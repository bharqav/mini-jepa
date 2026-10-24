"""Training pipeline: LoRA/frozen backbone + trainable head, grad accumulation,
clipping, cosine schedule + warmup, eval/best-checkpoint, resume, JSONL logs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import (
    TASK_TYPES,
    DecisionDataset,
    LengthBucketBatchSampler,
    collate_with_pad,
    group_split,
    load_questions,
    synthetic_records,
)
from .evaluate import evaluate
from .model import build_network
from .run_logger import RunLogger, load_checkpoint
from .tokenize import encode_example, get_tokenizer
from .utils import autocast, get_device, seed_all, to_device

TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}


def accumulation_schedule(num_batches: int, accumulation_steps: int) -> list[tuple[int, bool]]:
    """Return each microbatch's loss divisor and optimizer-step boundary.

    A final partial window is retained and averaged by its actual size, rather
    than being silently dropped or underweighted.
    """
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    schedule: list[tuple[int, bool]] = []
    for start in range(0, num_batches, accumulation_steps):
        window_size = min(accumulation_steps, num_batches - start)
        schedule.extend((window_size, False) for _ in range(window_size - 1))
        schedule.append((window_size, True))
    return schedule


def init_backbone_from_jepa(network, source) -> None:
    """Load a JEPA context encoder into a frozen decision backbone.

    The JEPA checkpoint stores the raw ``Qwen3Model`` state dict. LoRA-wrapped
    backbones expose the base model under ``base_model.model``; loading there
    keeps the adapter initialisation (if any) untouched.
    """
    from safetensors.torch import load_file

    path = Path(source)
    if path.is_dir():
        path = path / "context.safetensors"
    state = load_file(str(path))
    target = network.backbone
    if hasattr(target, "base_model"):
        target = getattr(target.base_model, "model", target.base_model)
    target_names = set(target.state_dict())
    mapped = {}
    for key, value in state.items():
        if key in target_names:
            mapped[key] = value
            continue
        if key.endswith(".weight"):
            candidate = key[: -len(".weight")] + ".base_layer.weight"
            if candidate in target_names:
                mapped[candidate] = value
    missing, unexpected = target.load_state_dict(mapped, strict=False)
    lora_missing = [key for key in missing if ".lora_" in key]
    missing = [key for key in missing if ".lora_" not in key]
    print(
        f"initialised backbone from JEPA {path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})",
        flush=True,
    )
    if lora_missing:
        print(f"  randomly initialised LoRA adapters: {len(lora_missing)}", flush=True)
    if missing:
        print(f"  missing keys: {missing[:5]}", flush=True)
    if unexpected:
        print(f"  unexpected keys: {unexpected[:5]}", flush=True)


def make_loader(questions, config: dict[str, Any], tokenizer, shuffle: bool, seed: int = 0):
    data = config["data"]
    ds = DecisionDataset(
        questions,
        data["max_state_tokens"],
        data["max_choice_tokens"],
        tokenizer,
        permute=data.get("permute_candidates", False),
        seed=seed,
    )
    pad_id = tokenizer.pad_token_id or 0
    bs = data["batch_size"]
    kw = {
        "num_workers": data.get("num_workers", 0),
        "persistent_workers": data.get("num_workers", 0) > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    # Length bucketing reduces padding waste for eval-style loading. Training
    # keeps plain shuffling so batch composition stays seed-reproducible.
    if not shuffle and data.get("bucket_by_length", True):
        lengths = []
        for q in questions:
            try:
                ex = encode_example(
                    q["state"],
                    q["instructions"],
                    q["choices"],
                    TASK_NAMES[q["task_type"]],
                    data["max_state_tokens"],
                    data["max_choice_tokens"],
                    tokenizer,
                )
                lengths.append(len(ex["input_ids"]))
            except Exception:
                lengths.append(data["max_state_tokens"])
        return torch.utils.data.DataLoader(
            ds,
            batch_sampler=LengthBucketBatchSampler(lengths, bs, shuffle=False, seed=seed),
            collate_fn=partial(collate_with_pad, pad_id=pad_id),
            **kw,
        )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=partial(collate_with_pad, pad_id=pad_id),
        **kw,
    )


def get_questions(split: str, config: dict[str, Any]):
    data = config["data"]
    if data.get("synthetic"):
        n = (
            data.get("max_train_questions", 60)
            if split == "train"
            else data.get("max_test_questions", 20)
        )
        recs = synthetic_records(n or 20, seed=config.get("seed", 0))
        splits = group_split(recs, seed=config.get("seed", 0))
        return splits["train"] if split == "train" else splits["test"]
    key = "max_train_questions" if split == "train" else "max_test_questions"
    ds = load_questions(
        "train" if split == "train" else "test",
        data.get(key),
        config.get("seed", 0),
        subset=data.get("subset", "data"),
        revision=data.get("revision"),
    )
    return [
        {
            "state": r["state"],
            "task_type": r["task_type"],
            "instructions": r["instructions"],
            "choices": r["choices"],
            "label": r["label"],
            "example_id": r.get("example_id", ""),
            "provenance": r.get("provenance", ""),
            "domain": r.get("domain", ""),
        }
        for r in ds
    ]


def resolve_checkpoint(resume: str) -> Path:
    p = Path(resume)
    if not (p / "meta.json").exists():
        p = Path("runs") / resume / "checkpoints" / "final"
    if not (p / "meta.json").exists():
        raise FileNotFoundError(f"no checkpoint at {resume} or {p}")
    return p


def train(config: dict[str, Any], run_name: str, resume: str | None = None) -> Path:
    seed_all(config.get("seed", 0))
    device = get_device()
    if resume:
        rdir = resolve_checkpoint(resume)
        print(f"Resuming head/adapter weights from {rdir} (optimizer/schedule restart fresh)")
        config = {**config, "resumed_from": str(rdir)}
    logger = RunLogger(config, run_name, device)
    model, lora, optim, log = config["model"], config["lora"], config["optim"], config["logging"]
    tokenizer = get_tokenizer(model["name"], revision=model.get("revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    network = build_network(
        model["name"],
        model.get("num_layers"),
        lora["r"],
        lora["alpha"],
        lora["dropout"],
        lora["last_k_layers"],
        config["head"],
        gradient_checkpointing=model.get("gradient_checkpointing", False),
        revision=model.get("revision"),
        bidirectional=bool(model.get("bidirectional", False)),
    )
    if model.get("init_from_jepa") and not resume:
        init_backbone_from_jepa(network, model["init_from_jepa"])
    if resume:
        net_tmp, _ = load_checkpoint(rdir, torch.device("cpu"))
        network.head.load_state_dict(net_tmp.head.state_dict())
        from peft import PeftModel

        if isinstance(network.backbone, PeftModel) and isinstance(net_tmp.backbone, PeftModel):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            set_peft_model_state_dict(
                network.backbone, load_file(rdir / "lora" / "adapter_model.safetensors")
            )
    network = network.to(device)
    n_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
    print(
        f"trainable: backbone={n_params(network.backbone.parameters()):,} "
        f"head={n_params(network.head.parameters()):,}"
    )

    train_loader = make_loader(
        get_questions("train", config), config, tokenizer, True, config.get("seed", 0)
    )
    test_loader = None
    if config["data"].get("max_test_questions", 0):
        test_loader = make_loader(get_questions("test", config), config, tokenizer, False)

    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    groups = ([{"params": lora_params, "lr": float(optim["lr_lora"])}] if lora_params else []) + [
        {"params": network.head.parameters(), "lr": float(optim["lr_head"])}
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(optim.get("weight_decay", 0.01)))
    accum = int(optim.get("grad_accum_steps", 1))
    schedule = accumulation_schedule(len(train_loader), accum)
    total_steps = max(optim["epochs"] * sum(should_step for _, should_step in schedule), 1)
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(
        optimizer, int(optim.get("warmup_ratio", 0.05) * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and optim.get("precision", "bf16") == "fp16"
    )

    step, best_acc = 0, -1.0
    network.train()
    for epoch in range(optim["epochs"]):
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        optimizer.zero_grad()
        for micro, batch in enumerate(train_loader, 1):
            window_size, should_step = schedule[micro - 1]
            batch = to_device(batch, device)
            with autocast(device, optim.get("precision", "bf16")):
                logits = network(**batch)
                loss = F.cross_entropy(logits, batch["labels"])
            if scaler.is_enabled():
                scaler.scale(loss / window_size).backward()
            else:
                (loss / window_size).backward()
            loss_value = float(loss.detach().item())
            correct = int((logits.argmax(-1) == batch["labels"]).sum())
            tot_loss += loss_value * len(logits)
            tot_correct += correct
            tot_n += len(logits)
            if not should_step:
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(optim.get("grad_clip", 1.0)))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            sched.step()
            optimizer.zero_grad()
            step += 1
            if step % log.get("log_every", 20) == 0:
                logger.log(
                    step,
                    split="train",
                    epoch=epoch,
                    loss=loss_value,
                    accuracy=correct / len(logits),
                )
            if step % log.get("save_every", 500) == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if test_loader is not None and step % log.get("eval_every", 500) == 0:
                m = evaluate(network, test_loader, device)
                logger.log(step, split="test", **m)
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    logger.save_checkpoint(network, name="best", step=step)
        logger.log(
            step,
            split="train_epoch",
            epoch=epoch,
            loss=tot_loss / max(tot_n, 1),
            accuracy=tot_correct / max(tot_n, 1),
        )

    summary = {
        "final_train_loss": tot_loss / max(tot_n, 1),
        "final_train_accuracy": tot_correct / max(tot_n, 1),
    }
    if test_loader is not None:
        m = evaluate(network, test_loader, device)
        logger.log(step, split="test", **m)
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            logger.save_checkpoint(network, name="best", step=step)
        summary["test"] = m
        summary["best_test_accuracy"] = best_acc
    ckpt = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(ckpt), **summary)
    return logger.run_dir
