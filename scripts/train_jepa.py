"""Pretrain a JEPA-style latent predictor on Qwen3-0.6B decision text.

Example:
    python scripts/train_jepa.py configs/jepa_qwen.yaml --name jepa-qwen
    python scripts/train_jepa.py configs/jepa_qwen.yaml --name jepa-qwen --probe

The run directory contains ``context.safetensors`` (the trained context
encoder), predictor/mask weights, config, metrics and an eval summary. The
``--probe`` flag then trains a frozen-backbone decision head from that context
encoder so the representation can be compared with an uninitialised baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_qwen.data import DATASET_NAME
from jev_qwen.hub import resolve_model_revision
from jev_qwen.jepa import (
    JEPAConfig,
    JEPAState,
    collapse_diagnostics,
    representation_similarity,
)
from jev_qwen.jepa_data import build_packed_dataset, make_batches
from jev_qwen.model import load_backbone
from jev_qwen.tokenize import get_tokenizer
from jev_qwen.train import get_questions
from jev_qwen.utils import autocast, env_fingerprint, get_device, seed_all


def make_run_dir(root: str, name: str) -> Path:
    run_dir = Path(root) / f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_jepa(
    state: JEPAState,
    out_dir: Path,
    config: dict,
    *,
    steps: int,
    seed: int,
    max_len: int,
    n_sequences: int,
    eval_metrics: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    context = state.context.model
    if context is None:
        raise RuntimeError("only the Qwen-backed JEPA state can be checkpointed")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in context.state_dict().items()},
        str(out_dir / "context.safetensors"),
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in state.predictor.state_dict().items()},
        str(out_dir / "predictor.safetensors"),
    )
    save_file(
        {"mask_vector": state.ctx_mask_vector.detach().cpu().contiguous()},
        str(out_dir / "mask_vector.safetensors"),
    )
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    meta = {
        "model_name": config["model"]["name"],
        "model_revision": config["model"].get("revision")
        or resolve_model_revision(config["model"]["name"]),
        "num_layers": config["model"]["num_layers"],
        "hidden_dim": state.cfg.hidden_dim,
        "jepa": asdict(state.cfg),
        "steps": steps,
        "seed": seed,
        "max_len": max_len,
        "sequences": n_sequences,
        "eval": eval_metrics,
        "dataset": DATASET_NAME,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


@torch.no_grad()
def evaluate_jepa(
    state: JEPAState,
    sequences: list[list[int]],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    precision: str,
) -> dict[str, float]:
    state.eval()
    rng = torch.Generator().manual_seed(seed + 991)
    n_tokens = 0
    n_elements = 0
    mse_sum = 0.0
    cos_sum = 0.0
    cka_sum = 0.0
    std_sum = 0.0
    collapse_sum = 0.0
    n_batches = 0
    for batch in make_batches(sequences, batch_size, seed=seed, shuffle=False):
        batch = batch.to(device)
        with autocast(device, precision):
            out = state(batch, rng=rng)
        mask = out["target_mask"]
        pred = out["pred"][mask].float()
        target = out["target"][mask].float()
        count = pred.shape[0]
        n_tokens += count
        n_elements += pred.numel()
        mse_sum += float(F.mse_loss(pred, target, reduction="sum"))
        sim = representation_similarity(pred, target)
        cos_sum += sim["pred_target_cosine"] * count
        cka_sum += sim["linear_cka"] * count
        diag = collapse_diagnostics(pred)
        std_sum += diag["mean_std"] * count
        collapse_sum += diag["collapse_ratio"] * count
        n_batches += 1
    state.train()
    denom = max(n_tokens, 1)
    return {
        "eval_mse": mse_sum / max(n_elements, 1),
        "eval_cosine": cos_sum / denom,
        "eval_linear_cka": cka_sum / denom,
        "eval_mean_std": std_sum / denom,
        "eval_collapse_ratio": collapse_sum / denom,
        "eval_tokens": float(n_tokens),
        "eval_batches": float(n_batches),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretrain a Qwen-backed JEPA latent predictor")
    ap.add_argument("config", nargs="?", default="configs/jepa_qwen.yaml")
    ap.add_argument("--name", default="jepa-qwen")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train-questions", type=int, default=None)
    ap.add_argument("--max-test-questions", type=int, default=None)
    ap.add_argument(
        "--bidirectional", action="store_true", help="use a bidirectional context/target encoder"
    )
    ap.add_argument(
        "--freeze-target", action="store_true", help="do not EMA-update the target encoder"
    )
    ap.add_argument("--probe", action="store_true", help="train a frozen decision head afterwards")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.steps is not None:
        config["optim"]["steps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_train_questions is not None:
        config["data"]["max_train_questions"] = args.max_train_questions
    if args.max_test_questions is not None:
        config["data"]["max_test_questions"] = args.max_test_questions
    if args.bidirectional:
        config["jepa"]["bidirectional_context"] = True
    if args.freeze_target:
        config["jepa"]["freeze_target"] = True

    seed = int(config.get("seed", 0))
    seed_all(seed)
    device = get_device()
    model_cfg = config["model"]
    data_cfg = config["data"]
    jepa_cfg = config["jepa"]
    optim = config["optim"]
    logging = config["logging"]
    steps = int(optim["steps"])
    batch_size = int(optim.get("batch_size", 8))
    precision = optim.get("precision", "bf16")
    max_len = int(logging.get("max_len", 256))
    max_sequences = logging.get("max_sequences")
    eval_max_sequences = int(logging.get("eval_max_sequences", 256))

    tokenizer = get_tokenizer(model_cfg["name"], revision=model_cfg.get("revision"))
    records = get_questions("train", config)
    sequences = build_packed_dataset(
        records,
        tokenizer,
        max_len=max_len,
        max_sequences=max_sequences,
        seed=seed,
    )
    if not sequences:
        raise RuntimeError("no packed sequences; check the dataset/config")
    eval_records = get_questions("test", config)
    eval_sequences = build_packed_dataset(
        eval_records,
        tokenizer,
        max_len=max_len,
        max_sequences=eval_max_sequences,
        seed=seed + 1,
    )
    print(
        f"packed train={len(sequences)} eval={len(eval_sequences)} len={max_len} device={device}",
        flush=True,
    )

    backbone = load_backbone(
        model_cfg["name"], int(model_cfg["num_layers"]), revision=model_cfg.get("revision")
    )
    cfg = JEPAConfig(
        hidden_dim=backbone.config.hidden_size,
        num_layers=int(model_cfg["num_layers"]),
        predictor_layers=int(jepa_cfg.get("predictor_layers", 2)),
        predictor_dim=int(jepa_cfg.get("predictor_dim", 512)),
        mask_ratio=float(jepa_cfg.get("mask_ratio", 0.2)),
        mask_span=int(jepa_cfg.get("mask_span", 8)),
        ema_momentum=float(jepa_cfg.get("ema_momentum", 0.996)),
        loss_mse=float(jepa_cfg.get("loss_mse", 1.0)),
        loss_cosine=float(jepa_cfg.get("loss_cosine", 0.1)),
        var_weight=float(jepa_cfg.get("var_weight", 0.1)),
        cov_weight=float(jepa_cfg.get("cov_weight", 0.04)),
        var_target_std=float(jepa_cfg.get("var_target_std", 1.0)),
        bidirectional_context=bool(jepa_cfg.get("bidirectional_context", False)),
        freeze_target=bool(jepa_cfg.get("freeze_target", False)),
        seed=seed,
    )
    state = JEPAState(cfg, qwen_model=backbone).to(device)
    trainable = [p for p in state.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optim.get("lr", 3e-4)),
        weight_decay=float(optim.get("weight_decay", 0.01)),
    )
    warmup = int(optim.get("warmup_steps", 100))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    run_dir = make_run_dir(args.root, args.name)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "started_at": datetime.now().isoformat(),
                "device": str(device),
                "env": env_fingerprint(),
                "steps": steps,
                "seed": seed,
            },
            indent=2,
        )
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    rng = torch.Generator().manual_seed(seed)
    metrics_path = run_dir / "metrics.jsonl"
    started = time.time()
    step = 0
    state.train()
    optimizer.zero_grad()
    while step < steps:
        batches = make_batches(
            sequences, batch_size, seed=seed + step // max(len(sequences) // batch_size, 1)
        )
        for batch in batches:
            if step >= steps:
                break
            batch = batch.to(device)
            with autocast(device, precision):
                out = state(batch, rng=rng)
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(optim.get("grad_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            state.ema_update()
            step += 1
            if step % int(logging.get("log_every", 50)) == 0 or step == 1:
                mask = out["target_mask"]
                diag = collapse_diagnostics(out["pred"][mask])
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mse": float(out["mse"].detach()),
                    "cosine": float(out["cosine"].detach()),
                    "var_term": float(out["var_term"].detach()),
                    "cov_term": float(out["cov_term"].detach()),
                    "mean_std": diag["mean_std"],
                    "collapse_ratio": diag["collapse_ratio"],
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_sec": round(time.time() - started, 1),
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"step={step} loss={record['loss']:.4f} mse={record['mse']:.4f} "
                    f"cos={record['cosine']:.3f} std={record['mean_std']:.3f} "
                    f"collapse={record['collapse_ratio']:.2f}",
                    flush=True,
                )
            if step % int(logging.get("eval_every", 500)) == 0:
                metrics = evaluate_jepa(
                    state,
                    eval_sequences,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                    precision=precision,
                )
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, **metrics}) + "\n")
                print(
                    f"step={step} eval_mse={metrics['eval_mse']:.4f} "
                    f"eval_cos={metrics['eval_cosine']:.3f} "
                    f"eval_std={metrics['eval_mean_std']:.3f} "
                    f"eval_collapse={metrics['eval_collapse_ratio']:.2f}",
                    flush=True,
                )

    eval_metrics = evaluate_jepa(
        state,
        eval_sequences,
        device=device,
        batch_size=batch_size,
        seed=seed,
        precision=precision,
    )
    ckpt = save_jepa(
        state,
        run_dir / "jepa",
        config,
        steps=steps,
        seed=seed,
        max_len=max_len,
        n_sequences=len(sequences),
        eval_metrics=eval_metrics,
    )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "steps": steps,
        "seed": seed,
        "train_sequences": len(sequences),
        "eval_sequences": len(eval_sequences),
        "duration_sec": round(time.time() - started, 1),
        **eval_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    if args.probe:
        probe = copy.deepcopy(config.get("probe") or {})
        if not probe:
            raise ValueError("--probe requires a 'probe' section in the config")
        probe["seed"] = seed
        probe["model"] = {
            **model_cfg,
            "init_from_jepa": str(ckpt / "context.safetensors"),
        }
        probe["data"] = {**data_cfg, **probe.get("data", {})}
        del state, optimizer, backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from jev_qwen.train import train

        train(probe, f"{args.name}-probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
