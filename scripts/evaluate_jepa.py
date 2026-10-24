"""Linear and kNN probes for JEPA representations on standard text tasks.

This is the downstream-evidence half of the JEPA pipeline. It deliberately
compares against the frozen pretrained Qwen layers, a random-init Qwen encoder,
a frozen MLM encoder and a contrastive SimCSE encoder, because "not collapsed"
is not evidence of a useful representation.

Example:
    python scripts/evaluate_jepa.py \
        --jepa runs/jepa-qwen-.../jepa/context.safetensors \
        --tasks sst2,stsb \
        --baselines jepa,frozen_qwen,random_qwen,mlm_bert,simcse

Results are written to JSON (default: ``results/jepa_probe_results.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_qwen.hub import resolve_model_revision  # noqa: E402
from jev_qwen.model import load_backbone  # noqa: E402
from jev_qwen.utils import get_device  # noqa: E402

TASK_SPECS = {
    "sst2": {"config": "sst2", "kind": "classification", "num_classes": 2},
    "cola": {"config": "cola", "kind": "classification", "num_classes": 2},
    "stsb": {"config": "stsb", "kind": "regression", "num_classes": 0},
}

BASELINE_NAMES = ("jepa", "frozen_qwen", "random_qwen", "mlm_bert", "simcse")


@dataclass
class EncoderBundle:
    model: torch.nn.Module
    tokenizer: object
    bidirectional: bool
    pooling: str


def load_qwen_jepa(path: str, model_name: str, num_layers: int, revision: str | None):
    model = load_backbone(model_name, num_layers, revision=revision)
    state = load_file(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"JEPA checkpoint load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return model


def build_encoder(name: str, args: argparse.Namespace, device: torch.device) -> EncoderBundle:
    if name in {"jepa", "frozen_qwen", "random_qwen"}:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=resolve_model_revision(args.model, args.revision)
        )
        if name == "jepa":
            if not args.jepa:
                raise ValueError("--jepa is required for the jepa baseline")
            model = load_qwen_jepa(args.jepa, args.model, args.num_layers, args.revision)
            bidirectional = args.bidirectional
        elif name == "frozen_qwen":
            model = load_backbone(args.model, args.num_layers, revision=args.revision)
            bidirectional = args.bidirectional
        else:
            config = AutoConfig.from_pretrained(
                args.model, revision=resolve_model_revision(args.model, args.revision)
            )
            config.num_hidden_layers = args.num_layers
            model = Qwen3Model(config)
            bidirectional = args.bidirectional
        return EncoderBundle(model.to(device), tokenizer, bidirectional, args.pooling)

    if name in {"mlm_bert", "simcse"}:
        model_name = (
            "google-bert/bert-base-uncased"
            if name == "mlm_bert"
            else "princeton-nlp/sup-simcse-bert-base-uncased"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        return EncoderBundle(model.to(device), tokenizer, False, args.pooling)

    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")


@torch.no_grad()
def embed_texts(
    bundle: EncoderBundle,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    bundle.model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = bundle.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        if bundle.bidirectional:
            dtype = next(bundle.model.parameters()).dtype
            visible = mask[:, None, None, :].to(dtype)
            attention_mask = ((1.0 - visible) * torch.finfo(dtype).min).expand(
                -1, 1, mask.shape[1], -1
            )
        else:
            attention_mask = mask
        hidden = bundle.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()
        valid = mask.bool()
        if bundle.pooling == "last":
            lengths = valid.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=device), lengths]
        else:
            pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def task_examples(task: str, split: str, max_examples: int | None):
    spec = TASK_SPECS[task]
    ds = load_dataset("nyu-mll/glue", spec["config"], split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    if task == "stsb":
        return list(ds["sentence1"]), list(ds["sentence2"]), torch.tensor(ds["label"]).float()
    return list(ds["sentence"]), None, torch.tensor(ds["label"]).long()


def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([a, b, (a - b).abs()], dim=-1)


def standardize(train: torch.Tensor, test: torch.Tensor):
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train - mean) / std, (test - mean) / std


def ridge_fit(
    train: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float
) -> torch.Tensor:
    x = torch.cat([train, torch.ones(train.shape[0], 1)], dim=1)
    if num_classes:
        y = torch.nn.functional.one_hot(labels, num_classes).float()
    else:
        y = labels.float().unsqueeze(1)
    gram = x.T @ x + alpha * torch.eye(x.shape[1])
    return torch.linalg.solve(gram, x.T @ y)


def ridge_predict(weights: torch.Tensor, features: torch.Tensor, num_classes: int):
    x = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1)
    out = x @ weights
    return out.argmax(dim=1) if num_classes else out.squeeze(1)


def knn_predict(
    train: torch.Tensor,
    labels: torch.Tensor,
    test: torch.Tensor,
    k: int,
    num_classes: int,
) -> torch.Tensor:
    train = torch.nn.functional.normalize(train, dim=1)
    test = torch.nn.functional.normalize(test, dim=1)
    sim = test @ train.T
    values, indices = sim.topk(min(k, train.shape[0]), dim=1)
    weights = values.clamp(min=0.0) + 1e-6
    neighbour_labels = labels[indices]
    if num_classes:
        onehot = torch.nn.functional.one_hot(
            neighbour_labels, num_classes
        ).float() * weights.unsqueeze(-1)
        return onehot.sum(dim=1).argmax(dim=1)
    return (neighbour_labels.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = rankdata(a.detach().cpu().numpy())
    rb = rankdata(b.detach().cpu().numpy())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def effective_rank(features: torch.Tensor) -> float:
    x = features.float() - features.float().mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    probs = eigvals / eigvals.sum().clamp(min=1e-12)
    return float(torch.exp(-(probs * torch.log(probs + 1e-12)).sum()))


def evaluate_task(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    task: str,
    alpha: float,
    k: int,
) -> dict[str, float]:
    num_classes = TASK_SPECS[task]["num_classes"]
    train_x, test_x = standardize(train_x, test_x)
    rank = effective_rank(test_x)
    weights = ridge_fit(train_x, train_y, num_classes, alpha)
    linear = ridge_predict(weights, test_x, num_classes)
    knn = knn_predict(train_x, train_y, test_x, k, num_classes)
    if num_classes:
        return {
            "linear_accuracy": float((linear == test_y).float().mean()),
            "knn_accuracy": float((knn == test_y).float().mean()),
            "effective_rank": rank,
        }
    return {
        "linear_spearman": spearman(linear, test_y),
        "knn_spearman": spearman(knn, test_y),
        "effective_rank": rank,
    }


def retrieval_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    sim = a @ b.T
    target = torch.arange(a.shape[0])
    ranks = (sim > sim[target, target].unsqueeze(1)).sum(dim=1) + 1
    return {
        "retrieval_r1": float((ranks == 1).float().mean()),
        "retrieval_mrr": float((1.0 / ranks.float()).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Linear/kNN probes for JEPA representations")
    ap.add_argument("--jepa", default=None, help="path to context.safetensors")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--tasks", default="sst2,stsb")
    ap.add_argument("--baselines", default=",".join(BASELINE_NAMES))
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--output", default="results/jepa_probe_results.json")
    args = ap.parse_args()

    device = get_device()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    results: dict[str, dict[str, dict[str, float]]] = {t: {} for t in tasks}

    for task in tasks:
        if task not in TASK_SPECS:
            raise ValueError(f"unknown task {task!r}; choose from {list(TASK_SPECS)}")
        train_a, train_b, train_y = task_examples(task, "train", args.max_train)
        test_a, test_b, test_y = task_examples(task, "validation", args.max_eval)
        for name in baselines:
            bundle = build_encoder(name, args, device)
            if task == "stsb":
                train_x = pair_features(
                    embed_texts(bundle, train_a, device, args.batch_size, args.max_length),
                    embed_texts(bundle, train_b, device, args.batch_size, args.max_length),
                )
                test_x = pair_features(
                    embed_texts(bundle, test_a, device, args.batch_size, args.max_length),
                    embed_texts(bundle, test_b, device, args.batch_size, args.max_length),
                )
            else:
                train_x = embed_texts(bundle, train_a, device, args.batch_size, args.max_length)
                test_x = embed_texts(bundle, test_a, device, args.batch_size, args.max_length)
            metrics = evaluate_task(
                train_x, train_y, test_x, test_y, task, args.ridge_alpha, args.knn_k
            )
            if task == "stsb":
                metrics.update(
                    retrieval_metrics(
                        embed_texts(bundle, test_a, device, args.batch_size, args.max_length),
                        embed_texts(bundle, test_b, device, args.batch_size, args.max_length),
                    )
                )
            results[task][name] = metrics
            print(f"{task} {name}: {metrics}", flush=True)
            del bundle

    payload = {
        "model": args.model,
        "revision": args.revision,
        "num_layers": args.num_layers,
        "pooling": args.pooling,
        "jepa": args.jepa,
        "tasks": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
