"""Usage: python scripts/evaluate.py runs/<id>/checkpoints/best [--split test]"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from jev_qwen.evaluate import evaluate, order_sensitivity
from jev_qwen.run_logger import load_checkpoint
from jev_qwen.tokenize import get_tokenizer
from jev_qwen.train import get_questions, make_loader
from jev_qwen.utils import get_device

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--split", default="test")
ap.add_argument("--n-perm", type=int, default=5)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--out-dir", default="runs")
args = ap.parse_args()

device = get_device()
network, meta = load_checkpoint(args.checkpoint, device)
with open(f"{args.checkpoint}/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["batch_size"] = args.batch_size
if not cfg["data"].get("max_test_questions"):
    print("note: config has no test split; evaluating on train split")
    args.split = "train"
tok = get_tokenizer(meta["model_name"], revision=meta.get("model_revision"))
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
eval_questions = get_questions(args.split, cfg)
loader = make_loader(eval_questions, cfg, tok, False)
metrics = evaluate(network, loader, device)
print(json.dumps(metrics, indent=2))
try:
    sens = order_sensitivity(network, loader, device, args.n_perm)
    print(json.dumps(sens, indent=2))
    metrics.update(sens)
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"order-sensitivity skipped: {e}")
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "eval_result.json", "w") as f:
    json.dump({"checkpoint": args.checkpoint, **metrics}, f, indent=2)

# Calibration: seeded 50/50 fit/measure split; temperature fit on fit-rows,
# all post-scaling numbers on measure-rows. Diagrams need no matplotlib.
import random

from jev_qwen.calibration import (
    conf_hit_pairs,
    reliability_svg,
    risk_coverage,
    risk_coverage_svg,
    temperature_report,
)
from jev_qwen.evaluate import collect_rows, evaluate_by_domain

rows = collect_rows(network, loader, device)
rng = random.Random(0)
idx = list(range(len(rows)))
rng.shuffle(idx)
fit = [rows[i] for i in idx[: len(idx) // 2]]
measure = [rows[i] for i in idx[len(idx) // 2 :]]
trep = temperature_report(
    [r["logits"] for r in fit],
    [r["label"] for r in fit],
    [r["logits"] for r in measure],
    [r["label"] for r in measure],
)
pairs = conf_hit_pairs(
    [r["logits"] for r in measure], [r["label"] for r in measure], trep["temperature"]
)
rc = risk_coverage(pairs)

# Bootstrap CIs over eval rows (sampling noise).
from jev_qwen.stats import accuracy_ci, mean_ci

hits = [int(max(range(len(r["logits"])), key=r["logits"].__getitem__)) == r["label"] for r in rows]


def _nll(r):
    m = max(r["logits"])
    z = sum(math.exp(v - m) for v in r["logits"])
    return -(r["logits"][r["label"]] - m - math.log(z))


lls = [_nll(r) for r in rows]
cis = {"accuracy_ci": accuracy_ci(hits), "nll_ci": mean_ci(lls)}
cal = {
    "checkpoint": args.checkpoint,
    "temperature": trep,
    "risk_coverage": rc,
    "bootstrap": cis,
    "rows": rows,
}
with open(Path(args.out_dir) / "eval_calibration.json", "w") as f:
    json.dump(cal, f, indent=2)
Path(args.out_dir, "reliability.svg").write_text(reliability_svg(trep["post_bins"]))
Path(args.out_dir, "risk_coverage.svg").write_text(risk_coverage_svg(rc["curve"]))
print(
    json.dumps(
        {
            "temperature": trep["temperature"],
            "pre_nll": trep["pre_nll"],
            "post_nll": trep["post_nll"],
            "pre_ece": trep["pre_ece"],
            "post_ece": trep["post_ece"],
            "acc_at_0.5": rc["acc_at_0.5"],
            "acc_at_1.0": rc["acc_at_1.0"],
            "accuracy_ci": cis["accuracy_ci"],
        },
        indent=2,
    )
)

# Per-domain (difficulty-stratified) breakdown with bootstrap CIs.
dom = evaluate_by_domain(network, loader, device)
if "error" in dom:
    print("domain breakdown unavailable:", dom["error"])
else:
    with open(Path(args.out_dir) / "eval_domains.json", "w") as f:
        json.dump(dom, f, indent=2)
    print("\nper-domain accuracy (95% CI):")
    for d, s in dom["by_domain"].items():
        print(f"  {s['accuracy']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  n={s['n']:<5} {d}")
    print("\nby candidate count:")
    for d, s in dom["by_candidate_count"].items():
        print(f"  {s['accuracy']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  n={s['n']:<5} C={d}")
