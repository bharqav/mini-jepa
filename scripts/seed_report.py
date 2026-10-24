"""Aggregate independent seed runs into mean/std and a head-vs-head verdict.

Usage:
  python scripts/seed_report.py --pattern "runs/seed-compat-s*/summary.json" ...
Runs matched by experiment name; every entry must come from the same eval
protocol (same split, same seed semantics) or the comparison is invalid.

Also supports a PAIRED bootstrap between two systems evaluated on the SAME
rows, using the per-row hit lists saved by evaluate.py.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from jev_qwen.stats import compare_seeds, seed_summary

ap = argparse.ArgumentParser()
ap.add_argument(
    "--glob",
    action="append",
    required=True,
    help="glob of summary.json files for one system (repeatable)",
)
ap.add_argument(
    "--label", action="append", required=True, help="label for the preceding glob, same order"
)
ap.add_argument(
    "--paired",
    nargs=2,
    metavar=("A_CAL", "B_CAL"),
    help="two eval_calibration.json to paired-bootstrap",
)
ap.add_argument("--out", default="results/seed_report.json")
args = ap.parse_args()


def main() -> int:
    report = {"systems": {}}
    for label, pattern in zip(args.label, args.glob):
        files = sorted(glob.glob(pattern))
        rows = []
        for f in files:
            d = json.loads(Path(f).read_text())
            rows.append(
                {
                    "exp": d.get("checkpoint", f),
                    "test": d["test"],
                    "best_test_accuracy": d.get("best_test_accuracy"),
                }
            )
        if not rows:
            print(f"WARNING: no files matched {pattern}")
            continue
        accs = [r["test"]["accuracy"] for r in rows]
        report["systems"][label] = {
            "n_seeds": len(rows),
            "accuracy": seed_summary(accs),
            "nll": seed_summary([r["test"]["nll"] for r in rows]),
            "choice": seed_summary([r["test"]["accuracy_choice"] for r in rows]),
            "noul": seed_summary([r["test"]["accuracy_noul"] for r in rows]),
            "score": seed_summary([r["test"]["accuracy_score"] for r in rows]),
            "auc_noul": seed_summary([r["test"]["auc_noul"] for r in rows]),
            "per_seed_accuracy": accs,
            "runs": [r["exp"] for r in rows],
        }

    if len(args.label) >= 2 and len(report["systems"]) >= 2:
        a, b = args.label[0], args.label[1]
        report["comparison"] = compare_seeds(
            report["systems"][a]["per_seed_accuracy"], report["systems"][b]["per_seed_accuracy"]
        )

    if args.paired:
        from jev_qwen.stats import paired_delta_ci

        A = json.loads(Path(args.paired[0]).read_text())["rows"]
        B = json.loads(Path(args.paired[1]).read_text())["rows"]
        assert len(A) == len(B), "paired bootstrap needs identical row counts/order"
        hit = lambda r: int(max(range(len(r["logits"])), key=r["logits"].__getitem__)) == r["label"]
        report["paired_bootstrap_accuracy_delta"] = paired_delta_ci(
            [hit(r) for r in A], [hit(r) for r in B]
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
