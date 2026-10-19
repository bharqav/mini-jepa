"""Prepare data: load reference set (or synthetic fallback), validate, dedup,
group-split, save processed shards + data-quality report.

Usage: python scripts/prepare_data.py --max-train 20000 --max-test 2000
       python scripts/prepare_data.py --synthetic --synthetic-n 200
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from jev_qwen.data import (
    DATASET_NAME,
    dedup_records,
    group_split,
    load_questions,
    synthetic_records,
    validate_records,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    if args.synthetic:
        recs = synthetic_records(args.synthetic_n, args.seed)
    else:
        recs = []
        for split, n in (("train", args.max_train), ("test", args.max_test)):
            ds = load_questions(split, n, args.seed)
            recs += [
                {
                    "state": r["state"],
                    "task_type": r["task_type"],
                    "instructions": r["instructions"],
                    "choices": r["choices"],
                    "label": r["label"],
                    "example_id": r.get("example_id", ""),
                    "provenance": r.get("provenance", DATASET_NAME),
                }
                for r in ds
            ]
    report = {
        "source": "synthetic-v1" if args.synthetic else DATASET_NAME,
        "raw_n": len(recs),
        **validate_records(recs),
    }
    recs = dedup_records(recs)
    report["deduped_n"] = len(recs)
    splits = group_split(recs, args.seed)
    report["splits"] = {k: len(v) for k, v in splits.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        with open(out / f"{k}.jsonl", "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
    (out / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
