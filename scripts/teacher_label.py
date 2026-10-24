"""Usage: python scripts/teacher_label.py --in data/processed/train.jsonl
          --out data/processed/train_teacher.jsonl [--teacher local-lm]
          [--layers 8] [--limit 200] [--cache data/processed/teacher_cache.jsonl]

Labels records WITHOUT ground truth (or audits existing ones via agreement
stats). Teacher outputs never overwrite existing labels silently:
disagreements are flagged.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from jev_qwen.teacher import ExternalTeacher, LocalLMTeacher, TeacherCache, label_records

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--teacher", default="local-lm", choices=["local-lm", "external"])
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--cache", default="data/processed/teacher_cache.jsonl")
args = ap.parse_args()


def main() -> None:
    records = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    todo = [r for r in records if r.get("label") is None][: args.limit] or records[: args.limit]
    teacher = (
        LocalLMTeacher(num_layers=args.layers) if args.teacher == "local-lm" else ExternalTeacher()
    )
    report = label_records(todo, teacher, TeacherCache(args.cache))
    with open(args.out, "w") as f:
        for r in todo:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2))
    if isinstance(teacher, ExternalTeacher):
        print(
            json.dumps(
                {
                    "calls": teacher.calls,
                    "failures": teacher.failures,
                    "chars_sent": teacher.chars_sent,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
