"""CLI: single decision or batch file.

Single: python scripts/decide.py --checkpoint CKPT --state "..." --type choice
          --question "..." --candidate "id:text" --candidate "id:text"
Batch:  python scripts/decide.py --checkpoint CKPT --batch examples/request.json
"""

import argparse
import json
import sys

sys.path.insert(0, "src")
from jev_qwen.infer import DecisionEngine
from jev_qwen.schema import DecisionRequest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--type", dest="qtype", default="choice")
    ap.add_argument("--question", default=None)
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--batch", default=None)
    args = ap.parse_args()

    engine = DecisionEngine(checkpoint=args.checkpoint)
    if args.batch:
        with open(args.batch) as fh:
            payload = json.load(fh)
    else:
        cands = []
        for c in args.candidate:
            cid, _, text = c.partition(":")
            cands.append({"id": cid or text, "text": text or cid})
        payload = {
            "state": args.state,
            "questions": [
                {"id": "q0", "type": args.qtype, "question": args.question, "candidates": cands}
            ],
        }
    print(json.dumps(engine.decide(DecisionRequest(**payload)).model_dump(), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
