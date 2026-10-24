"""Usage: python scripts/train.py configs/smoke.yaml --name smoke [--resume ID]"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from jev_qwen.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--init-from-jepa",
        default=None,
        help="path to a JEPA context.safetensors (or its directory) for backbone init",
    )
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.init_from_jepa:
        cfg.setdefault("model", {})["init_from_jepa"] = args.init_from_jepa
    train(cfg, args.name or Path(args.config).stem, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
