"""Usage: python scripts/train.py configs/smoke.yaml --name smoke [--resume ID]"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from jev_qwen.train import train

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--name", default=None)
ap.add_argument("--resume", default=None)
args = ap.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)
train(cfg, args.name or Path(args.config).stem, args.resume)
