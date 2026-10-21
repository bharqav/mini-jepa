"""One-time conversion of legacy ``head.pt`` checkpoints to ``head.safetensors``.

This repository no longer loads pickle checkpoints because ``torch.load`` can
execute arbitrary code. Old checkpoints written before that change contain
``head.pt``; convert each one once, then delete the pickle.

Usage:
    python scripts/convert_checkpoints.py runs/my-run/checkpoints/best
    python scripts/convert_checkpoints.py runs/*/checkpoints/* --delete-old

The converter itself uses ``torch.load(..., weights_only=True)`` so even the
one-time migration cannot execute arbitrary code from the pickle. Back up the
checkpoint directory first if it matters to you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch
from safetensors.torch import save_file

HEAD_WEIGHTS = "head.safetensors"
LEGACY_HEAD_WEIGHTS = "head.pt"


def convert(checkpoint: Path, delete_old: bool = False, force: bool = False) -> Path:
    """Convert ``checkpoint/head.pt`` to ``checkpoint/head.safetensors``."""
    checkpoint = Path(checkpoint)
    legacy = checkpoint / LEGACY_HEAD_WEIGHTS
    target = checkpoint / HEAD_WEIGHTS
    if not legacy.exists():
        raise FileNotFoundError(f"{checkpoint}: no {LEGACY_HEAD_WEIGHTS} to convert")
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to overwrite")
    state = torch.load(legacy, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(
            f"{legacy}: expected a state dict of tensors; this file is not a plain "
            "head checkpoint and was not converted"
        )
    tensors = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(tensors, str(target))
    if delete_old:
        legacy.unlink()
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument(
        "--delete-old",
        action="store_true",
        help="remove head.pt after a successful conversion",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing head.safetensors")
    args = ap.parse_args()
    failed = 0
    for ckpt in args.checkpoints:
        try:
            out = convert(ckpt, delete_old=args.delete_old, force=args.force)
            print(f"converted {ckpt} -> {out.name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAILED {ckpt}: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
