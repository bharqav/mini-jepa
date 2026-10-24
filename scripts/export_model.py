"""Export: copy a checkpoint dir + write a manifest (base weights stay referenced)."""

import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def main() -> None:
    out = Path(args.out)
    shutil.copytree(args.checkpoint, out, dirs_exist_ok=True)
    meta = json.loads((out / "meta.json").read_text())
    meta["export_note"] = (
        "LoRA adapter + head only. Base weights loaded separately "
        f"from {meta['model_name']}; never vendored."
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
