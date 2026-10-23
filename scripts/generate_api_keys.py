"""Generate or rotate an API key entry.

Examples::

    # print a new single-key JSON document
    python scripts/generate_api_keys.py --key-id ci --scopes decide

    # add a key to an existing file (old keys stay valid until removed)
    python scripts/generate_api_keys.py --file /etc/jev-qwen/keys.json \
        --key-id ops --scopes decide,answer

The token is written only to the destination (stdout or --file); it is never
logged. The file is chmod 0600 on POSIX.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

SCOPES = ("decide", "answer")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a scoped jev-qwen API key")
    ap.add_argument("--key-id", default=f"key-{secrets.token_hex(3)}")
    ap.add_argument("--scopes", default="decide,answer")
    ap.add_argument("--rps", type=float, default=None)
    ap.add_argument("--burst", type=float, default=None)
    ap.add_argument("--file", default="", help="merge into this JSON file instead of stdout")
    args = ap.parse_args()

    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    unknown = set(scopes) - set(SCOPES)
    if unknown:
        ap.error(f"unknown scopes {sorted(unknown)}; choose from {list(SCOPES)}")
    if not scopes:
        ap.error("at least one scope is required")

    entry: dict[str, object] = {
        "token": secrets.token_urlsafe(32),
        "scopes": scopes,
    }
    if args.rps is not None:
        entry["rps"] = args.rps
    if args.burst is not None:
        entry["burst"] = args.burst

    if not args.file:
        print(json.dumps({args.key_id: entry}, indent=2))
        return 0

    path = Path(args.file)
    config: dict[str, object] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"cannot read existing key file {path}: {e}", file=sys.stderr)
            return 1
        if not isinstance(loaded, dict):
            print(f"existing key file {path} is not a JSON object", file=sys.stderr)
            return 1
        config = loaded
    config[args.key_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    print(f"wrote key {args.key_id!r} to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
