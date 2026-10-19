"""Pre-publication security scan. Run: python scripts/security_scan.py"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".venv",
    ".git",
    "runs",
    "data",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".ruff_cache",
}
EXTS = {".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".json", ".csv", ".cfg", ".ini"}

SECRET_PATTERNS = [
    ("github token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
    ("huggingface token", r"hf_[A-Za-z0-9]{25,}"),
    ("openai-style key", r"sk-[A-Za-z0-9]{20,}"),
    ("aws access key", r"AKIA[0-9A-Z]{16}"),
    ("private key block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("slack token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("google api key", r"AIza[0-9A-Za-z_-]{35}"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ("connection string", r"(?i)(postgres|mysql|mongodb)://[^\s\"']+:[^\s\"']+@"),
]

SENSITIVE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}

PLACEHOLDER_OK = re.compile(r"(?i)(example|placeholder|your[_-]?\w+|xxx+|<.*>|changeme|dummy)")


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SENSITIVE_FILES or p.suffix.lower() in EXTS:
            yield p


def main():
    findings = []
    scanned = 0
    for p in iter_files():
        scanned += 1
        rel = p.relative_to(ROOT)
        if p.name in SENSITIVE_FILES:
            findings.append((rel, "sensitive filename present", ""))
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, pat in SECRET_PATTERNS:
            for m in re.finditer(pat, text):
                line_no = text[: m.start()].count("\n") + 1
                snippet = text.splitlines()[line_no - 1].strip()[:100]
                if PLACEHOLDER_OK.search(snippet):
                    continue
                findings.append((rel, f"{name} at line {line_no}", snippet))
    print(f"Scanned {scanned} files.")
    if findings:
        print(f"\nPOTENTIAL SECRETS FOUND ({len(findings)}):")
        for rel, what, snip in findings:
            print(f"  {rel}: {what}")
            print(f"      {snip}")
        return 1
    print("No secrets or sensitive files detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
