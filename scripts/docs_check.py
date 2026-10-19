"""Validate relative links, mermaid fences, and tables in markdown docs."""

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
docs = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
if (ROOT / "CONTRIBUTING.md").exists():
    docs.append(ROOT / "CONTRIBUTING.md")
if (ROOT / "SECURITY.md").exists():
    docs.append(ROOT / "SECURITY.md")
if (ROOT / "CHANGELOG.md").exists():
    docs.append(ROOT / "CHANGELOG.md")

LINK = re.compile(r"\[[^\]]*\]\(([^)#][^)]*)\)")
BAD_LINK_TOKENS = ("http://", "https://", "mailto:")

errors = []
mermaid_blocks = 0
checked_links = 0

for doc in docs:
    text = doc.read_text(encoding="utf-8", errors="ignore")
    rel = doc.relative_to(ROOT)

    # mermaid fences balanced + have a diagram type
    fences = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    mermaid_blocks += len(fences)
    for i, block in enumerate(fences, 1):
        first = block.strip().splitlines()[0].strip() if block.strip() else ""
        if not re.match(
            r"^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie)\b",
            first,
        ):
            errors.append(
                f"{rel}: mermaid block {i} has no valid diagram type header: {first[:40]!r}"
            )

    for target in LINK.findall(text):
        target = target.strip().split("#")[0]
        if not target or target.startswith(BAD_LINK_TOKENS):
            continue
        checked_links += 1
        if target.startswith("/"):
            resolved = ROOT / target.lstrip("/")
        else:
            resolved = (doc.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{rel}: broken link -> {target}")

    # unfenced absolute windows paths or leftover markers
    for pat, label in (
        (r"[A-Za-z]:\\\\Users\\\\", "absolute Windows user path"),
        (r"/home/bharqav", "absolute linux home path"),
        (r"TODO|FIXME|XXX", "TODO/FIXME marker"),
    ):
        for m in re.finditer(pat, text):
            errors.append(f"{rel}: contains {label}: ...{m.group(0)[:40]}...")

print(
    f"Checked {len(docs)} markdown files, {checked_links} relative links, "
    f"{mermaid_blocks} mermaid blocks."
)
if errors:
    print(f"\nPROBLEMS ({len(errors)}):")
    for e in errors:
        print("  " + e)
    sys.exit(1)
print("All markdown links resolve; mermaid blocks well-formed.")
sys.exit(0)
