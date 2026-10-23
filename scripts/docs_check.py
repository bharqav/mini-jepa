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
if (ROOT / "THIRD_PARTY_NOTICES.md").exists():
    docs.append(ROOT / "THIRD_PARTY_NOTICES.md")
if (ROOT / "deploy" / "README.md").exists():
    docs.append(ROOT / "deploy" / "README.md")

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
        # Constructs that break GitHub's Mermaid renderer. Verified against
        # mermaid v11 (the engine GitHub uses): HTML breaks inside a Note, and
        # non-ASCII dashes anywhere in a block.
        for j, line in enumerate(block.splitlines(), 1):
            if re.search(r"(Note\s+(?:over|left of|right of)\s+[^:]+:.*)<br\s*/?>", line):
                errors.append(
                    f"{rel}: mermaid block {i} line {j} uses <br/> in a Note; "
                    "split it into multiple Note lines instead"
                )
            if "—" in line or "–" in line:
                errors.append(f"{rel}: mermaid block {i} line {j} contains a non-ASCII dash")
            if re.match(r"\s*participant\s+\w+\s+as\s+.*[()]", line):
                errors.append(
                    f"{rel}: mermaid block {i} line {j} has parentheses in a participant alias"
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

    # Leftover developer-local paths and unresolved markers. Patterns are
    # assembled from parts so this checker never matches its own source.
    dev_windows = "[A-Za-z]:" + chr(92) * 2 + "Users" + chr(92) * 2
    dev_linux = "/" + "home" + "/" + "bhar" + "qav"
    markers = "TO" + "DO|FIX" + "ME"
    for pat, label in (
        (dev_windows, "absolute Windows user path"),
        (dev_linux, "absolute linux home path"),
        (markers, "unresolved TODO/FIXME marker"),
    ):
        for _ in re.finditer(pat, text):
            errors.append(f"{rel}: contains {label}")

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
