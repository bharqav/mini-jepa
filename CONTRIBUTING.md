# Contributing to jev-qwen

Thanks for considering a contribution. This document describes the actual
workflow — if something here is out of date, an issue or PR fixing it is very
welcome.

## Quick start

```bash
git clone https://github.com/bharqav/jev-qwen.git
cd jev-qwen
make install            # venv + editable install with dev extras (CPU torch)
make check              # format-check + lint + tests + secret scan
```

For GPU work:

```bash
make install-gpu        # installs CUDA torch, then the package
make test-real          # full suite including real Qwen3 weights (~1.2 GB download)
```

## Development workflow

1. **Branch** from `main` with a descriptive name (`feat/...`, `fix/...`, `docs/...`).
2. **Make the change.** Keep it focused; unrelated refactors belong in their own PR.
3. **Validate locally** — this is the same set CI runs:

   ```bash
   make format           # auto-format
   make check            # all offline gates
   ```

4. **Run the real-weights tests** if your change touches model code:

   ```bash
   make test-real
   ```

5. **Open a PR** describing what changed and why.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add candidate-count stratified evaluation
fix: prevent stale `datasets` cache from dropping new columns
docs: document the two-path decision architecture
test: cover batch-vs-single forward agreement
ci: add CodeQL analysis
chore: add ruff configuration
```

## Quality gates

CI runs these and they must pass:

| Gate | Command | Notes |
|------|---------|-------|
| Lint | `ruff check src/ tests/ scripts/` | No new violations |
| Format | `ruff format --check` | Formatting is not optional |
| Tests | `pytest tests/` | Weight-free suite, must not need a download |
| Secrets | `python scripts/security_scan.py` | Must find nothing |
| Integration | `JEV_REAL_WEIGHTS=1 pytest tests/` | Separate job, downloads weights |

**Do not** suppress a failing assertion to get CI green. If a test encodes
wrong behaviour, fix the test *and* explain why in the PR description.

## Code conventions

- Line length 100, enforced by ruff format.
- Type hints on public functions; internal helpers may stay loose.
- Docstrings explain *why*, not *what* the code does.
- Anything that is not implemented must say so in its docstring. This project
  is strict about not presenting planned work as working code.

## Things worth knowing before you change them

- **Candidate-order invariance is a tested invariant, not an optimisation.**
  If you touch `tokenize.py`, `model.py`, or the collate path, the permutation
  and isolation tests must still pass. If you genuinely need to break
  invariance, say so loudly in the PR.
- **`data.py` cache fingerprints** are derived from function source
  (`_map_fingerprint`). If you change the output schema, bump
  `SCHEMA_VERSION` or users will silently get stale cached columns.
- **`num_workers: 0` is the default on purpose** — worker processes use
  fork/forkserver, which is unreliable on WSL2 and locked-down Windows. Raise
  it only after profiling.

## Reporting bugs

Open an issue with:

- what you expected vs. what happened
- the exact command
- your OS, Python version, and `python scripts/inspect_environment.py` output
- the relevant config file

## License

By contributing you agree that your contributions are licensed under the
project's MIT License.