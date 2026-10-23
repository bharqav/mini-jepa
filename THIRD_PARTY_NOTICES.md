# Third-party notices

This repository's own code is MIT-licensed (see `LICENSE`). The following
third-party components are used or downloaded at runtime; their licenses apply
to them, not to this repository.

## Models and datasets

| component | revision | license | notes |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | Backbone weights and tokenizer |
| `avbiswas/bev-decision-150K` | `11ed9b5b879e7f9524e7c89c1037d498c54206f0` | **unknown** | Dataset card declares `license: unknown`; clarify with the owner before redistribution or commercial use |

The dataset license is a real blocker for redistribution. Derived checkpoints
may inherit dataset terms even though the code is MIT. Do not assume the
Apache-2.0 backbone license covers the data.

## Runtime libraries

| package | license |
|---|---|
| `torch` | BSD-3-Clause |
| `transformers` | Apache-2.0 |
| `datasets` | Apache-2.0 |
| `peft` | Apache-2.0 |
| `safetensors` | Apache-2.0 |
| `numpy` | BSD-3-Clause |
| `pandas` | BSD-3-Clause |
| `fastparquet` | Apache-2.0 |
| `pydantic` | MIT |
| `fastapi` | MIT |
| `uvicorn` | BSD-3-Clause |
| `httpx` | BSD-3-Clause |
| `prometheus-client` | Apache-2.0 |
| `pyyaml` | MIT |
| `tqdm` | MPL-2.0 AND MIT |

Optional OpenTelemetry packages are Apache-2.0. Transitive dependencies carry
their own licenses; `pip-licenses` or your distribution's package metadata is
the authoritative source for a given environment.

## Development-only tools

`pytest` (MIT), `pytest-cov` (MIT), `ruff` (MIT), `mypy` (MIT), `pip-audit`
(Apache-2.0), `coverage` (Apache-2.0), and `types-PyYAML` (Apache-2.0) are used
for development and CI, not shipped in the runtime image.
