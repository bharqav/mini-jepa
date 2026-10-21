"""Pinned Hugging Face revisions.

Every hub fetch (backbone weights, tokenizer, dataset) resolves an explicit
commit SHA. Nothing in the training, inference or data path fetches an
implicit ``main``. A custom model or dataset name must supply its own
revision, otherwise construction fails loudly instead of silently tracking a
moving branch.
"""

from __future__ import annotations

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

DEFAULT_DATASET_NAME = "avbiswas/bev-decision-150K"
DEFAULT_DATASET_REVISION = "11ed9b5b879e7f9524e7c89c1037d498c54206f0"

MODEL_REVISIONS = {DEFAULT_MODEL_NAME: DEFAULT_MODEL_REVISION}
DATASET_REVISIONS = {DEFAULT_DATASET_NAME: DEFAULT_DATASET_REVISION}


def resolve_model_revision(model_name: str, revision: str | None = None) -> str:
    if revision:
        return revision
    if model_name in MODEL_REVISIONS:
        return MODEL_REVISIONS[model_name]
    raise ValueError(
        f"model {model_name!r} has no pinned revision; pass revision=<commit sha> "
        "instead of tracking a hub branch"
    )


def resolve_dataset_revision(dataset_name: str, revision: str | None = None) -> str:
    if revision:
        return revision
    if dataset_name in DATASET_REVISIONS:
        return DATASET_REVISIONS[dataset_name]
    raise ValueError(
        f"dataset {dataset_name!r} has no pinned revision; pass revision=<commit sha> "
        "instead of tracking a hub branch"
    )
