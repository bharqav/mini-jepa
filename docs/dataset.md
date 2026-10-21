# Dataset

## Reference data
`avbiswas/bev-decision-150K` (HF, parquet per subset: `data`, `hard_50k`,
`counterfactual_15k`, `numeric_temporal`, `skills` → train/test). Row schema:
`state: str`, `questions_json: str` (dict `qid → {type, instructions,
criteria, label}`), plus `domain`, `question_types`, `question_count`.

The dataset is fetched at a pinned revision
(`11ed9b5b879e7f9524e7c89c1037d498c54206f0`, `jev_qwen.hub`), both through
`datasets.load_dataset` and the `snapshot_download` fallback.

Type semantics (verified against `bev-train/dataset.py`, not invented):
- **choice**: candidates = `criteria` keys (with `key: description` text when
  non-empty); label = index of `label` key.
- **noul**: candidates = `["No", "Yes"]`; label = `int(label)`; output
  `P(Yes) = softmax(logits)[1]`.
- **score**: candidates = enumerated `criteria` levels (`"i: level"`);
  label = level index; output = expected level + normalized score
  `E[level] / (C-1) ∈ [0, 1]`.

## Pipeline (`scripts/prepare_data.py`)
Load → `validate_records` (malformed / impossible-label / exact-dupe /
conflict detection) → `dedup_records` → `group_split` (grouped by state
fingerprint BEFORE splitting, no leakage) → `{train,validation,test}.jsonl`
+ `data_quality_report.json`.

Measured slice (500 train rows / 100 test rows, `data/` subset): 600
questions, 0 malformed, 0 impossible labels, 0 exact dupes, 0 conflicts;
types {choice: 221, noul: 272, score: 107}; split 483/47/70 with zero shared
states across splits.

## Loader compatibility
`jev_qwen.data.load_questions` prefers HF `datasets` (pyarrow) and falls
back to `huggingface_hub` + `fastparquet` with an identical row schema, for
machines where pyarrow binaries are unavailable. ## Adapters and teacher data
- `jev_qwen/adapters.py`: `jsonl` adapter (any `{state, type, instructions,
  choices, label?}` JSONL → canonical rows, validated on load) and
  `synthetic` adapter (`<n>:<seed>`); new public datasets plug in via
  `@register` without touching core code.
- `scripts/teacher_label.py` + `jev_qwen/teacher.py`: optional offline
  labeling with a local frozen LM (genuine softmax confidences, no cost) or
  a user-configured external endpoint (retries, call/char accounting,
  provenance per label). Fingerprint-keyed cache, disagreement flagging,
  validation of every teacher output. Never required for the baseline;
  synthetic labels are always marked `provenance: synthetic-v1`, never passed
  off as ground truth.
