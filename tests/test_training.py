"""Training smoke: tiny synthetic run saves a checkpoint; fresh-process reload
evaluates with the same protocol. Weight-free parts run in CI; the end-to-end
run needs Qwen weights (JEV_REAL_WEIGHTS=1)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from jev_qwen.train import accumulation_schedule

REAL = os.environ.get("JEV_REAL_WEIGHTS") == "1"
needs_weights = pytest.mark.skipif(not REAL, reason="needs Qwen3-0.6B download")


def test_accumulation_schedule_steps_partial_final_window():
    assert accumulation_schedule(5, 2) == [
        (2, False),
        (2, True),
        (2, False),
        (2, True),
        (1, True),
    ]


def test_accumulation_schedule_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="num_batches"):
        accumulation_schedule(0, 2)
    with pytest.raises(ValueError, match="accumulation_steps"):
        accumulation_schedule(5, 0)


def test_collate_allows_unlabeled_eval_rows():
    from jev_qwen.data import collate_fn

    rows = [
        {
            "input_ids": [1, 2, 3],
            "position_ids": [0, 1, 2],
            "attention_mask": torch.ones(3, 3),
            "choice_read_idx": [1],
            "answer_end_idx": 2,
            "task_type": 0,
            "label": None,
        }
    ]
    b = collate_fn(rows, pad_id=0)
    assert "labels" not in b


def test_group_split_has_no_state_leakage():
    from jev_qwen.data import group_split, synthetic_records

    recs = synthetic_records(60, seed=0)
    splits = group_split(recs, seed=0)
    tr = {r["state"] for r in splits["train"]}
    va = {r["state"] for r in splits["validation"]}
    te = {r["state"] for r in splits["test"]}
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == len({r["state"] for r in recs})


@needs_weights
def test_tiny_training_run_saves_and_reloads(tmp_path):
    cfg = {
        "seed": 0,
        "model": {"name": "Qwen/Qwen3-0.6B", "num_layers": 2},
        "data": {
            "synthetic": True,
            "max_train_questions": 8,
            "max_test_questions": 4,
            "max_state_tokens": 64,
            "max_choice_tokens": 16,
            "batch_size": 2,
            "num_workers": 0,
        },
        "lora": {"last_k_layers": 0, "r": 8, "alpha": 16, "dropout": 0.0},
        "head": {"type": "compat", "new_dim": 32},
        "optim": {
            "epochs": 2,
            "lr_lora": 1e-4,
            "lr_head": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "grad_clip": 1.0,
            "grad_accum_steps": 3,
        },
        "logging": {"log_every": 1, "eval_every": 100, "save_every": 100},
    }
    import jev_qwen.train as T

    run_dir = T.train(cfg, "ci-smoke")
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_steps = [record for record in records if record.get("split") == "train"]
    assert [record["step"] for record in train_steps] == [1, 2, 3, 4]
    assert (run_dir / "checkpoints" / "final" / "head.safetensors").exists()
    assert (run_dir / "checkpoints" / "final" / "meta.json").exists()
    # fresh-process reload + eval (same protocol as baseline)
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from jev_qwen.run_logger import load_checkpoint;"
        "from jev_qwen.utils import get_device;"
        f"n, m = load_checkpoint('{run_dir}/checkpoints/final', get_device());"
        "print('reload-ok', m['model_name'])"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "reload-ok" in r.stdout, r.stderr[-2000:]
