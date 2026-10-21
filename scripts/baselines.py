"""Non-neural + prompted baselines on the SAME test questions as the ablation.

1. next-token scoring: frozen Qwen3-0.6B, options listed as A/B/C..., read the
   logits of the letter tokens after 'Answer:'. One forward per question.
2. prompted generation: greedy generate (max 12 new tokens), parse first
   option letter. Subset only (slow on CPU) — count reported.
3. majority-class baseline per question type.

Writes runs/baselines/metrics.json. CPU-runnable.
"""

import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")

from jev_qwen.data import TASK_TYPES
from jev_qwen.hub import DEFAULT_MODEL_NAME, resolve_model_revision
from jev_qwen.tokenize import get_tokenizer
from jev_qwen.train import get_questions
from jev_qwen.utils import get_device

LETTERS = "ABCDEFGHIJKLMNOP"


def load_test(n=100, subset="all"):
    cfg = {"data": {"max_test_questions": n, "subset": subset}}
    return get_questions("test", cfg)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gen-n", type=int, default=20)
    ap.add_argument(
        "--full-model-gen",
        action="store_true",
        help="use the full 28-layer LM for generation (fair); "
        "scoring always uses the truncated 4-layer LM",
    )
    ap.add_argument("--subset", default="all", help="dataset config name")
    args = ap.parse_args()
    device = get_device()
    revision = resolve_model_revision(DEFAULT_MODEL_NAME)
    tok = get_tokenizer(DEFAULT_MODEL_NAME, revision=revision)
    from transformers import Qwen3ForCausalLM

    lm = (
        Qwen3ForCausalLM.from_pretrained(
            DEFAULT_MODEL_NAME,
            dtype=torch.float32,
            revision=revision,
            num_hidden_layers=4,
        )
        .to(device)
        .eval()
    )
    recs = load_test(args.n, args.subset)
    # next-token scoring (choice + noul only; score levels are ordered text, skip)
    scored = [r for r in recs if r["task_type"] in (TASK_TYPES["choice"], TASK_TYPES["noul"])]
    nt_correct, nt_n, nt_ll, t0 = 0, 0, 0.0, time.time()
    for r in scored:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        prompt = f"State: {r['state']}\nQuestion: {r['instructions']}\n{opts}\nAnswer:"
        ids = tok.encode(prompt)
        with torch.no_grad():
            logits = lm(torch.tensor([ids]).to(device)).logits[0, -1]
        letter_ids = [
            tok.encode(" " + L)[-1] if len(tok.encode(" " + L)) else tok.encode(L)[-1]
            for L in LETTERS[: len(r["choices"])]
        ]
        try:
            sub = torch.stack([logits[i] for i in letter_ids])
        except IndexError:
            continue
        probs = torch.softmax(sub.float(), -1)
        pred = int(probs.argmax())
        nt_correct += pred == r["label"]
        nt_ll += float(-torch.log(probs[r["label"]]))
        nt_n += 1
    nt = {
        "accuracy": nt_correct / max(nt_n, 1),
        "nll": nt_ll / max(nt_n, 1),
        "n": nt_n,
        "seconds": round(time.time() - t0, 1),
    }
    print("next-token:", json.dumps(nt))
    # prompted generation subset
    gen_subset = [r for r in scored if len(r["choices"]) <= 4][: args.gen_n]
    glm = lm
    if args.full_model_gen:
        del lm
        glm = (
            Qwen3ForCausalLM.from_pretrained(
                DEFAULT_MODEL_NAME, dtype=torch.float32, revision=revision
            )
            .to(device)
            .eval()
        )
    g_correct, g_n, t0 = 0, 0, time.time()
    for r in gen_subset:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        prompt = f"State: {r['state']}\nQuestion: {r['instructions']}\n{opts}\nAnswer with the letter only:"
        ids = torch.tensor([tok.encode(prompt)]).to(device)
        with torch.no_grad():
            out = glm.generate(
                ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
            )
        text = tok.decode(out[0][len(ids[0]) :])
        m = re.search(r"[A-D]", text)
        if m:
            g_correct += LETTERS.index(m.group(0)) == r["label"]
            g_n += 1
    gen = {
        "accuracy": g_correct / max(g_n, 1),
        "parsed_n": g_n,
        "of_n": len(gen_subset),
        "seconds": round(time.time() - t0, 1),
    }
    print("generate:", json.dumps(gen))
    # majority baseline
    from collections import Counter

    maj = {}
    for t in ("choice", "noul", "score"):
        labs = [r["label"] for r in recs if r["task_type"] == TASK_TYPES[t]]
        maj[t] = (Counter(labs).most_common(1)[0][1] / len(labs)) if labs else 0.0
    out = Path("runs/baselines")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "next_token": nt,
                "generate": gen,
                "majority_per_type": maj,
                "n_test": len(recs),
                "device": str(device),
            },
            indent=2,
        )
    )
    print("majority:", json.dumps(maj))


if __name__ == "__main__":
    main()
