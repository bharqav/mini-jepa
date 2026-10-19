"""Small working latent-prediction experiment (CPU, no downloads).

Compares, on IDENTICAL synthetic batches:
  (a) causal-LM baseline objective (next-token cross-entropy from a linear head
      on context-encoder reps), vs
  (b) the JEPA latent-prediction objective (masked-span rep prediction).

Writes runs/jepa_smoke/metrics.json with genuine measured numbers:
predictor MSE trajectory, LM loss trajectory, collapse diagnostics, EMA drift.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_qwen.jepa import JEPAConfig, JEPAState, collapse_diagnostics, representation_similarity

VOCAB, SEQ, BATCH, STEPS = 64, 32, 8, 60


def synth_batch(rng):
    # structured pseudo-language: grammar positions correlate, so there is
    # learnable latent structure but no trivial memorization
    base = torch.randint(0, VOCAB, (BATCH, SEQ // 4), generator=rng)
    return base.repeat_interleave(4, dim=1)[:, :SEQ]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bidirectional",
        action="store_true",
        help="experimental non-canonical bidirectional context attention",
    )
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-name", default="jepa_smoke", help="runs/<out-name>/metrics.json")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    cfg = JEPAConfig(
        hidden_dim=32,
        num_layers=2,
        predictor_layers=1,
        predictor_dim=32,
        mask_ratio=0.25,
        mask_span=4,
        ema_momentum=0.99,
        seed=0,
        bidirectional_context=args.bidirectional,
    )
    st = JEPAState(cfg, qwen_model=None, vocab=VOCAB, tiny_dim=32, tiny_layers=2)
    lm_head = nn.Linear(32, VOCAB)
    opt = torch.optim.AdamW(
        list(st.context.parameters())
        + list(st.predictor.parameters())
        + list(lm_head.parameters()),
        lr=3e-3,
    )
    rng = torch.Generator().manual_seed(7)
    hist = []
    for step in range(args.steps):
        ids = synth_batch(rng)
        out = st(ids)
        lm = st.lm_baseline_loss(ids, lm_head)
        (out["loss"] + 0.1 * lm).backward()
        opt.step()
        opt.zero_grad()
        st.ema_update()
        if step % 10 == 0 or step == args.steps - 1:
            m = out["target_mask"]
            sim = representation_similarity(out["pred"][m], out["target"][m])
            d = collapse_diagnostics(out["pred"][m])
            hist.append(
                {
                    "step": step,
                    "jepa_loss": float(out["loss"].detach()),
                    "mse": float(out["mse"]),
                    "lm_loss": float(lm.detach()),
                    **sim,
                    **d,
                }
            )
            print(
                f"step={step} jepa={float(out['loss'].detach()):.4f} mse={float(out['mse']):.4f} "
                f"lm={float(lm.detach()):.4f} cos={sim['pred_target_cosine']:.3f} "
                f"mean_std={d['mean_std']:.3f} collapse={d['collapse_ratio']:.2f}"
            )
    # EMA drift: target must have moved toward context but not equaled it
    drift = torch.stack(
        [
            (pt - pc.detach()).abs().mean()
            for pt, pc in zip(st.target.parameters(), st.context.parameters())
        ]
    ).mean()
    out_dir = Path(__file__).resolve().parents[1] / "runs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "bidirectional": args.bidirectional,
                "final": hist[-1],
                "history": hist,
                "ema_mean_abs_drift": float(drift),
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
