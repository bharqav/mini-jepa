# Experiments

Numbers live in `results/` (`notes.md`, `comparison.csv`,
`final_metrics.json`) so they are version-controlled; raw per-run logs stay
in the git-ignored `runs/`. Everything was produced by executable scripts, nothing is estimated.

## Headline (GPU: RTX 5060 via WSL2, 20k questions, 12 layers + LoRA, 3 epochs)
Held-out test split, n=2,280 questions.

| system | acc | NLL | Brier | ECE | choice | noul | score | macro-F1 | AUC | order TV |
|---|---|---|---|---|---|---|---|---|---|---|
| majority |, |, |, |, | 0.293 | 0.544 | 0.269 |, |, |, |
| next-token (frozen) | 0.401 | 2.960 |, |, |, |, |, |, |, |, |
| zero-shot generation (full 28L, n=50) | 0.520 |, |, |, |, |, |, |, |, |, |
| reference head, 3 ep | 0.581 | 0.906 | 0.514 | 0.047 | 0.545 | 0.645 | 0.510 | 0.540 | 0.712 | 3.4e-6 |
| **compat head, 3 ep** | **0.600** | **0.874** | **0.493** | **0.048** | **0.567** | **0.665** | **0.514** | **0.544** | **0.743** | **1.2e-7** |

At equal epochs the compat head wins every metric, but the honest reading is
that **the head margin narrows with training**: +0.054 accuracy at 1 epoch,
+0.019 at 3 epochs, because the reference head gains more from extra epochs
(+0.076 vs +0.042). Part of the early margin was compensating for
undertraining. Epoch scaling for the compat head: 1 ep 0.559 → 3 ep 0.600 →
5 ep 0.614, with ECE degrading 0.048 → 0.082 at 5 epochs (overconfidence,
fitted T = 1.95), so **3 epochs is the default**.

## Early CPU ablation (300 questions, frozen), kept for the record
Established the head ranking before scaling: compat NLL 1.310 vs ref 1.677,
AUC 0.768 vs 0.544, choice still below majority. This scale could not
separate the heads on accuracy, which is why Episode 2 exists.

## KEPT / REJECTED improvements
- **KEPT** compat head: better ranking, better calibration, exact invariance,
  cheaper than the reference head.
- **KEPT** LoRA on the top kept layers: choice 0.225 → 0.300 at CPU scale,
  the single biggest lever found.
- **KEPT** learned per-task temperature (compat head).
- **REJECTED as unnecessary**: order augmentation is a provable no-op for
  the invariant head (bit-identical training loss); retained only because the
  reference head's recipe requires it. Candidate-order invariance is a
  property of the architecture, not of the augmentation.
- **NULL RESULT**: bidirectional JEPA context vs causal, MSE 1.1559 vs
  1.1613 at identical seeds; no measurable difference at toy scale.
- **WEAK TEACHER**: local 4-layer LM teacher agreed with dataset labels on
  1/5 audited rows; not used for training.

## Calibration
Temperature scaling fitted on a seeded 50% half of the eval rows, measured
on the other half (T = 1.014, NLL 0.933 → 0.922, ECE → 0.032). Reliability
and risk-coverage SVGs are emitted per checkpoint. Selective prediction
works: 0.688 accuracy at 50% coverage vs 0.572 overall.

## Efficiency (same GPU)
21 ms median single-question latency, 369 q/s at batch 16, 3.3 GB peak
training VRAM, 1.3 GB inference VRAM, zero tokens generated. Training is
reproducible to the bit: two runs of `improved.yaml` gave identical test
metrics.

## What would come next
Multi-epoch training, larger LoRA rank, harder subsets (`hard_50k`),
packed multi-question execution, and an external teacher with validation, all implemented or configured, none yet measured.