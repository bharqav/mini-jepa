# Experiments

Numbers live in `results/` (`notes.md`, `comparison.csv`,
`final_metrics.json`) so they are version-controlled; raw per-run logs stay
in the git-ignored `runs/`. Everything was produced by executable scripts, nothing is estimated.

## Headline (GPU: RTX 5060 via WSL2, 20k questions, 12 layers + LoRA, 3 epochs)
Held-out test split of 2,280 questions; the config evaluates the first 2,000
after a fixed shuffle, and that subset is identical across seeds. Numbers are
mean ± sample std across seeds 0, 1 and 2 (seed 0 is the original run
re-evaluated with the pinned revisions and current loader).

| system | acc | NLL | Brier | ECE | choice | noul | score | macro-F1 | AUC | order TV |
|---|---|---|---|---|---|---|---|---|---|---|
| majority |, |, |, |, | 0.293 | 0.544 | 0.269 |, |, |, |
| next-token (frozen) | 0.401 | 2.960 |, |, |, |, |, |, |, |, |
| zero-shot generation (full 28L, n=50) | 0.520 |, |, |, |, |, |, |, |, |, |
| reference head, 3 ep | 0.629 ± 0.031 | 0.799 ± 0.052 | 0.456 | 0.029 | 0.595 | 0.709 | 0.477 | 0.640 | 0.779 | 2.4e-6 |
| **compat head, 3 ep** | **0.667 ± 0.053** | **0.734 ± 0.084** | **0.422** | 0.041 | **0.662** | **0.718** | **0.510** | **0.730** | **0.808** | **9.0e-8** |

Per-seed accuracy: compat 0.6065 / 0.6935 / 0.7015; reference 0.5950 /
0.6545 / 0.6365. The compat gap is +0.0385 (SE 0.0352, t ≈ 1.10), **not
significant at 95%**. Compat is higher on accuracy, NLL, Brier, choice, noul,
score, macro-F1 and AUC; the reference head calibrates better at 3 epochs
(ECE 0.029 vs 0.041).

The earlier single-seed table (0.600 vs 0.581) was an unlucky seed-0 draw for
both heads and overstated the 3-epoch margin. At 1 epoch the compat margin is
+0.054 and significant (3 seeds, t = 3.26); at 3 epochs it is not established.
Epoch scaling for compat (seed 0 only): 1 ep 0.559 → 3 ep 0.600 → 5 ep 0.614,
ECE 0.048 → 0.082 at 5 epochs (fitted T = 1.95). Three epochs remains the
default on the seed-0 NLL/calibration ablation, but the epoch choice has not
been rerun across seeds, and the 5-epoch single run should not be compared
directly with the 3-seed 3-epoch mean.

## Early CPU ablation (300 questions, frozen), kept for the record
Established the head ranking before scaling: compat NLL 1.310 vs ref 1.677,
AUC 0.768 vs 0.544, choice still below majority. This scale could not
separate the heads on accuracy, which is why Episode 2 exists.

## KEPT / REJECTED improvements
- **KEPT** compat head: better ranking, better 1-epoch calibration, near-exact
  invariance (≤1e-7 TV), cheaper than the reference head; at 3 epochs the
  reference head calibrates better.
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
On the 1-epoch compat run, temperature scaling was fitted on a seeded 50% half
of the eval rows and measured on the other half (T = 1.014, NLL 0.933 → 0.922,
ECE → 0.032). Reliability and risk-coverage SVGs are emitted per checkpoint.
Selective prediction works: 0.688 accuracy at 50% coverage vs 0.572 overall.

## Efficiency (same GPU)
21 ms median single-question latency, 369 q/s at batch 16, 3.3 GB peak
training VRAM, 1.3 GB inference VRAM, zero tokens generated. Same-seed
reruns are bit-identical; different seeds are not (the 3-epoch compat
accuracy spans 0.6065-0.7015).

## What would come next
A 5-epoch seed sweep (to separate epoch scaling from seed noise), larger LoRA
rank, harder subsets (`hard_50k`), packed multi-question execution, and an
external teacher with validation, all implemented or configured, none yet
measured.