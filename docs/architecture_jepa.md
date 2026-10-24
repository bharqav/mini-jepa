# JEPA-Style Latent Predictor on Qwen3-0.6B, Architecture Note

> Status: **JEPA-inspired experimental adaptation**, NOT canonical JEPA.
> Do not describe this as "implementing JEPA for language models" without
> the qualifier. Canonical JEPA results (I-JEPA) cover images with
> bidirectional ViT encoders; porting the idea to a causal decoder-only LM
> involves the non-canonical choices listed below.

## 1. JEPA principles used (from I-JEPA: Assran et al., 2023)

| Principle | Implementation (`src/jev_qwen/jepa.py`) |
|---|---|
| Context encoder on masked input | `EncoderStack` (first-N Qwen blocks), sees mask-vector spans |
| Target encoder, independent params | `copy.deepcopy` of context init; `requires_grad_(False)`; assert no shared storage |
| Stop-gradient / EMA target updates | target forward under `torch.no_grad`, `.detach()` targets; `ema_update(m)` |
| Latent predictor | `LatentPredictor`: mask tokens + visible reps → predicted target reps |
| Representation-space loss | MSE (+ optional cosine) **only at masked positions** |
| Block masking | `sample_target_spans`: contiguous spans, not i.i.d. tokens |
| Collapse prevention | stop-grad + EMA (primary); VICReg variance hinge + covariance penalty (secondary) |
| Diagnostics | `collapse_diagnostics` (mean/min std, collapse ratio, cov norm), `representation_similarity` (MSE, cosine, linear CKA) |

## 2. Non-canonical adaptations (text + decoder-only backbone)

1. **Causal attention default.** Qwen3 blocks are causal decoders. The default
   keeps causal masks in both encoders ("causal JEPA-inspired variant").
   `bidirectional_context: true` is an explicitly experimental option.
2. **Target encoder = first-N blocks**, not the full 28-layer model (VRAM).
   `num_layers` is recorded in checkpoint meta.
3. **Targets = last-hidden-state vectors** at masked positions (not pooled
   span reps or mid-layer reps, configurable later).
4. **Masking in embedding space** with a learned mask vector (text is
   discrete; there is no pixel masking equivalent).

## 3. Information-flow / isolation guarantees

- `apply_context_masking` replaces target-span embeddings BEFORE the context
  encoder; the predictor receives only masked-context reps + mask tokens.
- Target encoder input is the unmasked sequence under `no_grad`; its outputs
  are `.detach()`ed before the loss.
- Verified by tests: `test_target_encoder_gets_no_gradient`,
  `test_context_input_hides_target_content`,
  `test_predictor_cannot_trivially_copy`, `test_ema_update_exact`.

## 4. Qwen block reuse plan

- Context encoder: `embed_tokens` + layers `[0, N)` + `norm` from
  `Qwen/Qwen3-0.6B`; original checkpoint file never modified.
- Target encoder: deep copy at init, EMA-maintained thereafter.
- Full-copy target encoder (all 28 layers) is possible but exceeds the 8 GB
  budget alongside training states, N=8 default, N=12 max with LoRA/8-bit.

## 5. Baseline comparison protocol

`scripts/jepa_smoke.py` trains both objectives on identical batches and logs
`jepa_loss`/`mse` (latent) vs `lm_loss` (causal next-token CE) plus collapse
diagnostics to `runs/jepa_smoke/metrics.json`. A drop in MSE is only accepted
as learning if `collapse_ratio == 0` and `mean_std` stays healthy.

## 6. Real Qwen3-0.6B pretraining pipeline (implemented)

The toy smoke path above is complemented by a real-weight pipeline:

| Component | Location |
|---|---|
| Packed text dataset | `src/jev_qwen/jepa_data.py` (`record_text`, `pack_sequences`, `build_packed_dataset`, `make_batches`) |
| Pretraining/eval CLI | `scripts/train_jepa.py` |
| Causal EMA config | `configs/jepa_qwen.yaml` |
| Frozen-target config | `configs/jepa_qwen_frozen.yaml` |
| Frozen decision probe | `configs/jepa_probe.yaml`, `configs/jepa_probe_bidir.yaml` |
| Backbone init from JEPA | `src/jev_qwen/train.py:init_backbone_from_jepa`, `scripts/train.py --init-from-jepa` |

**Data path.** `record_text` concatenates state, instructions and candidate
text but deliberately drops the label. Streams are tokenized, shuffled with a
seeded RNG, and packed into fixed-length rows separated by EOS, so the JEPA
masking path never mistakes padding for content.

**Objective.** The context encoder sees block-masked embeddings; the target
encoder sees the unmasked sequence under `no_grad`. Targets are the first-N
Qwen decoder blocks' last hidden states. The default target is an EMA copy of
the context encoder (`ema_momentum: 0.996`). `freeze_target: true` instead
keeps the pretrained Qwen target fixed, which avoids target drift at the cost
of a harder, fixed prediction problem. `bidirectional_context: true` uses full
attention in both encoders; the default remains causal.

**Checkpoint.** `save_jepa` writes `context.safetensors`, `predictor.safetensors`,
`mask_vector.safetensors` and `meta.json`. The context state dict is the raw
truncated `Qwen3Model` state dict, so it can initialise the decision backbone
without conversion.

**Probe.** `scripts/train.py --init-from-jepa <run>/jepa/context.safetensors`
loads the context weights into a frozen decision backbone, then trains only the
decision head. The uninitialised baseline uses the same config without
`--init-from-jepa`. A `bidirectional: true` probe config is provided so a
bidirectional JEPA encoder can be evaluated with matching full attention.

**Measured result (single seed, RTX 5060, 8 layers, 2,000 steps).** The
pretrained-Qwen causal frozen probe reaches 0.5745 accuracy; the causal EMA
JEPA probe reaches 0.3940; the causal frozen-target JEPA probe reaches 0.4975.
With matching bidirectional attention, the pretrained baseline is 0.5390 and
the bidirectional JEPA probe is 0.4185. Latent eval MSE is 12.33 for causal
EMA, 12.38 for causal frozen-target, and 6.42 for bidirectional, but JEPA
initialisation does **not** improve this downstream decision probe in either
attention mode. This is a negative result for the current protocol, not
evidence that JEPA is better. Full metrics:
`results/jepa_qwen_results.json`.
