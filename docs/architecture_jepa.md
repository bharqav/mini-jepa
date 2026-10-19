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
