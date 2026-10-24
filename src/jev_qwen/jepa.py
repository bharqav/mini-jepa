"""JEPA-style latent predictor on top of Qwen3 blocks (experimental, non-canonical).

Canonical JEPA ingredients (I-JEPA, Assran et al. 2023) reproduced here:
  1. Context encoder  f_theta   : trainable, sees masked input.
  2. Target encoder  f_theta_bar: same architecture, independent params,
     NEVER trained by gradient; updated by exponential moving average (EMA)
     of the context encoder, forward pass under torch.no_grad + stop-grad.
  3. Latent predictor g_phi: maps (context reps + mask tokens) -> predicted
     target reps at masked positions. Loss is computed IN REPRESENTATION
     SPACE (MSE / cosine), never next-token cross-entropy.
  4. Block masking: targets are contiguous spans; the context encoder input
     has those spans replaced by a learned mask vector, so the predictor
     cannot trivially copy target content through the context pathway.
  5. Collapse prevention: stop-grad + EMA (primary) plus VICReg-style
     variance hinge and covariance off-diagonal penalties on the predicted
     representations (secondary, configurable). Diagnostics report latent
     std, collapse ratio, covariance norm, predictor error, similarities.

Where this is NOT canonical JEPA (documented adaptations for a decoder-only
text backbone — do not claim otherwise):
  A. Qwen3 is a causal decoder-only transformer, not a bidirectional ViT.
     Default context/target attention is causal (a "causal JEPA-inspired
     variant"); bidirectional context attention is offered as an explicit
     experimental option, not a canonical claim.
  B. The target encoder is a copy of the first N Qwen blocks (configurable),
     not the full 28-layer model, to fit 8 GB VRAM. Which blocks/targets are
     used is recorded in every checkpoint's meta.json.
  C. Prediction targets are last-hidden-state vectors at masked positions.
     Alternatives (mean-pooled span reps, intermediate-layer reps) are
     configurable but only the default is smoke-tested here.
  D. Text is discrete: masking replaces token EMBEDDINGS with a learned mask
     vector (not pixel masking). Target-span position ids are preserved so
     the predictor knows WHERE to predict without seeing WHAT is there.

The original Qwen checkpoint is never modified in place: both encoders are
initialized from it via from_pretrained into separate modules, and only the
context encoder + predictor receive gradients.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class JEPAConfig:
    hidden_dim: int = 1024  # Qwen3-0.6B hidden size
    num_layers: int = 8  # first-N Qwen blocks reused per encoder
    predictor_layers: int = 2
    predictor_dim: int = 512
    mask_ratio: float = 0.20  # fraction of state tokens masked as targets
    mask_span: int = 4  # contiguous span length (block masking)
    ema_momentum: float = 0.996  # target encoder EMA momentum
    loss_mse: float = 1.0
    loss_cosine: float = 0.0
    var_weight: float = 1.0  # VICReg variance hinge weight
    cov_weight: float = 0.04  # VICReg covariance weight
    var_target_std: float = 1.0  # hinge: per-dim std should reach this
    bidirectional_context: bool = False  # experimental non-canonical option
    freeze_target: bool = False  # keep the pretrained target encoder fixed
    seed: int = 0


# ----------------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------------


def sample_target_spans(
    seq_len: int, mask_ratio: float, span: int, rng: torch.Generator
) -> torch.BoolTensor:
    """Sample contiguous target spans covering ~mask_ratio of positions.

    Returns boolean mask of shape [seq_len]; True = target position.
    The caller must REPLACE these positions in the context input (isolation).
    """
    n_target = max(span, int(round(seq_len * mask_ratio)))
    target = torch.zeros(seq_len, dtype=torch.bool)
    starts = torch.randperm(seq_len, generator=rng).tolist()
    covered = 0
    for s in starts:
        if covered >= n_target:
            break
        e = min(seq_len, s + span)
        new = int((~target[s:e]).sum())
        target[s:e] = True
        covered += new
    return target


def apply_context_masking(
    input_embeds: torch.Tensor, target_mask: torch.Tensor, mask_vector: torch.Tensor
) -> torch.Tensor:
    """Replace embeddings at target positions with the learned mask vector.

    This is the isolation boundary: after this op, the context pathway holds
    NO content information about target spans (only their positions, via
    position ids passed separately to the predictor).
    """
    out = input_embeds.clone()
    out[target_mask] = mask_vector.to(out.dtype)
    return out


# ----------------------------------------------------------------------------
# Backbones
# ----------------------------------------------------------------------------


class TinyCausalBlock(nn.Module):
    """Minimal decoder block used when Qwen weights are unavailable (CPU smoke
    tests, offline CI). Same interface contract as a Qwen block subset."""

    def __init__(self, dim: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        s = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(x.shape[-1])
        if attn_bias is not None:
            s = s + attn_bias
        else:  # causal default
            L = x.shape[1]
            causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
            s = s + causal
        x = x + self.o(torch.softmax(s, dim=-1) @ v)
        return x + self.mlp(self.n2(x))


class EncoderStack(nn.Module):
    """Embedding + N transformer blocks + final norm.

    Qwen path: wraps a TRUNCATED Qwen3Model (first `num_layers` decoder
    layers) and calls its own forward — the exact tested convention (RoPE,
    GQA shapes, mask handling), so no manual block surgery can drift.
    If None, a randomly initialized tiny stack is built (offline experiments).
    """

    def __init__(
        self,
        vocab: int = 512,
        dim: int = 128,
        num_layers: int = 2,
        qwen_model=None,
        max_len: int = 512,
    ):
        super().__init__()
        self.is_qwen = qwen_model is not None
        if self.is_qwen:
            self.model = qwen_model
            self.dim = qwen_model.config.hidden_size
        else:
            self.model = None
            self.embed = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([TinyCausalBlock(dim) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(dim)
            self.dim = dim
        self.pos = nn.Embedding(max_len, self.dim)  # learned positions for predictor-side use

    @property
    def num_blocks(self) -> int:
        return self.model.config.num_hidden_layers if self.is_qwen else len(self.blocks)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.is_qwen:
            return self.model.embed_tokens(input_ids)
        return self.embed(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        bidirectional: bool = False,
    ) -> torch.Tensor:
        if self.is_qwen:
            L = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            dev = (input_ids if input_ids is not None else inputs_embeds).device
            dtype = (
                inputs_embeds.dtype
                if inputs_embeds is not None
                else self.model.embed_tokens(input_ids).dtype
            )
            if position_ids is None:
                position_ids = torch.arange(L, device=dev).expand(
                    (input_ids if input_ids is not None else inputs_embeds).shape[0], L
                )
            if bidirectional:
                bias = torch.zeros(
                    1,
                    1,
                    L,
                    L,
                    device=dev,
                    dtype=dtype,
                )
            else:
                c = torch.triu(
                    torch.full((L, L), float("-inf"), device=dev, dtype=dtype),
                    diagonal=1,
                )
                bias = c.view(1, 1, L, L)
            out = self.model(
                input_ids=input_ids,
                attention_mask=bias,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            return out.last_hidden_state
        if inputs_embeds is None:
            assert input_ids is not None
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        if position_ids is not None:
            x = x + self.pos(position_ids % self.pos.num_embeddings)
        L = x.shape[1]
        if bidirectional:
            zero3d = torch.zeros(x.shape[0], L, L, device=x.device)
            for blk in self.blocks:
                x = blk(x, zero3d)
            return self.norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def truncate_qwen(qwen_model, num_layers: int):
    """Keep the first `num_layers` decoder blocks IN PLACE (caller must pass a
    model it owns). Returns the same model for chaining."""
    del qwen_model.layers[num_layers:]
    qwen_model.config.num_hidden_layers = num_layers
    return qwen_model


# ----------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------


class LatentPredictor(nn.Module):
    """Predicts target reps from visible context reps + mask tokens.

    Only ever receives the MASKED context encoding plus learned mask tokens
    carrying target position ids — never target-encoder outputs.
    """

    def __init__(self, dim: int, predictor_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.randn(predictor_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=predictor_dim,
                    nhead=8,
                    dim_feedforward=predictor_dim * 2,
                    batch_first=True,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(predictor_dim, dim)

    def forward(self, ctx_reps: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(ctx_reps)
        h[target_mask] = self.mask_token.to(h.dtype)
        for layer in self.layers:
            h = layer(h)
        return self.out_proj(h)


# ----------------------------------------------------------------------------
# Full JEPA module
# ----------------------------------------------------------------------------


class JEPAState(nn.Module):
    """Context encoder + EMA target encoder + latent predictor + mask vector."""

    def __init__(
        self,
        cfg: JEPAConfig,
        qwen_model=None,
        vocab: int = 512,
        tiny_dim: int = 128,
        tiny_layers: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        if qwen_model is not None:
            truncate_qwen(qwen_model, cfg.num_layers)
            self.context = EncoderStack(qwen_model=qwen_model)
            # Independent parameter set: deep-copy the TRUNCATED model so no
            # storage is shared with the context encoder.
            self.target = EncoderStack(qwen_model=copy.deepcopy(qwen_model))
        else:
            kw = {"vocab": vocab, "dim": tiny_dim, "num_layers": tiny_layers, "qwen_model": None}
            self.context = EncoderStack(**kw)
            self.target = copy.deepcopy(self.context)
        self.target.requires_grad_(False)
        self.freeze_target = cfg.freeze_target
        dim = self.context.dim
        self.predictor = LatentPredictor(dim, cfg.predictor_dim, cfg.predictor_layers)
        self.ctx_mask_vector = nn.Parameter(torch.randn(dim) * 0.02)
        assert not shares_parameters(self.context, self.target), (
            "target encoder must own independent parameters"
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.context.embed_tokens(input_ids)

    @torch.no_grad()
    def ema_update(self, momentum: float | None = None) -> None:
        if self.freeze_target:
            return
        m = self.cfg.ema_momentum if momentum is None else momentum
        for pc, pt in zip(self.context.parameters(), self.target.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)

    def forward(
        self, input_ids: torch.Tensor, rng: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        B, L = input_ids.shape
        rng = rng or torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        # Per-sample block masks (loop is fine at this scale; keeps spans exact).
        tmasks = torch.stack(
            [sample_target_spans(L, self.cfg.mask_ratio, self.cfg.mask_span, rng) for _ in range(B)]
        )
        tmasks = tmasks.to(input_ids.device)
        pos = torch.arange(L, device=input_ids.device).expand(B, L)

        # TARGET pathway: unmasked input, no grad — the stop-gradient boundary.
        with torch.no_grad():
            tgt_emb = self.target.embed_tokens(input_ids)
            tgt_reps = self.target(
                inputs_embeds=tgt_emb,
                position_ids=pos,
                bidirectional=self.cfg.bidirectional_context,
            )

        # CONTEXT pathway: masked input, gradients flow here + predictor only.
        ctx_emb = apply_context_masking(
            self.context.embed_tokens(input_ids), tmasks, self.ctx_mask_vector
        )
        ctx_reps = self.context(
            inputs_embeds=ctx_emb, position_ids=pos, bidirectional=self.cfg.bidirectional_context
        )
        pred_reps = self.predictor(ctx_reps, tmasks)
        loss, parts = jepa_loss(pred_reps, tgt_reps.detach(), tmasks, self.cfg)
        return {
            "loss": loss,
            **parts,
            "target_mask": tmasks,
            "pred": pred_reps,
            "target": tgt_reps.detach(),
            "context": ctx_reps,
        }

    def lm_baseline_loss(self, input_ids: torch.Tensor, lm_head: nn.Linear) -> torch.Tensor:
        """Causal-LM objective on the SAME batch for the controlled comparison:
        next-token cross-entropy from context-encoder reps (no masking)."""
        reps = self.context(input_ids=input_ids)
        return F.cross_entropy(
            lm_head(reps[:, :-1]).reshape(-1, lm_head.out_features), input_ids[:, 1:].reshape(-1)
        )


def shares_parameters(a: nn.Module, b: nn.Module) -> bool:
    ids = {id(p) for p in a.parameters()}
    return any(id(p) in ids for p in b.parameters())


# ----------------------------------------------------------------------------
# Loss + collapse-prevention terms
# ----------------------------------------------------------------------------


def jepa_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: JEPAConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p = pred[mask]  # [M, D]: loss is computed ONLY at target positions
    t = target[mask]
    mse = F.mse_loss(p, t)
    cos = torch.tensor(0.0, device=p.device)
    if cfg.loss_cosine > 0:
        cos = (1.0 - F.cosine_similarity(p, t, dim=-1)).mean()
    var = variance_hinge(p, cfg.var_target_std)
    cov = covariance_penalty(p)
    loss = cfg.loss_mse * mse + cfg.loss_cosine * cos + cfg.var_weight * var + cfg.cov_weight * cov
    return loss, {
        "mse": mse.detach(),
        "cosine": cos.detach(),
        "var_term": var.detach(),
        "cov_term": cov.detach(),
    }


def variance_hinge(x: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = x.std(dim=0) + eps
    return F.relu(target_std - std).mean()


def covariance_penalty(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off**2).sum() / x.shape[1]


# ----------------------------------------------------------------------------
# Diagnostics (collapse detection)
# ----------------------------------------------------------------------------


def collapse_diagnostics(reps: torch.Tensor, eps: float = 1e-3) -> dict[str, float]:
    """Input: representation matrix [N, D] (predicted or target reps)."""
    with torch.no_grad():
        std = reps.float().std(dim=0)
        xc = reps.float() - reps.float().mean(dim=0, keepdim=True)
        cov = (xc.T @ xc) / max(reps.shape[0] - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return {
            "mean_std": float(std.mean()),
            "min_std": float(std.min()),
            "collapse_ratio": float((std < eps).float().mean()),
            "cov_offdiag_norm": float((off**2).sum().sqrt()),
            "rep_norm": float(reps.float().norm(dim=-1).mean()),
        }


def representation_similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """How close are predictor outputs to targets? High cosine + low MSE with
    HIGH variance/diagnostics = learning; low MSE with collapsed reps = fraud."""
    with torch.no_grad():
        mse = float(F.mse_loss(a.float(), b.float()))
        cos = float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean())
        # linear CKA (simplified, batch-level)
        xa = a.float() - a.float().mean(0, keepdim=True)
        xb = b.float() - b.float().mean(0, keepdim=True)
        cka = float(
            (torch.norm(xb.T @ xa, p="fro") ** 2)
            / max(float(torch.norm(xa.T @ xa, p="fro") * torch.norm(xb.T @ xb, p="fro")), 1e-12)
        )
        return {"pred_target_mse": mse, "pred_target_cosine": cos, "linear_cka": min(cka, 1.0)}
