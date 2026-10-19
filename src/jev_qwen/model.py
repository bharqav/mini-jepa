"""Backbone + decision heads. Reference-faithful + improved CompatHead.

Reference head (ChoiceHead): self-attention over [task, choices..., answer],
then bilinear score answer.proj vs each choice.proj. Permutation-equivariant
in exact arithmetic (set attention has no order terms), but inter-candidate
edges + fp rounding can leak tiny order effects; we test the tolerance.

Improved head (CompatHead): NO inter-candidate attention. Each candidate is
scored independently against the answer vector with SHARED projections:
  score_i = (Wq a . Wk c_i)/sqrt(d) + task_bias[t]
Strictly order-invariant by construction (score_i depends only on (a, c_i)),
fewer FLOPs, and candidates remain comparable via shared Wq/Wk.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, new_dim: int):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.new_dim)
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        return self.o(torch.einsum("bqk,bkd->bqd", torch.softmax(scores, -1), v))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, new_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, new_dim), nn.GELU(), nn.Linear(new_dim, hidden_dim)
        )

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        return x + self.mlp(self.norm2(x))


class ChoiceHead(nn.Module):
    """Reference-faithful head."""

    def __init__(self, hidden_dim: int, new_dim: int, num_layers: int, num_task_types: int):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [AttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        B = choice_embeddings.shape[0]
        x = torch.cat(
            [
                self.task_embedding(task_type).unsqueeze(1),
                choice_embeddings,
                answer_embedding.unsqueeze(1),
            ],
            dim=1,
        )
        x = self.input_norm(x)
        pad = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=x.device),
                choice_mask,
                torch.ones(B, 1, dtype=torch.bool, device=x.device),
            ],
            dim=1,
        )
        for layer in self.layers:
            x = layer(x, pad)
        x = self.final_norm(x)
        logits = torch.einsum(
            "bd,bcd->bc", self.answer_proj(x[:, -1]), self.choice_proj(x[:, 1:-1])
        ) / math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))


class CompatHead(nn.Module):
    """Improved head: independent candidate compatibility with shared params."""

    def __init__(self, hidden_dim: int, new_dim: int, num_task_types: int):
        super().__init__()
        self.new_dim = new_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, new_dim)
        self.k_proj = nn.Linear(hidden_dim, new_dim)
        self.task_bias = nn.Embedding(num_task_types, 1)
        self.log_temp = nn.Parameter(torch.zeros(num_task_types))

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        h = self.norm(torch.cat([answer_embedding.unsqueeze(1), choice_embeddings], dim=1))
        a = self.q_proj(h[:, :1])
        c = self.k_proj(h[:, 1:])
        logits = torch.einsum("bqd,bcd->bc", a, c).squeeze(1) / math.sqrt(self.new_dim)
        logits = logits + self.task_bias(task_type).squeeze(-1).unsqueeze(1)
        temp = torch.exp(self.log_temp[task_type]).unsqueeze(1).clamp(0.05, 20.0)
        return (logits / temp).masked_fill(~choice_mask, -float("inf"))


class DecisionNetwork(nn.Module):
    def __init__(self, backbone, head, gradient_checkpointing: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            self.head.gradient_checkpointing = True

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        choice_read_idx,
        choice_mask,
        answer_end_idx,
        task_type,
        **kwargs,
    ):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = out.last_hidden_state.float()
        bi = torch.arange(hidden.shape[0], device=hidden.device)
        ce = hidden[bi[:, None], choice_read_idx]
        ae = hidden[bi, answer_end_idx]
        # Head always runs in fp32: bf16 logits are too coarse for a stable loss
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(ce, ae, choice_mask, task_type)


def load_backbone(model_name: str, num_layers: int | None = None):
    from transformers import Qwen3Model

    ov = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return Qwen3Model.from_pretrained(model_name, dtype=torch.float32, **ov)


def lora_target_regex(num_layers: int, last_k: int) -> str:
    ids = "|".join(str(i) for i in range(num_layers - last_k, num_layers))
    return rf".*layers\.({ids})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def build_head(head_cfg: dict, hidden_dim: int, num_task_types: int = 3) -> nn.Module:
    t = head_cfg.get("type", "reference")
    if t == "compat":
        return CompatHead(hidden_dim, head_cfg.get("new_dim", 512), num_task_types)
    return ChoiceHead(
        hidden_dim, head_cfg.get("new_dim", 512), head_cfg.get("num_layers", 2), num_task_types
    )


def build_network(
    model_name,
    num_layers,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_last_k_layers,
    head_config,
    gradient_checkpointing: bool = False,
):
    from peft import LoraConfig, get_peft_model

    backbone = load_backbone(model_name, num_layers)
    if lora_last_k_layers == 0:
        backbone.requires_grad_(False)
    else:
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_regex(
                    backbone.config.num_hidden_layers, lora_last_k_layers
                ),
            ),
        )
    return DecisionNetwork(
        backbone,
        build_head(head_config, backbone.config.hidden_size),
        gradient_checkpointing=gradient_checkpointing,
    )
