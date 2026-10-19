"""Run logging + self-contained checkpoints (LoRA adapter + head only;
base Qwen weights are referenced by id, never vendored)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


class RunLogger:
    def __init__(
        self,
        config: dict[str, Any],
        run_name: str,
        device,
        root: str = "runs",
        extra: dict[str, Any] | None = None,
    ):
        self.config = config
        self.exp_id = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
        self.run_dir = Path(root) / self.exp_id
        self.run_dir.mkdir(parents=True)
        self.start = time.time()
        from .utils import env_fingerprint

        run_info = {
            "exp_id": self.exp_id,
            "device": str(device),
            "started_at": datetime.now().isoformat(),
            "env": env_fingerprint(),
            **(extra or {}),
        }
        (self.run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        self.write_config(self.run_dir)
        print(f"Experiment {self.exp_id} -> {self.run_dir}", flush=True)

    def write_config(self, directory: Path) -> None:
        with open(directory / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def log(self, step: int, **metrics: Any) -> None:
        record = {"step": step, "time": round(time.time() - self.start, 1), **metrics}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record, default=float) + "\n")
        print(
            " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
            ),
            flush=True,
        )

    def save_checkpoint(self, network, name: str = "final", step: int | None = None) -> Path:
        from peft import PeftModel

        ckpt = self.run_dir / "checkpoints" / name
        ckpt.mkdir(parents=True, exist_ok=True)
        peak_vram = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            print(f"peak VRAM (allocated): {peak_vram} MB", flush=True)
        if isinstance(network.backbone, PeftModel):
            network.backbone.save_pretrained(ckpt / "lora")
        torch.save(
            {k: v.detach().cpu() for k, v in network.head.state_dict().items()}, ckpt / "head.pt"
        )
        self.write_config(ckpt)
        head_cfg = dict(self.config.get("head", {}))
        head_cfg["num_task_types"] = (
            network.head.task_embedding.num_embeddings
            if hasattr(network.head, "task_embedding")
            else 3
        )
        (ckpt / "meta.json").write_text(
            json.dumps(
                {
                    "exp_id": self.exp_id,
                    "step": step,
                    "model_name": self.config["model"]["name"],
                    "num_layers": network.backbone.config.num_hidden_layers,
                    "head_config": head_cfg,
                    "max_state_tokens": self.config["data"]["max_state_tokens"],
                    "max_choice_tokens": self.config["data"]["max_choice_tokens"],
                    "peak_vram_mb": peak_vram,
                },
                indent=2,
            )
        )
        print(f"Saved checkpoint -> {ckpt}", flush=True)
        return ckpt

    def finish(self, **summary: Any) -> None:
        summary["duration_sec"] = round(time.time() - self.start, 1)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


def load_checkpoint(ckpt_dir, device):
    """Rebuild network from a checkpoint folder (fresh process safe)."""
    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    with open(ckpt_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    from .model import build_network

    lora = cfg.get("lora", {})
    net = build_network(
        meta["model_name"],
        meta.get("num_layers"),
        lora.get("r", 8),
        lora.get("alpha", 16),
        lora.get("dropout", 0.0),
        lora.get("last_k_layers", 0),
        meta["head_config"],
        gradient_checkpointing=cfg.get("model", {}).get("gradient_checkpointing", False),
    )
    if (ckpt_dir / "lora").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        weights = load_file(ckpt_dir / "lora" / "adapter_model.safetensors")
        set_peft_model_state_dict(net.backbone, weights)
    net.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
    return net.to(device).eval(), meta
