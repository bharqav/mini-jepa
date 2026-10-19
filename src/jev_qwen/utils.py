"""Shared utilities: seeding, devices, timing, environment fingerprint."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from contextlib import contextmanager
from typing import Any

import torch


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device, precision: str = "bf16"):
    """bf16 preferred on CUDA (no loss scaling needed); fp16 fallback; off on CPU."""
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return torch.autocast("cuda", dtype=dtype)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@contextmanager
def cuda_sync_timing():
    """Synchronize CUDA around a timed block so GPU timings are honest."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield lambda: (
        torch.cuda.synchronize() if torch.cuda.is_available() else None,
        (time.perf_counter() - t0),
    )[1]


def env_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        try:
            free, total = torch.cuda.mem_get_info(0)
            info["vram_total_mb"] = total // (1024 * 2)
            info["vram_free_mb"] = free // (1024 * 2)
        except Exception:
            pass
    try:
        import transformers, peft, datasets  # noqa

        info["transformers"] = transformers.__version__
        info["peft"] = peft.__version__
        info["datasets"] = datasets.__version__
    except Exception:
        pass
    return info


def sha_short(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
