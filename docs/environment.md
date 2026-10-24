# Environment notes

Target: Windows 11, RTX 5060 Laptop 8.1 GB, CUDA 13.x driver.

## GPU execution: WSL2 is the supported path on this host

Windows application control (WinError 4551) blocks **every CUDA PyTorch
wheel** on this machine (`cu128` and `cu126`, verified across clean
reinstalls, same policy also blocks scikit-learn's OpenMP `.pyd`s and
pyarrow). The GPU path therefore runs inside WSL2, which is unaffected:

```bash
# in WSL (Ubuntu 26.04)
python3 -m venv ~/jev-wsl/.venv
~/jev-wsl/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/jev-wsl/.venv/bin/pip install -r requirements/base.txt
cd ~/jev-wsl && ~/jev-wsl/.venv/bin/python scripts/train.py configs/improved.yaml --name run
```

```mermaid
flowchart LR
    A[Windows host\nWinError 4551\nblocks CUDA wheels] --> B[WSL2 Ubuntu\nunaffected by policy\nNVIDIA driver 590.62]
    B --> C[PyTorch cu128 wheel\ninstalled in WSL venv\nCUDA_VISIBLE_DEVICES=0]
    C --> D[RTX 5060 Laptop\n8.1 GB VRAM\nCUDA 13.1 driver]
    D --> E[Training + inference\nbf16, 3.3 GB peak train\n1.3 GB peak inference]
```

GPU passthrough confirmed via `/usr/lib/wsl/lib/nvidia-smi`. All numbers in
`results/` were produced this way. Verified idle-GPU benchmark (10 repeats,
CUDA-synchronised, 0% utilisation): batch-1 median 28.0 ms, batch-4 median
37.9 ms, batch-8 median 29.1 ms, peak inference VRAM 1,348.9 MB allocated
/ 1,366.0 MB reserved on RTX 5060 (8.1 GB).

## Windows-native path (CPU, fully functional)

Verified working: torch CPU, transformers, peft, tokenizers, safetensors,
numpy, pandas, fastparquet, fastapi/uvicorn, Qwen3-0.6B weights, the full
test suite, smoke training, eval, CLI and API. Two fallbacks make this work
without scikit-learn/pyarrow: `macro_f1` has a pure-torch implementation,
and dataset loading falls back to `huggingface_hub` + `fastparquet`.
Both activate automatically only when needed; normal machines use the
standard libraries.

## Notes for other machines
- On an unconstrained GPU box, the Windows-native CUDA command works:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- `configs/baseline.yaml` / `improved.yaml` are tuned for 8 GB: micro-batch 8
  x accum 4, bf16, 12 of 28 layers.
