# Building dflash-robot

## Requirements

- NVIDIA GPU: sm_75+ (RTX 2080 Ti, RTX 3090, RTX 4090, A10, A40, H100)
- CUDA 12+ (tested with CUDA 13.2 via miniforge)
- CMake 3.18+
- C++17 compiler (GCC 15.2.0 tested)
- ~80 GB disk for models + build
- 22+ GB VRAM for Qwen3.6-27B target + draft

## Quick Build

```bash
git clone --recurse-submodules https://github.com/am423/dflash-robot.git
cd dflash-robot

# Configure (RTX 3090 = sm_86)
CUDAToolkit_ROOT=/home/am/.miniforge/targets/x86_64-linux \
CUDACXX=/home/am/.miniforge/bin/nvcc \
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

# Build all targets
cmake --build build -j$(nproc)
```

## CUDA Architecture Values

| GPU | CMAKE_CUDA_ARCHITECTURES |
|---|---|
| RTX 2080 Ti (Turing) | 75 |
| RTX 3090 / A40 (Ampere) | 86 |
| RTX 4090 (Ada) | 89 |
| H100 (Hopper) | 90 |
| Jetson AGX Thor (requires CUDA 13+) | 110 |
| Blackwell / DGX Spark | 120 |

## Built Binaries

| Binary | Purpose |
|---|---|
| test_generate | Autoregressive target-only decode |
| test_dflash | DFlash speculative decode |
| smoke_load_target | Load test for target GGUF |
| smoke_load_draft | Load test for draft safetensors |
| test_kv_quant | KV quantization unit test (no model needed) |
| test_flashprefill_kernels | FlashPrefill kernel numerics test |
| test_vs_oracle | Draft graph numerics vs PyTorch reference |
| smoke_draft_graph | Draft graph smoke test |
| smoke_target_forward | Target forward smoke test |
| pflash_daemon | Phase-split PFlash harness |

## Running Tests

```bash
# CPU/GPU unit tests (no model files needed)
./build/test_kv_quant
./build/test_flashprefill_kernels

# Model smoke tests (requires models/)
./build/smoke_load_target models/Qwen3.6-27B-Q4_K_M.gguf
./build/smoke_load_draft models/draft/model.safetensors
```

## Python Environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install transformers datasets huggingface_hub safetensors sentencepiece fastapi uvicorn jinja2 pytest
```

## Downloading Models

```bash
# Target: ~17 GB
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# Draft: ~3.46 GB (may require HuggingFace access)
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/
```

## Common Issues

- **CUDA not found**: Set CUDAToolkit_ROOT and CUDACXX to miniforge CUDA path
- **Missing /lib64 symlinks**: On Ubuntu 26+, create symlinks: `sudo ln -sf /usr/lib/x86_64-linux-gnu/libm.so.6 /usr/lib64/`
- **Submodules missing**: Run `git submodule update --init --recursive`
- **BF16 WMMA errors on sm_75**: FlashPrefill kernels auto-disable on Turing; draft falls back to ggml flash_attn_ext
