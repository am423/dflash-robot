# dflash-robot

GGUF-native DFlash speculative decoding runtime for local models.

DFlash uses a lightweight block diffusion draft model to propose multiple tokens in parallel, verified by the target model in one forward pass. This is faster than autoregressive speculative decoding because draft latency does not scale with block size.

dflash-robot is based on [Luce DFlash](https://github.com/Luce-Org/lucebox-hub/tree/main/dflash), the first GGUF port of DFlash, with adapter-based architecture for expanding to any GGUF model with a compatible DFlash draft.

## Current Status

v0.1.0 — Qwen3.6-27B on RTX 3090:

| Task | AR tok/s | DFlash tok/s | Speedup |
|---|---:|---:|---:|
| HumanEval | 37.54 | 108.42 | 2.89x |
| GSM8K | 37.79 | 72.61 | 1.92x |
| Math500 | 37.86 | 91.67 | 2.42x |
| Mean | 37.73 | 90.90 | 2.41x |

Draft: z-lab/Qwen3.5-27B-DFlash (cross-generation mismatch with Qwen3.6-27B target). Matched draft would improve speedup.

## Requirements

- NVIDIA GPU: sm_75+ (RTX 2080 Ti, RTX 3090, RTX 4090, A10, A40, H100)
- CUDA 12+
- 22+ GB VRAM for Qwen3.6-27B
- ~80 GB disk

## Quick Start

```bash
git clone --recurse-submodules https://github.com/am423/dflash-robot.git
cd dflash-robot

# Build
CUDAToolkit_ROOT=/path/to/cuda \
CUDACXX=/path/to/nvcc \
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

# Download models
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors --local-dir models/draft/

# Python env
python3 -m venv .venv
.venv/bin/pip install transformers datasets huggingface_hub safetensors sentencepiece jinja2

# Run
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_DRAFT=models/draft/model.safetensors \
.venv/bin/python scripts/run.py --prompt "def fibonacci(n):" --n-gen 128

# Inspect any GGUF
./build/dflash_inspect --target model.gguf --json
```

## Architecture

dflash-robot uses an adapter pattern for model generality:

- **ModelAdapter**: Reads GGUF target metadata, builds target prefill/verify graphs, exposes hidden states, manages KV/SSM cache rollback
- **DraftAdapter**: Loads draft models, validates compatibility, builds draft graph, provides candidate tokens
- **RuntimeOrchestrator**: Target prefill, hidden-state extraction, draft proposal, target verification, acceptance, rollback, metrics
- **CompatibilityRegistry**: Classifies any GGUF/draft pair as supported/incompatible/missing

## Model Support

See [docs/model-support.md](docs/model-support.md) for the full matrix.

Key distinction: "any GGUF" means runtime-generic target support. Acceleration still requires a compatible trained DFlash draft model.

## Documentation

- [docs/building.md](docs/building.md) — Build instructions
- [docs/model-support.md](docs/model-support.md) — Supported models and expansion roadmap
- [docs/architecture/gap-analysis.md](docs/architecture/gap-analysis.md) — z-lab vs Luce vs dflash-robot
- [docs/research/paper-to-code-map.md](docs/research/paper-to-code-map.md) — DFlash paper to code mapping
- [docs/research/luce-hardcoded-assumptions.md](docs/research/luce-hardcoded-assumptions.md) — Qwen35-specific assumptions audit

## Limitations

- Batch size 1, single-user local inference
- Greedy decoding only (temperature/top_p accepted but ignored in server)
- Qwen3.6-27B path only (Qwen35 adapter); more adapters planned
- No draft training pipeline yet
- CUDA/NVIDIA only

## Upstream

Based on [Luce DFlash](https://github.com/Luce-Org/lucebox-hub/tree/main/dflash) (MIT License).
Algorithm: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036v1).
See [UPSTREAM.md](UPSTREAM.md) for full attribution.

## License

MIT — see [LICENSE](LICENSE).
