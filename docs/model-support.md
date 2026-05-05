# dflash-robot Model Support Matrix

Generated: 2026-05-04

## Supported (dflash-robot v0.1)

| Target Model | Draft Model | Target Arch | dflash-robot Status | Post-DFlash tok/s | Speedup vs AR |
|---|---|---|---|---|---|
| Qwen3.6-27B-Q4_K_M | z-lab/Qwen3.5-27B-DFlash (cross-gen) | qwen35 | measured | 90.90 mean | 2.41x |

## Known DFlash Drafts (z-lab)

| Target | Draft Repo | dflash-robot Adapter | Status |
|---|---|---|---|
| Qwen3.5-27B | z-lab/Qwen3.5-27B-DFlash | qwen35_adapter | ready |
| Qwen3.6-27B | z-lab/Qwen3.6-27B-DFlash | qwen35_adapter | gated on HF |
| Qwen3.6-35B-A3B | z-lab/Qwen3.6-35B-A3B-DFlash | qwen35_adapter | planned v0.2 |
| Qwen3.5-35B-A3B | z-lab/Qwen3.5-35B-A3B-DFlash | qwen35_adapter | planned v0.2 |
| Qwen3.5-122B-A10B | z-lab/Qwen3.5-122B-A10B-DFlash | qwen35_adapter | too large for RTX 3090 |
| Qwen3-4B | z-lab/Qwen3-4B-DFlash-b16 | planned | v0.2 |
| Qwen3-8B | z-lab/Qwen3-8B-DFlash-b16 | planned | v0.2 |
| Llama-3.1-8B-Instruct | z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat | planned | v0.2 |
| Kimi-K2.5 | z-lab/Kimi-K2.5-DFlash | none | no adapter |
| Qwen3-Coder-30B-A3B | z-lab/Qwen3-Coder-30B-A3B-DFlash | qwen35_adapter | planned v0.2 |
| gpt-oss-20b | z-lab/gpt-oss-20b-DFlash | none | no adapter |
| gpt-oss-120b | z-lab/gpt-oss-120b-DFlash | none | no adapter |

## RTX 3090 Baseline Models Without Known DFlash Drafts

| Target Model | Baseline gen512 | Status |
|---|---|---|
| gemma-4-26B-A4B-UD-Q4_K_M | 131.4 t/s | no known DFlash draft |
| gemma-4-31B-it-Q4_K_M | 36.21 t/s | no known DFlash draft |

## How to Check Any GGUF

```bash
./build/dflash_inspect --target model.gguf --json
```

Returns JSON with compatibility_status:
- `supported` — adapter and draft available, ready to run
- `adapter_exists_draft_missing` — architecture supported but no draft found
- `incompatible_arch` — draft exists but for different architecture
- `unsupported_architecture` — no adapter for this architecture

## Expansion Roadmap

### v0.2
- Qwen3.6-35B-A3B support (z-lab draft exists, qwen35 adapter compatible)
- Dense transformer adapter for Llama-3.1-8B-Instruct
- Qwen3-4B/8B non-thinking support

### v1.0
- Draft training pipeline when z-lab recipe is available
- Adaptive block-size scheduling
- Sampling/rejection beyond greedy
