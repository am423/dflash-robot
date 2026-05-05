# Qwen3.6-35B-A3B DFlash Benchmark Report

Date: 2026-05-04
Hardware: NVIDIA GeForce RTX 3090 (24 GB VRAM), CUDA 13.2
Framework: dflash-robot (GGUF-native DFlash speculative decoding)

## Model Details

| Property | Value |
|---|---|
| Target | unsloth/Qwen3.6-35B-A3B-GGUF (Q4_K_M) |
| Target size | 20.10 GiB (733 tensors) |
| Target arch | qwen35moe |
| Target params | n_layer=40, n_embd=2048, n_head=16, n_head_kv=2 |
| Target MoE | 256 experts, 8 active, n_ff_expert=512 |
| Draft | z-lab/Qwen3.6-35B-A3B-DFlash |
| Draft size | 0.88 GiB (91 tensors) |
| Draft params | n_layer=5, hidden=2048, n_head=32, n_kv_head=4 |
| Total VRAM | ~21.0 GiB |
| dflash-robot commit | 3f648c8+ (graph module infrastructure, 35B fixes) |

## DFlash Smoke

```
[dflash] generated 64 tokens in 2.263s -> 28.29 tok/s
[dflash] 56 draft steps, accepted=64/896 (7.1% per step), avg commit/step=1.14
```

### Per-Step Timing

| Phase | Time (ms) |
|---|---|
| draft_build | 0.82 |
| draft_copyfeat | 0.16 |
| draft_compute | 3.89 |
| draft_logits | 2.69 |
| verify_build | 0.74 |
| verify_set | 0.49 |
| verify_compute | 31.49 |
| **Sum** | **40.29** |

## Comments

- DFlash smoke completed successfully without graph crashes
- Acceptance length is low (7.1%, avg 1.14 tokens/step)
- Per-step overhead dominated by verify_compute (31.49ms, 78% of step time)
- This is expected for a 35B MoE model running on a single 24GB GPU
- Full benchmark suite (HumanEval, GSM8K, Math500) pending

## Historical Baseline (llama-bench, different harness)

| Model | pp512 | gen512 |
|---|---|---|
| Qwen3.6-35B-A3B-UD-Q4_K_M | 3,119 ± 19 t/s | 146.6 ± 0.32 t/s |

Note: llama-bench baseline uses different harness than dflash-robot. Not directly comparable.
