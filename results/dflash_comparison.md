# dflash-robot Benchmark Comparison

Generated: 2026-05-04

## DFlash vs Autoregressive (Same Harness)

Qwen3.6-27B-Q4_K_M on RTX 3090 FE (24GB VRAM)
Draft: z-lab/Qwen3.5-27B-DFlash (cross-generation mismatch)
Benchmark: dflash-robot bench_llm.py, 10 prompts/dataset, n_gen=256

| Task | AR tok/s | DFlash tok/s | Acceptance Length | Speedup |
|---|---:|---:|---:|---:|
| HumanEval | 37.54 | 108.42 | 6.50 | **2.89x** |
| GSM8K | 37.79 | 72.61 | 4.34 | **1.92x** |
| Math500 | 37.86 | 91.67 | 5.40 | **2.42x** |
| **Mean** | **37.73** | **90.90** | **5.41** | **2.41x** |

## DFlash vs Historical llama-bench Baseline (Reference Only)

Note: Different harnesses. Historical baseline uses llama-bench gen512 with -r 5 warmup protocol. DFlash uses bench_llm.py with 10 prompts per dataset.

| Model | Historical gen512 | DFlash Mean tok/s | Notes |
|---|---:|---:|---|
| Qwen3.6-27B-Q4_K_M | 40.4 | 90.90 | Using Qwen3.5-27B draft (cross-gen mismatch) |

With matched Qwen3.6-27B-DFlash draft, speedup should improve (z-lab reports 5.05 AL vs current 5.41 with mismatch, but actual tok/s would be higher with matched draft).

## RTX 3090 Full Baseline (llama-bench, pre-DFlash)

Source: /home/am/benchmark-wiki/comparisons/gpu-inference-benchmark-results.md

| Model | Size | pp512 | gen128 | gen512 |
|---|---:|---:|---:|---:|
| gemma-4-26B-A4B-UD-Q4_K_M | 16 GB | 3,829 ± 14 | 134.8 ± 0.27 | 131.4 ± 0.07 |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 22 GB | 3,119 ± 19 | 146.3 ± 0.21 | 146.6 ± 0.32 |
| gemma-4-31B-it-Q4_K_M | 17 GB | 1,265 | 37.55 ± 0.02 | 36.21 ± 0.07 |
| Qwen3.6-27B-Q4_K_M | 17 GB | 1,345 ± 12 | 40.8 ± 0.04 | 40.4 ± 0.04 |

## Compatibility Matrix

| Model | Baseline gen512 | DFlash Status | Post-DFlash | Notes |
|---|---:|---|---:|---|
| Qwen3.6-27B-Q4_K_M | 40.4 | measured | 90.90 | cross-gen draft; matched draft would improve |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 146.6 | planned v0.2 | TBD | z-lab draft exists |
| gemma-4-26B-A4B-UD-Q4_K_M | 131.4 | no known draft | N/A | no z-lab DFlash draft |
| gemma-4-31B-it-Q4_K_M | 36.21 | no known draft | N/A | no z-lab DFlash draft |
