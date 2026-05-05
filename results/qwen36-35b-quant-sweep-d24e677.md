# Qwen3.6-35B-A3B DFlash Quant Sweep (d24e677)

All benchmarks: RTX 3090 (24GB), same prompt (12 tokens → 64 generated), z-lab/Qwen3.6-35B-A3B-DFlash draft.

| Quant | Size | AR tok/s | DFlash tok/s | Speedup | Accept % | Commit/step | Verify ms | Step ms |
|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 20.10 GiB | 105.20 | 82.94 | 0.79x | 20.8% | 4.27 | 26.37 | 51.44 |
| Q3_K_XL | 15.18 GiB | 104.66 | 81.93 | 0.78x | 21.2% | 4.27 | 27.36 | 52.08 |
| Q2_K_XL | 11.11 GiB | 103.32 | 80.11 | 0.78x | 19.1% | 4.00 | 26.77 | 49.93 |

## Key findings

1. **Lower quant DOES NOT improve DFlash speed.** IQ3_XXS/IQ2_XXS dequantization overhead offsets model size reduction. Verify compute stays constant at ~27ms.

2. **Q2_K_XL degrades token quality.** Different greedy token sequence from Q4_K_M at position 17. Q4_K_M is the reference quant.

3. **Speedup locked at 0.78x.** MoE verify (~27ms/step for 40 MoE layers) is the binding constraint across all quants. DFlash cannot exceed AR on RTX 3090.

4. **VRAM is not the bottleneck.** Even Q2_K_XL (11.1 GiB target + 0.9 GiB draft = 12 GiB) has plenty of headroom. The issue is compute, not memory.

## Architecture fixes applied (d24e677)
- Shared expert sigmoid gate (ffn_gate_inp_shexp)
- Capture layer IDs formula: (n_layer-3)/(N-1) not (n_layer-2)
- MoE expert weight renormalization (norm_w=true from llama.cpp qwen35moe:384)
