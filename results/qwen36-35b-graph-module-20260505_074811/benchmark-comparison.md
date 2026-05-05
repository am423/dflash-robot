# Qwen3.6-35B-A3B Graph Module Benchmark Comparison

Target: `Qwen3.6-35B-A3B-UD-Q4_K_M`
Draft: `z-lab/Qwen3.6-35B-A3B-DFlash`
Graph module path: `build_target_graph -> qwen35moe TargetGraphModule`
Prompt tokens: 12
Generated tokens: 64

| Mode | tok/s | total s | notes |
|---|---:|---:|---|
| AR (`test_generate`) | 103.16 | 0.620 | same target graph dispatcher |
| DFlash (`test_dflash --fast-rollback --ddtree --ddtree-budget=22`) | 98.45 | 0.650 | 16 draft steps, avg commit/step 4.00 |

Speedup vs AR: 0.954x

DFlash acceptance: 64/256 (25.0%)

Per-step timing highlights:
- draft_compute: 6.32 ms
- verify_compute: 28.87 ms
- total measured step sum: 40.43 ms

Validation completed:
- build passed
- `dflash_inspect` reports graph module `qwen35moe`
- `smoke_load_target` passed
- `smoke_target_forward` passed with finite logits
- `test_compatibility_registry` passed 17/17

Conclusion: the Qwen3.6-35B-A3B benchmark path now uses the qwen35moe graph module. DFlash is correct and functional, but this prompt is still MoE-verify-bound on RTX 3090.
