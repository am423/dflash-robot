# Changelog

## v0.1.0 (2026-05-04)

Initial release of dflash-robot.

### Features
- GGUF-native DFlash speculative decoding for Qwen3.6-27B on RTX 3090
- Adapter-based architecture (ModelAdapter, DraftAdapter, RuntimeOrchestrator, CompatibilityRegistry)
- dflash_inspect CLI for any GGUF model compatibility inspection
- DFlash paper-to-code mapping and research documentation
- Qwen35 adapter with target hidden-state extraction and KV injection support
- DDTree tree-structured verification with fast rollback
- Long-context support via TQ3/Q4 KV cache quantization
- OpenAI-compatible HTTP server (batch=1, greedy)

### Benchmarks
- Qwen3.6-27B on RTX 3090: 2.41x mean speedup (90.90 vs 37.73 tok/s AR)
- HumanEval: 2.89x, GSM8K: 1.92x, Math500: 2.42x

### Known Limitations
- Qwen3.6-27B only (qwen35 adapter); more adapters planned for v0.2
- Draft: Qwen3.5-27B-DFlash (cross-generation mismatch); matched Qwen3.6 draft is gated
- Batch size 1, greedy decoding only
- No draft training pipeline
- CUDA/NVIDIA only
