# dflash-robot Gap Analysis: z-lab Original vs Luce GGUF vs Paper Requirements

Generated: 2026-05-04

## What Luce Already Solved

1. **GGUF target loading**: Full GGUF metadata parsing and weight loading via ggml
2. **Safetensors draft loading**: Converts z-lab BF16 safetensors to ggml tensors at load time
3. **GGUF draft loading**: Optional Q8_0 quantized draft format
4. **Custom ggml ops**: Tree-mode SSM conv and gated delta net ops for rollback
5. **DDTree verification**: Tree-structured speculative verification with configurable budget
6. **Fast rollback**: SSM intermediate state capture/restore without full replay
7. **Long context**: TQ3_0/Q4_0 KV cache quantization, sliding FA window, target_feat ring buffer
8. **CUDA kernels**: FlashPrefill (block-sparse attention for drafter), BSA launcher
9. **Benchmark harness**: test_generate (AR baseline) and test_dflash (DFlash) with timing
10. **Server**: OpenAI-compatible HTTP server (batch=1, greedy)

## What z-lab Solved Differently

1. **Model generality**: Supports Qwen3, Qwen3.5, Qwen3.6, Llama-3.1, Kimi-K2.5, gpt-oss, Qwen Coder
2. **Runtime generality**: Works across Transformers, vLLM, SGLang, MLX backends
3. **Draft weight sharing**: Draft uses target's lm_head (no separate LM head in draft)
4. **Layer selection**: build_target_layer_ids computes uniformly-spaced layers from [1, n_layers-3]
5. **Hidden feature fusion**: Concatenates selected hidden states along feature dim
6. **KV injection**: Projects target features through draft's K/V projections, persists in draft KV cache
7. **Training**: Full training pipeline with random anchor sampling, position-decayed loss, Flex Attention
8. **Block diffusion**: Non-causal block attention within each draft block
9. **Multi-backend benchmark**: Same benchmark harness across vLLM/SGLang/Transformers/MLX

## What the Paper Requires That Luce Hardcodes

1. **Architecture dispatch**: Luce has only qwen35 adapters; need ModelAdapter interface
2. **Target hidden-state extraction**: Hardcoded to specific qwen35 layer indices
3. **Hidden feature fusion**: Fixed [5*hidden, ctx_len] layout
4. **Target-feature KV injection**: Coupled to qwen35 draft graph
5. **Block size**: Compile-time constants
6. **Draft depth**: Hardcoded to qwen3 0.6B 5-layer draft
7. **Vocab/tokenizer**: Assumes Qwen3 tokenizer
8. **Acceptance logic**: Works but not behind clean interface

## What dflash-robot Must Add

### v0.1 (Shipping)
1. ModelAdapter interface with qwen35 implementation
2. DraftAdapter interface with qwen3 dflash implementation
3. RuntimeOrchestrator wrapping existing decode loop
4. CompatibilityRegistry for model classification
5. dflash_inspect CLI for any GGUF inspection
6. Benchmark report tooling joining to existing baselines
7. Documentation of adapter APIs and model support matrix

### v0.2 (Next)
1. Qwen3.6-35B-A3B adapter (z-lab draft exists)
2. Dense transformer adapter (Llama-3.1-8B-Instruct)
3. Draft GGUF conversion generalization
4. Sampling/rejection beyond greedy
5. Adaptive block-size scheduling

### v1.0 (Future)
1. Draft training pipeline
2. Upstream llama.cpp PR for hidden-state extraction hooks
3. Multiple draft families (chain, tree, block diffusion)
4. Production server with sampling support

## Design Implications

1. **Adapter pattern is mandatory**: No code path should assume qwen35
2. **Hidden-state extraction must be first-class**: The paper's core insight (target hidden states encode future tokens) requires clean extraction API
3. **KV injection must be generic**: Draft's KV cache injection of target features is the key differentiator from EAGLE-style methods
4. **Metrics must measure net tokens/sec**: Acceptance length alone is misleading; must account for draft+verify+rollback overhead
5. **Block size must be runtime-configurable**: Paper shows larger training blocks generalize to smaller inference blocks
