# Graph Module Authoring Guide

## What a Graph Module Owns

A target graph module is a self-contained implementation for one GGUF
architecture family. It owns:

1. **GGUF metadata parsing** — reading model dimensions and architecture-specific parameters from the GGUF header
2. **Tensor binding** — wiring GGUF tensor names to graph operations, with validation
3. **Cache layout** — KV cache, SSM/conv state, hidden-state capture buffer, rollback snapshots
4. **Layer graph construction** — building the ggml compute graph for forward passes
5. **Hidden-state capture** — capturing target layer activations for DFlash draft conditioning
6. **Rollback/crop semantics** — restoring state after speculative token rejection
7. **Capability reporting** — machine-readable declaration of supported features

## How to Add a New Architecture

### 1. Identify GGUF Architecture

```bash
dflash_inspect --target model.gguf --json
```

Look at `target_arch` and `graph_module` fields.

### 2. Dump Metadata and Tensor Shapes

```bash
smoke_load_target model.gguf
```

Prints: n_layer, n_embd, n_head, n_head_kv, head_dim, full_attention_interval,
SSM parameters, MoE parameters, tensor shapes for sample layers.

### 3. Locate llama.cpp Source Graph

Find the architecture's graph builder in `deps/llama.cpp/src/llama-graph.cpp`.
This is the ground-truth implementation for graph structure, shapes, and ops.

Key patterns to extract:
- How are attention layers structured? (dense, GQA, SWA, etc.)
- What activation functions are used? (SiLU, SwiGLU, GELU, etc.)
- Are there any architecture-specific operations? (RoPE variants, SSM, MoE routing)

### 4. Classify Layer Types

Common patterns:
- **Dense attention**: Q/K/V projections, RoPE, flash attention, output projection
- **Hybrid SSM**: interleaved full-attention and state-space model layers
- **MoE FFN**: router, top-k expert selection, per-expert FFN, shared expert
- **Sliding window attention**: limited attention range for long contexts

For DFlash, also classify where target hidden states are captured and how the
compatible draft consumes them (`target_layer_ids`, `target_hidden_size`,
`fc.weight`, vocab/d2t mapping). Runtime graph support alone does not imply
acceleration; a compatible trained draft is still required.

### 5. Implement the Module

1. Create `include/graph/<arch>_module.h`
2. Create `src/graph/<arch>_module.cpp`
3. Register in `src/graph/model_graph_registry.cpp`
4. Add to `CMakeLists.txt`

### 6. Required Tests

Before marking an architecture as supported:

| Test | What it verifies |
|---|---|
| `smoke_load_target` | Metadata parsing, tensor binding |
| `dflash_inspect --json` | Registry lookup, capability reporting |
| `smoke_target_forward` | Single-token forward pass, no NaN |
| Multi-token prefill | Batch processing correctness |
| Cache rollback | SSM/KV state restore after rejection |
| Oracle comparison | Top-k logit match vs llama.cpp |

### 7. Add to Compatibility Registry

Update `src/graph/model_graph_registry.cpp` to register the new module,
then update `src/runtime/compatibility_registry.cpp` if needed for
draft compatibility classification.

## Qwen35MoE / Qwen3.6-35B-A3B Notes

`qwen35moe` is the first MoE target module. The module validates MoE metadata
and tensors, then uses the shared qwen35 graph builder; the builder selects the
MoE FFN path when MoE tensors are present.

Required metadata:
- `general.architecture = qwen35moe`
- `qwen35moe.expert_count`
- `qwen35moe.expert_used_count`
- `qwen35moe.expert_shared_feed_forward_length`

Required MoE tensors per layer:
- `ffn_gate_inp.weight`
- `ffn_gate_exps.weight`
- `ffn_up_exps.weight`
- `ffn_down_exps.weight`
- `ffn_gate_shexp.weight`
- `ffn_up_shexp.weight`
- `ffn_down_shexp.weight`
- `ffn_gate_inp_shexp.weight` when present

Correctness requirements:
1. Route MoE layers through the MoE FFN graph, not dense SwiGLU.
2. After router softmax + top-k, renormalize selected expert weights per token
   (`sum_rows`, clamp min `6.103515625e-5`, divide). This matches llama.cpp
   `norm_w=true` for qwen35moe.
3. Apply the shared expert sigmoid gate:
   `shared_out *= sigmoid(ffn_gate_inp_shexp @ cur)` before adding it to the
   routed expert output.
4. Verify target capture layer IDs against the draft config. For the known
   40-layer Qwen3.6-35B-A3B DFlash draft with five target captures, the fixed
   IDs are `{1,10,19,28,37}`.

Performance note: on RTX 3090 the current qwen35moe path is verify-compute
bound without fused ggml-cuda MoE kernels. Correct graph semantics do not imply
DFlash beats same-harness AR speed for this MoE target.

## Common Pitfalls

### ggml Shape/Broadcast

- `ggml_mul` requires B to be broadcastable to A (each dim equal or B.dim == 1)
- `ggml_mul_mat` is A @ B where A has ne[0]=out_dim, ne[1]=in_dim
- `ggml_mul_mat_id` for MoE: tensor layout is [out_dim, in_dim, n_expert]
- Always validate shapes with assertions or shape tracing before graph execution

### Cache Quantization

- KV cache type (Q8_0, TQ3_0) affects attention kernel selection
- TQ3_0 requires FWHT rotation in flash attention
- SSM state is always F32
- Target feature buffer uses BF16

### Target Feature Ring Buffer

- Shape: [capture_layers * n_embd, target_feat_cap]
- Written at position (kv_start + i) % cap
- Draft reads contiguous slice; wrap is invisible if cap is large enough

### Tokenizer/Vocab Mismatch

- Always use the same tokenizer that matches the target model
- Draft vocab must map to target vocab (via d2t mapping or identical vocab)
- EOS token IDs come from GGUF metadata

### MoE Correctness

- Do not leave stale `under_development` capability notes after MoE graph
  semantics are implemented; use precise performance limitations instead.
- Check all MoE graph call sites. A helper existing in the file is not enough
  if the main graph path still calls dense FFN unconditionally.
- For qwen35moe, selected expert weights must be renormalized per token and
  shared expert output must be gated with `sigmoid(ffn_gate_inp_shexp @ cur)`.
- A correct MoE graph may still be slower than AR on RTX 3090 because verify is
  compute-bound without fused ggml-cuda MoE kernels.
