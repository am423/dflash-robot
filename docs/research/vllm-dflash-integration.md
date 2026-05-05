# DFlash: z-lab/vLLM Integration Analysis

Source: https://github.com/jianc99/DFlash (inspected at /tmp/zlab-dflash-inspect)
Paper: https://arxiv.org/abs/2602.06036

## 1. Overview

DFlash ("Block Diffusion for Flash Speculative Decoding") is a lightweight
draft model that generates tokens in *blocks* via masked diffusion rather
than autoregressively. The draft attends to selected target-model hidden
states as cross-attention context, making it architecture-aware without
being a full copy of the target. It supports Qwen3/3.5, LLaMA-3.1, and
several other families.

Key insight: instead of a small autoregressive draft, DFlash uses a
block-diffusion model that can produce K tokens in one forward pass by
iteratively unmasking them, conditioned on the target's intermediate
representations.


## 2. vLLM Integration: method=dflash

### 2.1 Speculative Config JSON

From the README quick-start:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

The speculative-config JSON has three fields:
- "method": "dflash" — selects the DFlash speculative decoding path in vLLM
- "model": HF repo path for the draft model
- "num_speculative_tokens": how many tokens to draft per step (typically 15-16)

### 2.2 vLLM PR Dependency (PR 40898)

DFlash requires vLLM PR #40898. The README instructs:

```bash
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head"
```

The PR adds:
- Support for interleaved sliding-window attention (SWA) in vLLM's execution
- Correct handling of target hidden states extraction at arbitrary layer
  indices (not just the final layer)
- The plumbing to pass hidden states from the target model's intermediate
  layers to the draft model as cross-attention context

Without this PR, vLLM cannot capture mid-layer hidden states or pass them
to the speculative draft model — both essential for DFlash.


## 3. Architecture: How the Draft Model Works

### 3.1 DFlashDraftModel (model.py)

The draft model (`DFlashDraftModel`) is a subclass of `Qwen3PreTrainedModel`
containing:

- `layers`: N transformer decoder layers (`Qwen3DFlashDecoderLayer`)
- `fc`: Linear projection from `(num_target_layers * hidden_size) -> hidden_size`
- `hidden_norm`: RMSNorm on projected target features
- `norm`: Final RMSNorm
- `rotary_emb`: Rotary position embeddings
- `block_size`: Number of tokens drafted per block (from config)
- `mask_token_id`: Special token used for masked positions in diffusion

Config fields (from `dflash_config` in the model's config.json):
- `target_layer_ids`: Which target model layers to read (or auto-computed)
- `mask_token_id`: The [MASK] token id for block diffusion

### 3.2 build_target_layer_ids()

```python
def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
```

This computes which target model layers to extract hidden states from:
- For a single draft layer: picks the middle target layer
- For multiple draft layers: evenly spans from layer 1 to layer (N-3),
  distributing draft layers across target layers

Example: 28 target layers, 4 draft layers → target_layer_ids = [1, 10, 19, 24]

The idea is that the draft's N layers each attend to a corresponding
"sibling" layer in the target, getting progressively richer context.

### 3.3 extract_context_feature()

```python
def extract_context_feature(hidden_states, layer_ids):
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)
```

- Takes the target model's `output_hidden_states` (a tuple of tensors, one
  per layer + embedding)
- offset=1 skips the embedding layer output
- Selects hidden states at the configured layer indices
- Concatenates them along the feature dimension: shape becomes
  `[batch, seq, num_selected_layers * hidden_size]`

This concatenated tensor is then projected by `self.fc` (Linear) back to
`hidden_size` and normalized by `self.hidden_norm` before being used as
cross-attention context.

### 3.4 Cross-Attention in Qwen3DFlashAttention

The draft's attention layer differs from standard self-attention:

```python
# Q comes from the noised draft tokens
q = self.q_proj(hidden_states)        # hidden_states = noise_embedding

# K, V come from BOTH target context AND draft tokens
k_ctx = self.k_proj(target_hidden)     # from target model
k_noise = self.k_proj(hidden_states)   # from draft
v_ctx = self.v_proj(target_hidden)
v_noise = self.v_proj(hidden_states)

# Concatenate: context keys/values first, then noise keys/values
k = torch.cat([k_ctx, k_noise], dim=1)
v = torch.cat([v_ctx, v_noise], dim=1)
```

This means the draft tokens attend to:
1. The target model's hidden states (cross-attention context)
2. Their own representations (self-attention)

The attention is non-causal (`self.is_causal = False`) since diffusion
tokens don't follow a strict left-to-right order.

### 3.5 Block Diffusion Drafting (dflash_generate)

The generation loop works as follows:

**Prefill:**
1. Run target model on input_ids with `output_hidden_states=True`
2. Extract target hidden states via `extract_context_feature`
3. Sample first token from target logits

**Decode loop (each step drafts a block of `block_size` tokens):**
1. Take the current block positions from the output buffer
2. Embed them via `target.model.embed_tokens` → `noise_embedding`
3. Run draft model: `model(target_hidden, noise_embedding, ...)` → hidden states
4. Project through `target.lm_head` to get draft logits
5. **KV cache crop**: `past_key_values_draft.crop(start)` — after draft
   produces a block, crop the draft KV cache to the accepted position
6. Sample draft tokens from the draft logits (positions 1..block_size)
7. Run **target verification** on the full drafted block
8. **Acceptance**: count matching tokens via cumulative product of matches
9. Crop both target and draft KV caches to the new accepted position
10. Extract new target hidden states for accepted portion
11. Continue

### 3.6 KV Cache Crop for Acceptance

After verification:
```python
acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
start += acceptance_length + 1
past_key_values_target.crop(start)
```

- Cumulative product ensures only *contiguous* matches from the start are
  accepted (first mismatch terminates the block)
- Target KV cache is cropped to `start` (the new accepted position)
- Draft KV cache is also cropped: `past_key_values_draft.crop(start)`
- Target hidden states are re-extracted for only the accepted tokens:
  `extract_context_feature(...)[:, :acceptance_length + 1, :]`

This is critical: the draft's cross-attention context always reflects
verified, accepted tokens — not stale or speculative ones.


## 4. Benchmark Harness (benchmark.py)

### 4.1 What It Measures

The benchmark supports four backends: transformers, sglang, vllm, mlx.

**For transformers backend** (direct measurement):
- Runs each prompt twice: once with block_size=1 (baseline), once with
  block_size=D (draft-enabled)
- Measures: time_to_first_token, time_per_output_token, acceptance_lengths
- Reports: baseline vs DFlash throughput, speedup ratio, mean acceptance
  length, acceptance length histogram

**For server backends** (vllm, sglang):
- Sends requests via HTTP (OpenAI-compatible for vllm, /generate for sglang)
- Uses ThreadPoolExecutor for concurrent load
- Reports: total throughput (tok/s), mean acceptance length (sglang only
  via meta_info), total latency

### 4.2 Datasets

Shared across all backends: gsm8k, math500, humaneval, mbpp, mt-bench.
Auto-downloaded and cached as JSONL in cache/ directory.

### 4.3 Key Metrics

- **Acceptance length**: average number of tokens accepted per speculative
  step (includes the verified token, so block_size=15 with acceptance=8
  means 8 tokens verified per step)
- **Acceptance histogram**: distribution of acceptance lengths from 0 to
  block_size
- **Throughput**: total output tokens / wall-clock time
- **Speedup**: ratio of baseline TPOT to DFlash TPOT


## 5. What Translates to GGUF Runtime

### 5.1 IDEAS THAT TRANSLATE (doable in llama.cpp / GGUF)

1. **Block drafting concept**: Draft K tokens at once instead of
   autoregressively. In GGUF, this means running the draft model once
   per speculation step to produce block_size candidates, then verifying
   them in a single target forward pass. This is the core speedup idea.

2. **Cross-attention from target hidden states**: The draft model needs
   intermediate hidden states from the target model. In llama.cpp, we
   would need to:
   - Add a hook/callback at specific target model layers during inference
   - Pass these hidden state tensors to the draft model's forward pass
   - Implement the Qwen3DFlashAttention cross-attention pattern in GGML

3. **build_target_layer_ids logic**: The evenly-spaced layer selection
   algorithm is simple arithmetic and translates directly. This is just
   a configuration mapping: "draft layer i reads target layer j."

4. **extract_context_feature + fc projection**: Concatenating selected
   layer outputs and projecting via a linear layer is standard tensor
   ops easily expressed in GGML. The fc weight would be stored in the
   GGUF draft model file.

5. **KV cache crop for acceptance**: llama.cpp already has KV cache
   sequence management. Cropping to the accepted length after
   verification is conceptually straightforward — just reset the
   sequence length pointer.

6. **Acceptance via cumulative product**: The contiguous-match acceptance
   logic is trivial arithmetic.

7. **Masked diffusion block generation**: The iterative unmasking loop
   (mask tokens → run draft → unmask highest-confidence positions →
   repeat until full block) is a simple loop around the draft model.
   The mask_token_id and block_size are config parameters.

### 5.2 IDEAS THAT DO NOT TRANSLATE (or are very difficult)

1. **Using target.model.embed_tokens directly**: The draft's
   noise_embedding is computed by calling `target.model.embed_tokens()`
   on the draft block ids. In GGUF, the target and draft models are
   separate files with potentially different architectures. The draft
   would need its own embedding table, or the GGUF runtime would need
   to share embeddings across models (non-trivial with current GGML
   memory layout).

2. **target.lm_head reuse**: The draft model's output is projected
   through `target.lm_head`, not the draft's own head. This means the
   draft model has NO lm_head — it piggybacks on the target's. In GGUF,
   this requires either sharing the lm_head tensor or restructuring
   the draft model to include its own output projection.

3. **Seamless hidden state extraction at arbitrary layers**: vLLM PR
   40898 adds plumbing for this. In llama.cpp, the model runner does
   not currently expose intermediate layer outputs. Adding this requires
   modifying the compute graph to optionally capture and return hidden
   states at specific layers — a significant architectural change.

4. **DynamicCache.crop() pattern**: vLLM and transformers have Pythonic
   cache objects with .crop(). llama.cpp's KV cache is a pre-allocated
   buffer with manual index management. The crop-to-acceptance-length
   pattern exists conceptually but requires careful pointer arithmetic
   in the C/C++ code.

5. **Sliding window attention interleaving**: The draft's attention layers
   may use sliding window attention (SWA) at specific layers. The vLLM
   PR handles interleaved SWA + global attention. In GGML, this would
   need explicit attention mask construction per layer.

6. **HuggingFace AutoModel loading**: The draft model uses HF's
   from_pretrained with trust_remote_code. GGUF requires a pre-converted
   model file. The draft model's custom layers (Qwen3DFlashAttention,
   Qwen3DFlashDecoderLayer) would need to be mapped to GGML ops.

7. **The fc linear layer's large input dimension**: The fc layer projects
   from `(num_target_layers * hidden_size)` to `hidden_size`. For a 28-
   layer model with 4096 hidden size, that's 114688 × 4096 = ~470M
   parameters just for this one projection. This is a significant memory
   cost in a GGUF file.

### 5.3 Recommended GGUF Implementation Path

Phase 1 — Proof of concept (Transformers-like in GGUF):
- Convert DFlash draft model to GGUF with its own embedding + lm_head
- Implement basic block-draft-then-verify loop
- Hard-code target layer indices for a specific model pair

Phase 2 — Full integration:
- Add hidden-state capture hooks to llama.cpp's target model runner
- Implement Qwen3DFlashAttention cross-attention in GGML
- Implement the fc projection as a GGML linear layer
- Add KV cache crop/rollback support for speculative acceptance
- Store target_layer_ids and block_size in GGUF metadata

Phase 3 — Optimization:
- Batch the draft's block generation (parallel unmasking steps)
- Fuse the fc + hidden_norm into the compute graph
- Support sliding_window attention interleaving in draft layers


## 6. File Inventory

| File | Purpose |
|---|---|
| dflash/model.py | Core DFlash model + generation loop |
| dflash/model_mlx.py | Apple MLX backend implementation |
| dflash/benchmark.py | Multi-backend benchmark harness |
| dflash/__init__.py | Package init |
| pyproject.toml | Dependencies (transformers, sglang, vllm, mlx extras) |


## 7. Key Takeaways

1. DFlash is NOT a small autoregressive draft — it's a block-diffusion
   model that generates K tokens simultaneously via iterative unmasking.

2. The draft is deeply coupled to the target: it reads target hidden
   states at specific layers and uses the target's embedding + lm_head.
   This is what makes it powerful but also what makes GGUF integration
   challenging.

3. The vLLM integration depends on PR 40898 for hidden state extraction
   and interleaved SWA handling — without this PR, vLLM cannot run DFlash.

4. The acceptance mechanism (cumulative product of matches + KV crop) is
   standard speculative decoding but adapted for block-parallel drafting.

5. For GGUF: the core algorithmic ideas (block drafting, cross-attention
   conditioning, layer selection) are sound, but the implementation
   requires significant changes to llama.cpp's model runner to support
   hidden state capture and cross-model tensor sharing.
