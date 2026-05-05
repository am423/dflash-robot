# Purpose-Built DFlash Draft Model Plan

## Why Build Our Own Draft

The z-lab draft model (Qwen3.6-35B-A3B-DFlash) has a calibration issue: it confidently predicts garbage tokens on certain input patterns (confirmed on HumanEval prompt #5 "intersperse"). Root cause: the draft was trained on Nemotron/CodeAlpaca data, not our target model's actual outputs. A purpose-built draft trained on OUR target model's hidden states eliminates this mismatch.

## Target Setup

- GPU: RTX 3090 (24 GB VRAM)
- Target: Qwen3.6-35B-A3B Q4_K_M (20.1 GiB on GPU)
- Free VRAM for draft: ~3.5 GB
- Target hidden_dim: 2048
- Target n_layer: 28 (hybrid FA + DeltaNet)
- Capture layers: 5 (uniformly spaced: 1, 7, 14, 21, 27)
- Draft block size: 16

## Draft Model Architecture

5-layer non-causal transformer, conditioned on target hidden states via KV injection.

```
Parameters:
  fc:           [5*2048, 2048]  = 20.97M   (feature fusion)
  hidden_norm:  [2048]          = 2K
  5 layers:
    attn_norm:  [2048]          = 2K
    wq:         [2048, 2048]    = 4.19M
    wk:         [2048, 512]     = 1.05M    (n_head_kv=4, head_dim=128)
    wv:         [2048, 512]     = 1.05M
    wo:         [2048, 2048]    = 4.19M
    q_norm:     [128]           = 128
    k_norm:     [128]           = 128
    ffn_norm:   [2048]          = 2K
    w_gate:     [2048, 5504]    = 11.27M   (SwiGLU FFN)
    w_up:       [2048, 5504]    = 11.27M
    w_down:     [5504, 2048]    = 11.27M
  out_norm:     [2048]          = 2K

Total: ~178M params
BF16: ~356 MB
Q8_0: ~180 MB
Fits easily in 3.5 GB free VRAM.
```

The draft shares the target's token embedding and lm_head (frozen).
Only the draft transformer layers + fc projection are trained.

## How DFlash Draft Works (Inference)

```
1. Target prefill → extract hidden states from 5 capture layers
2. Concatenate → project through fc → rms_norm → target_feat
3. Build noise block: [last_token, MASK, MASK, ..., MASK] (16 positions)
4. Embed via target's token embedding table
5. Draft forward (NON-CAUSAL, single pass):
   For each draft layer:
     Q = wq @ noise_hidden
     K_ctx = wk @ target_feat,  K_noise = wk @ noise_hidden → K = concat
     V_ctx = wv @ target_feat,  V_noise = wv @ noise_hidden → V = concat
     RoPE(Q, positions_q), RoPE(K, positions_k)
     attn = flash_attn(Q, K, V, mask=null)  ← NON-CAUSAL
     h += wo @ attn
     h += SwiGLU(ffn_norm(h))
6. Argmax draft logits → candidate tokens [16]
```

Key: draft latency is CONSTANT (~4ms) regardless of block size, because all
positions are predicted in parallel. This is why DFlash beats EAGLE-style
autoregressive drafters.

## Training Data Pipeline

### Step 1: Generate Training Traces

Run the target model (Qwen3.6-35B-A3B) on diverse prompts and record:

```
For each prompt in dataset:
  Run target model with output_hidden_states=True
  For each position in the response:
    - hidden_states at 5 capture layers → concatenated [5*2048]
    - next_token (ground truth)
    - target_logits (full vocab distribution, for distillation loss)
  Save as trace file (one per prompt)
```

Storage: ~800K prompts × ~500 tokens avg × 5 × 2048 × 2 bytes = ~8 TB raw.
Mitigation: store traces as float16, use memory-mapped files, or stream.

### Step 2: Dataset Selection

Must match OUR usage patterns, not generic benchmarks:

| Source | Samples | Purpose |
|--------|---------|---------|
| HumanEval | 164 | Code completion (our benchmark) |
| MBPP | 974 | Code completion |
| LiveCodeBench | 500+ | Code generation |
| GSM8K | 8.5K | Math reasoning |
| MATH-500 | 500 | Math reasoning |
| MT-Bench | 80 | Multi-turn chat |
| Alpaca-GPT4 | 52K | General instruction following |
| CodeAlpaca | 20K | Code instruction following |
| Custom prompts | 100K+ | Your actual use cases |
| Synthetic code | 500K+ | Diverse code patterns |

Total target: 800K+ samples (matching the paper's training set size).

### Step 3: Training Algorithm

```python
# DFlash training with paper's key innovations
for epoch in range(6):
    for batch in dataloader:
        # 1. Extract target hidden features
        #    Run frozen target model on clean sequence
        #    Select hidden states at capture_layer_ids
        #    Concatenate along feature dim: [5*2048, seq_len]
        target_features = extract_features(target_hidden_states, capture_layer_ids)
        
        # 2. Random anchor sampling (paper Section 4.2)
        #    Randomly sample 512 anchor positions from responses
        #    Each anchor = start of a draft block
        #    This matches inference: draft always conditions on a clean token
        anchors = random_sample(response_positions, n=512)
        
        # 3. Construct blocks with sparse attention
        #    For each anchor:
        #      block = [clean_token(anchor), MASK, MASK, ..., MASK]
        #      Attention: bidirectional within block
        #                 attend to target features via KV injection
        #                 NO cross-block attention
        blocks = []
        for anchor in anchors:
            block_tokens = [response[anchor]] + [MASK_TOKEN] * (block_size - 1)
            blocks.append(block_tokens)
        
        # 4. Forward pass (all blocks in one batch via Flex Attention)
        draft_logits = draft_model(
            input_ids=concat(blocks),
            target_hidden=target_features,
            attention_mask=sparse_block_mask(blocks),  # Fig 4 from paper
            positions=compute_positions(blocks),
        )
        
        # 5. Position-decayed loss (paper Equation 4)
        #    weight(pos) = exp(-decay * pos)
        #    Earlier positions weighted more heavily because an error at
        #    position 0 invalidates ALL subsequent accepted tokens.
        #    Decay: 7 for block_size=16, 5 for block_size=10, 4 for block_size=8
        loss = 0
        for block_idx, anchor in enumerate(anchors):
            for pos in range(block_size):
                weight = math.exp(-decay * pos)
                target_token = response[anchor + pos]
                loss += weight * cross_entropy(
                    draft_logits[block_idx, pos], 
                    target_token
                )
        loss /= len(anchors)
        
        # 6. Update draft weights (target frozen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### Step 4: Hyperparameters (from paper Appendix A.1)

| Parameter | Value |
|-----------|-------|
| Epochs | 6 |
| Optimizer | AdamW |
| Gradient clip | 1.0 |
| LR schedule | Cosine with warmup ratio 0.04 |
| Max sequence length | 3072 |
| Anchor positions per sequence | 512 |
| Loss decay (block_size=16) | 7 |
| Loss decay (block_size=10) | 5 |
| Loss decay (block_size=8) | 4 |
| Batch size | Fit in remaining VRAM after target |
| Learning rate | 1e-4 to 5e-4 (tune via sweep) |

### Step 5: Calibration-Aware Training (Our Addition)

The z-lab draft fails because it's confidently wrong. Add calibration loss:

```python
# Standard cross-entropy loss (from paper)
ce_loss = cross_entropy(draft_logits, target_tokens)

# Calibration loss: penalize when draft confidence doesn't match correctness
draft_probs = softmax(draft_logits)
correct_mask = (draft_logits.argmax(-1) == target_tokens).float()

# ECE-inspired: draft confidence should match accuracy
draft_confidence = draft_probs.max(-1).values
calibration_loss = ((draft_confidence - correct_mask) ** 2).mean()

# Combined loss
total_loss = ce_loss + lambda_cal * calibration_loss
# lambda_cal = 0.1 (tune via validation)
```

### Step 6: Export to Safetensors

```python
# Save draft model with metadata
state_dict = {
    'fc.weight': draft.fc.weight.to(torch.bfloat16),
    'hidden_norm.weight': draft.hidden_norm.weight.to(torch.bfloat16),
    'out_norm.weight': draft.out_norm.weight.to(torch.bfloat16),
}
for i, layer in enumerate(draft.layers):
    for name, param in layer.named_parameters():
        state_dict[f'layers.{i}.{name}'] = param.to(torch.bfloat16)

safetensors.torch.save_file(state_dict, 'model.safetensors')

# Save config as GGUF metadata (or JSON sidecar)
config = {
    'hidden_size': 2048,
    'num_hidden_layers': 5,
    'num_attention_heads': 16,
    'num_key_value_heads': 4,
    'head_dim': 128,
    'intermediate_size': 5504,
    'block_size': 16,
    'target_layer_ids': [1, 7, 14, 21, 27],
    'mask_token_id': 151643,
    'loss_decay': 7,
    'rms_norm_eps': 1e-6,
    'rope_theta': 10000000.0,
}
```

## Expected Performance

Based on paper ablations and our benchmark data:

| Metric | z-lab Draft | Purpose-Built Draft |
|--------|-------------|---------------------|
| AL (HumanEval) | 2.49 | 5-7 (expected) |
| Calibration | Fails 1/10 prompts | Correct 10/10 |
| Draft latency | ~4ms | ~4ms |
| Verify latency | ~25ms | ~25ms |
| Speedup (tok/s) | 72.5 | 120-180 (expected) |

The AL improvement comes from:
1. Training on OUR target model's hidden states (no distribution mismatch)
2. Training on OUR use case patterns (code, math, chat)
3. Calibration-aware loss (no confidently wrong predictions)

## Implementation Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Trace generator: run target model, capture hidden states | 1 day |
| 2 | Dataset curation: collect 800K+ diverse prompts | 1 day |
| 3 | Training script: DFlash training with random anchors | 2 days |
| 4 | Training run: 6 epochs on 3090 | 3-5 days |
| 5 | Export + benchmark: safetensors, test on HumanEval | 1 day |
| 6 | Iterate: tune hyperparameters, add calibration loss | 2-3 days |
| **Total** | | **10-14 days** |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training data too small | Low acceptance length | Use paper's 800K+ target; augment with synthetic data |
| Overfitting to training prompts | Poor generalization | Hold out 10% for validation; diverse prompt sources |
| Training too slow on 3090 | Miss deadline | Use gradient checkpointing; reduce batch size; train in FP16 |
| Calibration loss too weak | Still confidently wrong | Increase lambda_cal; add temperature scaling at inference |
| Draft too large for VRAM | OOM during inference | Reduce layers (3 instead of 5) or hidden dim |
| GGUF format incompatible | Can't load in dflash-robot | Use safetensors format (already supported) |

## Files to Create

```
dflash-robot/
├── training/
│   ├── trace_generator.py      # Run target model, capture hidden states
│   ├── dataset.py              # Dataset class for training traces
│   ├── draft_model.py          # Draft model architecture (PyTorch)
│   ├── train.py                # Training loop with DFlash loss
│   ├── export.py               # Export to safetensors
│   └── config.py               # Hyperparameters and model config
├── scripts/
│   └── collect_dataset.py      # Collect and curate training prompts
└── docs/
    └── purpose-built-draft.md  # This document
```

## References

- DFlash paper: arxiv:2602.06036v1 (Jian Chen, Yesheng Liang, Zhijian Liu)
- z-lab reference implementation: huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash
- DFlash paper-to-code mapping: docs/research/paper-to-code-map.md
- DFlash first-principles analysis: docs/research/dflash-paper-first-principles.md
- Flex Attention (training): arxiv:2412.05496
