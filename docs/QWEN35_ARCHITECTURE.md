QWEN3.5 / QWEN3.6 ARCHITECTURE REFERENCE
==========================================
Ground truth for dflash-robot graph module implementation.
Sources: HF config.json, llama.cpp src/models/qwen35.cpp, qwen35moe.cpp,
         delta-net-base.cpp, llama-model.cpp tensor creation, dflash-robot internal.h


==============================================================================
1. MODEL FAMILIES AND NAMING
==============================================================================

Qwen3.5 = the "dense" hybrid family (4B, 9B, 27B)
  GGUF arch: "qwen35"     enum: LLM_ARCH_QWEN35
  HF model_type: qwen3_5
  HF class: Qwen3_5ForConditionalGeneration

Qwen3.6 = a _pretrained checkpoint_ of the same architecture (preview)
  GGUF arch: "qwen35"     (same as Qwen3.5!)
  HF model_type: qwen3_5  (same!)
  Qwen3.6-27B has identical architecture to Qwen3.5-27B.
  The ONLY difference is training data / checkpoint quality.

Qwen3.5/Qwen3.6 MoE = MoE variant of the hybrid architecture
  GGUF arch: "qwen35moe"  enum: LLM_ARCH_QWEN35MOE
  HF model_type: qwen3_5_moe
  HF class: Qwen3_5MoeForConditionalGeneration
  Models: Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B

CONCLUSION: Qwen3.5 and Qwen3.6 use IDENTICAL graph ops. No architectural difference.


==============================================================================
2. HYBRID ARCHITECTURE: FULL-ATTENTION vs DELTA-NET LAYER INTERLEAVING
==============================================================================

Pattern: every 4th layer is full-attention, rest are DeltaNet.
  full_attention_interval = 4
  is_recurrent(il) = ((il + 1) % 4 != 0)
  Layers 0,1,2   -> DeltaNet (linear attention)
  Layer  3        -> FULL ATTENTION
  Layers 4,5,6   -> DeltaNet
  Layer  7        -> FULL ATTENTION
  ...
  Layer  63 (last for 64-layer) -> FULL ATTENTION

Total for 64-layer model (27B):
  Full-attention layers: 16  (at il=3,7,11,...,63)
  DeltaNet layers:       48  (the remaining)

Total for 40-layer model (35B-A3B):
  Full-attention layers: 10  (at il=3,7,11,...,39)
  DeltaNet layers:       30  (the remaining)

KEY: The layer_type list in HF config is:
  ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
  This is 0-indexed: indices 0,1,2 are linear, index 3 is full, etc.


==============================================================================
3. FULL-ATTENTION BLOCK
==============================================================================

3.1 PARAMETERS (both 27B and 35B-A3B)
--------------------------------------

27B (Qwen3.5-27B / Qwen3.6-27B):
  hidden_size = 5120
  head_dim    = 256
  num_attention_heads (n_head)    = 24
  num_key_value_heads (n_head_kv) = 4       (GQA ratio = 6:1)
  q_dim  = n_head * head_dim = 6144
  kv_dim = n_head_kv * head_dim = 1024
  rope_sections = [11, 11, 10]   (M-RoPE 3-axis)
  n_rot = 2 * (11 + 11 + 10) = 64   (= partial_rotary_factor 0.25 * head_dim)
  rope_theta = 10000000.0
  attention_bias = false
  attn_output_gate = true  (packed Q || gate in single projection)
  output_gate_type = "swish" (sigmoid gate in 3.6 config)

35B-A3B (Qwen3.5-35B-A3B / Qwen3.6-35B-A3B):
  hidden_size = 2048
  head_dim    = 256
  num_attention_heads (n_head)    = 16
  num_key_value_heads (n_head_kv) = 2       (GQA ratio = 8:1)
  q_dim  = n_head * head_dim = 4096
  kv_dim = n_head_kv * head_dim = 512
  rope_sections = [11, 11, 10]   (same M-RoPE 3-axis)
  n_rot = 64
  rope_theta = 10000000.0
  attention_bias = false
  attn_output_gate = true

3.2 TENSOR NAMES AND SHAPES
-----------------------------

Per full-attention layer (layer il where (il+1)%4==0):

  blk.{il}.attn_norm.weight      [hidden_size]               RMS-norm before attn
  blk.{il}.attn_post_norm.weight [hidden_size]               RMS-norm before FFN

  blk.{il}.attn_q.weight         [hidden_size, q_dim*2]      PACKED Q+gate projection
  blk.{il}.attn_k.weight         [hidden_size, kv_dim]       K projection
  blk.{il}.attn_v.weight         [hidden_size, kv_dim]       V projection
  blk.{il}.attn_output.weight    [q_dim, hidden_size]        output projection

  blk.{il}.attn_q_norm.weight    [head_dim]                  per-head Q RMS norm
  blk.{il}.attn_k_norm.weight    [head_dim]                  per-head K RMS norm

  27B shapes:
    wq: [5120, 12288]  = [5120, 2*6144]   (Q || gate packed)
    wk: [5120, 1024]
    wv: [5120, 1024]
    wo: [6144, 5120]
    q_norm: [256]
    k_norm: [256]

  35B-A3B shapes:
    wq: [2048, 8192]  = [2048, 2*4096]
    wk: [2048, 512]
    wv: [2048, 512]
    wo: [4096, 2048]
    q_norm: [256]
    k_norm: [256]

3.3 GRAPH OPERATIONS (llama.cpp's build_layer_attn)
----------------------------------------------------

Step 1: Pre-attention norm
  cur = rms_norm(inpL, attn_norm) * attn_norm

Step 2: Packed Q projection (Q + gate)
  QG = wq @ cur                        # [q_dim*2, n_tokens]
  QG = reshape(QG, [head_dim*2, n_head, n_tokens])

  Q    = view(QG, offset=0,  stride=head_dim*2)   # [head_dim, n_head, n_tokens]
  gate = view(QG, offset=head_dim, stride=head_dim*2) # [head_dim, n_head, n_tokens]
  gate = cont_2d(gate, q_dim, n_tokens)            # [q_dim, n_tokens]

Step 3: Q norm
  Q = rms_norm(Q, q_norm) * q_norm     (per-head RMS norm)

Step 4: K, V projections
  Kcur = wk @ cur                       # [kv_dim, n_tokens]
  Vcur = wv @ cur
  Kcur = reshape(Kcur, [head_dim, n_head_kv, n_tokens])
  Kcur = rms_norm(Kcur, k_norm) * k_norm   (per-head RMS norm)
  Vcur = reshape(Vcur, [head_dim, n_head_kv, n_tokens])

Step 5: M-RoPE (multi-axis rotary position embedding)
  rope_type = GGML_ROPE_TYPE_IMROPE (=MROPE with interleaved=True)
  sections = [11, 11, 10, 0]  (4th section is 0)
  n_rot = 2 * (11+11+10+0) = 64

  Q    = rope_multi(Q,    positions, n_rot=64, sections=[11,11,10,0], theta=1e7)
  Kcur = rope_multi(Kcur, positions, n_rot=64, sections=[11,11,10,0], theta=1e7)

  NOTE: positions is [4 * n_tokens] i32 for M-RoPE (4 axis values per token).

Step 6: GQA Flash Attention (causal)
  kq_scale = 1/sqrt(head_dim) = 1/16
  attn = flash_attn_ext(Q, K_cache, V_cache, mask, kq_scale)

Step 7: Sigmoid gate
  gate_sigmoid = sigmoid(gate)
  attn = attn * gate_sigmoid             # element-wise gating

Step 8: Output projection
  out = wo @ attn                        # [hidden_size, n_tokens]


==============================================================================
4. GATED DELTANET BLOCK
==============================================================================

4.1 SSM PARAMETERS
-------------------

                          27B       35B-A3B
  ssm_d_inner             6144      4096
  ssm_d_state             128       128       (= key head dim)
  ssm_dt_rank             48        32        (= number of value heads!)
  ssm_n_group             16        16        (= number of key heads)
  ssm_d_conv              4         4         (1D causal conv kernel size)

DERIVED DIMENSIONS:
  head_k_dim    = ssm_d_state                    = 128
  num_k_heads   = ssm_n_group                    = 16
  num_v_heads   = ssm_dt_rank                    = 48 / 32
  head_v_dim    = ssm_d_inner / ssm_dt_rank      = 128

  key_dim       = head_k_dim * num_k_heads       = 2048  (both models!)
  value_dim     = head_v_dim * num_v_heads       = 6144 / 4096  (= ssm_d_inner)
  conv_channels = key_dim*2 + value_dim           = 10240 / 8192

  27B:    conv_channels = 2*2048 + 6144 = 10240
  35B:    conv_channels = 2*2048 + 4096 = 8192

4.2 TENSOR NAMES AND SHAPES
-----------------------------

Per DeltaNet layer (layer il where (il+1)%4 != 0):

  blk.{il}.attn_norm.weight      [hidden_size]               pre-DeltaNet norm
  blk.{il}.attn_post_norm.weight [hidden_size]               pre-FFN norm

  blk.{il}.attn_qkv.weight       [hidden_size, conv_channels]
      = fused QKV projection (Q || K || V concatenated on output dim)
      Shape: [hidden_size, 2*key_dim + value_dim]
      27B:   [5120, 10240]
      35B:   [2048, 8192]

  blk.{il}.attn_gate.weight      [hidden_size, value_dim]    "z" gate projection
      27B:   [5120, 6144]
      35B:   [2048, 4096]

  blk.{il}.ssm_conv1d.weight     [ssm_d_conv, conv_channels]  1D causal conv
      27B:   [4, 10240]
      35B:   [4, 8192]

  blk.{il}.ssm_dt.bias           [num_v_heads]                alpha bias (per-head)
      27B:   [48]
      35B:   [32]

  blk.{il}.ssm_a                 [num_v_heads]                -A_log parameter
      27B:   [48]
      35B:   [32]

  blk.{il}.ssm_beta.weight       [hidden_size, num_v_heads]   beta projection
      27B:   [5120, 48]
      35B:   [2048, 32]

  blk.{il}.ssm_alpha.weight      [hidden_size, num_v_heads]   alpha projection
      27B:   [5120, 48]
      35B:   [2048, 32]

  blk.{il}.ssm_norm.weight       [head_v_dim]                 output norm
      27B:   [128]
      35B:   [128]

  blk.{il}.ssm_out.weight        [value_dim, hidden_size]     output projection
      27B:   [6144, 5120]
      35B:   [4096, 2048]

4.3 GRAPH OPERATIONS (llama.cpp's build_layer_attn_linear)
-----------------------------------------------------------

Step 1: Pre-DeltaNet norm
  cur = rms_norm(inpL, attn_norm) * attn_norm

Step 2: QKV + gate projections
  qkv_mixed = wqkv @ cur           # [conv_channels, n_tokens]
  z         = wqkv_gate @ cur      # [value_dim, n_tokens]

Step 3: Beta projection
  beta = ssm_beta @ cur             # [num_v_heads, n_tokens]
  beta = reshape(beta, [1, num_v_heads, n_tokens, 1])
  beta = sigmoid(beta)

Step 4: Alpha -> gate
  alpha = ssm_alpha @ cur           # [num_v_heads, n_tokens]
  alpha = reshape(alpha, [num_v_heads, n_tokens, 1])
  alpha = alpha + ssm_dt_bias       # [num_v_heads] broadcast
  alpha = softplus(alpha)
  gate  = alpha * ssm_a             # elementwise: -A_log.exp() * softplus
  gate  = reshape(gate, [1, num_v_heads, n_tokens, 1])

Step 5: Conv state prep
  conv_state: [ssm_d_conv-1, conv_channels]  (= [3, conv_channels]) persistent
  qkv_T = transpose(qkv_mixed)                # [n_tokens, conv_channels, 1]
  conv_input = concat(conv_state, qkv_T, dim0) # [3+n_tokens, conv_channels, 1]
  Save last (ssm_d_conv-1) rows back to conv_state.

Step 6: 1D causal conv + SiLU
  conv_out = ssm_conv(conv_input, ssm_conv1d)  # [conv_channels, n_tokens, 1]
  conv_out = silu(conv_out)

Step 7: Split into Q, K, V
  Layout of conv_channels:
    [0 .. key_dim)                    -> Q
    [key_dim .. 2*key_dim)            -> K
    [2*key_dim .. 2*key_dim+value_dim) -> V

  q_c = view(conv_out, [head_k_dim, num_k_heads, n_tokens, 1], offset=0)
  k_c = view(conv_out, [head_k_dim, num_k_heads, n_tokens, 1], offset=key_dim)
  v_c = view(conv_out, [head_v_dim, num_v_heads, n_tokens, 1], offset=2*key_dim)

Step 8: L2 normalize Q, K
  q_c = l2_norm(q_c, eps=1e-6)
  k_c = l2_norm(k_c, eps=1e-6)

Step 9: Repeat Q, K to match V head count (if num_k_heads != num_v_heads)
  q_c = repeat(q_c, [head_k_dim, num_v_heads, n_tokens, 1])
  k_c = repeat(k_c, [head_k_dim, num_v_heads, n_tokens, 1])
  NOTE: Only needed when not using fused GDN kernel that broadcasts internally.

Step 10: Gated DeltaNet recurrence
  ssm_state: [head_v_dim, head_v_dim, num_v_heads] persistent, f32
  s = reshape(ssm_state, [head_v_dim, head_v_dim, num_v_heads, 1])

  result = ggml_gated_delta_net(q_c, k_c, v_c, gate, beta, s)
  The fused kernel returns:
    result = [output | final_state | (optional)intermediate_states]
  output:     [head_v_dim, num_v_heads, n_tokens, 1]
  new_state:  [head_v_dim, head_v_dim, num_v_heads, 1]
  Save new_state back to ssm_state.

Step 11: Gated output normalization
  z_4d = reshape(z, [head_v_dim, num_v_heads, n_tokens, 1])
  output_norm = rms_norm(output, eps=1e-6) * ssm_norm
  output_gated = output_norm * silu(z_4d)

Step 12: Output projection
  flat = reshape(output_gated, [value_dim, n_tokens])
  out  = ssm_out @ flat              # [hidden_size, n_tokens]

4.4 RECURRENT STATE SIZES
--------------------------

Per DeltaNet layer:
  conv_state: [ssm_d_conv-1, conv_channels] = [3, conv_channels] * 4 bytes
    27B:    3 * 10240 * 4 = 122,880 bytes  (~120 KB)
    35B:    3 * 8192  * 4 = 98,304 bytes   (~96 KB)

  ssm_state: [head_v_dim, head_v_dim, num_v_heads] = [128, 128, num_v_heads] * 4 bytes
    27B:    128*128*48*4 = 3,145,728 bytes (~3 MB)
    35B:    128*128*32*4 = 2,097,152 bytes (~2 MB)

Total SSM state per token verification step (all delta-net layers):
  27B: 48 layers * 3 MB = 144 MB
  35B: 30 layers * 2 MB = 60 MB


==============================================================================
5. MIXTURE-OF-EXPERTS FFN (35B-A3B only)
==============================================================================

5.1 PARAMETERS
--------------

  num_experts (n_expert)          = 256
  num_experts_per_tok (n_expert_used) = 8
  moe_intermediate_size (n_ff_exp) = 512    per-expert FFN width
  shared_expert_intermediate_size (n_ff_shared) = 512
  hidden_act = "silu"             (SwiGLU for both expert and shared FFN)
  router: softmax over 256 experts, top-8 selection

5.2 EVERY LAYER has MoE FFN (both full-attn and DeltaNet layers)
-----------------------------------------------------------------

In qwen35moe, ALL 40 layers use MoE FFN. There is no dense FFN at all.
(llama.cpp asserts ffn_gate_inp != nullptr for every layer)

5.3 TENSOR NAMES AND SHAPES (per layer)
-----------------------------------------

  blk.{il}.ffn_gate_inp.weight          [hidden_size, n_expert]
      = router projection
      [2048, 256]

  blk.{il}.ffn_gate_exps.weight         [n_expert, n_ff_exp, hidden_size]
      = expert gate weights (3D, transposed for mul_mat_id)
      [256, 512, 2048]  or equivalently [hidden_size, n_ff_exp, n_expert]

  blk.{il}.ffn_up_exps.weight           [n_expert, n_ff_exp, hidden_size]
      = expert up weights
      [256, 512, 2048]

  blk.{il}.ffn_down_exps.weight         [n_expert, hidden_size, n_ff_exp]
      = expert down weights
      [256, 2048, 512]

  blk.{il}.ffn_gate_inp_shexp.weight    [hidden_size]
      = SHARED EXPERT GATE (scalar per token, sigmoid-activated)
      [2048]          ** CRITICAL: THIS TENSOR IS MISSING FROM dflash-robot **

  blk.{il}.ffn_gate_shexp.weight        [hidden_size, n_ff_shared]
      = shared expert gate projection (SwiGLU gate half)
      [2048, 512]

  blk.{il}.ffn_up_shexp.weight          [hidden_size, n_ff_shared]
      = shared expert up projection (SwiGLU up half)
      [2048, 512]

  blk.{il}.ffn_down_shexp.weight        [n_ff_shared, hidden_size]
      = shared expert down projection
      [512, 2048]

5.4 MoE FFN GRAPH OPERATIONS (llama.cpp's build_layer_ffn)
-----------------------------------------------------------

Step 1: Router
  logits = ffn_gate_inp @ cur            # [n_expert, n_tokens] = [256, n_tokens]
  probs  = softmax(logits)

Step 2: Top-k selection
  selected = argsort_top_k(probs, k=8)  # [8, n_tokens] i32 indices

Step 3: Expert SwiGLU (via ggml_mul_mat_id)
  cur_3d = reshape(cur, [hidden, 1, n_tokens])
  gate   = mul_mat_id(ffn_gate_exps, cur_3d, selected)  # [n_ff_exp, 8, n_tokens]
  up     = mul_mat_id(ffn_up_exps,   cur_3d, selected)
  swiglu = swiglu_split(gate, up)                        # [n_ff_exp, 8, n_tokens]
  experts = mul_mat_id(ffn_down_exps, swiglu, selected)  # [hidden, 8, n_tokens]

Step 4: Weighted sum
  weights = get_rows(probs_3d, selected)  # [1, 8, n_tokens]
  experts = experts * weights
  moe_out = sum(experts, dim=1)           # [hidden, n_tokens]

Step 5: SHARED EXPERT (with gate!)
  sh_gate = ffn_gate_shexp @ cur          # [n_ff_shared, n_tokens]
  sh_up   = ffn_up_shexp @ cur            # [n_ff_shared, n_tokens]
  sh_gu   = swiglu_split(sh_gate, sh_up)
  sh_down = ffn_down_shexp @ sh_gu        # [hidden, n_tokens]

  ** CRITICAL GATE: **
  shared_gate = sigmoid(ffn_gate_inp_shexp @ cur)  # [1, n_tokens] per-token gate
  sh_down = sh_down * shared_gate                   # gate the shared expert

Step 6: Combine
  ffn_out = moe_out + sh_down


==============================================================================
6. LAYER STRUCTURE SUMMARY (per layer, applies to BOTH attn types)
==============================================================================

For every layer il (0 to n_layer-1):

  inpSA = inpL                                          # save residual
  cur   = rms_norm(inpL, attn_norm) * attn_norm         # pre-block norm

  if is_recurrent(il):
      cur = build_delta_net_block(cur, ...)              # DeltaNet
  else:
      cur = build_full_attn_block(cur, ...)              # full attention

  cur = cur + inpSA                                      # residual connection

  ffn_residual = cur
  cur = rms_norm(cur, attn_post_norm) * attn_post_norm   # post-attn norm

  if is_moe:
      cur = build_moe_ffn(cur, ...)                      # MoE FFN
  else:
      cur = build_swiglu_ffn(cur, ...)                   # dense SwiGLU

  cur = cur + ffn_residual                               # FFN residual

  inpL = cur   # output for next layer


==============================================================================
7. FFN (Dense, 27B only)
==============================================================================

For 27B: SwiGLU FFN on every layer
  w_gate: [hidden_size, n_ff] = [5120, 17408]
  w_up:   [hidden_size, n_ff] = [5120, 17408]
  w_down: [n_ff, hidden_size] = [17408, 5120]

  ffn_out = w_down @ (silu(w_gate @ x) * (w_up @ x))


==============================================================================
8. HIDDEN STATE CAPTURE FOR SPECULATIVE DECODING
==============================================================================

8.1 TARGET LAYER IDS (DFlash feature capture)
----------------------------------------------

The DFlash draft model receives hidden states from specific target model layers
as "context features". These are captured AFTER the layer's residual + FFN.

27B (64 target layers, 5 draft layers):
  target_layer_ids = [1, 16, 31, 46, 61]
  Formula: ids[k] = 1 + k * 15   (step = (64-2)/(5-1) = 15.5 -> 15)

35B-A3B (40 target layers, 8 draft layers):
  target_layer_ids = [1, 10, 19, 28, 37]
  Formula: ids[k] = 1 + k * 9    (step = (40-2)/(8-1) = 5.4 -> ~5, but given 9)

  NOTE: The 35B-A3B DFlash config says num_target_layers=40, but it captures
  at 5 target positions (not 8), feeding into 8 draft layers.
  Looking at dflash config: target_layer_ids=[1,10,19,28,37], num_hidden_layers=8
  This means 5 target feature vectors concatenated -> 5*2048=10240 -> fc -> 2048

8.2 CAPTURE MECHANISM (in target graph)
----------------------------------------

After each layer il runs, if il is in capture_layer_ids:

  cur_2d = reshape(cur, [hidden, n_tokens])   # post-FFN output

  # Write into target_feat ring buffer:
  slot = view_2d(target_feat, hidden, n_tokens,
                 col_stride, offset=kv_start*col_stride + cap_idx*hidden*elt)
  ggml_cpy(cur_2d, slot)

  target_feat layout: [N_CAPTURE_LAYERS * hidden, target_feat_cap]
    27B:   [5*5120, 4096] = [25600, 4096]
    35B:   [5*2048, 4096] = [10240, 4096]

  Ring buffer: position P maps to slot P % target_feat_cap
  Handles wrap-around with two ggml_cpy ops per capture.

8.3 HOW DFLASH DRAFT USES TARGET HIDDEN STATES
-----------------------------------------------

From z-lab reference (dflash/model.py):

  1. Target model runs forward, collecting hidden_states from all layers.
  2. extract_context_feature(hidden_states, target_layer_ids):
       # offset=1 convention: hidden_states[layer_id + 1]
       selected = [hidden_states[lid+1] for lid in target_layer_ids]
       return torch.cat(selected, dim=-1)   # [batch, seq, N*hidden]

  3. In draft model forward():
       target_hidden = self.hidden_norm(self.fc(target_hidden))
       # fc: [N*hidden, hidden] linear projection
       # hidden_norm: RMSNorm

  4. Each draft decoder layer receives target_hidden as cross-attention context:
       k_ctx = k_proj(target_hidden)    # K from target features
       v_ctx = v_proj(target_hidden)    # V from target features
       k_noise = k_proj(hidden_states)  # K from draft's own hidden
       v_noise = v_proj(hidden_states)  # V from draft's own hidden
       k = cat([k_ctx, k_noise], dim=1) # concat along seq dimension
       v = cat([v_ctx, v_noise], dim=1)
       # Then standard attention: Q from draft, K/V from cat(context, draft)


==============================================================================
9. CRITICAL BUGS IN dflash-robot (35B-A3B MoE)
==============================================================================

BUG #1: MISSING SHARED EXPERT GATE (ffn_gate_inp_shexp)
---------------------------------------------------------

  LOCATION: internal.h TargetLayer struct, gguf_target_loader.cpp, build_moe_ffn()

  llama.cpp qwen35moe.cpp loads this tensor per layer:
    layer.ffn_gate_inp_shexp = create_tensor(
        tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), { n_embd }, 0);

  And uses it in build_layer_ffn:
    shared_gate = ffn_gate_inp_shexp @ cur    # [1, n_tokens]
    shared_gate = sigmoid(shared_gate)
    ffn_shexp   = ffn_shexp * shared_gate     # per-token gating

  dflash-robot:
    - TargetLayer struct has NO ffn_gate_inp_shexp field
    - gguf_target_loader.cpp does NOT load "ffn_gate_inp_shexp.weight"
    - build_moe_ffn() adds shared expert output WITHOUT gating

  IMPACT: The shared expert contributes ~100% unmodulated instead of being
  gated per-token. This corrupts ALL MoE layer outputs, causing the draft
  model to see wrong target hidden states and produce wrong logits.
  DIRECT CAUSE OF THE 7% ACCEPTANCE RATE.

  FIX REQUIRED:
    a) Add `ggml_tensor * ffn_gate_inp_shexp` to TargetLayer in internal.h
    b) Load it in gguf_target_loader.cpp:
       L.ffn_gate_inp_shexp = fnd("ffn_gate_inp_shexp.weight");
    c) In build_moe_ffn(), after computing sh_down:
       if (L.ffn_gate_inp_shexp) {
           ggml_tensor * shared_gate = ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, cur);
           shared_gate = ggml_sigmoid(ctx, shared_gate);
           sh_down = ggml_mul(ctx, sh_down, shared_gate);
       }
       moe_out = ggml_add(ctx, moe_out, sh_down);

BUG #2: n_ff_expert vs n_ff_exp naming confusion
---------------------------------------------------

  internal.h uses n_ff_expert (line 144).
  GGUF metadata key is "qwen35moe.expert_feed_forward_length".
  llama.cpp uses hparams.n_ff_exp.
  gguf_target_loader.cpp reads it into out.n_ff_expert.
  build_moe_ffn() uses w.n_ff_expert -> 0 because the field was set but
  the actual FFN width comes from the expert tensor shapes.

  Verify that n_ff_expert == 512 (35B-A3B) is being loaded correctly.


==============================================================================
10. COMPLETE DIMENSIONS TABLE
==============================================================================

                          Qwen3.5-27B    Qwen3.6-35B-A3B
  ---- Core ----
  hidden_size             5120           2048
  vocab_size              248320         248320
  n_layer                 64             40
  full_attn_interval      4              4
  n_full_attn             16             10
  n_delta_net             48             30

  ---- Full Attention ----
  head_dim                256            256
  n_head                  24             16
  n_head_kv               4              2
  q_dim                   6144           4096
  kv_dim                  1024           512
  n_rot (rope dims)       64             64
  rope_sections           [11,11,10,0]   [11,11,10,0]
  rope_theta              1e7            1e7

  ---- DeltaNet SSM ----
  ssm_d_inner             6144           4096
  ssm_d_state (head_k_dim) 128           128
  ssm_dt_rank (num_v_heads) 48           32
  ssm_n_group (num_k_heads) 16           16
  ssm_d_conv              4              4
  head_v_dim              128            128
  key_dim                 2048           2048
  value_dim               6144           4096
  conv_channels           10240          8192

  ---- FFN ----
  is_moe                  false          true
  n_ff (dense)            17408          N/A
  n_expert                N/A            256
  n_expert_used           N/A            8
  n_ff_exp (per expert)   N/A            512
  n_ff_shared             N/A            512

  ---- DFlash Draft ----
  target_layer_ids        [1,16,31,46,61] [1,10,19,28,37]
  n_draft_layers          5              8
  draft_hidden            5120           2048
  draft_head_dim          128            128
  draft_n_head            32             32
  draft_n_head_kv         8              4
  draft_n_ff              17408          6144
  draft_block_size        16             16
  draft_mask_token_id     248070         248070
  draft_rope              (no scaling)   yarn(factor=64)

  ---- RMS Norm ----
  rms_norm_eps            1e-6           1e-6


==============================================================================
11. GGUF TENSOR NAME CONVENTION
==============================================================================

Prefix: blk.{il}.

Full-attention layer tensors:
  blk.{il}.attn_norm.weight
  blk.{il}.attn_post_norm.weight
  blk.{il}.attn_q.weight         (packed Q+gate)
  blk.{il}.attn_k.weight
  blk.{il}.attn_v.weight
  blk.{il}.attn_output.weight
  blk.{il}.attn_q_norm.weight
  blk.{il}.attn_k_norm.weight

DeltaNet layer tensors:
  blk.{il}.attn_norm.weight
  blk.{il}.attn_post_norm.weight
  blk.{il}.attn_qkv.weight       (fused QKV)
  blk.{il}.attn_gate.weight      (z gate)
  blk.{il}.ssm_conv1d.weight
  blk.{il}.ssm_dt.bias
  blk.{il}.ssm_a                 (no suffix - the raw -A_log parameter)
  blk.{il}.ssm_beta.weight
  blk.{il}.ssm_alpha.weight
  blk.{il}.ssm_norm.weight
  blk.{il}.ssm_out.weight

Dense FFN tensors (27B):
  blk.{il}.ffn_gate.weight
  blk.{il}.ffn_up.weight
  blk.{il}.ffn_down.weight

MoE FFN tensors (35B-A3B):
  blk.{il}.ffn_gate_inp.weight           (router)
  blk.{il}.ffn_gate_exps.weight          (expert gate, 3D)
  blk.{il}.ffn_up_exps.weight            (expert up, 3D)
  blk.{il}.ffn_down_exps.weight          (expert down, 3D)
  blk.{il}.ffn_gate_inp_shexp.weight     (shared expert gate - MISSING from dflash-robot!)
  blk.{il}.ffn_gate_shexp.weight         (shared expert SwiGLU gate)
  blk.{il}.ffn_up_shexp.weight           (shared expert SwiGLU up)
  blk.{il}.ffn_down_shexp.weight         (shared expert down)

Global tensors:
  token_embd.weight              [vocab, hidden]
  output_norm.weight             [hidden]
  output.weight                  [vocab, hidden]  (lm_head, may tie with token_embd)


==============================================================================
12. DELTA-NET RECURRENCE MATH (gated_delta_net)
==============================================================================

The Gated DeltaNet implements a linear attention variant with gated recurrence:

For each token t:
  State update:
    S_t = diag(g_t) * S_{t-1} + beta_t * (v_t outer k_t)
  where:
    g_t = exp(softplus(alpha_t + dt_bias) * (-A_log))  # per-head decay
    beta_t = sigmoid(beta_proj(x_t))                    # per-head write strength

  Output computation:
    o_t = S_t @ q_t                                       # [head_v_dim]
    o_t = l2_norm(q_t)^T @ S_t  but with L2-normed q,k    # normalized attention

  Output gating:
    z_t = wqkv_gate @ x_t                                # [value_dim]
    out_t = rms_norm(o_t, ssm_norm) * silu(z_t)          # gated output

  Key insight: Q and K are L2-normalized before the recurrence, not scaled.
  The beta controls how much new information overwrites the state.
  The gate (g_t) controls exponential decay of old state.

State shape: [head_v_dim x head_v_dim] per value head
  This is a low-rank outer-product state, NOT a scalar per head.


==============================================================================
13. QWEN3.5 vs QWEN3.6 vs QWEN3NEXT DIFFERENCES
==============================================================================

Qwen3Next (qwen3next arch):
  - Earlier version of the same hybrid architecture
  - Different broadcasting pattern for DeltaNet heads
  - Uses LLAMA_ROPE_TYPE_IMROPE (same as Qwen3.5)
  - GGUF arch: "qwen3next"

Qwen3.5 (qwen35 arch):
  - Production version
  - Key/value head broadcasting: [k0_v0, k0_v1, k0_v2, k0_v3, k1_v4, ...]
    (each key head broadcasts to num_v_heads/num_k_heads value heads contiguously)
  - Uses LLAMA_ROPE_TYPE_IMROPE

Qwen3.6 (same qwen35 arch):
  - Same architecture as Qwen3.5
  - Different pretrained checkpoint (preview)
  - GGUF arch: "qwen35" (identical!)

Qwen3.5 MoE (qwen35moe arch):
  - MoE variant with 256 experts, top-8 routing
  - Has GATED shared expert (ffn_gate_inp_shexp)
  - Same hybrid attn pattern as dense

No architectural differences between Qwen3.5 and Qwen3.6 whatsoever.


==============================================================================
END OF DOCUMENT
==============================================================================
