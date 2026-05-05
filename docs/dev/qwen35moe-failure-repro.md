# qwen35moe Failure Reproduction

Date: 2026-05-04
Git: 3f648c8 feat: modular draft loader, qwen35moe target support

## Command

```
DFLASH_TARGET=/home/am/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
./build/smoke_target_forward \
  "/home/am/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
```

## Expected

Target forward pass succeeds, produces logits.

## Actual

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24126 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24126 MiB
[loader] eos_id=248046 eos_chat_id=-1
[target] target loaded: 733 tensors on GPU 20.10 GiB, tok_embd 515 MiB CPU-only (q8_0)
[cache] attn_k=10 attn_v=10 ssm=30 conv=30
ggml.c:2222: GGML_ASSERT(ggml_can_repeat(b, a)) failed
```

## Backtrace

```
#4  ggml_print_backtrace
#5  ggml_abort
#6  ggml_mul_impl
#7  ggml_mul
#8  dflash27b::build_qwen35_graph
#9  main
```

## Root Cause

`build_qwen35_graph()` at lines 1331-1333 unconditionally calls `build_swiglu_ffn()`:

```cpp
ggml_tensor * ffn = build_swiglu_ffn(ctx, post, L);
```

`build_swiglu_ffn` expects dense FFN weights with shape [hidden, intermediate], but MoE layers have different tensor shapes for their expert weights, causing a `ggml_can_repeat` assertion failure inside ggml_mul.

The fix is in `build_single_layer()` at line 1215-1217 which already has the conditional:

```cpp
ggml_tensor * ffn  = L.ffn_gate_inp
    ? build_moe_ffn(ctx, gf, w, L, post)
    : build_swiglu_ffn(ctx, post, L);
```

But `build_qwen35_graph()` (lines 1331-1333) was never updated with this same conditional.

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM
- CUDA: 13.2 (miniforge)
- Build: cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
- Target: unsloth/Qwen3.6-35B-A3B-GGUF, Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
- Loader output: 733 tensors, 20.10 GiB, arch qwen35moe, 256 experts / 8 active
