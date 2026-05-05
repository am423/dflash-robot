# Luce DFlash Hardcoded Assumptions Audit

Generated: 2026-05-04
Source: /tmp/lucebox-hub-dflash-inspect/dflash/

## Architecture Name / Qwen35 Specific

| File | Line | Context |
|---|---|---|
| _prefill_hook.py | 12 | `Qwen3.5/3.6-27B), so the pipeline is:` |
| bench_he.py | 306 | `default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B"),` |
| bench_he.py | 308 | `"$DFLASH_TOKENIZER, then Qwen/Qwen3.5-27B. Override for "` |
| bench_llm.py | 11 | `DFLASH_TOKENIZER HF tokenizer repo (default Qwen/Qwen3.5-27B; matches run.py)` |
| bench_llm.py | 32 | `TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B")` |
| chat.py | 96 | `tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",` |
| convert_dflash_to_gguf.py | 26 | `qwen3.5-27b-dflash-draft.gguf` |
| convert_dflash_to_gguf.py | 45 | `ARCH                = "qwen35-dflash-draft"` |
| convert_dflash_to_gguf.py | 161 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft")` |
| delta_net_chunked.cpp | 44 | `// GDA only in our port — Qwen3.5 delta-net uses gate scalar per head` |
| detokenize.py | 10 | `ap.add_argument("--model", default="Qwen/Qwen3.5-27B")` |
| dflash27b.h | 2 | `// Qwen3.5-27B with the z-lab/Qwen3.5-27B-DFlash draft model on a single RTX 3090.` |
| dflash27b.h | 2 | `// Qwen3.5-27B with the z-lab/Qwen3.5-27B-DFlash draft model on a single RTX 3090.` |
| dflash27b.h | 24 | `// Qwen3.5-27B qwen35 hybrid uses 24 Q heads, 4 KV heads, 256 head_dim, which` |
| dflash27b.h | 24 | `// Qwen3.5-27B qwen35 hybrid uses 24 Q heads, 4 KV heads, 256 head_dim, which` |
| gguf_draft_loader.cpp | 8 | `// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /` |
| gguf_draft_loader.cpp | 135 | `if (std::string(arch) != "qwen35-dflash-draft") {` |
| gguf_draft_loader.cpp | 137 | `" (expected qwen35-dflash-draft)");` |
| gguf_draft_loader.cpp | 144 | `const char * A = "qwen35-dflash-draft";` |
| gguf_target_loader.cpp | 1 | `// Loads Qwen3.5-27B qwen35 hybrid from a GGUF file on disk into a ggml` |
| gguf_target_loader.cpp | 1 | `// Loads Qwen3.5-27B qwen35 hybrid from a GGUF file on disk into a ggml` |
| gguf_target_loader.cpp | 4 | `// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See` |
| gguf_target_loader.cpp | 5 | `// unsloth/Qwen3.5-27B-GGUF or ddh0/Qwen3.5-GGUF for reference.` |
| gguf_target_loader.cpp | 5 | `// unsloth/Qwen3.5-27B-GGUF or ddh0/Qwen3.5-GGUF for reference.` |
| gguf_target_loader.cpp | ... | (24 total matches) |
| internal.h | 33 | `// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──` |
| internal.h | 33 | `// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──` |
| internal.h | 35 | `// Qwen3.5 uses two kinds of blocks interleaved:` |
| internal.h | 203 | `// build_qwen35_graph() call.` |
| internal.h | 423 | `// One entry per delta-net layer (48 for qwen35-27b). Only populated when` |
| internal.h | ... | (8 total matches) |
| kv_quant.cpp | 4 | `// resolution that was previously inlined in qwen35_target_graph.cpp.` |
| kv_quant.cpp | 164 | `// Layer 2: legacy shorthand (last wins, mirrors qwen35_target_graph.cpp:96-108)` |
| quantize_draft_q8.py | 31 | `ARCH                = "qwen35-dflash-draft"` |
| quantize_draft_q8.py | 137 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft-Q8_0")` |
| qwen35_target_graph.cpp | 1 | `// Forward pass of Qwen3.5 (qwen35 hybrid) in pure ggml.` |
| qwen35_target_graph.cpp | 1 | `// Forward pass of Qwen3.5 (qwen35 hybrid) in pure ggml.` |
| qwen35_target_graph.cpp | 3 | `// Translates llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp` into` |
| qwen35_target_graph.cpp | 5 | `// via TargetWeights, supporting any Qwen3.5-dense model size (4B, 9B, 27B).` |
| qwen35_target_graph.cpp | 34 | `// ─── File-local constants (architecture-invariant across all qwen35 sizes) ──` |
| qwen35_target_graph.cpp | ... | (9 total matches) |
| run.py | 8 | `Auto-applies the Qwen3.5/3.6 chat template unless --raw is passed.` |
| safetensors_draft.cpp | 1 | `// Loads z-lab/Qwen3.5-27B-DFlash draft weights from an HF safetensors file` |
| server.py | 59 | `_QWEN35_FAMILY_TOKENIZERS = {` |
| server.py | 60 | `"Qwen3.5-27B": "Qwen/Qwen3.5-27B",` |
| server.py | 60 | `"Qwen3.5-27B": "Qwen/Qwen3.5-27B",` |
| server.py | 66 | `default = "Qwen/Qwen3.5-27B"` |
| server.py | 82 | `for known, repo in _QWEN35_FAMILY_TOKENIZERS.items():` |
| server_tools.py | 1194 | `ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B",` |
| smoke_load_target.cpp | 1 | `// Smoke test for the GGUF target loader. Loads Qwen3.5-27B from a GGUF,` |
| smoke_load_target.cpp | 4 | `// Usage: smoke_load_target <path/to/qwen35.gguf>` |
| smoke_load_target.cpp | 23 | `std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);` |
| smoke_target_forward.cpp | 1 | `// Smoke test for the qwen35 target forward graph.` |
| smoke_target_forward.cpp | 3 | `// Loads Qwen3.5-27B from GGUF, creates a target cache, builds the forward` |
| smoke_target_forward.cpp | 10 | `// Usage: smoke_target_forward <qwen35.gguf>` |
| smoke_target_forward.cpp | 32 | `std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);` |
| smoke_target_forward.cpp | 87 | `QwenGraphOutputs go = build_qwen35_graph(gctx, gf, w, cache, gi);` |
| smoke_target_forward.cpp | ... | (6 total matches) |
| test_dflash.cpp | 4 | `//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).` |
| test_dflash.cpp | 4 | `//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).` |
| test_dflash.cpp | 4 | `//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).` |
| test_dflash.cpp | 769 | `ggml_tensor * layer_out = dflash27b::build_qwen35_layer(` |
| test_dflash.cpp | 863 | `QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);` |
| test_dflash.cpp | ... | (8 total matches) |
| test_generate.cpp | 1 | `// End-to-end generation test for our qwen35 target forward.` |
| test_generate.cpp | 11 | `//   test_generate <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>` |
| test_generate.cpp | 97 | `QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);` |
| test_generate.cpp | 127 | `"usage: %s <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>\n", argv[0]);` |
| tokenize_prompt.py | 2 | `Tokenize a prompt string using the Qwen3.5 HF tokenizer (via transformers)` |
| tokenize_prompt.py | 22 | `ap.add_argument("--model", default="Qwen/Qwen3.5-27B",` |

## Qwen3 References

| File | Line | Context |
|---|---|---|
| _prefill_hook.py | 5 | `in-process Qwen3-0.6B drafter + FlashPrefill scoring (BSA), then emits the` |
| _prefill_hook.py | 11 | `plumbing. The drafter and target use *different* tokenizers (Qwen3-0.6B vs` |
| _prefill_hook.py | 12 | `Qwen3.5/3.6-27B), so the pipeline is:` |
| _prefill_hook.py | 60 | `drafter_gguf: Optional[Path]                       # drafter weights (Qwen3-0.6B BF16 GGUF)` |
| _prefill_hook.py | 89 | `help="Path to the drafter Qwen3-0.6B BF16 GGUF used by "` |
| _prefill_hook.py | ... | (8 total matches) |
| bench_agent_loop.py | 32 | `TARGET        = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| bench_agent_loop.py | 32 | `TARGET        = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| bench_agent_loop.py | 33 | `DRAFT         = Path.home() / "models/qwen3.6-27b-dflash"` |
| bench_he.py | 24 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| bench_he.py | 306 | `default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B"),` |
| bench_he.py | 308 | `"$DFLASH_TOKENIZER, then Qwen/Qwen3.5-27B. Override for "` |
| bench_he.py | 309 | `"Qwen3.6 or other variants, e.g. "` |
| bench_he.py | 310 | `"--target-tokenizer Qwen/Qwen3.6-27B")` |
| bench_llm.py | 7 | `DFLASH_TARGET    path to target Qwen3.6-27B-Q4_K_M.gguf (or 3.5)` |
| bench_llm.py | 11 | `DFLASH_TOKENIZER HF tokenizer repo (default Qwen/Qwen3.5-27B; matches run.py)` |
| bench_llm.py | 25 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| bench_llm.py | 32 | `TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B")` |
| chat.py | 20 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| chat.py | 96 | `tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",` |
| convert_dflash_to_gguf.py | 26 | `qwen3.5-27b-dflash-draft.gguf` |
| convert_dflash_to_gguf.py | 45 | `ARCH                = "qwen35-dflash-draft"` |
| convert_dflash_to_gguf.py | 161 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft")` |
| delta_net_chunked.cpp | 44 | `// GDA only in our port — Qwen3.5 delta-net uses gate scalar per head` |
| detokenize.py | 10 | `ap.add_argument("--model", default="Qwen/Qwen3.5-27B")` |
| dflash27b.h | 2 | `// Qwen3.5-27B with the z-lab/Qwen3.5-27B-DFlash draft model on a single RTX 3090.` |
| dflash27b.h | 2 | `// Qwen3.5-27B with the z-lab/Qwen3.5-27B-DFlash draft model on a single RTX 3090.` |
| dflash27b.h | 24 | `// Qwen3.5-27B qwen35 hybrid uses 24 Q heads, 4 KV heads, 256 head_dim, which` |
| dflash27b.h | 24 | `// Qwen3.5-27B qwen35 hybrid uses 24 Q heads, 4 KV heads, 256 head_dim, which` |
| dflash27b.h | 27 | `// qwen3_dflash_graph.cpp which consume these as draft-side constants.` |
| flashprefill.h | 2 | `// the in-process Qwen3-0.6B drafter (speculative prefill scoring).` |
| gen_oracle.py | 4 | `../../megaqwen3_27b_dflash/reference/.` |
| gen_oracle.py | 29 | `HERE, "..", "..", "megaqwen3_27b_dflash", "reference"))` |
| gguf_draft_loader.cpp | 5 | `// graph builder (qwen3_dflash_graph.cpp) doesn't care about tensor storage` |
| gguf_draft_loader.cpp | 8 | `// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /` |
| gguf_draft_loader.cpp | 135 | `if (std::string(arch) != "qwen35-dflash-draft") {` |
| gguf_draft_loader.cpp | 137 | `" (expected qwen35-dflash-draft)");` |
| gguf_draft_loader.cpp | 144 | `const char * A = "qwen35-dflash-draft";` |
| gguf_target_loader.cpp | 1 | `// Loads Qwen3.5-27B qwen35 hybrid from a GGUF file on disk into a ggml` |
| gguf_target_loader.cpp | 1 | `// Loads Qwen3.5-27B qwen35 hybrid from a GGUF file on disk into a ggml` |
| gguf_target_loader.cpp | 4 | `// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See` |
| gguf_target_loader.cpp | 4 | `// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See` |
| gguf_target_loader.cpp | 5 | `// unsloth/Qwen3.5-27B-GGUF or ddh0/Qwen3.5-GGUF for reference.` |
| gguf_target_loader.cpp | ... | (25 total matches) |
| internal.h | 33 | `// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──` |
| internal.h | 33 | `// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──` |
| internal.h | 35 | `// Qwen3.5 uses two kinds of blocks interleaved:` |
| internal.h | 203 | `// build_qwen35_graph() call.` |
| internal.h | 423 | `// One entry per delta-net layer (48 for qwen35-27b). Only populated when` |
| internal.h | ... | (8 total matches) |
| kv_quant.cpp | 4 | `// resolution that was previously inlined in qwen35_target_graph.cpp.` |
| kv_quant.cpp | 164 | `// Layer 2: legacy shorthand (last wins, mirrors qwen35_target_graph.cpp:96-108)` |
| pflash_daemon.cpp | 3 | `// Loads the Qwen3-0.6B PFlash drafter once, then accepts stdin commands:` |
| pflash_daemon.cpp | 13 | `#include "qwen3_drafter.h"` |
| pflash_daemon.cpp | 71 | `std::fprintf(stderr, "usage: %s <qwen3-0.6b.gguf> [--stream-fd=N]\n", argv[0]);` |
| phase_split_dual_gpu.py | 4 | `This phase-split harness is intentionally PFlash-only. It keeps the Qwen3-0.6B` |
| phase_split_dual_gpu.py | 36 | `DEFAULT_DRAFTER = env_path("PFLASH_PHASE_DRAFTER", ROOT / "models" / "Qwen3-0.6B-BF16.gguf")` |
| phase_split_dual_gpu.py | 37 | `DEFAULT_TOKENIZER = os.environ.get("PFLASH_PHASE_TOKENIZER", "Qwen/Qwen3-0.6B")` |
| quantize_draft_q8.py | 31 | `ARCH                = "qwen35-dflash-draft"` |
| quantize_draft_q8.py | 137 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft-Q8_0")` |
| qwen35_target_graph.cpp | 1 | `// Forward pass of Qwen3.5 (qwen35 hybrid) in pure ggml.` |
| qwen35_target_graph.cpp | 1 | `// Forward pass of Qwen3.5 (qwen35 hybrid) in pure ggml.` |
| qwen35_target_graph.cpp | 3 | `// Translates llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp` into` |
| qwen35_target_graph.cpp | 5 | `// via TargetWeights, supporting any Qwen3.5-dense model size (4B, 9B, 27B).` |
| qwen35_target_graph.cpp | 34 | `// ─── File-local constants (architecture-invariant across all qwen35 sizes) ──` |
| qwen35_target_graph.cpp | ... | (9 total matches) |
| qwen3_0p6b_drafter.h | 1 | `// Custom Qwen3-0.6B drafter forward, in dflash, replacing libllama.` |
| qwen3_0p6b_drafter.h | 8 | `//   bool load_qwen3_0p6b_drafter(path, backend, out)  → load GGUF weights` |
| qwen3_0p6b_drafter.h | 9 | `//   bool forward_qwen3_0p6b_drafter(weights, ids, out_q_capture, out_k_capture)` |
| qwen3_0p6b_drafter.h | 10 | `//   void free_qwen3_0p6b_drafter(weights)` |
| qwen3_0p6b_drafter.h | 30 | `struct Qwen3DrafterLayer {` |
| qwen3_0p6b_drafter.h | ... | (14 total matches) |
| qwen3_0p6b_graph.cpp | 1 | `// Custom forward for the Qwen3-0.6B drafter, replacing libllama.` |
| qwen3_0p6b_graph.cpp | 33 | `#include "qwen3_0p6b_drafter.h"` |
| qwen3_0p6b_graph.cpp | 101 | `bool forward_qwen3_0p6b_drafter(` |
| qwen3_0p6b_graph.cpp | 102 | `const Qwen3DrafterWeights & w,` |
| qwen3_0p6b_graph.cpp | 108 | `set_last_error("forward_qwen3_0p6b_drafter: weights not loaded");` |
| qwen3_0p6b_graph.cpp | ... | (10 total matches) |
| qwen3_0p6b_loader.cpp | 1 | `// GGUF loader for Qwen3-0.6B drafter. Reads weights from a BF16 GGUF file` |
| qwen3_0p6b_loader.cpp | 2 | `// produced by `convert_hf_to_gguf.py Qwen/Qwen3-0.6B`. Sets up ggml tensors` |
| qwen3_0p6b_loader.cpp | 25 | `#include "qwen3_0p6b_drafter.h"` |
| qwen3_0p6b_loader.cpp | 44 | `std::fprintf(stderr, "[qwen3-0.6b] missing tensor: %s\n", name);` |
| qwen3_0p6b_loader.cpp | 68 | `bool load_qwen3_0p6b_drafter(const std::string & path,` |
| qwen3_0p6b_loader.cpp | ... | (19 total matches) |
| qwen3_dflash_graph.cpp | 2 | `// (5-layer non-causal Qwen3-flavored block-diffusion model).` |
| qwen3_dflash_graph.cpp | 15 | `// Semantics match megaqwen3_27b_dflash/reference/dflash_reference.py exactly:` |
| qwen3_drafter.cpp | 1 | `// Qwen3-0.6B drafter for pflash speculative prefill, hosted in-process.` |
| qwen3_drafter.cpp | 4 | `//   - qwen3_0p6b_loader.cpp : mmap GGUF + populate ggml tensors on backend` |
| qwen3_drafter.cpp | 5 | `//   - qwen3_0p6b_graph.cpp  : custom forward (per-layer ggml + FP CUDA kernel)` |
| qwen3_drafter.cpp | 8 | `// Single-pass forward at full S using a custom Qwen3-0.6B graph with the` |
| qwen3_drafter.cpp | 16 | `#include "qwen3_drafter.h"` |
| qwen3_drafter.cpp | ... | (10 total matches) |
| qwen3_drafter.h | 1 | `// In-process Qwen3-0.6B drafter for pflash speculative prefill.` |
| qwen3_drafter.h | 5 | `// subprocess integration. Drafter uses our custom Qwen3-0.6B forward` |
| qwen3_drafter.h | 6 | `// (qwen3_0p6b_graph.cpp + qwen3_0p6b_loader.cpp) which calls our FlashPrefill` |
| qwen3_drafter.h | 6 | `// (qwen3_0p6b_graph.cpp + qwen3_0p6b_loader.cpp) which calls our FlashPrefill` |
| qwen3_drafter.h | 21 | `#include "qwen3_0p6b_drafter.h"` |
| qwen3_drafter.h | ... | (7 total matches) |
| run.py | 8 | `Auto-applies the Qwen3.5/3.6 chat template unless --raw is passed.` |
| run.py | 10 | `Default target is Qwen3.6-27B-Q4_K_M.gguf. Override with `--target` or the` |
| run.py | 12 | `The HF tokenizer repo defaults to `Qwen/Qwen3.6-27B` and can be overridden via` |
| run.py | 27 | `"models/Qwen3.6-27B-Q4_K_M.gguf"),` |
| run.py | 99 | `tok_repo = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")` |
| safetensors_draft.cpp | 1 | `// Loads z-lab/Qwen3.5-27B-DFlash draft weights from an HF safetensors file` |
| server.py | 45 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| server.py | 59 | `_QWEN35_FAMILY_TOKENIZERS = {` |
| server.py | 60 | `"Qwen3.5-27B": "Qwen/Qwen3.5-27B",` |
| server.py | 60 | `"Qwen3.5-27B": "Qwen/Qwen3.5-27B",` |
| server.py | 61 | `"Qwen3.6-27B": "Qwen/Qwen3.6-27B",` |
| server.py | ... | (8 total matches) |
| server_tools.py | 56 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| server_tools.py | 129 | `# Qwen3.6 chat template emits:` |
| server_tools.py | 139 | `# `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`:` |
| server_tools.py | 139 | `# `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`:` |
| server_tools.py | 140 | `#   vllm/reasoning/qwen3_reasoning_parser.py` |
| server_tools.py | ... | (13 total matches) |
| smoke_load_target.cpp | 1 | `// Smoke test for the GGUF target loader. Loads Qwen3.5-27B from a GGUF,` |
| smoke_load_target.cpp | 4 | `// Usage: smoke_load_target <path/to/qwen35.gguf>` |
| smoke_load_target.cpp | 23 | `std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);` |
| smoke_qwen3_0p6b_forward.cpp | 1 | `// Smoke test for the custom Qwen3-0.6B drafter forward path.` |
| smoke_qwen3_0p6b_forward.cpp | 9 | `//   smoke_qwen3_0p6b_forward <gguf_path> <seq_len_or_FILE:path> [keep_ratio]` |
| smoke_qwen3_0p6b_forward.cpp | 11 | `//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf 140000 0.02` |
| smoke_qwen3_0p6b_forward.cpp | 11 | `//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf 140000 0.02` |
| smoke_qwen3_0p6b_forward.cpp | 12 | `//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf FILE:/tmp/niah_32k.bin 0.05` |
| smoke_qwen3_0p6b_forward.cpp | ... | (7 total matches) |
| smoke_target_forward.cpp | 1 | `// Smoke test for the qwen35 target forward graph.` |
| smoke_target_forward.cpp | 3 | `// Loads Qwen3.5-27B from GGUF, creates a target cache, builds the forward` |
| smoke_target_forward.cpp | 10 | `// Usage: smoke_target_forward <qwen35.gguf>` |
| smoke_target_forward.cpp | 32 | `std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);` |
| smoke_target_forward.cpp | 87 | `QwenGraphOutputs go = build_qwen35_graph(gctx, gf, w, cache, gi);` |
| smoke_target_forward.cpp | ... | (6 total matches) |
| test_dflash.cpp | 4 | `//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).` |
| test_dflash.cpp | 4 | `//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).` |
| test_dflash.cpp | 4 | `//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).` |
| test_dflash.cpp | 24 | `#include "qwen3_drafter.h"` |
| test_dflash.cpp | 769 | `ggml_tensor * layer_out = dflash27b::build_qwen35_layer(` |
| test_dflash.cpp | ... | (11 total matches) |
| test_full_compress_cache.py | 17 | `- Qwen3-0.6B-BF16 drafter GGUF` |
| test_full_compress_cache.py | 34 | `TARGET        = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| test_full_compress_cache.py | 34 | `TARGET        = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| test_full_compress_cache.py | 35 | `DRAFT         = Path.home() / "models/qwen3.6-27b-dflash"` |
| test_full_compress_cache.py | 36 | `DRAFTER_GGUF  = Path.home() / "models/Qwen3-0.6B-BF16.gguf"` |
| test_generate.cpp | 1 | `// End-to-end generation test for our qwen35 target forward.` |
| test_generate.cpp | 11 | `//   test_generate <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>` |
| test_generate.cpp | 97 | `QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);` |
| test_generate.cpp | 127 | `"usage: %s <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>\n", argv[0]);` |
| test_multi_turn_prefix_cache.py | 12 | `Prereqs: model files at ~/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf` |
| test_multi_turn_prefix_cache.py | 12 | `Prereqs: model files at ~/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf` |
| test_multi_turn_prefix_cache.py | 13 | `and ~/models/qwen3.6-27b-dflash/model.safetensors. Skipped if missing.` |
| test_multi_turn_prefix_cache.py | 29 | `TARGET = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| test_multi_turn_prefix_cache.py | 29 | `TARGET = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| test_multi_turn_prefix_cache.py | ... | (6 total matches) |
| test_server_prefix_cache.py | 5 | `Prereqs: model files at ~/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf and` |
| test_server_prefix_cache.py | 5 | `Prereqs: model files at ~/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf and` |
| test_server_prefix_cache.py | 6 | `~/models/qwen3.6-27b-dflash/model.safetensors. Skipped if missing.` |
| test_server_prefix_cache.py | 14 | `TARGET = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| test_server_prefix_cache.py | 14 | `TARGET = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"` |
| test_server_prefix_cache.py | ... | (6 total matches) |
| tokenize_prompt.py | 2 | `Tokenize a prompt string using the Qwen3.5 HF tokenizer (via transformers)` |
| tokenize_prompt.py | 22 | `ap.add_argument("--model", default="Qwen/Qwen3.5-27B",` |

## Hardcoded Dimensions

| File | Line | Context |
|---|---|---|
| convert_dflash_to_gguf.py | 47 | `N_LAYER             = 5` |
| convert_dflash_to_gguf.py | 48 | `N_HEAD              = 32          # query heads` |
| convert_dflash_to_gguf.py | 57 | `BLOCK_SIZE          = 16` |
| convert_dflash_to_gguf.py | 169 | `# n_embd_head = n_embd / n_head heuristic (DFlash has n_embd=5120` |
| flashprefill.h | 40 | `int   block_size       = 128;   // K stride; query block size = K block size` |
| internal.h | 92 | `int64_t          n_embd = 0;` |
| internal.h | 120 | `int n_head                  = 24;` |
| internal.h | 122 | `int n_layer                 = 64;` |
| internal.h | 123 | `int n_embd                  = 5120;` |
| quantize_draft_q8.py | 33 | `N_LAYER             = 5` |
| quantize_draft_q8.py | 34 | `N_HEAD              = 32` |
| quantize_draft_q8.py | 43 | `BLOCK_SIZE          = 16` |
| quantize_draft_q8.py | 46 | `Q8_0_BLOCK_SIZE     = 32   # elements per Q8_0 block` |
| qwen3_0p6b_drafter.h | 53 | `std::vector<Qwen3DrafterLayer> layers;  // size = n_layer = 28` |
| qwen3_0p6b_drafter.h | 56 | `int n_layer    = 28;` |
| qwen3_0p6b_drafter.h | 57 | `int n_head     = 16;` |
| qwen3_0p6b_drafter.h | 59 | `int n_embd     = 1024;` |
| spike_thin_copy.cpp | 23 | `constexpr int BLOCK_SIZE  = 256;` |

## Target Layer ID References

| File | Line | Context |
|---|---|---|
| dflash27b.h | 41 | `// target_layer_ids = {1, 16, 31, 46, 61}  (0-indexed into target layers)` |
| gen_oracle.py | 63 | `fc_in_dim = len(cfg.target_layer_ids) * hidden          # 25600` |
| internal.h | 445 | `int                   layer_idx,` |
| qwen35_target_graph.cpp | 1090 | `// layer_idx: which of the 64 layers to build (0-based).` |
| qwen35_target_graph.cpp | 1098 | `int                   layer_idx,` |
| qwen35_target_graph.cpp | 1109 | `const TargetLayer & L = w.layers[layer_idx];` |
| qwen35_target_graph.cpp | 1110 | `const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);` |
| qwen35_target_graph.cpp | 1117 | `for (int il = 0; il < layer_idx; il++) {` |
| qwen35_target_graph.cpp | ... | (9 total matches) |
| test_dflash.cpp | 721 | `int layer_idx,` |
| test_dflash.cpp | 733 | `const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);` |
| test_dflash.cpp | 770 | `sg.ctx, sg.gf, w, cache, layer_idx,` |

## Block Size Assumptions

| File | Line | Context |
|---|---|---|
| convert_dflash_to_gguf.py | 57 | `BLOCK_SIZE          = 16` |
| flashprefill.h | 40 | `int   block_size       = 128;   // K stride; query block size = K block size` |
| quantize_draft_q8.py | 43 | `BLOCK_SIZE          = 16` |
| quantize_draft_q8.py | 46 | `Q8_0_BLOCK_SIZE     = 32   # elements per Q8_0 block` |
| spike_thin_copy.cpp | 23 | `constexpr int BLOCK_SIZE  = 256;` |

## Architecture String References

| File | Line | Context |
|---|---|---|
| _prefill_hook.py | 92 | `ap.add_argument("--prefill-drafter-tokenizer", default="Qwen/Qwen3-0.6B",` |
| bench_he.py | 24 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| bench_he.py | 306 | `default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B"),` |
| bench_he.py | 309 | `"Qwen3.6 or other variants, e.g. "` |
| bench_llm.py | 25 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| bench_llm.py | 32 | `TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B")` |
| chat.py | 20 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| chat.py | 96 | `tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",` |
| convert_dflash_to_gguf.py | 42 | `# DFlash 27B draft architecture constants` |
| convert_dflash_to_gguf.py | 45 | `ARCH                = "qwen35-dflash-draft"` |
| convert_dflash_to_gguf.py | 160 | `# Architecture metadata` |
| convert_dflash_to_gguf.py | 161 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft")` |
| detokenize.py | 10 | `ap.add_argument("--model", default="Qwen/Qwen3.5-27B")` |
| gguf_draft_loader.cpp | 8 | `// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /` |
| gguf_draft_loader.cpp | 128 | `int64_t arch_id = gguf_find_key(gctx, "general.architecture");` |
| gguf_draft_loader.cpp | 130 | `set_last_error("missing general.architecture in draft GGUF");` |
| gguf_draft_loader.cpp | 135 | `if (std::string(arch) != "qwen35-dflash-draft") {` |
| gguf_draft_loader.cpp | 144 | `const char * A = "qwen35-dflash-draft";` |
| gguf_target_loader.cpp | 4 | `// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See` |
| gguf_target_loader.cpp | 4 | `// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See` |
| gguf_target_loader.cpp | 207 | `int64_t arch_id = gguf_find_key(gctx, "general.architecture");` |
| gguf_target_loader.cpp | 209 | `set_last_error("missing general.architecture");` |
| gguf_target_loader.cpp | 214 | `if (std::string(arch) != "qwen35") {` |
| gguf_target_loader.cpp | ... | (20 total matches) |
| internal.h | 178 | `// Architecture metadata (populated by loader).` |
| pflash_daemon.cpp | 13 | `#include "qwen3_drafter.h"` |
| phase_split_dual_gpu.py | 36 | `DEFAULT_DRAFTER = env_path("PFLASH_PHASE_DRAFTER", ROOT / "models" / "Qwen3-0.6B-BF16.gguf")` |
| phase_split_dual_gpu.py | 37 | `DEFAULT_TOKENIZER = os.environ.get("PFLASH_PHASE_TOKENIZER", "Qwen/Qwen3-0.6B")` |
| quantize_draft_q8.py | 28 | `# DFlash 27B draft architecture constants (must match dflash27b.h)` |
| quantize_draft_q8.py | 31 | `ARCH                = "qwen35-dflash-draft"` |
| quantize_draft_q8.py | 136 | `# Architecture metadata (identical to convert_dflash_to_gguf.py)` |
| quantize_draft_q8.py | 137 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft-Q8_0")` |
| qwen35_target_graph.cpp | 8 | `// Architecture highlights:` |
| qwen35_target_graph.cpp | 34 | `// ─── File-local constants (architecture-invariant across all qwen35 sizes) ──` |
| qwen3_0p6b_drafter.h | 55 | `// Architecture metadata.` |
| qwen3_0p6b_graph.cpp | 33 | `#include "qwen3_0p6b_drafter.h"` |
| qwen3_0p6b_loader.cpp | 25 | `#include "qwen3_0p6b_drafter.h"` |
| qwen3_0p6b_loader.cpp | 80 | `out.n_embd     = (int)get_u32(gctx, "qwen3.embedding_length", 1024);` |
| qwen3_0p6b_loader.cpp | 81 | `out.n_ff       = (int)get_u32(gctx, "qwen3.feed_forward_length", 3072);` |
| qwen3_0p6b_loader.cpp | 82 | `out.n_head     = (int)get_u32(gctx, "qwen3.attention.head_count", 16);` |
| qwen3_0p6b_loader.cpp | 83 | `out.n_head_kv  = (int)get_u32(gctx, "qwen3.attention.head_count_kv", 8);` |
| qwen3_0p6b_loader.cpp | ... | (9 total matches) |
| qwen3_drafter.cpp | 16 | `#include "qwen3_drafter.h"` |
| qwen3_drafter.cpp | 17 | `#include "qwen3_0p6b_drafter.h"` |
| qwen3_drafter.h | 21 | `#include "qwen3_0p6b_drafter.h"` |
| run.py | 99 | `tok_repo = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")` |
| safetensors_draft.cpp | 338 | `// The CMAKE_CUDA_ARCHITECTURES list tells us the minimum supported arch.` |
| server.py | 45 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| server.py | 60 | `"Qwen3.5-27B": "Qwen/Qwen3.5-27B",` |
| server.py | 60 | `"Qwen3.5-27B": "Qwen/Qwen3.5-27B",` |
| server.py | 61 | `"Qwen3.6-27B": "Qwen/Qwen3.6-27B",` |
| server.py | 61 | `"Qwen3.6-27B": "Qwen/Qwen3.6-27B",` |
| server.py | ... | (6 total matches) |
| server_tools.py | 56 | `str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),` |
| server_tools.py | 1194 | `ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B",` |
| smoke_qwen3_0p6b_forward.cpp | 16 | `#include "qwen3_drafter.h"` |
| test_dflash.cpp | 24 | `#include "qwen3_drafter.h"` |
| tokenize_prompt.py | 22 | `ap.add_argument("--model", default="Qwen/Qwen3.5-27B",` |

## Mask Token References

| File | Line | Context |
|---|---|---|
| convert_dflash_to_gguf.py | 56 | `MASK_TOKEN_ID       = 248070` |
| convert_dflash_to_gguf.py | 180 | `writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)` |
| convert_dflash_to_gguf.py | 180 | `writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)` |
| dflash27b.h | 39 | `#define DFLASH27B_DRAFT_MASK_TOKEN_ID  248070` |
| quantize_draft_q8.py | 42 | `MASK_TOKEN_ID       = 248070` |
| quantize_draft_q8.py | 154 | `writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)` |
| quantize_draft_q8.py | 154 | `writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)` |
| test_dflash.cpp | 1542 | `const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;` |

## Draft Model References

| File | Line | Context |
|---|---|---|
| convert_dflash_to_gguf.py | 26 | `qwen3.5-27b-dflash-draft.gguf` |
| convert_dflash_to_gguf.py | 45 | `ARCH                = "qwen35-dflash-draft"` |
| convert_dflash_to_gguf.py | 161 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft")` |
| gguf_draft_loader.cpp | 8 | `// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /` |
| gguf_draft_loader.cpp | 135 | `if (std::string(arch) != "qwen35-dflash-draft") {` |
| gguf_draft_loader.cpp | 137 | `" (expected qwen35-dflash-draft)");` |
| gguf_draft_loader.cpp | 144 | `const char * A = "qwen35-dflash-draft";` |
| quantize_draft_q8.py | 31 | `ARCH                = "qwen35-dflash-draft"` |
| quantize_draft_q8.py | 137 | `writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft-Q8_0")` |

## Summary

Key areas requiring generalization:

1. **Architecture dispatch**: qwen35_target_graph.cpp and qwen3_dflash_graph.cpp are qwen35-specific.
2. **Draft graph**: qwen3_dflash_graph.cpp hardcodes qwen3 attention/QKV shapes.
3. **Target loader**: gguf_target_loader.cpp has qwen35 metadata assumptions.
4. **Draft loader**: gguf_draft_loader.cpp uses "qwen35-dflash-draft" arch string.
5. **Block size**: Compile-time constants in many files.
6. **Target hidden states**: Hardcoded to 5 layers fused into [5*hidden, ctx_len] layout.
7. **SSM/DeltaNet ops**: Tree-mode ops are qwen35 hybrid-specific.
