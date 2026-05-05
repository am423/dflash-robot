// r0b0tlab trace generator: dump target hidden states + next tokens to binary files.
//
// Build: cmake --build build --target trace_dump
// Usage: ./build/trace_dump <target.gguf> <prompt_ids.bin> <output_trace.pt> [--max-tokens=N]
//
// prompt_ids.bin: raw int32 token IDs (use tokenize_prompt.py to generate)
// output_trace.pt: torch .pt file readable by training/dataset.py
//
// This avoids downloading the 45GB HF model. We use the GGUF we already have.

#include "internal.h"
#include "dflash_graph.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <sys/stat.h>

using namespace dflash27b;

// Helper: create directory
static bool mkdir_p(const char * path) {
    std::string cmd = std::string("mkdir -p ") + path;
    return system(cmd.c_str()) == 0;
}

// Read binary file of int32 token IDs
static std::vector<int32_t> read_prompt_bin(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return {}; }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int32_t> ids(sz / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(ids.data()), sz);
    return ids;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: trace_dump <target.gguf> <prompt_ids.bin> <output.pt> [--max-tokens=N]\n");
        return 2;
    }
    const char * target_path = argv[1];
    const char * prompt_path = argv[2];
    const char * output_path = argv[3];
    int max_tokens = 128;
    for (int i = 4; i < argc; i++) {
        if (std::strncmp(argv[i], "--max-tokens=", 13) == 0) {
            max_tokens = std::atoi(argv[i] + 13);
        }
    }

    // Init CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    // Load target
    TargetWeights w;
    if (!load_target_gguf(target_path, backend, w)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return 1;
    }

    // Read prompt token IDs
    std::vector<int32_t> prompt_ids = read_prompt_bin(prompt_path);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "empty or missing prompt\n");
        return 1;
    }
    int n_prompt = (int)prompt_ids.size();

    const int hidden = w.n_embd;               // 2048
    const int n_capture = 5;                   // number of capture layers
    const int feat_dim = n_capture * hidden;   // 5 * 2048 = 10240
    const int max_ctx = std::max(4096, n_prompt + max_tokens + 64);

    // Create cache (prefill only, we don't need rollback buffers)
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/16, backend, cache,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    // Allocate embedding buffer (CPU)
    std::vector<float> embed_buf(hidden);

    // Allocate target_feat host buffer for reading from GPU
    std::vector<uint16_t> feat_host;  // bf16 on GPU
    size_t feat_host_bytes = 0;

    // Output buffers: we save hidden states + next tokens
    std::vector<uint16_t> all_features;  // bf16, concatenated
    std::vector<int32_t> all_next_tokens;

    // ── Token-segmented prefill (generate first token) ───────────────
    {
        StepGraph sg;
        int kv_start = 0;
        for (int i = 0; i < n_prompt; i++) {
            const int chunk_end = std::min(i + 1, n_prompt);
            const int n_chunk = chunk_end - i;

            // Embed
            if (!w.embedder.embed(&prompt_ids[i], n_chunk, embed_buf.data())) return 1;

            // Build graph step
            if (!build_target_step(sg, w, cache, backend, kv_start, n_chunk,
                                   /*with_mask=*/false, /*capture=*/false,
                                   /*capture_delta_intermediate=*/false, 0)) {
                std::fprintf(stderr, "prefill build %d failed\n", i);
                return 1;
            }

            // Set input
            ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                    sizeof(float) * hidden * n_chunk);

            std::vector<int32_t> pos4(4 * n_chunk);
            for (int j = 0; j < n_chunk; j++) {
                int p = kv_start + j;
                pos4[0 * n_chunk + j] = p;
                pos4[1 * n_chunk + j] = p;
                pos4[2 * n_chunk + j] = p;
                pos4[3 * n_chunk + j] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4.data(), 0,
                                    sizeof(int32_t) * 4 * n_chunk);

            // Compute
            auto st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) return 1;

            kv_start += n_chunk;
        }

        // After prefill, kv_start == n_prompt
    }

    // Generate first token
    int last_tok = -1;
    {
        StepGraph sg;
        const int kv_start = n_prompt;

        if (!build_target_step(sg, w, cache, backend, kv_start, /*n_tokens=*/1,
                               /*with_mask=*/false, /*capture=*/true)) {
            std::fprintf(stderr, "first token build failed\n");
            return 1;
        }

        // Embed the last prompt token to get first output
        if (!w.embedder.embed(&prompt_ids.back(), 1, embed_buf.data())) return 1;
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0, sizeof(float) * hidden);

        std::vector<int32_t> pos4(4);
        for (int d = 0; d < 4; d++) pos4[d] = kv_start;
        ggml_backend_tensor_set(sg.positions, pos4.data(), 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) return 1;

        // Read argmax token
        ggml_backend_tensor_get(sg.argmax_tokens, &last_tok, 0, sizeof(int32_t));

        std::printf("[step %d/%d] token=%d kv_start=%d\n", 0, max_tokens, last_tok, kv_start);
    }

    // ── Decode loop: autoregressive generation with hidden state capture ─
    cache.cur_pos = n_prompt;
    resize_prefill_cache(cache);  // promote to full decode cache

    for (int step = 0; step < max_tokens && last_tok >= 0; step++) {
        StepGraph sg;
        const int kv_start = n_prompt + step;

        if (!build_target_step(sg, w, cache, backend, kv_start, /*n_tokens=*/1,
                               /*with_mask=*/false, /*capture=*/true)) {
            std::fprintf(stderr, "decode step %d build failed\n", step);
            break;
        }

        // Embed
        int32_t tok = last_tok;
        if (!w.embedder.embed(&tok, 1, embed_buf.data())) return 1;
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0, sizeof(float) * hidden);

        std::vector<int32_t> pos4(4);
        for (int d = 0; d < 4; d++) pos4[d] = kv_start;
        ggml_backend_tensor_set(sg.positions, pos4.data(), 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) return 1;

        // Read next token
        int32_t next_tok;
        ggml_backend_tensor_get(sg.argmax_tokens, &next_tok, 0, sizeof(int32_t));

        // Read captured hidden states from target_feat ring buffer
        // cache.target_feat: [N*hidden, cap] bf16
        // The current position's features are at slot: kv_start % target_feat_cap
        if (cache.target_feat) {
            const int slot = kv_start % cache.target_feat_cap;
            const size_t col_bytes = (size_t)feat_dim * sizeof(uint16_t);
            const size_t offset = (size_t)slot * cache.target_feat->nb[1];

            // Ensure host buffer is sized
            size_t needed = (size_t)feat_dim;
            if (feat_host.size() < needed) {
                feat_host.resize(needed);
            }

            // Copy from GPU to host (bf16 → host uint16)
            ggml_backend_tensor_get(cache.target_feat, feat_host.data(), offset, col_bytes);

            // Append to output
            all_features.insert(all_features.end(), feat_host.begin(), feat_host.begin() + feat_dim);
            all_next_tokens.push_back(next_tok);
        }

        last_tok = next_tok;
        cache.cur_pos = kv_start + 1;

        if ((step + 1) % 10 == 0 || step < 3) {
            std::printf("[step %d/%d] next=%d pos=%d\n", step + 1, max_tokens, next_tok, cache.cur_pos);
        }

        // Check EOS
        if (next_tok == w.eos_id || (w.eos_chat_id >= 0 && next_tok == w.eos_chat_id)) {
            std::printf("[eos] stopping at step %d\n", step);
            break;
        }
    }

    // ── Write output trace file ──────────────────────────────────────
    int n_steps = (int)all_next_tokens.size();
    std::printf("[trace] %d positions captured\n", n_steps);

    // Write as a simple binary format that dataset.py can read
    // Format: [int32 n_steps] [feat_dim * n_steps uint16] [n_steps int32]
    {
        FILE * f = std::fopen(output_path, "wb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", output_path); return 1; }
        std::fwrite(&n_steps, sizeof(int32_t), 1, f);
        std::fwrite(all_features.data(), sizeof(uint16_t), all_features.size(), f);
        std::fwrite(all_next_tokens.data(), sizeof(int32_t), all_next_tokens.size(), f);
        std::fclose(f);
    }

    std::printf("[trace] written to %s (%zu bytes)\n", output_path,
                sizeof(int32_t) + all_features.size() * sizeof(uint16_t) + all_next_tokens.size() * sizeof(int32_t));

    free_target_cache(cache);
    free_target_weights(w);
    return 0;
}
