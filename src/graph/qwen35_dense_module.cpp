// Qwen35 dense/hybrid graph module implementation.
// Delegates to existing internal functions for the dense architecture.

#include "graph/qwen35_dense_module.h"
#include "internal.h"

namespace dflash27b {

TargetGraphModuleCapabilities Qwen35DenseModule::capabilities() const {
    return {
        "qwen35_dense",
        {"qwen35"},
        true,   // supports_kv_cache
        true,   // supports_ssm_cache
        true,   // supports_hidden_capture
        true,   // supports_rollback
        true,   // supports_tree_verify
        {}      // known_limitations
    };
}

bool Qwen35DenseModule::bind_metadata(const gguf_context * gctx,
                                       TargetWeights & out,
                                       std::string & err) const {
    // Validate architecture
    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id < 0) {
        err = "missing general.architecture";
        return false;
    }
    const char * arch = gguf_get_val_str(gctx, arch_id);
    if (std::string(arch) != "qwen35") {
        err = std::string("expected arch qwen35, got ") + arch;
        return false;
    }
    // Full metadata binding is done by load_target_gguf.
    // Module validates that is_moe is false.
    out.is_moe = false;
    return true;
}

bool Qwen35DenseModule::bind_tensors(ggml_context * meta_ctx,
                                      TargetWeights & out,
                                      std::string & err) const {
    // Tensor binding is done by load_target_gguf.
    // Module validates that dense FFN tensors exist.
    for (int il = 0; il < out.n_layer; il++) {
        const TargetLayer & L = out.layers[il];
        if (!L.w_gate || !L.w_up || !L.w_down) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "layer %d: missing dense FFN tensors (qwen35)", il);
            err = buf;
            return false;
        }
    }
    return true;
}

bool Qwen35DenseModule::create_cache(const TargetWeights & w,
                                      int max_ctx,
                                      int max_verify_tokens,
                                      ggml_backend_t backend,
                                      TargetCache & out,
                                      bool prefill_only) const {
    return dflash27b::create_target_cache(w, max_ctx, max_verify_tokens,
                                          backend, out, prefill_only);
}

QwenGraphOutputs Qwen35DenseModule::build_graph(ggml_context * ctx,
                                                 ggml_cgraph * gf,
                                                 const TargetWeights & w,
                                                 TargetCache & cache,
                                                 const QwenGraphInputs & in) const {
    return dflash27b::build_qwen35_graph(ctx, gf, w, cache, in);
}

}  // namespace dflash27b
