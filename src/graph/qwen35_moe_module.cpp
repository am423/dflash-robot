// Qwen35 MoE graph module implementation (qwen35moe architecture).
//
// This module validates qwen35moe metadata/tensors and delegates graph
// construction to the shared qwen35 builder. The shared builder selects the
// MoE FFN path when TargetWeights::is_moe / layer MoE tensors are present.
// Current performance limitation is ggml-cuda MoE verify cost, not missing MoE
// graph semantics.

#include "graph/qwen35_moe_module.h"
#include "internal.h"

namespace dflash27b {

TargetGraphModuleCapabilities Qwen35MoeModule::capabilities() const {
    return {
        "qwen35moe",
        {"qwen35moe"},
        true,   // supports_kv_cache
        true,   // supports_ssm_cache
        true,   // supports_hidden_capture
        true,   // supports_rollback
        true,   // supports_tree_verify
        {"moe_verify_compute_bound_without_fused_cuda_moe",
         "performance_not_expected_to_exceed_ar_on_rtx3090"}  // known_limitations
    };
}

bool Qwen35MoeModule::bind_metadata(const gguf_context * gctx,
                                     TargetWeights & out,
                                     std::string & err) const {
    // Validate architecture
    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id < 0) {
        err = "missing general.architecture";
        return false;
    }
    const char * arch = gguf_get_val_str(gctx, arch_id);
    if (std::string(arch) != "qwen35moe") {
        err = std::string("expected arch qwen35moe, got ") + arch;
        return false;
    }
    // Validate MoE metadata
    if (out.expert_count == 0 || out.expert_used_count == 0) {
        err = "MoE model missing expert metadata";
        return false;
    }
    return true;
}

bool Qwen35MoeModule::bind_tensors(ggml_context * meta_ctx,
                                    TargetWeights & out,
                                    std::string & err) const {
    // Validate MoE FFN tensors exist for every layer
    for (int il = 0; il < out.n_layer; il++) {
        const TargetLayer & L = out.layers[il];
        if (!L.ffn_gate_inp || !L.ffn_gate_exps ||
            !L.ffn_up_exps   || !L.ffn_down_exps) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "layer %d: missing MoE FFN tensors (qwen35moe)", il);
            err = buf;
            return false;
        }
    }
    return true;
}

bool Qwen35MoeModule::create_cache(const TargetWeights & w,
                                    int max_ctx,
                                    int max_verify_tokens,
                                    ggml_backend_t backend,
                                    TargetCache & out,
                                    bool prefill_only) const {
    return dflash27b::create_target_cache(w, max_ctx, max_verify_tokens,
                                          backend, out, prefill_only);
}

QwenGraphOutputs Qwen35MoeModule::build_graph(ggml_context * ctx,
                                               ggml_cgraph * gf,
                                               const TargetWeights & w,
                                               TargetCache & cache,
                                               const QwenGraphInputs & in) const {
    return dflash27b::build_qwen35_graph(ctx, gf, w, cache, in);
}

}  // namespace dflash27b
