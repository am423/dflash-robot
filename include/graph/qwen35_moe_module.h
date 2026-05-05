// Qwen35 MoE graph module (qwen35moe architecture).
// Handles Qwen3.6 MoE models like Qwen3.6-35B-A3B.
// Same layer structure as qwen35 dense but uses MoE FFN with
// expert routing instead of dense SwiGLU FFN.

#pragma once

#include "graph/target_graph_module.h"

namespace dflash27b {

class Qwen35MoeModule : public TargetGraphModule {
public:
    TargetGraphModuleCapabilities capabilities() const override;
    bool bind_metadata(const gguf_context * gctx,
                       TargetWeights & out,
                       std::string & err) const override;
    bool bind_tensors(ggml_context * meta_ctx,
                      TargetWeights & out,
                      std::string & err) const override;
    bool create_cache(const TargetWeights & w,
                      int max_ctx,
                      int max_verify_tokens,
                      ggml_backend_t backend,
                      TargetCache & out,
                      bool prefill_only) const override;
    QwenGraphOutputs build_graph(ggml_context * ctx,
                                 ggml_cgraph * gf,
                                 const TargetWeights & w,
                                 TargetCache & cache,
                                 const QwenGraphInputs & in) const override;
};

}  // namespace dflash27b
