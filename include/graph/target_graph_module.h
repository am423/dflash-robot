// Target graph module interface.
// Each GGUF architecture family gets one module implementing this interface.
// The module owns metadata parsing, tensor binding, cache layout, graph
// construction, hidden-state capture, and rollback/crop semantics.
//
// Not installed, not exposed in the public API.

#pragma once

#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

// Forward from internal.h
namespace dflash27b {
struct TargetWeights;
struct TargetCache;
struct QwenGraphInputs;
struct QwenGraphOutputs;
}

namespace dflash27b {

// Machine-readable capabilities for a target graph module.
struct TargetGraphModuleCapabilities {
    std::string              name;                  // "qwen35_dense", "qwen35moe", etc.
    std::vector<std::string> gguf_arches;           // e.g. ["qwen35"]
    bool                     supports_kv_cache      = false;
    bool                     supports_ssm_cache     = false;
    bool                     supports_hidden_capture = false;
    bool                     supports_rollback      = false;
    bool                     supports_tree_verify   = false;
    std::vector<std::string> known_limitations;     // e.g. "batch_size_1_only"
};

// Abstract interface for a target model graph module.
// One module per GGUF architecture family.
class TargetGraphModule {
public:
    virtual ~TargetGraphModule() = default;

    // Return capabilities descriptor.
    virtual TargetGraphModuleCapabilities capabilities() const = 0;

    // Parse GGUF metadata keys into TargetWeights. Fills architecture-agnostic
    // fields (n_layer, n_embd, etc.) and architecture-specific fields
    // (ssm params, MoE params). Returns false + err on failure.
    virtual bool bind_metadata(const gguf_context * gctx,
                               TargetWeights & out,
                               std::string & err) const = 0;

    // Wire layer tensor pointers in meta_ctx to TargetWeights::layers.
    // Validates that required tensors are present for every layer.
    virtual bool bind_tensors(ggml_context * meta_ctx,
                              TargetWeights & out,
                              std::string & err) const = 0;

    // Allocate target cache: KV, SSM, conv, rollback, target_feat.
    virtual bool create_cache(const TargetWeights & w,
                              int max_ctx,
                              int max_verify_tokens,
                              ggml_backend_t backend,
                              TargetCache & out,
                              bool prefill_only) const = 0;

    // Build a full-layer forward compute graph.
    // Called once per decode step or verify step.
    virtual QwenGraphOutputs build_graph(ggml_context * ctx,
                                         ggml_cgraph * gf,
                                         const TargetWeights & w,
                                         TargetCache & cache,
                                         const QwenGraphInputs & in) const = 0;
};

}  // namespace dflash27b
