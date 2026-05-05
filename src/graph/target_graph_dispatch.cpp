// Target graph-module dispatcher.
//
// Executables and tools should call build_target_graph() instead of directly
// calling build_qwen35_graph(). This keeps qwen35moe/Qwen3.6-35B-A3B on its
// registered TargetGraphModule path and makes future GGUF architecture modules
// usable without patching every benchmark/test harness.

#include "internal.h"
#include "graph/model_graph_registry.h"
#include "graph/target_graph_module.h"

namespace dflash27b {

QwenGraphOutputs build_target_graph(ggml_context * ctx,
                                    ggml_cgraph * gf,
                                    const TargetWeights & w,
                                    TargetCache & cache,
                                    const QwenGraphInputs & in) {
    std::string arch = w.arch;
    if (arch.empty()) {
        // Backward-compatible fallback for older in-memory TargetWeights.
        arch = w.is_moe ? "qwen35moe" : "qwen35";
    }

    TargetGraphModule * module = get_graph_module(arch);
    if (!module) {
        set_last_error("no target graph module registered for arch '" + arch + "'");
        return {};
    }
    return module->build_graph(ctx, gf, w, cache, in);
}

}  // namespace dflash27b
