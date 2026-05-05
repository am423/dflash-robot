#include "dflash_robot/adapters/qwen3_dflash_draft_adapter.h"
#include <algorithm>

namespace dflash_robot {

DraftCapabilities Qwen3DFlashDraftAdapter::capabilities() const {
    return DraftCapabilities{
        "qwen3_dflash",                      // family
        16,                                  // training_block_size
        16,                                  // max_inference_block_size
        5,                                   // n_layers
        5120,                                // hidden_size
        5,                                   // required_target_features
        {"qwen35"}                           // compatible_target_arches
    };
}

CompatibilityResult Qwen3DFlashDraftAdapter::validate(const ModelCapabilities& target) const {
    DraftCapabilities caps = capabilities();

    auto it = std::find(caps.compatible_target_arches.begin(),
                        caps.compatible_target_arches.end(),
                        target.arch);

    if (it == caps.compatible_target_arches.end()) {
        return {false, "incompatible_arch",
                "Draft family '" + caps.family + "' does not support target arch '" + target.arch + "'"};
    }

    return {true, "supported", ""};
}

}
