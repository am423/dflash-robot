#include "dflash_robot/compatibility_registry.h"
#include <algorithm>

namespace dflash_robot {

CompatibilityResult check_compatibility(const ModelCapabilities& target, const DraftCapabilities& draft) {
    // qwen35 + qwen3_dflash = supported
    if ((target.arch == "qwen35" || target.arch == "qwen35moe") &&
        draft.family == "qwen3_dflash") {
        return {true, "supported", ""};
    }

    // draft exists but target.arch not in draft.compatible_target_arches
    auto it = std::find(draft.compatible_target_arches.begin(),
                        draft.compatible_target_arches.end(),
                        target.arch);
    if (it == draft.compatible_target_arches.end()) {
        return {false, "incompatible_arch",
                "Draft family '" + draft.family + "' does not support target arch '" + target.arch + "'"};
    }

    return {false, "unsupported_architecture",
            "No known adapter for target arch '" + target.arch + "' with draft family '" + draft.family + "'"};
}

CompatibilityResult classify_model(const ModelCapabilities& target, const DraftCapabilities* draft_or_null) {
    if (draft_or_null) {
        return check_compatibility(target, *draft_or_null);
    }

    // No draft provided
    if (target.arch == "qwen35" || target.arch == "qwen35moe") {
        return {false, "adapter_exists_draft_missing",
                "No DFlash draft found for " + target.arch + " target"};
    }

    return {false, "unsupported_architecture",
            "No known adapter for target arch '" + target.arch + "'"};
}

}
