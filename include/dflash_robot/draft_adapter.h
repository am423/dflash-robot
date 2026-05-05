#pragma once
#include <string>
#include <vector>
#include "dflash_robot/model_adapter.h"

namespace dflash_robot {

struct DraftCapabilities {
    std::string family;
    int training_block_size;
    int max_inference_block_size;
    int n_layers;
    int hidden_size;
    int required_target_features;
    std::vector<std::string> compatible_target_arches;
};

struct CompatibilityResult {
    bool ok;
    std::string status;
    std::string reason;
};

class DraftAdapter {
public:
    virtual ~DraftAdapter() = default;
    virtual DraftCapabilities capabilities() const = 0;
    virtual CompatibilityResult validate(const ModelCapabilities& target) const = 0;
};

}
