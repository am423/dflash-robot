#pragma once
#include <string>
#include <vector>

namespace dflash_robot {

struct LayerSelection {
    std::vector<int> layer_ids;
    int hidden_size;
    int fused_size;
};

struct ModelCapabilities {
    std::string arch;
    int n_layers;
    int hidden_size;
    int vocab_size;
    bool supports_hidden_capture;
    bool supports_cache_rollback;
    bool has_ssm_state;
};

class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;
    virtual ModelCapabilities capabilities() const = 0;
    virtual LayerSelection default_layer_selection(int n_features) const = 0;
};

}
