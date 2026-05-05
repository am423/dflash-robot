#include "dflash_robot/adapters/qwen35_adapter.h"
#include <algorithm>
#include <cmath>

namespace dflash_robot {

Qwen35Adapter::Qwen35Adapter(int n_layers, int hidden_size, int vocab_size)
    : n_layers_(n_layers), hidden_size_(hidden_size), vocab_size_(vocab_size) {}

ModelCapabilities Qwen35Adapter::capabilities() const {
    return ModelCapabilities{
        "qwen35",
        n_layers_,
        hidden_size_,
        vocab_size_,
        true,   // supports_hidden_capture
        true,   // supports_cache_rollback
        true    // has_ssm_state
    };
}

LayerSelection Qwen35Adapter::default_layer_selection(int n_features) const {
    // Uniformly-spaced layers from [1, n_layers-3], matching z-lab's build_target_layer_ids.
    // For n_layers=64: range is [1, 61], step = 60/(n_features-1) = 15
    // Layers: {1, 16, 31, 46, 61}

    LayerSelection sel;
    sel.hidden_size = hidden_size_;
    sel.fused_size = hidden_size_ * n_features;

    if (n_features <= 0) {
        return sel;
    }

    if (n_features == 1) {
        sel.layer_ids.push_back(1);
        return sel;
    }

    int lo = 1;
    int hi = n_layers_ - 3;

    sel.layer_ids.reserve(n_features);
    for (int i = 0; i < n_features; ++i) {
        // Linearly interpolate between lo and hi
        int lid = lo + static_cast<int>(std::round(
            static_cast<double>(i) * (hi - lo) / (n_features - 1)));
        sel.layer_ids.push_back(lid);
    }

    return sel;
}

}
