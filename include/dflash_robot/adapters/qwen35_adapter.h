#pragma once
#include <string>
#include <vector>
#include "dflash_robot/model_adapter.h"

namespace dflash_robot {

// ModelAdapter for Qwen3.5 (qwen35 hybrid architecture).
// Default n_layers = 64 (Qwen3.5-27B).
class Qwen35Adapter : public ModelAdapter {
public:
    explicit Qwen35Adapter(int n_layers = 64, int hidden_size = 5120, int vocab_size = 248320);

    ModelCapabilities capabilities() const override;
    LayerSelection default_layer_selection(int n_features) const override;

private:
    int n_layers_;
    int hidden_size_;
    int vocab_size_;
};

}
