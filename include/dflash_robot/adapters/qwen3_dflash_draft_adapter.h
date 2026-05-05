#pragma once
#include <string>
#include <vector>
#include "dflash_robot/draft_adapter.h"

namespace dflash_robot {

// DraftAdapter for the z-lab Qwen3.5-27B-DFlash draft model.
class Qwen3DFlashDraftAdapter : public DraftAdapter {
public:
    DraftCapabilities capabilities() const override;
    CompatibilityResult validate(const ModelCapabilities& target) const override;
};

}
