#pragma once
#include <string>
#include "dflash_robot/model_adapter.h"
#include "dflash_robot/draft_adapter.h"

namespace dflash_robot {

CompatibilityResult check_compatibility(const ModelCapabilities& target, const DraftCapabilities& draft);
CompatibilityResult classify_model(const ModelCapabilities& target, const DraftCapabilities* draft_or_null);

}
