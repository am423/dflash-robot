#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include "dflash_robot/adapters/qwen3_dflash_draft_adapter.h"

using namespace dflash_robot;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

static void test_capabilities() {
    Qwen3DFlashDraftAdapter adapter;
    DraftCapabilities caps = adapter.capabilities();

    CHECK(caps.family == "qwen3_dflash", "family should be qwen3_dflash");
    CHECK(caps.training_block_size == 16, "training_block_size=16");
    CHECK(caps.max_inference_block_size == 16, "max_inference_block_size=16");
    CHECK(caps.n_layers == 5, "n_layers=5");
    CHECK(caps.required_target_features == 5, "required_target_features=5");
    CHECK(caps.compatible_target_arches.size() == 2, "two compatible arches");
    CHECK(caps.compatible_target_arches[0] == "qwen35", "compatible with qwen35");
    CHECK(caps.compatible_target_arches[1] == "qwen35moe", "compatible with qwen35moe");
}

static void test_validate_qwen35() {
    Qwen3DFlashDraftAdapter adapter;
    ModelCapabilities target{"qwen35", 64, 5120, 248320, true, true, true};

    CompatibilityResult result = adapter.validate(target);
    CHECK(result.ok == true, "qwen35 target should be ok");
    CHECK(result.status == "supported", "status should be supported");
    CHECK(result.reason.empty(), "reason should be empty");
}

static void test_validate_llama() {
    Qwen3DFlashDraftAdapter adapter;
    ModelCapabilities target{"llama", 32, 4096, 32000, false, false, false};

    CompatibilityResult result = adapter.validate(target);
    CHECK(result.ok == false, "llama target should fail");
    CHECK(result.status == "incompatible_arch", "status should be incompatible_arch");
    CHECK(result.reason.find("llama") != std::string::npos, "reason should mention llama");
}

static void test_validate_gemma() {
    Qwen3DFlashDraftAdapter adapter;
    ModelCapabilities target{"gemma", 28, 3584, 256000, false, false, false};

    CompatibilityResult result = adapter.validate(target);
    CHECK(result.ok == false, "gemma target should fail");
    CHECK(result.status == "incompatible_arch", "status should be incompatible_arch");
}

int main() {
    test_capabilities();
    test_validate_qwen35();
    test_validate_llama();
    test_validate_gemma();

    printf("test_draft_adapter: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
