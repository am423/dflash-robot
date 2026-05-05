#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include "dflash_robot/compatibility_registry.h"
#include "dflash_robot/adapters/qwen35_adapter.h"
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

static DraftCapabilities make_qwen3_dflash_caps() {
    Qwen3DFlashDraftAdapter adapter;
    return adapter.capabilities();
}

static ModelCapabilities make_qwen35_target() {
    return {"qwen35", 64, 5120, 248320, true, true, true};
}

static ModelCapabilities make_gemma_target() {
    return {"gemma", 28, 3584, 256000, false, false, false};
}

// --- check_compatibility ---

static void test_qwen35_with_dflash_draft() {
    ModelCapabilities target = make_qwen35_target();
    DraftCapabilities draft = make_qwen3_dflash_caps();

    CompatibilityResult result = check_compatibility(target, draft);
    CHECK(result.ok == true, "qwen35 + qwen3_dflash should be supported");
    CHECK(result.status == "supported", "status should be supported");
    CHECK(result.reason.empty(), "reason should be empty");
}

static void test_gemma_with_dflash_draft() {
    ModelCapabilities target = make_gemma_target();
    DraftCapabilities draft = make_qwen3_dflash_caps();

    CompatibilityResult result = check_compatibility(target, draft);
    CHECK(result.ok == false, "gemma + qwen3_dflash should fail");
    CHECK(result.status == "incompatible_arch", "status should be incompatible_arch");
    CHECK(result.reason.find("gemma") != std::string::npos, "reason mentions gemma");
}

static void test_llama_with_dflash_draft() {
    ModelCapabilities target{"llama", 32, 4096, 32000, false, false, false};
    DraftCapabilities draft = make_qwen3_dflash_caps();

    CompatibilityResult result = check_compatibility(target, draft);
    CHECK(result.ok == false, "llama + qwen3_dflash should fail");
    CHECK(result.status == "incompatible_arch", "status should be incompatible_arch");
}

// --- classify_model ---

static void test_classify_qwen35_no_draft() {
    ModelCapabilities target = make_qwen35_target();

    CompatibilityResult result = classify_model(target, nullptr);
    CHECK(result.ok == false, "qwen35 with no draft should fail");
    CHECK(result.status == "adapter_exists_draft_missing", "status adapter_exists_draft_missing");
    CHECK(result.reason.find("qwen35") != std::string::npos, "reason mentions qwen35");
}

static void test_classify_qwen35_with_draft() {
    ModelCapabilities target = make_qwen35_target();
    DraftCapabilities draft = make_qwen3_dflash_caps();

    CompatibilityResult result = classify_model(target, &draft);
    CHECK(result.ok == true, "qwen35 with draft should be supported");
    CHECK(result.status == "supported", "status should be supported");
}

static void test_classify_unsupported_no_draft() {
    ModelCapabilities target = make_gemma_target();

    CompatibilityResult result = classify_model(target, nullptr);
    CHECK(result.ok == false, "gemma with no draft should fail");
    CHECK(result.status == "unsupported_architecture", "status unsupported_architecture");
}

static void test_classify_unsupported_with_draft() {
    ModelCapabilities target = make_gemma_target();
    DraftCapabilities draft = make_qwen3_dflash_caps();

    CompatibilityResult result = classify_model(target, &draft);
    CHECK(result.ok == false, "gemma with dflash draft should fail");
    CHECK(result.status == "incompatible_arch", "status should be incompatible_arch");
}

int main() {
    test_qwen35_with_dflash_draft();
    test_gemma_with_dflash_draft();
    test_llama_with_dflash_draft();
    test_classify_qwen35_no_draft();
    test_classify_qwen35_with_draft();
    test_classify_unsupported_no_draft();
    test_classify_unsupported_with_draft();

    printf("test_compatibility_registry: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
