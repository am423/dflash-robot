#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <set>
#include "dflash_robot/adapters/qwen35_adapter.h"

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
    Qwen35Adapter adapter;
    ModelCapabilities caps = adapter.capabilities();

    CHECK(caps.arch == "qwen35", "arch should be qwen35");
    CHECK(caps.n_layers == 64, "n_layers should be 64");
    CHECK(caps.hidden_size == 5120, "hidden_size should be 5120");
    CHECK(caps.vocab_size == 248320, "vocab_size should be 248320");
    CHECK(caps.supports_hidden_capture == true, "supports_hidden_capture");
    CHECK(caps.supports_cache_rollback == true, "supports_cache_rollback");
    CHECK(caps.has_ssm_state == true, "has_ssm_state");
}

static void test_layer_selection_count() {
    Qwen35Adapter adapter;
    LayerSelection sel = adapter.default_layer_selection(5);

    CHECK(sel.layer_ids.size() == 5, "should return 5 layers for n_features=5");
    CHECK(sel.hidden_size == 5120, "hidden_size should be 5120");
    CHECK(sel.fused_size == 5 * 5120, "fused_size should be 5*5120");
}

static void test_layer_selection_range() {
    Qwen35Adapter adapter;
    LayerSelection sel = adapter.default_layer_selection(5);

    // All layer IDs must be in [1, 61] (i.e., [1, n_layers-3])
    for (int lid : sel.layer_ids) {
        CHECK(lid >= 1 && lid <= 61, "layer ID in valid range");
    }

    // Should be {1, 16, 31, 46, 61} for n_layers=64
    CHECK(sel.layer_ids[0] == 1,  "first layer should be 1");
    CHECK(sel.layer_ids[1] == 16, "second layer should be 16");
    CHECK(sel.layer_ids[2] == 31, "third layer should be 31");
    CHECK(sel.layer_ids[3] == 46, "fourth layer should be 46");
    CHECK(sel.layer_ids[4] == 61, "fifth layer should be 61");
}

static void test_layer_selection_single() {
    Qwen35Adapter adapter;
    LayerSelection sel = adapter.default_layer_selection(1);

    CHECK(sel.layer_ids.size() == 1, "should return 1 layer for n_features=1");
    CHECK(sel.layer_ids[0] == 1, "single layer should be 1");
}

static void test_layer_selection_custom_layers() {
    Qwen35Adapter adapter(32, 2048, 32000); // smaller model
    LayerSelection sel = adapter.default_layer_selection(5);

    // Range [1, 29], step = 28/4 = 7
    CHECK(sel.layer_ids.size() == 5, "should return 5 layers");
    CHECK(sel.layer_ids[0] == 1,  "first layer");
    CHECK(sel.layer_ids[4] == 29, "last layer (32-3)");
    CHECK(sel.hidden_size == 2048, "hidden_size matches");
}

int main() {
    test_capabilities();
    test_layer_selection_count();
    test_layer_selection_range();
    test_layer_selection_single();
    test_layer_selection_custom_layers();

    printf("test_model_adapter: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
