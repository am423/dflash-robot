// dflash_inspect — CLI tool to inspect a GGUF model file and report
// DFlash compatibility status.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

#include "gguf.h"
#include "dflash_robot/compatibility_registry.h"
#include "dflash_robot/adapters/qwen35_adapter.h"
#include "dflash_robot/adapters/qwen3_dflash_draft_adapter.h"
#include "graph/model_graph_registry.h"

using namespace dflash_robot;

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --target PATH [--draft PATH] [--json]\n"
        "\n"
        "Inspect a GGUF model file and report DFlash compatibility.\n"
        "\n"
        "Options:\n"
        "  --target PATH   Path to target GGUF model file (required)\n"
        "  --draft PATH    Path to draft GGUF model file (optional)\n"
        "  --json          Output in JSON format\n",
        prog);
}

static std::string read_gguf_string(const char* fname, const char* key) {
    struct gguf_context* ctx = gguf_init_from_file(fname, {.no_alloc=true, .ctx=nullptr});
    if (!ctx) return "";

    int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        gguf_free(ctx);
        return "";
    }
    const char* val = gguf_get_val_str(ctx, kid);
    std::string result(val ? val : "");
    gguf_free(ctx);
    return result;
}

static int read_gguf_int(const char* fname, const char* key) {
    struct gguf_context* ctx = gguf_init_from_file(fname, {.no_alloc=true, .ctx=nullptr});
    if (!ctx) return -1;

    int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        gguf_free(ctx);
        return -1;
    }

    // GGUF metadata uses u32 for most numeric fields; try u32 first, then i32
    enum gguf_type ktype = gguf_get_kv_type(ctx, kid);
    int result = -1;
    if (ktype == GGUF_TYPE_UINT32) {
        result = static_cast<int>(gguf_get_val_u32(ctx, kid));
    } else if (ktype == GGUF_TYPE_INT32) {
        result = static_cast<int>(gguf_get_val_i32(ctx, kid));
    } else if (ktype == GGUF_TYPE_UINT64) {
        result = static_cast<int>(gguf_get_val_u64(ctx, kid));
    } else if (ktype == GGUF_TYPE_INT64) {
        result = static_cast<int>(gguf_get_val_i64(ctx, kid));
    } else if (ktype == GGUF_TYPE_FLOAT32) {
        result = static_cast<int>(gguf_get_val_f32(ctx, kid));
    }
    gguf_free(ctx);
    return result;
}

static std::string infer_next_action(const std::string& status, const std::string& arch) {
    if (status == "supported") {
        return "Ready for DFlash speculative decoding";
    }
    if (status == "adapter_exists_draft_missing") {
        return "Obtain a DFlash draft model (GGUF or safetensors) for " + arch;
    }
    if (status == "incompatible_arch") {
        return "No DFlash draft available for this architecture";
    }
    return "Architecture not supported by dflash-robot";
}

int main(int argc, char** argv) {
    const char* target_path = nullptr;
    const char* draft_path = nullptr;
    bool json_output = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            target_path = argv[++i];
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            draft_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!target_path) {
        fprintf(stderr, "Error: --target PATH is required\n");
        usage(argv[0]);
        return 1;
    }

    // Read target metadata
    std::string arch = read_gguf_string(target_path, "general.architecture");
    if (arch.empty()) {
        fprintf(stderr, "Error: could not read 'general.architecture' from %s\n", target_path);
        return 1;
    }

    // Try architecture-specific keys: "{arch}.block_count", "{arch}.embedding_length", "{arch}.vocab_size"
    // Also try generic fallbacks
    std::string layer_key = arch + ".block_count";
    std::string hidden_key = arch + ".embedding_length";
    std::string vocab_key = arch + ".vocab_size";

    int n_layers = read_gguf_int(target_path, layer_key.c_str());
    int hidden_size = read_gguf_int(target_path, hidden_key.c_str());
    int vocab_size = read_gguf_int(target_path, vocab_key.c_str());

    // Build target capabilities
    ModelCapabilities target;
    target.arch = arch;
    target.n_layers = n_layers > 0 ? n_layers : 0;
    target.hidden_size = hidden_size > 0 ? hidden_size : 0;
    target.vocab_size = vocab_size > 0 ? vocab_size : 0;
    // These would need adapter registration; default to false for unknown arches
    target.supports_hidden_capture = false;
    target.supports_cache_rollback = false;
    target.has_ssm_state = false;

    // If we have a qwen35 or qwen35moe adapter, fill in capabilities
    if (arch == "qwen35" || arch == "qwen35moe") {
        Qwen35Adapter adapter(target.n_layers, target.hidden_size, target.vocab_size);
        ModelCapabilities full = adapter.capabilities();
        target.supports_hidden_capture = full.supports_hidden_capture;
        target.supports_cache_rollback = full.supports_cache_rollback;
        target.has_ssm_state = full.has_ssm_state;
    }

    // If draft path given, try to read draft metadata
    DraftCapabilities* draft_caps_ptr = nullptr;
    DraftCapabilities draft_caps;

    if (draft_path) {
        std::string draft_arch = read_gguf_string(draft_path, "general.architecture");
        // For now, we only know the qwen3_dflash draft family
        // A real system would look up adapters by draft model metadata
        if (!draft_arch.empty()) {
            Qwen3DFlashDraftAdapter draft_adapter;
            draft_caps = draft_adapter.capabilities();
            draft_caps_ptr = &draft_caps;
        }
    }

    // Classify
    CompatibilityResult result = classify_model(target, draft_caps_ptr);

    // Get graph module info
    dflash27b::init_graph_registry();
    dflash27b::GraphModuleInfo gminfo = dflash27b::get_graph_module_info(target.arch);

    // Build next_action
    std::string next_action = infer_next_action(result.status, target.arch);

    // Output
    if (json_output) {
        printf("{\n");
        printf("  \"target_path\": \"%s\",\n", target_path);
        printf("  \"target_arch\": \"%s\",\n", target.arch.c_str());
        printf("  \"target_dims\": {\n");
        printf("    \"n_layers\": %d,\n", target.n_layers);
        printf("    \"hidden_size\": %d,\n", target.hidden_size);
        printf("    \"vocab_size\": %d\n", target.vocab_size);
        printf("  },\n");
        printf("  \"compatibility_status\": \"%s\",\n", result.status.c_str());
        printf("  \"reason\": \"%s\",\n", result.reason.c_str());
        printf("  \"next_action\": \"%s\",\n", next_action.c_str());
        printf("  \"graph_module\": {\n");
        printf("    \"name\": \"%s\",\n", gminfo.module_name.c_str());
        printf("    \"available\": %s,\n", gminfo.available ? "true" : "false");
        printf("    \"reason\": \"%s\"\n", gminfo.reason.c_str());
        printf("  }\n");
        printf("}\n");
    } else {
        printf("Target: %s\n", target_path);
        printf("  Architecture: %s\n", target.arch.c_str());
        printf("  Layers: %d\n", target.n_layers);
        printf("  Hidden size: %d\n", target.hidden_size);
        printf("  Vocab size: %d\n", target.vocab_size);
        printf("\nCompatibility: %s\n", result.status.c_str());
        if (!result.reason.empty()) {
            printf("Reason: %s\n", result.reason.c_str());
        }
        printf("Next action: %s\n", next_action.c_str());
    }

    return result.ok ? 0 : 1;
}
