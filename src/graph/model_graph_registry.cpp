// Graph module registry implementation.
// Registers qwen35_dense and qwen35moe modules.

#include "graph/model_graph_registry.h"
#include "graph/target_graph_module.h"
#include "graph/qwen35_dense_module.h"
#include "graph/qwen35_moe_module.h"

#include <mutex>
#include <unordered_map>

namespace dflash27b {
namespace {

std::mutex g_registry_mutex;
bool g_initialized = false;
std::unordered_map<std::string, TargetGraphModule *> g_registry;
std::vector<TargetGraphModule *> g_modules_owned;  // for cleanup

void ensure_init() {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (g_initialized) return;

    // Register qwen35 dense module
    {
        auto * mod = new Qwen35DenseModule();
        g_modules_owned.push_back(mod);
        auto caps = mod->capabilities();
        for (const auto & arch : caps.gguf_arches) {
            g_registry[arch] = mod;
        }
    }

    // Register qwen35moe module
    {
        auto * mod = new Qwen35MoeModule();
        g_modules_owned.push_back(mod);
        auto caps = mod->capabilities();
        for (const auto & arch : caps.gguf_arches) {
            g_registry[arch] = mod;
        }
    }

    g_initialized = true;
}

}  // namespace

void init_graph_registry() {
    ensure_init();
}

TargetGraphModule * get_graph_module(const std::string & arch) {
    ensure_init();
    auto it = g_registry.find(arch);
    return (it != g_registry.end()) ? it->second : nullptr;
}

GraphModuleInfo get_graph_module_info(const std::string & arch) {
    ensure_init();
    auto it = g_registry.find(arch);
    if (it != g_registry.end()) {
        return {arch, it->second->capabilities().name, true, ""};
    }
    return {arch, "", false, "no graph module registered for arch '" + arch + "'"};
}

std::vector<const TargetGraphModule *> list_graph_modules() {
    ensure_init();
    std::vector<const TargetGraphModule *> result;
    for (auto * mod : g_modules_owned) {
        result.push_back(mod);
    }
    return result;
}

}  // namespace dflash27b
