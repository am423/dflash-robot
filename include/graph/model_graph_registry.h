// Model graph module registry.
// Maps GGUF architecture strings to TargetGraphModule instances.
// Unknown architectures report "graph_module_missing" through
// the existing compatibility registry path.

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace dflash27b {

class TargetGraphModule;

// Information about a registered module's availability for a given arch.
struct GraphModuleInfo {
    std::string arch;               // e.g. "qwen35", "qwen35moe"
    std::string module_name;        // e.g. "qwen35_dense"
    bool        available = false;  // true if a module is registered
    std::string reason;             // empty if available, or e.g. "no graph module for arch 'gemma4'"
};

// Initialize the registry. Must be called before any module queries.
// Registers all built-in modules.
void init_graph_registry();

// Return the registered module for the given architecture, or nullptr.
TargetGraphModule * get_graph_module(const std::string & arch);

// Return info for a given architecture (never null; check .available).
GraphModuleInfo get_graph_module_info(const std::string & arch);

// List all registered modules (for dflash_inspect and tooling).
std::vector<const TargetGraphModule *> list_graph_modules();

}  // namespace dflash27b
