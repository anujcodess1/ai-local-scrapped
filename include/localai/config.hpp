#pragma once

#include <string>
#include <filesystem>

namespace localai {

struct EngineConfig {
    std::filesystem::path model_path;
    std::filesystem::path runtime_path;
    int threads = 6;
    int context_size = 2048;
    int batch_size = 256;
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 40;
    float min_p = 0.05f;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    int max_tokens = 1024;
    std::string mode = "Balanced";
    // Custom persona; falls back to kDefaultSystemPrompt when empty.
    std::string system_prompt;
};

class ConfigManager {
public:
    static EngineConfig load(const std::filesystem::path& base_dir);
    static bool save(const std::filesystem::path& base_dir, const EngineConfig& config);
    static std::string default_system_prompt();
};

}
