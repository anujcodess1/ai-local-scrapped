#include "localai/config.hpp"
#include "localai/prompt_builder.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace localai {

namespace {
    std::string extract_json_value(const std::string& json, const std::string& key) {
        std::string pattern = "\"" + key + "\"";
        size_t pos = json.find(pattern);
        if (pos == std::string::npos) return "";

        size_t colon = json.find(':', pos);
        if (colon == std::string::npos) return "";

        size_t start = json.find_first_not_of(" \t\r\n", colon + 1);
        if (start == std::string::npos) return "";

        if (json[start] == '\"') {
            size_t end = json.find('\"', start + 1);
            if (end != std::string::npos) {
                return json.substr(start + 1, end - start - 1);
            }
        } else {
            size_t end = json.find_first_of(",\r\n}", start);
            if (end != std::string::npos) {
                return json.substr(start, end - start);
            }
        }
        return "";
    }
}

EngineConfig ConfigManager::load(const std::filesystem::path& base_dir) {
    EngineConfig cfg;
    cfg.model_path = base_dir / "models" / "1b" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    cfg.runtime_path = base_dir / "bin" / "llama-cli.exe";

    std::filesystem::path config_file = base_dir / "config" / "config.json";
    if (std::filesystem::exists(config_file)) {
        std::ifstream file(config_file);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();

            std::string th = extract_json_value(content, "threads");
            if (!th.empty()) cfg.threads = std::stoi(th);

            std::string ctx = extract_json_value(content, "context_size");
            if (!ctx.empty()) cfg.context_size = std::stoi(ctx);

            std::string bs = extract_json_value(content, "batch_size");
            if (!bs.empty()) cfg.batch_size = std::stoi(bs);

            std::string md = extract_json_value(content, "mode");
            if (!md.empty()) cfg.mode = md;

            std::string sp = extract_json_value(content, "system_prompt");
            if (!sp.empty()) cfg.system_prompt = sp;

            auto read_float = [&content](const char* key, float fallback) {
                std::string v = extract_json_value(content, key);
                if (!v.empty()) {
                    try { return std::stof(v); } catch (...) { /* keep fallback */ }
                }
                return fallback;
            };

            cfg.temperature    = read_float("temperature", cfg.temperature);
            cfg.top_p          = read_float("top_p", cfg.top_p);
            cfg.min_p          = read_float("min_p", cfg.min_p);
            cfg.repeat_penalty = read_float("repeat_penalty", cfg.repeat_penalty);

            auto read_int = [&content](const char* key, int fallback) {
                std::string v = extract_json_value(content, key);
                if (!v.empty()) {
                    try { return std::stoi(v); } catch (...) { /* keep fallback */ }
                }
                return fallback;
            };

            cfg.top_k        = read_int("top_k", cfg.top_k);
            cfg.repeat_last_n = read_int("repeat_last_n", cfg.repeat_last_n);
            cfg.max_tokens   = read_int("max_tokens", cfg.max_tokens);
        }
    }

    // Optional custom persona: config/system_prompt.txt overrides the built-in one.
    std::filesystem::path persona_file = base_dir / "config" / "system_prompt.txt";
    if (std::filesystem::exists(persona_file)) {
        std::ifstream pf(persona_file, std::ios::binary);
        if (pf.is_open()) {
            std::stringstream pbuf;
            pbuf << pf.rdbuf();
            std::string persona = pbuf.str();
            // Trim trailing whitespace/newlines.
            while (!persona.empty() && std::isspace(static_cast<unsigned char>(persona.back()))) {
                persona.pop_back();
            }
            if (!persona.empty()) cfg.system_prompt = persona;
        }
    }

    return cfg;
}

std::string ConfigManager::default_system_prompt() {
    return kDefaultSystemPrompt;
}

bool ConfigManager::save(const std::filesystem::path& base_dir, const EngineConfig& config) {
    std::filesystem::path config_dir = base_dir / "config";
    std::filesystem::create_directories(config_dir);
    std::filesystem::path config_file = config_dir / "config.json";

    std::ofstream file(config_file);
    if (!file.is_open()) return false;

    file << "{\n";
    file << "  \"threads\": " << config.threads << ",\n";
    file << "  \"context_size\": " << config.context_size << ",\n";
    file << "  \"batch_size\": " << config.batch_size << ",\n";
    file << "  \"temperature\": " << config.temperature << ",\n";
    file << "  \"top_p\": " << config.top_p << ",\n";
    file << "  \"top_k\": " << config.top_k << ",\n";
    file << "  \"min_p\": " << config.min_p << ",\n";
    file << "  \"repeat_penalty\": " << config.repeat_penalty << ",\n";
    file << "  \"repeat_last_n\": " << config.repeat_last_n << ",\n";
    file << "  \"max_tokens\": " << config.max_tokens << ",\n";
    file << "  \"mode\": \"" << config.mode << "\",\n";
    file << "}\n";

    return true;
}

}
