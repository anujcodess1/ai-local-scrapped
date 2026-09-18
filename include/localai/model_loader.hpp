#pragma once

#include <string>
#include <filesystem>
#include <cstdint>

namespace localai {

struct ModelValidationResult {
    bool valid = false;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    std::string error_message;
};

class ModelLoader {
public:
    static ModelValidationResult validate_gguf(const std::filesystem::path& filepath);
};

}
