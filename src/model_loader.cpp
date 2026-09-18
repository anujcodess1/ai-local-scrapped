#include "localai/model_loader.hpp"
#include <fstream>
#include <array>

namespace localai {

ModelValidationResult ModelLoader::validate_gguf(const std::filesystem::path& filepath) {
    ModelValidationResult result;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        result.error_message = "Failed to open file: " + filepath.string();
        return result;
    }

    std::array<char, 4> magic{};
    file.read(magic.data(), 4);
    if (file.gcount() != 4) {
        result.error_message = "File too small to read magic header";
        return result;
    }

    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        result.error_message = "Invalid header signature (not a valid GGUF file)";
        return result;
    }

    file.read(reinterpret_cast<char*>(&result.version), sizeof(result.version));
    file.read(reinterpret_cast<char*>(&result.tensor_count), sizeof(result.tensor_count));
    file.read(reinterpret_cast<char*>(&result.kv_count), sizeof(result.kv_count));

    if (file.good()) {
        result.valid = true;
    } else {
        result.error_message = "Failed reading GGUF header fields";
    }

    return result;
}

}
