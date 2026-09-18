#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "gguf_reader.hpp"

namespace localai {

struct LayerInfo {
    std::string name;
    uint64_t param_count;
    uint64_t byte_size;
    GGMLType dtype;
    std::vector<uint64_t> shape;
};

struct ModelArchitecture {
    std::string arch_type;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t embedding_dim;
    uint32_t feed_forward_dim;
    uint32_t vocab_size;
    uint32_t context_length;
    float rope_freq_base;
    std::string quantization_type;
};

struct TensorStats {
    uint64_t total_params;
    uint64_t total_bytes;
    uint32_t n_tensors;
    std::vector<LayerInfo> layers;
    std::unordered_map<std::string, uint64_t> bytes_by_dtype;
    std::unordered_map<std::string, uint32_t> count_by_dtype;
};

class TensorInspector {
public:
    static ModelArchitecture extract_architecture(const GGUFFile& gguf);
    static TensorStats analyze_tensors(const GGUFFile& gguf);
    static std::string format_byte_size(uint64_t bytes);
    static std::string format_param_count(uint64_t params);
};

}
