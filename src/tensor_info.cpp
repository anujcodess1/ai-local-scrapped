#include "localai/tensor_info.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace localai {

ModelArchitecture TensorInspector::extract_architecture(const GGUFFile& gguf) {
    ModelArchitecture arch;
    arch.arch_type = gguf.get_string("general.architecture", "llama");
    std::string prefix = arch.arch_type + ".";

    arch.n_layers          = static_cast<uint32_t>(gguf.get_int(prefix + "block_count", 0));
    arch.n_heads           = static_cast<uint32_t>(gguf.get_int(prefix + "attention.head_count", 0));
    arch.n_kv_heads        = static_cast<uint32_t>(gguf.get_int(prefix + "attention.head_count_kv", arch.n_heads));
    arch.embedding_dim     = static_cast<uint32_t>(gguf.get_int(prefix + "embedding_length", 0));
    arch.feed_forward_dim  = static_cast<uint32_t>(gguf.get_int(prefix + "feed_forward_length", 0));
    arch.vocab_size        = static_cast<uint32_t>(gguf.get_int(prefix + "vocab_size", 0));
    arch.context_length    = static_cast<uint32_t>(gguf.get_int(prefix + "context_length", 0));
    arch.rope_freq_base    = gguf.get_float(prefix + "rope.freq_base", 500000.0f);

    std::string ftype = gguf.get_string("general.file_type", "");
    if (!ftype.empty()) {
        arch.quantization_type = ftype;
    } else {
        int64_t ft = gguf.get_int("general.file_type", 0);
        switch (ft) {
            case 0:  arch.quantization_type = "F32"; break;
            case 1:  arch.quantization_type = "F16"; break;
            case 2:  arch.quantization_type = "Q4_0"; break;
            case 3:  arch.quantization_type = "Q4_1"; break;
            case 7:  arch.quantization_type = "Q8_0"; break;
            case 15: arch.quantization_type = "Q4_K_M"; break;
            case 17: arch.quantization_type = "Q5_K_M"; break;
            case 18: arch.quantization_type = "Q6_K"; break;
            default: arch.quantization_type = "Q" + std::to_string(ft); break;
        }
    }

    return arch;
}

TensorStats TensorInspector::analyze_tensors(const GGUFFile& gguf) {
    TensorStats stats{};
    stats.n_tensors = static_cast<uint32_t>(gguf.tensors.size());

    for (const auto& tensor : gguf.tensors) {
        LayerInfo layer;
        layer.name = tensor.name;
        layer.param_count = tensor.n_elements;
        layer.byte_size = tensor.byte_size;
        layer.dtype = tensor.dtype;
        layer.shape = tensor.dimensions;

        stats.total_params += tensor.n_elements;
        stats.total_bytes += tensor.byte_size;

        const char* dtype_str = "UNKNOWN";
        switch (tensor.dtype) {
            case GGMLType::F32:  dtype_str = "F32"; break;
            case GGMLType::F16:  dtype_str = "F16"; break;
            case GGMLType::Q4_0: dtype_str = "Q4_0"; break;
            case GGMLType::Q4_1: dtype_str = "Q4_1"; break;
            case GGMLType::Q4_K: dtype_str = "Q4_K"; break;
            case GGMLType::Q5_K: dtype_str = "Q5_K"; break;
            case GGMLType::Q6_K: dtype_str = "Q6_K"; break;
            case GGMLType::Q8_0: dtype_str = "Q8_0"; break;
            case GGMLType::BF16: dtype_str = "BF16"; break;
            default: dtype_str = "OTHER"; break;
        }

        stats.bytes_by_dtype[dtype_str] += tensor.byte_size;
        stats.count_by_dtype[dtype_str]++;
        stats.layers.push_back(std::move(layer));
    }

    return stats;
}

std::string TensorInspector::format_byte_size(uint64_t bytes) {
    std::ostringstream oss;
    if (bytes >= 1024ULL * 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024 * 1024)) << " GB";
    } else if (bytes >= 1024ULL * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024)) << " MB";
    } else if (bytes >= 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / 1024.0) << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

std::string TensorInspector::format_param_count(uint64_t params) {
    std::ostringstream oss;
    if (params >= 1000000000ULL) {
        oss << std::fixed << std::setprecision(2) << (params / 1e9) << "B";
    } else if (params >= 1000000ULL) {
        oss << std::fixed << std::setprecision(2) << (params / 1e6) << "M";
    } else if (params >= 1000ULL) {
        oss << std::fixed << std::setprecision(1) << (params / 1e3) << "K";
    } else {
        oss << params;
    }
    return oss.str();
}

}
