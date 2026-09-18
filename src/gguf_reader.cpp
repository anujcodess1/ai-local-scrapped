#include "localai/gguf_reader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>

namespace localai {

bool GGUFFile::has_key(const std::string& key) const {
    return metadata_index.find(key) != metadata_index.end();
}

std::string GGUFFile::get_string(const std::string& key, const std::string& fallback) const {
    auto it = metadata_index.find(key);
    if (it == metadata_index.end()) return fallback;
    const auto& val = metadata[it->second].value;
    if (auto* s = std::get_if<std::string>(&val)) return *s;
    return fallback;
}

int64_t GGUFFile::get_int(const std::string& key, int64_t fallback) const {
    auto it = metadata_index.find(key);
    if (it == metadata_index.end()) return fallback;
    const auto& val = metadata[it->second].value;
    if (auto* v = std::get_if<int32_t>(&val)) return *v;
    if (auto* v = std::get_if<uint32_t>(&val)) return *v;
    if (auto* v = std::get_if<int64_t>(&val)) return *v;
    if (auto* v = std::get_if<uint64_t>(&val)) return static_cast<int64_t>(*v);
    if (auto* v = std::get_if<int16_t>(&val)) return *v;
    if (auto* v = std::get_if<uint16_t>(&val)) return *v;
    if (auto* v = std::get_if<int8_t>(&val)) return *v;
    if (auto* v = std::get_if<uint8_t>(&val)) return *v;
    return fallback;
}

float GGUFFile::get_float(const std::string& key, float fallback) const {
    auto it = metadata_index.find(key);
    if (it == metadata_index.end()) return fallback;
    const auto& val = metadata[it->second].value;
    if (auto* v = std::get_if<float>(&val)) return *v;
    if (auto* v = std::get_if<double>(&val)) return static_cast<float>(*v);
    return fallback;
}

std::vector<std::string> GGUFFile::get_string_array(const std::string& key) const {
    auto it = metadata_index.find(key);
    if (it == metadata_index.end()) return {};
    const auto& val = metadata[it->second].value;
    if (auto* v = std::get_if<std::vector<std::string>>(&val)) return *v;
    return {};
}

std::string GGUFReader::read_gguf_string(std::ifstream& stream) {
    uint64_t length = 0;
    stream.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (length == 0 || length > 1024 * 1024) return "";
    std::string result(length, '\0');
    stream.read(result.data(), length);
    return result;
}

GGUFValue GGUFReader::read_value(std::ifstream& stream, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:   { uint8_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::INT8:    { int8_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::UINT16:  { uint16_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::INT16:   { int16_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::UINT32:  { uint32_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::INT32:   { int32_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::UINT64:  { uint64_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::INT64:   { int64_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::FLOAT32: { float v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::FLOAT64: { double v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
        case GGUFValueType::BOOL:    { uint8_t v; stream.read(reinterpret_cast<char*>(&v), sizeof(v)); return static_cast<bool>(v); }
        case GGUFValueType::STRING:  { return read_gguf_string(stream); }
        case GGUFValueType::ARRAY: {
            uint32_t element_type;
            uint64_t count;
            stream.read(reinterpret_cast<char*>(&element_type), sizeof(element_type));
            stream.read(reinterpret_cast<char*>(&count), sizeof(count));

            if (static_cast<GGUFValueType>(element_type) == GGUFValueType::STRING) {
                std::vector<std::string> arr;
                arr.reserve(count);
                for (uint64_t i = 0; i < count; ++i) {
                    arr.push_back(read_gguf_string(stream));
                }
                return arr;
            } else if (static_cast<GGUFValueType>(element_type) == GGUFValueType::INT32) {
                std::vector<int32_t> arr(count);
                for (uint64_t i = 0; i < count; ++i) {
                    stream.read(reinterpret_cast<char*>(&arr[i]), sizeof(int32_t));
                }
                return arr;
            } else {
                uint64_t elem_size = 0;
                switch (static_cast<GGUFValueType>(element_type)) {
                    case GGUFValueType::UINT8: case GGUFValueType::INT8: case GGUFValueType::BOOL: elem_size = 1; break;
                    case GGUFValueType::UINT16: case GGUFValueType::INT16: elem_size = 2; break;
                    case GGUFValueType::UINT32: case GGUFValueType::INT32: case GGUFValueType::FLOAT32: elem_size = 4; break;
                    case GGUFValueType::UINT64: case GGUFValueType::INT64: case GGUFValueType::FLOAT64: elem_size = 8; break;
                    default: elem_size = 4; break;
                }
                stream.seekg(count * elem_size, std::ios::cur);
                return std::vector<int32_t>{};
            }
        }
    }
    return uint8_t(0);
}

uint64_t GGUFReader::compute_tensor_size(GGMLType dtype, uint64_t n_elements) {
    switch (dtype) {
        case GGMLType::F32:     return n_elements * 4;
        case GGMLType::F16:     return n_elements * 2;
        case GGMLType::BF16:    return n_elements * 2;
        case GGMLType::Q4_0:    return (n_elements / 32) * 18;
        case GGMLType::Q4_1:    return (n_elements / 32) * 20;
        case GGMLType::Q5_0:    return (n_elements / 32) * 22;
        case GGMLType::Q5_1:    return (n_elements / 32) * 24;
        case GGMLType::Q8_0:    return (n_elements / 32) * 34;
        case GGMLType::Q8_1:    return (n_elements / 32) * 36;
        case GGMLType::Q2_K:    return (n_elements / 256) * 84;
        case GGMLType::Q3_K:    return (n_elements / 256) * 110;
        case GGMLType::Q4_K:    return (n_elements / 256) * 144;
        case GGMLType::Q5_K:    return (n_elements / 256) * 176;
        case GGMLType::Q6_K:    return (n_elements / 256) * 210;
        case GGMLType::Q8_K:    return (n_elements / 256) * 292;
        case GGMLType::I8:      return n_elements;
        case GGMLType::I16:     return n_elements * 2;
        case GGMLType::I32:     return n_elements * 4;
        case GGMLType::I64:     return n_elements * 8;
        case GGMLType::F64:     return n_elements * 8;
        default:                return n_elements * 2;
    }
}

const char* GGUFReader::dtype_name(GGMLType dtype) {
    switch (dtype) {
        case GGMLType::F32:     return "F32";
        case GGMLType::F16:     return "F16";
        case GGMLType::BF16:    return "BF16";
        case GGMLType::Q4_0:    return "Q4_0";
        case GGMLType::Q4_1:    return "Q4_1";
        case GGMLType::Q5_0:    return "Q5_0";
        case GGMLType::Q5_1:    return "Q5_1";
        case GGMLType::Q8_0:    return "Q8_0";
        case GGMLType::Q8_1:    return "Q8_1";
        case GGMLType::Q2_K:    return "Q2_K";
        case GGMLType::Q3_K:    return "Q3_K";
        case GGMLType::Q4_K:    return "Q4_K";
        case GGMLType::Q5_K:    return "Q5_K";
        case GGMLType::Q6_K:    return "Q6_K";
        case GGMLType::Q8_K:    return "Q8_K";
        case GGMLType::I8:      return "I8";
        case GGMLType::I16:     return "I16";
        case GGMLType::I32:     return "I32";
        default:                return "UNKNOWN";
    }
}

GGUFFile GGUFReader::parse(const std::filesystem::path& filepath) {
    GGUFFile gguf;
    std::ifstream stream(filepath, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + filepath.string());
    }

    stream.read(reinterpret_cast<char*>(&gguf.header.magic), sizeof(gguf.header.magic));
    if (gguf.header.magic != 0x46554747) {
        throw std::runtime_error("Invalid GGUF magic bytes in weight file");
    }

    stream.read(reinterpret_cast<char*>(&gguf.header.version), sizeof(gguf.header.version));
    stream.read(reinterpret_cast<char*>(&gguf.header.tensor_count), sizeof(gguf.header.tensor_count));
    stream.read(reinterpret_cast<char*>(&gguf.header.metadata_kv_count), sizeof(gguf.header.metadata_kv_count));

    gguf.metadata.reserve(gguf.header.metadata_kv_count);
    for (uint64_t i = 0; i < gguf.header.metadata_kv_count; ++i) {
        GGUFMetadata kv;
        kv.key = read_gguf_string(stream);

        uint32_t value_type;
        stream.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));
        kv.type = static_cast<GGUFValueType>(value_type);
        kv.value = read_value(stream, kv.type);

        gguf.metadata_index[kv.key] = gguf.metadata.size();
        gguf.metadata.push_back(std::move(kv));

        if (!stream.good()) break;
    }

    gguf.tensors.reserve(gguf.header.tensor_count);
    for (uint64_t i = 0; i < gguf.header.tensor_count; ++i) {
        TensorDescriptor td;
        td.name = read_gguf_string(stream);
        stream.read(reinterpret_cast<char*>(&td.n_dims), sizeof(td.n_dims));

        td.dimensions.resize(td.n_dims);
        for (uint32_t d = 0; d < td.n_dims; ++d) {
            stream.read(reinterpret_cast<char*>(&td.dimensions[d]), sizeof(uint64_t));
        }

        uint32_t raw_dtype;
        stream.read(reinterpret_cast<char*>(&raw_dtype), sizeof(raw_dtype));
        td.dtype = static_cast<GGMLType>(raw_dtype);

        stream.read(reinterpret_cast<char*>(&td.offset), sizeof(td.offset));

        td.n_elements = 1;
        for (auto dim : td.dimensions) {
            td.n_elements *= dim;
        }
        td.byte_size = compute_tensor_size(td.dtype, td.n_elements);

        gguf.tensor_index[td.name] = gguf.tensors.size();
        gguf.tensors.push_back(std::move(td));

        if (!stream.good()) break;
    }

    uint64_t current_pos = stream.tellg();
    uint64_t alignment = 32;
    gguf.data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

    return gguf;
}

}
