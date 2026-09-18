#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <cstdint>
#include <variant>

namespace localai {

enum class GGUFValueType : uint32_t {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,
    STRING   = 8,
    ARRAY    = 9,
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
};

enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
};

using GGUFValue = std::variant<
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t,
    float, double,
    bool, std::string,
    std::vector<int32_t>,
    std::vector<std::string>
>;

struct GGUFMetadata {
    std::string key;
    GGUFValueType type;
    GGUFValue value;
};

struct TensorDescriptor {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dimensions;
    GGMLType dtype;
    uint64_t offset;
    uint64_t n_elements;
    uint64_t byte_size;
};

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct GGUFFile {
    GGUFHeader header;
    std::vector<GGUFMetadata> metadata;
    std::vector<TensorDescriptor> tensors;
    std::unordered_map<std::string, size_t> metadata_index;
    std::unordered_map<std::string, size_t> tensor_index;
    uint64_t data_offset;

    bool has_key(const std::string& key) const;
    std::string get_string(const std::string& key, const std::string& fallback = "") const;
    int64_t get_int(const std::string& key, int64_t fallback = 0) const;
    float get_float(const std::string& key, float fallback = 0.0f) const;
    std::vector<std::string> get_string_array(const std::string& key) const;
};

class GGUFReader {
public:
    static GGUFFile parse(const std::filesystem::path& filepath);

private:
    static std::string read_gguf_string(std::ifstream& stream);
    static GGUFValue read_value(std::ifstream& stream, GGUFValueType type);
    static uint64_t compute_tensor_size(GGMLType dtype, uint64_t n_elements);
    static const char* dtype_name(GGMLType dtype);
};

}
