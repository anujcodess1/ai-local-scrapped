#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include "gguf_reader.hpp"
#include "tensor_info.hpp"

namespace localai {

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};

struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};

struct block_q4_K {
    uint16_t d;          // fp16 super-block scale
    uint16_t dmin;       // fp16 super-block min
    uint8_t scales[12];  // packed 6-bit scales & mins
    uint8_t qs[128];     // 4-bit quants for 256 values
};

struct block_q6_K {
    uint8_t ql[128];     // low 4 bits of 256 values
    uint8_t qh[64];      // high 2 bits of 256 values
    int8_t scales[16];   // 16 8-bit scales
    uint16_t d;          // fp16 super-block scale
};
#pragma pack(pop)

struct TransformerWeights {
    const uint8_t* mmap_data = nullptr;
    uint64_t file_size = 0;

    const void* token_embd = nullptr;
    GGMLType token_embd_type = GGMLType::F32;

    const float* output_norm = nullptr;

    const void* output_weight = nullptr;
    GGMLType output_weight_type = GGMLType::F32;

    struct LayerWeights {
        const void* attn_q = nullptr;
        GGMLType attn_q_type = GGMLType::F32;

        const void* attn_k = nullptr;
        GGMLType attn_k_type = GGMLType::F32;

        const void* attn_v = nullptr;
        GGMLType attn_v_type = GGMLType::F32;

        const void* attn_output = nullptr;
        GGMLType attn_output_type = GGMLType::F32;

        const float* attn_norm = nullptr;

        const void* ffn_gate = nullptr;
        GGMLType ffn_gate_type = GGMLType::F32;

        const void* ffn_down = nullptr;
        GGMLType ffn_down_type = GGMLType::F32;

        const void* ffn_up = nullptr;
        GGMLType ffn_up_type = GGMLType::F32;

        const float* ffn_norm = nullptr;
    };

    std::vector<LayerWeights> layers;
};

struct RunState {
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> xb2;
    std::vector<float> hb;
    std::vector<float> hb2;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> att;
    std::vector<float> logits;

    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

class ThreadPool;

class PureLlamaEngine {
public:
    PureLlamaEngine();
    ~PureLlamaEngine();

    bool load_model(const std::filesystem::path& filepath);
    bool is_loaded() const { return weights_.mmap_data != nullptr; }
    void reset_cache();

    void forward(int token, int pos, bool compute_logits = true);
    int sample(float temperature = 0.7f, float top_p = 0.9f, int top_k = 40, float repeat_penalty = 1.1f, const std::vector<int32_t>& recent_tokens = {});

    const ModelArchitecture& get_architecture() const { return arch_; }
    const std::vector<float>& get_logits() const { return state_.logits; }

    void set_threads(int threads);
    int get_threads() const { return threads_; }

    static void dequantize_row(float* y, const void* vx, GGMLType type, int n);

private:
    ModelArchitecture arch_{};
    GGUFFile gguf_{};
    TransformerWeights weights_{};
    RunState state_{};
    int threads_ = 8;
    std::unique_ptr<ThreadPool> pool_;

#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int file_fd_ = -1;
#endif

    const void* get_tensor_data(const std::string& name, GGMLType& type_out) const;
    void allocate_state();
    void embed_token(int token, float* out_x) const;

    static void rmsnorm(float* o, const float* x, const float* weight, int size, float eps = 1e-5f);
    static void matmul(float* xout, const float* x, const void* w, GGMLType type, int n, int d, ThreadPool* pool = nullptr);
};

}
