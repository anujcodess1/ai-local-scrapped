#include "localai/model_forward.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <unordered_set>
#include <immintrin.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace localai {

namespace {

// Fast, portable IEEE-754 FP16 to FP32 conversion
inline float fp16_to_fp32(uint16_t h) {
    uint32_t w = static_cast<uint32_t>(h & 0x7FFF) << 13;
    uint32_t v = static_cast<uint32_t>(h & 0x8000) << 16;
    if ((h & 0x7C00) == 0x7C00) {
        v |= 0x7F800000 | w;
    } else if ((h & 0x7C00) != 0) {
        v |= w + 0x38000000;
    } else if (w != 0) {
        int shift = 0;
        while ((w & 0x00800000) == 0) {
            w <<= 1;
            shift++;
        }
        w &= 0x007FFFFF;
        v |= ((127 - 15 - shift + 1) << 23) | w;
    }
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j] >> 6) << 4);
    }
}

void dequantize_row_q4_k(const block_q4_K* x, float* y, int64_t k) {
    const int nb = static_cast<int>(k / 256);
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;

            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q6_k(const block_q6_K* x, float* y, int64_t k) {
    const int nb = static_cast<int>(k / 256);
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

} // namespace

void PureLlamaEngine::dequantize_row(float* y, const void* vx, GGMLType type, int n) {
    if (type == GGMLType::F32) {
        std::memcpy(y, vx, n * sizeof(float));
    } else if (type == GGMLType::Q4_K) {
        dequantize_row_q4_k(static_cast<const block_q4_K*>(vx), y, n);
    } else if (type == GGMLType::Q6_K) {
        dequantize_row_q6_k(static_cast<const block_q6_K*>(vx), y, n);
    } else {
        std::memset(y, 0, n * sizeof(float));
    }
}

class ThreadPool {
public:
    explicit ThreadPool(int num_threads) : stop_(false), generation_(0), active_workers_(0) {
        int n = std::max(1, num_threads);
        workers_.reserve(n - 1);
        for (int i = 1; i < n; ++i) {
            workers_.emplace_back([this, i, n]() {
                uint64_t my_gen = 0;
                while (true) {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_work_.wait(lock, [this, my_gen]() {
                            return stop_ || generation_ > my_gen;
                        });
                        if (stop_) return;
                        my_gen = generation_;
                    }

                    int total = current_total_;
                    int chunk = (total + n - 1) / n;
                    int start = i * chunk;
                    int end = std::min(total, start + chunk);
                    if (start < end && current_fn_) {
                        (*current_fn_)(start, end);
                    }

                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        --active_workers_;
                        if (active_workers_ == 0) {
                            cv_done_.notify_all();
                        }
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void parallel_for(int total, const std::function<void(int start, int end)>& fn) {
        int n = static_cast<int>(workers_.size()) + 1;
        if (n <= 1 || total <= 256) {
            fn(0, total);
            return;
        }

        {
            std::unique_lock<std::mutex> lock(mutex_);
            current_fn_ = &fn;
            current_total_ = total;
            active_workers_ = n - 1;
            ++generation_;
        }
        cv_work_.notify_all();

        int chunk = (total + n - 1) / n;
        int end_0 = std::min(total, chunk);
        if (end_0 > 0) {
            fn(0, end_0);
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this]() { return active_workers_ == 0; });
        current_fn_ = nullptr;
    }

    int size() const { return static_cast<int>(workers_.size()) + 1; }

private:
    std::vector<std::thread> workers_;
    const std::function<void(int, int)>* current_fn_ = nullptr;
    int current_total_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    uint64_t generation_;
    int active_workers_;
    bool stop_;
};

PureLlamaEngine::PureLlamaEngine() {
    pool_ = std::make_unique<ThreadPool>(threads_);
}

void PureLlamaEngine::set_threads(int threads) {
    threads_ = (threads > 0) ? threads : 8;
    pool_ = std::make_unique<ThreadPool>(threads_);
}

PureLlamaEngine::~PureLlamaEngine() {
#if defined(_WIN32)
    if (weights_.mmap_data) {
        UnmapViewOfFile(weights_.mmap_data);
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
    }
#else
    if (weights_.mmap_data && weights_.mmap_data != MAP_FAILED) {
        munmap(const_cast<uint8_t*>(weights_.mmap_data), weights_.file_size);
    }
    if (file_fd_ >= 0) {
        close(file_fd_);
    }
#endif
}

const void* PureLlamaEngine::get_tensor_data(const std::string& name, GGMLType& type_out) const {
    auto it = gguf_.tensor_index.find(name);
    if (it == gguf_.tensor_index.end()) return nullptr;

    const auto& desc = gguf_.tensors[it->second];
    type_out = desc.dtype;
    return weights_.mmap_data + gguf_.data_offset + desc.offset;
}

bool PureLlamaEngine::load_model(const std::filesystem::path& filepath) {
    try {
        gguf_ = GGUFReader::parse(filepath);
    } catch (...) {
        return false;
    }

    arch_ = TensorInspector::extract_architecture(gguf_);

#if defined(_WIN32)
    file_handle_ = CreateFileW(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER size;
    GetFileSizeEx(file_handle_, &size);
    weights_.file_size = static_cast<uint64_t>(size.QuadPart);

    mapping_handle_ = CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) {
        CloseHandle(file_handle_);
        return false;
    }

    weights_.mmap_data = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (!weights_.mmap_data) return false;
#else
    file_fd_ = open(filepath.c_str(), O_RDONLY);
    if (file_fd_ < 0) return false;

    struct stat sb;
    fstat(file_fd_, &sb);
    weights_.file_size = sb.st_size;

    weights_.mmap_data = static_cast<const uint8_t*>(mmap(nullptr, weights_.file_size, PROT_READ, MAP_SHARED, file_fd_, 0));
    if (weights_.mmap_data == MAP_FAILED) return false;
#endif

    GGMLType type;
    weights_.token_embd = get_tensor_data("token_embd.weight", weights_.token_embd_type);
    weights_.output_norm = static_cast<const float*>(get_tensor_data("output_norm.weight", type));
    weights_.output_weight = get_tensor_data("output.weight", weights_.output_weight_type);
    if (!weights_.output_weight) {
        weights_.output_weight = weights_.token_embd;
        weights_.output_weight_type = weights_.token_embd_type;
    }

    weights_.layers.resize(arch_.n_layers);
    for (uint32_t i = 0; i < arch_.n_layers; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        auto& l = weights_.layers[i];

        l.attn_q = get_tensor_data(prefix + "attn_q.weight", l.attn_q_type);
        l.attn_k = get_tensor_data(prefix + "attn_k.weight", l.attn_k_type);
        l.attn_v = get_tensor_data(prefix + "attn_v.weight", l.attn_v_type);
        l.attn_output = get_tensor_data(prefix + "attn_output.weight", l.attn_output_type);
        l.attn_norm = static_cast<const float*>(get_tensor_data(prefix + "attn_norm.weight", type));

        l.ffn_gate = get_tensor_data(prefix + "ffn_gate.weight", l.ffn_gate_type);
        l.ffn_down = get_tensor_data(prefix + "ffn_down.weight", l.ffn_down_type);
        l.ffn_up = get_tensor_data(prefix + "ffn_up.weight", l.ffn_up_type);
        l.ffn_norm = static_cast<const float*>(get_tensor_data(prefix + "ffn_norm.weight", type));
    }

    allocate_state();
    return true;
}

void PureLlamaEngine::allocate_state() {
    int dim = arch_.embedding_dim;
    int hidden_dim = arch_.feed_forward_dim;
    int max_seq = std::min(arch_.context_length, 2048u);
    int n_kv_heads = arch_.n_kv_heads;
    int head_dim = dim / arch_.n_heads;
    int kv_dim = n_kv_heads * head_dim;

    state_.x.resize(dim);
    state_.xb.resize(dim);
    state_.xb2.resize(dim);
    state_.hb.resize(hidden_dim);
    state_.hb2.resize(hidden_dim);
    state_.q.resize(dim);
    state_.k.resize(kv_dim);
    state_.v.resize(kv_dim);
    state_.att.resize(arch_.n_heads * max_seq);
    state_.logits.resize(arch_.vocab_size);

    state_.key_cache.assign(arch_.n_layers * max_seq * kv_dim, 0.0f);
    state_.value_cache.assign(arch_.n_layers * max_seq * kv_dim, 0.0f);
}

void PureLlamaEngine::reset_cache() {
    std::fill(state_.key_cache.begin(), state_.key_cache.end(), 0.0f);
    std::fill(state_.value_cache.begin(), state_.value_cache.end(), 0.0f);
}

void PureLlamaEngine::embed_token(int token, float* out_x) const {
    int dim = arch_.embedding_dim;
    if (weights_.token_embd_type == GGMLType::F32) {
        const float* emb = static_cast<const float*>(weights_.token_embd) + static_cast<size_t>(token) * dim;
        std::memcpy(out_x, emb, dim * sizeof(float));
    } else if (weights_.token_embd_type == GGMLType::Q6_K) {
        const block_q6_K* blocks = static_cast<const block_q6_K*>(weights_.token_embd) + static_cast<size_t>(token) * (dim / 256);
        dequantize_row_q6_k(blocks, out_x, dim);
    } else if (weights_.token_embd_type == GGMLType::Q4_K) {
        const block_q4_K* blocks = static_cast<const block_q4_K*>(weights_.token_embd) + static_cast<size_t>(token) * (dim / 256);
        dequantize_row_q4_k(blocks, out_x, dim);
    }
}

void PureLlamaEngine::rmsnorm(float* o, const float* x, const float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; ++j) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0f / std::sqrt(ss);

    for (int j = 0; j < size; ++j) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float dot_f32(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float dot_q4_k_row(const block_q4_K* b_row, const float* x, int nb) {
    __m256 v_total = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const float d_val = fp16_to_fp32(b_row[b].d);
        const float min_val = fp16_to_fp32(b_row[b].dmin);
        const uint8_t* q = b_row[b].qs;
        const float* x_ptr = x + b * 256;
        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, b_row[b].scales, &sc, &m);
            const float d1 = d_val * sc;
            const float m1 = min_val * m;

            get_scale_min_k4(is + 1, b_row[b].scales, &sc, &m);
            const float d2 = d_val * sc;
            const float m2 = min_val * m;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 sumx1 = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 8) {
                __m256 vx = _mm256_loadu_ps(x_ptr + l);
                sumx1 = _mm256_add_ps(sumx1, vx);

                __m128i q_raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
                __m128i q_lo = _mm_and_si128(q_raw, _mm_set1_epi8(0x0F));
                __m256i q_i32 = _mm256_cvtepu8_epi32(q_lo);
                __m256 q_f = _mm256_cvtepi32_ps(q_i32);
                acc1 = _mm256_fmadd_ps(q_f, vx, acc1);
            }

            __m256 acc2 = _mm256_setzero_ps();
            __m256 sumx2 = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 8) {
                __m256 vx = _mm256_loadu_ps(x_ptr + 32 + l);
                sumx2 = _mm256_add_ps(sumx2, vx);

                __m128i q_raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
                __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q_raw, 4), _mm_set1_epi8(0x0F));
                __m256i q_i32 = _mm256_cvtepu8_epi32(q_hi);
                __m256 q_f = _mm256_cvtepi32_ps(q_i32);
                acc2 = _mm256_fmadd_ps(q_f, vx, acc2);
            }

            __m256 vd1 = _mm256_set1_ps(d1);
            __m256 vm1 = _mm256_set1_ps(m1);
            __m256 vd2 = _mm256_set1_ps(d2);
            __m256 vm2 = _mm256_set1_ps(m2);

            v_total = _mm256_fmadd_ps(vd1, acc1, v_total);
            v_total = _mm256_fnmadd_ps(vm1, sumx1, v_total);
            v_total = _mm256_fmadd_ps(vd2, acc2, v_total);
            v_total = _mm256_fnmadd_ps(vm2, sumx2, v_total);

            q += 32;
            x_ptr += 64;
            is += 2;
        }
    }
    return hsum256_ps(v_total);
}

static inline float dot_q6_k_row(const block_q6_K* b_row, const float* x, int nb) {
    __m256 v_total = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const float d_val = fp16_to_fp32(b_row[b].d);
        const uint8_t* ql = b_row[b].ql;
        const uint8_t* qh = b_row[b].qh;
        const int8_t* sc = b_row[b].scales;
        const float* x_ptr = x + b * 256;

        for (int n_step = 0; n_step < 256; n_step += 128) {
            for (int half = 0; half < 2; ++half) {
                int is = half;
                int l_base = half * 16;

                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();

                for (int l_off = 0; l_off < 16; l_off += 8) {
                    int l = l_base + l_off;
                    __m128i ql_raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
                    __m128i ql_32 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l + 32));
                    __m128i qh_raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));

                    __m128i q1 = _mm_or_si128(_mm_and_si128(ql_raw, _mm_set1_epi8(0x0F)),
                                              _mm_slli_epi16(_mm_and_si128(qh_raw, _mm_set1_epi8(0x03)), 4));
                    q1 = _mm_sub_epi8(q1, _mm_set1_epi8(32));

                    __m128i q2 = _mm_or_si128(_mm_and_si128(ql_32, _mm_set1_epi8(0x0F)),
                                              _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 2), _mm_set1_epi8(0x03)), 4));
                    q2 = _mm_sub_epi8(q2, _mm_set1_epi8(32));

                    __m128i q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql_raw, 4), _mm_set1_epi8(0x0F)),
                                              _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 4), _mm_set1_epi8(0x03)), 4));
                    q3 = _mm_sub_epi8(q3, _mm_set1_epi8(32));

                    __m128i q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql_32, 4), _mm_set1_epi8(0x0F)),
                                              _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 6), _mm_set1_epi8(0x03)), 4));
                    q4 = _mm_sub_epi8(q4, _mm_set1_epi8(32));

                    __m256 vq1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1));
                    __m256 vq2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2));
                    __m256 vq3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3));
                    __m256 vq4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4));

                    acc1 = _mm256_fmadd_ps(vq1, _mm256_loadu_ps(x_ptr + l + 0), acc1);
                    acc2 = _mm256_fmadd_ps(vq2, _mm256_loadu_ps(x_ptr + l + 32), acc2);
                    acc3 = _mm256_fmadd_ps(vq3, _mm256_loadu_ps(x_ptr + l + 64), acc3);
                    acc4 = _mm256_fmadd_ps(vq4, _mm256_loadu_ps(x_ptr + l + 96), acc4);
                }

                __m256 vsc1 = _mm256_set1_ps(d_val * sc[is + 0]);
                __m256 vsc2 = _mm256_set1_ps(d_val * sc[is + 2]);
                __m256 vsc3 = _mm256_set1_ps(d_val * sc[is + 4]);
                __m256 vsc4 = _mm256_set1_ps(d_val * sc[is + 6]);

                v_total = _mm256_fmadd_ps(vsc1, acc1, v_total);
                v_total = _mm256_fmadd_ps(vsc2, acc2, v_total);
                v_total = _mm256_fmadd_ps(vsc3, acc3, v_total);
                v_total = _mm256_fmadd_ps(vsc4, acc4, v_total);
            }
            x_ptr += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return hsum256_ps(v_total);
}

void PureLlamaEngine::matmul(float* xout, const float* x, const void* w, GGMLType type, int n, int d, ThreadPool* pool) {
    if (!w) {
        std::memset(xout, 0, d * sizeof(float));
        return;
    }

    auto worker = [&](int start_row, int end_row) {
        if (type == GGMLType::F32) {
            const float* weights = static_cast<const float*>(w);
            for (int i = start_row; i < end_row; ++i) {
                xout[i] = dot_f32(weights + static_cast<size_t>(i) * n, x, n);
            }
        } else if (type == GGMLType::Q4_K) {
            const block_q4_K* blocks = static_cast<const block_q4_K*>(w);
            const int nb = n / 256;
            for (int i = start_row; i < end_row; ++i) {
                xout[i] = dot_q4_k_row(blocks + static_cast<size_t>(i) * nb, x, nb);
            }
        } else if (type == GGMLType::Q6_K) {
            const block_q6_K* blocks = static_cast<const block_q6_K*>(w);
            const int nb = n / 256;
            for (int i = start_row; i < end_row; ++i) {
                xout[i] = dot_q6_k_row(blocks + static_cast<size_t>(i) * nb, x, nb);
            }
        } else {
            for (int i = start_row; i < end_row; ++i) {
                xout[i] = 0.0f;
            }
        }
    };

    if (!pool || d <= 256) {
        worker(0, d);
    } else {
        pool->parallel_for(d, worker);
    }
}

void PureLlamaEngine::forward(int token, int pos, bool compute_logits) {
    int dim = arch_.embedding_dim;
    int hidden_dim = arch_.feed_forward_dim;
    int head_dim = dim / arch_.n_heads;
    int kv_dim = arch_.n_kv_heads * head_dim;
    int max_seq = std::min(arch_.context_length, 2048u);

    embed_token(token, state_.x.data());

    for (uint32_t l = 0; l < arch_.n_layers; ++l) {
        const auto& layer = weights_.layers[l];

        rmsnorm(state_.xb.data(), state_.x.data(), layer.attn_norm, dim);

        matmul(state_.q.data(), state_.xb.data(), layer.attn_q, layer.attn_q_type, dim, dim, pool_.get());
        matmul(state_.k.data(), state_.xb.data(), layer.attn_k, layer.attn_k_type, dim, kv_dim, nullptr);
        matmul(state_.v.data(), state_.xb.data(), layer.attn_v, layer.attn_v_type, dim, kv_dim, nullptr);

        for (int i = 0; i < dim; i += 2) {
            float freq = 1.0f / std::pow(arch_.rope_freq_base, static_cast<float>(i % head_dim) / static_cast<float>(head_dim));
            float val = pos * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);
            float q0 = state_.q[i];
            float q1 = state_.q[i + 1];
            state_.q[i] = q0 * fcr - q1 * fci;
            state_.q[i + 1] = q0 * fci + q1 * fcr;
            if (i < kv_dim) {
                float k0 = state_.k[i];
                float k1 = state_.k[i + 1];
                state_.k[i] = k0 * fcr - k1 * fci;
                state_.k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        int loff = l * max_seq * kv_dim;
        float* key_cache_row = state_.key_cache.data() + loff + pos * kv_dim;
        float* value_cache_row = state_.value_cache.data() + loff + pos * kv_dim;
        std::memcpy(key_cache_row, state_.k.data(), kv_dim * sizeof(float));
        std::memcpy(value_cache_row, state_.v.data(), kv_dim * sizeof(float));

        int kv_mul = arch_.n_heads / arch_.n_kv_heads;
        for (uint32_t h = 0; h < arch_.n_heads; ++h) {
            float* q = state_.q.data() + h * head_dim;
            float* att = state_.att.data() + h * max_seq;

            for (int t = 0; t <= pos; ++t) {
                float* k = state_.key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q[i] * k[i];
                }
                score /= std::sqrt(static_cast<float>(head_dim));
                att[t] = score;
            }

            float max_val = att[0];
            for (int t = 1; t <= pos; ++t) {
                if (att[t] > max_val) max_val = att[t];
            }
            float exp_sum = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                att[t] = std::exp(att[t] - max_val);
                exp_sum += att[t];
            }
            for (int t = 0; t <= pos; ++t) {
                att[t] /= exp_sum;
            }

            float* xb = state_.xb.data() + h * head_dim;
            std::memset(xb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                float* v = state_.value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att[t];
                for (int i = 0; i < head_dim; ++i) {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(state_.xb2.data(), state_.xb.data(), layer.attn_output, layer.attn_output_type, dim, dim, pool_.get());

        for (int i = 0; i < dim; ++i) {
            state_.x[i] += state_.xb2[i];
        }

        rmsnorm(state_.xb.data(), state_.x.data(), layer.ffn_norm, dim);

        matmul(state_.hb.data(), state_.xb.data(), layer.ffn_gate, layer.ffn_gate_type, dim, hidden_dim, pool_.get());
        matmul(state_.hb2.data(), state_.xb.data(), layer.ffn_up, layer.ffn_up_type, dim, hidden_dim, pool_.get());

        for (int i = 0; i < hidden_dim; ++i) {
            float val = state_.hb[i];
            val *= (1.0f / (1.0f + std::exp(-val))); // SwiGLU activation
            val *= state_.hb2[i];
            state_.hb[i] = val;
        }

        matmul(state_.xb.data(), state_.hb.data(), layer.ffn_down, layer.ffn_down_type, hidden_dim, dim, pool_.get());

        for (int i = 0; i < dim; ++i) {
            state_.x[i] += state_.xb[i];
        }
    }

    if (compute_logits) {
        rmsnorm(state_.x.data(), state_.x.data(), weights_.output_norm, dim);
        matmul(state_.logits.data(), state_.x.data(), weights_.output_weight, weights_.output_weight_type, dim, arch_.vocab_size, pool_.get());
    }
}

int PureLlamaEngine::sample(float temperature, float top_p, int top_k, float repeat_penalty, const std::vector<int32_t>& recent_tokens) {
    if (state_.logits.empty()) return 0;

    std::vector<float> logits = state_.logits;

    // Apply repetition penalty
    if (repeat_penalty != 1.0f && !recent_tokens.empty()) {
        std::unordered_set<int32_t> seen(recent_tokens.begin(), recent_tokens.end());
        for (int32_t tok : seen) {
            if (tok >= 0 && tok < static_cast<int32_t>(logits.size())) {
                if (logits[tok] > 0.0f) {
                    logits[tok] /= repeat_penalty;
                } else {
                    logits[tok] *= repeat_penalty;
                }
            }
        }
    }

    // Greedy sampling if temperature <= 0
    if (temperature <= 0.0f) {
        return static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    // Scale by temperature
    for (float& l : logits) {
        l /= temperature;
    }

    // Top-K filtering
    int k = (top_k > 0) ? std::min(top_k, static_cast<int>(logits.size())) : static_cast<int>(logits.size());
    struct TokenScore {
        int id;
        float score;
    };

    std::vector<TokenScore> candidates;
    candidates.reserve(logits.size());
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        candidates.push_back({i, logits[i]});
    }

    if (k < static_cast<int>(candidates.size())) {
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                          [](const TokenScore& a, const TokenScore& b) { return a.score > b.score; });
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end(),
                  [](const TokenScore& a, const TokenScore& b) { return a.score > b.score; });
    }

    // Softmax over top-k
    float max_s = candidates.front().score;
    float sum_exp = 0.0f;
    for (auto& c : candidates) {
        c.score = std::exp(c.score - max_s);
        sum_exp += c.score;
    }
    for (auto& c : candidates) {
        c.score /= sum_exp;
    }

    // Top-P (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            cum += candidates[i].score;
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
        float new_sum = 0.0f;
        for (const auto& c : candidates) new_sum += c.score;
        for (auto& c : candidates) c.score /= new_sum;
    }

    std::vector<float> probs;
    probs.reserve(candidates.size());
    for (const auto& c : candidates) {
        probs.push_back(c.score);
    }

    static std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int chosen_idx = dist(gen);
    return candidates[chosen_idx].id;
}

}
