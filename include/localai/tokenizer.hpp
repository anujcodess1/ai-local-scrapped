#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "gguf_reader.hpp"

namespace localai {

struct TokenData {
    std::string text;
    float score;
    int32_t type;
};

enum class TokenType : int32_t {
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

struct VocabStats {
    uint32_t total_tokens;
    uint32_t normal_tokens;
    uint32_t special_tokens;
    uint32_t byte_tokens;
    uint32_t unused_tokens;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t pad_token_id;
    std::string model_type;
};

class Tokenizer {
public:
    Tokenizer() = default;
    void load_from_gguf(const GGUFFile& gguf);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& tokens) const;
    std::string decode_token(int32_t token_id) const;

    int32_t bos_token() const { return bos_id_; }
    int32_t eos_token() const { return eos_id_; }
    int32_t pad_token() const { return pad_id_; }
    uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }

    VocabStats compute_stats() const;

    bool is_special_token(int32_t token_id) const;
    bool is_byte_token(int32_t token_id) const;
    int32_t token_to_id(const std::string& token) const;

private:
    std::vector<TokenData> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int> merge_ranks_;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    std::string tokenizer_model_;

    std::vector<int32_t> bpe_encode(const std::string& text) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
};

}
