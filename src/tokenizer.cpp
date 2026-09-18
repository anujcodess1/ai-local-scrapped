#include "localai/tokenizer.hpp"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <regex>

namespace localai {

void Tokenizer::load_from_gguf(const GGUFFile& gguf) {
    tokenizer_model_ = gguf.get_string("tokenizer.ggml.model", "gpt2");

    auto tokens = gguf.get_string_array("tokenizer.ggml.tokens");
    auto scores_it = gguf.metadata_index.find("tokenizer.ggml.scores");
    auto types_it = gguf.metadata_index.find("tokenizer.ggml.token_type");

    vocab_.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        vocab_[i].text = tokens[i];
        vocab_[i].score = 0.0f;
        vocab_[i].type = 1;
        token_to_id_[tokens[i]] = static_cast<int32_t>(i);
    }

    if (scores_it != gguf.metadata_index.end()) {
        const auto& val = gguf.metadata[scores_it->second].value;
        if (auto* arr = std::get_if<std::vector<int32_t>>(&val)) {
            for (size_t i = 0; i < arr->size() && i < vocab_.size(); ++i) {
                vocab_[i].score = static_cast<float>((*arr)[i]);
            }
        }
    }

    if (types_it != gguf.metadata_index.end()) {
        const auto& val = gguf.metadata[types_it->second].value;
        if (auto* arr = std::get_if<std::vector<int32_t>>(&val)) {
            for (size_t i = 0; i < arr->size() && i < vocab_.size(); ++i) {
                vocab_[i].type = (*arr)[i];
            }
        }
    }

    auto merges = gguf.get_string_array("tokenizer.ggml.merges");
    merge_ranks_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        merge_ranks_[merges[i]] = static_cast<int>(i);
    }

    bos_id_ = static_cast<int32_t>(gguf.get_int("tokenizer.ggml.bos_token_id", 128000));
    eos_id_ = static_cast<int32_t>(gguf.get_int("tokenizer.ggml.eos_token_id", 128001));
    pad_id_ = static_cast<int32_t>(gguf.get_int("tokenizer.ggml.padding_token_id", -1));
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> chunks;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                chunks.push_back(current);
                current.clear();
            }
            current += c;
        } else if (std::ispunct(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                chunks.push_back(current);
                current.clear();
            }
            current += c;
            chunks.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        chunks.push_back(current);
    }
    return chunks;
}

std::vector<int32_t> Tokenizer::bpe_encode(const std::string& text) const {
    if (text.empty()) return {};

    auto it = token_to_id_.find(text);
    if (it != token_to_id_.end()) {
        return {it->second};
    }

    std::vector<std::string> symbols;
    for (size_t i = 0; i < text.size(); ) {
        uint8_t byte = static_cast<uint8_t>(text[i]);
        size_t char_len = 1;
        if (byte >= 0xF0) char_len = 4;
        else if (byte >= 0xE0) char_len = 3;
        else if (byte >= 0xC0) char_len = 2;
        symbols.push_back(text.substr(i, char_len));
        i += char_len;
    }

    while (symbols.size() > 1) {
        int best_rank = 1000000000;
        size_t best_idx = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string pair = symbols[i] + " " + symbols[i + 1];
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                found = true;
            }
        }

        if (!found) break;

        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto found = token_to_id_.find(sym);
        if (found != token_to_id_.end()) {
            ids.push_back(found->second);
        } else {
            for (uint8_t byte : sym) {
                std::string byte_token = "<0x" + std::string(1, "0123456789ABCDEF"[byte >> 4]) +
                                         std::string(1, "0123456789ABCDEF"[byte & 0xF]) + ">";
                auto byte_it = token_to_id_.find(byte_token);
                if (byte_it != token_to_id_.end()) {
                    ids.push_back(byte_it->second);
                }
            }
        }
    }
    return ids;
}

static const auto s_byte_to_unicode = []() {
    std::unordered_map<uint8_t, std::string> m;
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool in_bs = false;
        for (int x : bs) {
            if (x == b) { in_bs = true; break; }
        }
        if (in_bs) {
            m[static_cast<uint8_t>(b)] = std::string(1, static_cast<char>(b));
        } else {
            uint32_t cp = 256 + n;
            n++;
            std::string s;
            if (cp < 320) {
                s.push_back(static_cast<char>(0xC4));
                s.push_back(static_cast<char>(0x80 + (cp - 256)));
            } else {
                s.push_back(static_cast<char>(0xC5));
                s.push_back(static_cast<char>(0x80 + (cp - 320)));
            }
            m[static_cast<uint8_t>(b)] = s;
        }
    }
    return m;
}();

static std::string byte_encode_text(const std::string& text) {
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) {
        auto it = s_byte_to_unicode.find(c);
        if (it != s_byte_to_unicode.end()) {
            result += it->second;
        } else {
            result.push_back(static_cast<char>(c));
        }
    }
    return result;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> all_ids;
    size_t pos = 0;

    while (pos < text.size()) {
        if (text[pos] == '<' && pos + 1 < text.size() && text[pos + 1] == '|') {
            size_t end = text.find("|>", pos);
            if (end != std::string::npos) {
                std::string special = text.substr(pos, end + 2 - pos);
                auto it = token_to_id_.find(special);
                if (it != token_to_id_.end()) {
                    all_ids.push_back(it->second);
                    pos = end + 2;
                    continue;
                }
            }
        }

        size_t next_special = text.find("<|", pos);
        std::string regular = (next_special == std::string::npos) ? text.substr(pos) : text.substr(pos, next_special - pos);
        if (!regular.empty()) {
            auto chunks = pre_tokenize(regular);
            for (const auto& chunk : chunks) {
                std::string byte_chunk = byte_encode_text(chunk);
                auto ids = bpe_encode(byte_chunk);
                all_ids.insert(all_ids.end(), ids.begin(), ids.end());
            }
        }

        if (next_special == std::string::npos) break;
        pos = next_special;
    }

    return all_ids;
}

static std::string clean_bpe_piece(const std::string& raw) {
    if (raw.empty()) return "";

    // Hex byte tokens like <0x0A> or <0x20>
    if (raw.size() == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        auto hex_val = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int h1 = hex_val(raw[3]);
        int h2 = hex_val(raw[4]);
        if (h1 >= 0 && h2 >= 0) {
            return std::string(1, static_cast<char>((h1 << 4) | h2));
        }
    }

    // Standard GPT-2 byte-to-unicode reverse mapping:
    // GPT-2 maps bytes 0..255 to printable Unicode characters.
    // Bytes 33..126, 161..172, 174..255 are mapped to themselves.
    // The remaining 68 bytes are mapped to code points 256..323:
    // b=0..32 map to 256..288 (b=32 ' ' -> 288 = 0x0120 'Ġ', b=10 '\n' -> 266 = 0x010A 'Ċ', b=9 '\t' -> 265 = 0x0109 'ĉ')
    // b=127..160 map to 289..322
    // b=173 maps to 323
    static const auto rev_map = []() {
        std::unordered_map<uint32_t, uint8_t> m;
        std::vector<int> bs;
        for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
        for (int i = 161; i <= 172; ++i) bs.push_back(i);
        for (int i = 174; i <= 255; ++i) bs.push_back(i);

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool in_bs = false;
            for (int x : bs) {
                if (x == b) { in_bs = true; break; }
            }
            if (!in_bs) {
                m[256 + n] = static_cast<uint8_t>(b);
                n++;
            }
        }
        return m;
    }();

    std::string result;
    result.reserve(raw.size());

    for (size_t i = 0; i < raw.size(); ) {
        uint8_t c = static_cast<uint8_t>(raw[i]);
        if (c == 0xC4 && i + 1 < raw.size()) {
            uint8_t c2 = static_cast<uint8_t>(raw[i + 1]);
            uint32_t cp = 256 + (c2 - 0x80);
            auto it = rev_map.find(cp);
            if (it != rev_map.end()) {
                result.push_back(static_cast<char>(it->second));
                i += 2;
                continue;
            }
        } else if (c == 0xC5 && i + 1 < raw.size()) {
            uint8_t c2 = static_cast<uint8_t>(raw[i + 1]);
            uint32_t cp = 320 + (c2 - 0x80);
            auto it = rev_map.find(cp);
            if (it != rev_map.end()) {
                result.push_back(static_cast<char>(it->second));
                i += 2;
                continue;
            }
        }
        result.push_back(raw[i]);
        i++;
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t id : tokens) {
        if (id >= 0 && id < static_cast<int32_t>(vocab_.size())) {
            if (vocab_[id].type == static_cast<int32_t>(TokenType::CONTROL)) continue;
            result += decode_token(id);
        }
    }
    return result;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id >= 0 && token_id < static_cast<int32_t>(vocab_.size())) {
        return clean_bpe_piece(vocab_[token_id].text);
    }
    return "";
}

bool Tokenizer::is_special_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(vocab_.size())) return false;
    return vocab_[token_id].type == static_cast<int32_t>(TokenType::CONTROL) ||
           vocab_[token_id].type == static_cast<int32_t>(TokenType::USER_DEFINED);
}

bool Tokenizer::is_byte_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(vocab_.size())) return false;
    return vocab_[token_id].type == static_cast<int32_t>(TokenType::BYTE);
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    return -1;
}

VocabStats Tokenizer::compute_stats() const {
    VocabStats stats{};
    stats.total_tokens = static_cast<uint32_t>(vocab_.size());
    stats.bos_token_id = bos_id_;
    stats.eos_token_id = eos_id_;
    stats.pad_token_id = pad_id_;
    stats.model_type = tokenizer_model_;

    for (const auto& tok : vocab_) {
        switch (static_cast<TokenType>(tok.type)) {
            case TokenType::NORMAL: ++stats.normal_tokens; break;
            case TokenType::CONTROL:
            case TokenType::USER_DEFINED: ++stats.special_tokens; break;
            case TokenType::BYTE: ++stats.byte_tokens; break;
            case TokenType::UNUSED: ++stats.unused_tokens; break;
            default: ++stats.normal_tokens; break;
        }
    }
    return stats;
}

}
