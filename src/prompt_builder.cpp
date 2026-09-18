#include "localai/prompt_builder.hpp"
#include <sstream>

namespace localai {

void PromptBuilder::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
}

void PromptBuilder::add_message(const std::string& role, const std::string& content) {
    messages_.push_back({role, content});
}

void PromptBuilder::clear_history() {
    messages_.clear();
}

std::string PromptBuilder::build_llama3_prompt() const {
    std::ostringstream ss;

    ss << "<|begin_of_text|>";

    if (!system_prompt_.empty()) {
        ss << "<|start_header_id|>system<|end_header_id|>\n\n"
           << system_prompt_ << "<|eot_id|>";
    }

    for (const auto& msg : messages_) {
        ss << "<|start_header_id|>" << msg.role << "<|end_header_id|>\n\n"
           << msg.content << "<|eot_id|>";
    }

    ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";

    return ss.str();
}

std::string PromptBuilder::build_chatml_prompt() const {
    std::ostringstream ss;

    if (!system_prompt_.empty()) {
        ss << "<|im_start|>system\n" << system_prompt_ << "<|im_end|>\n";
    }

    for (const auto& msg : messages_) {
        ss << "<|im_start|>" << msg.role << "\n"
           << msg.content << "<|im_end|>\n";
    }

    ss << "<|im_start|>assistant\n";

    return ss.str();
}

std::string PromptBuilder::build_raw_prompt() const {
    std::ostringstream ss;

    if (!system_prompt_.empty()) {
        ss << "System: " << system_prompt_ << "\n\n";
    }

    for (const auto& msg : messages_) {
        if (msg.role == "user") {
            ss << "User: " << msg.content << "\n";
        } else {
            ss << "Assistant: " << msg.content << "\n";
        }
    }

    ss << "Assistant: ";
    return ss.str();
}

std::string PromptBuilder::build(const std::string& format) const {
    if (format == "llama3") return build_llama3_prompt();
    if (format == "chatml") return build_chatml_prompt();
    return build_raw_prompt();
}

std::vector<int32_t> PromptBuilder::build_tokenized(const Tokenizer& tokenizer, const std::string& format) const {
    std::string prompt = build(format);
    return tokenizer.encode(prompt);
}

uint32_t PromptBuilder::estimate_token_count(const Tokenizer& tokenizer) const {
    std::string prompt = build();
    auto tokens = tokenizer.encode(prompt);
    return static_cast<uint32_t>(tokens.size());
}

void PromptBuilder::trim_to_fit(const Tokenizer& tokenizer, uint32_t max_tokens) {
    while (messages_.size() > 2) {
        uint32_t count = estimate_token_count(tokenizer);
        if (count <= max_tokens) break;
        messages_.erase(messages_.begin());
    }
}

}
