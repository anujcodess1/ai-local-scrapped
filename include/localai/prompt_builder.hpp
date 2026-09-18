#pragma once

#include <string>
#include <vector>
#include "tokenizer.hpp"

namespace localai {

struct ChatMessage {
    std::string role;
    std::string content;
};

// Strong default persona: deliberate reasoning + precise, varied vocabulary.
inline const char* kDefaultSystemPrompt =
    "You are a rigorous, articulate reasoning assistant.\n"
    "Before answering: silently work through the problem step by step - identify what is "
    "actually being asked, break it into sub-problems, consider edge cases, and verify your "
    "conclusion before stating it.\n"
    "When you answer:\n"
    "1. State your reasoning explicitly: give the chain of thought that led to your conclusion, "
    "then the conclusion itself. Prefer structure (short numbered steps or clear paragraphs) over walls of text.\n"
    "2. Use precise, varied vocabulary. Choose the exact word, not the familiar one; avoid repeating "
    "the same adjective or phrase twice in one answer. No filler, no hedging cliches like 'it is "
    "important to note'.\n"
    "3. Be substantive. If you are uncertain, say precisely what is uncertain and why, then give your "
    "best-supported answer anyway.\n"
    "4. Never invent facts. If you lack the information, say so plainly.";

class PromptBuilder {
public:
    PromptBuilder() = default;

    void set_system_prompt(const std::string& prompt);
    void add_message(const std::string& role, const std::string& content);
    void clear_history();

    std::string build_llama3_prompt() const;
    std::string build_chatml_prompt() const;
    std::string build_raw_prompt() const;

    std::string build(const std::string& format = "llama3") const;

    std::vector<int32_t> build_tokenized(const Tokenizer& tokenizer, const std::string& format = "llama3") const;

    size_t message_count() const { return messages_.size(); }
    const std::vector<ChatMessage>& messages() const { return messages_; }

    uint32_t estimate_token_count(const Tokenizer& tokenizer) const;

    void trim_to_fit(const Tokenizer& tokenizer, uint32_t max_tokens);

private:
    std::string system_prompt_;
    std::vector<ChatMessage> messages_;
};

}
