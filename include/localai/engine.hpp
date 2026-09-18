#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include "config.hpp"
#include "prompt_builder.hpp"
#include "tokenizer.hpp"
#include "model_forward.hpp"

namespace localai {

class InferenceEngine {
public:
    explicit InferenceEngine(EngineConfig config, const Tokenizer* tokenizer = nullptr);
    int run_interactive_session();
    bool execute_single_prompt(const std::string& prompt);

private:
    EngineConfig config_;
    Tokenizer tokenizer_;
    PureLlamaEngine pure_engine_;
    bool use_pure_engine_ = false;

    std::vector<ChatMessage> history_;
    std::string format_chat_prompt() const;
    bool run_inference_turn(const std::string& prompt, std::string& response_out);
    bool run_pure_inference(const std::string& prompt, std::string& response_out);
    bool run_subprocess_inference(const std::string& prompt, std::string& response_out);
    bool should_filter_line(const std::string& line) const;
};

}
