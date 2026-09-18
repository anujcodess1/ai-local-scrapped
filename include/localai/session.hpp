#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include "prompt_builder.hpp"

namespace localai {

struct SessionMetrics {
    uint32_t total_tokens_generated;
    uint32_t total_tokens_consumed;
    uint32_t total_turns;
    double total_inference_time_ms;
    double avg_tokens_per_second;
    std::chrono::system_clock::time_point session_start;
    std::chrono::system_clock::time_point last_activity;
};

struct GenerationConfig {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
    int32_t max_tokens = 512;
    float min_p = 0.05f;
    uint32_t seed = 0;
};

class SessionManager {
public:
    SessionManager();

    void start_session(const std::string& model_name);
    void end_session();

    void record_generation(uint32_t input_tokens, uint32_t output_tokens, double inference_ms);

    PromptBuilder& prompt_builder() { return prompt_builder_; }
    const PromptBuilder& prompt_builder() const { return prompt_builder_; }

    GenerationConfig& generation_config() { return gen_config_; }
    const GenerationConfig& generation_config() const { return gen_config_; }

    SessionMetrics get_metrics() const { return metrics_; }
    std::string format_metrics() const;

    bool is_active() const { return active_; }
    const std::string& model_name() const { return model_name_; }

    void clear_context();

private:
    bool active_;
    std::string model_name_;
    PromptBuilder prompt_builder_;
    GenerationConfig gen_config_;
    SessionMetrics metrics_;
};

}
