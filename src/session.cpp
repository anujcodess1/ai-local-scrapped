#include "localai/session.hpp"
#include <sstream>
#include <iomanip>

namespace localai {

SessionManager::SessionManager()
    : active_(false) {}

void SessionManager::start_session(const std::string& model_name) {
    model_name_ = model_name;
    active_ = true;
    metrics_ = {};
    metrics_.session_start = std::chrono::system_clock::now();
    metrics_.last_activity = metrics_.session_start;
    prompt_builder_.clear_history();
}

void SessionManager::end_session() {
    active_ = false;
}

void SessionManager::record_generation(uint32_t input_tokens, uint32_t output_tokens, double inference_ms) {
    metrics_.total_tokens_consumed += input_tokens;
    metrics_.total_tokens_generated += output_tokens;
    metrics_.total_turns++;
    metrics_.total_inference_time_ms += inference_ms;
    metrics_.last_activity = std::chrono::system_clock::now();

    if (metrics_.total_inference_time_ms > 0) {
        metrics_.avg_tokens_per_second =
            (metrics_.total_tokens_generated * 1000.0) / metrics_.total_inference_time_ms;
    }
}

std::string SessionManager::format_metrics() const {
    std::ostringstream ss;

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        metrics_.last_activity - metrics_.session_start);

    ss << "session stats:\n"
       << "  model       : " << model_name_ << "\n"
       << "  turns       : " << metrics_.total_turns << "\n"
       << "  tokens in   : " << metrics_.total_tokens_consumed << "\n"
       << "  tokens out  : " << metrics_.total_tokens_generated << "\n"
       << "  avg speed   : " << std::fixed << std::setprecision(1)
       << metrics_.avg_tokens_per_second << " tok/s\n"
       << "  uptime      : " << duration.count() << "s\n";

    return ss.str();
}

void SessionManager::clear_context() {
    prompt_builder_.clear_history();
    metrics_.total_tokens_consumed = 0;
    metrics_.total_tokens_generated = 0;
}

}
