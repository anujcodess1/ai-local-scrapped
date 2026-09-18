#include "localai/engine.hpp"
#include "localai/terminal_ui.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <locale>

#if defined(_WIN32)
#define popen_cmd _popen
#define pclose_cmd _pclose
#else
#define popen_cmd popen
#define pclose_cmd pclose
#endif

namespace fs = std::filesystem;

namespace localai {

namespace {

std::string format_float(float value) {
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << std::fixed << std::setprecision(2) << value;
    return ss.str();
}

} // namespace

InferenceEngine::InferenceEngine(EngineConfig config, const Tokenizer* tokenizer)
    : config_(std::move(config)) {
    if (tokenizer) {
        tokenizer_ = *tokenizer;
    } else {
        try {
            GGUFFile gguf = GGUFReader::parse(config_.model_path);
            tokenizer_.load_from_gguf(gguf);
        } catch (...) {
        }
    }

    pure_engine_.set_threads(config_.threads);
    if (pure_engine_.load_model(config_.model_path)) {
        use_pure_engine_ = true;
    }
}

std::string InferenceEngine::format_chat_prompt() const {
    std::string prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
    if (!config_.system_prompt.empty()) {
        prompt += config_.system_prompt;
    } else {
        prompt += "You are LocalAI, a thoughtful, articulate, and intelligent AI assistant. Think clearly, reason step-by-step when appropriate, and respond in helpful, well-structured natural language.";
    }
    prompt += "<|eot_id|>";

    for (const auto& msg : history_) {
        prompt += "<|start_header_id|>" + msg.role + "<|end_header_id|>\n\n";
        prompt += msg.content + "<|eot_id|>";
    }
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt;
}

bool InferenceEngine::should_filter_line(const std::string& line) const {
    if (line.empty()) return true;
    if (line.find("Loading model") != std::string::npos) return true;
    if (line.find("build") != std::string::npos && line.find(":") != std::string::npos) return true;
    if (line.find("model") != std::string::npos && line.find(":") != std::string::npos) return true;
    if (line.find("ftype") != std::string::npos && line.find(":") != std::string::npos) return true;
    if (line.find("modalities") != std::string::npos && line.find(":") != std::string::npos) return true;
    if (line.find("available commands") != std::string::npos) return true;
    if (line.find("/exit") != std::string::npos) return true;
    if (line.find("/clear") != std::string::npos) return true;
    if (line.find("/regen") != std::string::npos) return true;
    if (line.find("/read") != std::string::npos) return true;
    if (line.find("/glob") != std::string::npos) return true;
    if (line.find("<|") != std::string::npos) return true;
    if (line.find("\xe2\x96") != std::string::npos) return true;
    if (line.find("\xe2\x96\x88") != std::string::npos) return true;
    if (line.find("\xe2\x96\x84") != std::string::npos) return true;
    if (line.find("\xe2\x96\x80") != std::string::npos) return true;
    if (line.rfind("> ", 0) == 0) return true;

    size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return true;

    return false;
}

bool InferenceEngine::run_pure_inference(const std::string& prompt, std::string& response_out) {
    std::vector<int32_t> tokens = tokenizer_.encode(prompt);
    if (tokens.empty()) {
        tokens.push_back(tokenizer_.bos_token() >= 0 ? tokenizer_.bos_token() : 128000);
    }

    pure_engine_.reset_cache();

    // Ingest prompt tokens into KV-cache (computing logits only on the final token)
    for (size_t i = 0; i < tokens.size(); ++i) {
        bool is_last = (i == tokens.size() - 1);
        pure_engine_.forward(tokens[i], static_cast<int>(i), is_last);
    }

    TerminalUI::print_ai_prefix();

    int cur_pos = static_cast<int>(tokens.size());
    int max_tokens = config_.max_tokens > 0 ? config_.max_tokens : 512;
    std::string full_response;

    int eos1 = tokenizer_.eos_token();
    int eos2 = 128001; // <|end_of_text|>
    int eos3 = 128009; // <|eot_id|>

    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(max_tokens);

    for (int step = 0; step < max_tokens; ++step) {
        int next_token = pure_engine_.sample(
            config_.temperature,
            config_.top_p,
            config_.top_k > 0 ? config_.top_k : 40,
            config_.repeat_penalty > 0.0f ? config_.repeat_penalty : 1.1f,
            generated_tokens
        );
        if (next_token == eos1 || next_token == eos2 || next_token == eos3) {
            break;
        }

        generated_tokens.push_back(next_token);
        std::string token_str = tokenizer_.decode_token(next_token);
        std::cout << token_str << std::flush;
        full_response += token_str;

        pure_engine_.forward(next_token, cur_pos++);
    }

    std::cout << "\n";
    response_out = full_response;
    return true;
}

bool InferenceEngine::run_subprocess_inference(const std::string& prompt, std::string& response_out) {
    fs::path base_dir = config_.model_path.parent_path();
    while (base_dir.has_parent_path() && base_dir.filename() != "LocalAI" && !fs::exists(base_dir / "cache")) {
        if (fs::exists(base_dir / "models")) break;
        base_dir = base_dir.parent_path();
    }
    fs::path prompt_file = base_dir / "cache" / "prompt.tmp";
    fs::create_directories(prompt_file.parent_path());

    {
        std::ofstream ofs(prompt_file, std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs << prompt;
    }

    std::string cmd = "\"" + config_.runtime_path.string() + "\"";
    cmd += " -m \"" + config_.model_path.string() + "\"";
    cmd += " -t " + std::to_string(config_.threads);
    cmd += " -c " + std::to_string(config_.context_size);
    cmd += " -b " + std::to_string(config_.batch_size);

    cmd += " --temp " + format_float(config_.temperature);
    cmd += " --top-p " + format_float(config_.top_p);
    cmd += " --top-k " + std::to_string(config_.top_k);
    cmd += " --min-p " + format_float(config_.min_p);
    cmd += " --repeat-penalty " + format_float(config_.repeat_penalty);
    cmd += " --repeat-last-n " + std::to_string(config_.repeat_last_n);
    cmd += " -n " + std::to_string(config_.max_tokens);
    cmd += " --log-disable --no-perf --simple-io --single-turn --no-jinja --no-display-prompt -f \"" + prompt_file.string() + "\" 2>&1";

    std::string wrapped_cmd = "\"" + cmd + "\"";

    FILE* pipe = popen_cmd(wrapped_cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[512];
    std::string line_buffer;
    bool ai_started = false;
    bool ai_prefix_printed = false;
    std::string full_response;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        line_buffer += buffer;
        size_t newline_pos;
        while ((newline_pos = line_buffer.find('\n')) != std::string::npos) {
            std::string line = line_buffer.substr(0, newline_pos);
            line_buffer.erase(0, newline_pos + 1);

            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            if (!ai_started) {
                if (should_filter_line(line)) {
                    continue;
                }
                ai_started = true;
            }

            if (line.rfind("[ Prompt:", 0) == 0 || line.rfind("Exiting", 0) == 0) {
                continue;
            }

            if (!ai_prefix_printed) {
                TerminalUI::print_ai_prefix();
                ai_prefix_printed = true;
            }

            std::cout << line << "\n";
            std::cout.flush();
            if (!full_response.empty()) full_response += "\n";
            full_response += line;
        }
    }

    if (!line_buffer.empty()) {
        if (line_buffer.back() == '\r') line_buffer.pop_back();
        if (ai_started && line_buffer.rfind("[ Prompt:", 0) != 0 && line_buffer.rfind("Exiting", 0) != 0) {
            if (!ai_prefix_printed) {
                TerminalUI::print_ai_prefix();
                ai_prefix_printed = true;
            }
            std::cout << line_buffer << "\n";
            std::cout.flush();
            if (!full_response.empty()) full_response += "\n";
            full_response += line_buffer;
        }
    }

    pclose_cmd(pipe);
    response_out = full_response;
    return !full_response.empty();
}

bool InferenceEngine::run_inference_turn(const std::string& prompt, std::string& response_out) {
    if (use_pure_engine_) {
        return run_pure_inference(prompt, response_out);
    }
    return run_subprocess_inference(prompt, response_out);
}

int InferenceEngine::run_interactive_session() {
    TerminalUI::print_info("Interactive session active. Type /exit to quit, /clear to reset context.");

    std::string input;
    while (true) {
        TerminalUI::print_prompt_prefix();
        if (!std::getline(std::cin, input)) {
            if (std::cin.eof()) {
                break;
            }
            std::cin.clear();
            continue;
        }

        size_t first = input.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            continue;
        }
        std::string trimmed = input.substr(first);
        size_t last = trimmed.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) {
            trimmed = trimmed.substr(0, last + 1);
        }

        if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"') {
            trimmed = trimmed.substr(1, trimmed.size() - 2);
        }

        if (trimmed == "/exit" || trimmed == "exit" || trimmed == "quit" || trimmed == ":q") {
            TerminalUI::print_info("Goodbye.");
            break;
        }

        if (trimmed == "/clear" || trimmed == "clear") {
            history_.clear();
            if (use_pure_engine_) {
                pure_engine_.reset_cache();
            }
            TerminalUI::print_info("Context cleared.");
            continue;
        }

        if (trimmed == "/help" || trimmed == "help") {
            TerminalUI::print_help();
            continue;
        }

        if (history_.size() > 16) {
            history_.erase(history_.begin(), history_.begin() + 2);
        }

        history_.push_back({"user", trimmed});
        std::string full_prompt = format_chat_prompt();
        std::string response;

        if (!run_inference_turn(full_prompt, response)) {
            TerminalUI::print_error("Failed to generate response.");
            if (!history_.empty()) {
                history_.pop_back();
            }
        } else {
            history_.push_back({"assistant", response});
        }
        std::cout << "\n";
    }

    return 0;
}

bool InferenceEngine::execute_single_prompt(const std::string& prompt) {
    history_.push_back({"user", prompt});
    std::string full_prompt = format_chat_prompt();
    std::string response;
    return run_inference_turn(full_prompt, response);
}

}
