#include "localai/terminal_ui.hpp"
#include <iostream>
#include <iomanip>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace localai {

// Custom UI identity: turns carry exactly two labels - `user` and `agent>`.
namespace {
inline constexpr std::string_view kUserLabel = "user";
inline constexpr std::string_view kAgentLabel = "agent>";
}

void TerminalUI::init_terminal() {
#if defined(_WIN32)
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }

    HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
    if (hIn != INVALID_HANDLE_VALUE) {
        DWORD inMode = 0;
        if (GetConsoleMode(hIn, &inMode)) {
            inMode |= (ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT | ENABLE_INSERT_MODE | ENABLE_QUICK_EDIT_MODE);
            SetConsoleMode(hIn, inMode);
        }
    }

    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
}

void TerminalUI::print_banner() {
    std::cout << colors::bold << colors::cyan << "LocalAI" << colors::reset
              << colors::gray << "  ·  local reasoning engine" << colors::reset << "\n";
}

void TerminalUI::print_system_status(const CPUInfo& cpu, const MemoryInfo& mem, const MemoryAssessment& assess) {
    std::cout << colors::gray << "hardware: " << colors::reset
              << cpu.brand << " | "
              << cpu.logical_threads << "T | "
              << std::fixed << std::setprecision(1) << mem.available_ram_gb << "/"
              << mem.total_ram_gb << " GB ram"
              << colors::gray << "  ·  " << assess.recommended_mode << colors::reset << "\n";
}

void TerminalUI::print_model_info(const ModelArchitecture& arch, const TensorStats& stats) {
    std::cout << colors::gray << "model:    " << colors::reset
              << arch.arch_type
              << " | " << arch.n_layers << " layers"
              << " | " << arch.n_heads << " heads"
              << " | " << arch.embedding_dim << " dim"
              << " | " << arch.quantization_type << "\n";

    std::cout << colors::gray << "weights:  " << colors::reset
              << TensorInspector::format_param_count(stats.total_params) << " params"
              << " | " << stats.n_tensors << " tensors"
              << " | " << TensorInspector::format_byte_size(stats.total_bytes) << "\n";
}

void TerminalUI::print_vocab_info(const VocabStats& vocab) {
    std::cout << colors::gray << "vocab:    " << colors::reset
              << vocab.total_tokens << " tokens"
              << " | " << vocab.special_tokens << " special"
              << " | bos=" << vocab.bos_token_id
              << " eos=" << vocab.eos_token_id << "\n";
}

void TerminalUI::print_prompt_prefix() {
    std::cout << colors::bold << colors::white << kUserLabel << colors::reset << " ";
    std::cout.flush();
}

void TerminalUI::print_ai_prefix() {
    std::cout << "\n" << colors::bold << colors::green << kAgentLabel << colors::reset << " ";
    std::cout.flush();
}

void TerminalUI::print_help() {
    std::cout << "\n" << colors::gray << "commands" << colors::reset << "\n"
              << "  " << colors::bold << "/exit" << colors::reset  << "   quit the session\n"
              << "  " << colors::bold << "/clear" << colors::reset << "  wipe conversation memory\n"
              << "  " << colors::bold << "/help" << colors::reset  << "   show this list\n\n";
}

void TerminalUI::print_info(const std::string& message) {
    std::cout << colors::gray << "· " << message << colors::reset << "\n\n";
}

void TerminalUI::print_error(const std::string& message) {
    std::cerr << colors::bold << colors::red << "error" << colors::reset
              << colors::red << "  " << message << colors::reset << "\n";
}

void TerminalUI::print_session_stats(const std::string& stats) {
    std::cout << colors::gray << stats << colors::reset << "\n";
}

}
