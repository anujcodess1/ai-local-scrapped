#pragma once

#include <string>
#include <string_view>
#include "hardware.hpp"
#include "memory_guard.hpp"
#include "tensor_info.hpp"
#include "tokenizer.hpp"

namespace localai {

namespace colors {
    inline constexpr std::string_view reset   = "\033[0m";
    inline constexpr std::string_view bold    = "\033[1m";
    inline constexpr std::string_view cyan    = "\033[36m";
    inline constexpr std::string_view green   = "\033[32m";
    inline constexpr std::string_view yellow  = "\033[33m";
    inline constexpr std::string_view red     = "\033[31m";
    inline constexpr std::string_view blue    = "\033[34m";
    inline constexpr std::string_view gray    = "\033[90m";
    inline constexpr std::string_view magenta = "\033[35m";
    inline constexpr std::string_view white   = "\033[97m";
}

class TerminalUI {
public:
    static void init_terminal();
    static void print_banner();
    static void print_system_status(const CPUInfo& cpu, const MemoryInfo& mem, const MemoryAssessment& assess);
    static void print_model_info(const ModelArchitecture& arch, const TensorStats& stats);
    static void print_vocab_info(const VocabStats& vocab);
    static void print_prompt_prefix();
    static void print_ai_prefix();
    static void print_help();
    static void print_info(const std::string& message);
    static void print_error(const std::string& message);
    static void print_session_stats(const std::string& stats);
};

}
