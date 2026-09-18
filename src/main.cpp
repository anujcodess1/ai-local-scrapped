#include <iostream>
#include <filesystem>
#include "localai/hardware.hpp"
#include "localai/memory_guard.hpp"
#include "localai/config.hpp"
#include "localai/terminal_ui.hpp"
#include "localai/engine.hpp"
#include "localai/gguf_reader.hpp"
#include "localai/tokenizer.hpp"
#include "localai/tensor_info.hpp"
#include "localai/session.hpp"

namespace fs = std::filesystem;
using namespace localai;

int main(int argc, char* argv[]) {
    TerminalUI::init_terminal();
    TerminalUI::print_banner();

    fs::path base_dir = fs::current_path();
    if (!fs::exists(base_dir / "models") && fs::exists(base_dir.parent_path() / "models")) {
        base_dir = base_dir.parent_path();
    }

    CPUInfo cpu = HardwareDetector::detect_cpu();
    MemoryInfo mem = HardwareDetector::detect_memory();

    EngineConfig config = ConfigManager::load(base_dir);
    if (config.threads <= 0) {
        config.threads = HardwareDetector::recommend_threads(cpu);
    }

    MemoryAssessment assessment = MemoryGuard::evaluate(config.model_path, config.context_size, mem);
    if (!config.mode.empty()) {
        assessment.recommended_mode = config.mode;
        if (config.context_size > 0) {
            assessment.recommended_context = config.context_size;
        }
    }
    config.context_size = assessment.recommended_context;
    config.mode = assessment.recommended_mode;

    TerminalUI::print_system_status(cpu, mem, assessment);

    if (!fs::exists(config.model_path)) {
        TerminalUI::print_error("Model not found at: " + config.model_path.string());
        std::cerr << "Run: python scripts/download_model.py\n";
        return 1;
    }

    if (!fs::exists(config.runtime_path)) {
        fs::path alt_path = base_dir / "bin" / "llama-cli.exe";
        if (!fs::exists(alt_path)) {
            alt_path = base_dir / "runtime" / "llama-cli.exe";
        }
        if (fs::exists(alt_path)) {
            config.runtime_path = alt_path;
        }
    }

    GGUFFile gguf;
    Tokenizer tokenizer;
    bool weights_loaded = false;

    try {
        gguf = GGUFReader::parse(config.model_path);
        weights_loaded = true;
    } catch (const std::exception& e) {
        TerminalUI::print_error(std::string("GGUF parse failed: ") + e.what());
    }

    if (weights_loaded) {
        ModelArchitecture arch = TensorInspector::extract_architecture(gguf);
        TensorStats stats = TensorInspector::analyze_tensors(gguf);

        TerminalUI::print_model_info(arch, stats);

        try {
            tokenizer.load_from_gguf(gguf);
            VocabStats vocab = tokenizer.compute_stats();
            TerminalUI::print_vocab_info(vocab);
        } catch (...) {
            TerminalUI::print_info("Vocab loading skipped");
        }
    }

    SessionManager session;
    std::string model_name = gguf.get_string("general.name", "Llama-3.2-1B");
    session.start_session(model_name);

    InferenceEngine engine(config, weights_loaded ? &tokenizer : nullptr);

    if (argc > 1) {
        std::string prompt;
        for (int i = 1; i < argc; ++i) {
            if (i > 1) prompt += " ";
            prompt += argv[i];
        }
        return engine.execute_single_prompt(prompt) ? 0 : 1;
    }

    return engine.run_interactive_session();
}
