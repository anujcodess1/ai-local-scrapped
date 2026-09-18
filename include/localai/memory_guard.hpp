#pragma once

#include <filesystem>
#include <cstdint>
#include "hardware.hpp"

namespace localai {

struct MemoryAssessment {
    bool safe = true;
    int recommended_context = 2048;
    std::string recommended_mode = "Balanced";
    double estimated_usage_mb = 0.0;
    double available_mb = 0.0;
};

class MemoryGuard {
public:
    static MemoryAssessment evaluate(const std::filesystem::path& model_path, int requested_context, const MemoryInfo& mem);
    static double estimate_footprint_mb(uint64_t model_bytes, int context_size);
};

}
