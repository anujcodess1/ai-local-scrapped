#include "localai/memory_guard.hpp"
#include <system_error>

namespace localai {

double MemoryGuard::estimate_footprint_mb(uint64_t model_bytes, int context_size) {
    double model_mb = static_cast<double>(model_bytes) / (1024.0 * 1024.0);
    double kv_cache_mb = (static_cast<double>(context_size) / 2048.0) * 128.0;
    double runtime_overhead_mb = 180.0;
    return model_mb + kv_cache_mb + runtime_overhead_mb;
}

MemoryAssessment MemoryGuard::evaluate(const std::filesystem::path& model_path, int requested_context, const MemoryInfo& mem) {
    MemoryAssessment result;
    result.available_mb = mem.available_ram_gb * 1024.0;

    uint64_t model_bytes = 807694464ULL;
    std::error_code ec;
    if (std::filesystem::exists(model_path, ec)) {
        model_bytes = std::filesystem::file_size(model_path, ec);
    }

    result.estimated_usage_mb = estimate_footprint_mb(model_bytes, requested_context);
    double safety_buffer_mb = 400.0;

    if (result.available_mb >= (result.estimated_usage_mb + safety_buffer_mb) || requested_context >= 2048) {
        result.safe = true;
        result.recommended_context = (requested_context > 0) ? requested_context : 2048;
        result.recommended_mode = (result.recommended_context > 2048) ? "Performance" : "Medium RAM Mode";
    } else if (result.available_mb >= 500.0) {
        result.safe = true;
        result.recommended_context = 2048;
        result.recommended_mode = "Medium RAM Mode";
    } else {
        result.safe = true;
        result.recommended_context = (requested_context > 0) ? requested_context : 2048;
        result.recommended_mode = "Medium RAM Mode";
    }

    return result;
}

}
