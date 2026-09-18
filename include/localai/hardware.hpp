#pragma once

#include <string>
#include <cstdint>

namespace localai {

struct CPUInfo {
    std::string brand;
    uint32_t physical_cores = 4;
    uint32_t logical_threads = 8;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;
};

struct MemoryInfo {
    uint64_t total_ram_bytes = 0;
    uint64_t available_ram_bytes = 0;
    double total_ram_gb = 0.0;
    double available_ram_gb = 0.0;
};

class HardwareDetector {
public:
    static CPUInfo detect_cpu();
    static MemoryInfo detect_memory();
    static int recommend_threads(const CPUInfo& cpu);
};

}
