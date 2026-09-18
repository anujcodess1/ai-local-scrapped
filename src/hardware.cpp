#include "localai/hardware.hpp"
#include <thread>
#include <algorithm>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#include <cpuid.h>
#endif

namespace localai {

CPUInfo HardwareDetector::detect_cpu() {
    CPUInfo info;
    info.logical_threads = std::max(1u, std::thread::hardware_concurrency());
    info.physical_cores = std::max(1u, info.logical_threads / 2);
    info.brand = "Intel Core i7-8650U";

#if defined(_WIN32)
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 1) {
        __cpuid(cpuInfo, 1);
        info.has_avx = (cpuInfo[2] & (1 << 28)) != 0;
        info.has_fma = (cpuInfo[2] & (1 << 12)) != 0;
    }

    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        info.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    }
#else
    info.has_avx = true;
    info.has_avx2 = true;
    info.has_fma = true;
#endif

    return info;
}

MemoryInfo HardwareDetector::detect_memory() {
    MemoryInfo mem;

#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        mem.total_ram_bytes = status.ullTotalPhys;
        mem.available_ram_bytes = status.ullAvailPhys;
        mem.total_ram_gb = static_cast<double>(status.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
        mem.available_ram_gb = static_cast<double>(status.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
    }
#else
    mem.total_ram_gb = 8.0;
    mem.available_ram_gb = 4.0;
#endif

    return mem;
}

int HardwareDetector::recommend_threads(const CPUInfo& cpu) {
    if (cpu.logical_threads <= 4) {
        return std::max(1, static_cast<int>(cpu.logical_threads) - 1);
    }
    return std::max(2, static_cast<int>(cpu.logical_threads) - 2);
}

}
