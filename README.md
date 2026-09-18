# LocalAI

<div align="center">

```text
  _                 _    _    ___ 
 | |   ___  __ __ _| |  /_\  |_ _|
 | |__/ _ \/ _/ _` | | / _ \  | | 
 |____\___/\__\__,_|_|/_/ \_\|___|
  High-Performance Local Reasoning Engine
```

[![Language](https://img.shields.io/badge/Language-C%2B%2B17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-555555?style=for-the-badge&logo=linux&logoColor=white)](#building-from-source)
[![SIMD](https://img.shields.io/badge/Acceleration-AVX2%20%2B%20FMA-D97706?style=for-the-badge)](#hardware-acceleration--simd-kernels)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20Native)-059669?style=for-the-badge)](#core-architecture)
[![Format](https://img.shields.io/badge/Model%20Format-GGUF%20(Llama--3)-7C3AED?style=for-the-badge)](#technical-specifications)
[![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)](LICENSE)

**A high-performance, completely native C++17 local AI reasoning engine.**  
Engineered from the ground up for low latency, zero memory allocations during generation, and total privacy on commodity CPUs without GPUs, Python runtimes, or external daemons.

[Key Capabilities](#key-capabilities) • [Architecture](#core-architecture) • [Quick Start](#quick-start) • [Interactive CLI](#interactive-cli-interface) • [Configuration](#configuration-reference) • [Building](#building-from-source) • [Specifications](#technical-specifications)

---

</div>

## Executive Summary

**LocalAI** executes transformer-based large language models directly in-process via optimized native C++17. By coupling custom 256-bit AVX2+FMA SIMD quantization kernels with a zero-allocation persistent threadpool, LocalAI delivers responsive local inference on standard desktop and mobile CPUs without requiring PyTorch, CUDA, Python, or background HTTP servers.

### Why LocalAI?

- **Zero Runtime Dependencies**: Single, standalone executable. No Python virtual environment, no external dynamic libraries, no server processes.
- **Hardware-Saturating SIMD**: Custom-vectorized `Q4_K` and `Q6_K` dot-product kernels unrolled with AVX2 and FMA instructions for maximum FLOP utilization.
- **Sub-Millisecond Thread Synchronization**: Generation-based persistent worker pool eliminates thread teardown and operating system scheduling overhead.
- **Native Byte-Level BPE**: Full support for Llama-3's 128,256-token vocabulary with unicode byte re-mapping, eliminating token artifacts (e.g. `Ġ` and `Ċ`).
- **Memory Safety & Guardrails**: Dynamic system memory profiling with **Medium RAM Mode** and strict KV-cache bounds to prevent system paging and OOM crashes.
- **100% Offline & Private**: Zero telemetry, zero external network requests, and absolute data isolation.

---

## Core Architecture

The diagram below illustrates the end-to-end inference flow through LocalAI's native subsystems:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     LOCALAI RUNTIME PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────┐       ┌──────────────────────┐       ┌───────────────────────────────────┐
│   GGUF Parser    │ ───>  │  Memory-Mapped File  │ ───>  │     Tensor Registry & Metadata    │
│ (Header & Tensors│       │   (Zero-Copy I/O)    │       │ (Llama-3 1B / Q4_K_M / Q6_K Layer)│
└──────────────────┘       └──────────────────────┘       └───────────────────────────────────┘
                                                                            │
          ┌─────────────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT PROCESSING                                          │
│  ┌───────────────────────────┐         ┌──────────────────────────────────────────────────┐ │
│  │   Llama-3 Chat Template   │  ───>   │             Byte-Level BPE Tokenizer             │ │
│  │  (System / User / Agent)  │         │ (128,256 Vocab, Merge Tree, Reverse Byte Mapper) │ │
│  └───────────────────────────┘         └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                               TRANSFORMER FORWARD ENGINE                                    │
│  ┌─────────────────────────────────┐         ┌────────────────────────────────────────────┐ │
│  │      Circular KV-Cache Bank     │  <───>  │       Multi-Head Self Attention            │ │
│  │ (2048 Context, Layer-Partition) │         │ (Rotary Positional Embeddings [RoPE], RMS) │ │
│  └─────────────────────────────────┘         └────────────────────────────────────────────┘ │
│                                                              │                              │
│                                                              ▼                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                       Persistent Zero-Allocation ThreadPool                            │ │
│  │  ┌────────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐ │ │
│  │  │ Thread 0 (Main Core)   │  │ Thread 1..N (Workers)  │  │ Atomic Generation Sync    │ │ │
│  │  └────────────────────────┘  └────────────────────────┘  └───────────────────────────┘ │ │
│  │                                                                                        │ │
│  │                 AVX2 + FMA SIMD Matrix-Vector Dot Product Kernels                      │ │
│  │        [dot_q4_k_row: 256-bit SIMD]       │      [dot_q6_k_row: 256-bit SIMD]          │ │
│  │        - 8-wide fused multiply-acc        │      - Dual 4-bit + 2-bit unpack           │ │
│  │        - Deferred horizontal sum          │      - Single-pass reduction               │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                              │
│                                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           SwiGLU Feed-Forward Network & Projections                    │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 NUCLEUS SAMPLING PIPELINE                                   │
│  ┌────────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │ Repetition Penalty Monitor │  │ Temperature & Top-K Cut  │  │ Nucleus Top-P (CDF Pass) │ │
│  │      (Recent N Tokens)     │  │        (Softmax)         │  │   (Mersenne Twister)     │ │
│  └────────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            VT100 STREAMING TERMINAL EMISSION                                │
│                   (Real-time Token Stream, ANSI Colors, Echo Buffering)                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

| Subsystem | Architectural Implementation |
| :--- | :--- |
| **Pure In-Process Engine** | Complete transformer forward pass implemented natively in C++17. Computes attention heads, RoPE positional embeddings, RMS normalization, SwiGLU activations, and quantized matrix multiplication without relying on external libraries or helper processes. |
| **Hardware SIMD (AVX2 + FMA)** | Hand-crafted 256-bit vectorized kernels for `Q4_K` and `Q6_K` quantized blocks. Uses `_mm256_fmadd_ps` to compute 8 parallel products per cycle with deferred horizontal summation, eliminating instruction stalls. |
| **Zero-Allocation ThreadPool** | Persistent worker threads spin on generation counters during matmul operations. Workloads are partitioned evenly across available physical and logical cores with zero runtime memory allocations. |
| **Byte-Level BPE Tokenizer** | Full support for Llama-3's 128,256-token vocabulary. Accurately maps special control tokens (`<|begin_of_text|>`, `<|eot_id|>`), pre-tokenizes UTF-8 sequences, and inverts GPT-2 byte encodings for clean string rendering. |
| **Nucleus Sampling Pipeline** | Comprehensive sampling pipeline featuring temperature scaling, Top-K pruning, Top-P (nucleus) cumulative mass filtering, and sliding-window repetition penalties. |
| **Adaptive Memory Guard** | Profiles host memory on startup and balances KV-cache memory allocation. Includes **Medium RAM Mode** (2048 context window, 256 batch size) for optimal throughput on 8 GB and 16 GB hardware. |
| **Interactive Terminal UI** | ANSI VT100 console interface with real-time token streaming, multi-turn conversational memory, line editing, and session management. |

---

## Quick Start

### 1. Download Model Weights

LocalAI is optimized for **Llama-3.2 1B Instruct GGUF** (`Q4_K_M`, ~807 MB).

Run the automated download script:
```bash
python scripts/download_model.py
```

*Custom download options:*
```bash
# Download to a custom path or specify a custom Hugging Face URL:
python scripts/download_model.py --dest models/1b --force
```

---

### 2. Run LocalAI

#### Interactive Conversational Mode
Launch the interactive terminal session with multi-turn conversation memory:
```powershell
.\bin\localai.exe
```

```text
LocalAI  ·  local reasoning engine
hardware: Intel Core i7-8650U | 8T | 1.1/7.9 GB ram  ·  Medium RAM Mode
model:    llama | 16 layers | 32 heads | 2048 dim | Q4_K_M
weights:  1.24B params | 147 tensors | 762.81 MB
vocab:    128256 tokens | 256 special | bos=128000 eos=128009
· Interactive session active. Type /exit to quit, /clear to reset context.

user hi
agent> Hello! How can I assist you today?

user What is the speed of light in m/s?
agent> The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 3.00 × 10^8 m/s).
```

#### Single-Turn Query Mode
Pass a prompt directly as a command-line argument for immediate, scripted execution:
```powershell
.\bin\localai.exe "Explain the difference between compiled and interpreted languages."
```

---

## Interactive CLI Interface

LocalAI features a dedicated interactive console interface designed for smooth interaction:

| Command | Action | Description |
| :--- | :--- | :--- |
| `<prompt>` | **Infer** | Submits input to the model, formats the turn with system context, and streams tokens in real time. |
| `/clear` | **Reset** | Purges KV-cache context and resets conversation history for a fresh session. |
| `/help` | **Help** | Displays interactive commands and engine options. |
| `/exit` | **Exit** | Flushes memory buffers and cleanly shuts down the application. |
| `Ctrl+C` | **Interrupt** | Halts current generation or exits cleanly from the prompt loop. |

---

## Configuration Reference

The engine is configured via `config/config.json`. Below is the complete configuration schema:

```json
{
  "model_path": "models/1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "threads": 8,
  "context_size": 2048,
  "batch_size": 256,
  "temperature": 0.6,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.15,
  "repeat_last_n": 64,
  "max_tokens": 512,
  "offline": true,
  "mode": "Medium RAM Mode",
  "system_prompt": "You are LocalAI, an intelligent, helpful, and logical AI assistant. Think carefully, reason step-by-step, and provide clear, accurate, and direct answers in natural language."
}
```

### Parameter Reference

| Parameter | Type | Default | Recommended | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model_path` | `string` | `"models/1b/..."` | Valid GGUF | Path to the GGUF model binary file. |
| `threads` | `int` | `8` | Logical CPU cores | Number of persistent worker threads in the SIMD threadpool. |
| `context_size` | `int` | `2048` | `1024` – `4096` | Maximum token sequence length allocated in the circular KV-cache. |
| `batch_size` | `int` | `256` | `128` – `512` | Token chunk size processed during prompt ingestion and prefill. |
| `temperature` | `float` | `0.6` | `0.2` – `0.8` | Logit scaling factor. Lower values increase determinism; higher values increase variety. |
| `top_p` | `float` | `0.9` | `0.8` – `0.95` | Cumulative probability threshold for nucleus sampling. |
| `top_k` | `int` | `40` | `20` – `50` | Filters candidate tokens to the Top-K highest-probability logits. |
| `repeat_penalty`| `float` | `1.15` | `1.10` – `1.20` | Multiplicative penalty applied to previously generated tokens. |
| `repeat_last_n` | `int` | `64` | `32` – `128` | Window of recent tokens checked by the repetition penalty filter. |
| `max_tokens` | `int` | `512` | `128` – `2048` | Maximum number of output tokens to generate per turn. |
| `offline` | `bool` | `true` | `true` | Enforces zero-telemetry and offline operation. |
| `mode` | `string` | `"Medium RAM Mode"` | Mode label | Operational memory profile (`Medium RAM Mode`, `Balanced`, `Low RAM`). |
| `system_prompt` | `string` | `"..."` | Custom Persona | Foundational system instruction pre-pended to all conversation sessions. |

---

## Technical Specifications

| Property | Value | Architectural Details |
| :--- | :--- | :--- |
| **Model Architecture** | Llama-3.2 (Instruct) | Rotary Position Embedding (RoPE), RMSNorm, SwiGLU MLP activations |
| **Parameter Count** | 1.24 Billion Parameters | 147 Tensors across 16 Transformer Layers |
| **Quantization Types** | `Q4_K_M` / `Q6_K` | 4-bit block quantization with 6-bit scales (`Q4_K`) and 6-bit weights (`Q6_K`) |
| **Attention Geometry** | 32 Query Heads / 8 KV Heads | Grouped-Query Attention (GQA) with 2048 hidden dimension |
| **Vocabulary Size** | 128,256 Tokens | Byte-level BPE with 256 special token identifiers |
| **Context Window** | 2,048 Tokens | Configurable up to model maximum (Medium RAM Mode: 2048 tokens) |
| **Memory Footprint** | ~763 MB (Weights) + ~128 MB (KV) | Fits entirely within < 1.0 GB physical RAM |
| **Host Requirements** | Standard x86_64 CPU | AVX2 + FMA instruction set support (Intel Haswell+, AMD Zen+) |

---

## Project Structure

```text
LocalAI/
├── CMakeLists.txt                # Cross-platform CMake build configuration
├── config/
│   ├── config.json               # Runtime parameters, sampling config, and system persona
│   └── model.json                # Model specifications and tensor architecture schema
├── include/localai/
│   ├── config.hpp                # Configuration parser and schema declarations
│   ├── engine.hpp                # Engine orchestration, turn handling, and session state
│   ├── gguf_reader.hpp           # GGUF binary parser (header, metadata, tensor catalog)
│   ├── hardware.hpp              # CPU topology and SIMD feature detection (AVX2/FMA)
│   ├── memory_guard.hpp          # Memory profiling and KV-cache bounds verification
│   ├── model_forward.hpp         # Transformer forward pass, ThreadPool, and SIMD matmul
│   ├── model_loader.hpp          # Model integrity verification and tensor loader
│   ├── prompt_builder.hpp        # Llama-3 chat template and role turn formatter
│   ├── sampler.hpp               # Top-K, Top-P, temperature, and repetition penalty sampling
│   ├── session.hpp               # Turn latency metrics and token throughput tracking
│   ├── tensor_info.hpp           # Tensor profiling and parameter calculation
│   ├── terminal_ui.hpp           # Windows console mode configuration and ANSI streaming UI
│   └── tokenizer.hpp             # Byte-Pair Encoding (BPE) engine and reverse byte mapper
├── scripts/
│   └── download_model.py         # Automated model downloader from Hugging Face
└── src/
    ├── config.cpp                # JSON config deserialization and default fallbacks
    ├── engine.cpp                # Token streaming loop, recovery logic, and turn dispatch
    ├── gguf_reader.cpp           # GGUF binary deserializer implementation
    ├── hardware.cpp              # CPUID hardware inspection and RAM queries
    ├── main.cpp                  # CLI entry point and execution mode coordinator
    ├── memory_guard.cpp          # RAM safety calculations and mode assignment
    ├── model_forward.cpp         # AVX2+FMA SIMD vector kernels and persistent ThreadPool
    ├── model_loader.cpp          # File format verification and loading
    ├── prompt_builder.cpp        # Llama-3 turn construction (<|start_header_id|> tags)
    ├── sampler.cpp               # Logit normalization, nucleus selection, and penalty pass
    ├── session.cpp               # Generation statistics and performance counters
    ├── tensor_info.cpp           # Model topology and tensor inspection
    ├── terminal_ui.cpp           # Console handle initialization and ANSI VT100 renderer
    └── tokenizer.cpp             # BPE vocabulary indexing, merges, and UTF-8 decoding
```

---

## Hardware Acceleration & SIMD Kernels

LocalAI incorporates bespoke SIMD kernels designed to maximize memory bandwidth and vector throughput:

### 1. Vectorized `Q4_K` Dot Product (`dot_q4_k_row`)
- **Block Layout**: Processes 256 weights per super-block partitioned into 32-weight sub-blocks.
- **Instruction Utilization**: Employs `_mm256_fmadd_ps` to compute 8 parallel single-precision multiply-accumulations per clock cycle.
- **Deferred Reduction**: Accumulates intermediate products directly into an AVX vector register (`v_total`) across the entire row, performing a single horizontal sum (`hsum256_ps`) at the conclusion of the vector pass.

### 2. Vectorized `Q6_K` Dot Product (`dot_q6_k_row`)
- **Dual-Stream Unpacking**: Simultaneously extracts lower 4-bit nibbles from `ql` and upper 2-bit scales from `qh` to reconstruct 6-bit signed weights in SIMD registers.
- **Single-Pass Scale Integration**: Pre-multiplies sub-block 8-bit scales into the accumulation register, minimizing memory fetches.

### 3. Persistent ThreadPool
- Worker threads are spawned at initialization and remain active throughout the application lifetime.
- Synchronization is governed by an atomic generation sequence counter, eliminating the latency of thread instantiation during time-sensitive forward passes.

---

## Building from Source

### Prerequisites
- A **C++17** compliant compiler:
  - **Clang++ 14+** (Recommended)
  - **GCC 9+**
  - **MSVC 2019+** (Visual Studio 16.0+)
- An **x86_64 CPU** with support for **AVX2** and **FMA** instruction sets (Intel 4th Gen Haswell+, AMD Zen+).

### Option 1: Direct Clang++ Compilation (Windows)
To build the optimized, standalone binary with static runtime linking:

```powershell
& "D:\tools\llvm\bin\clang++.exe" -std=c++17 -O3 -mavx2 -mfma -static -Iinclude (Get-ChildItem src/*.cpp).FullName -o bin\localai.exe
```

### Option 2: Cross-Platform CMake Build
Works identically on Windows, Linux, and macOS:

```bash
# Configure release build with native vectorization
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build localai binary
cmake --build build --config Release --parallel
```

The compiled binary will be placed in `bin/localai` (or `bin/Release/localai.exe` on Windows).

---

## Troubleshooting & FAQ

<details>
<summary><strong>Why does LocalAI not require a GPU or CUDA?</strong></summary>

LocalAI is engineered specifically for CPU-based inference. Through 4-bit/6-bit block quantization (`Q4_K_M`), memory bandwidth requirements are reduced by over 75%. Combined with AVX2/FMA SIMD vector instructions and multithreading, commodity CPUs deliver responsive token streaming speeds without the hardware cost or driver complexity of dedicated GPUs.
</details>

<details>
<summary><strong>Can I use other GGUF models?</strong></summary>

Yes. LocalAI natively parses any standard GGUF file adhering to the Llama architecture. To use a different model, download the GGUF file, place it in the `models/` directory, and update the `"model_path"` attribute in `config/config.json`.
</details>

<details>
<summary><strong>What is Medium RAM Mode?</strong></summary>

Medium RAM Mode dynamically dimensions the KV-cache to 2,048 context tokens and bounds activation buffers, ensuring the entire model and runtime context operate comfortably within ~890 MB of system RAM.
</details>

---

## License

This project is open-source software licensed under the **[MIT License](LICENSE)**.  
Built for private, local, and sovereign machine intelligence.
