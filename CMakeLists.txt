cmake_minimum_required(VERSION 3.16)
project(LocalAI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/config.cpp
    src/hardware.cpp
    src/memory_guard.cpp
    src/model_loader.cpp
    src/terminal_ui.cpp
    src/engine.cpp
    src/gguf_reader.cpp
    src/tokenizer.cpp
    src/tensor_info.cpp
    src/prompt_builder.cpp
    src/session.cpp
    src/sampler.cpp
    src/model_forward.cpp
)

add_executable(localai ${SOURCES})

if(MSVC)
    target_compile_options(localai PRIVATE /W4 /utf-8 /arch:AVX2)
else()
    target_compile_options(localai PRIVATE -Wall -Wextra -mavx2 -mfma)
endif()
