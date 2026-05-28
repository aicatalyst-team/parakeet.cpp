cmake_minimum_required(VERSION 3.18)
project(parakeet_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PARAKEET_BUILD_TESTS "Build ctest targets" OFF)
option(PARAKEET_BUILD_CLI   "Build parakeet-cli" ON)
option(PARAKEET_SHARED      "Build libparakeet as shared" OFF)
option(PARAKEET_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(PARAKEET_GGML_METAL  "Forward GGML_METAL" OFF)
option(PARAKEET_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(PARAKEET_GGML_HIPBLAS "Forward GGML_HIPBLAS" OFF)

set(GGML_CUDA    ${PARAKEET_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL   ${PARAKEET_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${PARAKEET_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIPBLAS ${PARAKEET_GGML_HIPBLAS} CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

set(PARAKEET_SRC
    src/parakeet.cpp
    src/common.cpp
    src/audio_io.cpp
    src/model_loader.cpp
    src/ggml_graph.cpp
    src/fft.cpp
    src/mel.cpp
    src/subsampling.cpp
    src/relpos_attention.cpp
    src/conformer.cpp
    src/pos_enc.cpp
    src/encoder.cpp
    src/ctc_decoder.cpp
    src/tokenizer.cpp
    src/search.cpp)

if(PARAKEET_SHARED)
  add_library(parakeet SHARED ${PARAKEET_SRC})
else()
  add_library(parakeet STATIC ${PARAKEET_SRC})
endif()
target_include_directories(parakeet PUBLIC include PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(parakeet PUBLIC ggml)

if(PARAKEET_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(PARAKEET_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
