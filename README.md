# parakeet.cpp

parakeet.cpp is a C++/ggml inference port of NVIDIA NeMo Parakeet ASR models, providing fast, dependency-light automatic speech recognition on CPU and GPU via the ggml tensor library. See `docs/superpowers/specs/` for the full design.

## Build (placeholder)

```sh
git submodule update --init --recursive
cmake -B build -DPARAKEET_BUILD_TESTS=ON
cmake --build build -j
```
