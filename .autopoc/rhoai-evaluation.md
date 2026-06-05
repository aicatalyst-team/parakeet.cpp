# RHOAI Evaluation: parakeet.cpp

## Strategy: Red Hat AI 2026

### Impact Dimensions (0-20 each)

| Dimension | Score | Justification |
|---|---|---|
| audience_value | 14/20 | Speech-to-text inference has broad enterprise value (transcription, accessibility, call center analytics). Parakeet.cpp targets ML/AI engineers and platform teams deploying ASR models. |
| strategic_alignment | 12/20 | Aligns with model-inference strategy area -- demonstrates self-hosted inference with quantization variants and CPU/GPU flexibility. However, it's a C++ CLI tool, not a serving runtime, so it doesn't directly integrate with vLLM/KServe/llm-d stack. |
| strategy_fit | 13/20 | Fits the "fast, flexible, efficient inference" narrative. Shows quantization (q4_k, q8_0), multi-backend (CPU, CUDA, Vulkan), and ggml-based inference -- relevant to accelerator optionality and optimized models. |
| platform_leverage | 8/20 | Limited platform showcase -- CLI tool doesn't naturally demonstrate routes, services, scaling. Can demonstrate GPU scheduling and PVC-backed model storage, but requires wrapping in HTTP service for fuller demo. |
| demo_potential | 14/20 | Speech-to-text is visually compelling. Feed audio, get transcript. Performance benchmarks show it beating whisper.cpp. Multiple model sizes and quantization variants allow scaling demos. |

**Impact Score: (14 + 12 + 13 + 8 + 14) / 5 = 12.2/20**

### Feasibility Dimensions (0-20 each)

| Dimension | Score | Justification |
|---|---|---|
| container_readiness | 18/20 | Existing multi-stage Dockerfile proven in CI. Multi-arch support. Clean binary + shared libs runtime. Only needs UBI base swap. |
| dependency_profile | 18/20 | Minimal: cmake, build-essential, git for build. libgomp1 for runtime. ggml vendored as submodule. No Python, no package managers, no external services. |
| reproduction_confidence | 16/20 | CI-proven builds, deterministic CMake, test fixtures committed. Need GGUF model download (125-860MB from HuggingFace). |
| complexity_sweet_spot | 16/20 | Well-scoped: one binary + one model file + one audio file = transcript. Streaming mode adds richness without complexity. |

**Feasibility Score: (18 + 18 + 16 + 16) / 4 = 17.0/20**

### Strategy Areas
- model-inference (primary)

### Capability Labels
- serving, quantization, accelerator-optionality, optimized-models

### Relationship
- validates-platform-story

### Strengths
- Near-zero external dependencies
- Existing Dockerfile with proven CI
- Compelling performance benchmarks (beats whisper.cpp and NeMo)
- MIT license
- CPU-first with optional GPU

### Risks
- CLI tool, not a server -- limited platform feature showcase without HTTP wrapper
- Requires GGUF model download from HuggingFace
- No health endpoint or readiness probe built-in
