# Blog Abstract: parakeet.cpp on OpenShift

**Thesis**: Deploying parakeet.cpp on OpenShift proves that lightweight C++ inference engines can run containerized speech-to-text workloads on enterprise Kubernetes with zero external dependencies, quantized models, and CPU-only execution.

**Target Audience**: Platform engineers and ML engineers evaluating self-hosted ASR (automatic speech recognition) solutions on OpenShift AI.

**Blog Type**: Red Hat Developer Blog

**Key Points**:
1. parakeet.cpp compiles to a single binary with no Python runtime, making it ideal for slim, secure container images on OpenShift
2. GGUF quantization (q4_k) reduces the 110M-parameter model to 126MB while preserving transcription accuracy
3. The entire deployment (build, deploy, test) completes in under 5 minutes on OpenShift using UBI-based multi-stage builds

**Products**: Red Hat OpenShift AI, Open Data Hub

**CTA**: Deploy your own ASR inference engine on OpenShift using parakeet.cpp

**Section Outline**:
1. What is parakeet.cpp?
2. Why containerize speech-to-text on OpenShift?
3. Building with UBI: multi-stage Dockerfile
4. Deploying as Kubernetes Jobs
5. Transcription results and accuracy
6. What we learned
7. Try it yourself
