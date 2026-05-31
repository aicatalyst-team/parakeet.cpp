#include "parakeet_capi.h"
#include "audio_io.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "env not set; skip\n"); return 77; }
    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }
    pk::Audio a;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", a) || a.samples.empty()) {
        std::fprintf(stderr, "wav load failed\n"); parakeet_capi_free(ctx); return 1;
    }
    char* single = parakeet_capi_transcribe_path_json(ctx, "tests/fixtures/speech.wav", 0);
    if (!single) { std::fprintf(stderr, "single failed: %s\n", parakeet_capi_last_error(ctx)); parakeet_capi_free(ctx); return 1; }
    std::string single_doc(single);
    parakeet_capi_free_string(single);

    std::vector<float> concat;
    concat.insert(concat.end(), a.samples.begin(), a.samples.end());
    concat.insert(concat.end(), a.samples.begin(), a.samples.end());
    int n_samples[2] = { (int)a.samples.size(), (int)a.samples.size() };
    char* batch = parakeet_capi_transcribe_pcm_batch_json(ctx, concat.data(), n_samples, 2, 16000, 0);
    if (!batch) { std::fprintf(stderr, "batch failed: %s\n", parakeet_capi_last_error(ctx)); parakeet_capi_free(ctx); return 1; }
    std::string doc(batch);
    parakeet_capi_free_string(batch);

    auto text_field = [](const std::string& s) -> std::string {
        size_t p = s.find("\"text\":\"");
        if (p == std::string::npos) return "";
        p += 8; size_t q = s.find('"', p);
        return s.substr(p, q - p);
    };
    std::string t = text_field(single_doc);
    bool is_array = !doc.empty() && doc.front() == '[' && doc.back() == ']';
    size_t cnt = 0, pos = 0;
    std::string needle = "\"text\":\"" + t + "\"";
    while ((pos = doc.find(needle, pos)) != std::string::npos) { ++cnt; pos += needle.size(); }
    bool ok = is_array && !t.empty() && cnt == 2;
    std::fprintf(stderr, "array=%d text='%s' count=%zu -> %s\n", is_array, t.c_str(), cnt, ok?"OK":"FAIL");
    parakeet_capi_free(ctx);
    return ok ? 0 : 1;
}
