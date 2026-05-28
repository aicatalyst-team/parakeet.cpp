#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
struct ggml_tensor;
struct ggml_context;
struct gguf_context;
namespace pk {
struct ParakeetConfig {
    std::string arch;
    // encoder
    uint32_t feat_in=0, d_model=0, n_layers=0, n_heads=0, ff_dim=0, conv_kernel=0;
    std::string conv_norm_type;
    uint32_t subsampling_factor=0, subsampling_conv_channels=0, pos_emb_max_len=5000;
    bool xscaling=true;
    // preprocessor
    uint32_t sample_rate=16000, n_mels=0, n_fft=0, win_length=0, hop_length=0;
    float preemph=0.0f, mag_power=2.0f, log_zero_guard=0.0f;
    std::string normalize;
    // transducer (optional)
    uint32_t pred_hidden=0, pred_rnn_layers=0, joint_hidden=0;
    std::string joint_activation;
    std::vector<int32_t> tdt_durations;
    // vocab
    uint32_t vocab_size=0, blank_id=0;
    std::vector<std::string> tokenizer_pieces;
};
class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    bool load(const std::string& path);
    const ParakeetConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const; // nullptr if absent
    ggml_context* ggml_ctx() const { return ctx_; }
private:
    ParakeetConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};
}
