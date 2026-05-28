#pragma once
#include "model_loader.hpp"
#include <vector>
namespace pk {
class MelFrontend {
public:
    explicit MelFrontend(const ModelLoader& ml);
    // samples: 16k mono. Out: feats row-major [n_mels, T] (frame-major inner = T).
    void compute(const std::vector<float>& samples, std::vector<float>& feats, int& n_mels, int& T) const;
private:
    int n_fft_, win_, hop_, n_mels_, n_bins_;
    float preemph_, mag_power_, log_guard_;
    bool per_feature_;
    std::vector<float> window_;  // [n_fft] (centered-padded Hann from GGUF)
    std::vector<float> fb_;      // [n_mels, n_bins] row-major (fb_[m*n_bins + b])
};
}
