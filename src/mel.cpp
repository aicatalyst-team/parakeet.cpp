#include "mel.hpp"
#include "fft.hpp"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace pk {

// CONSTANT epsilon added to std in NeMo normalize_batch (features.py: CONSTANT = 1e-5).
static constexpr double kNormEps = 1e-5;

MelFrontend::MelFrontend(const ModelLoader& ml) {
    const ParakeetConfig& c = ml.config();
    n_fft_     = (int)c.n_fft;
    win_       = (int)c.win_length;
    hop_       = (int)c.hop_length;
    n_mels_    = (int)c.n_mels;
    n_bins_    = n_fft_ / 2 + 1;
    preemph_   = c.preemph;
    mag_power_ = c.mag_power;
    log_guard_ = c.log_zero_guard;
    per_feature_ = (c.normalize == "per_feature");

    // --- Window: lift the Hann window from the GGUF and center-pad to n_fft. ---
    // NeMo builds torch.hann_window(win_length, periodic=False) and passes it to
    // torch.stft with win_length < n_fft, so torch zero-pads the window to n_fft,
    // centered: pad (n_fft - win_length) // 2 zeros on each side.
    window_.assign(n_fft_, 0.0f);
    ggml_tensor* w = ml.tensor("preprocessor.featurizer.window");
    if (w) {
        const int wlen = (int)w->ne[0];
        const float* wd = (const float*)w->data;
        if (wlen == n_fft_) {
            std::memcpy(window_.data(), wd, sizeof(float) * n_fft_);
        } else {
            const int left = (n_fft_ - wlen) / 2;
            for (int i = 0; i < wlen && (left + i) < n_fft_; ++i)
                window_[left + i] = wd[i];
        }
    }

    // --- Filterbank: preprocessor.featurizer.fb has ggml ne=[n_bins, n_mels, 1]
    // i.e. numpy [1, n_mels, n_bins] row-major -> fb_[m*n_bins + b]. ---
    fb_.assign((size_t)n_mels_ * n_bins_, 0.0f);
    ggml_tensor* fb = ml.tensor("preprocessor.featurizer.fb");
    if (fb) {
        const float* fd = (const float*)fb->data;
        std::memcpy(fb_.data(), fd, sizeof(float) * (size_t)n_mels_ * n_bins_);
    }
}

void MelFrontend::compute(const std::vector<float>& samples,
                          std::vector<float>& feats, int& n_mels, int& T) const {
    const int S = (int)samples.size();

    // ----- Step 2: Preemphasis -----
    // NeMo: x = cat((x[:,0], x[:,1:] - preemph * x[:,:-1])); first sample unchanged.
    std::vector<double> x((size_t)S);
    if (preemph_ > 0.0f && S > 0) {
        x[0] = samples[0];
        for (int t = 1; t < S; ++t)
            x[t] = (double)samples[t] - (double)preemph_ * (double)samples[t - 1];
    } else {
        for (int t = 0; t < S; ++t) x[t] = samples[t];
    }

    // seq_len = floor(S / hop) (NeMo get_seq_len for center=True:
    //   floor((S + n_fft//2*2 - n_fft) / hop) = floor(S / hop)).
    const int seq_len = (S > 0) ? (S / hop_) : 0;

    // ----- Step 3: STFT with center=True -----
    // Reflect-pad n_fft//2 on each side, then frame with hop, apply the window,
    // FFT -> keep bins 0..n_fft/2. Frame count = 1 + floor((S + 2*pad - n_fft)/hop).
    const int pad = n_fft_ / 2;
    const int padded_len = S + 2 * pad;
    const int n_frames = (padded_len >= n_fft_) ? (1 + (padded_len - n_fft_) / hop_) : 0;
    T = n_frames;
    n_mels = n_mels_;

    // Zero-padded signal. NeMo's stft uses pad_mode="constant" (features.py),
    // so center=True pads n_fft//2 ZEROS on each side (not reflection).
    // The original signal sits at [pad, pad+S); everything else is 0.
    std::vector<double> padded((size_t)padded_len, 0.0);
    for (int j = 0; j < S; ++j) padded[pad + j] = x[j];

    // Power spectrum per frame: power[b][t] = (re^2 + im^2) (mag then ^mag_power).
    // We store frame-major scratch; mel is computed directly into feats.
    feats.assign((size_t)n_mels_ * T, 0.0f);

    std::vector<float> frame((size_t)n_fft_);
    std::vector<float> re, im;
    std::vector<double> power((size_t)n_bins_);

    for (int t = 0; t < T; ++t) {
        const int start = t * hop_;
        for (int i = 0; i < n_fft_; ++i)
            frame[i] = (float)(padded[start + i] * (double)window_[i]);

        pk::rfft(frame, re, im);

        // ----- Step 4: power spectrum -----
        // NeMo: x = sqrt(re^2 + im^2) (guard=0 at inference), then x = x^mag_power.
        for (int b = 0; b < n_bins_; ++b) {
            const double mag = std::sqrt((double)re[b] * re[b] + (double)im[b] * im[b]);
            power[b] = (mag_power_ == 1.0f) ? mag : std::pow(mag, (double)mag_power_);
        }

        // ----- Step 5: mel projection: mel[m,t] = sum_b fb[m,b] * power[b] -----
        // ----- Step 6: log(mel + log_zero_guard) -----
        for (int m = 0; m < n_mels_; ++m) {
            const float* fbm = &fb_[(size_t)m * n_bins_];
            double acc = 0.0;
            for (int b = 0; b < n_bins_; ++b)
                acc += (double)fbm[b] * power[b];
            feats[(size_t)m * T + t] = (float)std::log(acc + (double)log_guard_);
        }
    }

    // ----- Step 7: per-feature normalization (normalize_batch, per_feature) -----
    // Per mel-bin (row): mean over the first `seq_len` frames, unbiased std
    // (ddof=1), std += CONSTANT, then (x - mean) / std. Frames >= seq_len are
    // zeroed (valid_mask). B=1.
    if (per_feature_ && T > 0) {
        const int valid = std::min(seq_len, T);
        for (int m = 0; m < n_mels_; ++m) {
            float* row = &feats[(size_t)m * T];
            double mean = 0.0;
            if (valid > 0) {
                for (int t = 0; t < valid; ++t) mean += row[t];
                mean /= (double)valid;
            }
            double var = 0.0;
            if (valid > 1) {
                for (int t = 0; t < valid; ++t) {
                    const double d = (double)row[t] - mean;
                    var += d * d;
                }
                var /= (double)(valid - 1); // unbiased (ddof=1)
            }
            double sd = std::sqrt(var); // NaN-guard not needed: valid>1 here
            sd += kNormEps;
            for (int t = 0; t < T; ++t) {
                if (t < valid) row[t] = (float)(((double)row[t] - mean) / sd);
                else           row[t] = 0.0f; // masked beyond seq_len
            }
        }
    }
}

} // namespace pk
