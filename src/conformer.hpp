#pragma once
#include "model_loader.hpp"
#include <vector>
namespace pk {

// A single FastConformer encoder layer (NeMo ConformerLayer), built per-layer
// from the GGUF weights. Mirrors NeMo's ConformerLayer.forward exactly:
//
//   r = x
//   r = r + 0.5 * feed_forward1(norm_feed_forward1(r))   # FFN1 (half-step)
//   r = r + self_attn(norm_self_att(r), pos_emb, mask)    # MHSA (RelPosAttention)
//   r = r + conv(norm_conv(r))                            # Conv module
//   r = r + 0.5 * feed_forward2(norm_feed_forward2(r))    # FFN2 (half-step)
//   out = norm_out(r)
//
// feed_forwardN  = linear2(silu(linear1(x))).
// conv (ConformerConvolution, operates on [d_model, T]):
//   pointwise_conv1 (d->2d, k=1) -> GLU(dim=channel) -> [zero padded time pos]
//   -> depthwise_conv (d->d, k=conv_kernel, groups=d, symmetric pad (k-1)/2)
//   -> batch_norm (inference affine from running stats, eps 1e-5)
//   -> SiLU -> pointwise_conv2 (d->d, k=1).
// All norm_* are LayerNorm (eps 1e-5). MHSA reuses pk::RelPosAttention.
//
// Layout convention (matches the rest of the port and the baseline GGUF):
//   x       row-major [T, d_model]      (d_model fastest)
//   pos_emb row-major [2T-1, d_model]   (d_model fastest)
//   out     row-major [T, d_model]      (d_model fastest)
//
// `valid_len` is the number of non-padding frames (frames >= valid_len are
// center-pad). It is threaded into RelPosAttention (key/query masking) and used
// to zero padded time positions before the depthwise conv, matching NeMo's
// pad_mask handling. Pass valid_len == T to disable masking.
class ConformerLayer {
public:
    ConformerLayer(const ModelLoader& ml, int layer_idx);

    // x: [T, d_model]; pos_emb: [pos_len=2T-1, d_model]; out: [T, d_model].
    void forward(const std::vector<float>& x, int T,
                 const std::vector<float>& pos_emb, int pos_len,
                 int valid_len,
                 std::vector<float>& out) const;

    // Same as forward(), but also returns the ConformerConvolution sub-module
    // output (NeMo `layers[i].conv` output, row-major [T, d_model]) for parity
    // localization against the baseline `l0_conv_out`.
    void forward_with_conv(const std::vector<float>& x, int T,
                           const std::vector<float>& pos_emb, int pos_len,
                           int valid_len,
                           std::vector<float>& out,
                           std::vector<float>& conv_out) const;

private:
    const ModelLoader& ml_;
    int layer_idx_;
    int d_model_;
    int n_heads_;
    int ff_dim_;
    int conv_kernel_;
};

} // namespace pk
