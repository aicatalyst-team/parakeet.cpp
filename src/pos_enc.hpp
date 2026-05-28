#pragma once
#include <vector>
namespace pk {

// Relative positional encoding table for the FastConformer encoder, matching
// NeMo's RelPositionalEncoding (multi_head_attention.py).
//
// For an input of length T, NeMo builds `pos_emb` over 2T-1 relative positions
// running from +(T-1) DOWN TO -(T-1) (inclusive). Each row p holds a sinusoid:
//   div_term[i] = exp(2i * -(log(10000)/d_model))           for i in [0, d_model/2)
//   pe[p, 2i]   = sin(position(p) * div_term[i])
//   pe[p, 2i+1] = cos(position(p) * div_term[i])
// where position(0) = T-1, position(1) = T-2, ..., position(2T-2) = -(T-1).
//
// (NeMo precomputes the table over a larger length and slices the center; the
// sliced positions are exactly +(T-1)..-(T-1), independent of the precompute
// length, so computing directly for T reproduces the same values.)
//
// Output: row-major [2T-1, d_model] (d_model fastest), i.e. out[p*d_model + c].
void rel_pos_encoding(int T, int d_model, std::vector<float>& out);

} // namespace pk
