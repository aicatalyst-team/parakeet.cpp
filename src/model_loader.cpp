#include "model_loader.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstring>
namespace pk {
static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : (uint32_t)gguf_get_val_u32(g,id);
}
static int32_t kv_i32(gguf_context* g, const char* k, int32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_i32(g,id);
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int64_t id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}
ModelLoader::~ModelLoader(){ if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_); }
bool ModelLoader::load(const std::string& path){
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ PK_LOG("gguf open failed: %s", path.c_str()); return false; }
    cfg_.arch        = kv_str(gguf_, "parakeet.arch");
    cfg_.feat_in     = kv_u32(gguf_, "parakeet.encoder.feat_in");
    cfg_.d_model     = kv_u32(gguf_, "parakeet.encoder.d_model");
    cfg_.n_layers    = kv_u32(gguf_, "parakeet.encoder.n_layers");
    cfg_.n_heads     = kv_u32(gguf_, "parakeet.encoder.n_heads");
    cfg_.ff_dim      = kv_u32(gguf_, "parakeet.encoder.ff_dim");
    cfg_.conv_kernel = kv_u32(gguf_, "parakeet.encoder.conv_kernel");
    cfg_.conv_norm_type = kv_str(gguf_, "parakeet.encoder.conv_norm_type", "batch_norm");
    cfg_.subsampling_factor = kv_u32(gguf_, "parakeet.encoder.subsampling_factor");
    cfg_.subsampling_conv_channels = kv_u32(gguf_, "parakeet.encoder.subsampling_conv_channels");
    cfg_.xscaling    = kv_bool(gguf_, "parakeet.encoder.xscaling", true);
    cfg_.pos_emb_max_len = kv_u32(gguf_, "parakeet.encoder.pos_emb_max_len", 5000);
    // cache-aware streaming / causal config (Phase 5). Absent for offline models
    // -> offline-safe defaults (regular style, no causal, streaming.present=false).
    cfg_.att_context_left  = kv_i32(gguf_, "parakeet.encoder.att_context_left", -1);
    cfg_.att_context_right = kv_i32(gguf_, "parakeet.encoder.att_context_right", -1);
    cfg_.att_context_style = kv_str(gguf_, "parakeet.encoder.att_context_style", "regular");
    cfg_.causal_downsampling = kv_bool(gguf_, "parakeet.encoder.causal_downsampling", false);
    cfg_.conv_causal = kv_bool(gguf_, "parakeet.encoder.conv_causal", false);
    if(cfg_.att_context_style != "regular"){
        StreamingCfg& s = cfg_.streaming;
        s.chunk_size = kv_i32_arr(gguf_, "parakeet.streaming.chunk_size");
        s.shift_size = kv_i32_arr(gguf_, "parakeet.streaming.shift_size");
        s.pre_encode_cache_size = kv_i32_arr(gguf_, "parakeet.streaming.pre_encode_cache_size");
        s.cache_drop_size = kv_i32(gguf_, "parakeet.streaming.cache_drop_size", 0);
        s.last_channel_cache_size = kv_i32(gguf_, "parakeet.streaming.last_channel_cache_size", 0);
        s.valid_out_len = kv_i32(gguf_, "parakeet.streaming.valid_out_len", 0);
        s.drop_extra_pre_encoded = kv_i32(gguf_, "parakeet.streaming.drop_extra_pre_encoded", 0);
        s.present = true;
    }
    cfg_.sample_rate = kv_u32(gguf_, "parakeet.preprocessor.sample_rate", 16000);
    cfg_.n_mels      = kv_u32(gguf_, "parakeet.preprocessor.n_mels");
    cfg_.n_fft       = kv_u32(gguf_, "parakeet.preprocessor.n_fft");
    cfg_.win_length  = kv_u32(gguf_, "parakeet.preprocessor.win_length");
    cfg_.hop_length  = kv_u32(gguf_, "parakeet.preprocessor.hop_length");
    cfg_.preemph     = kv_f32(gguf_, "parakeet.preprocessor.preemph", 0.0f);
    cfg_.mag_power   = kv_f32(gguf_, "parakeet.preprocessor.mag_power", 2.0f);
    cfg_.normalize   = kv_str(gguf_, "parakeet.preprocessor.normalize", "per_feature");
    cfg_.log_zero_guard = kv_f32(gguf_, "parakeet.preprocessor.log_zero_guard", 0.0f);
    cfg_.pred_hidden = kv_u32(gguf_, "parakeet.decoder.pred_hidden");
    cfg_.pred_rnn_layers = kv_u32(gguf_, "parakeet.decoder.pred_rnn_layers");
    cfg_.joint_hidden = kv_u32(gguf_, "parakeet.joint.joint_hidden");
    cfg_.joint_activation = kv_str(gguf_, "parakeet.joint.activation");
    cfg_.max_symbols = kv_u32(gguf_, "parakeet.decoding.max_symbols", 10);
    cfg_.vocab_size  = kv_u32(gguf_, "parakeet.vocab_size");
    cfg_.blank_id    = kv_u32(gguf_, "parakeet.blank_id");
    // durations array (stored as INT32 by the converter)
    { int64_t id = gguf_find_key(gguf_, "parakeet.tdt.durations");
      if(id>=0 && gguf_get_arr_type(gguf_,id)==GGUF_TYPE_INT32){
          size_t n = gguf_get_arr_n(gguf_,id);
          const int32_t* a = (const int32_t*)gguf_get_arr_data(gguf_,id);
          cfg_.tdt_durations.assign(a, a+n); } }
    // tokenizer pieces STRING array
    { int64_t id = gguf_find_key(gguf_, "parakeet.tokenizer.pieces");
      if(id>=0 && gguf_get_arr_type(gguf_,id)==GGUF_TYPE_STRING){
          size_t n = gguf_get_arr_n(gguf_,id);
          cfg_.tokenizer_pieces.resize(n);
          for(size_t i=0;i<n;++i)
              cfg_.tokenizer_pieces[i] = gguf_get_arr_str(gguf_,id,i); } }
    // tensors
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for(int64_t i=0;i<nt;++i){ const char* nm = gguf_get_tensor_name(gguf_,i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm); if(t) tensors_[nm]=t; }
    return cfg_.d_model>0 && cfg_.vocab_size>0;
}
ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it==tensors_.end()? nullptr : it->second;
}
}
