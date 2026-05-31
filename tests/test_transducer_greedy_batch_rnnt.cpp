#include "transducer_batch.hpp"
#include "rnnt.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "encoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <tuple>
static bool toks_equal(const std::vector<pk::TokenInfo>& a, const std::vector<pk::TokenInfo>& b){
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        if (a[i].id!=b[i].id||a[i].frame!=b[i].frame||a[i].span!=b[i].span) return false;
        if (std::fabs(a[i].conf-b[i].conf)>1e-4f) return false;
    }
    return true;
}
int main(){
    // RNNT-only model (the 110m anchor is TDT, not pure RNNT). Self-skip unless a
    // dedicated RNNT GGUF is provided.
    const char* gguf=std::getenv("PARAKEET_TEST_GGUF_RNNT"); const char* base=std::getenv("PARAKEET_TEST_BASELINE");
    if(!gguf||!base){ std::fprintf(stderr,"PARAKEET_TEST_GGUF_RNNT not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;
    const auto& cfg = ml.config();
    if (!cfg.tdt_durations.empty()){ std::fprintf(stderr,"model has TDT durations; not pure RNNT; skip\n"); return 77; }
    std::vector<float> mel; std::vector<int64_t> ms;
    if(!pktest::load_baseline(base,"mel",mel,ms)) return 1;
    const int n_mels=(int)ms[0], T0=(int)ms[1];
    pk::Encoder enc(ml);
    auto enc_row=[&](const std::vector<float>& m,int Tn){ std::vector<float> eo;int dm=0,to=0; enc.forward(m,n_mels,Tn,eo,dm,to);
        std::vector<float> r((size_t)to*dm); for(int t=0;t<to;++t)for(int c=0;c<dm;++c) r[(size_t)t*dm+c]=eo[(size_t)c*to+t]; return std::make_tuple(r,to,dm); };
    auto [e0,to0,dm]=enc_row(mel,T0);
    int T1=T0*3/4; std::vector<float> mel1((size_t)n_mels*T1);
    for(int m=0;m<n_mels;++m)for(int t=0;t<T1;++t) mel1[(size_t)m*T1+t]=mel[(size_t)m*T0+t];
    auto [e1,to1,dm1]=enc_row(mel1,T1);
    pk::PredictionNet pred(ml); pk::Joint joint(ml);
    const int blank=(int)cfg.blank_id, maxs=(int)cfg.max_symbols;
    std::vector<pk::TokenInfo> r0,r1;
    auto id0 = pk::rnnt_greedy(pred,joint,e0,to0,dm,blank,maxs,&r0);
    auto id1 = pk::rnnt_greedy(pred,joint,e1,to1,dm,blank,maxs,&r1);
    std::vector<std::vector<float>> encs={e0,e1}; std::vector<int> Ts={to0,to1};
    std::vector<std::vector<int32_t>> ids; std::vector<std::vector<pk::TokenInfo>> tk;
    std::vector<int32_t> no_dur{};
    pk::transducer_greedy_batch(pred,joint,encs,Ts,dm,no_dur,blank,maxs,ids,&tk);
    bool ok = ids.size()==2 && ids[0]==id0 && ids[1]==id1 && toks_equal(tk[0],r0) && toks_equal(tk[1],r1);
    std::fprintf(stderr,"item0 ids %zu/%zu words; item1 ids %zu/%zu -> %s\n", ids[0].size(),id0.size(),ids[1].size(),id1.size(), ok?"OK":"FAIL");
    return ok?0:1;
}
