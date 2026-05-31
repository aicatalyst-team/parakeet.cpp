#include "joint.hpp"
#include "prediction.hpp"
#include "encoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if (!ml.load(gguf)) return 1;
    std::vector<float> mel; std::vector<int64_t> ms;
    if (!pktest::load_baseline(base, "mel", mel, ms)) return 1;
    const int n_mels=(int)ms[0], T=(int)ms[1];
    pk::Encoder enc(ml); std::vector<float> eo; int dm=0,Tout=0;
    enc.forward(mel,n_mels,T,eo,dm,Tout);
    std::vector<float> encr((size_t)Tout*dm);
    for (int t=0;t<Tout;++t) for (int c=0;c<dm;++c) encr[(size_t)t*dm+c]=eo[(size_t)c*Tout+t];
    pk::Joint joint(ml);
    std::vector<float> ep; joint.precompute_enc_proj(encr, Tout, dm, ep);
    const int Hj = joint.joint_hidden(), Vp = joint.V_plus();
    pk::PredictionNet pred(ml); const int Hp = pred.hidden_size();
    std::vector<int> frames = {0, Tout/2, Tout-1};
    std::vector<int32_t> toks = {7, 13, 21};
    const int N = 3;
    std::vector<std::vector<float>> gs(N);
    pk::PredState z = pred.zero_state();
    for (int n=0;n<N;++n){ pk::PredState os; pred.step(toks[n], false, z, gs[n], os); }
    std::vector<std::vector<float>> ref(N);
    for (int n=0;n<N;++n) joint.step_logits(ep.data()+(size_t)frames[n]*Hj, gs[n].data(), Hp, ref[n]);
    std::vector<float> epg((size_t)Hj*N), gg((size_t)Hp*N);
    for (int n=0;n<N;++n){
        std::copy(ep.begin()+(size_t)frames[n]*Hj, ep.begin()+(size_t)(frames[n]+1)*Hj, epg.begin()+(size_t)n*Hj);
        std::copy(gs[n].begin(), gs[n].end(), gg.begin()+(size_t)n*Hp);
    }
    std::vector<float> lb; joint.step_logits_batch(epg.data(), gg.data(), Hp, N, lb);
    bool ok = (int)lb.size()==Vp*N;
    for (int n=0;n<N && ok;++n){
        std::vector<float> col(lb.begin()+(size_t)n*Vp, lb.begin()+(size_t)(n+1)*Vp);
        ok = pktest::compare(col, ref[n], "jointbatch", 1e-3f, 1e-3f) && ok;
    }
    return ok?0:1;
}
