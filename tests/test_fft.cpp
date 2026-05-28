#include "fft.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
int main() {
    const int N=512;
    std::vector<float> x(N);
    // pure cosine at bin k=8: magnitude should peak at bin 8
    for (int i=0;i<N;++i) x[i]=std::cos(2.0*M_PI*8*i/N);
    std::vector<float> re(N/2+1), im(N/2+1);
    pk::rfft(x, re, im);
    // find peak bin
    int peak=0; double best=-1;
    for (int k=0;k<=N/2;++k){ double m=re[k]*re[k]+im[k]*im[k]; if(m>best){best=m;peak=k;} }
    if (peak!=8){ std::fprintf(stderr,"peak at %d, expected 8\n",peak); return 1; }
    std::printf("fft ok: peak bin=%d\n", peak);
    return 0;
}
