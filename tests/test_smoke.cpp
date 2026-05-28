#include "parakeet.h"
#include <cstdio>
#include <cstring>

int main() {
    const char* v = parakeet_version();
    if (v == nullptr || std::strlen(v) == 0) {
        std::fprintf(stderr, "version string is empty\n");
        return 1;
    }
    std::printf("parakeet.cpp version: %s\n", v);
    return 0;
}
