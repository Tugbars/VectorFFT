/* demo_v4_score.c — score specific factorizations with V4, side by side.
 *
 * Usage: demo_v4_score <N> <K> <fac1> <fac2> ...
 *   where each fac is "RxRxRx..." (e.g., "8x32x16")
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../planner.h"
#include "../estimate_plan.h"

static int parse_fac(const char *s, int *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || v > 1024) return -1;
        out[n++] = (int)v;
        p = end;
        if (*p == 'x' || *p == 'X' || *p == '*' || *p == ',') p++;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s N K fac1 [fac2 ...]\n", argv[0]);
        return 1;
    }
    int    N = atoi(argv[1]);
    size_t K = (size_t)atoll(argv[2]);
    stride_cpu_info_t cpu = stride_detect_cpu();

    printf("V4 scores for N=%d K=%zu (lower = predicted faster):\n", N, K);
    for (int i = 3; i < argc; i++) {
        int fac[STRIDE_MAX_STAGES];
        int nf = parse_fac(argv[i], fac, STRIDE_MAX_STAGES);
        if (nf < 0) { printf("  %-32s  (parse err)\n", argv[i]); continue; }
        int prod = 1;
        for (int s = 0; s < nf; s++) prod *= fac[s];
        if (prod != N) {
            printf("  %-32s  (product %d != N=%d)\n", argv[i], prod, N);
            continue;
        }
        double sc = _vfft_proto_v4_score(fac, nf, K, N, &cpu);
        printf("  %-32s  v4=%12.0f\n", argv[i], sc);
    }
    return 0;
}
