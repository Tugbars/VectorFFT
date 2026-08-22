/* il_oop_order_verify2.c -- which ORDERING does each front door ship?
 * Build: python build.py --src benches/il_oop_order_verify2.c --vfft --compile */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"
#include "vfft.h"

static int fd_pair(int N, int *oR1, int *oR2)
{
    int iR1 = 0, iR2 = 0;
    for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--) {
        if (N % R2c) continue;
        int R1c = N / R2c;
        if (R1c < 3 || R1c > 64) continue;
        if (!vfft_il2p_leaf_fn(R2c, 0) || !vfft_il2p_mid_fn(R1c, 0)) continue;
        if (!iR1 || abs(R1c - R2c) < abs(iR1 - iR2)) { iR1 = R1c; iR2 = R2c; }
    }
    *oR1 = iR1; *oR2 = iR2; return iR1 != 0;
}

static const char *which(const double *got, const double *h, const double *s,
                         size_t nb, int have_s)
{
    if (memcmp(got, h, nb) == 0) return "HEUR";
    if (have_s && memcmp(got, s, nb) == 0) return "SWAP";
    return "other";
}

int main(void)
{
    static const int NS[] = { 21, 27, 33, 50, 75, 150, 300, 675, 128, 512 };
    printf("%-6s %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "N", "heur", "oop.def", "oop.nat", "ip.def", "ip.nat", "oop.scr");
    for (unsigned i = 0; i < sizeof NS / sizeof NS[0]; i++) {
        int N = NS[i], R1, R2;
        if (!fd_pair(N, &R1, &R2)) continue;
        size_t nb = 2 * (size_t)N * sizeof(double);
        double *zi = malloc(nb), *zh = malloc(nb), *zs = malloc(nb), *zt = malloc(nb);
        for (int n = 0; n < N; n++) { zi[2*n] = sin(0.7*n)+0.3; zi[2*n+1] = cos(0.31*n)-0.2; }
        vfft_il2p_plan_t *ph = vfft_il2p_create(N, R1, R2);
        vfft_il2p_plan_t *ps = (R1 != R2) ? vfft_il2p_create(N, R2, R1) : NULL;
        memset(zh, 0, nb); memset(zs, 0, nb);
        if (ph) vfft_il2p_execute_fwd(ph, zi, zh);
        if (ps) vfft_il2p_execute_fwd(ps, zi, zs);
        char pairs[16]; snprintf(pairs, 16, "%dx%d", R1, R2);
        const char *res[5];
        int pl[5] = { VFFT_OUTOFPLACE, VFFT_OUTOFPLACE, VFFT_INPLACE, VFFT_INPLACE, VFFT_OUTOFPLACE };
        int od[5] = { VFFT_ORDER_DEFAULT, VFFT_ORDER_NATURAL, VFFT_ORDER_DEFAULT,
                      VFFT_ORDER_NATURAL, VFFT_ORDER_SCRAMBLED };
        for (int a = 0; a < 5; a++) {
            vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.placement = pl[a]; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.order = od[a]; cfg.rigor = VFFT_MEASURE;
            memset(zt, 0, nb);
            vfft_plan h = vfft_create(&cfg);
            if (!h) { res[a] = "noplan"; continue; }
            if (pl[a] == VFFT_INPLACE) { memcpy(zt, zi, nb); vfft_execute(h, VFFT_FORWARD, zt, NULL, zt, NULL); }
            else vfft_execute(h, VFFT_FORWARD, zi, NULL, zt, NULL);
            vfft_destroy(h);
            res[a] = which(zt, zh, zs, nb, ps != NULL);
        }
        printf("%-6d %-8s %-8s %-8s %-8s %-8s %-8s\n", N, pairs,
               res[0], res[1], res[2], res[3], res[4]);
        free(zi); free(zh); free(zs); free(zt);
        if (ph) vfft_il2p_destroy(ph);
        if (ps) vfft_il2p_destroy(ps);
    }
    return 0;
}
