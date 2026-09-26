/* ztt_mt.h's COLUMN cut (_ztt_mt_columns, lines 126-133: ranges at multiples
 * of 4, stream tw + (lo/4)*twrec, twrec = (R-1)*8) replayed serially at
 * VW=8, vs the same cut at multiples of VW. Natural chain 8.8.8, N = 512. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"
static void run_cols(int T, int grain, vfft_ztt_kfn fn, const double *zin, double *zout, int oshift,
                     const double *tw, int twrec, size_t Ls, size_t OLs, size_t count, const size_t *rb)
{
    const size_t nq = count / grain; int w;
    for (w = 0; w < T; w++) {
        const size_t lo = grain * (nq * (size_t)w / (size_t)T), hi = grain * (nq * (size_t)(w + 1) / (size_t)T);
        if (hi == lo) continue;
        fn(zin + 2 * lo, 0, oshift ? zout + 2 * lo : zout, 0, tw ? tw + (lo / grain) * (size_t)twrec : 0,
           rb ? (const double *)(rb + lo) : 0, Ls, 1, OLs, 0, hi - lo);
    }
}
static void mt_natural(const vfft_ztt_plan_t *p, const double *zin, double *zout, int T, int grain)
{   /* ztt_mt.h:249-261 for a 3-stage chain, arm BLOCKS, out of place */
    const vfft_ztt_kfn *st = p->st_fwd; const double *tw = p->tw; double *W = zout;
    const size_t L = (size_t)p->L[p->nf - 1]; const int Rl = p->chain[p->nf - 1];
    int s;
    run_cols(T, grain, st[0], zin, W, 0, 0, 0, (size_t)p->ncol, 0, (size_t)p->ncol, p->rb);
    for (s = 1; s < p->nf - 1; s++) st[s](W, 0, W, 0, tw + p->twoff[s], 0, (size_t)p->L[s], (size_t)p->Gs[s], 0, 0, (size_t)p->L[s]);
    run_cols(T, grain, st[p->nf - 1], W, zout, 1, tw + p->twoff[p->nf - 1], (Rl - 1) * 2 * grain, L, L, L, 0);
}
int main(void)
{
    const int N = 512, ch[3] = { 8, 8, 8 };
    vfft_ztt_plan_t *p = _ztt_create(N, ch, 3, 0, 1);
    double *x = aligned_alloc(64, 16 * N), *ys = aligned_alloc(64, 16 * N), *ym = aligned_alloc(64, 16 * N);
    int i, T;
    for (i = 0; i < 2 * N; i++) x[i] = sin(1.3 * i);
    vfft_ztt_bind(p, 0);
    vfft_ztt_execute_fwd(p, x, ys);
    for (T = 2; T <= 5; T++) {
        int grain;
        for (grain = 4; grain <= 8; grain += 4) {
            double e = 0;
            memset(ym, 0, 16 * N);
            mt_natural(p, x, ym, T, grain);
            for (i = 0; i < 2 * N; i++) if (fabs(ym[i] - ys[i]) > e) e = fabs(ym[i] - ys[i]);
            printf("T=%d cut grain=%d (%s): max |threaded - serial| = %.3g%s\n", T, grain, grain == 4 ? "ztt_mt.h today" : "grain = VW", e,
                   e ? "  <-- WRONG, silently" : "  (bitwise)");
        }
    }
    return 0;
}
