/* test_oop_mt_slice.c — DEFINITIVE OOP MT correctness: whole-K vs K-SPLIT on the SAME plan object.
 * The public-API MT-vs-ST test can't pin the plan for K%8!=0 (OOP wisdom rejects those K -> pm/ps
 * calibrate independently -> different kinds -> spurious diff). Here we build ONE plan and run it two
 * ways: (a) whole-batch via vfft_oop_execute_fwd/bwd, (b) the exact lane-slice loop _oop_mt uses in
 * vfft.c (S=8 slabs, last=tail). Same plan + correct slicing => BIT-IDENTICAL. Covers LEAF + MODEB,
 * fwd + bwd, at even + odd K. Threads only decide WHO runs each disjoint slice, so serial==whole proves
 * the math; the slices are non-overlapping lane ranges so the threaded run is race-free by construction.
 * Build: python build.py --src test/test_oop_mt_slice.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

static vfft_proto_registry_t REG;
static int fails = 0;
#define SLAB 8

/* mirror of _oop_slice_fwd in vfft.c, driven over S=8 slabs (last carries the odd-K tail) */
static void kslice_fwd(const vfft_oop_plan_t *p, const double *sr, const double *si, double *dr, double *di)
{
    size_t K = p->K;
    for (size_t k0 = 0; k0 < K; k0 += SLAB) {
        size_t s = (k0 + SLAB <= K) ? SLAB : (K - k0);
        if (p->kind == VFFT_OOP_KIND_LEAF)
            p->leaf(sr + k0, si + k0, dr + k0, di + k0, 0, 0, K, 1, K, 1, s);
        else
            vfft_proto_execute_fwd_oop_jit(p->mb, sr + k0, si + k0, dr + k0, di + k0, s, p->mb_jit_fwd);
    }
}
/* mirror of _oop_slice_bwd in vfft.c (the NEW code): LEAF swap / MODEB per-slice copy + DIF-bwd */
static void kslice_bwd(const vfft_oop_plan_t *p, const double *sr, const double *si, double *dr, double *di)
{
    size_t K = p->K;
    for (size_t k0 = 0; k0 < K; k0 += SLAB) {
        size_t s = (k0 + SLAB <= K) ? SLAB : (K - k0);
        if (p->kind == VFFT_OOP_KIND_MODEB) {
            for (int e = 0; e < p->N; e++) {
                memcpy(dr + (size_t)e * K + k0, sr + (size_t)e * K + k0, s * sizeof(double));
                memcpy(di + (size_t)e * K + k0, si + (size_t)e * K + k0, s * sizeof(double));
            }
            if (p->mb_jit_bwd) p->mb_jit_bwd(p->mb, dr + k0, di + k0, s, p->mb->K, 0);
            else               vfft_proto_execute_bwd_generic(p->mb, dr + k0, di + k0, s);
        } else
            p->leaf(si + k0, sr + k0, di + k0, dr + k0, 0, 0, K, 1, K, 1, s);
    }
}

static double maxdiff(const double *a, const double *b, size_t n)
{ double m = 0; for (size_t i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > m) m = d; } return m; }

static void cell(const char *tag, vfft_oop_plan_t *p, int N, size_t K)
{
    if (!p) { printf("  %-6s N=%-4d K=%-3zu  plan NULL\n", tag, N, K); fails++; return; }
    size_t n = (size_t)N * K;
    double *sr=malloc(n*8),*si=malloc(n*8);
    double *wf_r=malloc(n*8),*wf_i=malloc(n*8),*kf_r=malloc(n*8),*kf_i=malloc(n*8);   /* fwd: whole / kslice */
    double *wb_r=malloc(n*8),*wb_i=malloc(n*8),*kb_r=malloc(n*8),*kb_i=malloc(n*8);   /* bwd */
    srand(101 + N + (int)K);
    for (size_t i=0;i<n;i++){ sr[i]=(double)rand()/RAND_MAX-.5; si[i]=(double)rand()/RAND_MAX-.5; }
    /* FWD: whole-K vs K-split (bit-exact expected) */
    vfft_oop_execute_fwd(p, sr, si, wf_r, wf_i);
    kslice_fwd(p, sr, si, kf_r, kf_i);
    double ef = maxdiff(wf_r,kf_r,n) > maxdiff(wf_i,kf_i,n) ? maxdiff(wf_r,kf_r,n) : maxdiff(wf_i,kf_i,n);
    /* BWD: whole-K vs K-split on the whole-K spectrum wf (bit-exact expected) */
    vfft_oop_execute_bwd(p, wf_r, wf_i, wb_r, wb_i);
    kslice_bwd(p, wf_r, wf_i, kb_r, kb_i);
    double eb = maxdiff(wb_r,kb_r,n) > maxdiff(wb_i,kb_i,n) ? maxdiff(wb_r,kb_r,n) : maxdiff(wb_i,kb_i,n);
    int bad = (ef != 0.0) || (eb != 0.0); if (bad) fails++;
    const char *kn = p->kind==VFFT_OOP_KIND_LEAF?"LEAF":p->kind==VFFT_OOP_KIND_MODEB?"MODEB":"BAILEY2";
    printf("  %-6s N=%-4d K=%-3zu rem%zu  fwd whole-vs-split=%.1e  bwd=%.1e  %s\n",
           kn, N, K, K&3, ef, eb, bad?"<FAIL>":"BIT-EXACT");
    vfft_oop_plan_destroy(p);
    free(sr);free(si);free(wf_r);free(wf_i);free(kf_r);free(kf_i);free(wb_r);free(wb_i);free(kb_r);free(kb_i);
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    vfft_proto_registry_init(&REG);
    printf("# OOP MT: whole-K vs S=8 K-SPLIT on the SAME plan (fwd + bwd). Must be BIT-EXACT.\n");
    int leafK[] = {8,16,23,31};
    int modebK[] = {8,16,23,31};
    int f256[] = {4,4,4,4};      /* 256 = 4^4 -> MODEB via explicit factors */
    printf("== LEAF (N=64, Rule-1) ==\n");
    for (int i=0;i<4;i++) cell("LEAF", vfft_oop_plan_create(64,(size_t)leafK[i],NULL,0,&REG), 64,(size_t)leafK[i]);
    /* Force MODEB directly via _vfft_oop_make_modeb — raw vfft_oop_plan_create(256,factors) hits
     * Rule 2 (BAILEY2) first. This is the same helper Rule 3 / dp / wisdom reach MODEB through. */
    printf("== MODEB (N=256, factors 4x4x4x4, forced via make_modeb) ==\n");
    for (int i=0;i<4;i++) cell("MODEB", _vfft_oop_make_modeb(256,(size_t)modebK[i],f256,NULL,4,&REG), 256,(size_t)modebK[i]);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: OOP MT K-split is BIT-EXACT vs whole-K (fwd+bwd, LEAF+MODEB, odd K)\n", fails);
    return fails ? 1 : 0;
}
