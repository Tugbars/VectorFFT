/* natorder_t2_norestrict.c — T2 POSITIVE CONTROL for the N=9 restrict-UB theory.
 * Claim: the aliased (dst==src) failure of radix9_n1_oop is caused ONLY by the __restrict__
 * qualifiers licensing gcc to re-load inputs after output stores (register pressure > 16 ymm on the
 * MONOLITHIC path). Test: compile THE EXACT SAME generated source with __restrict__ neutralized
 * (renamed symbol, no other change) and run both variants aliased against a separate-dst reference.
 *   expect: library (restrict) aliased  -> MISMATCH (T1 reproduction)
 *           norestrict        aliased  -> BIT-EXACT
 *           norestrict separate-dst    -> BIT-EXACT vs library separate-dst (same math sanity)
 * Build: python build.py --src test/natorder_t2_norestrict.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>   /* include system headers BEFORE neutralizing __restrict__ */
#include <stddef.h>
#include "executor.h"
#include "planner.h"
#include "oop_plan.h"    /* vfft_oop_leaf_fn / vfft_oop11_fn */

/* ---- the theory: same source, no restrict promise, new symbol ---- */
#define __restrict__                                   /* neutralize the alias promise */
#define radix9_n1_oop_fwd_avx2_UG_UG radix9_n1_oop_fwd_avx2_UG_UG_norestrict
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix9_n1_oop_avx2.c"
#undef radix9_n1_oop_fwd_avx2_UG_UG

static vfft_proto_registry_t REG;
#define N9 9

static int fails = 0;
static double maxdiff(const double *a, const double *b, size_t n)
{ double m=0; for(size_t i=0;i<n;i++){ double d=fabs(a[i]-b[i]); if(d>m)m=d; } return m; }

static void cell(vfft_oop11_fn lib, size_t K)
{
    size_t n = (size_t)N9 * K;
    double *xr=_aligned_malloc(n*8,64), *xi=_aligned_malloc(n*8,64);   /* pristine input   */
    double *rr=_aligned_malloc(n*8,64), *ri=_aligned_malloc(n*8,64);   /* reference (sep)  */
    double *ar=_aligned_malloc(n*8,64), *ai=_aligned_malloc(n*8,64);   /* aliased work     */
    srand(977 + (int)K);
    for (size_t i=0;i<n;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; }

    /* reference: library codelet, separate dst */
    lib(xr, xi, rr, ri, NULL, NULL, K, 1, K, 1, K);

    /* sanity: norestrict separate-dst == library separate-dst (same math) */
    double *sr=_aligned_malloc(n*8,64), *si=_aligned_malloc(n*8,64);
    radix9_n1_oop_fwd_avx2_UG_UG_norestrict(xr, xi, sr, si, NULL, NULL, K, 1, K, 1, K);
    double e_same = maxdiff(sr,rr,n) > maxdiff(si,ri,n) ? maxdiff(sr,rr,n) : maxdiff(si,ri,n);

    /* library, ALIASED (out==in) — expect the T1 corruption */
    memcpy(ar,xr,n*8); memcpy(ai,xi,n*8);
    lib(ar, ai, ar, ai, NULL, NULL, K, 1, K, 1, K);
    double e_lib = maxdiff(ar,rr,n) > maxdiff(ai,ri,n) ? maxdiff(ar,rr,n) : maxdiff(ai,ri,n);

    /* norestrict, ALIASED — the theory says bit-exact */
    memcpy(ar,xr,n*8); memcpy(ai,xi,n*8);
    radix9_n1_oop_fwd_avx2_UG_UG_norestrict(ar, ai, ar, ai, NULL, NULL, K, 1, K, 1, K);
    double e_nr = maxdiff(ar,rr,n) > maxdiff(ai,ri,n) ? maxdiff(ar,rr,n) : maxdiff(ai,ri,n);

    int ok = (e_same==0.0) && (e_nr==0.0);              /* lib aliased is EXPECTED broken; don't gate on it */
    if(!ok) fails++;
    printf("  K=%-3zu  same-math=%.1e  lib-aliased=%.3e%s  NORESTRICT-aliased=%.1e %s\n",
           K, e_same, e_lib, e_lib>1e-12?" (broken, expected)":" (?! passed this run)",
           e_nr, ok?"BIT-EXACT":"<FAIL>");
    _aligned_free(xr);_aligned_free(xi);_aligned_free(rr);_aligned_free(ri);
    _aligned_free(ar);_aligned_free(ai);_aligned_free(sr);_aligned_free(si);
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    vfft_proto_registry_init(&REG);
    vfft_oop11_fn lib = vfft_oop_leaf_fn(N9);
    if(!lib){ printf("no leaf fn for N=9\n"); return 2; }
    printf("# T2: radix9_n1_oop — same generated source, __restrict__ neutralized, aliased dst==src\n");
    size_t Ks[]={4,8,12,23,64};
    for(int i=0;i<5;i++) cell(lib, Ks[i]);
    printf(fails? "\nT2 FAIL: theory falsified — norestrict build still corrupts aliased (dataflow bug, not restrict-UB)\n"
                : "\nT2 PASS: restrict-UB confirmed as the sole cause — no-restrict N=9 is alias-safe\n");
    return fails?1:0;
}
