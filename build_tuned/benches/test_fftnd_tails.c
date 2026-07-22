/* test_fftnd_tails.c — the adversarial tail matrix for rank 2–4.
 *
 * The friendly-shape blind spot: multi-dim lane counts are PRODUCTS of axis
 * lengths, so odd axes push K % VW != 0 into every pass (exercising the
 * rem-aware in-place tail — on AVX-512 for the first time through multi-dim,
 * per the coverage map's "emit-present, untested"), and arbitrary R pushes
 * sub-VW row tails into the tiled pass. With strided rows, tails take the
 * PADDED path, which must preserve UNIFORM per-row natural order for any R —
 * validated here by the strongest gate available: per-bin comparison against
 * a long-double rank-d reference THROUGH the natorder maps (whose last-axis
 * identity fast-path is only correct if every row, tails included, is
 * natural).
 *
 * Cells:
 *   G1 (native, odd-K):  7x5x9   (K0=45, K1=9 — both odd; R=35, rem 3@VW8)
 *                        9x7x15  (K0=105, K1=15; R=63, rem 7)
 *                        3x7x5x9 (K=315/45/9 all odd; R=105, rem 1)
 *   G2 (strided tails):  9x5x8   (r8,  R=45, rem 5)   3x5x16 (r16, R=15, rem 7)
 *                        5x9x32  (r32, R=45, rem 5)   7x9x64 (r64, R=63, rem 7)
 * Gates per cell: roundtrip (scale-rel), per-bin vs long-double reference via
 * maps, MT bit T∈{1,2,4,8}; G2 additionally asserts strided actually resolved.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "fftnd_natorder.h"
#include "fftnd_r2c.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#define AALLOC(n) _aligned_malloc((n),64)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#endif
#define EPS53 1.1102230246251565e-16
typedef long double complex lcplx;
static const long double LPI = 3.14159265358979323846264338327950288L;
static int g_fail = 0;

static void ref_axis(lcplx *x, int rank, const int *N, int m) {
    size_t K = 1, O = 1;
    for (int i = m + 1; i < rank; i++) K *= (size_t)N[i];
    for (int i = 0; i < m; i++) O *= (size_t)N[i];
    int n = N[m];
    lcplx *tmp = (lcplx *)malloc((size_t)n * sizeof(lcplx));
    for (size_t o = 0; o < O; o++)
        for (size_t l = 0; l < K; l++) {
            lcplx *b = x + o * (size_t)n * K + l;
            for (int k = 0; k < n; k++) {
                lcplx a = 0;
                for (int j = 0; j < n; j++)
                    a += b[(size_t)j*K] * cexpl(-2.0L*LPI*I*((long double)j*k)/n);
                tmp[k] = a;
            }
            for (int k = 0; k < n; k++) b[(size_t)k*K] = tmp[k];
        }
    free(tmp);
}

static void cell(int rank, const int *N, int want_strided,
                 const vfft_proto_registry_t *reg) {
    size_t n = 1; for (int m=0;m<rank;m++) n *= (size_t)N[m];
    stride_set_num_threads(8);            /* scratch slots for the T sweep */
    stride_plan_t *p = stride_plan_nd(rank, N, (vfft_proto_registry_t *)reg);
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)p->override_data;
    int strided_on = 0;
#ifdef VFFT_STRIDED_ROWS
    strided_on = (d->srow_fwd != NULL);
#endif
    stride_set_num_threads(1);

    double *xr=AALLOC(n*8),*xi=AALLOC(n*8);
    double *sr=AALLOC(n*8),*si=AALLOC(n*8);
    double *cr=AALLOC(n*8),*ci=AALLOC(n*8);
    srand(41 + N[0] + rank);
    for (size_t i=0;i<n;i++){ xr[i]=2.0*rand()/RAND_MAX-1;
                              xi[i]=2.0*rand()/RAND_MAX-1; }

    /* roundtrip */
    memcpy(sr,xr,n*8); memcpy(si,xi,n*8);
    stride_execute_fwd(p,sr,si);
    memcpy(cr,sr,n*8); memcpy(ci,si,n*8);       /* keep spectrum */
    stride_execute_bwd(p,sr,si);
    double sc=(double)n, rt=0, mx=0;
    for (size_t i=0;i<n;i++){ if(fabs(xr[i])>mx)mx=fabs(xr[i]);
        double e=fabs(sr[i]-sc*xr[i])+fabs(si[i]-sc*xi[i]);
        if(e>rt)rt=e; }
    rt/=sc*(mx>0?mx:1);

    /* per-bin via maps (identity fast-path over strided tails is on trial) */
    int *maps[FFTND_MAX_RANK]={0};
    int mok = fftnd_natorder_maps(p, maps);
    double l2 = 1e30;
    if (mok) {
        lcplx *ref = malloc(n*sizeof(lcplx));
        for (size_t i=0;i<n;i++) ref[i]=(long double)xr[i]+I*(long double)xi[i];
        for (int m=0;m<rank;m++) ref_axis(ref,rank,N,m);
        double *nr=AALLOC(n*8),*ni=AALLOC(n*8);
        fftnd_natorder_gather(d, maps, cr, ci, nr, ni);
        long double e2=0,r2=0;
        for (size_t i=0;i<n;i++){
            long double dr=(long double)nr[i]-creall(ref[i]);
            long double di=(long double)ni[i]-cimagl(ref[i]);
            e2+=dr*dr+di*di;
            r2+=creall(ref[i])*creall(ref[i])+cimagl(ref[i])*cimagl(ref[i]);
        }
        l2=(double)sqrtl(e2/(r2>0?r2:1));
        free(ref); free(nr); free(ni);
        for (int m=0;m<rank;m++) free(maps[m]);
    }

    /* MT bit sweep */
    int Ts[3]={2,4,8}, mtb=1;
    for (int ti=0;ti<3;ti++){
        stride_set_num_threads(Ts[ti]);
        memcpy(sr,xr,n*8); memcpy(si,xi,n*8);
        stride_execute_fwd(p,sr,si);
        if (memcmp(sr,cr,n*8)||memcmp(si,ci,n*8)) mtb=0;
    }
    stride_set_num_threads(1);

    int ok = rt<1e-12 && mok && l2<10*EPS53 && mtb
             && (!want_strided || strided_on);
    if(!ok) g_fail++;
    printf("  r%d ", rank);
    for (int m=0;m<rank;m++) printf("%d%s",N[m],m+1<rank?"x":"");
    printf("  rt=%.1e perbin=%.2f eps  MT=%s  strided=%s  %s\n",
           rt, l2/EPS53, mtb?"EXACT":"NO",
           strided_on?"ON":"off", ok?"OK":"**FAIL**");

    free(xr);free(xi);free(sr);free(si);free(cr);free(ci);
    stride_plan_destroy(p);
}

static void r2c_cell(int rank, const int *N,
                     const vfft_proto_registry_t *reg) {
    size_t nre = 1; for (int m=0;m<rank;m++) nre *= (size_t)N[m];
    stride_plan_t *p = stride_plan_nd_r2c(rank, N,
                                          (vfft_proto_registry_t *)reg);
    if (!p) { g_fail++; printf("  r2c r%d create FAIL\n", rank); return; }
    stride_fftnd_r2c_data_t *d = (stride_fftnd_r2c_data_t *)p->override_data;
    size_t ncx = d->R * d->hp1;
    double *x=AALLOC(nre*8);
    double *sr=AALLOC(nre*8 > ncx*8 ? nre*8 : ncx*8), *si=AALLOC(ncx*8);
    srand(17+N[0]);
    for (size_t i=0;i<nre;i++) x[i]=2.0*rand()/RAND_MAX-1;
    memcpy(sr,x,nre*8);
    stride_execute_fwd(p,sr,si);
    /* per-bin along the packed axis vs long-double rank-d real-input ref,
     * row 0 (outer scrambled -- row 0 is the all-axes-origin pencil only if
     * we probe... keep it simple: roundtrip + Parseval-style energy check) */
    double efwd=0;
    { long double s2=0;
      for (size_t i=0;i<nre;i++) s2 += (long double)x[i]*x[i];
      long double f2=0;
      int nl=N[rank-1]; size_t h=d->hp1;
      for (size_t r=0;r<d->R;r++)
        for (size_t k=0;k<h;k++){
            long double m2=(long double)sr[r*h+k]*sr[r*h+k]
                          +(long double)si[r*h+k]*si[r*h+k];
            int kk=(int)k;
            int dup = (kk!=0 && !(nl%2==0 && kk==nl/2));
            f2 += dup ? 2*m2 : m2;
        }
      efwd = fabs((double)(f2/((long double)nre) - s2))
             / (double)(s2>0?s2:1);
    }
    stride_execute_bwd(p,sr,si);
    double rt=0,mx=0;
    for (size_t i=0;i<nre;i++){ if(fabs(x[i])>mx)mx=fabs(x[i]);
        double e=fabs(sr[i]-(double)nre*x[i]); if(e>rt)rt=e; }
    rt/=(double)nre*(mx>0?mx:1);
    int ok = rt<1e-12 && efwd<1e-12;
    if(!ok) g_fail++;
    printf("  r2c r%d ", rank);
    for (int m=0;m<rank;m++) printf("%d%s",N[m],m+1<rank?"x":"");
    printf("  rt=%.1e parseval=%.1e  %s\n", rt, efwd, ok?"OK":"**FAIL**");
    free(x);free(sr);free(si); stride_plan_destroy(p);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("adversarial tails (odd K, sub-VW row tails, strided padded tails)\n");
    { int a[3]={7,5,9};     cell(3,a,0,&reg); }
    { int a[3]={9,7,15};    cell(3,a,0,&reg); }
    { int a[4]={3,7,5,9};   cell(4,a,0,&reg); }
    { int a[3]={9,5,8};     cell(3,a,1,&reg); }
    { int a[3]={3,5,16};    cell(3,a,1,&reg); }
    { int a[3]={5,9,32};    cell(3,a,1,&reg); }
    { int a[3]={7,9,64};    cell(3,a,1,&reg); }
    /* r2c odd-R: rfft-family route-around (explicit-pack / non-fused
     * fallbacks) exercised through the rank-general layer; K_pad %% 8 = 4
     * axes stress the avx512 in-place tails inside the c2c passes. */
    { int a[3]={9,7,16};    r2c_cell(3,a,&reg); }
    { int a[3]={5,9,8};     r2c_cell(3,a,&reg); }
    { int a[4]={3,7,5,24};  r2c_cell(4,a,&reg); }
    printf(g_fail?"\n%d FAILURE(S)\n":"\nALL PASS\n",g_fail);
    return g_fail?1:0;
}
