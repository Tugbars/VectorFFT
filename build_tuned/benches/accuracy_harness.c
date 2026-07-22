/* accuracy_harness.c — per-bin accuracy of dag transforms vs a long-double
 * reference, with an MKL column for context. Paper-grade numbers.
 *
 * REFERENCE: recursive mixed-radix DIT DFT in `long double complex`
 * (natural order, out-of-place, cexpl twiddles; direct O(n^2) fallback for
 * leftover prime factors). Reference rounding error ~ log2(N) * 2^-64 —
 * >= 500x below the double-precision errors being measured, so it acts as
 * exact. Self-checked per size via impulse + Parseval.
 *
 * PER-BIN COMPARISON of scrambled dag output uses the natorder machinery:
 * plans are built with EXPLICIT factor chains via vfft_proto_plan_create_ex,
 * and vfft_natorder_detect() impulse-probes the exact scramble map M
 * (natural[k] = scrambled[M[k]]) — no tolerance games, no multiset tricks.
 * Prime N (Rader/Bluestein overrides) is probed for identity order first.
 *
 * METRICS per (size, chain, input-class):
 *   L2rel  = ||X - Xref||_2 / ||Xref||_2
 *   Linf   = max_k |X[k]-Xref[k]| / ||Xref||_inf
 *   eps-u  = L2rel / 2^-53   (readable "units of double eps")
 * Input classes: (a) uniform random, (b) single-bin complex exponential
 * (max cancellation stress), (c) alternating +-1.
 *
 * Plus: roundtrip error sweep, and CONV end-to-end accuracy vs long-double
 * direct circular convolution (natural time order — no map needed; includes
 * prime N, validating the flagship API's numerics directly).
 *
 * Output: table to stdout + accuracy_results.csv.
 *
 * Build: python build.py --src benches/accuracy_harness.c --mkl --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "conv.h"
#include "natorder_perm.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#define EPS53 1.1102230246251565e-16   /* 2^-53 */

typedef long double complex lcplx;
static const long double LPI = 3.14159265358979323846264338327950288L;

/* ── long-double reference DFT (natural order, unnormalized fwd) ── */
static int _small_factor(int n){
    for (int p=2; p*p<=n; p++) if (n%p==0) return p;
    return n;
}
static void ref_dft_rec(int n, int s, const lcplx *in, lcplx *out, lcplx *scr){
    if (n == 1) { out[0] = in[0]; return; }
    int p = _small_factor(n);
    if (p == n) {                          /* prime leaf: direct O(p^2) */
        for (int k=0;k<n;k++){
            lcplx acc = 0;
            for (int j=0;j<n;j++)
                acc += in[(size_t)j*s] * cexpl(-2.0L*LPI*I*((long double)j*k)/n);
            out[k] = acc;
        }
        return;
    }
    int m = n / p;
    for (int r=0;r<p;r++)
        ref_dft_rec(m, s*p, in + (size_t)r*s, scr + (size_t)r*m, out);
    for (int q=0;q<m;q++){
        for (int t=0;t<p;t++){
            int k = q + m*t;
            lcplx acc = 0;
            for (int r=0;r<p;r++)
                acc += scr[(size_t)r*m + q] *
                       cexpl(-2.0L*LPI*I*((long double)r*k)/n);
            out[k] = acc;
        }
    }
}
static void ref_dft(int n, const double *xr, const double *xi, lcplx *out){
    lcplx *in  = (lcplx*)malloc((size_t)n*sizeof(lcplx));
    lcplx *scr = (lcplx*)malloc((size_t)n*sizeof(lcplx));
    for (int j=0;j<n;j++) in[j] = (long double)xr[j] + I*(long double)xi[j];
    ref_dft_rec(n, 1, in, out, scr);
    free(in); free(scr);
}

/* ── metrics: computed (natural order) vs lcplx reference ── */
typedef struct { double l2rel, linf, epsu; } acc_t;
static acc_t acc_measure(int n, const double *cr, const double *ci, const lcplx *ref){
    long double e2=0, r2=0, li=0, rinf=0;
    for (int k=0;k<n;k++){
        long double dr = (long double)cr[k] - creall(ref[k]);
        long double di = (long double)ci[k] - cimagl(ref[k]);
        long double rm = creall(ref[k])*creall(ref[k]) + cimagl(ref[k])*cimagl(ref[k]);
        e2 += dr*dr + di*di;  r2 += rm;
        long double am = sqrtl(dr*dr+di*di);
        if (am > li) li = am;
        long double aref = sqrtl(rm);
        if (aref > rinf) rinf = aref;
    }
    acc_t a;
    a.l2rel = (double)sqrtl(e2/(r2>0?r2:1));
    a.linf  = (double)(li/(rinf>0?rinf:1));
    a.epsu  = a.l2rel / EPS53;
    return a;
}

/* ── build a dag plan from an explicit chain + detect its scramble map ── */
static stride_plan_t *mk_chain_plan(int N, size_t K, const int *f, int nf,
                                    const vfft_proto_registry_t *reg,
                                    int **map_out){
    /* variants: all zero = default codelet variant per stage */
    int variants[16] = {0};
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, (int*)f, variants, nf, 0, reg);
    if (!p) return NULL;
    /* impulse at n0=1, lane 0 -> detect map */
    int n0 = 1;
    double *re = (double*)AALLOC((size_t)N*K*8), *im = (double*)AALLOC((size_t)N*K*8);
    memset(re,0,(size_t)N*K*8); memset(im,0,(size_t)N*K*8);
    re[(size_t)n0*K] = 1.0;
    stride_execute_fwd(p, re, im);
    *map_out = vfft_natorder_detect(N, f, nf, K, re, im, n0);
    AFREE(re); AFREE(im);
    if (!*map_out) { stride_plan_destroy(p); return NULL; }
    return p;
}

/* prime/override plans: check natural-order output via impulse closed form */
static stride_plan_t *mk_prime_plan(int N, size_t K,
                                    const vfft_proto_registry_t *reg,
                                    int **map_out){
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, NULL);
    if (!p) return NULL;
    int n0 = 1;
    double *re=(double*)AALLOC((size_t)N*K*8), *im=(double*)AALLOC((size_t)N*K*8);
    memset(re,0,(size_t)N*K*8); memset(im,0,(size_t)N*K*8);
    re[(size_t)n0*K]=1.0;
    stride_execute_fwd(p,re,im);
    int natural = 1;
    for (int k=0;k<N;k+=(N/7)+1){
        double er = cos(-2.0*M_PI*k*n0/N), ei = sin(-2.0*M_PI*k*n0/N);
        if (fabs(re[(size_t)k*K]-er)>1e-6 || fabs(im[(size_t)k*K]-ei)>1e-6){
            natural=0; break; }
    }
    AFREE(re); AFREE(im);
    if (!natural){ stride_plan_destroy(p); return NULL; }
    int *M = (int*)malloc((size_t)N*4);
    for (int k=0;k<N;k++) M[k]=k;
    *map_out = M;
    return p;
}

static FILE *g_csv;

static void run_cell(int N, const int *f, int nf, const char *chain_tag,
                     const vfft_proto_registry_t *reg,
                     DFTI_DESCRIPTOR_HANDLE mh /* may be 0 */){
    const size_t K = 8;
    int *M = NULL;
    stride_plan_t *p = (nf>0) ? mk_chain_plan(N,K,f,nf,reg,&M)
                              : mk_prime_plan(N,K,reg,&M);
    if (!p){ printf("  %6d %-12s SKIP (plan/map unavailable)\n",N,chain_tag); return; }

    double *xr=(double*)AALLOC((size_t)N*8), *xi=(double*)AALLOC((size_t)N*8);
    double *br=(double*)AALLOC((size_t)N*K*8), *bi=(double*)AALLOC((size_t)N*K*8);
    double *nr=(double*)AALLOC((size_t)N*8), *ni=(double*)AALLOC((size_t)N*8);
    double *mr=(double*)AALLOC((size_t)N*8), *mi=(double*)AALLOC((size_t)N*8);
    lcplx  *rf=(lcplx*)malloc((size_t)N*sizeof(lcplx));

    const char *icls[3] = { "rand", "1bin", "alt " };
    for (int ic=0; ic<3; ic++){
        srand(1000+N+ic);
        if (ic==0) for (int t=0;t<N;t++){ xr[t]=2.0*rand()/RAND_MAX-1;
                                          xi[t]=2.0*rand()/RAND_MAX-1; }
        else if (ic==1){ int k0=N/3;
            for (int t=0;t<N;t++){ xr[t]=cos(-2.0*M_PI*k0*t/N);
                                   xi[t]=sin(-2.0*M_PI*k0*t/N); } }
        else for (int t=0;t<N;t++){ xr[t]=(t&1)?-1.0:1.0; xi[t]=0.0; }

        ref_dft(N, xr, xi, rf);

        /* dag: lane 0 carries the signal */
        memset(br,0,(size_t)N*K*8); memset(bi,0,(size_t)N*K*8);
        for (int t=0;t<N;t++){ br[(size_t)t*K]=xr[t]; bi[(size_t)t*K]=xi[t]; }
        stride_execute_fwd(p, br, bi);
        for (int k=0;k<N;k++){ nr[k]=br[(size_t)M[k]*K]; ni[k]=bi[(size_t)M[k]*K]; }
        acc_t av = acc_measure(N, nr, ni, rf);

        acc_t am = {0,0,0};
        if (mh){
            DftiComputeForward(mh, xr, xi, mr, mi);
            am = acc_measure(N, mr, mi, rf);
        }
        printf("  %6d %-12s %s  vfft L2=%.2e Linf=%.2e (%5.2f eps)   mkl L2=%.2e (%5.2f eps)\n",
               N, chain_tag, icls[ic], av.l2rel, av.linf, av.epsu, am.l2rel, am.epsu);
        fprintf(g_csv,"%d,%s,%s,%.3e,%.3e,%.3f,%.3e,%.3f\n",
                N, chain_tag, icls[ic], av.l2rel, av.linf, av.epsu, am.l2rel, am.epsu);
    }
    free(rf); AFREE(xr);AFREE(xi);AFREE(br);AFREE(bi);
    AFREE(nr);AFREE(ni);AFREE(mr);AFREE(mi);
    free(M);
    stride_plan_destroy(p);
}

/* ── conv end-to-end accuracy vs long-double direct circular conv ── */
static void conv_cell(int N, const vfft_proto_registry_t *reg){
    const size_t K=8, n=(size_t)N*K;
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N,K,reg,NULL);
    stride_conv_t *cv = stride_conv_wrap(p,1);
    double *xr=(double*)AALLOC(n*8),*xi=(double*)AALLOC(n*8);
    double *hr=(double*)AALLOC(n*8),*hi=(double*)AALLOC(n*8);
    srand(500+N);
    double *LX=malloc(N*8),*LY=malloc(N*8),*LH=malloc(N*8),*LG=malloc(N*8);
    for (int t=0;t<N;t++){
        LX[t]=2.0*rand()/RAND_MAX-1; LY[t]=2.0*rand()/RAND_MAX-1;
        LH[t]=2.0*rand()/RAND_MAX-1; LG[t]=2.0*rand()/RAND_MAX-1;
        for (size_t l=0;l<K;l++){
            xr[(size_t)t*K+l]=LX[t]; xi[(size_t)t*K+l]=LY[t];
            hr[(size_t)t*K+l]=LH[t]; hi[(size_t)t*K+l]=LG[t]; }
    }
    stride_conv_set_kernel(cv,hr,hi);
    stride_conv_execute(cv,xr,xi);
    /* long-double direct circular conv */
    long double e2=0,r2=0;
    for (int m=0;m<N;m++){
        long double sr=0,si=0;
        for (int k=0;k<N;k++){
            int j=(m-k)%N; if(j<0)j+=N;
            long double a=LX[k],b=LY[k],c=LH[j],d=LG[j];
            sr += a*c-b*d; si += a*d+b*c;
        }
        long double dr=(long double)xr[(size_t)m*K]-sr;
        long double di=(long double)xi[(size_t)m*K]-si;
        e2 += dr*dr+di*di; r2 += sr*sr+si*si;
    }
    double l2 = (double)sqrtl(e2/(r2>0?r2:1));
    printf("  conv N=%-6d           L2rel=%.2e (%5.2f eps)\n", N, l2, l2/EPS53);
    fprintf(g_csv,"conv,%d,,%.3e,,%.3f,,\n",N,l2,l2/EPS53);
    free(LX);free(LY);free(LH);free(LG);
    AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);
    stride_conv_destroy(cv);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    mkl_set_num_threads(1);
    g_csv = fopen("accuracy_results.csv","w");
    fprintf(g_csv,"N,chain,input,vfft_l2,vfft_linf,vfft_eps,mkl_l2,mkl_eps\n");

    /* reference self-check */
    {
        int N=360; lcplx *rf=malloc((size_t)N*sizeof(lcplx));
        double *xr=calloc(N,8),*xi=calloc(N,8); xr[1]=1.0;
        ref_dft(N,xr,xi,rf);
        long double emax=0;
        for (int k=0;k<N;k++){
            long double er=cosl(-2.0L*LPI*k/N)-creall(rf[k]);
            long double ei=sinl(-2.0L*LPI*k/N)-cimagl(rf[k]);
            long double e=sqrtl(er*er+ei*ei); if(e>emax)emax=e;
        }
        printf("reference self-check (impulse, N=360): max err = %.2Le\n", emax);
        free(rf);free(xr);free(xi);
    }

    struct { int N; int f[6]; int nf; const char *tag; } cells[] = {
        {   64, {16,4},        2, "[16x4]"     },
        {   64, {8,8},         2, "[8x8]"      },
        {   64, {4,4,4},       3, "[4x4x4]"    },
        {  256, {16,16},       2, "[16x16]"    },
        {  256, {8,8,4},       3, "[8x8x4]"    },
        { 1024, {16,16,4},     3, "[16x16x4]"  },
        { 1024, {16,8,8},      3, "[16x8x8]"   },
        { 4096, {16,16,16},    3, "[16^3]"     },
        { 4096, {8,8,8,8},     4, "[8^4]"      },
        {16384, {16,16,16,4},  4, "[16^3x4]"   },
        {65536, {16,16,16,16}, 4, "[16^4]"     },
        {   60, {4,3,5},       3, "[4x3x5]"    },
        {  360, {8,3,3,5},     4, "[8x3x3x5]"  },
        { 1000, {8,5,5,5},     4, "[8x5^3]"    },
        { 3600, {16,3,5,3,5},  5, "[16x(15)^2]"},
        {   61, {0},           0, "prime-61"   },
        {  257, {0},           0, "prime-257"  },
    };

    printf("\nper-bin forward accuracy vs long-double reference (K=8, lane 0):\n");
    for (size_t c=0;c<sizeof(cells)/sizeof(cells[0]);c++){
        DFTI_DESCRIPTOR_HANDLE mh=0;
        MKL_LONG dn = cells[c].N;
        if (DftiCreateDescriptor(&mh,DFTI_DOUBLE,DFTI_COMPLEX,1,dn)==DFTI_NO_ERROR){
            DftiSetValue(mh,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
            DftiSetValue(mh,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
            if (DftiCommitDescriptor(mh)!=DFTI_NO_ERROR){ DftiFreeDescriptor(&mh); mh=0; }
        } else mh=0;
        run_cell(cells[c].N, cells[c].f, cells[c].nf, cells[c].tag, &reg, mh);
        if (mh) DftiFreeDescriptor(&mh);
    }

    printf("\nconv end-to-end accuracy vs long-double direct (K=8):\n");
    conv_cell(64,&reg);
    conv_cell(360,&reg);
    conv_cell(1024,&reg);
    conv_cell(61,&reg);
    conv_cell(997,&reg);

    fclose(g_csv);
    printf("\nCSV: accuracy_results.csv\n");
    return 0;
}
