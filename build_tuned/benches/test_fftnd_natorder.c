/* test_fftnd_natorder.c — validate the per-axis natural-order maps.
 *
 * Gates:
 *   1. PER-BIN vs REFERENCE: fwd -> gather(maps) must match a long-double
 *      multi-dim reference DFT elementwise (L2 <= ~10 eps). This is ALSO the
 *      first per-bin external validation of the whole fftnd transform at
 *      rank > 1 (previous gates were roundtrip / Parseval / MKL-multiset).
 *   2. PRIME AXES -> IDENTITY: Rader/Bluestein axes emit natural order;
 *      the probe must return identity maps for them.
 *   3. GATHER∘SCATTER = id (and scatter∘gather = id): exact.
 *   4. SOLVER PATTERN: scatter a natural-order multiplier into scrambled
 *      placement once, run fwd -> pointwise -> bwd, compare against the
 *      gather-multiply-scatter route: bit-identical, no reorder on the
 *      per-solve path.
 *
 * Build: python build.py --src benches/test_fftnd_natorder.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "fftnd_natorder.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#define EPS53 1.1102230246251565e-16

typedef long double complex lcplx;
static const long double LPI = 3.14159265358979323846264338327950288L;
static int g_fail = 0;

/* long-double direct 1D DFT along one axis of a cube (O(N^2) per pencil;
 * test sizes are small) */
static void ref_axis(lcplx *x, int rank, const int *N, int m) {
    size_t K = 1, O = 1;
    for (int i = m + 1; i < rank; i++) K *= (size_t)N[i];
    for (int i = 0; i < m; i++) O *= (size_t)N[i];
    int n = N[m];
    lcplx *tmp = (lcplx *)malloc((size_t)n * sizeof(lcplx));
    for (size_t o = 0; o < O; o++)
        for (size_t l = 0; l < K; l++) {
            lcplx *base = x + o * (size_t)n * K + l;
            for (int k = 0; k < n; k++) {
                lcplx acc = 0;
                for (int j = 0; j < n; j++)
                    acc += base[(size_t)j * K] *
                           cexpl(-2.0L * LPI * I * ((long double)j * k) / n);
                tmp[k] = acc;
            }
            for (int k = 0; k < n; k++) base[(size_t)k * K] = tmp[k];
        }
    free(tmp);
}

static stride_plan_t *mk(int rank, const int *N,
                         const vfft_proto_registry_t *reg) {
    stride_fftnd_data_t tmp; memset(&tmp, 0, sizeof tmp);
    tmp.rank = rank;
    for (int m = 0; m < rank; m++) tmp.N[m] = N[m];
    _fftnd_fill_ok(&tmp);
    size_t B = _fftnd_choose_tile(N[rank-1], tmp.O[rank-1]);
    stride_plan_t *pl[FFTND_MAX_RANK] = {0};
    for (int m = 0; m < rank; m++) {
        size_t Kp = (m == rank-1) ? B : tmp.K[m];
        pl[m] = vfft_proto_auto_plan_dispatch(N[m], Kp, reg, NULL);
        if (!pl[m]) return NULL;
    }
    return stride_plan_nd_from(rank, N, B, -1, NULL, pl);
}

static void cell(int rank, const int *N, const vfft_proto_registry_t *reg) {
    size_t n = 1; for (int m=0;m<rank;m++) n *= (size_t)N[m];
    stride_plan_t *p = mk(rank, N, reg);
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)p->override_data;

    int *maps[FFTND_MAX_RANK] = {0};
    if (!fftnd_natorder_maps(p, maps)) {
        printf("  r%d ", rank);
        for (int m=0;m<rank;m++) printf("%d%s", N[m], m+1<rank?"x":"");
        printf("  MAP PROBE REFUSED\n");
        g_fail++;
        stride_plan_destroy(p);
        return;
    }
    /* prime axes must be identity */
    int prime_id = 1;
    for (int m=0;m<rank;m++){
        int Nm=N[m], is_prime = Nm>2;
        for (int f=2; f*f<=Nm; f++) if (Nm%f==0) { is_prime=0; break; }
        if (Nm<=2) is_prime=0;
        if (is_prime)
            for (int k=0;k<Nm;k++) if (maps[m][k]!=k) { prime_id=0; break; }
    }

    /* per-bin vs long-double reference */
    double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*sr=AALLOC(n*8),*si=AALLOC(n*8);
    double *nr=AALLOC(n*8),*ni=AALLOC(n*8);
    srand(77 + N[0]);
    lcplx *ref = (lcplx*)malloc(n*sizeof(lcplx));
    for (size_t i=0;i<n;i++){
        xr[i]=2.0*rand()/RAND_MAX-1; xi[i]=2.0*rand()/RAND_MAX-1;
        ref[i] = (long double)xr[i] + I*(long double)xi[i];
    }
    for (int m=0;m<rank;m++) ref_axis(ref, rank, N, m);
    memcpy(sr,xr,n*8); memcpy(si,xi,n*8);
    stride_execute_fwd(p, sr, si);
    fftnd_natorder_gather(d, maps, sr, si, nr, ni);
    long double e2=0,r2=0;
    for (size_t i=0;i<n;i++){
        long double dr=(long double)nr[i]-creall(ref[i]);
        long double di=(long double)ni[i]-cimagl(ref[i]);
        e2 += dr*dr+di*di;
        r2 += creall(ref[i])*creall(ref[i])+cimagl(ref[i])*cimagl(ref[i]);
    }
    double l2 = (double)sqrtl(e2/(r2>0?r2:1));

    /* gather∘scatter identity (exact) */
    double *tr=AALLOC(n*8),*ti=AALLOC(n*8),*ur=AALLOC(n*8),*ui=AALLOC(n*8);
    fftnd_natorder_scatter(d, maps, nr, ni, tr, ti);
    fftnd_natorder_gather(d, maps, tr, ti, ur, ui);
    int inv = !memcmp(ur,nr,n*8) && !memcmp(ui,ni,n*8) &&
              !memcmp(tr,sr,n*8) && !memcmp(ti,si,n*8);

    /* solver pattern: multiplier scattered once == gather-mul-scatter */
    double *mr=AALLOC(n*8),*mi=AALLOC(n*8),*msr=AALLOC(n*8),*msi=AALLOC(n*8);
    srand(5);
    for (size_t i=0;i<n;i++){ mr[i]=2.0*rand()/RAND_MAX-1;
                              mi[i]=2.0*rand()/RAND_MAX-1; }
    fftnd_natorder_scatter(d, maps, mr, mi, msr, msi);
    /* route A: multiply scrambled spectrum by scattered multiplier */
    for (size_t i=0;i<n;i++){
        double a=sr[i],b=si[i],c=msr[i],e=msi[i];
        tr[i]=a*c-b*e; ti[i]=a*e+b*c;
    }
    /* route B: gather, multiply natural, scatter */
    for (size_t i=0;i<n;i++){
        double a=nr[i],b=ni[i],c=mr[i],e=mi[i];
        ur[i]=a*c-b*e; ui[i]=a*e+b*c;
    }
    fftnd_natorder_scatter(d, maps, ur, ui, msr, msi);
    int solver = !memcmp(tr,msr,n*8) && !memcmp(ti,msi,n*8);

    int ok = l2 < 10*EPS53 && prime_id && inv && solver;
    if (!ok) g_fail++;
    printf("  r%d ", rank);
    for (int m=0;m<rank;m++) printf("%d%s", N[m], m+1<rank?"x":"");
    printf("  per-bin L2=%.2e (%4.2f eps)  prime-id=%s  inv=%s  solver=%s  %s\n",
           l2, l2/EPS53, prime_id?"Y":"N", inv?"EXACT":"NO",
           solver?"EXACT":"NO", ok?"OK":"**FAIL**");

    for (int m=0;m<rank;m++) free(maps[m]);
    free(ref);
    AFREE(xr);AFREE(xi);AFREE(sr);AFREE(si);AFREE(nr);AFREE(ni);
    AFREE(tr);AFREE(ti);AFREE(ur);AFREE(ui);
    AFREE(mr);AFREE(mi);AFREE(msr);AFREE(msi);
    stride_plan_destroy(p);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    printf("fftnd natural-order maps\n");

    int a[2]={32,48};       cell(2,a,&reg);
    int b[3]={16,12,20};    cell(3,b,&reg);
    int c[3]={8,61,4};      cell(3,c,&reg);    /* prime middle -> identity */
    int e[4]={8,12,10,16};  cell(4,e,&reg);
    int f[4]={13,8,8,7};    cell(4,f,&reg);    /* primes at both ends      */

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
