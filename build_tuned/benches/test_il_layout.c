/* test_il_layout.c — interleaved boundary support (P1a).
 *
 * Gates:
 *   1. KERNEL UNIT: il2sp∘sp2il and sp2il∘il2sp identity, bit-exact, odd
 *      tails included.
 *   2. WRAPPER BIT-EQUIVALENCE, every fft type: for a 1D lane-major plan
 *      (N=64,K=8), fftnd rank-2 (32x48), rank-3 (16x12x20 + prime 8x61x4),
 *      rank-4 (8x12x10x16): stride_il_fwd(z) output must memcmp-equal the
 *      manual route (il2sp -> split fwd -> sp2il). Conversions are exact,
 *      so equality is bit-level.
 *   3. IL roundtrip: il_bwd(il_fwd(z)) == Ntot * z.
 *   4. STRIDED + MT interplay: rank-3 32x32x64 with strided rows, T in
 *      {1,4}: IL output bit-identical across T.
 *   5. r2c IL-out: stride_plan_nd_r2c_il roundtrip == Ntot * x, and the
 *      interleaved spectrum pairs match the split plan's (re,im) bit-exact.
 *   6. TAX BENCH: 64^3 IL wrapper vs split, interleaved rounds — the P1a
 *      two-sweep ceiling, printed for the record.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftnd.h"
#include "fftnd_r2c.h"
#include "il_layout.h"
#include "generator/generated/registry.h"
#include <x86intrin.h>

#if defined(_WIN32)
#define AALLOC(n) _aligned_malloc((n),64)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#endif
static int g_fail = 0;
static void chk(const char *t, int ok){ if(!ok)g_fail++;
    printf("  %-44s %s\n", t, ok?"OK":"**FAIL**"); }

static stride_plan_t *mknd(int rank, const int *N,
                           const vfft_proto_registry_t *reg){
    return stride_plan_nd(rank, N, (vfft_proto_registry_t *)reg);
}

static void wrap_cell(const char *tag, stride_plan_t *p,
                      const vfft_proto_registry_t *reg){
    (void)reg;
    size_t n = (size_t)p->N * (p->K ? p->K : 1);
    stride_il_t *w = stride_il_wrap(p, 1);
    double *z=AALLOC(n*16),*z2=AALLOC(n*16),*zr=AALLOC(n*16);
    double *re=AALLOC(n*8),*im=AALLOC(n*8);
    srand(11+(int)n);
    for(size_t i=0;i<2*n;i++) z[i]=z2[i]=zr[i]=2.0*rand()/RAND_MAX-1;
    /* IL path */
    stride_il_fwd(w, z);
    /* manual route */
    vfft_il2sp(z2, re, im, n);
    stride_execute_fwd(w->plan, re, im);
    vfft_sp2il(re, im, z2, n);
    int bit = !memcmp(z, z2, n*16);
    /* IL roundtrip */
    stride_il_bwd(w, z);
    double sc=(double)w->plan->N, rt=0, mx=0;
    for(size_t i=0;i<2*n;i++){ if(fabs(zr[i])>mx)mx=fabs(zr[i]);
        double e=fabs(z[i]-sc*zr[i]); if(e>rt)rt=e; }
    rt/=sc*(mx>0?mx:1);
    char buf[96]; snprintf(buf,96,"%s bit=%s rt=%.1e",tag,bit?"EXACT":"NO",rt);
    chk(buf, bit && rt<1e-12);
    free(z);free(z2);free(zr);free(re);free(im);
    stride_il_destroy(w);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    printf("interleaved layout (P1a)\n");

    /* 1. kernel unit */
    {
        size_t n=1003;                         /* odd tail */
        double *z=AALLOC((n+1)*16),*re=AALLOC((n+1)*8),*im=AALLOC((n+1)*8);
        double *z2=AALLOC((n+1)*16);
        srand(1); for(size_t i=0;i<2*n;i++) z[i]=rand()*1e-9;
        vfft_il2sp(z,re,im,n); vfft_sp2il(re,im,z2,n);
        int ok=!memcmp(z,z2,n*16);
        for(size_t f=0;f<n && ok;f++)
            if(re[f]!=z[2*f]||im[f]!=z[2*f+1]) ok=0;
        chk("kernel il2sp/sp2il identity (n=1003)", ok);
        free(z);free(re);free(im);free(z2);
    }

    /* 2+3. wrapper across the fft types */
    wrap_cell("1D N=64 K=8 (lane-major IL)",
              vfft_proto_auto_plan_dispatch(64,8,&reg,NULL), &reg);
    { int a[2]={32,48};        wrap_cell("rank-2 32x48", mknd(2,a,&reg), &reg); }
    { int a[3]={16,12,20};     wrap_cell("rank-3 16x12x20", mknd(3,a,&reg), &reg); }
    { int a[3]={8,61,4};       wrap_cell("rank-3 8x61x4 (prime)", mknd(3,a,&reg), &reg); }
    { int a[4]={8,12,10,16};   wrap_cell("rank-4 8x12x10x16", mknd(4,a,&reg), &reg); }

    /* 4. strided + MT under IL */
    {
        int a[3]={32,32,64}; size_t n=(size_t)32*32*64;
        stride_set_num_threads(8);
        stride_il_t *w = stride_il_wrap(mknd(3,a,&reg), 1);
        double *z=AALLOC(n*16),*z1=AALLOC(n*16);
        srand(3); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
        stride_set_num_threads(1);
        memcpy(z1,z,n*16); stride_il_fwd(w,z1);
        double *z4=AALLOC(n*16);
        stride_set_num_threads(4);
        memcpy(z4,z,n*16); stride_il_fwd(w,z4);
        stride_set_num_threads(1);
        chk("rank-3 32x32x64 IL, T={1,4} bit", !memcmp(z1,z4,n*16));
        free(z);free(z1);free(z4); stride_il_destroy(w);
    }

    /* 5. r2c IL-out */
    {
        int a[3]={16,12,20};
        stride_plan_t *ps = stride_plan_nd_r2c(3,a,&reg);
        stride_plan_t *pi = stride_plan_nd_r2c_il(3,a,&reg);
        stride_fftnd_r2c_data_t *ds=(stride_fftnd_r2c_data_t*)ps->override_data;
        size_t nre=(size_t)16*12*20, ncx=ds->R*ds->hp1;
        double *x=AALLOC(nre*8);
        double *sr=AALLOC(nre*8),*si=AALLOC(ncx*8);
        double *zi=AALLOC(2*ncx*8 > nre*8 ? 2*ncx*8 : nre*8);
        srand(7); for(size_t i=0;i<nre;i++) x[i]=2.0*rand()/RAND_MAX-1;
        memcpy(sr,x,nre*8); stride_execute_fwd(ps,sr,si);
        memcpy(zi,x,nre*8); stride_execute_fwd(pi,zi,NULL);
        int bit=1;
        for(size_t f=0;f<ncx && bit;f++)
            if(zi[2*f]!=sr[f] || zi[2*f+1]!=si[f]) bit=0;
        stride_execute_bwd(pi,zi,NULL);
        double rt=0,mx=0;
        for(size_t i=0;i<nre;i++){ if(fabs(x[i])>mx)mx=fabs(x[i]);
            double e=fabs(zi[i]-(double)nre*x[i]); if(e>rt)rt=e; }
        rt/=(double)nre*mx;
        char b[96]; snprintf(b,96,"r2c IL-out 16x12x20 bit=%s rt=%.1e",
                             bit?"EXACT":"NO",rt);
        chk(b, bit && rt<1e-12);
        free(x);free(sr);free(si);free(zi);
        stride_plan_destroy(ps); stride_plan_destroy(pi);
    }

    /* 6. tax bench 64^3 */
    {
        int a[3]={64,64,64}; size_t n=(size_t)64*64*64;
        stride_il_t *w = stride_il_wrap(mknd(3,a,&reg), 1);
        double *z=AALLOC(n*16),*re=AALLOC(n*8),*im=AALLOC(n*8);
        srand(4); for(size_t i=0;i<2*n;i++) z[i]=rand()*1e-9;
        vfft_il2sp(z,re,im,n);
        int reps=6; double ts=1e18,ti=1e18;
        for(int w2=0;w2<2;w2++){ stride_execute_fwd(w->plan,re,im);
                                 stride_il_fwd(w,z); }
        for(int t=0;t<7;t++){ double t0=(double)__rdtsc();
            for(int i=0;i<reps;i++) stride_execute_fwd(w->plan,re,im);
            double v=((double)__rdtsc()-t0)/reps; if(v<ts)ts=v;
            t0=(double)__rdtsc();
            for(int i=0;i<reps;i++) stride_il_fwd(w,z);
            v=((double)__rdtsc()-t0)/reps; if(v<ti)ti=v; }
        printf("  64^3 split %.0f | IL(wrapper) %.0f | tax %.3fx  (P1a two-sweep ceiling)\n",
               ts, ti, ti/ts);
        free(z);free(re);free(im); stride_il_destroy(w);
    }

    printf(g_fail?"\n%d FAILURE(S)\n":"\nALL PASS\n",g_fail);
    return g_fail?1:0;
}
