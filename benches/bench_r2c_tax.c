/* r2c packing-tax attribution bench: public path, prof counters, ablation deltas. */
#include "src/core/vfft.c"   /* bound-prototype: same-TU access to _r2c_prof_* */
#include <mkl_dfti.h>
#include <time.h>
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(void){
    struct { int N; size_t K; int L; } cs[]={{512,256,40},{2000,4,400},{200,4,2000}};
    int NC=3;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    printf("%-12s %10s %10s %10s %10s %10s %10s\n","cell","tot_us","prof_pack","prof_inner","prof_post","mkl_us","il_sweep");
    for(int c=0;c<NC;c++){
        int N=cs[c].N; size_t K=cs[c].K; int L=cs[c].L;
        size_t H=(size_t)N/2+1;
        vfft_config_t cf; memset(&cf,0,sizeof cf);
        cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
        cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.wisdom=w;
        vfft_plan p=vfft_create(&cf);
        if(!p){ printf("[%d,%zu] create FAIL\n",N,K); continue; }
        double *x=aligned_alloc(64,(size_t)N*K*8);
        double *orr=aligned_alloc(64,H*K*8),*oi=aligned_alloc(64,H*K*8);
        double *z=aligned_alloc(64,2*H*K*8);
        srand(3+c); for(size_t i=0;i<(size_t)N*K;i++) x[i]=2.0*rand()/RAND_MAX-1;
        for(int wu=0;wu<3;wu++) vfft_execute(p,VFFT_FORWARD,x,NULL,orr,oi);
        double tv[11];
#ifdef VFFT_R2C_PROFILE
        _r2c_prof_pack=_r2c_prof_inner=_r2c_prof_post=0;
#endif
        for(int t=0;t<11;t++){ double t0=bnow();
            for(int it=0;it<L;it++) vfft_execute(p,VFFT_FORWARD,x,NULL,orr,oi);
            tv[t]=(bnow()-t0)/L; }
        qsort(tv,11,8,dcmp); double tot=tv[5];
        /* MODEL-B live activation (#1 completion): set the never-wired ls_fwd
         * when the prototype codelet's geometry matches, re-time, gate output. */
        double tot_lsb = 0; int lsact = 0;
        if (N == 512) {
            extern __typeof__(*((stride_r2c_data_t*)0)->ls_fwd)
                radix256_r2c_term_ls_r8_fwd_avx2;
            stride_plan_t *sp = ((struct vfft_plan_s *)p)->rplan
                ? ((struct vfft_plan_s *)p)->rplan->stride : NULL;
            if (sp && sp->override_fwd == _r2c_execute_fwd) {
                stride_r2c_data_t *rd = (stride_r2c_data_t *)sp->override_data;
                if (rd->half_N == 256 && !rd->inner->use_dif_forward &&
                    rd->inner->stages[rd->inner->num_stages-1].radix == 8) {
                    double *gr=aligned_alloc(64,H*K*8),*gi=aligned_alloc(64,H*K*8);
                    memcpy(gr,orr,H*K*8); memcpy(gi,oi,H*K*8);
                    rd->ls_fwd = radix256_r2c_term_ls_r8_fwd_avx2;
                    lsact = 1;
                    for(int wu=0;wu<3;wu++) vfft_execute(p,VFFT_FORWARD,x,NULL,orr,oi);
                    size_t bad=0; for(size_t i=0;i<H*K;i++){
                        double da=fabs(orr[i]-gr[i]),db=fabs(oi[i]-gi[i]);
                        double ma=fabs(gr[i])>1?fabs(gr[i]):1, mb=fabs(gi[i])>1?fabs(gi[i]):1;
                        if(da>1e-12*ma||db>1e-12*mb) bad++; }
                    printf("  MODEL-B ACTIVATED, output match BAD=%zu %s\n",bad,bad?"**FAIL**":"OK");
                    for(int t=0;t<11;t++){ double t0=bnow();
                        for(int it=0;it<L;it++) vfft_execute(p,VFFT_FORWARD,x,NULL,orr,oi);
                        tv[t]=(bnow()-t0)/L; }
                    qsort(tv,11,8,dcmp); tot_lsb=tv[5];
                    free(gr);free(gi);
                }
            }
        }
        double pp=0,pi=0,po=0;
#ifdef VFFT_R2C_PROFILE
        double iters=(double)(11*L);
        pp=_r2c_prof_pack/iters; pi=_r2c_prof_inner/iters; po=_r2c_prof_post/iters;
#endif
        /* MKL r2c CCE, row-batched (their home layout — noted) */
        DFTI_DESCRIPTOR_HANDLE mh;
        DftiCreateDescriptor(&mh,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
        DftiSetValue(mh,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(mh,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
        DftiSetValue(mh,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        DftiSetValue(mh,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
        DftiSetValue(mh,DFTI_OUTPUT_DISTANCE,(MKL_LONG)H);
        DftiCommitDescriptor(mh);
        MKL_Complex16 *mo=aligned_alloc(64,H*K*sizeof(MKL_Complex16));
        for(int wu=0;wu<3;wu++) DftiComputeForward(mh,x,mo);
        for(int t=0;t<11;t++){ double t0=bnow();
            for(int it=0;it<L;it++) DftiComputeForward(mh,x,mo);
            tv[t]=(bnow()-t0)/L; }
        qsort(tv,11,8,dcmp); double mkl=tv[5];
        /* split->z interleave of X: the pass a z-store recombine deletes */
        for(int t=0;t<11;t++){ double t0=bnow();
            for(int it=0;it<L;it++)
                for(size_t i=0;i<H*K;i++){ z[2*i]=orr[i]; z[2*i+1]=oi[i]; }
            tv[t]=(bnow()-t0)/L; }
        qsort(tv,11,8,dcmp); double sw=tv[5];
        printf("[%4d,%4zu] %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f%s\n",
               N,K,tot,pp,pi,po,mkl,sw, lsact?"":"");
        if (lsact) printf("  [ 512, 256] MODEL-B total: %.2f us (was %.2f, MKL %.2f -> %.3fx)\n",
                          tot_lsb, tot, mkl, mkl/tot_lsb);
        vfft_destroy(p);
        DftiFreeDescriptor(&mh);
        free(x);free(orr);free(oi);free(z);free(mo);
    }
    if(w) vfft_wisdom_free(w);
    return 0;
}
