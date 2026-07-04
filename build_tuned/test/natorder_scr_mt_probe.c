/* natorder_scr_mt_probe.c — isolate the SCR-MT 256/64 failure: is the two-phase RANGE decomposition
 * wrong (logic), or a thread RACE? Replicates _scr_fwd_mt's ranges SERIALLY (no pool) and compares
 * each phase to the ST full-K path. Serial-ranged == ST => logic ok (bug is a race); != => logic bug.
 * Build: python build.py --src test/natorder_scr_mt_probe.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "natorder_perm.h"
#include "natorder_scatter.h"

static vfft_proto_registry_t REG;

static double maxd(const double *a, const double *b, size_t n){ double m=0; for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]); if(d>m)m=d;} return m; }

static void probe(int N, size_t K, const int *f, const int *v, int nf, int T)
{
    size_t n = (size_t)N*K;
    stride_plan_t *p = vfft_proto_plan_create_ex(N,K,f,v,nf,0,&REG);
    double *cre=calloc(n,8),*cim=calloc(n,8); cre[K]=1.0;
    vfft_proto_execute_fwd(p,cre,cim,K);
    int *M=vfft_natorder_detect(N,f,nf,K,cre,cim,1); free(cre); free(cim);
    int *IM=malloc(N*4); vfft_natorder_inv_perm(N,M,IM);
    natorder_scr_t s; natorder_scr_build(&s,p,N,K,M,IM);

    double *ur=malloc(n*8),*ui=malloc(n*8), *x=malloc(n*8),*xi=malloc(n*8);
    srand(9); for(size_t i=0;i<n;i++){x[i]=(double)rand()/RAND_MAX-.5; xi[i]=(double)rand()/RAND_MAX-.5;}

    /* ── ST full path ── */
    memcpy(ur,x,n*8); memcpy(ui,xi,n*8);
    natorder_scr_fwd(&s,ur,ui,K);
    double *out_st=malloc(n*8),*out_st_i=malloc(n*8); memcpy(out_st,ur,n*8); memcpy(out_st_i,ui,n*8);
    /* keep the ST scratch (post-MODEB, pre-terminator) for phase-1 compare: recompute scratch only */
    double *scr_st_r=malloc(n*8),*scr_st_i=malloc(n*8);
    memcpy(ur,x,n*8); memcpy(ui,xi,n*8);
    vfft_proto_execute_fwd_oop(&s.sub, ur, ui, s.scr_re, s.scr_im, K);
    memcpy(scr_st_r,s.scr_re,n*8); memcpy(scr_st_i,s.scr_im,n*8);

    /* ── serial-ranged MODEB (phase 1), Sv=roundup(K/T,8) lane slabs ── */
    size_t Sv=((K/(size_t)T)+7)&~(size_t)7;
    memcpy(ur,x,n*8); memcpy(ui,xi,n*8);
    memset(s.scr_re,0,n*8); memset(s.scr_im,0,n*8);
    for(size_t k0=0;k0<K;k0+=Sv){ size_t S=(k0+Sv<=K)?Sv:(K-k0);
        vfft_proto_execute_fwd_oop(&s.sub, ur+k0, ui+k0, s.scr_re+k0, s.scr_im+k0, S); }
    double e1=maxd(scr_st_r,s.scr_re,n)>maxd(scr_st_i,s.scr_im,n)?maxd(scr_st_r,s.scr_re,n):maxd(scr_st_i,s.scr_im,n);

    /* ── serial-ranged terminator (phase 2) on the (now serial-ranged) scratch, q-split ── */
    double *out_mt=calloc(n,8),*out_mt_i=calloc(n,8);
    int P=s.P, per=(P+T-1)/T;
    for(int q0=0;q0<P;q0+=per){ int q1=(q0+per<P)?q0+per:P;
        natorder_scr_term_range(&s, out_mt, out_mt_i, q0, q1); }
    double e2=maxd(out_st,out_mt,n)>maxd(out_st_i,out_mt_i,n)?maxd(out_st,out_mt,n):maxd(out_st_i,out_mt_i,n);

    printf("N=%-5d K=%-3zu R=%-2d P=%-3d Sv=%-3zu per=%-3d T=%d | phase1(MODEB range)=%.1e  phase2(term range)=%.1e  %s\n",
        N,K,s.R,s.P,Sv,per,T,e1,e2,(e1<1e-11&&e2<1e-11)?"RANGE-OK (bug=race)":"<RANGE LOGIC BUG>");

    natorder_scr_free(&s); free(M); free(IM); free(ur); free(ui); free(x); free(xi);
    free(out_st); free(out_st_i); free(scr_st_r); free(scr_st_i); free(out_mt); free(out_mt_i);
    vfft_proto_plan_destroy(p);
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    vfft_proto_registry_init(&REG);
    printf("# SCR-MT range decomposition (serial): phase1 MODEB-range vs ST, phase2 term-range vs ST\n");
    int v2[]={0,2};
    { int f[]={16,16}; probe(256, 64, f, v2, 2, 4); }   /* the FAILING cell */
    { int f[]={64,64}; probe(4096,64, f, v2, 2, 4); }   /* the PASSING cell */
    { int f[]={16,16}; probe(256, 64, f, v2, 2, 2); }   /* fewer threads */
    { int f[]={16,16}; probe(256, 64, f, v2, 2, 8); }
    { int f[]={8,16};  probe(128, 64, f, v2, 2, 4); }
    return 0;
}
