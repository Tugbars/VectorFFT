/* il_ragged_count_probe2.c -- RESEARCH PROBE (not a gate).
 * Which ENGINE serves a ragged-count cell, and is the front door bitwise
 * equal to a directly-built il2p plan on the heuristic pair? */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"
#include "il_prime.h"
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int fd_pair(int N, int *oR1, int *oR2)          /* vfft.c:4952 copy */
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
static int ilprime_pair(int M, int *oR1, int *oR2)     /* il_prime.h:106 copy */
{
    int bR1 = 0, bR2 = 0;
    for (int R2 = (M < 64 ? M : 64); R2 >= 3; R2--) {
        if (M % R2) continue;
        int R1 = M / R2;
        if (R1 < 3 || R1 > 64) continue;
        if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;
        if (!bR1 || abs(R1 - R2) < abs(bR1 - bR2)) { bR1 = R1; bR2 = R2; }
    }
    *oR1 = bR1; *oR2 = bR2; return bR1 != 0;
}
static void naive(const double *z, double *o, int N, int sign)
{
    for (int k = 0; k < N; k++) { double sr=0,si=0;
        for (int n = 0; n < N; n++) { double a = sign*2.0*M_PI*(double)k*(double)n/(double)N;
            double c=cos(a), s=sin(a); sr += z[2*n]*c - z[2*n+1]*s; si += z[2*n]*s + z[2*n+1]*c; }
        o[2*k]=sr; o[2*k+1]=si; }
}
static double relerr(const double *a, const double *b, int N)
{ double nu=0,de=0; for (int i=0;i<2*N;i++){double d=a[i]-b[i];nu+=d*d;de+=b[i]*b[i];} return de>0?sqrt(nu/de):sqrt(nu); }

int main(void)
{
    static const int NS[] = { 9, 15, 21, 25, 27, 33, 45, 50, 75, 150, 225, 300, 675,
                              31, 61, 127, 11, 13, 17, 19, 6, 12, 20, 36, 100, 144 };
    printf("%-5s %-9s %-5s %-9s %-9s %-12s %-12s %s\n",
           "N", "fd_pair", "odd", "il3p", "ilprime", "fd_oop_err", "fd_ip_err", "fd==il2p?");
    for (unsigned i = 0; i < sizeof NS/sizeof NS[0]; i++) {
        int N = NS[i], R1=0, R2=0;
        int ok = fd_pair(N, &R1, &R2);
        char pair[24]; if (ok) snprintf(pair,sizeof pair,"%dx%d",R1,R2); else snprintf(pair,sizeof pair,"-");
        int odd = ok && ((R1&1)||(R2&1));
        int cR2,cA,cB; char ch[24];
        if (vfft_il3p_default_chain(N,&cR2,&cA,&cB)) snprintf(ch,sizeof ch,"%d/%dx%d",cR2,cA,cB); else snprintf(ch,sizeof ch,"-");
        int pR1=0,pR2=0; char pp[24];
        { vfft_ilprime_plan_t *q = vfft_ilprime_create(N);
          if (q) { int M = q->M; if (ilprime_pair(M,&pR1,&pR2)) snprintf(pp,sizeof pp,"M=%d %dx%d",M,pR1,pR2); else snprintf(pp,sizeof pp,"M=%d chain",M); vfft_ilprime_destroy(q);} 
          else snprintf(pp,sizeof pp,"-"); }

        double *zi=(double*)calloc(2*(size_t)N,sizeof(double));
        double *zo=(double*)calloc(2*(size_t)N,sizeof(double));
        double *zr=(double*)calloc(2*(size_t)N,sizeof(double));
        double *zd=(double*)calloc(2*(size_t)N,sizeof(double));
        double *zp=(double*)calloc(2*(size_t)N,sizeof(double));
        for (int n=0;n<N;n++){ zi[2*n]=sin(0.7*n)+0.3; zi[2*n+1]=cos(0.31*n)-0.2; }
        naive(zi,zr,N,-1);
        vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
        cfg.transform=VFFT_C2C; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
        cfg.placement=VFFT_OUTOFPLACE; cfg.layout=VFFT_LAYOUT_INTERLEAVED;
        cfg.order=VFFT_ORDER_DEFAULT; cfg.rigor=VFFT_MEASURE;
        double eo=-1, ep=-1; const char *same="-";
        vfft_plan h=vfft_create(&cfg);
        if (h){ vfft_execute(h,VFFT_FORWARD,zi,NULL,zo,NULL); eo=relerr(zo,zr,N); vfft_destroy(h);} 
        cfg.placement=VFFT_INPLACE;
        vfft_plan h2=vfft_create(&cfg);
        if (h2){ memcpy(zp,zi,2*(size_t)N*sizeof(double)); vfft_execute(h2,VFFT_FORWARD,zp,NULL,zp,NULL); ep=relerr(zp,zr,N); vfft_destroy(h2);} 
        if (ok){ vfft_il2p_plan_t *p=vfft_il2p_create(N,R1,R2);
                 if (p){ vfft_il2p_execute_fwd(p,zi,zd);
                         same = memcmp(zd,zo,2*(size_t)N*sizeof(double))==0 ? "BITWISE" : "differs";
                         vfft_il2p_destroy(p);} }
        printf("%-5d %-9s %-5s %-9s %-9s %-12.3e %-12.3e %s\n",
               N, pair, odd?"YES":"no", ch, pp, eo, ep, same);
        free(zi);free(zo);free(zr);free(zd);free(zp);
    }
    return 0;
}
