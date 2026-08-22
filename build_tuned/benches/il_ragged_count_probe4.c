/* il_ragged_count_probe4.c -- RESEARCH PROBE (not a gate).
 * When the heuristic pair puts R>=32 in one slot and an ODD partner count in
 * the other, the structural BLOCKED default (il2p.h:507 / :492) is refused
 * because blocked kernels carry no odd-count tail.  Which cells is that, and
 * does create really fall back to the monolithic kernel? */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static int fd_pair(int N,int *oR1,int *oR2)            /* vfft.c:4952 copy */
{ int a=0,b=0; for(int R2c=(N<64?N:64);R2c>=4;R2c--){ if(N%R2c) continue; int R1c=N/R2c;
    if(R1c<3||R1c>64) continue; if(!vfft_il2p_leaf_fn(R2c,0)||!vfft_il2p_mid_fn(R1c,0)) continue;
    if(!a||abs(R1c-R2c)<abs(a-b)){a=R1c;b=R2c;} } *oR1=a;*oR2=b; return a!=0; }
static void naive(const double *z,double *o,int N,int sign)
{ for(int k=0;k<N;k++){double sr=0,si=0; for(int n=0;n<N;n++){double t=sign*2.0*M_PI*(double)k*n/(double)N,c=cos(t),s=sin(t);
  sr+=z[2*n]*c-z[2*n+1]*s; si+=z[2*n]*s+z[2*n+1]*c;} o[2*k]=sr;o[2*k+1]=si;} }
static double relerr(const double*a,const double*b,int N)
{ double nu=0,de=0; for(int i=0;i<2*N;i++){double d=a[i]-b[i];nu+=d*d;de+=b[i]*b[i];} return de>0?sqrt(nu/de):sqrt(nu);} 
int main(void)
{
    static const int NS[]={480,864,1728,1600,2400,96,192,512,1024,4096,240,800};
    printf("%-6s %-9s %-6s %-10s %-10s %-10s %-10s %s\n",
           "N","pair","odd","leaf_f","mid_f","t2t_b","n1_b_r2","fwd_err");
    for(unsigned i=0;i<sizeof NS/sizeof NS[0];i++){
        int N=NS[i],R1=0,R2=0; if(!fd_pair(N,&R1,&R2)){printf("%-6d no-pair\n",N);continue;}
        vfft_il2p_plan_t *p=vfft_il2p_create(N,R1,R2);
        if(!p){printf("%-6d %dx%d create-NULL\n",N,R1,R2);continue;}
        const char *lf = (p->leaf_f==vfft_il2p_leaf_fn(R2,0))?"MONO":"blocked";
        const char *mf = (p->mid_f ==vfft_il2p_mid_fn (R1,0))?"MONO":"blocked";
        const char *tb = (p->t2t_b ==vfft_il2p_t2t_bwd_fn(R1))?"MONO":"blocked";
        const char *nb = (p->n1_b_r2==vfft_il2p_n1_bwd_fn(R2))?"MONO":"blocked";
        double *zi=(double*)calloc(2*(size_t)N,sizeof(double));
        double *zo=(double*)calloc(2*(size_t)N,sizeof(double));
        double *zr=(double*)calloc(2*(size_t)N,sizeof(double));
        for(int n=0;n<N;n++){zi[2*n]=sin(0.7*n)+0.3; zi[2*n+1]=cos(0.31*n)-0.2;}
        naive(zi,zr,N,-1); vfft_il2p_execute_fwd(p,zi,zo);
        char pr[16]; snprintf(pr,sizeof pr,"%dx%d",R1,R2);
        printf("%-6d %-9s %-6s %-10s %-10s %-10s %-10s %.3e\n",
               N,pr,((R1&1)||(R2&1))?"YES":"no",lf,mf,tb,nb,relerr(zo,zr,N));
        vfft_il2p_destroy(p); free(zi);free(zo);free(zr);
    }
    return 0;
}
