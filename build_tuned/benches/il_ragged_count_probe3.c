/* il_ragged_count_probe3.c -- RESEARCH PROBE (not a gate).
 * In-place INTERLEAVED at ragged-count N, with an EXPLICIT natural order,
 * plus the matched roundtrip.  Also: does VFFT_NO_IL2P change the OOP
 * answer (i.e. was il2p the engine)? */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static void naive(const double *z, double *o, int N, int sign)
{ for (int k=0;k<N;k++){double sr=0,si=0; for(int n=0;n<N;n++){double a=sign*2.0*M_PI*(double)k*n/(double)N,c=cos(a),s=sin(a);
  sr+=z[2*n]*c-z[2*n+1]*s; si+=z[2*n]*s+z[2*n+1]*c;} o[2*k]=sr;o[2*k+1]=si;} }
static double relerr(const double *a,const double *b,int N)
{ double nu=0,de=0; for(int i=0;i<2*N;i++){double d=a[i]-b[i];nu+=d*d;de+=b[i]*b[i];} return de>0?sqrt(nu/de):sqrt(nu); }

static double run(int N,int placement,int order,const double *zi,const double *ref,int *created)
{
    vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.placement=(vfft_placement_t)placement; cfg.layout=VFFT_LAYOUT_INTERLEAVED;
    cfg.order=order; cfg.rigor=VFFT_MEASURE;
    vfft_plan h=vfft_create(&cfg); *created = h!=NULL;
    if(!h) return -1.0;
    double *a=(double*)calloc(2*(size_t)N,sizeof(double));
    double *b=(double*)calloc(2*(size_t)N,sizeof(double));
    memcpy(a,zi,2*(size_t)N*sizeof(double));
    if(placement==VFFT_INPLACE){ vfft_execute(h,VFFT_FORWARD,a,NULL,a,NULL); memcpy(b,a,2*(size_t)N*sizeof(double)); }
    else vfft_execute(h,VFFT_FORWARD,a,NULL,b,NULL);
    double e=relerr(b,ref,N);
    vfft_destroy(h); free(a); free(b);
    return e;
}
static double roundtrip(int N,int placement,int order,const double *zi)
{
    vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.placement=(vfft_placement_t)placement; cfg.layout=VFFT_LAYOUT_INTERLEAVED;
    cfg.order=order; cfg.rigor=VFFT_MEASURE;
    vfft_plan h=vfft_create(&cfg); if(!h) return -1.0;
    double *a=(double*)calloc(2*(size_t)N,sizeof(double));
    double *b=(double*)calloc(2*(size_t)N,sizeof(double));
    double *c=(double*)calloc(2*(size_t)N,sizeof(double));
    memcpy(a,zi,2*(size_t)N*sizeof(double));
    if(placement==VFFT_INPLACE){ vfft_execute(h,VFFT_FORWARD,a,NULL,a,NULL); vfft_execute(h,VFFT_BACKWARD,a,NULL,a,NULL); memcpy(c,a,2*(size_t)N*sizeof(double)); }
    else { vfft_execute(h,VFFT_FORWARD,a,NULL,b,NULL); vfft_execute(h,VFFT_BACKWARD,b,NULL,c,NULL); }
    for(int i=0;i<2*N;i++) c[i]/=(double)N;
    double e=relerr(c,zi,N);
    vfft_destroy(h); free(a);free(b);free(c);
    return e;
}
int main(void)
{
    static const int NS[]={9,15,21,25,27,33,45,50,75,150,225,300,675,12,20,36,100,144,192,96,128,512};
    printf("%-5s %-12s %-12s %-12s %-12s %-12s\n","N","oop_DEF","ip_DEF","ip_NAT","rt_ip_DEF","rt_oop_DEF");
    for(unsigned i=0;i<sizeof NS/sizeof NS[0];i++){
        int N=NS[i],cr;
        double *zi=(double*)calloc(2*(size_t)N,sizeof(double));
        double *ref=(double*)calloc(2*(size_t)N,sizeof(double));
        for(int n=0;n<N;n++){zi[2*n]=sin(0.7*n)+0.3; zi[2*n+1]=cos(0.31*n)-0.2;}
        naive(zi,ref,N,-1);
        double a=run(N,VFFT_OUTOFPLACE,VFFT_ORDER_DEFAULT,zi,ref,&cr);
        double b=run(N,VFFT_INPLACE,VFFT_ORDER_DEFAULT,zi,ref,&cr);
        double c=run(N,VFFT_INPLACE,VFFT_ORDER_NATURAL,zi,ref,&cr);
        double d=roundtrip(N,VFFT_INPLACE,VFFT_ORDER_DEFAULT,zi);
        double e=roundtrip(N,VFFT_OUTOFPLACE,VFFT_ORDER_DEFAULT,zi);
        printf("%-5d %-12.3e %-12.3e %-12.3e %-12.3e %-12.3e\n",N,a,b,c,d,e);
        free(zi);free(ref);
    }
    return 0;
}
