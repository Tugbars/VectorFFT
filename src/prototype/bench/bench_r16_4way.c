/* bench_r16_4way.c — Hand, Topo, SU, SU+Anno for R=16 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>
#include "../radix16_handcoded.h"

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace(double*, double*, const double*, const double*, size_t, size_t);
__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace_su(double*, double*, const double*, const double*, size_t, size_t);
__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace_su_anno(double*, double*, const double*, const double*, size_t, size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)!=0)exit(1);return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double mre(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]),s=fabs(a[i])+fabs(b[i])+1e-30,r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *hr,*hi,*tr,*ti,*sr,*si,*ar,*ai,*twr,*twi;
static void cH(){radix16_t1_dit_fwd_avx512(hr,hi,twr,twi,K,K);}
static void cT(){radix16_t1_dit_fwd_avx512_gen_inplace(tr,ti,twr,twi,K,K);}
static void cS(){radix16_t1_dit_fwd_avx512_gen_inplace_su(sr,si,twr,twi,K,K);}
static void cA(){radix16_t1_dit_fwd_avx512_gen_inplace_su_anno(ar,ai,twr,twi,K,K);}

int main(int c,char**v){
    K=c>1?atoi(v[1]):1024;
    if (K%8!=0||K<8){fprintf(stderr,"K%%8\n");return 1;}
    double *in_re=aa(16*K),*in_im=aa(16*K);
    hr=aa(16*K);hi=aa(16*K);tr=aa(16*K);ti=aa(16*K);sr=aa(16*K);si=aa(16*K);ar=aa(16*K);ai=aa(16*K);
    twr=aa(15*K);twi=aa(15*K);
    fr(in_re,16*K,1);fr(in_im,16*K,2);fr(twr,15*K,3);fr(twi,15*K,4);
    memcpy(hr,in_re,16*K*8);memcpy(hi,in_im,16*K*8);
    memcpy(tr,in_re,16*K*8);memcpy(ti,in_im,16*K*8);
    memcpy(sr,in_re,16*K*8);memcpy(si,in_im,16*K*8);
    memcpy(ar,in_re,16*K*8);memcpy(ai,in_im,16*K*8);
    cH();cT();cS();cA();
    double et=mre(hr,tr,16*K),es=mre(hr,sr,16*K),ea=mre(hr,ar,16*K);
    if(et>1e-9||es>1e-9||ea>1e-9){printf("CORRECTNESS topo=%.2e su=%.2e anno=%.2e\n",et,es,ea);return 2;}
    double tH=bn(cH,5000,7),tT=bn(cT,5000,7),tS=bn(cS,5000,7),tA=bn(cA,5000,7);
    printf("K=%5zu Hand=%6.0f Topo=%6.0f SU=%6.0f SUAnno=%6.0f | T/H=%.3f S/H=%.3f S/T=%.3f A/T=%.3f\n",
           K,tH,tT,tS,tA,tT/tH,tS/tH,tS/tT,tA/tT);
    return 0;
}
