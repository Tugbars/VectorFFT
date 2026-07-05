#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static int fails=0;
static void naive(const double*re,const double*im,int N,size_t K,double*Xr,double*Xi){for(int k=0;k<N;k++){double sr=0,si=0;for(int n=0;n<N;n++){double a=-2.0*M_PI*k*n/N,c=cos(a),s=sin(a);sr+=re[(size_t)n*K]*c-im[(size_t)n*K]*s;si+=re[(size_t)n*K]*s+im[(size_t)n*K]*c;}Xr[k]=sr;Xi[k]=si;}}
static void cell(int N,size_t K){size_t n=(size_t)N*K;double*re=malloc(n*8),*im=malloc(n*8),*x=malloc(n*8),*xi=malloc(n*8),*Xr=malloc((size_t)N*8),*Xi=malloc((size_t)N*8);
 srand(3+N+(int)K);for(size_t i=0;i<n;i++){x[i]=(double)rand()/RAND_MAX-.5;xi[i]=(double)rand()/RAND_MAX-.5;}naive(x,xi,N,K,Xr,Xi);double sc=0;for(int k=0;k<N;k++)if(fabs(Xr[k])>sc)sc=fabs(Xr[k]);
 vfft_config_t c;memset(&c,0,sizeof c);c.transform=VFFT_C2C;c.placement=VFFT_INPLACE;c.rigor=VFFT_MEASURE;c.dims=1;c.n[0]=N;c.howmany=K;c.nthreads=1;c.order=VFFT_ORDER_NATURAL;
 printf("  N=%-5d K=%-3zu racing...",N,K);fflush(stdout);vfft_plan p=vfft_create(&c);if(!p){printf(" NULL\n");fails++;goto d;}
 memcpy(re,x,n*8);memcpy(im,xi,n*8);vfft_execute(p,VFFT_FORWARD,re,im,re,im);double eF=0;for(int k=0;k<N;k++){double d1=fabs(re[(size_t)k*K]-Xr[k]),d2=fabs(im[(size_t)k*K]-Xi[k]);if(d1>eF)eF=d1;if(d2>eF)eF=d2;}eF/=(sc>0?sc:1);
 vfft_execute(p,VFFT_BACKWARD,re,im,re,im);double eR=0,inv=1.0/N;for(size_t i=0;i<n;i++){double d1=fabs(re[i]*inv-x[i]),d2=fabs(im[i]*inv-xi[i]);if(d1>eR)eR=d1;if(d2>eR)eR=d2;}
 int bad=(eF>1e-9)||(eR>1e-9);if(bad)fails++;printf(" fwd=%.1e rt=%.1e %s\n",eF,eR,bad?"<FAIL>":"ok");vfft_destroy(p);
 d:free(re);free(im);free(x);free(xi);free(Xr);free(Xi);}
int main(void){setvbuf(stdout,NULL,_IONBF,0);SetThreadAffinityMask(GetCurrentThread(),1);
 system("rmdir /s /q natorder_scrdit_wis 2>nul");system("mkdir natorder_scrdit_wis 2>nul");putenv("VFFT_WISDOM_DIR=natorder_scrdit_wis");
 printf("# SCR DIT-injection: does SCR win the L2-resident 4096/4 cell (its bakeoff home)?\n");
 cell(4096,4);cell(16384,4);cell(4096,8);
 printf("\n# winning nat_mode per cell (1=FREE 3=SCR 4=PURE 5=PSWAP; col after exec_me):\n");
 system("type natorder_scrdit_wis\\spike_wisdom.txt 2>nul | findstr /R \"^4096 ^16384\"");
 printf(fails?"\n%d FAIL\n":"\nALL CORRECT\n",fails);return fails?1:0;}
