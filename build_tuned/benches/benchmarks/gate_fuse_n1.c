#define _POSIX_C_SOURCE 200809L
#define MPI 3.14159265358979323846
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef RN
#define RN 32
#endif
#ifndef FN
#define FN radix32_n1_dit_fwd_avx512
#endif
extern void FN(double*,double*,const double*,const double*,size_t,size_t);
int main(void){
  int N=RN; size_t me=8, ios=me, buf=(size_t)N*ios;
  double *re,*im,*re0,*im0;
  posix_memalign((void**)&re,64,buf*sizeof(double));
  posix_memalign((void**)&im,64,buf*sizeof(double));
  re0=malloc(buf*sizeof(double)); im0=malloc(buf*sizeof(double));
  srand(7); for(size_t i=0;i<buf;i++){re0[i]=(rand()%2000)/100.0-10.0; im0[i]=(rand()%2000)/100.0-10.0;}
  memcpy(re,re0,buf*sizeof(double));memcpy(im,im0,buf*sizeof(double));
  double tw[256]={0}; FN(re,im,tw,tw,ios,me);
  double maxabs=0;
  for(size_t col=0;col<me;col++){
    for(int n=0;n<N;n++){double sr=0,si=0;
      for(int k=0;k<N;k++){double a=-2.0*MPI*n*k/N,c=cos(a),s=sin(a);
        double xr=re0[(size_t)k*ios+col],xi=im0[(size_t)k*ios+col];
        sr+=xr*c-xi*s; si+=xr*s+xi*c;}
      double gr=re[(size_t)n*ios+col],gi=im[(size_t)n*ios+col];
      double e=fabs(gr-sr)+fabs(gi-si); if(e>maxabs)maxabs=e;}
  }
  printf("max_abs=%.2e\n",maxabs); return (maxabs<1e-9)?0:1;
}
