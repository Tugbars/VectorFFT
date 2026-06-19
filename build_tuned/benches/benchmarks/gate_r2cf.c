/* r2cf leaf gates: vs real-input DFT formula; skipped-row sentinels. */
#include <stdio.h>
#include <math.h>
#include <stddef.h>
typedef void(*r2cf_fn)(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix4_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix8_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix5_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix2_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix16_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix32_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
enum {K=8};
static double x[32*K], yr[32*K], yi[32*K];
static void gate(const char*nm,int n,r2cf_fn f){
  int half=n/2, im_hi=(n%2==0)?half-1:half;
  for(int i=0;i<n*K;i++) x[i]=sin(0.7*i)+0.2*cos(1.9*i);
  for(int i=0;i<n*K;i++){ yr[i]=7777.0; yi[i]=7777.0; }
  f(x, yr, yi, (ptrdiff_t)K, (ptrdiff_t)K, (ptrdiff_t)K, K);
  double m=0;
  for(int lane=0;lane<K;lane++)
    for(int k=0;k<=half;k++){
      double sr=0,si=0;
      for(int t=0;t<n;t++){ double th=-2.0*M_PI*k*t/n;
        sr+=x[t*K+lane]*cos(th); si+=x[t*K+lane]*sin(th); }
      double dr=fabs(sr-yr[k*K+lane]); if(dr>m)m=dr;
      if(k>=1 && k<=im_hi){ double di=fabs(si-yi[k*K+lane]); if(di>m)m=di; }
    }
  int sent_ok = 1;
  for(int lane=0;lane<K;lane++){
    if(yi[0*K+lane]!=7777.0) sent_ok=0;
    if(n%2==0 && yi[half*K+lane]!=7777.0) sent_ok=0;
  }
  printf("%-14s max|err| = %.2e  %s | skipped-rows untouched: %s\n",
         nm, m, m<1e-13?"PASS":"FAIL", sent_ok?"PASS":"FAIL");
}
int main(void){
  gate("r2cf r=4", 4, radix4_r2cf_avx512);
  gate("r2cf r=8", 8, radix8_r2cf_avx512);
  gate("r2cf r=5", 5, radix5_r2cf_avx512);
  gate("r2cf r=2", 2, radix2_r2cf_avx512);
  gate("r2cf r=16",16, radix16_r2cf_avx512);
  gate("r2cf r=32",32, radix32_r2cf_avx512);
  return 0;
}
