#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#define N 256
#define HALF 128
#define VL 8
/* the codelet under test */
void radix256_r2c_term_k1_fwd_avx512(
  const double*in_re,const double*in_im,double*Xp_re,double*Xp_im,
  double*Xm_re,double*Xm_im,ptrdiff_t is,size_t vl);

int main(void){
  /* Build random real input x[0..N-1], compute Z = (N/2)-pt complex FFT of
     pair-packed reals, lay Z[k] and Z[m] into the codelet's input rows, run it,
     and compare X[k],X[m] to brute r2c. VL lanes all identical (broadcast). */
  double x[N];
  srand(12345);
  for(int i=0;i<N;i++) x[i]=(double)rand()/RAND_MAX*2.0-1.0;
  /* Z */
  double zr[HALF], zi[HALF];
  for(int kk=0;kk<HALF;kk++){double sr=0,si=0;
    for(int j=0;j<HALF;j++){double zjr=x[2*j],zji=x[2*j+1];
      double th=-2.0*M_PI*(double)(kk*j)/HALF;
      sr+=zjr*cos(th)-zji*sin(th); si+=zjr*sin(th)+zji*cos(th);}
    zr[kk]=sr; zi[kk]=si;}
  int k=1, m=HALF-k;
  /* codelet input: row0 = Z[k], row1 = Z[m] (is = VL row stride) */
  double in_re[2*VL], in_im[2*VL], Xp_re[VL],Xp_im[VL],Xm_re[VL],Xm_im[VL];
  for(int v=0;v<VL;v++){ in_re[v]=zr[k]; in_im[v]=zi[k]; in_re[VL+v]=zr[m]; in_im[VL+v]=zi[m]; }
  radix256_r2c_term_k1_fwd_avx512(in_re,in_im,Xp_re,Xp_im,Xm_re,Xm_im,VL,VL);
  /* brute r2c reference */
  double xkr=0,xki=0,xmr=0,xmi=0;
  for(int n=0;n<N;n++){
    xkr+=x[n]*cos(-2.0*M_PI*k*n/N); xki+=x[n]*sin(-2.0*M_PI*k*n/N);
    xmr+=x[n]*cos(-2.0*M_PI*m*n/N); xmi+=x[n]*sin(-2.0*M_PI*m*n/N);}
  double e1=fabs(Xp_re[0]-xkr),e2=fabs(Xp_im[0]-xki),e3=fabs(Xm_re[0]-xmr),e4=fabs(Xm_im[0]-xmi);
  double me=fmax(fmax(e1,e2),fmax(e3,e4));
  printf("X[k]: got(%.6f,%.6f) ref(%.6f,%.6f)\n",Xp_re[0],Xp_im[0],xkr,xki);
  printf("X[m]: got(%.6f,%.6f) ref(%.6f,%.6f)\n",Xm_re[0],Xm_im[0],xmr,xmi);
  printf("MAX ERR = %.3e  %s\n", me, me<1e-10?"PASS":"FAIL");
  return me<1e-10?0:1;
}
