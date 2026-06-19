#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#define N 256
#define HALF 128
#define R 8
#define M 16
#define VL 8
void radix256_r2c_term_ls_r8_fwd_avx512(
  const double*ink_re,const double*ink_im,const double*inm_re,const double*inm_im,
  double*Xp_re,double*Xp_im,double*Xm_re,double*Xm_im,
  const double*tw_re,const double*tw_im,ptrdiff_t is_leg,ptrdiff_t osp,ptrdiff_t osm,size_t vl);

int main(void){
  double x[N]; srand(31337);
  for(int i=0;i<N;i++) x[i]=(double)rand()/RAND_MAX*2.0-1.0;
  /* sub_j[k] = M-point DFT of subsequence z[j+p*R], z[t]=x[2t]+i x[2t+1] */
  double subr[R][M], subi[R][M];
  for(int j=0;j<R;j++)for(int k=0;k<M;k++){double sr=0,si=0;
    for(int p=0;p<M;p++){int t=j+p*R; double zr=x[2*t],zi=x[2*t+1];
      double th=-2.0*M_PI*(double)(p*k)/M; sr+=zr*cos(th)-zi*sin(th); si+=zr*sin(th)+zi*cos(th);}
    subr[j][k]=sr; subi[j][k]=si;}
  /* brute r2c ref */
  double maxerr=0;
  for(int k=1;k<M/2;k++){            /* interior column pairs */
    int mk=M-k;
    /* codelet inputs: legs strided by is_leg. Lay col k legs contiguous (is_leg=VL),
       col m-k legs in a second buffer. */
    double ink_re[R*VL],ink_im[R*VL],inm_re[R*VL],inm_im[R*VL];
    for(int j=0;j<R;j++)for(int v=0;v<VL;v++){
      ink_re[j*VL+v]=subr[j][k];  ink_im[j*VL+v]=subi[j][k];
      inm_re[j*VL+v]=subr[j][mk]; inm_im[j*VL+v]=subi[j][mk];}
    /* packed twiddles: [0..R-1]=W_half^{j*k}, [R..2R-1]=W_half^{j*mk}, [2R..3R-1]=W_N^{k+s*M} */
    double tw_re[3*R],tw_im[3*R];
    for(int j=0;j<R;j++){double th=-2.0*M_PI*(double)(j*k)/HALF; tw_re[j]=cos(th); tw_im[j]=sin(th);}
    for(int j=0;j<R;j++){double th=-2.0*M_PI*(double)(j*mk)/HALF; tw_re[R+j]=cos(th); tw_im[R+j]=sin(th);}
    for(int s=0;s<R;s++){int f=k+s*M; double th=-2.0*M_PI*(double)f/N; tw_re[2*R+s]=cos(th); tw_im[2*R+s]=sin(th);}
    double Xp_re[R*VL],Xp_im[R*VL],Xm_re[R*VL],Xm_im[R*VL];
    radix256_r2c_term_ls_r8_fwd_avx512(ink_re,ink_im,inm_re,inm_im,
      Xp_re,Xp_im,Xm_re,Xm_im,tw_re,tw_im,VL,VL,VL,VL);
    for(int s=0;s<R;s++){
      int f=k+s*M, mir=HALF-f;
      double xfr=0,xfi=0,xmr=0,xmi=0;
      for(int n=0;n<N;n++){xfr+=x[n]*cos(-2.0*M_PI*f*n/N);xfi+=x[n]*sin(-2.0*M_PI*f*n/N);
        xmr+=x[n]*cos(-2.0*M_PI*mir*n/N);xmi+=x[n]*sin(-2.0*M_PI*mir*n/N);}
      double e=fmax(fmax(fabs(Xp_re[s*VL]-xfr),fabs(Xp_im[s*VL]-xfi)),
                   fmax(fabs(Xm_re[s*VL]-xmr),fabs(Xm_im[s*VL]-xmi)));
      if(e>maxerr)maxerr=e;
    }
  }
  printf("model(b) codelet end-to-end vs brute r2c: MAX ERR %.3e  %s\n", maxerr, maxerr<1e-10?"PASS":"FAIL");
  return maxerr<1e-10?0:1;
}
