/* P0 gate (section 60): hc2hc DIT/DIF and hc2c DIT codelets at r=4
 * vs the ported contract formulas, K=8 lanes, all rows, all lanes.
 * tw row 0 is NaN-poisoned to prove the trivial twiddle is unused. */
#include <stdio.h>
#include <math.h>
#include <stddef.h>
typedef void(*fn7)(const double*,const double*,double*,double*,
                   const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
void radix4_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
void radix4_hc2hc_dif_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
void radix4_hc2c_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
enum {N=4, K=8};
static double xr[N*K], xi[N*K], twr[N], twi[N], yr[N*K], yi[N*K];

static void fill(void){
  for(int i=0;i<N*K;i++){ xr[i]=sin(0.7*i)+0.2; xi[i]=cos(1.3*i)-0.1; }
  /* unified slot convention: leg s loads slot s-1; last slot dead */
  twr[N-1]=NAN; twi[N-1]=NAN;
  for(int s=1;s<N;s++){ twr[s-1]=cos(0.31*s); twi[s-1]=sin(0.31*s); }
}
/* reference helpers on one lane */
static void byw(double*ar,double*ai,int lane){
  for(int s=1;s<N;s++){
    double a=xr[s*K+lane], b=xi[s*K+lane], wr=twr[s-1], wi=twi[s-1];
    ar[s]=a*wr-b*wi; ai[s]=a*wi+b*wr;
  }
  ar[0]=xr[0*K+lane]; ai[0]=xi[0*K+lane];
}
static void dft4(const double*ar,const double*ai,double*Rr,double*Ri){
  for(int j=0;j<N;j++){ double sr=0,si=0;
    for(int k=0;k<N;k++){ double th=-2.0*M_PI*j*k/N;
      sr+=ar[k]*cos(th)-ai[k]*sin(th);
      si+=ar[k]*sin(th)+ai[k]*cos(th); }
    Rr[j]=sr; Ri[j]=si; }
}
static double check(const char*nm, fn7 f, void(*ref)(int,double*,double*)){
  fill(); f(xr,xi,yr,yi,twr,twi,(ptrdiff_t)K,(ptrdiff_t)K,K);
  double m=0;
  for(int lane=0;lane<K;lane++){
    double rr[N], ri[N]; ref(lane,rr,ri);
    for(int i=0;i<N;i++){
      double dr=fabs(rr[i]-yr[i*K+lane]), di=fabs(ri[i]-yi[i*K+lane]);
      if(dr>m)m=dr; if(di>m)m=di; }
  }
  printf("%-22s max|err| = %.2e  %s\n", nm, m, (m<1e-13 && m==m)?"PASS":"FAIL");
  return m;
}
/* DIT hc2hc: sym1(sym2(DFT(byw(x)))) */
static void ref_dit(int lane,double*Or,double*Oi){
  double ar[N],ai[N],Rr[N],Ri[N],r2[N],i2[N];
  byw(ar,ai,lane); dft4(ar,ai,Rr,Ri);
  for(int i=0;i<N;i++){ if(2*i<N){r2[i]=Rr[i];i2[i]=Ri[i];} else {r2[i]=-Ri[i];i2[i]=Rr[i];} }
  for(int i=0;i<N;i++){ Or[i]=r2[i]; Oi[i]=i2[N-1-i]; }
}
/* DIF hc2hc: byw_post(DFT(sym2i(sym1(x)))) */
static void ref_dif(int lane,double*Or,double*Oi){
  double s1r[N],s1i[N],s2r[N],s2i[N],Rr[N],Ri[N];
  for(int i=0;i<N;i++){ s1r[i]=xr[i*K+lane]; s1i[i]=xi[(N-1-i)*K+lane]; }
  for(int i=0;i<N;i++){ if(2*i<N){s2r[i]=s1r[i];s2i[i]=s1i[i];} else {s2r[i]=s1i[i];s2i[i]=-s1r[i];} }
  dft4(s2r,s2i,Rr,Ri);
  Or[0]=Rr[0]; Oi[0]=Ri[0];
  for(int s=1;s<N;s++){ double wr=twr[s-1], wi=twi[s-1];
    Or[s]=Rr[s]*wr-Ri[s]*wi; Oi[s]=Rr[s]*wi+Ri[s]*wr; }
}
/* DIT hc2c: sym(DFT(byw(x))) */
static void ref_hc2c(int lane,double*Or,double*Oi){
  double ar[N],ai[N],Rr[N],Ri[N];
  byw(ar,ai,lane); dft4(ar,ai,Rr,Ri);
  for(int i=0;i<N;i++){ Or[i]=Rr[i]; Oi[i]=(2*i<N)?Ri[i]:-Ri[i]; }
}
int main(void){
  double a=check("hc2hc_dit r=4", radix4_hc2hc_dit_fwd_avx512, ref_dit);
  double b=check("hc2hc_dif r=4", radix4_hc2hc_dif_fwd_avx512, ref_dif);
  double c=check("hc2c_dit  r=4", radix4_hc2c_dit_fwd_avx512,  ref_hc2c);
  return (a<1e-13 && b<1e-13 && c<1e-13 && a==a && b==b && c==c)?0:1;
}
