/* hc2c_nat formula gate: y = sym(DFT_r(byw(x))), 4-pointer placement.
 * Low slots s <= s* -> Rp/Ip + s*osp (im = +Im G);
 * upper slots      -> Rm/Im + (r-1-s)*osm (im = -Im G, conjugated). */
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
#define N 4
#define K 8
void radix4_hc2c_nat_fwd_avx512(const double*,const double*,double*,double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
static double xr[N*K], xi[N*K], twr[N], twi[N];
static double Rp[N*K], Ip[N*K], Rm[N*K], Im_[N*K];
int main(void){
  int r=4; int sstar = (r%2==0)? r/2-1 : (r-1)/2;
  for(int s=0;s<N;s++)for(int k=0;k<K;k++){
    xr[s*K+k]=sin(0.7*s+0.13*k); xi[s*K+k]=cos(0.4*s+0.21*k);}
  twr[N-1]=NAN; twi[N-1]=NAN;
  for(int s=1;s<N;s++){ twr[s-1]=cos(0.31*s); twi[s-1]=sin(0.31*s); }
  memset(Rp,0,sizeof Rp); memset(Ip,0,sizeof Ip);
  memset(Rm,0,sizeof Rm); memset(Im_,0,sizeof Im_);
  radix4_hc2c_nat_fwd_avx512(xr,xi,Rp,Ip,Rm,Im_,twr,twi,(ptrdiff_t)K,(ptrdiff_t)K,(ptrdiff_t)K,K);
  double err=0;
  for(int lane=0;lane<K;lane++){
    double cr[N], ci[N];
    cr[0]=xr[0*K+lane]; ci[0]=xi[0*K+lane];
    for(int s=1;s<N;s++){
      double a=xr[s*K+lane], b=xi[s*K+lane], wr=twr[s-1], wi=twi[s-1];
      cr[s]=a*wr-b*wi; ci[s]=a*wi+b*wr;
    }
    for(int s=0;s<N;s++){
      double gr=0, gi=0;
      for(int j=0;j<N;j++){
        double th=-2.0*M_PI*j*s/N;
        gr+=cr[j]*cos(th)-ci[j]*sin(th);
        gi+=cr[j]*sin(th)+ci[j]*cos(th);
      }
      double mr, mi;
      if(s<=sstar){ mr=Rp[s*K+lane]; mi=Ip[s*K+lane]; }
      else        { mr=Rm[(r-1-s)*K+lane]; mi=-Im_[(r-1-s)*K+lane]; }
      double e1=fabs(mr-gr), e2=fabs(mi-gi);
      if(e1>err)err=e1; if(e2>err)err=e2;
    }
  }
  printf("hc2c_nat r=%-3d        max|err| = %.2e  %s\n",r,err,err<1e-11?"PASS":"FAIL");
  return err<1e-11?0:1;
}
