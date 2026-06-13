/* P2 composition harness: r2cf leaves + one hc2hc stage (+ k=0 r2cf
 * column + k=m/2 direct column) == brute packed rdft.
 * Geometry per section 62 derivation. Ping-pong: x -> B (leaf) -> A. */
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
typedef void(*r2cf_fn)(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
typedef void(*hc_fn)(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
void radix2_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix4_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix8_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
void radix2_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
void radix4_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
void radix8_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
static r2cf_fn r2cf_of(int r){return r==2?radix2_r2cf_avx512:r==4?radix4_r2cf_avx512:radix8_r2cf_avx512;}
static hc_fn hc_of(int r){return r==2?radix2_hc2hc_dit_fwd_avx512:r==4?radix4_hc2hc_dit_fwd_avx512:radix8_hc2hc_dit_fwd_avx512;}
enum {K=8, NMAX=64};
static double x[NMAX*K], A[NMAX*K], B[NMAX*K], twr[16*K], twi[16*K];

static void brute(int N,int lane,int p,double*re,double*im){
  double sr=0,si=0;
  for(int t=0;t<N;t++){ double th=-2.0*M_PI*p*t/N;
    sr+=x[t*K+lane]*cos(th); si+=x[t*K+lane]*sin(th); }
  *re=sr; *im=si;
}
static int compose(int N,int r1,int rho){
  int m=N/r1, S=N/rho;
  for(int i=0;i<N*K;i++) x[i]=sin(0.7*i)+0.2*cos(1.9*i+0.3);
  memset(A,0,sizeof A); memset(B,0,sizeof B);

  /* LEAF: rho-point r2cf, S groups batched in one call (vl=S*K) */
  r2cf_of(rho)(x, B, B+(size_t)N*K, (ptrdiff_t)(S*K), (ptrdiff_t)(S*K),
               -(ptrdiff_t)(S*K), (size_t)(S*K));

  /* leaf instrument: B vs per-child brute packed */
  double lm=0;
  for(int g=0;g<S;g++) for(int lane=0;lane<K;lane++)
    for(int t=0;t<rho;t++){
      double zr=0,zi=0;
      for(int u=0;u<rho;u++){ double th=-2.0*M_PI*t*u/rho;
        zr+=x[(g+S*u)*K+lane]*cos(th); zi+=x[(g+S*u)*K+lane]*sin(th); }
      double exp = (2*t<=rho)? zr : ({double zr2=0,zi2=0;(void)zr2;
        int tt=rho-t; double a=0,b2=0;
        for(int u=0;u<rho;u++){ double th=-2.0*M_PI*tt*u/rho;
          a+=x[(g+S*u)*K+lane]*cos(th); b2+=x[(g+S*u)*K+lane]*sin(th); }
        (void)a; b2;});
      double d=fabs(exp - B[(g+S*t)*K+lane]); if(d>lm)lm=d;
    }
  printf("  leaf  N=%-3d (r1=%d,rho=%d): max|err| = %.2e %s\n",N,r1,rho,lm,lm<1e-13?"PASS":"FAIL");

  /* STAGE: radix r1 over child size m, Q=1 */
  /* k=0 column: r2cf_r1 */
  r2cf_of(r1)(B, A, A+(size_t)N*K, (ptrdiff_t)K, (ptrdiff_t)(m*K),
              -(ptrdiff_t)(m*K), (size_t)K);
  /* interior columns */
  int kmax = (m%2==0)? m/2-1 : (m-1)/2;
  for(int k=1;k<=kmax;k++){
    for(int j=0;j<r1;j++) for(int lane=0;lane<K;lane++){
      double th=2.0*M_PI*j*k/N;
      twr[j*K+lane]=cos(th); twi[j*K+lane]=-sin(th);
    }
    hc_of(r1)(B+(size_t)(r1*k)*K, B+(size_t)(r1*(m-k))*K,
              A+(size_t)k*K, A+(size_t)(m-k)*K,
              twr, twi, (ptrdiff_t)K, (ptrdiff_t)(m*K), (size_t)K);
  }
  /* k=m/2 column, m even: direct */
  if(m%2==0){
    for(int lane=0;lane<K;lane++){
      double Xr[16],Xi[16];
      for(int s=0;s<r1;s++){ double sr=0,si=0;
        for(int j=0;j<r1;j++){
          double c=B[(size_t)(j+r1*(m/2))*K+lane];
          double th=-2.0*M_PI*j*(m/2 + (double)s*m)/N;
          sr+=c*cos(th); si+=c*sin(th);
        } Xr[s]=sr; Xi[s]=si; }
      for(int s=0;s<r1;s++){ int p=m/2+s*m;
        if(p<=N/2) A[(size_t)p*K+lane]=Xr[s];
        else       A[(size_t)p*K+lane]=Xi[r1-1-s];
      }
    }
  }
  /* compare vs brute packed */
  double cm=0;
  for(int lane=0;lane<K;lane++) for(int p=0;p<N;p++){
    double re,im,exp;
    if(p<=N/2){ brute(N,lane,p,&re,&im); exp=re; }
    else      { brute(N,lane,N-p,&re,&im); exp=im; }
    double d=fabs(exp-A[(size_t)p*K+lane]); if(d>cm)cm=d;
  }
  printf("  TOTAL N=%-3d (r1=%d,rho=%d): max|err| = %.2e %s\n",N,r1,rho,cm,cm<1e-12?"PASS":"FAIL");
  return cm<1e-12;
}
int main(void){
  int ok=1;
  ok &= compose(16,4,4);
  ok &= compose(8,2,4);
  ok &= compose(8,4,2);
  return ok?0:1;
}
