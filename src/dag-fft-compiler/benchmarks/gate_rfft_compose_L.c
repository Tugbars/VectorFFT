/* P2 L-stage composition harness: the executor reference loop.
 * Factors f[0..L-1], f[0]=outermost combine, f[L-1]=leaf.
 * Ping-pong planes; per-depth instrument localizes failures.
 * Geometry: section 62 derivation + Q-fold (vl = Q*K). */
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
typedef void(*r2cf_fn)(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
typedef void(*hc_fn)(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(3) DECL(4) DECL(5) DECL(7) DECL(8) DECL(16)
static r2cf_fn r2cf_of(int r){switch(r){case 2:return radix2_r2cf_avx512;case 3:return radix3_r2cf_avx512;case 4:return radix4_r2cf_avx512;case 5:return radix5_r2cf_avx512;case 7:return radix7_r2cf_avx512;case 8:return radix8_r2cf_avx512;default:return radix16_r2cf_avx512;}}
static hc_fn hc_of(int r){switch(r){case 2:return radix2_hc2hc_dit_fwd_avx512;case 3:return radix3_hc2hc_dit_fwd_avx512;case 4:return radix4_hc2hc_dit_fwd_avx512;case 5:return radix5_hc2hc_dit_fwd_avx512;case 7:return radix7_hc2hc_dit_fwd_avx512;case 8:return radix8_hc2hc_dit_fwd_avx512;default:return radix16_hc2hc_dit_fwd_avx512;}}
enum {K=8, NMAX=512};
static double x[NMAX*K], P0[NMAX*K], P1[NMAX*K], twr[16*NMAX*K], twi[16*NMAX*K];

static double verify_depth(const double*pl,int N,int Qp){
  int np=N/Qp; double mx=0;
  for(int q=0;q<Qp;q++) for(int lane=0;lane<K;lane++)
    for(int p=0;p<np;p++){
      int bin = (p<=np/2)? p : np-p;
      double zr=0,zi=0;
      for(int t=0;t<np;t++){ double th=-2.0*M_PI*(double)bin*t/np;
        zr+=x[(q+(size_t)Qp*t)*K+lane]*cos(th);
        zi+=x[(q+(size_t)Qp*t)*K+lane]*sin(th); }
      double ex=(p<=np/2)? zr : zi;
      double d=fabs(ex - pl[(q+(size_t)Qp*p)*K+lane]); if(d>mx)mx=d;
    }
  return mx;
}

static int compose(const int*f,int nf,int verbose){
  int N=1; for(int i=0;i<nf;i++) N*=f[i];
  for(int i=0;i<N*K;i++) x[i]=sin(0.7*i)+0.2*cos(1.9*i+0.3);
  memset(P0,0,sizeof P0); memset(P1,0,sizeof P1);
  double *cur=P0, *nxt=P1;

  int rho=f[nf-1]; size_t S=(size_t)(N/rho);
  r2cf_of(rho)(x, cur, cur+(size_t)N*K, (ptrdiff_t)(S*K), (ptrdiff_t)(S*K),
               -(ptrdiff_t)(S*K), S*K);
  if(verbose){ double e=verify_depth(cur,N,(int)S);
    printf("    depth leaf  (Q=%-3zu): %.2e %s\n",S,e,e<1e-12?"ok":"FAIL"); }

  for(int d=nf-2; d>=0; d--){
    int r=f[d]; size_t Q=1; for(int i=0;i<d;i++) Q*=(size_t)f[i];
    int np=N/(int)Q, m=np/r;
    size_t vl=Q*K;
    /* k=0 column: r2cf batched over q */
    r2cf_of(r)(cur, nxt, nxt+(size_t)N*K, (ptrdiff_t)vl,
               (ptrdiff_t)((size_t)m*vl), -(ptrdiff_t)((size_t)m*vl), vl);
    /* interior columns: hc2hc batched over q */
    int kmax=(m%2==0)? m/2-1 : (m-1)/2;
    for(int k=1;k<=kmax;k++){
      for(int j=1;j<r;j++){ double th=2.0*M_PI*(double)j*k/np;
        double cr=cos(th), ci=-sin(th);
        for(size_t v=0;v<vl;v++){ twr[(size_t)j*vl+v]=cr; twi[(size_t)j*vl+v]=ci; } }
      hc_of(r)(cur+(Q*(size_t)(r*k))*K,
               cur+(Q*(size_t)(r*(m-k)))*K,
               nxt+(Q*(size_t)k)*K,
               nxt+(Q*(size_t)(m-k))*K,
               twr, twi, (ptrdiff_t)vl, (ptrdiff_t)((size_t)m*vl), vl);
    }
    /* k = m/2 column (m even): direct, batched over (q,lane) */
    if(m%2==0){
      for(size_t q=0;q<Q;q++) for(int lane=0;lane<K;lane++){
        double Xr[16],Xi[16];
        for(int s=0;s<r;s++){ double sr=0,si=0;
          for(int j=0;j<r;j++){
            double c=cur[(q+Q*(size_t)(j+r*(m/2)))*K+lane];
            double th=-2.0*M_PI*(double)j*((double)m/2.0+(double)s*m)/np;
            sr+=c*cos(th); si+=c*sin(th);
          } Xr[s]=sr; Xi[s]=si; }
        for(int s=0;s<r;s++){ int p=m/2+s*m;
          if(p<=np/2) nxt[(q+Q*(size_t)p)*K+lane]=Xr[s];
          else        nxt[(q+Q*(size_t)p)*K+lane]=Xi[r-1-s];
        }
      }
    }
    double*t=cur; cur=nxt; nxt=t;
    if(verbose){ double e=verify_depth(cur,N,(int)Q);
      printf("    depth d=%-2d (Q=%-3zu): %.2e %s\n",d,Q,e,e<1e-12?"ok":"FAIL"); }
  }
  double e=verify_depth(cur,N,1);
  char fs[64]; int off=0;
  for(int i=0;i<nf;i++) off+=snprintf(fs+off,sizeof fs-off,"%d%s",f[i],i<nf-1?",":"");
  printf("  N=%-4d (%s): max|err| = %.2e %s\n",N,fs,e,e<1e-12?"PASS":"FAIL");
  return e<1e-12;
}
int main(void){
  int ok=1;
  { int f[]={4,4};      ok&=compose(f,2,0); }
  { int f[]={2,4,4};    ok&=compose(f,3,1); }   /* first Q>1 */
  { int f[]={4,4,4};    ok&=compose(f,3,0); }
  { int f[]={2,4,8};    ok&=compose(f,3,0); }
  { int f[]={2,4,4,4};  ok&=compose(f,4,0); }   /* Q up to 8 */
  { int f[]={4,4,16};   ok&=compose(f,3,0); }   /* the target shape */
  { int f[]={2,4,4,8};  ok&=compose(f,4,0); }
  { int f[]={5,4};      ok&=compose(f,2,0); }   /* odd combine radix */
  { int f[]={3,8};      ok&=compose(f,2,0); }
  { int f[]={4,5};      ok&=compose(f,2,0); }   /* ODD m: no Nyquist col */
  { int f[]={2,3,2};    ok&=compose(f,3,0); }   /* m=2 stage, no interior */
  { int f[]={7,3,5};    ok&=compose(f,3,0); }   /* all-odd, odd m */
  printf(ok? "ALL PASS\n":"FAILURES\n");
  return ok?0:1;
}
