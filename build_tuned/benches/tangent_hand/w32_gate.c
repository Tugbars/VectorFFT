/* w32_gate.c — correctness gate for the radix-32 tangent kernels.
 * mid (w32tg):  16 cols, Ls=OLs=16, 31-record their-fold table, d=w512^{lk};
 *               golden Y[o] = sum_l W32^{ol} d_l x_l, out[2*(o*16+k)].
 * leaf (w32tgL): 16 cols, Ls=16, OLs=32, no table;
 *               golden DFT-32, corner-turn out[2*(k*32+o)]. */
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

void radix32_z_w32tg_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_w32tgL_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;
static cx W32(double p){return cexp(-2.0*PI*I*p/32.0);}
static uint64_t lcg=0x9E3779B97F4A7C15ull;
static double rnd(void){lcg=lcg*6364136223846793005ull+1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0;}

static void gen32_theirs(double *T,int kc){
    for(int l=1;l<32;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=cimag(d);r[4+col*2+1]=-cimag(d);} }
}

int main(void){
    double *zin =_aligned_malloc(1024*sizeof(double),64);
    double *zout=_aligned_malloc(1024*sizeof(double),64);
    double *T   =_aligned_malloc(8*248*sizeof(double),64);
    for(int g=0;g<8;g++) gen32_theirs(T+g*248,2*g);

    double worst_m=0, worst_l=0;
    for(int trial=0;trial<20;trial++){
        for(int i=0;i<1024;i++) zin[i]=rnd();
        /* ---- mid ---- */
        radix32_z_w32tg_fwd_avx2(zin,0,zout,0,T,0,16,0,16,0,16);
        for(int k=0;k<16;k++){
            cx x[32];
            for(int l=0;l<32;l++) x[l]=zin[2*(l*16+k)]+I*zin[2*(l*16+k)+1];
            for(int o=0;o<32;o++){
                cx ref=0;
                for(int l=0;l<32;l++)
                    ref+=W32((double)o*l)*cexp(-2.0*PI*I*(double)l*k/512.0)*x[l];
                cx got=zout[2*(o*16+k)]+I*zout[2*(o*16+k)+1];
                double e=cabs(got-ref); if(e>worst_m)worst_m=e;
            }
        }
        /* ---- leaf ---- */
        radix32_z_w32tgL_fwd_avx2(zin,0,zout,0,(double*)0,0,16,0,32,0,16);
        for(int k=0;k<16;k++){
            cx x[32];
            for(int l=0;l<32;l++) x[l]=zin[2*(l*16+k)]+I*zin[2*(l*16+k)+1];
            for(int o=0;o<32;o++){
                cx ref=0;
                for(int l=0;l<32;l++) ref+=W32((double)o*l)*x[l];
                cx got=zout[2*(k*32+o)]+I*zout[2*(k*32+o)+1];
                double e=cabs(got-ref); if(e>worst_l)worst_l=e;
            }
        }
    }
    printf("w32tg  (mid,  pre-twiddled DFT-32): worst %.3e  %s\n",worst_m,
           worst_m<1e-10?"CORRECT":"WRONG");
    printf("w32tgL (leaf, corner-turn DFT-32):  worst %.3e  %s\n",worst_l,
           worst_l<1e-10?"CORRECT":"WRONG");
    return (worst_m<1e-10&&worst_l<1e-10)?0:1;
}
