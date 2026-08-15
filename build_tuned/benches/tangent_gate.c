/* tangent_gate.c — tolerance gate for every SHIPPED tangent codelet
 * (src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/). Conventions:
 *   MID fwd : Y[o] = sum_l W_R^{ol} . w512^{lk} . x_l      (pre-twiddle)
 *   LEAF    : Y[o] = sum_l W_R^{ol} x_l, stored corner-turned (no table)
 * Build (from the repo root):
 *   gcc -O2 -static -o tangent_gate build_tuned/benches/tangent_gate.c \
 *       src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/*.c -lm
 */
#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#define K(n) void n(const double*,const double*,double*,double*,const double*, \
                    const double*,size_t,size_t,size_t,size_t,size_t)
K(radix8_z_t2tan_fwd_avx2);
K(radix8_z_n1ttan_fwd_avx2);
K(radix16_z_t2tan_fwd_avx2);
K(radix16_z_n1ttan_fwd_avx2);
K(radix32_z_t2bw32_fwd_avx2);
K(radix32_z_n1tbw32_fwd_avx2);
typedef void (*krn)(const double*,const double*,double*,double*,const double*,
                    const double*,size_t,size_t,size_t,size_t,size_t);

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

/* VTW2 records, our fold [-s,+s]; sg = -1 fwd */
static void genT(double *T, int R, int kc, double sg){
    for(int l=1;l<R;l++){
        double *r = T + (size_t)(l-1)*8;
        for(int c=0;c<2;c++){
            int kk = kc + c;
            cx d = cexp(sg*2.0*PI*I*(double)l*kk/512.0);
            r[c*2] = creal(d); r[c*2+1] = creal(d);
            r[4+c*2] = -cimag(d); r[4+c*2+1] = cimag(d);
        }
    }
}

static double gate_mid(krn fn, int R, int cols, double sg){
    int rec = (R-1)*8, N2 = R*cols*2;
    double *zin = _aligned_malloc((size_t)N2*8, 64);
    double *zo  = _aligned_malloc((size_t)N2*8, 64);
    double *T   = _aligned_malloc((size_t)(cols/2)*rec*8, 64);
    for(int i=0;i<N2;i++) zin[i] = rnd();
    for(int g=0; g<cols/2; g++) genT(T + (size_t)g*rec, R, 2*g, sg);
    fn(zin,0,zo,0,T,0,cols,0,cols,0,cols);
    double w = 0;
    for(int k=0;k<cols;k++) for(int o=0;o<R;o++){
        cx ref = 0;
        for(int l=0;l<R;l++){
            cx xv = zin[2*((size_t)l*cols+k)] + I*zin[2*((size_t)l*cols+k)+1];
            ref += cexp(sg*2.0*PI*I*(double)o*l/(double)R)
                 * cexp(sg*2.0*PI*I*(double)l*k/512.0) * xv;
        }
        cx got = zo[2*((size_t)o*cols+k)] + I*zo[2*((size_t)o*cols+k)+1];
        double e = cabs(got-ref); if(e>w) w = e;
    }
    _aligned_free(zin); _aligned_free(zo); _aligned_free(T);
    return w;
}

static double gate_leaf(krn fn, int R, int cols){
    int N2 = R*cols*2;
    double *zin = _aligned_malloc((size_t)N2*8, 64);
    double *zo  = _aligned_malloc((size_t)N2*8, 64);
    for(int i=0;i<N2;i++) zin[i] = rnd();
    fn(zin,0,zo,0,(double*)0,0,cols,0,R,0,cols);
    double w = 0;
    for(int k=0;k<cols;k++) for(int o=0;o<R;o++){
        cx ref = 0;
        for(int l=0;l<R;l++){
            cx xv = zin[2*((size_t)l*cols+k)] + I*zin[2*((size_t)l*cols+k)+1];
            ref += cexp(-2.0*PI*I*(double)o*l/(double)R) * xv;
        }
        cx got = zo[2*((size_t)k*R+o)] + I*zo[2*((size_t)k*R+o)+1];
        double e = cabs(got-ref); if(e>w) w = e;
    }
    _aligned_free(zin); _aligned_free(zo);
    return w;
}

int main(void){
    struct { const char *nm; double e; } r[5];
    r[0].nm = "radix8_z_t2tan_fwd       (R8  mid,  bit-identical)   ";
    r[0].e  = gate_mid(radix8_z_t2tan_fwd_avx2, 8, 32, -1.0);
    r[1].nm = "radix8_z_n1ttan_fwd      (R8  leaf, bit-identical)   ";
    r[1].e  = gate_leaf(radix8_z_n1ttan_fwd_avx2, 8, 16);
    r[2].nm = "radix16_z_t2tan_fwd      (R16 mid,  wing full-kernel)";
    r[2].e  = gate_mid(radix16_z_t2tan_fwd_avx2, 16, 32, -1.0);
    r[3].nm = "radix16_z_n1ttan_fwd     (R16 leaf, corner-turn)     ";
    r[3].e  = gate_leaf(radix16_z_n1ttan_fwd_avx2, 16, 16);
    r[4].nm = "radix32_z_t2bw32_fwd     (R32 mid,  wing32 2.16)     ";
    r[4].e  = gate_mid(radix32_z_t2bw32_fwd_avx2, 32, 16, -1.0);

    int ok = 1;
    for(int i=0;i<5;i++){
        int pass = r[i].e < 1e-10;
        printf("%s  %.3e  %s\n", r[i].nm, r[i].e, pass ? "CORRECT" : "*** WRONG ***");
        ok &= pass;
    }
    printf("\n%s\n", ok ? "ALL 5 SHIPPED TANGENT CODELETS GATED CORRECT"
                        : "GATE FAILURE - DO NOT SHIP");
    return ok ? 0 : 1;
}
