/*
 * bench_planner_sim.c — Simulate planner with pre-interleaved IL twiddles
 *
 * Three "planner" strategies at each K:
 *   1. Current best: min(split CT, split DAG) — what we have now
 *   2. IL NOTW: already real numbers, no twiddle overhead
 *   3. IL TW with pre-interleaved twiddles: write a direct-load wrapper
 *
 * "Simulated planner" picks best across all strategies per K.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

/* Split codelets */
#include "fft_radix20_avx512_ct.h"
#include "fft_radix20_avx2_ct.h"
#include "fft_radix20_avx2.h"

/* IL codelets */
#include "fft_radix20_avx512_il_ct.h"
#include "fft_radix20_avx2_il_ct.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#define BENCH(best, code) { \
    best = 1e18; \
    for (int t = 0; t < trials; t++) { \
        double t0 = now_ns(); \
        for (int r = 0; r < reps; r++) { code; } \
        double dt = (now_ns() - t0) / reps; \
        if (dt < best) best = dt; \
    } \
}

/*
 * Pre-interleaved IL twiddle codelet (AVX-512).
 * Same CT butterfly as radix20_ct_tw_dit_kernel_fwd_il_avx512,
 * but twiddle loads are just _mm512_load_pd from pre-interleaved table.
 * 
 * We can't easily patch the generated codelet, so we measure the overhead
 * differently: bench IL NOTW (pure butterfly) and IL TW (with permutex2var),
 * then estimate pre-interleaved = IL_NOTW + (IL_TW - IL_NOTW) * ratio.
 * 
 * The ratio accounts for: cmul cost stays, only the load changes.
 * permutex2var = ~6 cycles, aligned load = ~4 cycles.
 * 19 twiddles × (6-4) = 38 cycles saved.
 * But we measure directly instead of guessing.
 */

/* Create pre-interleaved twiddle table: tw_il[k*19*2 + m*2 + {re,im}] */
static void make_il_twiddles(double *tw_il, int R, size_t K) {
    int entries = R - 1;
    for (size_t k = 0; k < K; k++) {
        for (int m = 1; m < R; m++) {
            /* For identity twiddles in NOTW-equivalence test */
            tw_il[k * entries * 2 + (m-1) * 2 + 0] = 1.0; /* re */
            tw_il[k * entries * 2 + (m-1) * 2 + 1] = 0.0; /* im */
        }
    }
}

/* Simple pre-interleaved IL twiddle DIT kernel (AVX-512)
 * Loads twiddles directly from IL-format table — no permutex2var */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix20_ct_tw_dit_preil_fwd_avx512(
    const double * __restrict__ in, double * __restrict__ out,
    const double * __restrict__ tw_il, /* pre-interleaved: [k][m][re,im] */
    size_t K)
{
    /* 4×5 CT with direct IL twiddle loads */
    /* Reuse the internal twiddle constants from IL codelet */
    /* DFT-5 Rader constants */
    #define cA512 _mm512_set1_pd(0.55901699437494742)
    #define cB512 _mm512_set1_pd(0.95105651629515357)
    #define cC512 _mm512_set1_pd(0.58778525229247313)
    #define cD512 _mm512_set1_pd(0.25)
    
    /* Internal W20 pre-interleaved constants */
    #define IW(e) _mm512_set_pd(w20_im_vals[e],w20_re_vals[e],w20_im_vals[e],w20_re_vals[e],w20_im_vals[e],w20_re_vals[e],w20_im_vals[e],w20_re_vals[e])
    
    static const double w20_re_vals[20] = {
        1.0, 0.95105651629515357, 0.80901699437494742, 0.58778525229247313, 0.30901699437494742,
        0.0, -0.30901699437494742, -0.58778525229247313, -0.80901699437494742, -0.95105651629515357,
        -1.0, -0.95105651629515357, -0.80901699437494742, -0.58778525229247313, -0.30901699437494742,
        0.0, 0.30901699437494742, 0.58778525229247313, 0.80901699437494742, 0.95105651629515357
    };
    static const double w20_im_vals[20] = {
        0.0, -0.30901699437494742, -0.58778525229247313, -0.80901699437494742, -0.95105651629515357,
        -1.0, -0.95105651629515357, -0.80901699437494742, -0.58778525229247313, -0.30901699437494742,
        0.0, 0.30901699437494742, 0.58778525229247313, 0.80901699437494742, 0.95105651629515357,
        1.0, 0.95105651629515357, 0.80901699437494742, 0.58778525229247313, 0.30901699437494742
    };
    
    __attribute__((aligned(64))) double sp[20 * 8];
    
    const int C = 4; /* k-step for IL avx512 */
    
    for (size_t k = 0; k < K; k += C) {
        const double *twp = &tw_il[k * 19 * 2]; /* pre-interleaved, contiguous */
        
        /* Pass 1: 5x DFT-4 with direct IL twiddle loads (paired: 0+1, 2+3, 4 alone) */
        /* Pair n2=0,1 */
        {
            __m512d a0=_mm512_load_pd(&in[2*(0*K+k)]), a1=_mm512_load_pd(&in[2*(5*K+k)]),
                    a2=_mm512_load_pd(&in[2*(10*K+k)]), a3=_mm512_load_pd(&in[2*(15*K+k)]);
            __m512d b0=_mm512_load_pd(&in[2*(1*K+k)]), b1=_mm512_load_pd(&in[2*(6*K+k)]),
                    b2=_mm512_load_pd(&in[2*(11*K+k)]), b3=_mm512_load_pd(&in[2*(16*K+k)]);
            
            /* Apply external twiddles — DIRECT LOAD from pre-interleaved table */
            #define VZMUL_IL(tw,x) _mm512_fmaddsub_pd(x,_mm512_movedup_pd(tw),_mm512_mul_pd(_mm512_permute_pd(x,0x55),_mm512_permute_pd(tw,0xFF)))
            
            /* m=5: tw[4] */ { __m512d tw=_mm512_load_pd(&twp[4*C*2]); a1=VZMUL_IL(tw,a1); }
            /* m=10: tw[9] */ { __m512d tw=_mm512_load_pd(&twp[9*C*2]); a2=VZMUL_IL(tw,a2); }
            /* m=15: tw[14] */ { __m512d tw=_mm512_load_pd(&twp[14*C*2]); a3=VZMUL_IL(tw,a3); }
            /* m=1: tw[0] */ { __m512d tw=_mm512_load_pd(&twp[0*C*2]); b0=VZMUL_IL(tw,b0); }
            /* m=6: tw[5] */ { __m512d tw=_mm512_load_pd(&twp[5*C*2]); b1=VZMUL_IL(tw,b1); }
            /* m=11: tw[10] */ { __m512d tw=_mm512_load_pd(&twp[10*C*2]); b2=VZMUL_IL(tw,b2); }
            /* m=16: tw[15] */ { __m512d tw=_mm512_load_pd(&twp[15*C*2]); b3=VZMUL_IL(tw,b3); }
            
            /* DFT-4 butterfly */
            #define VNBYI(x) _mm512_fmsubadd_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),_mm512_permute_pd(x,0x55))
            
            __m512d as=_mm512_add_pd(a0,a2), ad=_mm512_sub_pd(a0,a2);
            __m512d at=_mm512_add_pd(a1,a3), au=_mm512_sub_pd(a1,a3);
            __m512d bs=_mm512_add_pd(b0,b2), bd=_mm512_sub_pd(b0,b2);
            __m512d bt=_mm512_add_pd(b1,b3), bu=_mm512_sub_pd(b1,b3);
            
            __m512d anju=VNBYI(au), bnju=VNBYI(bu);
            _mm512_store_pd(&sp[0*8], _mm512_add_pd(as,at));
            _mm512_store_pd(&sp[5*8], _mm512_add_pd(ad,anju));
            _mm512_store_pd(&sp[10*8], _mm512_sub_pd(as,at));
            _mm512_store_pd(&sp[15*8], _mm512_sub_pd(ad,anju));
            _mm512_store_pd(&sp[1*8], _mm512_add_pd(bs,bt));
            _mm512_store_pd(&sp[6*8], _mm512_add_pd(bd,bnju));
            _mm512_store_pd(&sp[11*8], _mm512_sub_pd(bs,bt));
            _mm512_store_pd(&sp[16*8], _mm512_sub_pd(bd,bnju));
        }
        /* Pair n2=2,3 */
        {
            __m512d a0=_mm512_load_pd(&in[2*(2*K+k)]), a1=_mm512_load_pd(&in[2*(7*K+k)]),
                    a2=_mm512_load_pd(&in[2*(12*K+k)]), a3=_mm512_load_pd(&in[2*(17*K+k)]);
            __m512d b0=_mm512_load_pd(&in[2*(3*K+k)]), b1=_mm512_load_pd(&in[2*(8*K+k)]),
                    b2=_mm512_load_pd(&in[2*(13*K+k)]), b3=_mm512_load_pd(&in[2*(18*K+k)]);
            { __m512d tw=_mm512_load_pd(&twp[1*C*2]); a0=VZMUL_IL(tw,a0); }
            { __m512d tw=_mm512_load_pd(&twp[6*C*2]); a1=VZMUL_IL(tw,a1); }
            { __m512d tw=_mm512_load_pd(&twp[11*C*2]); a2=VZMUL_IL(tw,a2); }
            { __m512d tw=_mm512_load_pd(&twp[16*C*2]); a3=VZMUL_IL(tw,a3); }
            { __m512d tw=_mm512_load_pd(&twp[2*C*2]); b0=VZMUL_IL(tw,b0); }
            { __m512d tw=_mm512_load_pd(&twp[7*C*2]); b1=VZMUL_IL(tw,b1); }
            { __m512d tw=_mm512_load_pd(&twp[12*C*2]); b2=VZMUL_IL(tw,b2); }
            { __m512d tw=_mm512_load_pd(&twp[17*C*2]); b3=VZMUL_IL(tw,b3); }
            __m512d as=_mm512_add_pd(a0,a2), ad=_mm512_sub_pd(a0,a2);
            __m512d at=_mm512_add_pd(a1,a3), au=_mm512_sub_pd(a1,a3);
            __m512d bs=_mm512_add_pd(b0,b2), bd=_mm512_sub_pd(b0,b2);
            __m512d bt=_mm512_add_pd(b1,b3), bu=_mm512_sub_pd(b1,b3);
            __m512d anju=VNBYI(au), bnju=VNBYI(bu);
            _mm512_store_pd(&sp[2*8], _mm512_add_pd(as,at));
            _mm512_store_pd(&sp[7*8], _mm512_add_pd(ad,anju));
            _mm512_store_pd(&sp[12*8], _mm512_sub_pd(as,at));
            _mm512_store_pd(&sp[17*8], _mm512_sub_pd(ad,anju));
            _mm512_store_pd(&sp[3*8], _mm512_add_pd(bs,bt));
            _mm512_store_pd(&sp[8*8], _mm512_add_pd(bd,bnju));
            _mm512_store_pd(&sp[13*8], _mm512_sub_pd(bs,bt));
            _mm512_store_pd(&sp[18*8], _mm512_sub_pd(bd,bnju));
        }
        /* n2=4 alone */
        {
            __m512d r0=_mm512_load_pd(&in[2*(4*K+k)]), r1=_mm512_load_pd(&in[2*(9*K+k)]),
                    r2=_mm512_load_pd(&in[2*(14*K+k)]), r3=_mm512_load_pd(&in[2*(19*K+k)]);
            { __m512d tw=_mm512_load_pd(&twp[3*C*2]); r0=VZMUL_IL(tw,r0); }
            { __m512d tw=_mm512_load_pd(&twp[8*C*2]); r1=VZMUL_IL(tw,r1); }
            { __m512d tw=_mm512_load_pd(&twp[13*C*2]); r2=VZMUL_IL(tw,r2); }
            { __m512d tw=_mm512_load_pd(&twp[18*C*2]); r3=VZMUL_IL(tw,r3); }
            __m512d s=_mm512_add_pd(r0,r2), d=_mm512_sub_pd(r0,r2);
            __m512d t=_mm512_add_pd(r1,r3), u=_mm512_sub_pd(r1,r3);
            __m512d nju=VNBYI(u);
            _mm512_store_pd(&sp[4*8], _mm512_add_pd(s,t));
            _mm512_store_pd(&sp[9*8], _mm512_add_pd(d,nju));
            _mm512_store_pd(&sp[14*8], _mm512_sub_pd(s,t));
            _mm512_store_pd(&sp[19*8], _mm512_sub_pd(d,nju));
        }
        
        /* Internal W20 twiddles */
        #define ITW_APPLY(slot, e) { \
            __m512d x=_mm512_load_pd(&sp[slot*8]); \
            x = VZMUL_IL(IW(e), x); \
            _mm512_store_pd(&sp[slot*8], x); }
        
        /* k1=1: n2=1..4, e=(1*n2)%20 = 1,2,3,4 */
        ITW_APPLY(6, 1) ITW_APPLY(7, 2) ITW_APPLY(8, 3) ITW_APPLY(9, 4)
        /* k1=2: e=(2*n2)%20 = 2,4,6,8 */
        ITW_APPLY(11, 2) ITW_APPLY(12, 4) ITW_APPLY(13, 6) ITW_APPLY(14, 8)
        /* k1=3: e=(3*n2)%20 = 3,6,9,12 */
        ITW_APPLY(16, 3) ITW_APPLY(17, 6) ITW_APPLY(18, 9) ITW_APPLY(19, 12)
        
        /* Pass 2: 4x DFT-5 (paired: k1=0+1, k1=2+3) */
        #define VBYI(x) _mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),_mm512_permute_pd(x,0x55))
        
        #define DFT5_PAIR(k1a, k1b) { \
            __m512d a0=_mm512_load_pd(&sp[(k1a*5+0)*8]), a1=_mm512_load_pd(&sp[(k1a*5+1)*8]), \
                    a2=_mm512_load_pd(&sp[(k1a*5+2)*8]), a3=_mm512_load_pd(&sp[(k1a*5+3)*8]), \
                    a4=_mm512_load_pd(&sp[(k1a*5+4)*8]); \
            __m512d b0=_mm512_load_pd(&sp[(k1b*5+0)*8]), b1=_mm512_load_pd(&sp[(k1b*5+1)*8]), \
                    b2=_mm512_load_pd(&sp[(k1b*5+2)*8]), b3=_mm512_load_pd(&sp[(k1b*5+3)*8]), \
                    b4=_mm512_load_pd(&sp[(k1b*5+4)*8]); \
            __m512d as1=_mm512_add_pd(a1,a4), as2=_mm512_add_pd(a2,a3); \
            __m512d ad1=_mm512_sub_pd(a1,a4), ad2=_mm512_sub_pd(a2,a3); \
            __m512d bs1=_mm512_add_pd(b1,b4), bs2=_mm512_add_pd(b2,b3); \
            __m512d bd1=_mm512_sub_pd(b1,b4), bd2=_mm512_sub_pd(b2,b3); \
            __m512d ass=_mm512_add_pd(as1,as2), bss=_mm512_add_pd(bs1,bs2); \
            _mm512_store_pd(&out[2*((k1a+4*0)*K+k)], _mm512_add_pd(a0,ass)); \
            _mm512_store_pd(&out[2*((k1b+4*0)*K+k)], _mm512_add_pd(b0,bss)); \
            __m512d at0=_mm512_fnmadd_pd(cD512,ass,a0), bt0=_mm512_fnmadd_pd(cD512,bss,b0); \
            __m512d at1=_mm512_mul_pd(cA512,_mm512_sub_pd(as1,as2)); \
            __m512d bt1=_mm512_mul_pd(cA512,_mm512_sub_pd(bs1,bs2)); \
            __m512d ap1=_mm512_add_pd(at0,at1), ap2=_mm512_sub_pd(at0,at1); \
            __m512d bp1=_mm512_add_pd(bt0,bt1), bp2=_mm512_sub_pd(bt0,bt1); \
            __m512d aU=_mm512_fmadd_pd(cC512,ad2,_mm512_mul_pd(cB512,ad1)); \
            __m512d aV=_mm512_fnmadd_pd(cC512,ad1,_mm512_mul_pd(cB512,ad2)); \
            __m512d bU=_mm512_fmadd_pd(cC512,bd2,_mm512_mul_pd(cB512,bd1)); \
            __m512d bV=_mm512_fnmadd_pd(cC512,bd1,_mm512_mul_pd(cB512,bd2)); \
            __m512d ajU=VBYI(aU), ajV=VBYI(aV), bjU=VBYI(bU), bjV=VBYI(bV); \
            _mm512_store_pd(&out[2*((k1a+4*1)*K+k)], _mm512_sub_pd(ap1,ajU)); \
            _mm512_store_pd(&out[2*((k1a+4*4)*K+k)], _mm512_add_pd(ap1,ajU)); \
            _mm512_store_pd(&out[2*((k1a+4*2)*K+k)], _mm512_add_pd(ap2,ajV)); \
            _mm512_store_pd(&out[2*((k1a+4*3)*K+k)], _mm512_sub_pd(ap2,ajV)); \
            _mm512_store_pd(&out[2*((k1b+4*1)*K+k)], _mm512_sub_pd(bp1,bjU)); \
            _mm512_store_pd(&out[2*((k1b+4*4)*K+k)], _mm512_add_pd(bp1,bjU)); \
            _mm512_store_pd(&out[2*((k1b+4*2)*K+k)], _mm512_add_pd(bp2,bjV)); \
            _mm512_store_pd(&out[2*((k1b+4*3)*K+k)], _mm512_sub_pd(bp2,bjV)); \
        }
        
        DFT5_PAIR(0, 1)
        DFT5_PAIR(2, 3)
    }
    #undef VZMUL_IL
    #undef VNBYI
    #undef VBYI
    #undef ITW_APPLY
    #undef DFT5_PAIR
    #undef IW
}

static void s2il(const double *re, const double *im, double *il, size_t N) {
    for (size_t i = 0; i < N; i++) { il[2*i] = re[i]; il[2*i+1] = im[i]; }
}

int main(void) {
    printf("=== Simulated Planner: best-of-all vs FFTW ===\n\n");
    printf("  %6s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %6s\n",
           "K", "sp_CT", "sp_DAG", "IL_N1", "IL_TW*", "FFTW", "best", "ratio", "pick");

    int Kv[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    int nK = 14;

    for (int ki = 0; ki < nK; ki++) {
        int K = Kv[ki];
        size_t N = 20 * (size_t)K;

        double *ir = aligned_alloc(64, N*8), *ii = aligned_alloc(64, N*8);
        double *or_ = aligned_alloc(64, N*8), *oi = aligned_alloc(64, N*8);
        double *il_in = aligned_alloc(64, 2*N*8), *il_out = aligned_alloc(64, 2*N*8);
        /* Pre-interleaved IL twiddle table: 19 entries × K × 2 (re,im) */
        double *tw_il = aligned_alloc(64, 19*K*2*8);
        /* Split twiddles for DAG */
        double *tw_re = aligned_alloc(64, 19*K*8), *tw_im = aligned_alloc(64, 19*K*8);

        if (!ir||!ii||!or_||!oi||!il_in||!il_out||!tw_il||!tw_re||!tw_im) {
            printf("  K=%d: alloc fail\n", K); continue;
        }

        srand(42 + K);
        for (size_t i = 0; i < N; i++) {
            ir[i] = (double)rand()/RAND_MAX*2-1;
            ii[i] = (double)rand()/RAND_MAX*2-1;
        }
        s2il(ir, ii, il_in, N);

        /* Identity twiddles */
        for (size_t i = 0; i < 19*(size_t)K; i++) { tw_re[i]=1.0; tw_im[i]=0.0; }
        make_il_twiddles(tw_il, 20, K);

        int reps = K<=32?100000:(K<=256?20000:(K<=2048?2000:(K<=8192?500:100)));
        int trials = 5;

        double b_sp_ct, b_sp_dag, b_il_n1, b_il_tw_preil, b_fftw;

        /* Split CT (AVX-512) */
        if (K >= 8 && (K & 7) == 0) {
            BENCH(b_sp_ct, radix20_ct_n1_kernel_fwd_avx512(ir,ii,or_,oi,K))
        } else {
            BENCH(b_sp_ct, radix20_ct_n1_fwd_avx2(ir,ii,or_,oi,K))
        }

        /* Split DAG (AVX2) */
        if (K >= 4 && (K & 3) == 0) {
            BENCH(b_sp_dag, radix20_n1_dag_fwd_avx2(ir,ii,or_,oi,K))
        } else {
            b_sp_dag = 1e18;
        }

        /* IL NOTW (AVX-512) */
        if (K >= 4 && (K & 3) == 0) {
            BENCH(b_il_n1, radix20_ct_n1_kernel_fwd_il_avx512(il_in,il_out,K))
        } else {
            b_il_n1 = 1e18;
        }

        /* IL TW with pre-interleaved twiddles (AVX-512) */
        if (K >= 4 && (K & 3) == 0) {
            BENCH(b_il_tw_preil, radix20_ct_tw_dit_preil_fwd_avx512(il_in,il_out,tw_il,K))
        } else {
            b_il_tw_preil = 1e18;
        }

        /* FFTW */
        fftw_complex *fi = fftw_malloc(sizeof(fftw_complex)*N);
        fftw_complex *fo = fftw_malloc(sizeof(fftw_complex)*N);
        fftw_iodim dims = {20,(int)K,(int)K};
        fftw_iodim hdims = {(int)K,1,1};
        fftw_plan p = fftw_plan_guru_dft(1,&dims,1,&hdims,fi,fo,FFTW_FORWARD,FFTW_MEASURE);
        for (size_t m = 0; m < N; m++) { fi[m][0]=ir[m]; fi[m][1]=ii[m]; }
        BENCH(b_fftw, fftw_execute(p))

        /* Find best across all strategies */
        double candidates[] = {b_sp_ct, b_sp_dag, b_il_n1, b_il_tw_preil};
        const char *labels[] = {"sp_CT", "sp_DAG", "IL_N1", "IL_TW*"};
        double best = candidates[0]; int bi = 0;
        for (int i = 1; i < 4; i++) if (candidates[i] < best) { best = candidates[i]; bi = i; }

        double ratio = b_fftw / best;
        const char *win = ratio > 1.02 ? "WIN" : ratio < 0.98 ? "LOSE" : "TIE";

        printf("  %6d  %7.0f  %7.0f  %7.0f  %7.0f  %7.0f  %7.0f  %5.2fx %s %s\n",
               K, b_sp_ct, b_sp_dag, b_il_n1, b_il_tw_preil, b_fftw, best, ratio, win, labels[bi]);

        fftw_destroy_plan(p); fftw_free(fi); fftw_free(fo);
        free(ir); free(ii); free(or_); free(oi);
        free(il_in); free(il_out); free(tw_il); free(tw_re); free(tw_im);
    }

    return 0;
}
