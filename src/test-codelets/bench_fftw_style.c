/**
 * bench_fftw_style.c — FFTW-style recursive CT executor
 *
 * FFTW's actual plan for N=32768 (from fftw_print_plan):
 *
 *   (dft-ct-dit/16
 *     (dftw-direct-16 "t2sv_16")          ← outermost twiddle stage
 *     (dft-vrank>=1-x16/1                 ← loop 16x
 *       (dft-ct-dit/8
 *         (dftw-direct-8 "t2sv_8")        ← R=8 twiddle stage
 *         (dft-vrank>=1-x8/1              ← loop 8x
 *           (dft-ct-dit/32
 *             (dftw-direct-32 "t1sv_32")  ← R=32 twiddle stage
 *             (dft-vrank>=1-x32/1         ← loop 32x
 *               (dft-direct-8 "n1_8")))))))  ← DFT-8 leaf
 *
 * DIT execution order (inner first):
 *   1. 32 × DFT-8 on 256 contiguous elements      (4KB, L1)
 *   2. R=32 twiddle on those 256 elements
 *   3. Repeat 1-2 eight times → 2048 elements      (32KB, L1)
 *   4. R=8 twiddle on those 2048 elements
 *   5. Repeat 1-4 sixteen times → 32768 elements
 *   6. R=16 twiddle on all 32768 elements           (512KB, L2)
 *
 * Cache: steps 1-4 stay in L1. Only step 6 touches full array.
 * NO transposes. NO global permutation. Just recursive staging.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_n1_k1.h"
#include "fft_n1_k1_simd.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_notw.h"
#include "fft_radix16_avx2_dit_tw.h"

#define N_FFT 32768

static inline double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void init_tw(double *twr, double *twi, size_t R, size_t K) {
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / (double)(R * K);
            twr[(n-1)*K+k] = cos(a); twi[(n-1)*K+k] = sin(a);
        }
}

static size_t *build_perm(const size_t *rx, size_t ns, size_t N) {
    size_t *p = malloc(N * sizeof(size_t));
    for (size_t i = 0; i < N; i++) {
        size_t rem=i, rev=0, str=N;
        for (size_t s=0;s<ns;s++){str/=rx[s];rev+=(rem%rx[s])*str;rem/=rx[s];}
        p[i]=rev;
    }
    return p;
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN NODE — FFTW-style recursive CT
 *
 * Three node types:
 *   DIRECT — leaf DFT using n1 codelet (no twiddle)
 *   CT_DIT — one radix-R stage: execute child, then apply twiddle
 *   VLOOP  — repeat child plan vl times at stride vs
 * ═══════════════════════════════════════════════════════════════ */

typedef void (*notw_fn)(const double*,const double*,double*,double*,size_t);
typedef void (*tw_fn)(const double*,const double*,double*,double*,
                      const double*,const double*,size_t);

typedef enum { NODE_DIRECT, NODE_CT_DIT, NODE_VLOOP } node_type;

typedef struct plan_node {
    node_type type;

    /* NODE_DIRECT: n1 codelet, applied once */
    notw_fn n1_fn;
    size_t n1_K;           /* stride parameter for n1 codelet */

    /* NODE_CT_DIT: twiddle codelet + child */
    tw_fn tw_fn;
    size_t tw_R, tw_K;     /* radix and stride for twiddle stage */
    double *tw_re, *tw_im; /* twiddle table */
    struct plan_node *child;

    /* NODE_VLOOP: repeat child vl times at stride vs */
    size_t vl, vs;
    struct plan_node *loop_child;
} plan_node;

/* ═══════════════════════════════════════════════════════════════
 * RECURSIVE EXECUTOR
 *
 * The trick: we need ping-pong buffers because our codelets are
 * out-of-place. FFTW's codelets are in-place (twiddle operates on
 * the data where it sits). We emulate this with buffer management.
 *
 * For DIT at each CT level:
 *   1. Execute child: perm_in → buf_a
 *   2. Twiddle stage: buf_a → buf_b
 *   3. Next level treats buf_b as input
 *
 * The number of stages determines which buffer holds the final result.
 * We handle this by having the executor return which buffer has the data.
 * ═══════════════════════════════════════════════════════════════ */

/* Execute plan node. Returns 0 if result is in (a,ai), 1 if in (b,bi).
 * Input is always at (src_re, src_im).
 * Two scratch buffers a,b are available. */
static int rexec(const plan_node *p,
                  const double *src_re, const double *src_im,
                  double *a_re, double *a_im,
                  double *b_re, double *b_im)
{
    switch (p->type) {

    case NODE_DIRECT:
        /* Leaf: apply n1 codelet, output to a */
        p->n1_fn(src_re, src_im, a_re, a_im, p->n1_K);
        return 0; /* result in a */

    case NODE_CT_DIT: {
        /* DIT: execute child first, then apply twiddle stage */

        /* Child writes to a (or b, depending on child's depth) */
        int child_buf = rexec(p->child, src_re, src_im, a_re, a_im, b_re, b_im);

        /* Twiddle reads from child's output buffer, writes to the other */
        double *in_r, *in_i, *out_r, *out_i;
        int result_buf;
        if (child_buf == 0) {
            in_r = a_re; in_i = a_im; out_r = b_re; out_i = b_im;
            result_buf = 1;
        } else {
            in_r = b_re; in_i = b_im; out_r = a_re; out_i = a_im;
            result_buf = 0;
        }

        /* Apply twiddle stage: R groups of K elements */
        size_t R = p->tw_R, K = p->tw_K;
        size_t ng = 1; /* CT_DIT node processes exactly one "super-group" */
        (void)ng;
        p->tw_fn(in_r, in_i, out_r, out_i, p->tw_re, p->tw_im, K);
        return result_buf;
    }

    case NODE_VLOOP: {
        /* Loop: execute child vl times.
         * All iterations must write to the same buffer.
         * We enforce this by making every iteration use the same
         * a/b buffers at the correct offsets. */
        int result_buf = -1;
        for (size_t v = 0; v < p->vl; v++) {
            size_t off = v * p->vs;
            int buf = rexec(p->loop_child,
                             src_re + off, src_im + off,
                             a_re + off, a_im + off,
                             b_re + off, b_im + off);
            if (result_buf < 0) result_buf = buf;
            /* All iterations produce into the same buffer (a or b)
             * because the child plan structure is identical each time */
        }
        return result_buf;
    }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * BUILD PLAN for N=32768
 *
 * Our factorization: 16 × 16 × 8 × 16 (inner→outer)
 * Mimicking FFTW's structure:
 *   Inner: R=16 K=1 (n1 codelet, no twiddle)
 *   Stage 1: R=16 K=16 (twiddle)
 *   Vloop ×8
 *   Stage 2: R=8 K=256 (twiddle)
 *   Vloop ×16
 *   Stage 3: R=16 K=2048 (twiddle) — unavoidable full-array pass
 *
 * But wait — the input needs permutation. FFTW avoids this because
 * its n1 codelets access data at stride K. The permutation is implicit
 * in the strided access pattern.
 *
 * For our DIT codelets: the n1 codelet at K=1 reads contiguous data
 * and produces natural-order output. The twiddle codelet at K=16
 * reads/writes at stride K.
 *
 * The permutation question: in our flat executor we explicitly permute
 * input. In FFTW's structure the permutation is handled by the
 * combination of strided access + recursive structure.
 *
 * For V1: keep the explicit permutation but apply it only to the
 * sub-problem that's about to be processed, not the whole array.
 * ═══════════════════════════════════════════════════════════════ */

static void r16_k1_wrap(const double *ir, const double *ii,
                          double *or_, double *oi, size_t K) {
    (void)K; dft16_k1_fwd_avx2(ir, ii, or_, oi);
}

/* Build the plan tree */
static double *tw_s1_re, *tw_s1_im;
static double *tw_s2_re, *tw_s2_im;
static double *tw_s3_re, *tw_s3_im;
static size_t *g_perm;
static plan_node g_nodes[20]; /* static allocation for simplicity */

static void build_plan(void) {
    /* Twiddles */
    tw_s1_re = aligned_alloc(32, 15*16*8);  tw_s1_im = aligned_alloc(32, 15*16*8);
    tw_s2_re = aligned_alloc(32, 7*256*8);  tw_s2_im = aligned_alloc(32, 7*256*8);
    tw_s3_re = aligned_alloc(32, 15*2048*8);tw_s3_im = aligned_alloc(32, 15*2048*8);
    init_tw(tw_s1_re, tw_s1_im, 16, 16);
    init_tw(tw_s2_re, tw_s2_im, 8, 256);
    init_tw(tw_s3_re, tw_s3_im, 16, 2048);

    /* Permutation (still needed for DIT input ordering) */
    size_t rx[] = {16, 16, 8, 16};
    g_perm = build_perm(rx, 4, N_FFT);

    /* Node 0: innermost — R=16 K=1, n1 codelet */
    g_nodes[0] = (plan_node){
        .type = NODE_DIRECT,
        .n1_fn = r16_k1_wrap,
        .n1_K = 1
    };

    /* Node 1: vloop — 16 copies of node 0, stride 16 */
    g_nodes[1] = (plan_node){
        .type = NODE_VLOOP,
        .vl = 16, .vs = 16,
        .loop_child = &g_nodes[0]
    };

    /* Node 2: R=16 K=16 twiddle stage */
    g_nodes[2] = (plan_node){
        .type = NODE_CT_DIT,
        .tw_fn = (tw_fn)radix16_tw_flat_dit_kernel_fwd_avx2,
        .tw_R = 16, .tw_K = 16,
        .tw_re = tw_s1_re, .tw_im = tw_s1_im,
        .child = &g_nodes[1]
    };

    /* Node 3: vloop — 8 copies of node 2, stride 256
     * Each iteration processes a 256-element sub-problem */
    g_nodes[3] = (plan_node){
        .type = NODE_VLOOP,
        .vl = 8, .vs = 256,
        .loop_child = &g_nodes[2]
    };

    /* Node 4: R=8 K=256 twiddle stage */
    g_nodes[4] = (plan_node){
        .type = NODE_CT_DIT,
        .tw_fn = (tw_fn)radix8_tw_dit_kernel_fwd_avx2,
        .tw_R = 8, .tw_K = 256,
        .tw_re = tw_s2_re, .tw_im = tw_s2_im,
        .child = &g_nodes[3]
    };

    /* Node 5: vloop — 16 copies of node 4, stride 2048
     * Each iteration processes a 2048-element sub-problem */
    g_nodes[5] = (plan_node){
        .type = NODE_VLOOP,
        .vl = 16, .vs = 2048,
        .loop_child = &g_nodes[4]
    };

    /* Node 6: R=16 K=2048 outermost twiddle */
    g_nodes[6] = (plan_node){
        .type = NODE_CT_DIT,
        .tw_fn = (tw_fn)radix16_tw_flat_dit_kernel_fwd_avx2,
        .tw_R = 16, .tw_K = 2048,
        .tw_re = tw_s3_re, .tw_im = tw_s3_im,
        .child = &g_nodes[5]
    };
}

/* Top-level execute: permute input, then run recursive plan */
static void execute_recursive(const double *in_re, const double *in_im,
                                double *out_re, double *out_im,
                                double *a_re, double *a_im,
                                double *b_re, double *b_im)
{
    /* Permute input into a */
    for (size_t i = 0; i < N_FFT; i++) {
        a_re[i] = in_re[g_perm[i]];
        a_im[i] = in_im[g_perm[i]];
    }

    /* Execute recursive plan. Input is in a, uses a and b as ping-pong. */
    int result = rexec(&g_nodes[6], a_re, a_im, a_re, a_im, b_re, b_im);

    /* Copy result to output */
    double *res_re = (result == 0) ? a_re : b_re;
    double *res_im = (result == 0) ? a_im : b_im;
    if (res_re != out_re) { memcpy(out_re, res_re, N_FFT*8); memcpy(out_im, res_im, N_FFT*8); }
}

/* ═══════════════════════════════════════════════════════════════
 * FLAT EXECUTOR (comparison)
 * ═══════════════════════════════════════════════════════════════ */

static void exec_flat(const double *ir, const double *ii,
                      double *or_, double *oi,
                      double *a, double *ai, double *b, double *bi)
{
    for (size_t i=0;i<N_FFT;i++){a[i]=ir[g_perm[i]];ai[i]=ii[g_perm[i]];}
    for(size_t g=0;g<2048;g++){size_t o=g*16;
        r16_k1_wrap(a+o,ai+o,b+o,bi+o,1);}
    for(size_t g=0;g<128;g++){size_t o=g*256;
        radix16_tw_flat_dit_kernel_fwd_avx2(b+o,bi+o,a+o,ai+o,tw_s1_re,tw_s1_im,16);}
    for(size_t g=0;g<16;g++){size_t o=g*2048;
        radix8_tw_dit_kernel_fwd_avx2(a+o,ai+o,b+o,bi+o,tw_s2_re,tw_s2_im,256);}
    radix16_tw_flat_dit_kernel_fwd_avx2(b,bi,or_,oi,tw_s3_re,tw_s3_im,2048);
}

/* ═══════════════════════════════════════════════════════════════ */

static double bench_fftw(int reps) {
    double *ri=fftw_malloc(N_FFT*8),*ii_=fftw_malloc(N_FFT*8);
    double *ro=fftw_malloc(N_FFT*8),*io=fftw_malloc(N_FFT*8);
    for(size_t i=0;i<N_FFT;i++){ri[i]=(double)rand()/RAND_MAX;ii_[i]=(double)rand()/RAND_MAX;}
    fftw_iodim d={.n=N_FFT,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=N_FFT,.os=N_FFT};
    fftw_plan p=fftw_plan_guru_split_dft(1,&d,1,&h,ri,ii_,ro,io,FFTW_MEASURE);
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii_,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii_);fftw_free(ro);fftw_free(io);
    return best;
}

int main(void)
{
    srand(42);
    printf("================================================================\n");
    printf("  N=%d: FFTW-style recursive CT vs Flat vs FFTW\n", N_FFT);
    printf("  Factorization: 16 × 16 × 8 × 16 (inner→outer)\n");
    printf("  Sub-problem sizes: 256 (L1), 2048 (L1), 32768 (L2)\n");
    printf("================================================================\n\n");

    build_plan();

    double *in_re=aligned_alloc(32,N_FFT*8),*in_im=aligned_alloc(32,N_FFT*8);
    double *out_re=aligned_alloc(32,N_FFT*8),*out_im=aligned_alloc(32,N_FFT*8);
    double *a=aligned_alloc(32,N_FFT*8),*ai=aligned_alloc(32,N_FFT*8);
    double *b=aligned_alloc(32,N_FFT*8),*bi=aligned_alloc(32,N_FFT*8);
    for(size_t i=0;i<N_FFT;i++){in_re[i]=(double)rand()/RAND_MAX-.5;in_im[i]=(double)rand()/RAND_MAX-.5;}

    /* FFTW reference */
    double *fr=fftw_malloc(N_FFT*8),*fi=fftw_malloc(N_FFT*8);
    double *fo_r=fftw_malloc(N_FFT*8),*fo_i=fftw_malloc(N_FFT*8);
    memcpy(fr,in_re,N_FFT*8);memcpy(fi,in_im,N_FFT*8);
    fftw_iodim d={.n=N_FFT,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=N_FFT,.os=N_FFT};
    fftw_plan fp=fftw_plan_guru_split_dft(1,&d,1,&h,fr,fi,fo_r,fo_i,FFTW_ESTIMATE);
    fftw_execute(fp);

    printf("Correctness vs FFTW:\n");

    exec_flat(in_re,in_im,out_re,out_im,a,ai,b,bi);
    double e=0;for(size_t i=0;i<N_FFT;i++){double d_=fabs(out_re[i]-fo_r[i])+fabs(out_im[i]-fo_i[i]);if(d_>e)e=d_;}
    printf("  Flat:       %.2e %s\n",e,e<1e-10?"OK":"FAIL");

    execute_recursive(in_re,in_im,out_re,out_im,a,ai,b,bi);
    e=0;for(size_t i=0;i<N_FFT;i++){double d_=fabs(out_re[i]-fo_r[i])+fabs(out_im[i]-fo_i[i]);if(d_>e)e=d_;}
    printf("  Recursive:  %.2e %s\n",e,e<1e-10?"OK":"FAIL");

    if(e>1e-10){printf("\nFAIL — skipping perf\n");return 1;}

    int reps=200;
    printf("\nBenchmarking (%d reps, 7 trials, best-of)...\n\n",reps);

    double fftw_ns=bench_fftw(reps);

    for(int i=0;i<20;i++) exec_flat(in_re,in_im,out_re,out_im,a,ai,b,bi);
    double flat_best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++) exec_flat(in_re,in_im,out_re,out_im,a,ai,b,bi);
        double ns=(now_ns()-t0)/reps;if(ns<flat_best)flat_best=ns;}

    for(int i=0;i<20;i++) execute_recursive(in_re,in_im,out_re,out_im,a,ai,b,bi);
    double rec_best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++) execute_recursive(in_re,in_im,out_re,out_im,a,ai,b,bi);
        double ns=(now_ns()-t0)/reps;if(ns<rec_best)rec_best=ns;}

    printf("  %-28s %8.0f ns\n","FFTW_MEASURE:",fftw_ns);
    printf("  %-28s %8.0f ns  (%.2fx vs FFTW)\n","Flat (stage-by-stage):",flat_best,fftw_ns/flat_best);
    printf("  %-28s %8.0f ns  (%.2fx vs FFTW)\n","Recursive CT (FFTW-style):",rec_best,fftw_ns/rec_best);
    printf("\n  Recursive vs Flat: %.2fx\n",flat_best/rec_best);

    fftw_destroy_plan(fp);fftw_free(fr);fftw_free(fi);fftw_free(fo_r);fftw_free(fo_i);
    return 0;
}
