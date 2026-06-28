/* test_trig_oddk.c — arbitrary-K tail correctness for the dag trig (r2r) codelets.
 *
 * The trig codelets have the uniform ABI fn(const double *in, double *out, K),
 * processing K independent N-point real-to-real transforms laid out leg-major
 * with stride K:  in[leg*K + b]  (leg in 0..N-1, batch column b in 0..K-1).
 *
 * Each batch column b is an INDEPENDENT N-point transform of input column b.
 * So we validate the arbitrary-K tail by self-consistency, with NO dependence
 * on the exact transform normalization:
 *
 *   1. Run the codelet at an even reference K=8 over random input  -> out_ref.
 *   2. For odd K in {1,3,5,7}: repack the first K columns of the SAME input
 *      into a K-strided buffer, run the codelet -> out_k, and compare
 *      out_k[leg*K+b] vs out_ref[leg*8+b] for every leg and b<K.
 *
 * Masked tail columns (rem>=2) run the SAME __m256d FMA arithmetic as the bulk
 * -> bit-exact. The scalar tail column (rem==1, force-mono scalar render) may
 * differ only by FMA-vs-scalar last-bit rounding (~1e-16). Tolerance 1e-11.
 *
 * Coverage: K=7 exercises bulk(k=0..3) + masked tail(rem=3); K=5 exercises
 * bulk(k=0..3) + scalar tail(rem=1); K=1 exercises scalar-only (no bulk).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef void (*trig_fn)(const double *in, double *out, size_t K);

/* Direct externs of the dag trig-avx2 codelets (no plan shell, no K guard). */
#define DECL(name) extern void name(const double*, double*, size_t)
DECL(radix8_dct2_avx2);  DECL(radix16_dct2_avx2);  DECL(radix32_dct2_avx2);  DECL(radix64_dct2_avx2);
DECL(radix8_dct3_avx2);  DECL(radix16_dct3_avx2);  DECL(radix32_dct3_avx2);  DECL(radix64_dct3_avx2);
DECL(radix8_dct4_avx2);  DECL(radix16_dct4_avx2);  DECL(radix32_dct4_avx2);  DECL(radix64_dct4_avx2);
DECL(radix8_dst2_avx2);  DECL(radix16_dst2_avx2);  DECL(radix32_dst2_avx2);  DECL(radix64_dst2_avx2);
DECL(radix8_dst3_avx2);  DECL(radix16_dst3_avx2);  DECL(radix32_dst3_avx2);  DECL(radix64_dst3_avx2);
DECL(radix8_dst4_avx2);  DECL(radix16_dst4_avx2);  DECL(radix32_dst4_avx2);  DECL(radix64_dst4_avx2);
DECL(radix8_dht_avx2);   DECL(radix16_dht_avx2);   DECL(radix32_dht_avx2);   DECL(radix64_dht_avx2);

/* Direct O(N^2) FFTW-convention references (column 0), for the kinds without
 * boundary terms. Confirms the regen didn't corrupt the bulk AND that the codelet
 * definition/order matches FFTW. dct3/dst3 (boundary terms) use self-consistency only. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static double col0(const double *x, int N, int K, int n) { return x[(size_t)n*K + 0]; }
static double ref_dct2(const double*x,int N,int K,int k){double s=0;for(int n=0;n<N;n++)s+=col0(x,N,K,n)*cos(M_PI*(2*n+1)*k/(2.0*N));return 2*s;}
static double ref_dst2(const double*x,int N,int K,int k){double s=0;for(int n=0;n<N;n++)s+=col0(x,N,K,n)*sin(M_PI*(2*n+1)*(k+1)/(2.0*N));return 2*s;}
static double ref_dct4(const double*x,int N,int K,int k){double s=0;for(int n=0;n<N;n++)s+=col0(x,N,K,n)*cos(M_PI*(2*n+1)*(2*k+1)/(4.0*N));return 2*s;}
static double ref_dst4(const double*x,int N,int K,int k){double s=0;for(int n=0;n<N;n++)s+=col0(x,N,K,n)*sin(M_PI*(2*n+1)*(2*k+1)/(4.0*N));return 2*s;}
static double ref_dht (const double*x,int N,int K,int k){double s=0;for(int n=0;n<N;n++){double a=2.0*M_PI*k*n/N;s+=col0(x,N,K,n)*(cos(a)+sin(a));}return s;}
typedef double (*ref_fn)(const double*,int,int,int);
static ref_fn ref_for(const char*kind){
    if(!strcmp(kind,"dct2"))return ref_dct2; if(!strcmp(kind,"dst2"))return ref_dst2;
    if(!strcmp(kind,"dct4"))return ref_dct4; if(!strcmp(kind,"dst4"))return ref_dst4;
    if(!strcmp(kind,"dht")) return ref_dht;  return 0; /* dct3/dst3: self-consistency only */
}

typedef struct { const char *kind; int N; trig_fn fn; } cell_t;
static const cell_t CELLS[] = {
    {"dct2", 8,radix8_dct2_avx2},{"dct2",16,radix16_dct2_avx2},{"dct2",32,radix32_dct2_avx2},{"dct2",64,radix64_dct2_avx2},
    {"dct3", 8,radix8_dct3_avx2},{"dct3",16,radix16_dct3_avx2},{"dct3",32,radix32_dct3_avx2},{"dct3",64,radix64_dct3_avx2},
    {"dct4", 8,radix8_dct4_avx2},{"dct4",16,radix16_dct4_avx2},{"dct4",32,radix32_dct4_avx2},{"dct4",64,radix64_dct4_avx2},
    {"dst2", 8,radix8_dst2_avx2},{"dst2",16,radix16_dst2_avx2},{"dst2",32,radix32_dst2_avx2},{"dst2",64,radix64_dst2_avx2},
    {"dst3", 8,radix8_dst3_avx2},{"dst3",16,radix16_dst3_avx2},{"dst3",32,radix32_dst3_avx2},{"dst3",64,radix64_dst3_avx2},
    {"dst4", 8,radix8_dst4_avx2},{"dst4",16,radix16_dst4_avx2},{"dst4",32,radix32_dst4_avx2},{"dst4",64,radix64_dst4_avx2},
    {"dht",  8,radix8_dht_avx2}, {"dht", 16,radix16_dht_avx2}, {"dht", 32,radix32_dht_avx2}, {"dht", 64,radix64_dht_avx2},
};
#define NCELLS ((int)(sizeof(CELLS)/sizeof(CELLS[0])))

#define KREF 8
static const int KTEST[] = {7, 5, 3, 1};   /* masked rem3 ; scalar rem1 ; masked rem1->scalar ; scalar only */
#define NKTEST ((int)(sizeof(KTEST)/sizeof(KTEST[0])))

int main(void) {
    const double TOL = 1e-11;
    int fails = 0;
    double worst = 0.0;
    printf("# trig arbitrary-K tail correctness (per-column vs even K=%d reference)\n", KREF);
    printf("# %-6s %4s | %-8s %-8s %-8s %-8s\n", "kind", "N", "K=7", "K=5", "K=3", "K=1");

    for (int c = 0; c < NCELLS; c++) {
        int N = CELLS[c].N;
        trig_fn fn = CELLS[c].fn;
        size_t pad = 8;                       /* slack so maskload/store never touches OOB */
        double *in_ref  = malloc((size_t)(N*KREF + pad) * sizeof(double));
        double *out_ref = malloc((size_t)(N*KREF + pad) * sizeof(double));
        srand(1000 + c);
        for (int i = 0; i < N*KREF; i++) in_ref[i] = (double)rand() / RAND_MAX - 0.5;
        for (size_t i = N*KREF; i < (size_t)(N*KREF+pad); i++) in_ref[i] = 0.0;
        fn(in_ref, out_ref, KREF);

        /* Independent even-K bulk check vs direct FFTW-convention formula (column 0). */
        double referr = -1.0;
        ref_fn rf = ref_for(CELLS[c].kind);
        if (rf) {
            referr = 0.0;
            for (int k = 0; k < N; k++) {
                double e = fabs(out_ref[(size_t)k*KREF + 0] - rf(in_ref, N, KREF, k));
                if (e > referr) referr = e;
            }
            if (referr > 1e-9) fails++;       /* bulk corrupt or convention mismatch */
        }

        printf("  %-6s %4d |", CELLS[c].kind, N);
        for (int t = 0; t < NKTEST; t++) {
            int K = KTEST[t];
            double *in_k  = malloc((size_t)(N*K + pad) * sizeof(double));
            double *out_k = malloc((size_t)(N*K + pad) * sizeof(double));
            for (int leg = 0; leg < N; leg++)
                for (int b = 0; b < K; b++)
                    in_k[leg*K + b] = in_ref[leg*KREF + b];
            for (size_t i = N*K; i < (size_t)(N*K+pad); i++) in_k[i] = 0.0;
            fn(in_k, out_k, (size_t)K);

            double maxerr = 0.0;
            for (int leg = 0; leg < N; leg++)
                for (int b = 0; b < K; b++) {
                    double e = fabs(out_k[leg*K + b] - out_ref[leg*KREF + b]);
                    if (e > maxerr) maxerr = e;
                }
            if (maxerr > worst) worst = maxerr;
            printf(" %.1e%s", maxerr, maxerr > TOL ? "*" : " ");
            if (maxerr > TOL) fails++;
            free(in_k); free(out_k);
        }
        if (referr >= 0.0) printf("  refK8=%.1e%s", referr, referr > 1e-9 ? "*" : "");
        printf("\n");
        free(in_ref); free(out_ref);
    }
    printf("\n# worst err = %.2e   %s (tol=%.0e, '*' marks fail)\n",
           worst, fails ? "FAIL" : "PASS", TOL);
    return fails ? 1 : 0;
}
