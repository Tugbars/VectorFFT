/* half_store_bench.c -- does AVX-512 want half-store twins of the forward
 * column leaf? The AVX2 tree has them (src/dag-fft-compiler/codelets/zil/avx2/
 * shared/col/half/README.md); this bench times the AVX-512 leaf with its plane
 * stores at three widths, the way the 2D real r2c plan times its leaf forms.
 *
 * THE QUESTION. The 2D real r2c column pass runs in place at the row pitch
 * hp1 = N2/2 + 1, odd for every even N2. At AVX2 a vector is 32 bytes, a
 * quarter of the leaf's plane stores cross a 64-byte line there, and the half
 * twins (each vector stored as two 128-bit halves) race against the full-store
 * leaf for that plan. At AVX-512 a vector is 64 bytes: at an odd pitch three
 * rows in four start off a line boundary, so three of four plane stores split.
 *
 * THE ARMS, from the SHIPPED AVX-512 kernel source (no new generated file).
 * Compiled with -DHALF_STORE_ARM=<width> -DHALF_STORE_KERNEL="<kernel .c>"
 * -D<symbol>=half_store_st<width>_<symbol>, this file is one arm: it includes
 * the kernel's source with _mm512_storeu_pd redefined, under its own name.
 *   st512  as shipped: one 512-bit store
 *   st256  two 256-bit halves (vmovupd ymm, vextractf64x4 to memory)
 *   st128  four 128-bit quarters (vmovupd xmm, vextractf64x2 to memory), the
 *          AVX2 twin's store width
 * The tail (the masked k-pass, the ymm and xmm ladder) keeps its stores, as
 * the AVX2 twin keeps its VEX-128 tail. Compiled plainly, this file is the
 * bench; half_store_bench.sh builds both (gauntlet/build.py loaded as a
 * module, as src/tools/gates/twin_gate_build.py does), the arms at the codelet
 * library's flags.
 *
 * THE KERNELS (HALF_STORE_KERNELS): every forward column leaf with an AVX2
 * half twin whose every _mm512_storeu_pd is a plane store of the wide path,
 * the stores the AVX2 twin splits, and nothing else under that name: 13 of the
 * 32. Left out, because their wide path also stores a half-DFT spill into the
 * stack scratch S[] with _mm512_storeu_pd, which the AVX2 twin keeps whole:
 * the n1c leaves at radix 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 37,
 * 41, 43, 47 and the blocked n1cb48, n1cb84 (radix 32) and n1cb88 (radix 64).
 * At radix 3, 5, 6, 10 and 12 the AVX2 half twin carries an older arithmetic
 * body than its full twin; its stores are still exactly the wide plane
 * stores, which is the rule the arms here follow.
 *
 * CORRECTNESS FIRST. Before any timing, every arm of every kernel runs in
 * place on random input at an odd and an even pitch, at column counts that
 * reach the wide path and every tail arm (count mod 4 = 0, 1, 2, 3), and its
 * whole buffer -- the gaps past the count and guard bands on both sides
 * included -- must be BITWISE the st512 arm's, and the st512 arm bitwise the
 * library's own kernel. A mismatch stops the program with the case.
 *
 * TIMING. One sample is the leaf stage of the column pass as _il2d_col_pass
 * runs it (il2d_cols.h; _il2d_race_forms in il2d_tier.h times the whole pass
 * with it): N1/R calls, each over all hp1 columns of R rows, Ls = OLs = the
 * pitch, in place; the plane re-laid from its seed before every sample, as
 * the race does, and the sample sized to about 1 ms on st512 (one pass at
 * least; in place the values grow by up to R a pass, so R^reps stays finite).
 * A cell is a real 2D r2c shape N1 x N2 whose column leaf is radix R (N1 a
 * multiple of R); its arms are st512, st256 and st128 at the odd pitch hp1,
 * and the CONTROL, st512 at the even pitch hp1 + 1 over the same hp1 columns.
 * The default cells: leaves 4, 8, 16, 3, 5, 7; N2 = 64, 256, 1024, 4096
 * (hp1 = 33, 129, 513, 2049); N1 = R times the power of two that puts the
 * plane near 256 KiB (inside L2), 4 MiB (in L3) and 64 MiB (past it).
 *
 * PROTOCOL (the one-thread protocol, bench_scope.h): the library's
 * measurement scope -- pinned to one physical core, its SMT sibling held, the
 * process at HIGH priority, the machine's measurement lock -- on Intel and on
 * AMD alike; the arms ALTERNATE within a round (forward, then backward the
 * next round); 25 rounds; 200 ms idle between rounds. Reported per cell and
 * arm: the median and the spread (min..max) of the rounds, and the ratio of
 * each split arm's median to st512's at the odd pitch.
 *
 * Usage: half_store_bench [--check] [--quick] [--leaves 4,8,16,3,5,7 | all]
 *                         [--rounds N] [--pace-ms M] [--core C] [--out DIR]
 *   --check   the correctness check alone (no timing): what Intel SDE can show
 *   --quick   one cell (radix 16, 64 x 64) and 2 rounds: the build and the run
 *             proven end to end, nothing measured
 *   --out     write DIR/table.txt (the table) and DIR/rows.csv (every sample)
 * Exit 2 on a CPU without AVX-512 (checked before any AVX-512 instruction).
 * Build and run: ./gauntlet/half_store_bench.sh on an AVX-512 machine. Under
 * Intel SDE (sde -spr -- ./gauntlet/half_store_bench.exe --check) the check is
 * exact and the timings mean nothing.
 */

/* radix, the kernel's file under codelets/zil/avx512/, its symbol */
#define HALF_STORE_KERNELS(X)                                          \
    X(2, "shared/col/radix2_z_n1c_avx512.c", radix2_z_n1c_fwd_avx512)    \
    X(3, "shared/col/radix3_z_n1c_avx512.c", radix3_z_n1c_fwd_avx512)    \
    X(4, "shared/col/radix4_z_n1c_avx512.c", radix4_z_n1c_fwd_avx512)    \
    X(5, "shared/col/radix5_z_n1c_avx512.c", radix5_z_n1c_fwd_avx512)    \
    X(6, "shared/col/radix6_z_n1c_avx512.c", radix6_z_n1c_fwd_avx512)    \
    X(7, "shared/col/radix7_z_n1c_avx512.c", radix7_z_n1c_fwd_avx512)    \
    X(8, "shared/col/radix8_z_n1c_avx512.c", radix8_z_n1c_fwd_avx512)    \
    X(10, "shared/col/radix10_z_n1c_avx512.c", radix10_z_n1c_fwd_avx512) \
    X(12, "shared/col/radix12_z_n1c_avx512.c", radix12_z_n1c_fwd_avx512) \
    X(14, "shared/col/radix14_z_n1c_avx512.c", radix14_z_n1c_fwd_avx512) \
    X(16, "shared/col/radix16_z_n1c_avx512.c", radix16_z_n1c_fwd_avx512) \
    X(22, "shared/col/radix22_z_n1c_avx512.c", radix22_z_n1c_fwd_avx512) \
    X(26, "shared/col/radix26_z_n1c_avx512.c", radix26_z_n1c_fwd_avx512)

#ifdef HALF_STORE_ARM
/* ── ONE ARM: the kernel's source, its plane stores at the arm's width ────── */
#include <immintrin.h>
#include <stddef.h>
#if HALF_STORE_ARM == 256
#define _mm512_storeu_pd(p, v)                                         \
    do                                                                 \
    {                                                                  \
        double *const hs_p_ = (p);                                     \
        const __m512d hs_v_ = (v);                                     \
        _mm256_storeu_pd(hs_p_, _mm512_castpd512_pd256(hs_v_));        \
        _mm256_storeu_pd(hs_p_ + 4, _mm512_extractf64x4_pd(hs_v_, 1)); \
    } while (0)
#elif HALF_STORE_ARM == 128
#define _mm512_storeu_pd(p, v)                                    \
    do                                                            \
    {                                                             \
        double *const hs_p_ = (p);                                \
        const __m512d hs_v_ = (v);                                \
        _mm_storeu_pd(hs_p_, _mm512_castpd512_pd128(hs_v_));      \
        _mm_storeu_pd(hs_p_ + 2, _mm512_extractf64x2_pd(hs_v_, 1)); \
        _mm_storeu_pd(hs_p_ + 4, _mm512_extractf64x2_pd(hs_v_, 2)); \
        _mm_storeu_pd(hs_p_ + 6, _mm512_extractf64x2_pd(hs_v_, 3)); \
    } while (0)
#elif HALF_STORE_ARM != 512
#error "HALF_STORE_ARM is 512, 256 or 128"
#endif
#include HALF_STORE_KERNEL

#else /* ── THE BENCH ─────────────────────────────────────────────────────── */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bench_scope.h"                  /* the library's measurement scope; vfft.h, vfft_diagnostics.h */
#include "common/support/race_timing.h"   /* vfft_now_ns: the clock every race times with */

#define LEAF_ARGS const double *, const double *, double *, double *, const double *, const double *, \
                  size_t, size_t, size_t, size_t, size_t
typedef void (*leaf_fn)(LEAF_ARGS);

#define DECLARE(R, file, sym)                   \
    extern void sym(LEAF_ARGS);                 \
    extern void half_store_st512_##sym(LEAF_ARGS); \
    extern void half_store_st256_##sym(LEAF_ARGS); \
    extern void half_store_st128_##sym(LEAF_ARGS);
HALF_STORE_KERNELS(DECLARE)

typedef struct
{
    int R;
    const char *file, *name;
    leaf_fn lib;      /* the library's own kernel (the codelet library) */
    leaf_fn arm[3];   /* st512, st256, st128 */
} kernel_t;
#define ROW(R, file, sym) {R, file, #sym, sym, {half_store_st512_##sym, half_store_st256_##sym, half_store_st128_##sym}},
static const kernel_t KERNELS[] = {HALF_STORE_KERNELS(ROW)};
#define NKERNELS ((int)(sizeof KERNELS / sizeof KERNELS[0]))

enum { ST512, ST256, ST128, ST512_EVEN, NARMS };
static const char *const ARM[NARMS] = {"st512", "st256", "st128", "st512_even"};

static void pace(int ms)
{
    struct timespec ts;
    if (ms <= 0)
        return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static double *alloc_d(size_t n)
{
    double *p = (double *)vfft_malloc(n * sizeof(double));   /* the alignment every plane the door serves has */
    if (!p)
    {
        fprintf(stderr, "half_store_bench: out of memory (%zu doubles)\n", n);
        exit(1);
    }
    return p;
}

/* ── THE CORRECTNESS CHECK ───────────────────────────────────────────────── */

static unsigned long long g_rng = 0x9E3779B97F4A7C15ull;
static double rnd(void)
{   /* xorshift64*: [-1, 1) */
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (double)((g_rng * 0x2545F4914F6CDD1Dull) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

#define GUARD 64   /* doubles of canary on each side of the buffer */
static double canary(void)
{   /* a quiet NaN no arithmetic produces: a store into a gap or a guard shows */
    const unsigned long long b = 0x7FF8DEADBEEF0001ull;
    double d;
    memcpy(&d, &b, sizeof d);
    return d;
}

/* runs fn in place on a copy of seed, as the column pass calls the leaf */
static void run_case(leaf_fn fn, const double *seed, double *buf, size_t n, size_t pitch, size_t count)
{
    double *z = buf + GUARD;
    memcpy(buf, seed, n * sizeof(double));
    fn(z, NULL, z, NULL, NULL, NULL, pitch, 0, pitch, 0, count);
}

static void mismatch(const kernel_t *k, const char *what, size_t pitch, size_t count,
                     const double *got, const double *want, size_t n)
{
    size_t i;
    for (i = 0; i < n && !memcmp(got + i, want + i, sizeof(double)); i++)
        ;
    if (i < GUARD || i >= n - GUARD)
        printf("half_store_bench: MISMATCH radix %d (%s): %s at pitch %zu, count %zu: guard double %zu "
               "was written (%a)\n", k->R, k->name, what, pitch, count, i, got[i]);
    else
    {
        const size_t j = i - GUARD, row = j / (2 * pitch), col = (j % (2 * pitch)) / 2;
        printf("half_store_bench: MISMATCH radix %d (%s): %s at pitch %zu, count %zu: row %zu, column %zu "
               "(%s) holds %a where st512 holds %a\n", k->R, k->name, what, pitch, count, row, col,
               (j & 1) ? "im" : "re", got[i], want[i]);
    }
    exit(1);
}

static int check_all(void)
{
    static const size_t PITCH[2] = {33, 34};
    static const size_t COUNT[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 31, 32, 33};
    const int ncount = (int)(sizeof COUNT / sizeof COUNT[0]);
    int ki, pi, ci, a, cases = 0;
    for (ki = 0; ki < NKERNELS; ki++)
    {
        const kernel_t *k = &KERNELS[ki];
        for (pi = 0; pi < 2; pi++)
        {
            const size_t P = PITCH[pi], n = 2 * (size_t)k->R * P + 2 * GUARD;
            double *seed = alloc_d(n), *ref = alloc_d(n), *got = alloc_d(n);
            for (ci = 0; ci < ncount && COUNT[ci] <= P; ci++)
            {
                const size_t c = COUNT[ci];
                size_t i, r, col;
                for (i = 0; i < n; i++)
                    seed[i] = canary();
                for (r = 0; r < (size_t)k->R; r++)
                    for (col = 0; col < c; col++)
                    {
                        seed[GUARD + 2 * (r * P + col)] = rnd();
                        seed[GUARD + 2 * (r * P + col) + 1] = rnd();
                    }
                run_case(k->arm[ST512], seed, ref, n, P, c);
                run_case(k->lib, seed, got, n, P, c);
                if (memcmp(got, ref, n * sizeof(double)))
                    mismatch(k, "the library's kernel against the st512 arm", P, c, got, ref, n);
                for (a = ST256; a <= ST128; a++)
                {
                    run_case(k->arm[a], seed, got, n, P, c);
                    if (memcmp(got, ref, n * sizeof(double)))
                        mismatch(k, a == ST256 ? "st256 against st512" : "st128 against st512", P, c, got, ref, n);
                }
                cases++;
            }
            vfft_free(seed);
            vfft_free(ref);
            vfft_free(got);
        }
    }
    printf("correctness: %d kernels, %d cases (pitch 33 and 34, counts 1..33): st256 and st128 bitwise st512, "
           "st512 bitwise the library's kernel\n", NKERNELS, cases);
    return 0;
}

/* ── THE TIMING ──────────────────────────────────────────────────────────── */

typedef struct
{
    const kernel_t *k;
    int N1, N2;
    size_t hp1;
} cell_t;

/* the leaf stage of _il2d_col_pass: blocks of R rows, all columns, in place */
static void leaf_pass(leaf_fn fn, double *z, int N1, int R, size_t pitch, size_t count)
{
    const size_t nb = (size_t)(N1 / R);
    size_t b;
    for (b = 0; b < nb; b++)
    {
        double *p = z + 2 * b * (size_t)R * pitch;
        fn(p, NULL, p, NULL, NULL, NULL, pitch, 0, pitch, 0, count);
    }
}

static double sample(leaf_fn fn, double *z, const double *seed, size_t n, const cell_t *c, size_t pitch, int reps)
{
    double t0;
    int i;
    memcpy(z, seed, n * sizeof(double));   /* the plane re-laid before every sample (_il2d_ip_reset) */
    t0 = vfft_now_ns();
    for (i = 0; i < reps; i++)
        leaf_pass(fn, z, c->N1, c->k->R, pitch, c->hp1);
    return (vfft_now_ns() - t0) / reps;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *v, int n, double *lo, double *hi)
{
    double s[1024];
    memcpy(s, v, (size_t)n * sizeof(double));
    qsort(s, (size_t)n, sizeof(double), cmp_d);
    *lo = s[0];
    *hi = s[n - 1];
    return (n & 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/* N1 = R * 2^j, the plane N1 x hp1 complex nearest `bytes` */
static int pick_n1(int R, size_t hp1, double bytes)
{
    const double ideal = bytes / (16.0 * (double)hp1 * (double)R);
    int j = ideal > 1.0 ? (int)lround(log2(ideal)) : 0;
    return R << j;
}

static void plane_size(char *buf, size_t n, double bytes)
{
    if (bytes >= 1048576.0)
        snprintf(buf, n, "%.1f MiB", bytes / 1048576.0);
    else
        snprintf(buf, n, "%.0f KiB", bytes / 1024.0);
}

int main(int argc, char **argv)
{
    static const int DEFAULT_LEAVES[] = {4, 8, 16, 3, 5, 7};
    static const int N2S[] = {64, 256, 1024, 4096};
    static const double PLANE_BYTES[] = {256.0 * 1024, 4.0 * 1048576, 64.0 * 1048576};
    int leaves[NKERNELS + 1], nleaves = 0, check_only = 0, quick = 0, rounds = 25, pace_ms = 200, core = -1;
    const char *out = NULL;
    cell_t cells[NKERNELS * 4 * 3];
    int ncells = 0, ci, i, a;
    FILE *ft = NULL, *fr = NULL;
    char scope[320], line[1024];
    time_t now = time(NULL);

    __builtin_cpu_init();   /* first: this TU carries no AVX-512 code, the arms and the library do */
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512dq") ||
        !__builtin_cpu_supports("avx512vl"))
    {
        printf("half_store_bench: this CPU has no AVX-512 (avx512f, avx512dq, avx512vl). Run it on an AVX-512 "
               "machine; Intel SDE runs the correctness check (sde -spr -- half_store_bench.exe --check)\n");
        return 2;
    }
    for (i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--check"))
            check_only = 1;
        else if (!strcmp(argv[i], "--quick"))
            quick = 1;
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc)
            rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pace-ms") && i + 1 < argc)
            pace_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--core") && i + 1 < argc)
            core = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out = argv[++i];
        else if (!strcmp(argv[i], "--leaves") && i + 1 < argc)
        {
            const char *s = argv[++i];
            if (!strcmp(s, "all"))
                for (nleaves = 0; nleaves < NKERNELS; nleaves++)
                    leaves[nleaves] = KERNELS[nleaves].R;
            else
                while (*s && nleaves < NKERNELS)
                {
                    char *e;
                    const long r = strtol(s, &e, 10);
                    if (e == s)
                        break;
                    leaves[nleaves++] = (int)r;
                    s = (*e == ',') ? e + 1 : e;
                }
        }
        else
        {
            fprintf(stderr, "usage: half_store_bench [--check] [--quick] [--leaves 4,8,16,3,5,7 | all] "
                            "[--rounds N] [--pace-ms M] [--core C] [--out DIR]\n");
            return 2;
        }
    }
    if (rounds < 1 || rounds > 1024)
    {
        fprintf(stderr, "half_store_bench: --rounds takes 1..1024\n");
        return 2;
    }

    check_all();
    if (check_only)
        return 0;

    if (quick)
    {   /* one cell, 2 rounds: proves the build and the run, measures nothing */
        leaves[0] = 16;
        nleaves = 1;
        rounds = 2;
    }
    else if (!nleaves)
        for (; nleaves < (int)(sizeof DEFAULT_LEAVES / sizeof DEFAULT_LEAVES[0]); nleaves++)
            leaves[nleaves] = DEFAULT_LEAVES[nleaves];
    for (i = 0; i < nleaves; i++)
    {
        const kernel_t *k = NULL;
        int n2, pl;
        for (a = 0; a < NKERNELS; a++)
            if (KERNELS[a].R == leaves[i])
                k = &KERNELS[a];
        if (!k)
        {
            fprintf(stderr, "half_store_bench: no radix-%d leaf here (the kernels: 2 3 4 5 6 7 8 10 12 14 16 22 26)\n",
                    leaves[i]);
            return 2;
        }
        for (n2 = 0; n2 < 4; n2++)
            for (pl = 0; pl < 3; pl++)
            {
                cell_t *c = &cells[ncells];
                if (quick && (n2 || pl))
                    continue;
                c->k = k;
                c->N2 = N2S[n2];
                c->hp1 = (size_t)c->N2 / 2 + 1;
                c->N1 = quick ? 64 : pick_n1(k->R, c->hp1, PLANE_BYTES[pl]);
                ncells++;
            }
    }

    bench_scope(core, 1, bench_guard_mode());   /* pinned, sibling held, HIGH priority, the measurement lock */
    vfft_measure_describe(scope, sizeof scope);
    if (out)
    {
        snprintf(line, sizeof line, "%s/table.txt", out);
        ft = fopen(line, "w");
        snprintf(line, sizeof line, "%s/rows.csv", out);
        fr = fopen(line, "w");
        if (!ft || !fr)
        {
            fprintf(stderr, "half_store_bench: cannot write into %s\n", out);
            return 1;
        }
        fprintf(fr, "leaf,N1,N2,pitch,count,arm,round,position,reps,ns\n");
    }
    snprintf(line, sizeof line,
             "# half_store_bench %s: the AVX-512 forward column leaf, its plane stores as one 512-bit store (st512,\n"
             "#   as shipped), two 256-bit halves (st256), four 128-bit quarters (st128)\n"
             "# host: %s; build %s, gcc %s\n",
             strtok(ctime(&now), "\n"), vfft_wisdom_identity(), vfft_wisdom_build(), __VERSION__);
    printf("%s", line);   /* the scope's own line is already out (bench_scope) */
    if (ft)
        fprintf(ft, "%s# measurement scope: %s\n", line, scope);
    snprintf(line, sizeof line,
             "# a time: one leaf stage of the 2D real r2c column pass (N1/R calls over all hp1 columns, in place),\n"
             "#   us, median [min..max] of %d rounds; arms alternate within a round, %d ms idle between rounds\n"
             "# st512/st256/st128 at the odd pitch hp1 = N2/2 + 1; st512_even: the control, st512 at pitch hp1 + 1\n"
             "# ratios of medians: st256/st512, st128/st512 (below 1.00: the split arm is faster);\n"
             "#   even/odd = st512_even/st512 (what the odd pitch costs the shipped leaf)\n"
             "%-5s %-14s %5s %9s %5s  %-25s %-25s %-25s %-25s %7s %7s %7s\n",
             rounds, pace_ms, "leaf", "N1 x N2", "pitch", "plane", "reps",
             "st512", "st256", "st128", "st512_even", "256/512", "128/512", "even/odd");
    printf("%s", line);
    if (ft)
        fputs(line, ft);
    fflush(stdout);

    for (ci = 0; ci < ncells; ci++)
    {
        const cell_t *c = &cells[ci];
        const int R = c->k->R;
        const size_t pitch[NARMS] = {c->hp1, c->hp1, c->hp1, c->hp1 + 1};
        const size_t n_odd = 2 * (size_t)c->N1 * c->hp1, n_even = 2 * (size_t)c->N1 * (c->hp1 + 1);
        double *z = alloc_d(n_odd), *seed = alloc_d(n_odd), *ze = alloc_d(n_even), *seede = alloc_d(n_even);
        leaf_fn fn[NARMS];
        double ns[NARMS][1024], med[NARMS], lo[NARMS], hi[NARMS], est, t0, bytes;
        int reps, cap, r, p;
        char ps[32], cell[32];
        size_t q;
        for (q = 0; q < n_odd; q++)
            seed[q] = 1.0 + 1e-6 * (double)(q & 1023);   /* the race's seed (_il2d_race_forms) */
        for (q = 0; q < n_even; q++)
            seede[q] = 1.0 + 1e-6 * (double)(q & 1023);
        fn[ST512] = c->k->arm[0];
        fn[ST256] = c->k->arm[1];
        fn[ST128] = c->k->arm[2];
        fn[ST512_EVEN] = c->k->arm[0];
        /* warm every arm once, then size the sample on st512 (the incumbent, as the race sizes it) */
        for (a = 0; a < NARMS; a++)
            sample(fn[a], a == ST512_EVEN ? ze : z, a == ST512_EVEN ? seede : seed,
                   a == ST512_EVEN ? n_even : n_odd, c, pitch[a], 1);
        memcpy(z, seed, n_odd * sizeof(double));
        t0 = vfft_now_ns();
        leaf_pass(fn[ST512], z, c->N1, R, c->hp1, c->hp1);
        est = vfft_now_ns() - t0;
        cap = (int)(290.0 / log10((double)R));   /* R^reps under 1e290 from a plane seeded near 1 */
        reps = (int)ceil(1.0e6 / (est > 1.0 ? est : 1.0));
        if (reps > cap)
            reps = cap;
        if (reps < 1)
            reps = 1;
        for (r = 0; r < rounds; r++)
        {
            pace(pace_ms);   /* every round after the same idle, the first included */
            for (p = 0; p < NARMS; p++)
            {
                a = (r & 1) ? NARMS - 1 - p : p;   /* forward, then backward: no arm always first */
                ns[a][r] = sample(fn[a], a == ST512_EVEN ? ze : z, a == ST512_EVEN ? seede : seed,
                                  a == ST512_EVEN ? n_even : n_odd, c, pitch[a], reps);
                if (fr)
                    fprintf(fr, "%d,%d,%d,%zu,%zu,%s,%d,%d,%d,%.1f\n", R, c->N1, c->N2, pitch[a], c->hp1,
                            ARM[a], r, p, reps, ns[a][r]);
            }
        }
        for (a = 0; a < NARMS; a++)
            med[a] = median(ns[a], rounds, &lo[a], &hi[a]);
        bytes = (double)n_odd * sizeof(double);
        plane_size(ps, sizeof ps, bytes);
        snprintf(cell, sizeof cell, "%d x %d", c->N1, c->N2);
        {
            char col[NARMS][40];
            for (a = 0; a < NARMS; a++)
                snprintf(col[a], sizeof col[a], "%.2f [%.2f..%.2f]", med[a] / 1e3, lo[a] / 1e3, hi[a] / 1e3);
            snprintf(line, sizeof line, "r%-4d %-14s %5zu %9s %5d  %-25s %-25s %-25s %-25s %7.3f %7.3f %7.3f\n",
                     R, cell, c->hp1, ps, reps, col[ST512], col[ST256], col[ST128], col[ST512_EVEN],
                     med[ST256] / med[ST512], med[ST128] / med[ST512], med[ST512_EVEN] / med[ST512]);
        }
        printf("%s", line);
        fflush(stdout);
        if (ft)
            fputs(line, ft);
        if (fr)
            fflush(fr);
        vfft_free(z);
        vfft_free(seed);
        vfft_free(ze);
        vfft_free(seede);
    }
    if (ft)
        fclose(ft);
    if (fr)
        fclose(fr);
    printf("half_store_bench: %d cells, %d rounds%s\n", ncells, rounds, out ? "" : " (no --out: nothing written)");
    return 0;
}

#endif /* HALF_STORE_ARM */
