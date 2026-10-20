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
 * half_store_bench.sh copies each kernel's source into the bench's object
 * folder with its plane stores -- the 512-bit stores into zout in the wide
 * loop -- spelled HALF_STORE_PLANE, and proves kernel by kernel that these are,
 * one for one by destination, the stores the AVX2 half twin splits, and that
 * the 512-bit stores it leaves alone (into the stack scratch S[]) are, one for
 * one, the ones the twin keeps whole; any difference stops the build. Compiled
 * with -DHALF_STORE_ARM=<width> -DHALF_STORE_KERNEL="<the copy>"
 * -D<symbol>=half_store_st<width>_<symbol>, this file is one arm: the copy,
 * its plane stores at the arm's width, under its own name.
 *   st512  as shipped: one 512-bit store
 *   st256  two 256-bit halves (vmovupd ymm, vextractf64x4 to memory)
 *   st128  four 128-bit quarters (vmovupd xmm, vextractf64x2 to memory), the
 *          AVX2 twin's store width
 * The S[] stores stay whole, as the AVX2 twin keeps its S[] stores whole, and
 * the tail (the masked k-pass, the ymm and xmm ladder) keeps its stores, as
 * the AVX2 twin keeps its VEX-128 tail. Compiled plainly, this file is the
 * bench; half_store_bench.sh builds both (gauntlet/build.py loaded as a
 * module, as src/tools/gates/twin_gate_build.py does), the arms at the codelet
 * library's flags, and reads each arm's machine code back for its store widths.
 *
 * THE KERNELS (HALF_STORE_KERNELS): the 32 forward column leaves with an AVX2
 * half twin, n1c at 29 radices from 2 to 47 and the blocked n1cb48 and n1cb84
 * (radix 32) and n1cb88 (radix 64). In the odd leaves from radix 9 up and in
 * the blocked leaves the wide loop also stores half-DFT spills into S[]. At
 * radix 3, 5, 6, 10 and 12 the AVX2 half twin carries an older arithmetic body
 * than its full twin, and at the odd radices from 23 up its body runs in
 * another order than the AVX-512 one; the stores it splits are still, by
 * destination, the AVX-512 kernel's plane stores, which is what the arms follow.
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
 * pitch, in place, repeated to about 1 ms on st512. The plane is re-laid from
 * the race's seed before the sample, as the race does, and again, untimed,
 * every so many passes that the values stay finite (in place they grow by up
 * to R a pass). A cell is a real 2D r2c shape N1 x N2 whose column leaf is
 * radix R (N1 a multiple of R). Its arms, all timed in every round:
 *   st512, st256, st128  at the odd pitch hp1
 *   st512 again          st512 a second time at the odd pitch: an arm against
 *                        itself, the noise of a ratio
 *   st512 even           the CONTROL: st512 over the same hp1 columns at the
 *                        even pitch hp1 rounded up to a multiple of 4 complex
 *                        (64 bytes, so every row starts on a cache line and no
 *                        plane store splits one; at hp1 + 1 every other row
 *                        would still start 32 bytes into a line)
 * The default cells, for the leaves r2, r4, r8, r16, r32b48, r32b84, r64b88,
 * r5 and r9:
 *   - the 2D real r2c cells where the 14900KF's forms race picked the
 *     half-store leaf (BANKED), at their own N1 x N2;
 *   - the plane-size sweep: N2 = 64, 512, 4096 (hp1 = 33, 257, 2049) and N1 =
 *     R times the power of two that puts the plane near 256 KiB (inside L2),
 *     4 MiB (in L3) and 64 MiB (past it).
 *
 * PROTOCOL (the one-thread protocol, bench_scope.h): the library's
 * measurement scope -- pinned to one physical core, its SMT sibling held, the
 * process at HIGH priority, the machine's measurement lock -- on Intel and on
 * AMD alike; the arms ALTERNATE within a round (forward, then backward the
 * next round); 25 rounds; 200 ms idle between rounds, each idle followed by
 * untimed samples of every arm, 5 ms at least, before the round's first timed
 * sample (a core runs slow for the first milliseconds after a sleep:
 * VFFT_RACE_PACE_MS, race.h). Reported per cell and arm: the median and the
 * spread (min..max) of the rounds, and the ratio of each arm's median to
 * st512's.
 *
 * Usage: half_store_bench [--check] [--quick] [--leaves r16,32,... | all]
 *                         [--rounds N] [--pace-ms M] [--core C] [--out DIR]
 *   --check   the correctness check alone (no timing): what Intel SDE can show
 *   --quick   one cell (r16, 64 x 64) and 2 rounds: the build and the run
 *             proven end to end, nothing measured
 *   --leaves  the leaves timed, by name (r32b48) or by radix (32: both radix-32
 *             leaves); each with its BANKED cells and the sweep
 *   --out     write DIR/table.md (the table) and DIR/rows.csv (every sample)
 * Exit 2 on a CPU without AVX-512 (checked before any AVX-512 instruction).
 * Build and run: ./gauntlet/half_store_bench.sh on an AVX-512 machine. Under
 * Intel SDE (sde -spr -- ./gauntlet/half_store_bench.exe --check) the check is
 * exact and the timings mean nothing.
 */

/* radix, the leaf's name, the kernel's file under codelets/zil/avx512/, its
 * symbol, the AVX2 half twin's file under codelets/zil/avx2/ (read by
 * half_store_bench.sh for the proof) */
#define HALF_STORE_KERNELS(X)                                                                                           \
    X(2, "r2", "shared/col/radix2_z_n1c_avx512.c", radix2_z_n1c_fwd_avx512, "shared/col/half/radix2_z_n1ch_avx2.c")     \
    X(3, "r3", "shared/col/radix3_z_n1c_avx512.c", radix3_z_n1c_fwd_avx512, "shared/col/half/radix3_z_n1ch_avx2.c")     \
    X(4, "r4", "shared/col/radix4_z_n1c_avx512.c", radix4_z_n1c_fwd_avx512, "shared/col/half/radix4_z_n1ch_avx2.c")     \
    X(5, "r5", "shared/col/radix5_z_n1c_avx512.c", radix5_z_n1c_fwd_avx512, "shared/col/half/radix5_z_n1ch_avx2.c")     \
    X(6, "r6", "shared/col/radix6_z_n1c_avx512.c", radix6_z_n1c_fwd_avx512, "shared/col/half/radix6_z_n1ch_avx2.c")     \
    X(7, "r7", "shared/col/radix7_z_n1c_avx512.c", radix7_z_n1c_fwd_avx512, "shared/col/half/radix7_z_n1ch_avx2.c")     \
    X(8, "r8", "shared/col/radix8_z_n1c_avx512.c", radix8_z_n1c_fwd_avx512, "shared/col/half/radix8_z_n1ch_avx2.c")     \
    X(9, "r9", "shared/col/radix9_z_n1c_avx512.c", radix9_z_n1c_fwd_avx512, "shared/col/half/radix9_z_n1ch_avx2.c")     \
    X(10, "r10", "shared/col/radix10_z_n1c_avx512.c", radix10_z_n1c_fwd_avx512, "shared/col/half/radix10_z_n1ch_avx2.c") \
    X(11, "r11", "shared/col/radix11_z_n1c_avx512.c", radix11_z_n1c_fwd_avx512, "shared/col/half/radix11_z_n1ch_avx2.c") \
    X(12, "r12", "shared/col/radix12_z_n1c_avx512.c", radix12_z_n1c_fwd_avx512, "shared/col/half/radix12_z_n1ch_avx2.c") \
    X(13, "r13", "shared/col/radix13_z_n1c_avx512.c", radix13_z_n1c_fwd_avx512, "shared/col/half/radix13_z_n1ch_avx2.c") \
    X(14, "r14", "shared/col/radix14_z_n1c_avx512.c", radix14_z_n1c_fwd_avx512, "shared/col/half/radix14_z_n1ch_avx2.c") \
    X(15, "r15", "shared/col/radix15_z_n1c_avx512.c", radix15_z_n1c_fwd_avx512, "shared/col/half/radix15_z_n1ch_avx2.c") \
    X(16, "r16", "shared/col/radix16_z_n1c_avx512.c", radix16_z_n1c_fwd_avx512, "shared/col/half/radix16_z_n1ch_avx2.c") \
    X(17, "r17", "shared/col/radix17_z_n1c_avx512.c", radix17_z_n1c_fwd_avx512, "shared/col/half/radix17_z_n1ch_avx2.c") \
    X(19, "r19", "shared/col/radix19_z_n1c_avx512.c", radix19_z_n1c_fwd_avx512, "shared/col/half/radix19_z_n1ch_avx2.c") \
    X(21, "r21", "shared/col/radix21_z_n1c_avx512.c", radix21_z_n1c_fwd_avx512, "shared/col/half/radix21_z_n1ch_avx2.c") \
    X(22, "r22", "shared/col/radix22_z_n1c_avx512.c", radix22_z_n1c_fwd_avx512, "shared/col/half/radix22_z_n1ch_avx2.c") \
    X(23, "r23", "shared/col/radix23_z_n1c_avx512.c", radix23_z_n1c_fwd_avx512, "shared/col/half/radix23_z_n1ch_avx2.c") \
    X(25, "r25", "shared/col/radix25_z_n1c_avx512.c", radix25_z_n1c_fwd_avx512, "shared/col/half/radix25_z_n1ch_avx2.c") \
    X(26, "r26", "shared/col/radix26_z_n1c_avx512.c", radix26_z_n1c_fwd_avx512, "shared/col/half/radix26_z_n1ch_avx2.c") \
    X(27, "r27", "shared/col/radix27_z_n1c_avx512.c", radix27_z_n1c_fwd_avx512, "shared/col/half/radix27_z_n1ch_avx2.c") \
    X(29, "r29", "shared/col/radix29_z_n1c_avx512.c", radix29_z_n1c_fwd_avx512, "shared/col/half/radix29_z_n1ch_avx2.c") \
    X(31, "r31", "shared/col/radix31_z_n1c_avx512.c", radix31_z_n1c_fwd_avx512, "shared/col/half/radix31_z_n1ch_avx2.c") \
    X(32, "r32b48", "shared/col/blocked/radix32_z_n1cb48_avx512.c", radix32_z_n1cb48_fwd_avx512,                        \
      "shared/col/half/radix32_z_n1cb48h_avx2.c")                                                                       \
    X(32, "r32b84", "shared/col/blocked/radix32_z_n1cb84_avx512.c", radix32_z_n1cb84_fwd_avx512,                        \
      "shared/col/half/radix32_z_n1cb84h_avx2.c")                                                                       \
    X(37, "r37", "shared/col/radix37_z_n1c_avx512.c", radix37_z_n1c_fwd_avx512, "shared/col/half/radix37_z_n1ch_avx2.c") \
    X(41, "r41", "shared/col/radix41_z_n1c_avx512.c", radix41_z_n1c_fwd_avx512, "shared/col/half/radix41_z_n1ch_avx2.c") \
    X(43, "r43", "shared/col/radix43_z_n1c_avx512.c", radix43_z_n1c_fwd_avx512, "shared/col/half/radix43_z_n1ch_avx2.c") \
    X(47, "r47", "shared/col/radix47_z_n1c_avx512.c", radix47_z_n1c_fwd_avx512, "shared/col/half/radix47_z_n1ch_avx2.c") \
    X(64, "r64b88", "shared/col/blocked/radix64_z_n1cb88_avx512.c", radix64_z_n1cb88_fwd_avx512,                        \
      "shared/col/half/radix64_z_n1cb88h_avx2.c")

#ifdef HALF_STORE_ARM
/* ── ONE ARM: the kernel's copy, its plane stores at the arm's width ────────── */
#include <immintrin.h>
#include <stddef.h>
#if HALF_STORE_ARM == 512
#define HALF_STORE_PLANE(p, v) _mm512_storeu_pd((p), (v))
#elif HALF_STORE_ARM == 256
#define HALF_STORE_PLANE(p, v)                                         \
    do                                                                 \
    {                                                                  \
        double *const hs_p_ = (p);                                     \
        const __m512d hs_v_ = (v);                                     \
        _mm256_storeu_pd(hs_p_, _mm512_castpd512_pd256(hs_v_));        \
        _mm256_storeu_pd(hs_p_ + 4, _mm512_extractf64x4_pd(hs_v_, 1)); \
    } while (0)
#elif HALF_STORE_ARM == 128
#define HALF_STORE_PLANE(p, v)                                      \
    do                                                              \
    {                                                               \
        double *const hs_p_ = (p);                                  \
        const __m512d hs_v_ = (v);                                  \
        _mm_storeu_pd(hs_p_, _mm512_castpd512_pd128(hs_v_));        \
        _mm_storeu_pd(hs_p_ + 2, _mm512_extractf64x2_pd(hs_v_, 1)); \
        _mm_storeu_pd(hs_p_ + 4, _mm512_extractf64x2_pd(hs_v_, 2)); \
        _mm_storeu_pd(hs_p_ + 6, _mm512_extractf64x2_pd(hs_v_, 3)); \
    } while (0)
#else
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

#define DECLARE(R, name, file, sym, twin)          \
    extern void sym(LEAF_ARGS);                    \
    extern void half_store_st512_##sym(LEAF_ARGS); \
    extern void half_store_st256_##sym(LEAF_ARGS); \
    extern void half_store_st128_##sym(LEAF_ARGS);
HALF_STORE_KERNELS(DECLARE)

typedef struct
{
    int R;
    const char *name, *file, *sym;
    leaf_fn lib;      /* the library's own kernel (the codelet library) */
    leaf_fn arm[3];   /* st512, st256, st128 */
} kernel_t;
#define ROW(R, name, file, sym, twin) \
    {R, name, file, #sym, sym, {half_store_st512_##sym, half_store_st256_##sym, half_store_st128_##sym}},
static const kernel_t KERNELS[] = {HALF_STORE_KERNELS(ROW)};
#define NKERNELS ((int)(sizeof KERNELS / sizeof KERNELS[0]))

/* the default leaves: the radices whose half-store leaf the 14900KF's race
 * picked (2, 4, 16, 32, 64), radix 8, and two odd leaves (5; 9, whose wide
 * loop also stores into S[]) */
static const char *const DEFAULT_LEAVES[] = {"r2", "r4", "r8", "r16", "r32b48", "r32b84", "r64b88", "r5", "r9"};

/* the 2D real r2c cells where the 14900KF's forms race picked the half-store
 * leaf (src/wisdom/14900KF/wisdom2_2d.txt: the rows whose forms= ends in h,
 * b84h or b88h), with the leaf it picked. Not here: the five rows with
 * hp1 = N2/2 + 1 = 3 -- 4 x 4 (radix 4), 128 x 4 and 256 x 4 (r32b48),
 * 2048 x 4 and 4096 x 4 (r64b88) -- where the AVX-512 leaf runs its masked
 * 3-column pass alone: no 512-bit plane store, the three arms the same code. */
static const struct
{
    const char *leaf;
    int N1, N2;
} BANKED[] = {
    {"r2", 2, 16},      {"r2", 2, 32},      {"r2", 2, 128},     {"r2", 2, 1024},    {"r2", 2, 2048},
    {"r4", 4, 16},      {"r4", 4, 4096},
    {"r16", 16, 1024},  {"r16", 16, 2048},  {"r16", 16, 4096},  {"r16", 16, 8192},  {"r16", 512, 4096},
    {"r16", 512, 8192},
    {"r32b84", 32, 16}, {"r32b84", 32, 512}, {"r32b84", 128, 8},
    {"r64b88", 64, 32}, {"r64b88", 64, 128}, {"r64b88", 64, 512}, {"r64b88", 64, 1024},
};
#define NBANKED ((int)(sizeof BANKED / sizeof BANKED[0]))

enum { ST512, ST256, ST128, ST512_AGAIN, ST512_EVEN, NARMS };
static const char *const ARM[NARMS] = {"st512", "st256", "st128", "st512_again", "st512_even"};

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
        printf("half_store_bench: MISMATCH %s (%s): %s at pitch %zu, count %zu: guard double %zu "
               "was written (%a)\n", k->name, k->sym, what, pitch, count, i, got[i]);
    else
    {
        const size_t j = i - GUARD, row = j / (2 * pitch), col = (j % (2 * pitch)) / 2;
        printf("half_store_bench: MISMATCH %s (%s): %s at pitch %zu, count %zu: row %zu, column %zu "
               "(%s) holds %a where st512 holds %a\n", k->name, k->sym, what, pitch, count, row, col,
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
    int N1, N2, banked;
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

/* one sample, ns per leaf stage: reps stages, timed; the plane re-laid from
 * its seed (_il2d_ip_reset) before every `chunk` of them, untimed */
static double sample(leaf_fn fn, double *z, const double *seed, size_t n, const cell_t *c, size_t pitch, int reps,
                     int chunk)
{
    double ns = 0.0, t0;
    int done = 0, i, m;
    while (done < reps)
    {
        m = reps - done < chunk ? reps - done : chunk;
        memcpy(z, seed, n * sizeof(double));
        t0 = vfft_now_ns();
        for (i = 0; i < m; i++)
            leaf_pass(fn, z, c->N1, c->k->R, pitch, c->hp1);
        ns += vfft_now_ns() - t0;
        done += m;
    }
    return ns / reps;
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
    else if (bytes >= 1024.0)
        snprintf(buf, n, "%.0f KiB", bytes / 1024.0);
    else
        snprintf(buf, n, "%.0f B", bytes);
}

/* ns, four digits that matter, no exponent */
static void ns4(char *buf, size_t n, double v)
{
    snprintf(buf, n, v >= 1000.0 ? "%.0f" : v >= 100.0 ? "%.1f" : v >= 10.0 ? "%.2f" : "%.3f", v);
}

static int add_cell(cell_t *cells, int ncells, const kernel_t *k, int N1, int N2, int banked)
{
    int i;
    for (i = 0; i < ncells; i++)
        if (cells[i].k == k && cells[i].N1 == N1 && cells[i].N2 == N2)
            return ncells;   /* a sweep cell that is a banked one is timed once, with the banked */
    cells[ncells].k = k;
    cells[ncells].N1 = N1;
    cells[ncells].N2 = N2;
    cells[ncells].hp1 = (size_t)N2 / 2 + 1;
    cells[ncells].banked = banked;
    return ncells + 1;
}

/* `s` is a leaf's name (r32b48), a radix (32: every leaf of that radix) or all */
static int select_leaves(const char *s, int *sel)
{
    int ki, any = 0;
    if (!strcmp(s, "all"))
    {
        for (ki = 0; ki < NKERNELS; ki++)
            sel[ki] = 1;
        return 1;
    }
    for (ki = 0; ki < NKERNELS; ki++)
        if (!strcmp(s, KERNELS[ki].name) || (s[0] >= '0' && s[0] <= '9' && atoi(s) == KERNELS[ki].R))
            sel[ki] = any = 1;
    return any;
}

int main(int argc, char **argv)
{
    static const int N2S[] = {64, 512, 4096};
    static const double PLANE_BYTES[] = {256.0 * 1024, 4.0 * 1048576, 64.0 * 1048576};
    static cell_t cells[NKERNELS * 9 + NBANKED];
    int sel[NKERNELS] = {0}, nsel = 0, check_only = 0, quick = 0, rounds = 25, pace_ms = 200, core = -1;
    const char *out = NULL;
    int ncells = 0, ci, i, a, ki;
    FILE *ft = NULL, *fr = NULL;
    char scope[320], line[2048];
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
            char list[256], *s;
            snprintf(list, sizeof list, "%s", argv[++i]);
            for (s = strtok(list, ","); s; s = strtok(NULL, ","))
                if (!select_leaves(s, sel))
                {
                    fprintf(stderr, "half_store_bench: no leaf %s here (the leaves:", s);
                    for (ki = 0; ki < NKERNELS; ki++)
                        fprintf(stderr, " %s", KERNELS[ki].name);
                    fprintf(stderr, ")\n");
                    return 2;
                }
            nsel = 1;
        }
        else
        {
            fprintf(stderr, "usage: half_store_bench [--check] [--quick] [--leaves r16,32,... | all] "
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
        for (ki = 0; KERNELS[ki].R != 16; ki++)
            ;
        ncells = add_cell(cells, 0, &KERNELS[ki], 64, 64, 0);
        rounds = 2;
    }
    else
    {
        if (!nsel)
            for (i = 0; i < (int)(sizeof DEFAULT_LEAVES / sizeof DEFAULT_LEAVES[0]); i++)
                select_leaves(DEFAULT_LEAVES[i], sel);
        for (i = 0; i < NBANKED; i++)
            for (ki = 0; ki < NKERNELS; ki++)
                if (sel[ki] && !strcmp(KERNELS[ki].name, BANKED[i].leaf))
                    ncells = add_cell(cells, ncells, &KERNELS[ki], BANKED[i].N1, BANKED[i].N2, 1);
        for (ki = 0; ki < NKERNELS; ki++)
        {
            int n2, pl;
            if (!sel[ki])
                continue;
            for (n2 = 0; n2 < (int)(sizeof N2S / sizeof N2S[0]); n2++)
                for (pl = 0; pl < (int)(sizeof PLANE_BYTES / sizeof PLANE_BYTES[0]); pl++)
                    ncells = add_cell(cells, ncells, &KERNELS[ki],
                                      pick_n1(KERNELS[ki].R, (size_t)N2S[n2] / 2 + 1, PLANE_BYTES[pl]), N2S[n2], 0);
        }
    }

    bench_scope(core, 1, bench_guard_mode());   /* pinned, sibling held, HIGH priority, the measurement lock */
    vfft_measure_describe(scope, sizeof scope);
    if (out)
    {
        snprintf(line, sizeof line, "%s/table.md", out);
        ft = fopen(line, "w");
        snprintf(line, sizeof line, "%s/rows.csv", out);
        fr = fopen(line, "w");
        if (!ft || !fr)
        {
            fprintf(stderr, "half_store_bench: cannot write into %s\n", out);
            return 1;
        }
        fprintf(ft, "```text\n");
        fprintf(fr, "leaf,R,N1,N2,pitch,count,arm,round,position,reps,ns\n");
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
             "# a time: ns for one leaf stage of the 2D real r2c column pass (N1/R calls over all hp1 columns, in\n"
             "#   place), median [min..max] of %d rounds; the arms alternate within a round; before every round\n"
             "#   %d ms idle, then 5 ms at least of untimed samples\n"
             "# st512, st256, st128 and st512 again (st512 timed a second time) at the odd pitch hp1 = N2/2 + 1;\n"
             "#   st512 even: the control, st512 at the even pitch hp1 rounded up to a multiple of 4 complex\n"
             "#   (64 bytes: every row starts on a cache line)\n"
             "# ratios of medians to st512's: 256/512, 128/512 (below 1.000: the split arm is faster);\n"
             "#   again/512 (st512 against itself: the noise of a ratio); even/512 (what the odd pitch costs\n"
             "#   the shipped leaf)\n"
             "%-7s %-12s %5s %9s %8s  %-28s %-28s %-28s %-28s %-28s %7s %7s %9s %8s\n",
             rounds, pace_ms, "leaf", "N1 x N2", "pitch", "plane", "reps", "st512", "st256", "st128",
             "st512 again", "st512 even", "256/512", "128/512", "again/512", "even/512");
    printf("%s", line);
    if (ft)
        fputs(line, ft);
    fflush(stdout);

    for (ci = 0; ci < ncells; ci++)
    {
        const cell_t *c = &cells[ci];
        const int R = c->k->R;
        const int chunk = (int)(290.0 / log10((double)R));   /* stages between re-lays: R^chunk under 1e290 */
        const size_t pe = (c->hp1 + 3) & ~(size_t)3;          /* the control's pitch: a multiple of 64 bytes */
        const size_t pitch[NARMS] = {c->hp1, c->hp1, c->hp1, c->hp1, pe};
        const size_t n_odd = 2 * (size_t)c->N1 * c->hp1, n_even = 2 * (size_t)c->N1 * pe;
        double *z = alloc_d(n_odd), *seed = alloc_d(n_odd), *ze = alloc_d(n_even), *seede = alloc_d(n_even);
        double *zb[NARMS], *sb[NARMS];
        size_t nb[NARMS];
        leaf_fn fn[NARMS];
        double ns[NARMS][1024], med[NARMS], lo[NARMS], hi[NARMS], est = 0.0, t0;
        int reps, r, p, m;
        char ps[32], cell[32], col[NARMS][64], v[3][24];
        size_t q;
        if (!quick && (ci == 0 || c->banked != cells[ci - 1].banked))
        {
            snprintf(line, sizeof line, c->banked
                     ? "# the cells where the 14900KF's 2D r2c forms race picked the half-store leaf\n"
                     : "# the plane-size sweep: N1 x N2 near 256 KiB (inside L2), 4 MiB (in L3), 64 MiB (past it)\n");
            printf("%s", line);
            if (ft)
                fputs(line, ft);
        }
        for (q = 0; q < n_odd; q++)
            seed[q] = 1.0 + 1e-6 * (double)(q & 1023);   /* the race's seed (_il2d_race_forms) */
        for (q = 0; q < n_even; q++)
            seede[q] = 1.0 + 1e-6 * (double)(q & 1023);
        fn[ST512] = fn[ST512_AGAIN] = fn[ST512_EVEN] = c->k->arm[0];
        fn[ST256] = c->k->arm[1];
        fn[ST128] = c->k->arm[2];
        for (a = 0; a < NARMS; a++)
        {
            zb[a] = a == ST512_EVEN ? ze : z;
            sb[a] = a == ST512_EVEN ? seede : seed;
            nb[a] = a == ST512_EVEN ? n_even : n_odd;
        }
        /* warm every arm once, then size the sample on st512 (the incumbent, as the race sizes it) */
        for (a = 0; a < NARMS; a++)
            sample(fn[a], zb[a], sb[a], nb[a], c, pitch[a], 1, chunk);
        for (m = 1; m <= (1 << 22); m *= 2)
        {   /* a stage's time from at least 0.2 ms of stages */
            est = sample(fn[ST512], z, seed, n_odd, c, c->hp1, m, chunk);
            if (est * m >= 2.0e5)
                break;
        }
        est = ceil(1.0e6 / (est > 0.5 ? est : 0.5));   /* the stages in 1 ms */
        reps = est > (double)(1 << 24) ? 1 << 24 : est < 1.0 ? 1 : (int)est;
        for (r = 0; r < rounds; r++)
        {
            pace(pace_ms);   /* every round after the same idle, the first included */
            t0 = vfft_now_ns();
            do   /* the core wakes slow from the idle: untimed samples, in the round's order, 5 ms at least */
                for (p = 0; p < NARMS; p++)
                {
                    a = (r & 1) ? NARMS - 1 - p : p;
                    sample(fn[a], zb[a], sb[a], nb[a], c, pitch[a], reps, chunk);
                }
            while (vfft_now_ns() - t0 < 5.0e6);
            for (p = 0; p < NARMS; p++)
            {
                a = (r & 1) ? NARMS - 1 - p : p;   /* forward, then backward: no arm always first */
                ns[a][r] = sample(fn[a], zb[a], sb[a], nb[a], c, pitch[a], reps, chunk);
                if (fr)
                    fprintf(fr, "%s,%d,%d,%d,%zu,%zu,%s,%d,%d,%d,%.3f\n", c->k->name, R, c->N1, c->N2, pitch[a],
                            c->hp1, ARM[a], r, p, reps, ns[a][r]);
            }
        }
        for (a = 0; a < NARMS; a++)
        {
            med[a] = median(ns[a], rounds, &lo[a], &hi[a]);
            ns4(v[0], sizeof v[0], med[a]);
            ns4(v[1], sizeof v[1], lo[a]);
            ns4(v[2], sizeof v[2], hi[a]);
            snprintf(col[a], sizeof col[a], "%s [%s..%s]", v[0], v[1], v[2]);
        }
        plane_size(ps, sizeof ps, (double)n_odd * sizeof(double));
        snprintf(cell, sizeof cell, "%d x %d", c->N1, c->N2);
        snprintf(line, sizeof line, "%-7s %-12s %5zu %9s %8d  %-28s %-28s %-28s %-28s %-28s %7.3f %7.3f %9.3f %8.3f\n",
                 c->k->name, cell, c->hp1, ps, reps, col[ST512], col[ST256], col[ST128], col[ST512_AGAIN],
                 col[ST512_EVEN], med[ST256] / med[ST512], med[ST128] / med[ST512], med[ST512_AGAIN] / med[ST512],
                 med[ST512_EVEN] / med[ST512]);
        printf("%s", line);
        fflush(stdout);
        if (ft)
        {
            fputs(line, ft);
            fflush(ft);
        }
        if (fr)
            fflush(fr);
        vfft_free(z);
        vfft_free(seed);
        vfft_free(ze);
        vfft_free(seede);
    }
    if (ft)
    {
        fprintf(ft, "```\n");
        fclose(ft);
    }
    if (fr)
        fclose(fr);
    printf("half_store_bench: %d cells, %d rounds%s\n", ncells, rounds, out ? "" : " (no --out: nothing written)");
    return 0;
}

#endif /* HALF_STORE_ARM */
