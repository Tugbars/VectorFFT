/* bench_ingest_diag.c — s0 ingest STORE-ADDRESSING attribution probe
 * (Phase 0b follow-up; docs/roadmap/cascade_load_path_restructure.md).
 *
 * THE 2x2 KERNEL MATRIX (register lattice x store pattern). All four arms
 * read the SAME interleaved zin through the SAME 4 leg streams (stride N/4
 * complex, each leg cursor advancing 64 B/iter):
 *
 *                        SECTIONED stores             LINEAR stores
 *                        (4 cursors N/4 cx apart,     (1 cursor, 256 B/iter =
 *                         64 B/iter each, RATE-        4x the per-cursor rate,
 *                         MATCHED to the leg loads)    sweeps mod 4KB vs loads)
 *   deint-only lattice   a. s0s      (incumbent,      d. s0s_lin (s0s's 16-op
 *   (16 p5 / 16 cx)         real production object)      network, stores
 *                                                        rewritten linear)
 *   fused-turn lattice   c. s0t_sect (18-op turn      b. s0t     (challenger,
 *   (18 p5 / 16 cx)         lattice, MKL's OBSERVED      spike_s0t_ingest.c
 *                           section geometry)            body VERBATIM)
 *
 * READOUT: charge follows the STORE column (b,d slow / a,c fast at N>=4096)
 *            => store pattern guilty, turn lattice innocent;
 *          charge follows the LATTICE row (b,c slow / a,d fast)
 *            => Phase 0's op census missed something in the turn lattice;
 *          only b slow => the specific rho plane layout;
 *          nothing reproduces in cold mode => hot-loop regime artifact.
 *
 * PLANE LAYOUTS (c and d compute DIFFERENT plane layouts from a and b — fine
 * for cost attribution; each arm therefore has its OWN scalar index-map
 * reference, smoke-gated below):
 *   a. s0s      radix-4 output row r at section r*Ls; per column group q:
 *               [4 re][4 im] at doubles 2*r*Ls + 8q  (output digit = section)
 *   b. s0t      ZTURN rho: record of butterfly position p at doubles 8p,
 *               [re Y0..Y3][im Y0..Y3] (output digit innermost = lanes,
 *               granule index = p, natural order, ONE linear cursor)
 *   c. s0t_sect same record interior as b, but record p -> section
 *               bitrev2(p mod 4) = {0->0, 1->2, 2->1, 3->3} at byte offsets
 *               {0, 4N, 8N, 12N}, granule index in section = p div 4
 *               == MKL's ingest store geometry (disasm ground report:
 *               fn 0x1825c3800, stores r13 / +rbp*2 / +rbp*4 / +r15*2,
 *               shared +0x40/iter cursor)
 *   d. s0s_lin  s0s's records on ONE linear cursor: column group q at
 *               doubles 32q, row r at 32q + 8r (+4 im)
 *
 * TWO REGIMES:
 *   hot  — Phase 0b reproduction: ONE slot, inner-repeat hot loop sized to a
 *          ~1.2 ms region from an 8-pass incumbent estimate, identical inner
 *          for all arms; zin at +0 mod 4KB, out at +16N+skew (default +64).
 *   cold — one-shot regime per the regime ground report ("dram" spec): ring
 *          of R slots, footprint >= 80 MB (> 2x the 36 MB L3), slot_stride =
 *          roundup(16N + 16N + 8192, 4096) so EVERY slot base is 0 mod 4KB
 *          (Phase 0b plane geometry reproduced in every slot: zin_k = slot+0,
 *          out_k = slot + 16N + skew); no prime — the 64 MB cachebust before
 *          each sample leaves the walked slots cold; timed region = K
 *          distinct-slot one-shot passes under ONE QPC pair, K =
 *          ceil(150000 ns / est_cold_ns), floor 32, cap R (=> region >=
 *          ~150 us, timer overhead < 0.05%, per-pass ring-advance from a
 *          precomputed pointer table); est_cold calibrated ONCE per cell with
 *          the incumbent on ring slots after a bust, cursor then reset; ring
 *          cursor RESET to slot 0 at the start of EVERY region so all arms
 *          walk the same slots in the same order (same pages, same DTLB).
 *
 * DISCIPLINE (each item = a Phase 0b scar):
 *   pin logical core 2 (SetProcessAffinityMask AND SetThreadAffinityMask,
 *   mask 4, FATAL exit(2) if refused), HIGH_PRIORITY_CLASS, FTZ/DAZ, NO
 *   sibling spinner; 64 MB cachebust FIRST (it is work, it heats) THEN
 *   Sleep(pace_ms) THEN the timed region; rotation: rep r executes arms
 *   ((r+j) % 5), reps FORCED UP to a multiple of 5; arm 4 = s0s again
 *   (CONTROL, band 0.98-1.02 evaluated on MEDIANS — the Phase 0b logs prove
 *   best-of gates spuriously); every per-rep sample printed for offline
 *   median post-processing; MEDIAN is the primary statistic (min/max also
 *   printed); Sleep(5*pace) between regimes and between cells.
 *
 * USAGE: bench_ingest_diag [N|all] [hot|cold|both] [reps] [pace_ms]
 *                          [--smoke] [--skew=S]
 *   defaults:               all     both          15     1000     skew=64
 *   S = out-plane offset mod 4KB: multiple of 64, 0..4096. skew=0 overlays
 *   the rate-matched arms' stores exactly on their loads mod 4KB (the
 *   H-alias crater probe); 2048 is the far point.
 *
 * SMOKE (always runs before timing; --smoke = run only this): each arm vs
 * its own scalar reference at N=2048 and 4096, <= 1e-13 + all outputs
 * finite, PASS/FAIL per arm. Any failure => exit(1), no timing.
 *
 * BUILD (mandated gcc 15.2, from build_tuned/benches):
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma -c bench_ingest_diag.c
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma -c
 *     ..\..\src\dag-fft-compiler\codelets\zil\avx2\radix4_z_s0s_avx2.c
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma <both .o>
 *     -o bench_ingest_diag.exe
 *
 * EXIT: 0 ok, 1 smoke fail, 2 fatal, 3 completed but >=1 control VOID.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

/* arm b: the Phase-0-proven challenger body + its scalar reference,
 * included VERBATIM (its self-test main renamed away). Also provides the
 * pinned _M_IM mask reused by s0t_sect_fwd below (same TU). */
#define main spike_s0t_ingest_main
#include "spike_s0t_ingest.c"
#undef main

/* arm a: the REAL production incumbent, linked as its own object. */
extern void radix4_z_s0s_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);

/* ====================================================================== */
/* arm c: s0t_sect — s0t's 18-op fused-turn lattice VERBATIM, stores      */
/* rewritten to MKL's observed section geometry: 4 cursors at byte        */
/* offsets {0, 4N, 8N, 12N}, each advancing 64 B/iter (rate-matched to    */
/* the leg loads); record->section = bitrev2(position mod 4):            */
/* {k->sec0, k+1->sec2, k+2->sec1, k+3->sec3}.                            */
/* ====================================================================== */
__attribute__((noinline, target("avx2,fma")))
void s0t_sect_fwd(const double * __restrict zin, double * __restrict sp,
                  size_t N)
{
    const size_t Ls = N / 4;
    double * __restrict s0 = sp;                /* byte 0        (r13)     */
    double * __restrict s1 = sp + (N >> 1);     /* byte 4N  (r13+rbp*2)    */
    double * __restrict s2 = sp + N;            /* byte 8N  (r13+rbp*4)    */
    double * __restrict s3 = sp + N + (N >> 1); /* byte 12N (r13+r15*2)    */
    for (size_t k = 0; k + 4 <= Ls; k += 4) {
        /* ---- half A: positions k, k+1 (lattice VERBATIM from s0t) ---- */
        const __m256d a0 = _mm256_loadu_pd(&zin[2*(0*Ls + k)]);
        const __m256d a1 = _mm256_loadu_pd(&zin[2*(1*Ls + k)]);
        const __m256d a2 = _mm256_loadu_pd(&zin[2*(2*Ls + k)]);
        const __m256d a3 = _mm256_loadu_pd(&zin[2*(3*Ls + k)]);
        const __m256d at0 = _mm256_add_pd(a0, a2);
        const __m256d at1 = _mm256_sub_pd(a0, a2);
        const __m256d at2 = _mm256_add_pd(a1, a3);
        const __m256d at3 = _mm256_sub_pd(a1, a3);
        const __m256d ar  = _mm256_xor_pd(_mm256_permute_pd(at3, 0x5), _M_IM);
        const __m256d aY0 = _mm256_add_pd(at0, at2);
        const __m256d aY2 = _mm256_sub_pd(at0, at2);
        const __m256d aY1 = _mm256_add_pd(at1, ar);
        const __m256d aY3 = _mm256_sub_pd(at1, ar);
        const __m256d au0 = _mm256_unpacklo_pd(aY0, aY1);
        const __m256d au1 = _mm256_unpackhi_pd(aY0, aY1);
        const __m256d au2 = _mm256_unpacklo_pd(aY2, aY3);
        const __m256d au3 = _mm256_unpackhi_pd(aY2, aY3);
        _mm256_storeu_pd(&s0[2*k + 0], _mm256_permute2f128_pd(au0, au2, 0x20)); /* re(k)   -> sec0 */
        _mm256_storeu_pd(&s0[2*k + 4], _mm256_permute2f128_pd(au1, au3, 0x20)); /* im(k)   -> sec0 */
        _mm256_storeu_pd(&s2[2*k + 0], _mm256_permute2f128_pd(au0, au2, 0x31)); /* re(k+1) -> sec2 */
        _mm256_storeu_pd(&s2[2*k + 4], _mm256_permute2f128_pd(au1, au3, 0x31)); /* im(k+1) -> sec2 */
        /* ---- half B: positions k+2, k+3 ---- */
        const __m256d b0 = _mm256_loadu_pd(&zin[2*(0*Ls + k) + 4]);
        const __m256d b1 = _mm256_loadu_pd(&zin[2*(1*Ls + k) + 4]);
        const __m256d b2 = _mm256_loadu_pd(&zin[2*(2*Ls + k) + 4]);
        const __m256d b3 = _mm256_loadu_pd(&zin[2*(3*Ls + k) + 4]);
        const __m256d bt0 = _mm256_add_pd(b0, b2);
        const __m256d bt1 = _mm256_sub_pd(b0, b2);
        const __m256d bt2 = _mm256_add_pd(b1, b3);
        const __m256d bt3 = _mm256_sub_pd(b1, b3);
        const __m256d br  = _mm256_xor_pd(_mm256_permute_pd(bt3, 0x5), _M_IM);
        const __m256d bY0 = _mm256_add_pd(bt0, bt2);
        const __m256d bY2 = _mm256_sub_pd(bt0, bt2);
        const __m256d bY1 = _mm256_add_pd(bt1, br);
        const __m256d bY3 = _mm256_sub_pd(bt1, br);
        const __m256d bu0 = _mm256_unpacklo_pd(bY0, bY1);
        const __m256d bu1 = _mm256_unpackhi_pd(bY0, bY1);
        const __m256d bu2 = _mm256_unpacklo_pd(bY2, bY3);
        const __m256d bu3 = _mm256_unpackhi_pd(bY2, bY3);
        _mm256_storeu_pd(&s1[2*k + 0], _mm256_permute2f128_pd(bu0, bu2, 0x20)); /* re(k+2) -> sec1 */
        _mm256_storeu_pd(&s1[2*k + 4], _mm256_permute2f128_pd(bu1, bu3, 0x20)); /* im(k+2) -> sec1 */
        _mm256_storeu_pd(&s3[2*k + 0], _mm256_permute2f128_pd(bu0, bu2, 0x31)); /* re(k+3) -> sec3 */
        _mm256_storeu_pd(&s3[2*k + 4], _mm256_permute2f128_pd(bu1, bu3, 0x31)); /* im(k+3) -> sec3 */
    }
}

/* ====================================================================== */
/* arm d: s0s_lin — s0s's 16-op deint lattice VERBATIM (load edge + SU    */
/* body copied from radix4_z_s0s_avx2.c), the 8 sectioned stores          */
/* rewritten to ONE linear cursor advancing 256 B/iter (s0t's store rate).*/
/* ====================================================================== */
__attribute__((noinline, target("avx2,fma")))
void s0s_lin_fwd(const double * __restrict zin, double * __restrict zout,
                 size_t N)
{
    const size_t Ls = N / 4;
    for (size_t k = 0; k + 4 <= Ls; k += 4) {
        /* Z load edge (DEINT) — verbatim */
        const __m256d _zl_0 = _mm256_loadu_pd(&zin[2*(size_t)k]);
        const __m256d _zh_0 = _mm256_loadu_pd(&zin[2*(size_t)k + 4]);
        const __m256d lane_re_0 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_0, _zh_0), 0xD8);
        const __m256d lane_im_0 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_0, _zh_0), 0xD8);
        const __m256d _zl_1 = _mm256_loadu_pd(&zin[2*((size_t)1*Ls + k)]);
        const __m256d _zh_1 = _mm256_loadu_pd(&zin[2*((size_t)1*Ls + k) + 4]);
        const __m256d lane_re_1 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_1, _zh_1), 0xD8);
        const __m256d lane_im_1 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_1, _zh_1), 0xD8);
        const __m256d _zl_2 = _mm256_loadu_pd(&zin[2*((size_t)2*Ls + k)]);
        const __m256d _zh_2 = _mm256_loadu_pd(&zin[2*((size_t)2*Ls + k) + 4]);
        const __m256d lane_re_2 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_2, _zh_2), 0xD8);
        const __m256d lane_im_2 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_2, _zh_2), 0xD8);
        const __m256d _zl_3 = _mm256_loadu_pd(&zin[2*((size_t)3*Ls + k)]);
        const __m256d _zh_3 = _mm256_loadu_pd(&zin[2*((size_t)3*Ls + k) + 4]);
        const __m256d lane_re_3 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_3, _zh_3), 0xD8);
        const __m256d lane_im_3 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_3, _zh_3), 0xD8);
        /* SU-scheduled body — verbatim */
        const __m256d t0 = lane_re_3;
        const __m256d t2 = lane_im_3;
        const __m256d t5 = lane_re_1;
        const __m256d t10 = _mm256_sub_pd(t5, t0);
        const __m256d t24 = _mm256_add_pd(t0, t5);
        const __m256d t6 = lane_im_1;
        const __m256d t8 = _mm256_sub_pd(t6, t2);
        const __m256d t23 = _mm256_add_pd(t2, t6);
        const __m256d t13 = lane_re_2;
        const __m256d t14 = lane_im_2;
        const __m256d t16 = lane_re_0;
        const __m256d t20 = _mm256_sub_pd(t16, t13);
        const __m256d t21 = _mm256_sub_pd(t20, t8);
        const __m256d t31 = _mm256_add_pd(t8, t20);
        const __m256d t27 = _mm256_add_pd(t13, t16);
        const __m256d t28 = _mm256_sub_pd(t27, t24);
        const __m256d t33 = _mm256_add_pd(t24, t27);
        const __m256d t17 = lane_im_0;
        const __m256d t18 = _mm256_sub_pd(t17, t14);
        const __m256d t22 = _mm256_add_pd(t10, t18);
        const __m256d t32 = _mm256_sub_pd(t18, t10);
        const __m256d t26 = _mm256_add_pd(t14, t17);
        const __m256d t30 = _mm256_sub_pd(t26, t23);
        const __m256d t34 = _mm256_add_pd(t23, t26);
        /* LINEAR store cursor — s0s's records, one 256-B run per iter    */
        _mm256_storeu_pd(&zout[8*k +  0], t33);   /* row0 re, cols k..k+3 */
        _mm256_storeu_pd(&zout[8*k +  4], t34);   /* row0 im              */
        _mm256_storeu_pd(&zout[8*k +  8], t31);   /* row1 re              */
        _mm256_storeu_pd(&zout[8*k + 12], t32);   /* row1 im              */
        _mm256_storeu_pd(&zout[8*k + 16], t28);   /* row2 re              */
        _mm256_storeu_pd(&zout[8*k + 20], t30);   /* row2 im              */
        _mm256_storeu_pd(&zout[8*k + 24], t21);   /* row3 re              */
        _mm256_storeu_pd(&zout[8*k + 28], t22);   /* row3 im              */
    }
}

/* ================== scalar index-map references ======================= */
/* Shared forward radix-4 butterfly (identical math to s0t_ref; the s0s   */
/* SU body decodes to the same rows: t33/t34=Y0, t31/t32=Y1, t28/t30=Y2,  */
/* t21/t22=Y3). Only the index maps differ between references.           */
static void bfly4_at(const double *zin, size_t Ls, size_t p,
                     double *Yr, double *Yi)
{
    double xr[4], xi[4];
    for (int l = 0; l < 4; l++) {
        xr[l] = zin[2*((size_t)l*Ls + p)];
        xi[l] = zin[2*((size_t)l*Ls + p) + 1];
    }
    const double t0r = xr[0]+xr[2], t0i = xi[0]+xi[2];
    const double t1r = xr[0]-xr[2], t1i = xi[0]-xi[2];
    const double t2r = xr[1]+xr[3], t2i = xi[1]+xi[3];
    const double t3r = xr[1]-xr[3], t3i = xi[1]-xi[3];
    Yr[0] = t0r + t2r;  Yi[0] = t0i + t2i;
    Yr[2] = t0r - t2r;  Yi[2] = t0i - t2i;
    Yr[1] = t1r + t3i;  Yi[1] = t1i - t3r;   /* Y1 = t1 - i*t3 (fwd) */
    Yr[3] = t1r - t3i;  Yi[3] = t1i + t3r;   /* Y3 = t1 + i*t3       */
}

/* a. s0s: row r at section r*Ls; column group q: [4re][4im] at 2*r*Ls+8q */
static void s0s_ref(const double *zin, double *out, size_t N)
{
    const size_t Ls = N / 4;
    for (size_t p = 0; p < Ls; p++) {
        double Yr[4], Yi[4];
        bfly4_at(zin, Ls, p, Yr, Yi);
        const size_t q = p >> 2, j = p & 3;
        for (int r = 0; r < 4; r++) {
            out[2*(size_t)r*Ls + 8*q + j]     = Yr[r];
            out[2*(size_t)r*Ls + 8*q + 4 + j] = Yi[r];
        }
    }
}

/* b. s0t: uses the spike's own s0t_ref (included above) — rho = 8p + a0. */

/* c. s0t_sect: record p -> section bitrev2(p mod 4), granule p div 4,    */
/*    record interior identical to b (output digit in lanes).            */
static void s0t_sect_ref(const double *zin, double *out, size_t N)
{
    static const size_t secmap[4] = { 0, 2, 1, 3 };   /* bitrev2 */
    const size_t Ls = N / 4;
    for (size_t p = 0; p < Ls; p++) {
        double Yr[4], Yi[4];
        bfly4_at(zin, Ls, p, Yr, Yi);
        const size_t base = secmap[p & 3] * (N >> 1) + 8*(p >> 2);
        for (int a = 0; a < 4; a++) {
            out[base + a]     = Yr[a];
            out[base + 4 + a] = Yi[a];
        }
    }
}

/* d. s0s_lin: column group q at 32q; row r at 32q + 8r (+4 im).          */
static void s0s_lin_ref(const double *zin, double *out, size_t N)
{
    const size_t Ls = N / 4;
    for (size_t p = 0; p < Ls; p++) {
        double Yr[4], Yi[4];
        bfly4_at(zin, Ls, p, Yr, Yi);
        const size_t q = p >> 2, j = p & 3;
        for (int r = 0; r < 4; r++) {
            out[32*q + 8*(size_t)r + j]     = Yr[r];
            out[32*q + 8*(size_t)r + 4 + j] = Yi[r];
        }
    }
}

/* ========================== arm dispatch ============================== */
typedef void (*arm_fn)(const double *zin, double *out, size_t N);
typedef void (*ref_fn)(const double *zin, double *out, size_t N);

static void arm_s0s(const double *zin, double *out, size_t N)
{
    radix4_z_s0s_fwd_avx2(zin, 0, out, 0, 0, 0,
                          N/4, 0, N/4, 0, N/4);
}
static void arm_s0t(const double *zin, double *out, size_t N)
{ s0t_fwd(zin, out, N); }
static void arm_s0t_sect(const double *zin, double *out, size_t N)
{ s0t_sect_fwd(zin, out, N); }
static void arm_s0s_lin(const double *zin, double *out, size_t N)
{ s0s_lin_fwd(zin, out, N); }

#define NARMS 5   /* 0=s0s(inc) 1=s0t 2=s0t_sect 3=s0s_lin 4=s0s(CONTROL) */
static const arm_fn g_arm[NARMS] =
    { arm_s0s, arm_s0t, arm_s0t_sect, arm_s0s_lin, arm_s0s };
static const char *g_nm[NARMS] =
    { "s0s", "s0t", "s0t_sect", "s0s_lin", "ctl" };

/* ============================ utilities =============================== */
static double now_ns(void)
{
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

static volatile double g_sink = 0.0;
static char *g_bust = NULL;
static void cachebust(void)                       /* 64 MB > 36 MB L3 */
{
    const size_t n = 64u << 20; double s = 0.0;
    for (size_t i = 0; i < n; i += 64) { g_bust[i] = (char)(i ^ 0x5a); s += g_bust[i]; }
    g_sink += s;
}

static int all_finite(const double *p, size_t n)
{
    for (size_t i = 0; i < n; i++) if (!isfinite(p[i])) return 0;
    return 1;
}

static void fill_zin(double *zin, size_t N)       /* deterministic spike fill */
{
    for (size_t i = 0; i < N; i++) {
        zin[2*i]     = sin(0.7 * (double)i) + (double)i * 1e-3;
        zin[2*i + 1] = cos(1.3 * (double)i);
    }
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void arm_stats(const double *smp, int reps, int arm,
                      double *med, double *mn, double *mx)
{
    double v[256];                    /* reps capped well below this */
    for (int r = 0; r < reps; r++) v[r] = smp[r * NARMS + arm];
    qsort(v, (size_t)reps, sizeof(double), cmp_d);
    *mn = v[0]; *mx = v[reps - 1];
    *med = (reps & 1) ? v[reps / 2]
                      : 0.5 * (v[reps / 2 - 1] + v[reps / 2]);
}

/* ============================== ring ================================== */
/* slot_stride multiple of 4KB => every slot base is 0 mod 4KB; zin at    */
/* +0, out at +16N+skew => out mod 4KB == skew for every slot (16N is a   */
/* 4KB multiple at all timed N).                                          */
typedef struct { const double *zin; double *out; } slot_t;
typedef struct { int R; size_t stride; char *arena; slot_t *tab; } ring_t;

static size_t slot_stride(size_t N)
{
    return (32*N + 8192 + 4095) & ~(size_t)4095;
}
static int cold_slots(size_t N)
{
    const size_t need = (size_t)80 << 20;         /* >= 80 MB > 2x L3 */
    return (int)((need + slot_stride(N) - 1) / slot_stride(N));
}

static ring_t *make_ring(size_t N, int R, size_t skew)
{
    ring_t *rg = (ring_t *)malloc(sizeof(ring_t));
    if (!rg) { fprintf(stderr, "FATAL: ring alloc\n"); exit(2); }
    rg->R = R;
    rg->stride = slot_stride(N);
    rg->arena = (char *)_aligned_malloc((size_t)R * rg->stride, 4096);
    rg->tab = (slot_t *)malloc((size_t)R * sizeof(slot_t));
    if (!rg->arena || !rg->tab) { fprintf(stderr, "FATAL: arena alloc\n"); exit(2); }
    const size_t plane = 16 * N;                  /* bytes */
    for (int i = 0; i < R; i++) {
        rg->tab[i].zin = (const double *)(rg->arena + (size_t)i * rg->stride);
        rg->tab[i].out = (double *)(rg->arena + (size_t)i * rg->stride
                                    + plane + skew);
    }
    fill_zin((double *)rg->tab[0].zin, N);        /* slot 0 computed ...  */
    for (int i = 1; i < R; i++)                   /* ... others copied    */
        memcpy((void *)rg->tab[i].zin, rg->tab[0].zin, plane);
    for (int i = 0; i < R; i++)                   /* commit out pages     */
        memset(rg->tab[i].out, 0, plane);
    return rg;
}
static void free_ring(ring_t *rg)
{
    _aligned_free(rg->arena); free(rg->tab); free(rg);
}

/* ============================== smoke ================================= */
static int smoke_one(const char *nm, arm_fn f, ref_fn ref,
                     const double *zin, double *out, double *rbuf, size_t N)
{
    const size_t nd = 2 * N;
    for (size_t i = 0; i < nd; i++) { out[i] = 123.25; rbuf[i] = -987.5; }
    f(zin, out, N);
    ref(zin, rbuf, N);
    double mx = 0.0;
    for (size_t i = 0; i < nd; i++) {
        double d = fabs(out[i] - rbuf[i]); if (d > mx) mx = d;
    }
    int ok = (mx <= 1e-13) && all_finite(out, nd) && all_finite(rbuf, nd);
    printf("    %-9s vs its scalar ref: maxabs=%.3e  %s\n",
           nm, mx, ok ? "PASS" : "**FAIL**");
    return ok;
}

static int smoke_all(void)
{
    static const size_t Ns[2] = { 2048, 4096 };
    static const ref_fn refs[4] = { s0s_ref, s0t_ref, s0t_sect_ref, s0s_lin_ref };
    int ok = 1;
    for (int t = 0; t < 2; t++) {
        const size_t N = Ns[t], nd = 2 * N;
        double *zin = (double *)_mm_malloc(nd * sizeof(double), 4096);
        double *out = (double *)_mm_malloc(nd * sizeof(double), 4096);
        double *rb  = (double *)_mm_malloc(nd * sizeof(double), 4096);
        if (!zin || !out || !rb) { fprintf(stderr, "FATAL: smoke alloc\n"); exit(2); }
        fill_zin(zin, N);
        printf("  SMOKE N=%zu (4 arms, each vs its own index-map reference)\n", N);
        for (int a = 0; a < 4; a++)
            ok &= smoke_one(g_nm[a], g_arm[a], refs[a], zin, out, rb, N);
        _mm_free(zin); _mm_free(out); _mm_free(rb);
    }
    printf("  SMOKE %s\n\n", ok ? "PASS" : "**FAIL**");
    return ok;
}

/* ============================ timed cell ============================== */
static int run_regime(size_t N, ring_t *rg, int hot, int reps, int pace_ms,
                      size_t skew)
{
    long inner = 1; int K = 1; double est;

    if (hot) {                    /* inner sized ONCE from the incumbent  */
        const slot_t *s = &rg->tab[0];
        g_arm[0](s->zin, s->out, N);          /* warm x3 */
        g_arm[0](s->zin, s->out, N);
        g_arm[0](s->zin, s->out, N);
        double t0 = now_ns();
        for (int i = 0; i < 8; i++) g_arm[0](s->zin, s->out, N);
        est = (now_ns() - t0) / 8.0;
        if (est < 1.0) est = 1.0;
        inner = (long)(1.2e6 / est) + 1;
        if (inner < 1) inner = 1;
        if (inner > 2000000) inner = 2000000;
    } else {                      /* est_cold: incumbent on cold slots    */
        cachebust();
        double t0 = now_ns();
        for (int i = 0; i < 8; i++)
            g_arm[0](rg->tab[i].zin, rg->tab[i].out, N);
        est = (now_ns() - t0) / 8.0;
        if (est < 1.0) est = 1.0;
        K = (int)(150000.0 / est) + 1;        /* region >= ~150 us */
        if (K < 32) K = 32;
        if (K > rg->R) K = rg->R;
    }

    printf("  [%s] N=%-6zu skew=%zu arms: 0=s0s(inc) 1=s0t 2=s0t_sect "
           "3=s0s_lin 4=s0s(CONTROL)  %s=%ld  est=%.0fns\n",
           hot ? "hot " : "cold", N, skew,
           hot ? "inner" : "K", hot ? inner : (long)K, est);
    fflush(stdout);

    double *smp = (double *)malloc((size_t)reps * NARMS * sizeof(double));
    if (!smp) { fprintf(stderr, "FATAL: smp alloc\n"); exit(2); }

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j < NARMS; j++) {
            const int a = (j + r) % NARMS;    /* rotation */
            arm_fn f = g_arm[a];
            cachebust();                      /* bust FIRST (it heats) */
            Sleep((DWORD)pace_ms);            /* THEN cool             */
            double t;
            if (hot) {
                const slot_t *s = &rg->tab[0];
                double t0 = now_ns();
                for (long i = 0; i < inner; i++) f(s->zin, s->out, N);
                t = (now_ns() - t0) / (double)inner;
            } else {                          /* cursor RESET to slot 0 */
                const slot_t *tab = rg->tab;
                double t0 = now_ns();
                for (int i = 0; i < K; i++) f(tab[i].zin, tab[i].out, N);
                t = (now_ns() - t0) / (double)K;
            }
            smp[r * NARMS + a] = t;
        }
    }
    g_sink += rg->tab[0].out[0];

    for (int r = 0; r < reps; r++)
        printf("    rep %2d: s0s=%9.1f s0t=%9.1f sect=%9.1f lin=%9.1f "
               "ctl=%9.1f\n", r,
               smp[r*NARMS+0], smp[r*NARMS+1], smp[r*NARMS+2],
               smp[r*NARMS+3], smp[r*NARMS+4]);

    double med[NARMS], mn[NARMS], mx[NARMS];
    for (int a = 0; a < NARMS; a++)
        arm_stats(smp, reps, a, &med[a], &mn[a], &mx[a]);

    const double r_s0t  = med[1] / med[0];
    const double r_sect = med[2] / med[0];
    const double r_lin  = med[3] / med[0];
    const double r_ctl  = med[4] / med[0];
    const int ctl_ok = (r_ctl >= 0.98 && r_ctl <= 1.02);

    printf("  med: s0s=%.1f s0t=%.1f sect=%.1f lin=%.1f ctl=%.1f\n",
           med[0], med[1], med[2], med[3], med[4]);
    printf("  min: s0s=%.1f s0t=%.1f sect=%.1f lin=%.1f ctl=%.1f\n",
           mn[0], mn[1], mn[2], mn[3], mn[4]);
    printf("  max: s0s=%.1f s0t=%.1f sect=%.1f lin=%.1f ctl=%.1f\n",
           mx[0], mx[1], mx[2], mx[3], mx[4]);
    printf("SUMMARY N=%zu regime=%s skew=%zu %s=%ld est=%.0f "
           "med_s0s=%.1f med_s0t=%.1f med_sect=%.1f med_lin=%.1f "
           "med_ctl=%.1f r_s0t=%.4f r_sect=%.4f r_lin=%.4f r_ctl=%.4f "
           "ctl=%s\n\n",
           N, hot ? "hot" : "cold", skew,
           hot ? "inner" : "K", hot ? inner : (long)K, est,
           med[0], med[1], med[2], med[3], med[4],
           r_s0t, r_sect, r_lin, r_ctl,
           ctl_ok ? "PASS" : "VOID");
    fflush(stdout);
    free(smp);
    return ctl_ok;
}

/* ================================ main ================================ */
int main(int argc, char **argv)
{
    size_t cells_all[4] = { 2048, 4096, 8192, 16384 };
    size_t cells[8]; int ncells = 0;
    char mode = '2';              /* 'h' hot, 'c' cold, '2' both */
    int reps = 15, pace_ms = 1000, smoke_only = 0, pos = 0;
    size_t skew = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--smoke") == 0) { smoke_only = 1; continue; }
        if (strncmp(argv[i], "--skew=", 7) == 0) {
            skew = (size_t)atoi(argv[i] + 7);
            if (skew > 4096 || (skew & 63)) {
                fprintf(stderr, "FATAL: --skew must be a multiple of 64, "
                        "0..4096\n");
                return 2;
            }
            continue;
        }
        switch (pos++) {
        case 0:
            if (strcmp(argv[i], "all") == 0) break;
            cells[ncells++] = (size_t)atoi(argv[i]); break;
        case 1:
            if      (!strcmp(argv[i], "hot"))  mode = 'h';
            else if (!strcmp(argv[i], "cold")) mode = 'c';
            else                               mode = '2';
            break;
        case 2: reps = atoi(argv[i]); break;
        case 3: pace_ms = atoi(argv[i]); break;
        }
    }
    if (ncells == 0) { memcpy(cells, cells_all, sizeof(cells_all)); ncells = 4; }
    if (reps < NARMS) reps = NARMS;
    if (reps % NARMS) reps += NARMS - reps % NARMS;   /* multiple of 5 */
    if (reps > 250) reps = 250;                       /* arm_stats cap */
    for (int i = 0; i < ncells; i++)
        if (cells[i] < 1024 || (cells[i] & (cells[i] - 1))) {
            fprintf(stderr, "FATAL: N must be a power of two >= 1024\n");
            return 2;
        }

    /* pin logical core 2, both process and thread; FATAL on failure */
    if (!SetProcessAffinityMask(GetCurrentProcess(), 4)) {
        fprintf(stderr, "FATAL: SetProcessAffinityMask(4) failed\n"); return 2;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), 4)) {
        fprintf(stderr, "FATAL: SetThreadAffinityMask(4) failed\n"); return 2;
    }
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
        fprintf(stderr, "WARN: HIGH_PRIORITY_CLASS not set\n");
    _mm_setcsr(_mm_getcsr() | 0x8040);                /* FTZ|DAZ */

    g_bust = (char *)malloc(64u << 20);
    if (!g_bust) { fprintf(stderr, "FATAL: bust alloc\n"); return 2; }
    memset(g_bust, 1, 64u << 20);

    printf("bench_ingest_diag: mode=%s reps=%d (x5 arms) pace=%dms "
           "cellpace=%dms skew=%zu %s\n",
           mode == '2' ? "both" : (mode == 'h' ? "hot" : "cold"),
           reps, pace_ms, 5 * pace_ms, skew,
           smoke_only ? "[SMOKE ONLY]" : "");
    printf("arms: a=s0s(16p5,sect) b=s0t(18p5,lin) c=s0t_sect(18p5,sect) "
           "d=s0s_lin(16p5,lin) + ctl=s0s\n\n");

    if (!smoke_all()) return 1;
    if (smoke_only) return 0;

    int any_void = 0;
    for (int ci = 0; ci < ncells; ci++) {
        const size_t N = cells[ci];
        printf("== N=%zu ==\n", N);
        if (mode != 'c') {
            ring_t *rg = make_ring(N, 1, skew);
            if (!run_regime(N, rg, 1, reps, pace_ms, skew)) any_void = 1;
            free_ring(rg);
            if (mode == '2') Sleep((DWORD)(5 * pace_ms));
        }
        if (mode != 'h') {
            ring_t *rg = make_ring(N, cold_slots(N), skew);
            if (!run_regime(N, rg, 0, reps, pace_ms, skew)) any_void = 1;
            free_ring(rg);
        }
        if (ci + 1 < ncells) Sleep((DWORD)(5 * pace_ms));
    }
    (void)g_sink;
    if (any_void) {
        printf("NOTE: at least one control out of band -- those cells are "
               "VOID; re-run with more pacing/reps.\n");
        return 3;
    }
    return 0;
}
