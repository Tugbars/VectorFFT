/* bench_zturn_final.c — THE DECISIVE MEASUREMENT (docs/roadmap/
 * cascade_load_path_restructure.md §6, Phase 4): production zsplit vs the
 * ZTURN-S full-cascade prototype, WHOLE TRANSFORM, front door, both
 * directions, one binary containing both routes.
 *
 * ARMS (3):
 *   0 = vfft_zsplit_execute_{fwd,bwd}  (production incumbent,
 *       src/core/oop/zsplit.h, production t2q default per cell)
 *   1 = vfft_zturn_execute_{fwd,bwd}   (ZTURN-S challenger,
 *       src/core/oop/zturn_proto.h — gate-proven memcmp-EXACT vs production
 *       under the Gamma transpose; build_tuned/benches/zturn_proto_gate.c)
 *   2 = vfft_zsplit_execute_{fwd,bwd}  (CONTROL — the same incumbent code
 *       in the rotated slot; band 0.98–1.02 ON MEDIANS or the cell is VOID)
 * FORWARD and BACKWARD are SEPARATE timed passes with separate SUMMARY
 * lines. Same chain per cell for both engines: vfft_zturn_create consumes
 * vfft_zsplit_default_chain internally; the bench ASSERTS at create that
 * the two plans' chains are identical (FATAL exit 2 otherwise).
 *
 * REGIMES (both implemented exactly as bench_ingest_diag.c):
 *   hot  — ONE slot; inner-repeat loop sized ONCE from an 8-pass incumbent
 *          estimate to a ~1.2 ms region; IDENTICAL inner for all arms.
 *   cold — one-shot ring: R slots, footprint >= 80 MB (> 2x the 36 MB L3),
 *          slot_stride = roundup(16N + 16N + 8192, 4096) so EVERY slot base
 *          is 0 mod 4KB (identical plane geometry in every slot: zin_k =
 *          slot+0, out_k = slot+16N+64); timed region = K distinct-slot
 *          one-shot passes under ONE QPC pair, K = ceil(150 us / est_cold),
 *          floor 32, cap R (region >= ~150 us, timer overhead < 0.05%);
 *          est_cold calibrated ONCE per (cell,dir) with the incumbent on
 *          ring slots after a bust; ring cursor RESET to slot 0 at the
 *          start of EVERY region so every arm walks the same slots in the
 *          same order (same pages, same DTLB).
 *
 * BWD INPUT: every slot's zin plane holds the LEGACY comb (production fwd
 * of the natural fill), and arm 1 reads the SAME plane as a ZTURN comb.
 * The two combs differ by the pure per-row (N/32 x 4) transpose Gamma, so
 * the plane is a valid, all-finite comb under either interpretation, and
 * the address walk is IDENTICAL for all arms — ONE shared input plane and
 * ONE shared output plane per slot (per-arm output buffers once fabricated
 * a bogus 1.51x). Value correctness of both bwd engines is gated in SMOKE.
 *
 * ENGINE SCRATCH: both plans' internal 2N-double scratch planes (zsplit
 * p->sp, zturn p->plane) are RE-POINTED into ONE 4KB-aligned cell arena at
 * deterministic mod-4KB offsets — incumbent at +128, challenger at +192;
 * slot zin sits at 0 and slot out at +64 mod 4KB — so no two live streams
 * share a mod-4KB class by heap accident (the 4KB-aliasing trap, hit twice;
 * §4.9995(c) measured 27% from allocation alignment alone). The original
 * pointers are restored before destroy.
 *
 * DISCIPLINE (every item = a paid-for scar; bench_phase0b/bench_ingest_diag):
 *   pin process AND thread to mask 4 (logical core 2), FATAL exit(2) if
 *   either refuses; HIGH_PRIORITY_CLASS; FTZ/DAZ; NO sibling spinner;
 *   64 MB cachebust FIRST (it is work, it heats) THEN Sleep(pace_ms) THEN
 *   the timed region; rotation: rep r executes arms (r%3,(r+1)%3,(r+2)%3),
 *   reps FORCED UP to a multiple of 3; every per-rep sample printed (check
 *   bimodality offline); MEDIANS are the primary statistic (min/max also
 *   printed); control band 0.98–1.02 evaluated ON MEDIANS (the Phase 0b
 *   logs prove best-of gates spuriously); Sleep(5*pace) between passes,
 *   regimes and cells.
 *
 * SMOKE GATES (every cell, EVERY LAUNCH, before any timing; --smoke = run
 * only these — a bench that times a wrong transform is worse than no bench):
 *   G-fwd: zturn fwd == production fwd under the gate transpose Gamma
 *          out_z[l*(N/8)+4k'+j] == out_legacy[l*(N/8)+j*(N/32)+k'],
 *          EXACT (memcmp per double — the zturn_proto_gate.c Gamma logic);
 *   G-bwd: zturn bwd(zturn comb) == production bwd(legacy comb), EXACT;
 *   all four outputs finite. Any failure: the cell is NOT timed, exit 1.
 *
 * SUMMARY (machine-parsable, one line per (N, regime, dir)):
 *   SUMMARY N=<n> regime=hot|cold dir=fwd|bwd med_inc=<ns> med_chal=<ns>
 *   med_ctl=<ns> r_chal=<r> r_ctl=<r> ctl=PASS|VOID
 *
 * USAGE: bench_zturn_final [N|all] [hot|cold|both] [reps] [pace_ms] [--smoke]
 *   defaults:                all     both          15      1000
 * EXIT: 0 ok, 1 smoke fail, 2 fatal, 3 completed but >=1 control VOID
 *   (identical semantics to bench_phase0b / bench_ingest_diag).
 *
 * BUILD (gcc 15.2 MANDATED for all timed objects — gcc 13 folds leg loads;
 * from the repo root, KDIR = src/dag-fft-compiler/codelets/zil/avx2):
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma
 *     -c KDIR/radix{4,8}_z_{s0s,msg}[_bwd]_avx2.c
 *        KDIR/radix8_z_sterm{,2,_bwd}_avx2.c            (11 kernel objects)
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma -I src/core/oop
 *     -c build_tuned/benches/bench_zturn_final.c
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma bench_zturn_final.o
 *     <11 kernel .o> -o build_tuned/benches/bench_zturn_final.exe -lm
 * Both engines link the SAME production kernel objects (verify: nm shows
 * exactly one T per radix*_z_* symbol in the exe).
 *
 * B1 SEAM (-DZT_STFB_SINK; the zturn_proto_gate.c ZTURN_GEN_KERNELS
 * precedent): swaps ONLY the challenger's backward TERMINATOR for the SUNK
 * generated kernel radix8_z_stf_r4sk_bwd_avx2 (--zp-sink store-at-def
 * interleave; bit-identical output, spill census 33 -> 6). The rest of the
 * bwd path (msg mids, s0tb ingest) and the whole fwd path stay the
 * prototype route. Link KDIR/radix8_z_stf_r4sk_bwd_avx2.o in that build.
 * Baseline build (no define) is macro-inert: ZT_EXEC_BWD is
 * vfft_zturn_execute_bwd, behavior byte-stable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include "zsplit.h"        /* production plan + kernels (src/core/oop) */
#include "zturn_proto.h"   /* ZTURN-S prototype (same production msg objects) */

/* ================= B1 SEAM: sunk bwd terminator substitution ============
 * -DZT_STFB_SINK swaps the challenger's bwd terminator for the generated
 * SUNK kernel (see file header). The executor below is vfft_zturn_execute_bwd
 * (zturn_proto.h) verbatim except the stfb call: same production msg mid
 * loop, same _vfft_zturn_s0tb_bwd ingest, and the generated symbol takes
 * the GATE-0-proven arg tuple (OLs = count = N/8). Both smoke and timing
 * go through ZT_EXEC_BWD, so the sink build's smoke gates the swapped path. */
#ifdef ZT_STFB_SINK
#define ZSK_ABI const double *, const double *, double *, double *, \
                const double *, const double *, size_t, size_t, size_t, \
                size_t, size_t
void radix8_z_stf_r4sk_bwd_avx2(ZSK_ABI);
static void zt_exec_bwd_sink(const vfft_zturn_plan_t *p,
                             const double *zin, double *zout)
{
    radix8_z_stf_r4sk_bwd_avx2(zin, 0, p->plane, 0, p->tzqb, 0,
                               0, 0, (size_t)p->N / 8, 0, (size_t)p->N / 8);
    for (int s = p->nf - 2; s >= 1; s--) {
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_bwd_avx2 : radix4_z_msg_bwd_avx2;
        f(p->plane, 0, p->plane, 0, p->twzb[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    _vfft_zturn_s0tb_bwd(p->plane, zout, (size_t)p->N);
}
#define ZT_EXEC_BWD zt_exec_bwd_sink
#else
#define ZT_EXEC_BWD vfft_zturn_execute_bwd
#endif

/* ------------------------------------------------------------------ timing */
static double now_ns(void)
{
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

static volatile double g_sink = 0.0;
static char *g_bust = NULL;
static void cachebust(void)                        /* 64 MB > 36 MB L3 */
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

static uint64_t g_rng;                             /* smoke fill (gate LCG) */
static double rnd(void)
{
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return 2.0 * ((double)(g_rng >> 11) * 0x1p-53) - 1.0;
}

static void fill_zin(double *zin, size_t N)        /* deterministic spike fill */
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

#define NARMS 3   /* 0 = zsplit(inc)  1 = zturn(chal)  2 = zsplit(CONTROL) */

static void arm_stats(const double *smp, int reps, int arm,
                      double *med, double *mn, double *mx)
{
    double v[256];                                 /* reps capped below     */
    for (int r = 0; r < reps; r++) v[r] = smp[r * NARMS + arm];
    qsort(v, (size_t)reps, sizeof(double), cmp_d);
    *mn = v[0]; *mx = v[reps - 1];
    *med = (reps & 1) ? v[reps / 2]
                      : 0.5 * (v[reps / 2 - 1] + v[reps / 2]);
}

/* --------------------------------------------------------------- engines */
static vfft_zsplit_plan_t *g_pp = NULL;            /* per-cell */
static vfft_zturn_plan_t  *g_zp = NULL;

static void run_engine(int arm, int dir, const double *zin, double *out)
{
    if (arm == 1) {
        if (dir) ZT_EXEC_BWD(g_zp, zin, out);      /* B1 seam: see header */
        else     vfft_zturn_execute_fwd(g_zp, zin, out);
    } else {                                       /* arms 0 and 2: SAME code */
        if (dir) vfft_zsplit_execute_bwd(g_pp, zin, out);
        else     vfft_zsplit_execute_fwd(g_pp, zin, out);
    }
}

/* ------------------------------------------------------------------ ring */
/* slot_stride multiple of 4KB => every slot base is 0 mod 4KB; zin at +0,
 * out at +16N+64 => out mod 4KB == 64 for every slot (16N is a 4KB multiple
 * at all timed N). Identical geometry to bench_ingest_diag (skew=64).      */
#define SKEW 64
typedef struct { double *zin; double *out; } slot_t;
typedef struct { int R; size_t stride; char *arena; slot_t *tab; } ring_t;

static size_t slot_stride(size_t N)
{
    return (32*N + 8192 + 4095) & ~(size_t)4095;
}
static int cold_slots(size_t N)
{
    const size_t need = (size_t)80 << 20;          /* >= 80 MB > 2x L3 */
    return (int)((need + slot_stride(N) - 1) / slot_stride(N));
}

static ring_t *make_ring(size_t N, int R)
{
    ring_t *rg = (ring_t *)malloc(sizeof(ring_t));
    if (!rg) { fprintf(stderr, "FATAL: ring alloc\n"); exit(2); }
    rg->R = R;
    rg->stride = slot_stride(N);
    rg->arena = (char *)_aligned_malloc((size_t)R * rg->stride, 4096);
    rg->tab = (slot_t *)malloc((size_t)R * sizeof(slot_t));
    if (!rg->arena || !rg->tab) { fprintf(stderr, "FATAL: arena alloc\n"); exit(2); }
    const size_t plane = 16 * N;                   /* bytes */
    for (int i = 0; i < R; i++) {
        rg->tab[i].zin = (double *)(rg->arena + (size_t)i * rg->stride);
        rg->tab[i].out = (double *)(rg->arena + (size_t)i * rg->stride
                                    + plane + SKEW);
    }
    return rg;
}
static void free_ring(ring_t *rg)
{
    _aligned_free(rg->arena); free(rg->tab); free(rg);
}

/* natural fill into every slot's zin (slot 0 computed, others copied);
 * out planes committed. */
static void ring_fill_natural(ring_t *rg, size_t N)
{
    const size_t plane = 16 * N;
    fill_zin(rg->tab[0].zin, N);
    for (int i = 1; i < rg->R; i++)
        memcpy(rg->tab[i].zin, rg->tab[0].zin, plane);
    for (int i = 0; i < rg->R; i++)
        memset(rg->tab[i].out, 0, plane);
}

/* LEGACY comb (production fwd of the natural fill) into every slot's zin.
 * One untimed production pass into a heap temp, then copied.              */
static void ring_fill_comb(ring_t *rg, size_t N)
{
    const size_t plane = 16 * N;
    double *nat  = (double *)malloc(plane);
    double *comb = (double *)malloc(plane);
    if (!nat || !comb) { fprintf(stderr, "FATAL: comb alloc\n"); exit(2); }
    fill_zin(nat, N);
    vfft_zsplit_execute_fwd(g_pp, nat, comb);
    for (int i = 0; i < rg->R; i++)
        memcpy(rg->tab[i].zin, comb, plane);
    free(nat); free(comb);
}

/* ----------------------------------------------------------------- smoke */
/* Gamma logic verbatim from zturn_proto_gate.c GATE 1/2: EXACT memcmp per
 * double. Own small arena, 64-B skews.                                     */
static int smoke_cell(int N)
{
    const size_t nd = 2 * (size_t)N, K2 = (size_t)N / 32, OL = (size_t)N / 8;
    const size_t sz = nd * 8, stride = sz + 64;
    char *ar = (char *)_aligned_malloc(5 * stride + 64, 4096);
    if (!ar) { fprintf(stderr, "FATAL: smoke alloc\n"); exit(2); }
    double *x        = (double *)(ar + 0 * stride);
    double *out_ref  = (double *)(ar + 1 * stride);
    double *out_z    = (double *)(ar + 2 * stride);
    double *back_ref = (double *)(ar + 3 * stride);
    double *back_z   = (double *)(ar + 4 * stride);

    g_rng = 0x5EED0000 + (uint64_t)N;
    for (size_t i = 0; i < nd; i++) x[i] = rnd();

    /* fwd: production and zturn, compare under Gamma, EXACT */
    vfft_zsplit_execute_fwd(g_pp, x, out_ref);
    vfft_zturn_execute_fwd(g_zp, x, out_z);
    long badf = 0;
    for (size_t l = 0; l < 8; l++)
        for (size_t k2 = 0; k2 < K2; k2++)
            for (size_t j = 0; j < 4; j++) {
                const size_t zi = l * OL + 4 * k2 + j;
                const size_t li = l * OL + j * K2 + k2;
                if (memcmp(&out_z[2*zi],     &out_ref[2*li],     8)) badf++;
                if (memcmp(&out_z[2*zi + 1], &out_ref[2*li + 1], 8)) badf++;
            }
    /* bwd: each engine on ITS OWN comb, outputs both natural, EXACT */
    vfft_zsplit_execute_bwd(g_pp, out_ref, back_ref);
    ZT_EXEC_BWD(g_zp, out_z, back_z);              /* B1 seam: see header */
    long badb = 0;
    for (size_t i = 0; i < nd; i++)
        if (memcmp(&back_z[i], &back_ref[i], 8)) badb++;

    const int fin = all_finite(out_ref, nd) && all_finite(out_z, nd)
                 && all_finite(back_ref, nd) && all_finite(back_z, nd);
    const int ok = (badf == 0) && (badb == 0) && fin;
    printf("  SMOKE N=%-6d fwd-Gamma EXACT: %ld/%zu bad | bwd EXACT: %ld/%zu bad"
           " | finite: %s | %s\n",
           N, badf, nd, badb, nd, fin ? "yes" : "NO", ok ? "PASS" : "**FAIL**");
    _aligned_free(ar);
    return ok;
}

/* ------------------------------------------------------------ timed pass */
static int run_pass(size_t N, ring_t *rg, int hot, int dir, int reps,
                    int pace_ms)
{
    const char *dnm = dir ? "bwd" : "fwd";
    long inner = 1; int K = 1; double est;

    if (hot) {                    /* inner sized ONCE from the incumbent  */
        const slot_t *s = &rg->tab[0];
        run_engine(0, dir, s->zin, s->out);         /* warm x3 */
        run_engine(0, dir, s->zin, s->out);
        run_engine(0, dir, s->zin, s->out);
        double t0 = now_ns();
        for (int i = 0; i < 8; i++) run_engine(0, dir, s->zin, s->out);
        est = (now_ns() - t0) / 8.0;
        if (est < 1.0) est = 1.0;
        inner = (long)(1.2e6 / est) + 1;
        if (inner < 1) inner = 1;
        if (inner > 2000000) inner = 2000000;
    } else {                      /* est_cold: incumbent on cold slots    */
        cachebust();
        double t0 = now_ns();
        for (int i = 0; i < 8; i++)
            run_engine(0, dir, rg->tab[i].zin, rg->tab[i].out);
        est = (now_ns() - t0) / 8.0;
        if (est < 1.0) est = 1.0;
        K = (int)(150000.0 / est) + 1;              /* region >= ~150 us */
        if (K < 32) K = 32;
        if (K > rg->R) K = rg->R;
    }

    printf("  [%s/%s] N=%-6zu arms: 0=zsplit(inc) 1=zturn(chal) "
           "2=zsplit(CONTROL)  %s=%ld  est=%.0fns\n",
           hot ? "hot " : "cold", dnm, N,
           hot ? "inner" : "K", hot ? inner : (long)K, est);
    fflush(stdout);

    double *smp = (double *)malloc((size_t)reps * NARMS * sizeof(double));
    if (!smp) { fprintf(stderr, "FATAL: smp alloc\n"); exit(2); }

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j < NARMS; j++) {
            const int a = (j + r) % NARMS;          /* rotation */
            cachebust();                            /* bust FIRST (it heats) */
            Sleep((DWORD)pace_ms);                  /* THEN cool             */
            double t;
            if (hot) {
                const slot_t *s = &rg->tab[0];
                double t0 = now_ns();
                for (long i = 0; i < inner; i++)
                    run_engine(a, dir, s->zin, s->out);
                t = (now_ns() - t0) / (double)inner;
            } else {                                /* cursor RESET to slot 0 */
                const slot_t *tab = rg->tab;
                double t0 = now_ns();
                for (int i = 0; i < K; i++)
                    run_engine(a, dir, tab[i].zin, tab[i].out);
                t = (now_ns() - t0) / (double)K;
            }
            smp[r * NARMS + a] = t;
        }
    }
    g_sink += rg->tab[0].out[0];

    for (int r = 0; r < reps; r++)
        printf("    rep %2d (order %d,%d,%d): inc=%9.1f  chal=%9.1f  "
               "ctl=%9.1f\n", r, r % NARMS, (r + 1) % NARMS, (r + 2) % NARMS,
               smp[r*NARMS+0], smp[r*NARMS+1], smp[r*NARMS+2]);

    double med[NARMS], mn[NARMS], mx[NARMS];
    for (int a = 0; a < NARMS; a++)
        arm_stats(smp, reps, a, &med[a], &mn[a], &mx[a]);

    const double r_chal = med[1] / med[0];
    const double r_ctl  = med[2] / med[0];
    const int ctl_ok = (r_ctl >= 0.98 && r_ctl <= 1.02);

    printf("  med: inc=%.1f chal=%.1f ctl=%.1f\n", med[0], med[1], med[2]);
    printf("  min: inc=%.1f chal=%.1f ctl=%.1f\n", mn[0], mn[1], mn[2]);
    printf("  max: inc=%.1f chal=%.1f ctl=%.1f\n", mx[0], mx[1], mx[2]);
    printf("SUMMARY N=%zu regime=%s dir=%s med_inc=%.1f med_chal=%.1f "
           "med_ctl=%.1f r_chal=%.4f r_ctl=%.4f ctl=%s\n\n",
           N, hot ? "hot" : "cold", dnm, med[0], med[1], med[2],
           r_chal, r_ctl, ctl_ok ? "PASS" : "VOID");
    fflush(stdout);
    free(smp);
    return ctl_ok;
}

/* one regime = fwd pass then bwd pass on the SAME ring (zin refilled) */
static int run_regime(size_t N, int hot, int reps, int pace_ms)
{
    int ok = 1;
    ring_t *rg = make_ring(N, hot ? 1 : cold_slots(N));
    ring_fill_natural(rg, N);
    if (!run_pass(N, rg, hot, 0, reps, pace_ms)) ok = 0;
    Sleep((DWORD)(5 * pace_ms));
    ring_fill_comb(rg, N);                          /* bwd input, see header */
    if (!run_pass(N, rg, hot, 1, reps, pace_ms)) ok = 0;
    free_ring(rg);
    return ok;
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv)
{
    int cells_all[4] = { 2048, 4096, 8192, 16384 };
    int cells[8]; int ncells = 0;
    char mode = '2';              /* 'h' hot, 'c' cold, '2' both */
    int reps = 15, pace_ms = 1000, smoke_only = 0, pos = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--smoke") == 0) { smoke_only = 1; continue; }
        switch (pos++) {
        case 0:
            if (strcmp(argv[i], "all") == 0) break;
            cells[ncells++] = atoi(argv[i]); break;
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
    if (reps % NARMS) reps += NARMS - reps % NARMS;   /* multiple of 3 */
    if (reps > 249) reps = 249;                       /* arm_stats cap */

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

    printf("bench_zturn_final: mode=%s reps=%d (x3 arms) pace=%dms "
           "cellpace=%dms %s\n",
           mode == '2' ? "both" : (mode == 'h' ? "hot" : "cold"),
           reps, pace_ms, 5 * pace_ms, smoke_only ? "[SMOKE ONLY]" : "");
    printf("arms: 0=vfft_zsplit(inc) 1=vfft_zturn(chal) 2=vfft_zsplit(CTL); "
           "medians primary; ctl band 0.98-1.02 on medians\n\n");

    int any_void = 0, any_smoke_fail = 0;
    for (int ci = 0; ci < ncells; ci++) {
        const int N = cells[ci];
        if (N < 1024 || (N & (N - 1))) {
            fprintf(stderr, "FATAL: N must be a power of two >= 1024\n");
            return 2;
        }
        int chain[VFFT_ZSPLIT_MAX_NF];
        const int nf = vfft_zsplit_default_chain(N, chain);
        if (!nf) { printf("N=%d: no calibrated chain, skip\n", N); continue; }
        g_pp = vfft_zsplit_create(N, chain, nf);
        g_zp = vfft_zturn_create(N);
        if (!g_pp || !g_zp) {
            fprintf(stderr, "FATAL: N=%d plan create failed\n", N); return 2;
        }
        /* SAME chain per cell for both engines — assert at create */
        if (g_zp->nf != nf || memcmp(g_zp->chain, chain, (size_t)nf * sizeof(int))) {
            fprintf(stderr, "FATAL: N=%d chain mismatch zsplit vs zturn\n", N);
            return 2;
        }

        /* re-point both engines' scratch into ONE cell arena (see header):
         * inc scratch at +128 mod 4KB, chal scratch at +192 mod 4KB. */
        const size_t plane_b = 16 * (size_t)N;
        const size_t s1 = (plane_b + 128 + 4095) & ~(size_t)4095;
        char *cell_arena = (char *)_aligned_malloc(s1 + plane_b + 4096, 4096);
        if (!cell_arena) { fprintf(stderr, "FATAL: cell arena\n"); return 2; }
        double *sp_save = g_pp->sp;
        double *pl_save = g_zp->plane;
        g_pp->sp    = (double *)(cell_arena + 128);
        g_zp->plane = (double *)(cell_arena + s1 + 192);

        char cs[64]; int o = 0;
        for (int s = 0; s < nf; s++)
            o += sprintf(cs + o, "%d%s", chain[s], s < nf - 1 ? "." : "");
        printf("== N=%d chain=%s (both engines) t2q=%d ==\n", N, cs, g_pp->t2q);

        if (!smoke_cell(N)) {
            any_smoke_fail = 1;
            printf("  smoke FAILED -- cell NOT timed\n\n");
        } else if (!smoke_only) {
            if (mode != 'c') {
                if (!run_regime((size_t)N, 1, reps, pace_ms)) any_void = 1;
                if (mode == '2') Sleep((DWORD)(5 * pace_ms));
            }
            if (mode != 'h') {
                if (!run_regime((size_t)N, 0, reps, pace_ms)) any_void = 1;
            }
        } else {
            printf("\n");
        }

        g_pp->sp = sp_save;                 /* restore before destroy */
        g_zp->plane = pl_save;
        _aligned_free(cell_arena);
        vfft_zturn_destroy(g_zp);  g_zp = NULL;
        vfft_zsplit_destroy(g_pp); g_pp = NULL;
        if (ci + 1 < ncells && !smoke_only) Sleep((DWORD)(5 * pace_ms));
    }
    (void)g_sink;
    if (any_smoke_fail) return 1;
    if (any_void) {
        printf("NOTE: at least one control out of band -- those cells are "
               "VOID; re-run with more pacing/reps.\n");
        return 3;
    }
    return 0;
}
