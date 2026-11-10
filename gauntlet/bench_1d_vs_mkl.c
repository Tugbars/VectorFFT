/* bench_1d_vs_mkl.c -- THE canonical bench against Intel MKL (gauntlet/README.md).
 *
 * One cell per process: our plan through the library's front door on the wisdom directory
 * given (a row there replays, a miss races and banks), MKL's DFTI descriptor for the same
 * transform, both timed inside the library's measurement scope (bench_scope.h: the pin to a
 * P-core, the SMT-sibling guard, the priority, the machine's measurement lock), one csv row
 * with both times, the ratio and the correctness columns. MKL is optional at build time
 * (build.py --mkl; without it the row carries our numbers and the correctness checks). The
 * gauntlet's cells take other comparators: --cmp fftw (bound at run time from $VFFT_FFTW_DLL,
 * else the repository's gauntlet/fftw/libfftw3-3.dll, else vcpkg's fftw3.dll, no build flag:
 * the 1D/2D/3D c2c cells and the real cells, out of
 * place, at one thread or under --mt with an FFTW built with its threads API) and --cmp kfr
 * (a --kfr build; one thread).
 *
 * Protocol (docs/design/measurement_arms.md): a warm-up, then timing windows of best-of-5
 * samples; cachebust and a cool-down between the two engines; the engine order set by the
 * flip argument, alternated across processes so neither side always runs first; the
 * threaded cells (--mt, T = $VFFT_MT, default 8) run both sides at T, never paced, and
 * print the library's engagement counters so a serial fallback cannot pass as threaded.
 *
 * Usage: bench_1d_vs_mkl [mode flags] [wisdom] [csv] [pace_ms] [N] [K] [cool_ms] [flip] [core]
 *   wisdom   a file inside the wisdom directory (names the directory)   csv  the row's file
 *   pace_ms  the idle between trials        N  the cell; a shape N1xN2[xN3] for a rank-2/3 cell
 *   K        the batch count (N2 where a mode says so)   cool_ms  the idle between the engines
 *   flip     1 = MKL first                  core  the P-core to pin (-1 = none)
 * Modes, leading flags in any order (no mode flag: this usage, exit 2):
 *   1D c2c    --k1nat / --k1noop / --k1zip / --k1dir the natural in-place and out-of-place forms;
 *             --mt threaded; --cmp mkl|fftw|kfr the comparator
 *   1D real   --realfwd / --realbwd (r2c / c2r); --zr2c the earlier real form
 *   2D / 3D   --2dilnat / --3dilnat the gauntlet's 2D and 3D c2c cells; --2drealnat / --3dreal the
 *             real cells; --realip / --realown the real cell in place; --2dil, --3dil, --2dreal
 *             the earlier forms
 *   batches   --lmc2c / --lmr2c / --lmc2r the lane-major 1D batches; --lm2d / --lm3d the lane-major
 *             2D / 3D c2c batches; --lm2dr2c / --lm2dc2r and --lm3dr2c / --lm3dc2r the 2D / 3D real
 *             batches (the contiguous queue timed beside them); --ilmt the threaded
 *             transform-contiguous batch
 *   FP32      --f32 (a build with VFFT_HAS_FLOAT: build.py --float) beside --k1noop [--mt] or --k1nat:
 *             the K=1 natural cell on float data through vfftf.h, MKL DFTI_SINGLE; K > 1 in the K slot
 *             is the contiguous batch; plan z:ilp-f32, path nat-oop-f32 / nat-ip-f32 (run_k1z_cell_f32)
 * Build: python gauntlet/build.py --compile --mkl --vfft, or the CMake target vfft-gauntlet.
 *   MKL is linked LP64 through mkl_rt: ILP64 corrupts the DFTI strides array ("Inconsistent
 *   configuration parameters" at DftiCommit).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "threads.h" /* thread_pool_resize: --ilmt and --3dil tear this TU's pool down before an MKL arm */
#include "env.h"     /* vfft_env_init + vfft_pin_thread */
#include "vfft_diagnostics.h"   /* the engagement counters (and, since 2026-10-07, the measurement scope as a tool
                                 * and vfft_isa()): the header, not hand-written prototypes -- vfft_ztt_mt_passes()
                                 * had been an IMPLICIT declaration here until the header's own arrived through
                                 * bench_scope.h and conflicted with it */
#include "common/support/race_timing.h" /* vfft_now_ns: the one monotonic clock */
#include "il/rank1/k1_fourstep_band.h" /* vfft_k1fs_band: the upper band 2^19..2^22 (2026-09-15) */
#include "il/rank1/ztt.h"            /* vfft_ztt_odd_band: the --k1nat/--k1noop direct cell at 2^a*odd (2026-09-15) */
#include "wisdom2_oop.h"        /* vfft_oop_wisdom_entry_t (run_k1z_cell's verdict argument) and,
                                 * underneath, the wisdom2 store (vw2_open / vw2_scan) */
#include "zr2c.h"               /* --zr2c: D2 interleaved real folds (zr2c.h, Phase 1) */
#include "vfft.h"               /* K=1 kind-4 cascade cells: public front door
                                 * (vfft_create serves the banked route+chain
                                 * verdict). Requires build.py --vfft. */
#include "il/planning/policy_il.h"    /* THE admission law: the bench asks, it does not
                                 * keep its own copy (planning_policy_design.md L11) */

#ifdef VFFT_HAS_KFR
#include "kfr_arm.h"   /* the KFR comparator arm (C++ behind a C interface; wired 2026-09-25, untested) */
static int g_cmp_kfr = 0;   /* --cmp kfr: the K=1 cell's comparator is KFR instead of MKL */
#endif
/* ── THE FFTW COMPARATOR ARM (--cmp fftw, 2026-09-28) ───────────────────────
 * Needs NO build flag and NO link: ref_fftw.h binds fftw3 at runtime from
 * $VFFT_FFTW_DLL by absolute path and asserts the library is genuine FFTW
 * (MKL exports 92 fftw_* wrappers, so a linked fftw3 can be MKL in disguise).
 * The arm therefore compiles into every build of this bench and is inert
 * until --cmp fftw selects it.
 *
 * WHY HERE AND NOT IN bench_1d_vs_fftw.c: that separate bench has drifted
 * from this one's protocol (measured 2026-09-28: 1 of this file's 16
 * discipline markers -- it has neither the two-window timing nor the sibling
 * guard). A comparator that does not share the timed path cannot be compared
 * with one that does, so FFTW becomes an ARM of the canonical bench and
 * inherits the pacing, the cool-downs, the cachebust, the flip, the pin and
 * the control cell by construction, exactly as the KFR arm does. */
#include "ref_fftw.h"
static int        g_cmp_fftw = 0;
static fftwx_api_t g_fx_arm;
static int        g_fx_arm_ok = -1;   /* -1 = not yet bound */
/* $VFFT_FFTW_WIS: the arm's wisdom file, imported at bind and re-exported
 * after every plan it builds, so the run's first MEASURE verdict for a
 * problem serves every later process of the run (one process per cell): the
 * comparator holds still across cells, directions and method races, as our
 * banked wisdom does. Unset: each process measures afresh. */
static const char *fftw_wis_path(void)
{
    const char *e = getenv("VFFT_FFTW_WIS");
    return (e && *e) ? e : NULL;
}
static void fftw_wis_export(void)
{
    if (fftw_wis_path())
        g_fx_arm.export_wisdom_to_filename(fftw_wis_path());
}
static int fftw_arm_bind(void)
{
    char err[512];
    if (g_fx_arm_ok < 0)
    {
        g_fx_arm_ok = fftwx_bind(&g_fx_arm, err, sizeof err) ? 1 : 0;
        if (!g_fx_arm_ok)
            fprintf(stderr, "--cmp fftw: %s\n", err);
        else
        {
            const int wis = fftw_wis_path() ? g_fx_arm.import_wisdom_from_filename(fftw_wis_path()) : 0;
            printf("# comparator: %s (%s) wisdom %s\n", g_fx_arm.version, g_fx_arm.dll_path,
                   fftw_wis_path() ? (wis ? "replayed" : "cold") : "per process");
        }
    }
    return g_fx_arm_ok;
}
/* The arm at the cell's thread count (2026-10-19), called before every plan it
 * builds: T > 1 needs the DLL's threads API (ref_fftw.h's optional group: a
 * build with --enable-threads --with-combined-threads, or CMake ENABLE_THREADS
 * + WITH_COMBINED_THREADS; vcpkg's fftw3.dll has none) -- fftw_init_threads
 * once, then fftw_plan_with_nthreads(T), so FFTW plans and runs at the T our
 * plan and MKL do. Returns 0, with the reason on stderr, when T > 1 and the
 * DLL cannot thread: a threaded cell against a serial FFTW is not a
 * comparison, and main exits 2 on it before any cell runs. */
static int fftw_arm_threads(int T)
{
    static int inited = 0;
    if (!fftw_arm_bind())
        return 0;
    if (T <= 1)
    {
        if (inited)
            g_fx_arm.plan_with_nthreads(1);
        return 1;
    }
    if (!g_fx_arm.init_threads || !g_fx_arm.plan_with_nthreads)
    {
        fprintf(stderr, "--cmp fftw --mt: %s exports no threads API (fftw_init_threads / fftw_plan_with_nthreads), "
                        "so it cannot run the cell at T=%d; a threaded run needs an FFTW built with "
                        "--enable-threads --with-combined-threads (CMake: ENABLE_THREADS + WITH_COMBINED_THREADS), "
                        "named in $VFFT_FFTW_DLL\n", g_fx_arm.dll_path, T);
        return 0;
    }
    if (!inited)
    {
        if (!g_fx_arm.init_threads())
        {
            fprintf(stderr, "--cmp fftw --mt: fftw_init_threads failed in %s\n", g_fx_arm.dll_path);
            return 0;
        }
        inited = 1;
    }
    g_fx_arm.plan_with_nthreads(T);
    return 1;
}
#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

#ifndef BENCH_K
#define BENCH_K 4 /* K=4 only — the cells we calibrated (MEASURE) */
#endif

static void pace(int ms)
{
    if (ms <= 0)
        return;
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
/* inter-trial idle (env VFFT_TRIAL_PACE_MS) — lets the best-of-5 min reflect a
 * cooler core on big cells that heat-soak between trials. 0 = back-to-back. */
static int g_trial_pace_ms = 0;
static double *alloc_d(size_t n)
{
    double *p = NULL;
    if (!(p = vfft_aligned_alloc(n * sizeof(double))))
    {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return p;
}
static void free_d(double *p) { vfft_aligned_free(p); }
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = alloc_d(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++)
        j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++)
        a += j[i];
    (void)a;
    free_d(j);
}
static int reps_for(size_t total)
{
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0)
        return atoi(e);
    int r = (int)(2e6 / (total + 1));
    if (r < 8)
        r = 8;
    if (r > 100000)
        r = 100000;
    return r;
}

/* MT thread count (--mt). 1 = single-thread (legacy path, byte-identical timing).
 * MKL gets mkl_set_num_threads(g_mt). */
static int g_mt = 1;
/* ════════════════════════════════════════════════════════════════════════
 * K=1 SCRAMBLED-cascade cells (kind-4 oop_wisdom lines) — FRONT-DOOR path.
 *
 * A "N 1 4 zs_t2q cc_chain ns [zs_route zt_t2q]" line (src/core/oop/
 * oop_wisdom.h is the single source of format truth; the trailing route pair
 * is the OPTIONAL Phase-5 route axis) carries NO c2c factorization — it is
 * the banked verdict of the create-time route race. The only measurement
 * that makes sense for these cells is the PUBLIC front door serving that
 * verdict: vfft_create (C2C, OUT-OF-PLACE, order=SCRAMBLED, howmany=1) reads
 * the kind-4 line from the bundle's oop_wisdom.txt and builds the zsplit /
 * ZTURN-S cascade with the banked chain + terminator pick (pure read, no
 * race; VFFT_ZRACE_VERBOSE=1 prints the served route). Execution rides the
 * cascade's interleaved z contract (sim == dim == NULL).
 *
 * Fairness mirrors the retired stride sweep's measure_ab exactly: same warmup(10)/best-of-5/reps_for
 * shape, cachebust + cool_ms idle BETWEEN engines, flip order. MKL is the
 * interleaved OOP descriptor — the cascade's like-for-like (split REAL_REAL
 * storage is the batched paths' contract, not K=1 z's; zsplit_api_gate.c
 * precedent).
 * ════════════════════════════════════════════════════════════════════════ */
static int g_k1dir = 0;              /* --k1dir: time K=1 IL in-place BOTH
                                      * directions in ONE process. The falsifier
                                      * for "the backward is slow because it runs
                                      * MONOLITHIC codelets while the forward runs
                                      * BLOCKED ones": VFFT_NO_ILBLK forces the
                                      * FORWARD monolithic and touches only
                                      * mid_f/leaf_f, so the backward column is an
                                      * unchanged CONTROL by construction. If
                                      * fwd(NO_ILBLK) ~= bwd, the kernel class is
                                      * the whole story. Implies --k1zip/--k1nat.
                                      * NOTE wisdom2 section 5: a banked il_kv
                                      * BEATS VFFT_NO_ILBLK, so this is valid only
                                      * on a cell with none (n=1024 has none). */
static int g_k1zip = 0;              /* --k1zip: K=1 kind-4 cells IN-PLACE
                                      * (both engines) — the apples-to-
                                      * apples in-place interleaved cell */
static void bench_pin_one_thread(void); /* the one-thread protocol helper (defined beside ilmt_pin_pcores) */
static long g_cell_engaged = -1;        /* the threaded 2D/3D cell: the tier's MT counter over the timed executes; -1 = one thread */
static int g_k1_direct_cell = 0;  /* the policy's admission for the direct K=1 cell */
static int g_k1noop_mt = 0;          /* --k1noop --mt: the odd-N flat DIT's threaded verdict
                                       * (il_flatdit_mt.h) vs MKL at the same T — the
                                       * two-team protocol of --3dil --mt (traps a-d) */
static int g_k3nat = 0;              /* --3dilnat (2026-09-24): the 3D interleaved gauntlet cell, the shape N1xN2xN3 in the N slot */
static int g_k2nat = 0;              /* --2dilnat (2026-09-23): the 2D interleaved gauntlet
                                      * cell -- the K=1 natural cell's protocol on a
                                      * dims=2 plan, N1 x N2, OOP, natural, one thread;
                                      * MKL DFTI 2D out of place; one csv row per flip */
static int g_k1nat = 0;              /* --k1nat (B6): --k1zip discipline +
                                      * order=NATURAL on our side. MKL is
                                      * ALWAYS natural — this mode finally
                                      * removes the permanent scrambled-vs-
                                      * natural caveat: both engines produce
                                      * THE SAME spectrum, and the
                                      * correctness column becomes a CROSS-
                                      * ENGINE elementwise compare (stronger
                                      * than roundtrip, which cannot gate
                                      * ordering). Implies g_k1zip. */
static vw2_store_t g_k1z_store;      /* the LIVE store — what the front door serves */
static int g_k1z_oopw_loaded = 0;
static const char *g_k1z_wpath = NULL;

/* The bundle directory: the positional wisdom argument's parent (the file
 * itself is only a locator now — the store lives beside it, and the front
 * door resolves the same directory). */
static const char *k1z_dir(void)
{
    static char dir[600];
    static int done = 0;
    const char *b1, *b2, *base;
    if (done) return dir;
    done = 1;
    snprintf(dir, sizeof dir, ".");
    if (!g_k1z_wpath) return dir;
    b1 = strrchr(g_k1z_wpath, '/');
    b2 = strrchr(g_k1z_wpath, '\\');
    base = b1 ? b1 : b2;
    if (b1 && b2) base = (b1 > b2) ? b1 : b2;
    if (base)
    {
        size_t dl = (size_t)(base - g_k1z_wpath); /* dir WITHOUT the separator */
        if (dl == 0) dl = 1;                      /* "/oop_wisdom.txt" -> "/"  */
        if (dl >= sizeof dir) dl = sizeof dir - 1;
        memcpy(dir, g_k1z_wpath, dl);
        dir[dl] = 0;
    }
    return dir;
}

/* Caller-owned front-door bundle rooted at the wisdom argument's directory.
 * The old basename contract (the file HAD to be named oop_wisdom.txt so the
 * bundle would read the very lines this process parsed) is GONE: the front
 * door now serves the wisdom2 store in that directory, and this bench reads
 * its verdicts from the same store — so bundle and bench agree by
 * construction rather than by filename coincidence. */
static const char *k1z_bundle(void)
{
    static const char *W = NULL;
    static int tried = 0;
    if (tried || !g_k1z_wpath)
        return W;
    tried = 1;
    W = (k1z_dir());
    if (!W)
        fprintf(stderr, "k1z: vfft_wisdom_load(%s) failed — cells skipped\n", k1z_dir());
    return W;
}

/* the gap between the two timing windows: 40 ms idle, then >= 10 ms of the
 * engine's own work untimed (the post-idle state, when it strikes, is sticky
 * for at least that long; the second window then starts on a settled core).
 * The same shape serves the MKL timer through kzb_window_gap. */
#define K1Z_WINDOW_IDLE_MS 40
#define K1Z_WINDOW_WARM_NS 1.0e7
static void k1z_window_gap(vfft_plan h, int dir, double *z0, double *S)
{
    const double tw0 = vfft_now_ns();
    pace(K1Z_WINDOW_IDLE_MS);
    do
        g_k1zip ? vfft_execute(h, dir, S, S)
                : vfft_execute(h, dir, z0, S);
    while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
}

static double k1z_time_vfft_d(vfft_plan h, double *z0, double *S, size_t total,
                              int dir)
{
    /* --k1zip: aliased interleaved form (S, NULL, S, NULL) on the evolving
     * buffer — mirrors the MKL in-place arm's discipline exactly. */
    if (g_k1zip)
        memcpy(S, z0, 2 * total * sizeof(double));
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: the pool rebuilt before this arm starts up, the
         * workers' cache partition settles, the caller core is back at full
         * clock (3D_mt_il_strategy.md, the transient rule) */
        const double tw = vfft_now_ns();
        do
            vfft_execute(h, dir, z0, S);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            g_k1zip ? vfft_execute(h, dir, S, S)
                    : vfft_execute(h, dir, z0, S);
    int reps = reps_for(total);
    double best = 1e18;
    /* TWO WINDOWS (2026-09-21): best-of-5, then again after k1z_window_gap().
     * The gauntlet read the same banked chain3 plan 1.1-1.5x slower than the
     * race had, sticky for >= 10 ms after any idle (the 300 ms cool before
     * this arm in flip 1) with the core clock, the memory and the buffer
     * placement all verified unchanged (chain3_skew_probe.c): a state of the
     * MACHINE, not of the plan, that a single 15 ms window sits inside
     * whole. Two windows, separated, give the engine a second chance to be
     * measured at its own speed; MKL's timer takes the same two windows. */
    for (int win = 0; win < 2; win++)
    {
        if (win) k1z_window_gap(h, dir, z0, S);
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                g_k1zip ? vfft_execute(h, dir, S, S)
                        : vfft_execute(h, dir, z0, S);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    return best;
}

static double k1z_time_vfft(vfft_plan h, double *z0, double *S, size_t total)
{
    return k1z_time_vfft_d(h, z0, S, total, VFFT_FORWARD);
}

#ifdef VFFT_HAS_MKL
static double k1z_time_mkl(int N, const double *z0, size_t total)
{
    /* --k1zip: DFTI_INPLACE, single-buffer compute. Timing loops run on the
     * evolving buffer (the dataflow is data-independent) — same discipline as
     * the vfft in-place arm, so neither engine pays a refill in the loop. */
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, g_k1zip ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: MKL's parked team wakes and settles */
        const double tw = vfft_now_ns();
        do
            DftiComputeForward(d, zi, zo);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            g_k1zip ? DftiComputeForward(d, zi) : DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    /* TWO WINDOWS, the same shape as k1z_time_vfft_d (2026-09-21) */
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                g_k1zip ? DftiComputeForward(d, zi) : DftiComputeForward(d, zi, zo);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                g_k1zip ? DftiComputeForward(d, zi) : DftiComputeForward(d, zi, zo);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}
#ifdef VFFT_HAS_KFR
/* the KFR twin of k1z_time_mkl (2026-09-25, UNTESTED): the same windows,
 * trials, reps and pacing; in place under --k1zip; single thread only
 * (KFR's DFT does not thread, so --cmp kfr refuses --mt in main). */
static double k1z_time_kfr(int N, const double *z0, size_t total)
{
    void *p = kfr_c2c_create(N);
    if (!p)
        return 0;
    size_t tsz = kfr_c2c_temp_size(p);
    unsigned char *temp = tsz ? (unsigned char *)alloc_d((tsz + 7) / 8) : NULL;
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
#define KFR_RUN() (g_k1zip ? kfr_c2c_forward_inplace(p, zi, temp) : kfr_c2c_forward(p, zi, zo, temp))
    for (int w = 0; w < 10; w++)
        KFR_RUN();
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                KFR_RUN();
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                KFR_RUN();
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
#undef KFR_RUN
    free_d(zi);
    free_d(zo);
    if (temp) free_d((double *)temp);
    kfr_c2c_destroy(p);
    return best;
}
#endif
/* the K=1 cell's comparator: MKL, or KFR under --cmp kfr (2026-09-25) */
static double k1z_time_cmp_mkl_or_kfr(int N, const double *z0, size_t total)
{
#ifdef VFFT_HAS_KFR
    if (g_cmp_kfr)
        return k1z_time_kfr(N, z0, total);
#endif
    return k1z_time_mkl(N, z0, total);
}
#endif

/* The FFTW arm's timed path: the twin of k1z_time_mkl / k1z_time_kfr, window
 * for window -- 10 warm-ups, then TWO timing windows (the second preceded by
 * the idle and the re-warm), best of 5 trials each, reps_for(total) reps per
 * trial, g_trial_pace_ms between trials. In place under --k1zip, mirroring the
 * MKL arm's DFTI_INPLACE. A plan is built ONCE here and destroyed after, so no
 * planning cost is inside any timed window; FFTW_MEASURE matches the MKL arm's
 * "commit once, then time" shape and $VFFT_FFTW_WIS (imported by the caller)
 * makes it a wisdom replay. --mt: planned at T (fftw_arm_threads) and warmed
 * >= 5 ms untimed as the MKL arm is -- FFTW's workers are spawned at the first
 * threaded execute, under the P-core mask main set. */
static double k1z_time_fftw(int N, const double *z0, size_t total)
{
    if (!fftw_arm_threads(g_k1noop_mt ? g_mt : 1))
        return 0;
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    fftwx_plan p = g_k1zip
        ? g_fx_arm.plan_dft_1d(N, (fftwx_complex *)zi, (fftwx_complex *)zi,
                               FFTWX_FORWARD, FFTWX_MEASURE)
        : g_fx_arm.plan_dft_1d(N, (fftwx_complex *)zi, (fftwx_complex *)zo,
                               FFTWX_FORWARD, FFTWX_MEASURE);
    if (!p) { free_d(zi); free_d(zo); return 0; }
    fftw_wis_export();
    memcpy(zi, z0, 2 * total * sizeof(double));   /* AFTER planning: MEASURE scribbles */
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: FFTW's team comes up and settles */
        const double tw = vfft_now_ns();
        do
            g_fx_arm.execute(p);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            g_fx_arm.execute(p);
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                g_fx_arm.execute(p);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                g_fx_arm.execute(p);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    g_fx_arm.destroy_plan(p);
    free_d(zi);
    free_d(zo);
    return best;
}

/* THE comparator entry. Always defined -- the FFTW arm needs no build flag, so
 * a bench built without MKL still has a comparator when --cmp fftw selects it.
 * Returns 0 when none is active, which the caller reports as "no comparator"
 * exactly as a no-MKL build did before. */
static double k1z_time_cmp(int N, const double *z0, size_t total)
{
    if (g_cmp_fftw)
        return k1z_time_fftw(N, z0, total);
#ifdef VFFT_HAS_MKL
    return k1z_time_cmp_mkl_or_kfr(N, z0, total);
#else
    (void)N; (void)z0; (void)total;
    return 0;
#endif
}

/* A CELL'S ROW IS REPLACED, NEVER DUPLICATED (2026-09-21). A gauntlet re-runs
 * the cells a fix touched into the SAME csv, so an earlier row for the same
 * (N, K, plan, path, flip) is dropped and the new one takes its place at the
 * end; every other row is untouched. Returns 1 when a row was replaced, 0
 * when the file holds no such row (the caller then appends as before). The
 * file is rewritten through a fresh handle: the appending handle stays open
 * and idle (flushed first), and a gauntlet csv is a few hundred KB. Lines are
 * copied without their line ending and re-terminated, so the rewrite keeps
 * the text-mode convention the appending handle writes with.
 * THE FLIP IS FOUND BY NAME (2026-10-04): the column `flip` of the file's own
 * header line, wherever a schema puts it -- a threaded run's `engaged`
 * column follows it in the 1D rows, so a key on the LAST field matched the
 * two flips of a cell whose engaged counts agree (every four-step: its 2D
 * child's threading is not counted) and the second flip erased the first.
 * A header without `flip` keys on the last field. */
static const char *g_csv_path = NULL;

/* field `col` (0-based) of the csv line s (n chars): *f, *fn; 0 when the
 * line has fewer fields; col < 0 = the last field */
static int _k1z_field(const char *s, size_t n, int col, const char **f, size_t *fn)
{
    size_t i = 0, j;
    int c;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    if (col < 0)
    {
        for (i = n; i > 0 && s[i - 1] != ','; i--) ;
        *f = s + i;
        *fn = n - i;
        return 1;
    }
    for (c = 0; c < col; c++)
    {
        while (i < n && s[i] != ',') i++;
        if (i >= n) return 0;
        i++;
    }
    for (j = i; j < n && s[j] != ','; j++) ;
    *f = s + i;
    *fn = j - i;
    return 1;
}

/* the 0-based index of column `name` in the header line h (n chars); -1 when absent */
static int _k1z_col_index(const char *h, size_t n, const char *name)
{
    const size_t ln = strlen(name);
    const char *f;
    size_t fn;
    int c;
    for (c = 0; _k1z_field(h, n, c, &f, &fn); c++)
        if (fn == ln && memcmp(f, name, ln) == 0) return c;
    return -1;
}

static int _k1z_row_matches(const char *line, size_t n, const char *key, size_t klen, int fcol,
                            const char *fl, size_t fln)
{
    const char *f;
    size_t fn;
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) n--;
    if (n < klen || memcmp(line, key, klen) != 0) return 0;
    return _k1z_field(line, n, fcol, &f, &fn) && fn == fln && memcmp(f, fl, fln) == 0;
}

static int k1z_csv_replace(const char *path, const char *row)
{
    FILE *f;
    char *buf, *p, *nl;
    const char *fl;
    long len;
    size_t klen, fln, ln;
    int kf = 0, hits = 0, fcol;
    if (!path) return 0;
    for (p = (char *)row; *p; p++)
        if (*p == ',' && ++kf == 4) break;
    if (kf < 4) return 0;
    klen = (size_t)(p - row) + 1;              /* "N,K,plan,path," */
    f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return 0; }
    buf = (char *)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return 0; }
    len = (long)fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[len] = 0;
    nl = strchr(buf, '\n');
    fcol = _k1z_col_index(buf, nl ? (size_t)(nl - buf) : strlen(buf), "flip");   /* the header's own flip column */
    if (!_k1z_field(row, strlen(row), fcol, &fl, &fln)) { free(buf); return 0; }
    for (p = buf; *p; p = nl)
    {
        nl = strchr(p, '\n');
        nl = nl ? nl + 1 : p + strlen(p);
        if (_k1z_row_matches(p, (size_t)(nl - p), row, klen, fcol, fl, fln)) hits++;
    }
    if (!hits) { free(buf); return 0; }
    f = fopen(path, "w");
    if (!f) { free(buf); return 0; }
    for (p = buf; *p; p = nl)
    {
        nl = strchr(p, '\n');
        nl = nl ? nl + 1 : p + strlen(p);
        if (_k1z_row_matches(p, (size_t)(nl - p), row, klen, fcol, fl, fln)) continue;
        ln = (size_t)(nl - p);
        while (ln && (p[ln - 1] == '\n' || p[ln - 1] == '\r')) ln--;
        if (!ln) continue;
        fwrite(p, 1, ln, f);
        fputc('\n', f);
    }
    fputs(row, f);
    fclose(f);
    free(buf);
    return 1;
}

static void run_k1z_cell(int N, const vfft_oop_wisdom_entry_t *ze,
                         FILE *out, int cool_ms, int flip)
{
    /* plan descriptor = the banked verdict: decoded cascade chain + route.
     * ze == NULL is the --k1nat sub-2048 direct cell (Phase B5): no kind-4
     * line exists for the IL-tier band — the front door serves the @nat
     * ILP verdict instead, and the label says so. */
    char plan_s[64];
    (void)ze;   /* the kind-4 (cascade) lines are gone (2026-09-15): every K=1 cell is the IL tier's */
    snprintf(plan_s, sizeof plan_s, "z:ilp");
    const char *path = "ilp";
    if (g_k1nat)
        path = g_k1zip ? "nat-ip" : "nat-oop";
        /* --k1noop: order=NATURAL OOP never attaches the cascade — the
         * kind-4 label would lie; the real engine is the K=1 IL route or the
         * convert fallback (exactly what D1 measures). --k1nat (2026-09-21):
         * natural IN-PLACE gets its own label so the in-place gauntlet's rows
         * never key-collide with the out-of-place ones. */

    if (!g_k1noop_mt) bench_pin_one_thread();   /* the one-thread protocol (once per process) */
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d %-16s   SKIP (front-door bundle unavailable)\n", N, plan_s);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = g_k1zip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    cfg.rank = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = g_k1nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; /* k1z cells run the committed z contract */
    cfg.nthreads = g_k1noop_mt ? g_mt : 1;
    cfg.wisdom_dir = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d %-16s   vfft_create FAILED\n", N, plan_s);
        return;
    }

    size_t total = (size_t)N; /* K = 1 */
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total), *rt = alloc_d(2 * total);
    srand(42 + N + 1);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* roundtrip gate through the API (matched-permutation: bwd inverts fwd).
     * --k1zip: aliased both legs — the rt buffer carries the round trip. */
    if (g_k1zip)
    {
        memcpy(rt, z0, 2 * total * sizeof(double));
        vfft_execute(h, VFFT_FORWARD, rt, rt);
        vfft_execute(h, VFFT_BACKWARD, rt, rt);
    }
    else
    {
        vfft_execute(h, VFFT_FORWARD, z0, S);
        vfft_execute(h, VFFT_BACKWARD, S, rt);
    }
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < 2 * total; i++)
    {
        double e = fabs(rt[i] * inv - z0[i]), m = fabs(z0[i]);
        if (e > maxerr)
            maxerr = e;
        if (m > maxmag)
            maxmag = m;
    }
    double rel = maxmag > 0 ? maxerr / maxmag : maxerr;
    /* The FFTW arm's twin of the cross-engine gate below: both engines
     * natural, same input, same spectrum, so `rel` becomes an ELEMENTWISE
     * comparison rather than a roundtrip (which cannot gate ordering). Same
     * reasoning and same overwrite as the MKL branch. FFTW's plan here has the
     * placement of the cell (in place under --k1zip, else out of place): the
     * same problem the timed arm plans, so one MEASURE search serves both --
     * a threaded 1D search is minutes long, and an in-place plan would be a
     * second one. */
    if (g_cmp_fftw && g_k1nat && fftw_arm_threads(cfg.nthreads))
    {
        double *zf = alloc_d(2 * total), *zv = alloc_d(2 * total);
        double *zo = g_k1zip ? zf : alloc_d(2 * total);
        fftwx_plan p = g_fx_arm.plan_dft_1d(N, (fftwx_complex *)zf,
                                            (fftwx_complex *)zo,
                                            FFTWX_FORWARD, FFTWX_MEASURE);
        if (p)
        {
            fftw_wis_export();
            memcpy(zf, z0, 2 * total * sizeof(double));   /* after planning */
            memcpy(zv, z0, 2 * total * sizeof(double));
            g_fx_arm.execute(p);
            if (g_k1zip)
                vfft_execute(h, VFFT_FORWARD, zv, zv);
            else
                vfft_execute(h, VFFT_FORWARD, z0, zv);
            double xe = 0.0, xm = 0.0;
            for (size_t i = 0; i < 2 * total; i++)
            {
                double e = fabs(zv[i] - zo[i]), m = fabs(zo[i]);
                if (e > xe) xe = e;
                if (m > xm) xm = m;
            }
            rel = xm > 0 ? xe / xm : xe;
            g_fx_arm.destroy_plan(p);
        }
        free_d(zf);
        free_d(zv);
        if (!g_k1zip) free_d(zo);
    }
#ifdef VFFT_HAS_MKL
    if (g_k1nat && !g_cmp_fftw)
    {
        /* --k1nat: the correctness column is the CROSS-ENGINE elementwise
         * compare — both engines natural, same input, same spectrum. This is
         * the check the scrambled modes structurally cannot run (🔴 roundtrip
         * cannot gate ordering; it holds under ANY self-consistent
         * permutation). Overwrites `rel` on purpose: a natural mode whose
         * spectrum does not match MKL's elementwise is WRONG no matter how
         * clean its roundtrip is. */
        DFTI_DESCRIPTOR_HANDLE d = NULL;
        if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                                 (MKL_LONG)N) == DFTI_NO_ERROR)
        {
            DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
            if (DftiCommitDescriptor(d) == DFTI_NO_ERROR)
            {
                double *zm = alloc_d(2 * total), *zv = alloc_d(2 * total);
                memcpy(zm, z0, 2 * total * sizeof(double));
                memcpy(zv, z0, 2 * total * sizeof(double));
                DftiComputeForward(d, zm);
                if (g_k1zip)
                    vfft_execute(h, VFFT_FORWARD, zv, zv);
                else
                { /* --k1noop: OOP handle — aliased calls are not its
                   * contract; run z0 -> zv through the OOP signature. */
                    vfft_execute(h, VFFT_FORWARD, z0, zv);
                }
                double xe = 0.0, xm = 0.0;
                for (size_t i = 0; i < 2 * total; i++)
                {
                    double e = fabs(zv[i] - zm[i]), m = fabs(zm[i]);
                    if (e > xe) xe = e;
                    if (m > xm) xm = m;
                }
                rel = xm > 0 ? xe / xm : xe;
                free_d(zm);
                free_d(zv);
            }
            DftiFreeDescriptor(&d);
        }
    }
#endif

    /* A/B — measure_ab's fairness shape (cachebust + cool between engines, flip).
     * --mt: the two-team protocol — the library pool torn down before the
     * MKL arm (trap a), rebuilt before ours; cool_ms >= 300 after MKL so its
     * team parks (trap b); the timing helpers warm >= 5 ms each. */
    double vns = 0, mns = 0;
    long eng = 0;
    /* The comparator arm is no longer MKL-only: the FFTW arm binds at runtime,
     * so this A/B runs in every build. k1z_time_cmp returns 0 when none is
     * active, which reproduces the old no-MKL behaviour (ns only, ratio 0). */
    if (flip)
    { /* comparator first */
        mns = k1z_time_cmp(N, z0, total);
        cachebust();
        pace(cool_ms);
        {
            const long e0 = vfft_ilfd_mt_passes() + vfft_ztt_mt_passes();
            vns = k1z_time_vfft(h, z0, S, total);
            eng = vfft_ilfd_mt_passes() + vfft_ztt_mt_passes() - e0;
        }
    }
    else
    { /* vfft first */
        {
            const long e0 = vfft_ilfd_mt_passes() + vfft_ztt_mt_passes();
            vns = k1z_time_vfft(h, z0, S, total);
            eng = vfft_ilfd_mt_passes() + vfft_ztt_mt_passes() - e0;
        }
        cachebust();
        pace(cool_ms);
        mns = k1z_time_cmp(N, z0, total);
    }
    /* --k1dir: same cell, backward, same process/buffers/discipline. bwd/fwd
     * is the number that matters -- it is INTERNAL to one run, so it survives
     * the thermal drift that makes cross-run ns incomparable on this host. */
    double bns = 0;
    if (g_k1dir)
        bns = k1z_time_vfft_d(h, z0, S, total, VFFT_BACKWARD);
    double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    double vgf = (vns > 0) ? 5.0 * N * log2((double)N) / vns : 0;
    printf("%-8d %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
           N, plan_s, path, vns, mns, vgf, ratio, rel);
    if (g_k1noop_mt)
        printf("         k1noop-mt N=%-6d T=%d: engaged %ld threaded executes in the timed arm%s\n",
               N, g_mt, eng, eng > 0 ? "" : " (SERIAL verdict or declined)");
    if (g_k1dir)
        printf("         k1dir N=%-6d fwd %10.0f  bwd %10.0f  bwd/fwd %6.3f  "
               "NO_ILBLK=%s\n",
               N, vns, bns, (vns > 0) ? bns / vns : 0.0,
               getenv("VFFT_NO_ILBLK") ? "1" : "0");
    if (out)
    {
        char row[320];
        snprintf(row, sizeof row, "%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%d",
                 N, 1, plan_s, path, vns, mns, vgf, ratio, rel,
                 vfft_plan_route(h), flip);
        if (g_k1noop_mt)   /* the threaded run's own column: threaded executes ENGAGED in the timed arm (0 = serial verdict) */
            snprintf(row + strlen(row), sizeof row - strlen(row), ",%ld", eng);
        strncat(row, "\n", sizeof row - strlen(row) - 1);
        fflush(out);
        if (!k1z_csv_replace(g_csv_path, row))   /* a re-run REPLACES the cell's row */
            fputs(row, out);
        fflush(out);
    }
    free_d(z0);
    free_d(S);
    free_d(rt);
    vfft_destroy(h);
}

#ifdef VFFT_HAS_FLOAT
/* ═════════════════════════════════════════════════════════════════════════
 * --f32 (2026-11-10): THE FP32 ARM of the K=1 natural cells, in a build with
 * VFFT_HAS_FLOAT (gauntlet/build.py --float). A leading flag beside --k1noop
 * (out of place; with --mt the threaded cell) or --k1nat (in place): the same
 * cell on FLOAT data, complex float pairs, through the public FP32 door only
 * (vfftf.h: vfftf_create / vfftf_execute / vfftf_destroy, the route from
 * vfftf_info), against MKL DFTI_SINGLE at the same placement and thread count.
 * The FP64 cell's protocol: the pin and the sibling guard (the threaded cell:
 * the P-cores and >= 5 ms untimed warm), 10 warm-ups, two timing windows of
 * best-of-5 with the idle + re-warm gap between them, cachebust + cool between
 * the engines, the engine order by the flip argument. K > 1 in the K slot is a
 * CONTIGUOUS batch (howmany = K, transform t at complex offset t N; MKL
 * NUMBER_OF_TRANSFORMS = K at distance N); K omitted or 1 is one transform.
 * The rt_err column: our float forward against MKL's DOUBLE forward of the
 * same float input (converted), the elementwise max error over the
 * reference's max magnitude; the FP32 limit is 1e-5, and the line says FAIL
 * above it or on a NaN (a build without MKL: the roundtrip against N x). The
 * csv row has the FP64 cell's columns with plan "z:ilp-f32" and path
 * "nat-oop-f32" / "nat-ip-f32", so it never replaces an FP64 row of a shared
 * csv. Under --mt the engaged column reads -1: vfftf.h has no engagement
 * counter (the line prints the plan's thread count from vfftf_info instead). */
#include "vfftf.h"
#define K1Z_F32_LIMIT 1e-5
static int g_f32 = 0;   /* --f32 */

static float *alloc_f(size_t n)
{
    float *p = (float *)vfft_malloc(n * sizeof(float));
    if (!p)
    {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return p;
}
static void free_f(float *p) { vfft_free(p); }

/* one execute of the cell: in place on the evolving buffer S, or z0 -> S */
#define K1Z_F32_EXEC(h, z0, S) (g_k1zip ? vfftf_execute((h), VFFT_FORWARD, (S), (S)) \
                                        : vfftf_execute((h), VFFT_FORWARD, (z0), (S)))

/* k1z_time_vfft_d's shape on the FP32 plan, forward */
static double k1z_time_vfft_f32(vfftf_plan h, const float *z0, float *S, size_t total)
{
    if (g_k1zip)
        memcpy(S, z0, 2 * total * sizeof(float));
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: the pool and the caller core settle (as the FP64 cell) */
        const double tw = vfft_now_ns();
        do
            vfftf_execute(h, VFFT_FORWARD, z0, S);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            K1Z_F32_EXEC(h, z0, S);
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {   /* k1z_window_gap: the idle, then >= 10 ms of the plan's own work untimed */
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                K1Z_F32_EXEC(h, z0, S);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                K1Z_F32_EXEC(h, z0, S);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    return best;
}

#ifdef VFFT_HAS_MKL
/* the cell's MKL descriptor at precision prec: K transforms at distance N (K = 1: one), in place or not */
static DFTI_DESCRIPTOR_HANDLE k1z_dfti_f32(enum DFTI_CONFIG_VALUE prec, int N, size_t K, int inplace)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, prec, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_PLACEMENT, inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    if (K > 1)
    {
        DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    }
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}

/* k1z_time_mkl's shape on DFTI_SINGLE (in place under --k1nat) */
static double k1z_time_mkl_f32(int N, size_t K, const float *z0, size_t total)
{
    DFTI_DESCRIPTOR_HANDLE d = k1z_dfti_f32(DFTI_SINGLE, N, K, g_k1zip);
    if (!d)
        return 0;
    float *zi = alloc_f(2 * total), *zo = alloc_f(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(float));
#define K1Z_F32_MKL() (g_k1zip ? DftiComputeForward(d, zi) : DftiComputeForward(d, zi, zo))
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: MKL's parked team wakes and settles */
        const double tw = vfft_now_ns();
        do
            DftiComputeForward(d, zi, zo);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            K1Z_F32_MKL();
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                K1Z_F32_MKL();
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                K1Z_F32_MKL();
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
#undef K1Z_F32_MKL
    free_f(zi);
    free_f(zo);
    DftiFreeDescriptor(&d);
    return best;
}

/* max_j |a_j - b_j| / max_j |b_j| over n values; NaN when a holds a NaN */
static double k1z_rel_f32(const float *a, const double *b, size_t n)
{
    double e = 0.0, m = 0.0;
    for (size_t j = 0; j < n; j++)
    {
        const double d = fabs((double)a[j] - b[j]);
        if (d != d)
            return NAN;
        if (d > e) e = d;
        if (fabs(b[j]) > m) m = fabs(b[j]);
    }
    return m > 0 ? e / m : e;
}

/* the rt_err column: our float forward against MKL's DOUBLE forward of the same float input;
 * -1 when MKL cannot make the descriptor */
static double k1z_xerr_f32(vfftf_plan h, int N, size_t K, const float *z0, size_t total)
{
    DFTI_DESCRIPTOR_HANDLE d = k1z_dfti_f32(DFTI_DOUBLE, N, K, 1);
    double *zr, rel;
    float *zv;
    if (!d)
        return -1.0;
    zr = (double *)vfft_malloc(2 * total * sizeof(double));
    zv = alloc_f(2 * total);
    if (!zr)
    {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    for (size_t i = 0; i < 2 * total; i++)
        zr[i] = (double)z0[i];
    DftiComputeForward(d, zr);
    if (g_k1zip)
    {
        memcpy(zv, z0, 2 * total * sizeof(float));
        vfftf_execute(h, VFFT_FORWARD, zv, zv);
    }
    else
        vfftf_execute(h, VFFT_FORWARD, z0, zv);
    rel = k1z_rel_f32(zv, zr, 2 * total);
    vfft_free(zr);
    free_f(zv);
    DftiFreeDescriptor(&d);
    return rel;
}
#endif

/* the comparator of the FP32 cell: MKL DFTI_SINGLE (0 = none in this build) */
static double k1z_time_cmp_f32(int N, size_t K, const float *z0, size_t total)
{
#ifdef VFFT_HAS_MKL
    return k1z_time_mkl_f32(N, K, z0, total);
#else
    (void)N; (void)K; (void)z0; (void)total;
    return 0;
#endif
}

static void run_k1z_cell_f32(int N, size_t K, FILE *out, int cool_ms, int flip)
{
    const char *plan_s = "z:ilp-f32";
    const char *path = g_k1zip ? "nat-ip-f32" : "nat-oop-f32";
    const int T = g_k1noop_mt ? g_mt : 1;
    if (!g_k1noop_mt) bench_pin_one_thread();   /* the one-thread protocol (once per process) */
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d %-16s   SKIP (front-door bundle unavailable)\n", N, plan_s);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = g_k1zip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    cfg.rank = 1;
    cfg.n[0] = N;
    cfg.howmany = K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = K > 1 ? VFFT_BATCH_CONTIGUOUS : VFFT_BATCH_DEFAULT;
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    vfftf_plan h = vfftf_create(&cfg);
    if (!h)
    {
        printf("%-8d %-16s   vfftf_create FAILED: %s\n", N, plan_s, vfft_error());
        return;
    }
    vfft_info_t info;
    memset(&info, 0, sizeof info);
    vfftf_info(h, &info);
    const char *route = (info.route && info.route[0]) ? info.route : "-";

    const size_t total = (size_t)N * K;
    float *z0 = alloc_f(2 * total), *S = alloc_f(2 * total), *rt = alloc_f(2 * total);
    srand(42 + N + 1);   /* the FP64 cell's seed */
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (float)((double)rand() / RAND_MAX - 0.5);

    /* the roundtrip (backward of forward against N x): the column of a build without MKL */
    if (g_k1zip)
    {
        memcpy(rt, z0, 2 * total * sizeof(float));
        vfftf_execute(h, VFFT_FORWARD, rt, rt);
        vfftf_execute(h, VFFT_BACKWARD, rt, rt);
    }
    else
    {
        vfftf_execute(h, VFFT_FORWARD, z0, S);
        vfftf_execute(h, VFFT_BACKWARD, S, rt);
    }
    double rel;
    {
        double e = 0.0, m = 0.0;
        const double inv = 1.0 / (double)N;
        rel = 0.0;
        for (size_t i = 0; i < 2 * total; i++)
        {
            const double d = fabs((double)rt[i] * inv - (double)z0[i]);
            if (d != d) { rel = NAN; break; }
            if (d > e) e = d;
            if (fabs((double)z0[i]) > m) m = fabs((double)z0[i]);
        }
        if (rel == 0.0)
            rel = m > 0 ? e / m : e;
    }
#ifdef VFFT_HAS_MKL
    {   /* the cross-engine column against the double-precision reference (overwrites the roundtrip,
         * as the FP64 cell's MKL compare does) */
        const double x = k1z_xerr_f32(h, N, K, z0, total);
        if (!(x < 0))
            rel = x;
    }
#endif

    double vns = 0, mns = 0;
    if (flip)
    { /* comparator first */
        mns = k1z_time_cmp_f32(N, K, z0, total);
        cachebust();
        pace(cool_ms);
        vns = k1z_time_vfft_f32(h, z0, S, total);
    }
    else
    { /* ours first */
        vns = k1z_time_vfft_f32(h, z0, S, total);
        cachebust();
        pace(cool_ms);
        mns = k1z_time_cmp_f32(N, K, z0, total);
    }
    const double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    const double vgf = (vns > 0) ? 5.0 * N * log2((double)N) * (double)K / vns : 0;
    const int bad = !(rel < K1Z_F32_LIMIT);
    printf("%-8d %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e  K=%d T=%d route=%s%s\n",
           N, plan_s, path, vns, mns, vgf, ratio, rel, (int)K, T, route,
           bad ? "   *** FAIL: above the FP32 limit 1e-5 ***" : "");
    if (g_k1noop_mt)
        printf("         k1noop-mt N=%-6d T=%d: engaged n/a at FP32 (vfftf.h has no counter); the plan runs on %d threads\n",
               N, g_mt, info.nthreads);
    if (out)
    {
        char row[320];
        snprintf(row, sizeof row, "%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%d",
                 N, (int)K, plan_s, path, vns, mns, vgf, ratio, rel, route, flip);
        if (g_k1noop_mt)   /* the threaded run's engaged column: no FP32 counter */
            strncat(row, ",-1", sizeof row - strlen(row) - 1);
        strncat(row, "\n", sizeof row - strlen(row) - 1);
        fflush(out);
        if (!k1z_csv_replace(g_csv_path, row))   /* a re-run REPLACES the cell's row */
            fputs(row, out);
        fflush(out);
    }
    free_f(z0);
    free_f(S);
    free_f(rt);
    vfftf_destroy(h);
}
#endif /* VFFT_HAS_FLOAT */

/* ═════════════════════════════════════════════════
 * --realfwd / --realbwd : the REAL gauntlet cell (2026-09-29): 1D at K=1, 1D
 * at K>1 (transform-contiguous rows: real rows at pitch N, CCE rows at pitch
 * N+2 -- the geometry FFTW's howmany means), and with --2drealnat the 2D
 * shape (N1 in the N slot, N2 in the K slot, N2 of either parity; real plane
 * N1 x N2, CCE plane N1 x (N2/2+1) pairs). The K=1 natural cell's protocol on a VFFT_R2C /
 * VFFT_C2R plan: pin + sibling guard, the front door on the store (OUT OF
 * PLACE, interleaved CCE, natural, one thread), the comparator's spectrum of
 * the same input as the reference (r2c: ours vs it elementwise; c2r: both
 * backwards fed that spectrum, checked against N*x -- never a roundtrip),
 * then the two timers with cachebust + cool between engines, best-of-5 in two
 * windows, the engine order per the driver's flip. Comparator: FFTW (--cmp
 * fftw: plan_dft_r2c_1d / plan_many_dft_r2c / plan_dft_r2c_2d and the c2r
 * twins, FFTW_MEASURE, out of place; FFTW's c2r destroys its input, so it
 * runs on a copy) or MKL (DFTI_REAL, CCE = COMPLEX_COMPLEX, NOT_INPLACE,
 * NUMBER_OF_TRANSFORMS = K with the row distances). GFLOPS = 2.5 T log2 T per
 * transform, T = the points (a real transform is half a complex one). Row =
 * the K=1 cell's columns (N,K,... or N1,N2,...); plan "z:r2c" / "z:c2r"
 * ("z:r2c2d" / "z:c2r2d"), path "nat-oop". Nothing else runs in the process. */
static int g_real = 0;    /* 1 = r2c (forward), 2 = c2r (backward) */
static int g_k2real = 0;  /* --2drealnat: the 2D real cell */
static int g_k3real = 0;  /* --3dreal (2026-10-07): the 3D real cell, the shape N1xN2xN3 in the N slot (fftnd_real_il.h) */
static int g_real_own = 0; /* --realown (2026-10-08): DOOR 2 -- ours on the plan's own plane at the pitch it chooses
                            * (owned_buffers = 1), the comparators at their own in-place layout; the 2D r2c cell; path "nat-ipo" */
static int g_real_ip = 0; /* --realip (2026-10-08): the real cell IN PLACE -- one padded plane, every row 2 (N2/2 + 1)
                           * doubles (FFTW's and MKL's in-place real layout), ours vs the comparator in place on it; the
                           * reference stays the comparator's out-of-place spectrum of the same input. The 2D r2c cell
                           * today (docs/roadmap/real_inplace_design.md); path "nat-ip". */

/* the cell's geometry: nd = 1 or 2; K blocks (1D rows; 2D: one block) of rp
 * reals and cp CCE doubles each */
typedef struct { int nd, N1, N2, N3; size_t K, rp, cp, pts; } real_geo_t;   /* nd = 1, 2 or 3 (2026-10-07) */

typedef void (*real_body_f)(void *ctx);
/* the K=1 cell's timing shape (k1z_time_vfft_d): 10 warm-ups, then TWO windows
 * of best-of-5, the second after the idle + re-warm gap. `total` sizes the
 * reps (reps_for) -- the complex-equivalent points/2 keeps the batch duration
 * on the c2c cells' scale. */
static double real_time_body(real_body_f f, void *ctx, size_t total)
{
    for (int w = 0; w < 10; w++)
        f(ctx);
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                f(ctx);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                f(ctx);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    return best;
}
typedef struct { vfft_plan h; int dir; double *src, *dst; } real_ours_t;
static void real_ours_body(void *v)
{
    real_ours_t *c = (real_ours_t *)v;
    vfft_execute(c->h, c->dir, c->src, c->dst);
}
typedef struct { fftwx_plan p; } real_fx_t;
static void real_fx_body(void *v) { g_fx_arm.execute(((real_fx_t *)v)->p); }
/* FFTW's plan for the geometry, over the caller's planes (planned BEFORE the
 * planes are filled: MEASURE scribbles), at the cell's thread count */
static fftwx_plan real_fx_plan(const real_geo_t *g, int c2r, double *a, double *b)
{
    const int N = g->N1, K = (int)g->K;
    fftwx_plan p;
    if (!fftw_arm_threads(g_k1noop_mt ? g_mt : 1))
        return NULL;
    if (g->nd == 3)
        p = c2r ? g_fx_arm.plan_dft_c2r_3d(g->N1, g->N2, g->N3, (fftwx_complex *)a, b, FFTWX_MEASURE)
                : g_fx_arm.plan_dft_r2c_3d(g->N1, g->N2, g->N3, a, (fftwx_complex *)b, FFTWX_MEASURE);
    else if (g->nd == 2)
        p = c2r ? g_fx_arm.plan_dft_c2r_2d(g->N1, g->N2, (fftwx_complex *)a, b, FFTWX_MEASURE)
                : g_fx_arm.plan_dft_r2c_2d(g->N1, g->N2, a, (fftwx_complex *)b, FFTWX_MEASURE);
    else if (K > 1)
        p = c2r ? g_fx_arm.plan_many_dft_c2r(1, &N, K, (fftwx_complex *)a, NULL, 1, (int)(g->cp / 2),
                                             b, NULL, 1, (int)g->rp, FFTWX_MEASURE)
                : g_fx_arm.plan_many_dft_r2c(1, &N, K, a, NULL, 1, (int)g->rp,
                                             (fftwx_complex *)b, NULL, 1, (int)(g->cp / 2), FFTWX_MEASURE);
    else
        p = c2r ? g_fx_arm.plan_dft_c2r_1d(N, (fftwx_complex *)a, b, FFTWX_MEASURE)
                : g_fx_arm.plan_dft_r2c_1d(N, a, (fftwx_complex *)b, FFTWX_MEASURE);
    if (p)
        fftw_wis_export();
    return p;
}
#ifdef VFFT_HAS_MKL
typedef struct { DFTI_DESCRIPTOR_HANDLE d; double *a, *b; int c2r; } real_mk_t;
static void real_mk_body(void *v)
{
    real_mk_t *c = (real_mk_t *)v;
    if (c->c2r) DftiComputeBackward(c->d, c->a, c->b);
    else        DftiComputeForward(c->d, c->a, c->b);
}
/* MKL's descriptor for the geometry, PER DIRECTION: this MKL applies the
 * distances and strides to the arrays handed to the compute call (the
 * --c2r cell's precedent: a backward descriptor with the distances
 * swapped), so c2r gets its own. The 2D CCE strides are set explicitly:
 * MKL's default output pitch is N2 complex, not N2/2+1 (measured
 * 2026-09-29), and the explicit {0, N2/2+1, 1} is
 * the layout FFTW and we write. */
static DFTI_DESCRIPTOR_HANDLE real_mk_desc(const real_geo_t *g, int c2r, int ip)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[3] = { g->N1, g->N2, g->N3 };
    const MKL_LONG rc = g->nd == 3 ? DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 3, dims)
                      : g->nd == 2 ? DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 2, dims)
                                   : DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)g->N1);
    if (rc != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_PLACEMENT, ip ? DFTI_INPLACE : DFTI_NOT_INPLACE);   /* --realip: one padded plane */
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (g->nd == 3)
    {   /* the 3D CCE strides, as the 2D ones: the explicit {0, N2 hp3, hp3, 1} is the layout FFTW and we write;
         * in place the real rows sit at the padded pitch 2 hp3 */
        const MKL_LONG hp3 = g->N3 / 2 + 1, rp3 = ip ? 2 * hp3 : g->N3;
        MKL_LONG rs[4] = { 0, (MKL_LONG)g->N2 * rp3, rp3, 1 }, cs[4] = { 0, (MKL_LONG)g->N2 * hp3, hp3, 1 };
        DftiSetValue(d, DFTI_INPUT_STRIDES, c2r ? cs : rs);
        DftiSetValue(d, DFTI_OUTPUT_STRIDES, c2r ? rs : cs);
    }
    else if (g->nd == 2)
    {   /* in place the real rows sit at the padded pitch 2 (N2/2 + 1) */
        MKL_LONG rs[3] = { 0, ip ? 2 * (g->N2 / 2 + 1) : g->N2, 1 }, cs[3] = { 0, g->N2 / 2 + 1, 1 };
        DftiSetValue(d, DFTI_INPUT_STRIDES, c2r ? cs : rs);
        DftiSetValue(d, DFTI_OUTPUT_STRIDES, c2r ? rs : cs);
    }
    else if (g->K > 1)
    {
        const MKL_LONG rd = (MKL_LONG)g->rp, cd = (MKL_LONG)(g->cp / 2);   /* reals, complex */
        DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)g->K);
        DftiSetValue(d, DFTI_INPUT_DISTANCE, c2r ? cd : rd);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, c2r ? rd : cd);
    }
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
#endif

/* the comparator's CCE spectrum of the real planes x into ref; 1 on success */
static int real_ref_spectrum(const real_geo_t *g, const double *x, double *ref)
{
    const size_t rn = g->K * g->rp + 2, cn = g->K * g->cp;
#ifdef VFFT_HAS_KFR
    if (g_cmp_kfr)
    {   /* --cmp kfr (2026-10-10): KFR's own r2c spectrum, the 1D K=1 cell (run_real_cell admits no other) */
        void *p = kfr_r2c_create(g->N1);
        size_t tsz;
        unsigned char *temp;
        double *xi, *zo;
        if (!p)
            return 0;
        tsz = kfr_r2c_temp_size(p);
        temp = tsz ? (unsigned char *)alloc_d((tsz + 7) / 8) : NULL;
        xi = alloc_d(rn);
        zo = alloc_d(cn);
        memcpy(xi, x, rn * sizeof(double));
        kfr_r2c_forward(p, xi, zo, temp);
        memcpy(ref, zo, cn * sizeof(double));
        free_d(xi);
        free_d(zo);
        if (temp) free_d((double *)temp);
        kfr_r2c_destroy(p);
        return 1;
    }
#endif
    if (g_cmp_fftw)
    {
        if (!fftw_arm_bind())
            return 0;
        double *xi = alloc_d(rn), *zo = alloc_d(cn);
        fftwx_plan p = real_fx_plan(g, 0, xi, zo);
        if (!p)
        {
            free_d(xi);
            free_d(zo);
            return 0;
        }
        memcpy(xi, x, rn * sizeof(double));   /* AFTER planning: MEASURE scribbles */
        g_fx_arm.execute(p);
        memcpy(ref, zo, cn * sizeof(double));
        g_fx_arm.destroy_plan(p);
        free_d(xi);
        free_d(zo);
        return 1;
    }
#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE d = real_mk_desc(g, 0, 0);   /* the reference: out of place, from the tight input */
        if (!d)
            return 0;
        double *xi = alloc_d(rn);
        memcpy(xi, x, rn * sizeof(double));
        DftiComputeForward(d, xi, ref);
        DftiFreeDescriptor(&d);
        free_d(xi);
        return 1;
    }
#else
    return 0;
#endif
}

/* the comparator's timed arm: its own plan, built and destroyed outside the
 * timed windows; the input seeded once (a dense transform is data-oblivious
 * after FFTW's c2r has consumed it) */
#ifdef VFFT_HAS_KFR
typedef struct { void *p; const double *in; double *out; unsigned char *temp; int c2r; } real_kfr_t;
static void real_kfr_body(void *v)
{
    real_kfr_t *c = (real_kfr_t *)v;
    if (c->c2r)
        kfr_c2r_backward(c->p, c->in, c->out, c->temp);
    else
        kfr_r2c_forward(c->p, c->in, c->out, c->temp);
}
#endif
static double real_time_cmp(const real_geo_t *g, int c2r, const double *x, const double *ref)
{
    const size_t rn = g->K * g->rp + 2, cn = g->K * g->cp, tot = g->pts / 2;
    /* --realip: x is the padded plane and the arm's one plane holds it (a == b) */
    const size_t an = g_real_ip ? cn : (c2r ? cn : rn), bn = c2r ? rn : cn;
#ifdef VFFT_HAS_KFR
    if (g_cmp_kfr)
    {   /* --cmp kfr (2026-10-10): KFR's real plan, built and destroyed outside the timed windows;
         * r2c from the reals, c2r (the same plan backward) from the CCE spectrum */
        real_kfr_t c;
        size_t tsz;
        double ns;
        c.p = kfr_r2c_create(g->N1);
        if (!c.p)
            return 0;
        c.c2r = c2r;
        tsz = kfr_r2c_temp_size(c.p);
        c.temp = tsz ? (unsigned char *)alloc_d((tsz + 7) / 8) : NULL;
        {
            double *xi = alloc_d(c2r ? cn : rn);
            memcpy(xi, c2r ? ref : x, (c2r ? cn : rn) * sizeof(double));
            c.in = xi;
            c.out = alloc_d(c2r ? rn : cn);
            if (c2r)
            {   /* the comparator does the transform it claims: once, before timing, KFR's c2r of the
                 * spectrum against N x (unnormalized, as ours) */
                double w = 0, m = 0;
                size_t i;
                real_kfr_body(&c);
                for (i = 0; i < (size_t)g->N1; i++)
                {
                    const double d = fabs(c.out[i] - (double)g->N1 * x[i]), q = fabs(x[i]);
                    if (d > w) w = d;
                    if (q > m) m = q;
                }
                if (!(m > 0 && w / ((double)g->N1 * m) < 1e-10))
                    printf("         KFR c2r N=%d: its output disagrees with N x (rel %.2e) -- the timing stands, the comparator is suspect\n",
                           g->N1, m > 0 ? w / ((double)g->N1 * m) : w);
                memcpy(xi, ref, cn * sizeof(double));
            }
            ns = real_time_body(real_kfr_body, &c, tot);
            free_d(xi);
        }
        free_d(c.out);
        if (c.temp) free_d((double *)c.temp);
        kfr_r2c_destroy(c.p);
        return ns;
    }
#endif
    if (g_cmp_fftw)
    {
        if (!fftw_arm_bind())
            return 0;
        double *a = alloc_d(an), *b = g_real_ip ? a : alloc_d(bn);
        fftwx_plan p = real_fx_plan(g, c2r, a, b);
        if (!p)
        {
            free_d(a);
            if (!g_real_ip) free_d(b);
            return 0;
        }
        memcpy(a, c2r ? ref : x, an * sizeof(double));   /* after planning */
        real_fx_t c = { p };
        double ns = real_time_body(real_fx_body, &c, tot);
        g_fx_arm.destroy_plan(p);
        free_d(a);
        if (!g_real_ip) free_d(b);
        return ns;
    }
#ifdef VFFT_HAS_MKL
    {
        real_mk_t c;
        c.d = real_mk_desc(g, c2r, g_real_ip);
        if (!c.d)
            return 0;
        c.a = alloc_d(an);
        c.b = g_real_ip ? c.a : alloc_d(bn);
        c.c2r = c2r;
        memcpy(c.a, c2r ? ref : x, an * sizeof(double));
        double ns = real_time_body(real_mk_body, &c, tot);
        DftiFreeDescriptor(&c.d);
        free_d(c.a);
        if (!g_real_ip) free_d(c.b);
        return ns;
    }
#else
    return 0;
#endif
}

static void run_real_cell(const real_geo_t *g, FILE *out, int cool_ms, int flip)
{
    const int c2r = (g_real == 2);
    const size_t rn = g->K * g->rp + 2, cn = g->K * g->cp, tot = g->pts / 2;
    const char *plan_s = g->nd == 3 ? (c2r ? "z:c2r3d" : "z:r2c3d") : g->nd == 2 ? (c2r ? "z:c2r2d" : "z:r2c2d") : (c2r ? "z:c2r" : "z:r2c");
    const char *path = g_real_own ? "nat-ipo" : g_real_ip ? "nat-ip" : "nat-oop";
    char shape[48];
    if (g->nd == 3)      snprintf(shape, sizeof shape, "%dx%dx%d", g->N1, g->N2, g->N3);
    else if (g->nd == 2) snprintf(shape, sizeof shape, "%dx%d", g->N1, g->N2);
    else                 snprintf(shape, sizeof shape, "%d", g->N1);
    if (g_real_ip && g->nd != 2 && g->nd != 3)
    {   /* --realip / --realown: the 2D and 3D cells, both directions (real_inplace_design.md) */
        printf("%-8s %-16s   SKIP (in place: the 2D and 3D cells)\n", shape, plan_s);
        return;
    }
#ifdef VFFT_HAS_KFR
    if (g_cmp_kfr && (g->nd != 1 || g->K != 1 || g_real_ip || (g->N1 & 1)))
    {   /* --cmp kfr (2026-10-10): the 1D r2c and c2r cells, K = 1, out of place, an even N (KFR's real DFT is even-only) */
        printf("%-8s %-16s   SKIP (--cmp kfr: the 1D real cell at an even N, K=1, out of place)\n", shape, plan_s);
        return;
    }
#endif
    if (!g_k1noop_mt) bench_pin_one_thread();   /* --mt: the threaded cell's two-team protocol (main) */
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%-8s %-16s   SKIP (front-door bundle unavailable)\n", shape, plan_s);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = g_real_ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;   /* --realip (2026-10-08) */
    cfg.flags |= (g_real_own) ? VFFT_FLAG_OWN_PITCH : 0;   /* --realown: door 2, the plan's own plane at its pitch */
    cfg.rank = g->nd;
    cfg.n[0] = g->N1;
    cfg.n[1] = g->nd >= 2 ? g->N2 : 0;
    cfg.n[2] = g->nd == 3 ? g->N3 : 0;
    cfg.howmany = g->K;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = g->K > 1 ? VFFT_BATCH_CONTIGUOUS : VFFT_BATCH_DEFAULT;
    cfg.order = VFFT_ORDER_NATURAL;   /* a real spectrum is natural */
    cfg.nthreads = g_k1noop_mt ? g_mt : 1;   /* --mt: the plan's thread snapshot; the comparator gets the same T */
    cfg.wisdom_dir = W;   /* wisdom_write 0: a bench never banks */
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8s %-16s   vfft_create FAILED\n", shape, plan_s);
        return;
    }
    double *x = alloc_d(rn), *ref = alloc_d(cn), *o = alloc_d(cn > rn ? cn : rn), *src = alloc_d(cn > rn ? cn : rn);
    srand(42 + g->N1 + 1);
    for (size_t i = 0; i < rn; i++)
        x[i] = 0.0;
    for (size_t t = 0; t < g->K; t++)
        for (size_t i = 0; i < g->rp; i++)   /* the points of one transform at any rank */
            x[t * g->rp + i] = (double)rand() / RAND_MAX - 0.5;
    const int haveref = real_ref_spectrum(g, x, ref);
    /* --realip: the rows of the last axis at the padded pitch 2 (N/2 + 1): the real input laid so (xp),
     * the CCE spectrum already is; a c2r's real output comes back at that pitch */
    const size_t ip_rows = g->nd == 3 ? (size_t)g->N1 * (size_t)g->N2 : (size_t)g->N1;
    const size_t ip_last = (size_t)(g->nd == 3 ? g->N3 : g->N2), ip_P = 2 * (ip_last / 2 + 1);
    double *xp = NULL;
    if (g_real_ip)
    {
        xp = alloc_d(cn);
        memset(xp, 0, cn * sizeof(double));
        for (size_t r = 0; r < ip_rows; r++)
            memcpy(xp + r * ip_P, x + r * ip_last, ip_last * sizeof(double));
        memcpy(src, c2r ? ref : xp, cn * sizeof(double));
    }
    else
        memcpy(src, c2r ? ref : x, (c2r ? cn : rn) * sizeof(double));
    for (size_t i = 0; i < (cn > rn ? cn : rn); i++)
        o[i] = -1.0e300;   /* poison: a refused execute leaves it */
    /* --realown: the plan's own plane and the input laid at its pitch */
    double *own = NULL, *xo = NULL;
    size_t own_P = 0, own_n = 0;
    if (g_real_own)
    {
        {   /* the plan's pitch and its plane's size (vfft_info); the plane is the caller's */
            vfft_info_t inf;
            if (vfft_info(h, &inf) == 0 && inf.in_doubles)
            {
                own_P = inf.stride;
                own = alloc_d(inf.in_doubles);
                memset(own, 0, inf.in_doubles * sizeof(double));
            }
        }
        if (!own || own_P < ip_P)
        {
            printf("%-8s %-16s   door 2: no plane at the plan's pitch (pitch %zu)\n", shape, plan_s, own_P);
            free_d(x); free_d(ref); free_d(o); free_d(src);
            if (xp) free_d(xp);
            if (own) free_d(own);
            vfft_destroy(h);
            return;
        }
        own_n = ip_rows * own_P;
        xo = alloc_d(own_n);
        memset(xo, 0, own_n * sizeof(double));
        for (size_t r = 0; r < ip_rows; r++)   /* the input rows at the plan's pitch: the reals, or the spectrum's rows */
            if (c2r)
                memcpy(xo + r * own_P, ref + r * ip_P, ip_P * sizeof(double));
            else
                memcpy(xo + r * own_P, x + r * ip_last, ip_last * sizeof(double));
        memcpy(own, xo, own_n * sizeof(double));
    }
    if (g_real_ip)
    {   /* in place: the volume transforms onto itself; r2c's CCE rows are the comparator's layout,
         * c2r's real rows come back at the padded pitch and are laid tight for the check; door 2's
         * rows come back at the plan's pitch and are laid at the comparator's */
        const double *in0 = g_real_own ? xo : (c2r ? ref : xp);
        double *buf = g_real_own ? own : src;
        const size_t bn = g_real_own ? own_n : cn;
        vfft_execute(h, c2r ? VFFT_BACKWARD : VFFT_FORWARD, buf, buf);
        if (memcmp(buf, in0, bn * sizeof(double)) != 0)   /* a refused execute leaves the input, and the poison */
        {
            if (g_real_own && c2r)
                for (size_t r = 0; r < ip_rows; r++)
                    memcpy(o + r * ip_last, buf + r * own_P, ip_last * sizeof(double));   /* the real rows, tight for the check */
            else if (g_real_own)
                for (size_t r = 0; r < ip_rows; r++)
                    memcpy(o + r * ip_P, buf + r * own_P, ip_P * sizeof(double));
            else if (c2r)
                for (size_t r = 0; r < ip_rows; r++)
                    memcpy(o + r * ip_last, src + r * ip_P, ip_last * sizeof(double));
            else
                memcpy(o, src, cn * sizeof(double));
        }
        memcpy(buf, in0, bn * sizeof(double));   /* the timed arm starts from the input */
    }
    else
        vfft_execute(h, c2r ? VFFT_BACKWARD : VFFT_FORWARD, src, o);
    if (o[0] == -1.0e300)
    {
        printf("%-8s %-16s   execute REFUSED\n", shape, plan_s);
        free_d(x); free_d(ref); free_d(o); free_d(src);
        if (xp) free_d(xp);
        if (xo) free_d(xo);
        if (own) free_d(own);
        vfft_destroy(h);
        return;
    }
    double rel = -1.0;
    if (haveref)
    {
        double w = 0, m = 0;
        const size_t pts1 = g->pts / g->K;   /* points per transform */
        if (c2r)
        {   /* backward vs pts1 * x, block by block */
            for (size_t t = 0; t < g->K; t++)
                for (size_t i = 0; i < pts1; i++)
                {
                    double d = fabs(o[t * g->rp + i] - (double)pts1 * x[t * g->rp + i]), q = fabs(x[t * g->rp + i]);
                    if (d > w) w = d;
                    if (q > m) m = q;
                }
            rel = m > 0 ? w / ((double)pts1 * m) : w;
        }
        else
        {   /* forward vs the comparator's CCE planes */
            for (size_t i = 0; i < cn; i++)
            {
                double d = fabs(o[i] - ref[i]), q = fabs(ref[i]);
                if (d > w) w = d;
                if (q > m) m = q;
            }
            rel = m > 0 ? w / m : w;
        }
    }
    double vns = 0, mns = 0;
    long eng = 0;
    real_ours_t oc = { h, c2r ? VFFT_BACKWARD : VFFT_FORWARD, g_real_own ? own : src,
                       g_real_own ? own : (g_real_ip ? src : o) };   /* in place: the one plane; door 2: the plan's */
/* the threaded executes engaged in the timed arm: the child's (ZTURN-T, the
 * four-step's 2D tier, the flat DIT), the real four-step's sweeps, the batch's
 * worker dispatches */
long vfft_il2d_row_mt_passes(void);   /* vfft.c: the 2D real row pass's threaded dispatches (not in the public header) */
#define REAL_ENG() (vfft_ztt_mt_passes() + vfft_il2d_col_mt_passes() + vfft_ilfd_mt_passes() + vfft_zfsr_mt_passes() + vfft_zr2c_fold_mt_passes() + vfft_zttr_mt_passes() + vfft_zrf_mt_passes() + vfft_tc_mt_dispatches() + vfft_ilnd_mt_passes() + vfft_il2d_row_mt_passes())   /* + the rank-3 tiers' counter (the 3D real cell, 2026-10-07); + the 2D real row pass's (the in-place door route, 2026-10-08) */
    if (flip)
    { /* comparator first; --mt: the two-team protocol of the K=1 c2c cell (our pool down while the comparator runs) */
        mns = real_time_cmp(g, c2r, g_real_ip ? xp : x, ref);
        cachebust();
        pace(cool_ms);
        {
            const long e0 = REAL_ENG();
            vns = real_time_body(real_ours_body, &oc, tot);
            eng = REAL_ENG() - e0;
        }
    }
    else
    {
        {
            const long e0 = REAL_ENG();
            vns = real_time_body(real_ours_body, &oc, tot);
            eng = REAL_ENG() - e0;
        }
        cachebust();
        pace(cool_ms);
        mns = real_time_cmp(g, c2r, g_real_ip ? xp : x, ref);
    }
#undef REAL_ENG
    if (g_k1noop_mt)
        printf("         real-mt %s T=%d: engaged %ld threaded executes in the timed arm%s\n",
               shape, g_mt, eng, eng > 0 ? "" : " (SERIAL verdict or declined)");
    double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    const double pts1 = (double)(g->pts / g->K);
    double vgf = (vns > 0) ? (double)g->K * 2.5 * pts1 * log2(pts1) / vns : 0;
    printf("%-8s %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
           shape, plan_s, path, vns, mns, vgf, ratio, rel);
    if (out)
    {
        char row[320];
        if (g->nd == 3 && g_k1noop_mt)   /* the threaded cell's engaged column before flip, as the 3D c2c row's */
            snprintf(row, sizeof row, "%d,%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%ld,%d\n",
                     g->N1, g->N2, g->N3, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), eng, flip);
        else if (g->nd == 3)
            snprintf(row, sizeof row, "%d,%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%d\n",
                     g->N1, g->N2, g->N3, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), flip);
        else if (g->nd == 2 && g_k1noop_mt)   /* the threaded 2D real cell (2026-10-08): engaged before flip, as the 3D row's */
            snprintf(row, sizeof row, "%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%ld,%d\n",
                     g->N1, g->N2, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), eng, flip);
        else if (g->nd == 2)
            snprintf(row, sizeof row, "%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%d\n",
                     g->N1, g->N2, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), flip);
        else if (g_k1noop_mt)   /* the threaded run's own column, as the K=1 c2c cell's */
            snprintf(row, sizeof row, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%d,%ld\n",
                     g->N1, g->K, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), flip, eng);
        else
            snprintf(row, sizeof row, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%d\n",
                     g->N1, g->K, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), flip);
        fflush(out);
        if (!k1z_csv_replace(g_csv_path, row))   /* a re-run REPLACES the cell's row */
            fputs(row, out);
        fflush(out);
    }
    free_d(x); free_d(ref); free_d(o); free_d(src);
    if (xp) free_d(xp);
    if (xo) free_d(xo);
    if (own) free_d(own);
    vfft_destroy(h);
}

/* ═════════════════════════════════════════════════
 * --2dilnat : the 2D INTERLEAVED gauntlet cell (2026-09-23). The K=1 natural
 * cell's protocol (run_k1z_cell) on a dims=2 plan: pin + sibling guard, the
 * front door on the store (OOP, natural, interleaved, K=1, one thread),
 * roundtrip gate bwd(fwd(x)) == N1*N2*x, then the CROSS-ENGINE elementwise
 * check against the comparator's 2D plan (MKL DFTI 2D, or FFTW's under --cmp
 * fftw; out of place, natural on both sides), then the two timers with
 * cachebust + cool between engines, best-of-5 in two windows, the engine
 * order per the driver's flip. GFLOPS = 5 T log2 T, T = N1*N2.
 * Row: N1,N2,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,flip
 * -- the 1D gauntlet's columns with (N1, N2) in place of (N, K), so the
 * row-replace key (first four fields + flip) and the driver's readers hold.
 * ═════════════════════════════════════════════════ */
#ifdef VFFT_HAS_MKL
static double k2z_time_mkl(int N1, int N2, int N3, const double *z0, size_t total)
{   /* N3 = 0: the 2D cell; N3 > 0: the 3D cell (2026-09-24) */
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[3];
    dims[0] = N1;
    dims[1] = N2;
    dims[2] = N3;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, N3 > 0 ? 3 : 2, dims) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: MKL's parked team wakes and settles (the threaded cell, 2026-09-24) */
        const double tw = vfft_now_ns();
        do
            DftiComputeForward(d, zi, zo);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                DftiComputeForward(d, zi, zo);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                DftiComputeForward(d, zi, zo);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}
#endif
#ifdef VFFT_HAS_KFR
/* KFR's 2D plan (N3 = 0) or 3D plan (N3 > 0), the shapes of the 2D/3D cell */
static void *k2z_kfr_create(int N1, int N2, int N3)
{
    return N3 > 0 ? kfr_c2c3d_create(N1, N2, N3) : kfr_c2c2d_create(N1, N2);
}

/* the KFR twin of k2z_time_mkl's one-thread path (2026-10-10): KFR's 2D or 3D
 * plan, the same windows, trials, reps and pacing (main refuses --cmp kfr
 * with --mt) */
static double k2z_time_kfr(int N1, int N2, int N3, const double *z0, size_t total)
{
    void *p = k2z_kfr_create(N1, N2, N3);
    if (!p)
        return 0;
    size_t tsz = kfr_c2c_temp_size(p);
    unsigned char *temp = tsz ? (unsigned char *)alloc_d((tsz + 7) / 8) : NULL;
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        kfr_c2c_forward(p, zi, zo, temp);
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                kfr_c2c_forward(p, zi, zo, temp);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                kfr_c2c_forward(p, zi, zo, temp);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    free_d(zi);
    free_d(zo);
    if (temp) free_d((double *)temp);
    kfr_c2c_destroy(p);
    return best;
}

/* the cross-engine check against KFR's 2D/3D spectrum: max |S - KFR| / max |KFR|,
 * or -1 when KFR refuses the shape */
static double k2z_kfr_xerr(int N1, int N2, int N3, const double *z0, const double *S, size_t total)
{
    void *p = k2z_kfr_create(N1, N2, N3);
    if (!p)
        return -1;
    size_t tsz = kfr_c2c_temp_size(p);
    unsigned char *temp = tsz ? (unsigned char *)alloc_d((tsz + 7) / 8) : NULL;
    double *zk = alloc_d(2 * total);
    kfr_c2c_forward(p, z0, zk, temp);
    double xe = 0.0, xm = 0.0;
    for (size_t i = 0; i < 2 * total; i++)
    {
        double e = fabs(S[i] - zk[i]), m = fabs(zk[i]);
        if (e > xe) xe = e;
        if (m > xm) xm = m;
    }
    free_d(zk);
    if (temp) free_d((double *)temp);
    kfr_c2c_destroy(p);
    return xm > 0 ? xe / xm : xe;
}
#endif
/* the FFTW twin of k2z_time_mkl (2026-10-19): fftw_plan_dft at rank 2 (N3 = 0)
 * or 3, out of place, FFTWX_MEASURE, the plan built once outside the timed
 * windows and $VFFT_FFTW_WIS re-exported after it; the input copied AFTER
 * planning (MEASURE scribbles); the threaded cell's >= 5 ms untimed warm, then
 * the same windows, trials, reps and pacing. --mt: planned at T
 * (fftw_arm_threads), FFTW's workers spawned at the first threaded execute
 * under the P-core mask main set. */
static double k2z_time_fftw(int N1, int N2, int N3, const double *z0, size_t total)
{
    int n[3];
    n[0] = N1;
    n[1] = N2;
    n[2] = N3;
    if (!fftw_arm_threads(g_k1noop_mt ? g_mt : 1))
        return 0;
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    fftwx_plan p = g_fx_arm.plan_dft(N3 > 0 ? 3 : 2, n, (fftwx_complex *)zi, (fftwx_complex *)zo,
                                     FFTWX_FORWARD, FFTWX_MEASURE);
    if (!p) { free_d(zi); free_d(zo); return 0; }
    fftw_wis_export();
    memcpy(zi, z0, 2 * total * sizeof(double));   /* AFTER planning: MEASURE scribbles */
    if (g_k1noop_mt)
    {   /* >= 5 ms untimed: FFTW's team comes up and settles, as MKL's does */
        const double tw = vfft_now_ns();
        do
            g_fx_arm.execute(p);
        while (vfft_now_ns() - tw < 5e6);
    }
    else
        for (int w = 0; w < 10; w++)
            g_fx_arm.execute(p);
    int reps = reps_for(total);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                g_fx_arm.execute(p);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                g_fx_arm.execute(p);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    g_fx_arm.destroy_plan(p);
    free_d(zi);
    free_d(zo);
    return best;
}

/* the cross-engine check against FFTW's 2D/3D spectrum: max |S - FFTW| / max |FFTW|,
 * or -1 when the DLL did not bind or FFTW cannot plan the shape (planned on empty
 * planes, the input copied after: MEASURE scribbles) */
static double k2z_fftw_xerr(int N1, int N2, int N3, const double *z0, const double *S, size_t total)
{
    int n[3];
    n[0] = N1;
    n[1] = N2;
    n[2] = N3;
    if (!fftw_arm_threads(g_k1noop_mt ? g_mt : 1))
        return -1;
    double *zi = alloc_d(2 * total), *zf = alloc_d(2 * total);
    fftwx_plan p = g_fx_arm.plan_dft(N3 > 0 ? 3 : 2, n, (fftwx_complex *)zi, (fftwx_complex *)zf,
                                     FFTWX_FORWARD, FFTWX_MEASURE);
    if (!p) { free_d(zi); free_d(zf); return -1; }
    fftw_wis_export();
    memcpy(zi, z0, 2 * total * sizeof(double));
    g_fx_arm.execute(p);
    double xe = 0.0, xm = 0.0;
    for (size_t i = 0; i < 2 * total; i++)
    {
        double e = fabs(S[i] - zf[i]), m = fabs(zf[i]);
        if (e > xe) xe = e;
        if (m > xm) xm = m;
    }
    g_fx_arm.destroy_plan(p);
    free_d(zi);
    free_d(zf);
    return xm > 0 ? xe / xm : xe;
}

/* THE 2D/3D cell's comparator entry: FFTW under --cmp fftw (no build flag, so
 * a bench built without MKL has a comparator here too, as the K=1 cell's
 * k1z_time_cmp does), KFR's 2D/3D plan under --cmp kfr (2026-10-10), else MKL.
 * Returns 0 when none is active, which the caller reports as "no comparator". */
static double k2z_time_cmp(int N1, int N2, int N3, const double *z0, size_t total)
{
    if (g_cmp_fftw)
        return k2z_time_fftw(N1, N2, N3, z0, total);
#ifdef VFFT_HAS_KFR
    if (g_cmp_kfr)
        return k2z_time_kfr(N1, N2, N3, z0, total);
#endif
#ifdef VFFT_HAS_MKL
    return k2z_time_mkl(N1, N2, N3, z0, total);
#else
    (void)N1; (void)N2; (void)N3; (void)z0; (void)total;
    return 0;
#endif
}

static void run_k2z_cell(int N1, int N2, int N3, FILE *out, int cool_ms, int flip)
{   /* N3 = 0: the 2D cell (N2 rides the K argument); N3 > 0: the 3D cell,
     * the same protocol on a dims=3 plan against MKL DFTI 3D (2026-09-24) */
    const int nd = N3 > 0 ? 3 : 2;
    const char *plan_s = nd == 3 ? "z:il3d" : "z:il2d", *path = "nat-oop";
    char shape[48];
    if (nd == 3)
        snprintf(shape, sizeof shape, "%dx%dx%d", N1, N2, N3);
    else
        snprintf(shape, sizeof shape, "%dx%d", N1, N2);
    if (N1 < 2 || N2 < 2 || (nd == 3 && N3 < 2))
    {
        printf("%s: a %dD cell needs every axis >= 2\n", shape, nd);
        return;
    }
    const int T = g_k1noop_mt ? g_mt : 1;   /* --mt: the threaded cell (2026-09-24); main's
                                              * threaded-cell setup already confined the process
                                              * to the eight P-cores, warmed MKL's team before the
                                              * caller's pin and left this TU's pool unsized */
    if (T == 1)
        bench_pin_one_thread();   /* the one-thread protocol + the sibling guard */
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%s %-8s   SKIP (front-door bundle unavailable)\n", shape, plan_s);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = nd;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.n[2] = nd == 3 ? N3 : 0;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%s %-8s   vfft_create FAILED\n", shape, plan_s);
        return;
    }
    size_t total = (size_t)N1 * (size_t)N2 * (size_t)(nd == 3 ? N3 : 1);
    g_cell_engaged = -1;
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total), *rt = alloc_d(2 * total);
    srand(42 + 131 * N1 + N2 + 7919 * N3);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;
    /* roundtrip gate through the API */
    vfft_execute(h, VFFT_FORWARD, z0, S);
    vfft_execute(h, VFFT_BACKWARD, S, rt);
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)total;
    for (size_t i = 0; i < 2 * total; i++)
    {
        double e = fabs(rt[i] * inv - z0[i]), m = fabs(z0[i]);
        if (e > maxerr) maxerr = e;
        if (m > maxmag) maxmag = m;
    }
    double rel = maxmag > 0 ? maxerr / maxmag : maxerr;
    if (g_cmp_fftw)
    {   /* the comparator's spectrum: FFTW's plan of the shape (2026-10-19); the
         * roundtrip error stays when FFTW cannot plan it */
        double xr = k2z_fftw_xerr(N1, N2, N3, z0, S, total);
        if (xr >= 0)
            rel = xr;
    }
#ifdef VFFT_HAS_KFR
    else if (g_cmp_kfr)
    {   /* the comparator's spectrum: KFR's 2D/3D plan (2026-10-10); the roundtrip
         * error stays when KFR refuses the shape */
        double xr = k2z_kfr_xerr(N1, N2, N3, z0, S, total);
        if (xr >= 0)
            rel = xr;
    }
#endif
#ifdef VFFT_HAS_MKL
    else
    {   /* the correctness column is the CROSS-ENGINE elementwise compare:
         * both engines natural, same input, same spectrum (as --k1nat) */
        DFTI_DESCRIPTOR_HANDLE d = NULL;
        MKL_LONG dims[3];
        dims[0] = N1;
        dims[1] = N2;
        dims[2] = N3;
        if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, nd, dims) == DFTI_NO_ERROR)
        {
            DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            if (DftiCommitDescriptor(d) == DFTI_NO_ERROR)
            {
                double *zm = alloc_d(2 * total);
                DftiComputeForward(d, z0, zm);
                double xe = 0.0, xm = 0.0;
                for (size_t i = 0; i < 2 * total; i++)
                {
                    double e = fabs(S[i] - zm[i]), m = fabs(zm[i]);
                    if (e > xe) xe = e;
                    if (m > xm) xm = m;
                }
                rel = xm > 0 ? xe / xm : xe;
                free_d(zm);
            }
            DftiFreeDescriptor(&d);
        }
    }
#endif
    double vns = 0, mns = 0;
    /* the engagement proof (mt_results_need_engagement_proof): the tier's MT
     * counter over the timed executes; a threaded plan whose counter did not
     * move ran SERIAL, and its row must say so */
#define K2Z_ENG() ((nd == 3) ? vfft_ilnd_mt_passes() : vfft_il2d_col_mt_passes())
    /* the A/B runs in every build, as the K=1 cell's does: the FFTW arm binds
     * at runtime, and k2z_time_cmp returns 0 when no comparator is active
     * (ns only, ratio 0, as a no-MKL build reported before) */
    if (flip)
    {
        mns = k2z_time_cmp(N1, N2, N3, z0, total);
        cachebust();
        pace(cool_ms);
        {
            const long e0 = K2Z_ENG();
            vns = k1z_time_vfft(h, z0, S, total);
            if (T > 1) g_cell_engaged = K2Z_ENG() - e0;
        }
    }
    else
    {
        {
            const long e0 = K2Z_ENG();
            vns = k1z_time_vfft(h, z0, S, total);
            if (T > 1) g_cell_engaged = K2Z_ENG() - e0;
        }
        cachebust();
        pace(cool_ms);
        mns = k2z_time_cmp(N1, N2, N3, z0, total);
    }
#undef K2Z_ENG
    double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    double vgf = (vns > 0) ? 5.0 * (double)total * log2((double)total) / vns : 0;
    printf("%11s %-8s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e  %s",
           shape, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h));
    if (T > 1)
        printf("  T=%d engaged %ld%s", T, g_cell_engaged, g_cell_engaged == 0 ? " (SERIAL)" : "");
    printf("\n");
    if (out)
    {
        char row[320], eng[24] = "";
        if (T > 1)
            snprintf(eng, sizeof eng, "%ld,", g_cell_engaged);   /* the column sits BEFORE flip: flip stays the row key's last field */
        if (nd == 3)
            snprintf(row, sizeof row, "%d,%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%s%d\n",
                     N1, N2, N3, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), eng, flip);
        else
            snprintf(row, sizeof row, "%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e,%s,%s%d\n",
                     N1, N2, plan_s, path, vns, mns, vgf, ratio, rel, vfft_plan_route(h), eng, flip);
        fflush(out);
        if (!k1z_csv_replace(g_csv_path, row))
            fputs(row, out);
        fflush(out);
    }
    free_d(z0);
    free_d(S);
    free_d(rt);
    vfft_destroy(h);
}

/* ════════════════════════════════════════════════════════════════════════
 * --kzb : Phase C1 gap map (il_coverage_plan.md) — K∈{2,3,4} INTERLEAVED
 * batched C2C, ours-as-is vs MKL. C0 pinned the contract: element e of
 * lane t at z[2*(e*K+t)] (lane-major). Our side: the PUBLIC front door,
 * OOP + order=NATURAL + layout=INTERLEAVED + howmany=K — today that is
 * the convert route (flat dein → split champions at K → flat inter);
 * K∈{2,3,4} have no OOP wisdom cells, so the first create per cell
 * calibrates champions (VFFT_MEASURE = quick DP) and banks them — create
 * sits OUTSIDE timing, later processes consume. MKL side, TWO arms:
 *   mirror = OUR layout exactly (COMPLEX_COMPLEX, NUMBER_OF_TRANSFORMS=K,
 *            DISTANCE=1, STRIDES={0,K}) — the routing-verdict number;
 *   home   = MKL's native batched layout (DISTANCE=N, unit stride) — a
 *            DIAGNOSTIC column for positioning (different memory contract,
 *            never the verdict input; C0's strawman rule).
 * Correctness = cross-engine elementwise vs the mirror arm (both natural,
 * identical layout; roundtrip cannot gate ordering). Fairness = the k1z
 * shape: warmup 10 / best-of-5 / reps_for, cachebust + cool between
 * engines, flip order per cell. No routing change anywhere — this mode
 * only MEASURES (C1's contract).
 * ════════════════════════════════════════════════════════════════════════ */
static int g_kzb = 0;
static int g_lm2d = 0;   /* --lm2d (2) / --lm3d (3): the lane-major 2D / 3D c2c batch cell (2026-10-15), the rank */
static int g_lm2dr = 0;  /* --lm2dr2c (1) / --lm2dc2r (2): the lane-major 2D real batch cell (2026-10-15) */
static int g_lm3dr = 0;  /* --lm3dr2c (1) / --lm3dc2r (2): the lane-major 3D real batch cell (2026-10-19) */
static int g_kzbr = 0;   /* --lmr2c = 1, --lmc2r = 2: the lane-major 1D REAL batch cells (2026-10-15), the
                          * same four arms as --kzb / --lmc2c over the real batch door (lm_batch_il.h) */

#ifdef VFFT_HAS_MKL
static double kzb_time_mkl(int N, int K, const double *z0, size_t total,
                           int home)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                             (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    { /* transform-contiguous: transform k at complex offset k*N */
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    }
    else
    { /* OUR lane-major contract: element n of lane k at complex n*K+k */
        MKL_LONG st[2] = { 0, (MKL_LONG)K };
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_INPUT_STRIDES, st);
        DftiSetValue(d, DFTI_OUTPUT_STRIDES, st);
    }
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    /* TWO WINDOWS, the same shape as k1z_time_vfft_d (2026-09-21): fairness
     * is the same protocol for both engines */
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                DftiComputeForward(d, zi, zo);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (t)
                pace(g_trial_pace_ms);
            double t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                DftiComputeForward(d, zi, zo);
            double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}

/* one-shot spectra for the correctness columns (mirror layout + the
 * home-vs-mirror MKL self-check). Returns 0 on descriptor failure. */
static int kzb_mkl_ref(int N, int K, const double *z0, size_t total,
                       int home, double *zout)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                             (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    {
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    }
    else
    {
        MKL_LONG st[2] = { 0, (MKL_LONG)K };
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_INPUT_STRIDES, st);
        DftiSetValue(d, DFTI_OUTPUT_STRIDES, st);
    }
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    DftiComputeForward(d, zi, zout);
    free_d(zi);
    DftiFreeDescriptor(&d);
    return 1;
}
#endif

/* LOOP arm: K sequential K=1 transforms over a TRANSFORM-CONTIGUOUS buffer
 * (transform k occupies [k*2N, (k+1)*2N) doubles) — "batching is an outer
 * loop", served by the shipped K=1 IL engines with ZERO new kernels and no
 * API change (each call is an ordinary K=1 call on a sub-buffer). This is
 * the honest like-for-like against MKL's HOME arm: both engines see the
 * same transform-contiguous memory. */
static double kzb_time_loop(vfft_plan h1, int N, int K, double *z0h,
                            double *S, size_t total)
{
    const size_t tn = 2 * (size_t)N;
    for (int w = 0; w < 10; w++)
        for (int k = 0; k < K; k++)
            vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, S + (size_t)k * tn);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_now_ns();
        for (int i = 0; i < reps; i++)
            for (int k = 0; k < K; k++)
                vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, S + (size_t)k * tn);
        double ns = (vfft_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

static void run_kzb_cell(int N, int K, FILE *out, int cool_ms, int flip)
{
    /* --kzb / --lmc2c (2026-10-15): the lane-major c2c batch DOOR (il/rank1/lm_batch_il.h, the column
     * pass over the K lanes) against MKL on the same layout; the loop and home arms stay as the
     * contiguous reference. --mt: the threaded cell, both sides at T, the engagement counted. */
    const int T = g_k1noop_mt ? g_mt : 1;
    if (T == 1)
        bench_pin_one_thread();   /* the one-thread protocol + the sibling guard */
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d K=%-3d   SKIP (front-door bundle unavailable)\n", N, K);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_PLACEMENT_OUT; /* C1: the story starts OOP */
    /* quick DP on the calibrate-on-miss */
    cfg.rank = 1;
    cfg.n[0] = N;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL; /* MKL is natural; cross-engine gate */
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    /* EXPLICIT lane-major: this handle IS the lane-major bridge arm — the
     * baseline the loop arm is measured against. Since 2026-08-04 the
     * default geometry is transform-contiguous, so leaving this implicit
     * would silently turn the "vfft" column into a second loop arm and the
     * speedup would read 1.0x. */
    cfg.batch = VFFT_BATCH_LANE_MAJOR;
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d K=%-3d   vfft_create FAILED\n", N, K);
        return;
    }

    /* K=1 handle for the LOOP arm (same front door, howmany=1, one thread) — the
     * transform-contiguous story needs no batched plan at all. */
    vfft_config_t c1 = cfg;
    c1.howmany = 1;
    c1.nthreads = 1;
    c1.batch = VFFT_BATCH_DEFAULT;
    vfft_plan h1 = vfft_create(&c1);

    size_t total = (size_t)N * (size_t)K;
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total);
    srand(42 + N + K);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* the SAME logical batch, transform-contiguous: transform k at k*2N */
    double *z0h = alloc_d(2 * total);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
        {
            size_t im = 2 * ((size_t)n * K + k);
            size_t ih = 2 * ((size_t)k * N + n);
            z0h[ih] = z0[im];
            z0h[ih + 1] = z0[im + 1];
        }

    double rel = -1.0, lrel = -1.0;
#ifdef VFFT_HAS_MKL
    { /* cross-engine gate vs the MIRROR arm (identical layout, both
       * natural) + MKL home-vs-mirror self-check (re-indexed): guards the
       * stride setup itself — a wrong mirror descriptor would make the
       * timing column measure a different transform. */
        double *zm = alloc_d(2 * total), *zh = alloc_d(2 * total);
        if (kzb_mkl_ref(N, K, z0, total, 0, zm))
        {
            vfft_execute(h, VFFT_FORWARD, z0, S);
            double xe = 0.0, xm = 0.0;
            for (size_t i = 0; i < 2 * total; i++)
            {
                double e = fabs(S[i] - zm[i]), m = fabs(zm[i]);
                if (e > xe) xe = e;
                if (m > xm) xm = m;
            }
            rel = xm > 0 ? xe / xm : xe;
            /* home self-check uses the SAME LOGICAL INPUT in home layout
             * (z0h, built above) — feeding raw z0 to both arms would
             * transform different logical vectors and the compare would be
             * meaningless. zh also gates the LOOP arm below. */
            if (kzb_mkl_ref(N, K, z0h, total, 1, zh))
            {
                double se = 0.0, sm = 0.0;
                for (int k = 0; k < K; k++)
                    for (int n = 0; n < N; n++)
                    {
                        size_t im = 2 * ((size_t)n * K + k);
                        size_t ih = 2 * ((size_t)k * N + n);
                        double er = fabs(zm[im] - zh[ih]);
                        double ei = fabs(zm[im + 1] - zh[ih + 1]);
                        double e = er > ei ? er : ei;
                        double m = fabs(zh[ih]);
                        if (e > se) se = e;
                        if (m > sm) sm = m;
                    }
                if (sm > 0 && se / sm > 1e-10)
                    fprintf(stderr,
                            "kzb: MKL mirror/home self-check DIFF %.2e at "
                            "N=%d K=%d — mirror strides suspect\n",
                            se / sm, N, K);
                /* LOOP-arm gate: K sequential K=1 transforms over the
                 * transform-contiguous buffer must reproduce MKL's home
                 * spectrum elementwise (both natural, same layout). */
                if (h1)
                {
                    const size_t tn = 2 * (size_t)N;
                    for (int k = 0; k < K; k++)
                        vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, S + (size_t)k * tn);
                    double le = 0.0, lm = 0.0;
                    for (size_t i = 0; i < 2 * total; i++)
                    {
                        double e = fabs(S[i] - zh[i]), m = fabs(zh[i]);
                        if (e > le) le = e;
                        if (m > lm) lm = m;
                    }
                    lrel = lm > 0 ? le / lm : le;
                }
            }
        }
        free_d(zm);
        free_d(zh);
    }
#endif

    /* ARMS — k1z fairness shape; the two MKL arms share MKL's slot, the
     * LOOP arm shares ours (it is our engine, different geometry). */
    double vns = 0, mns = 0, hns = 0, lns = 0;
    g_cell_engaged = -1;
    /* the engagement proof at T > 1 (mt_results_need_engagement_proof): the column pass's MT counter
     * over the timed executes; the loop arm runs its K = 1 plans serially either way */
#define KZB_OURS() do { \
        \
        { const long e0 = vfft_il2d_col_mt_passes(); \
          vns = k1z_time_vfft(h, z0, S, total); \
          if (T > 1) g_cell_engaged = vfft_il2d_col_mt_passes() - e0; } \
        if (h1) { pace(g_trial_pace_ms); lns = kzb_time_loop(h1, N, K, z0h, S, total); } \
    } while (0)
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = kzb_time_mkl(N, K, z0, total, 0);
        pace(g_trial_pace_ms);
        hns = kzb_time_mkl(N, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        KZB_OURS();
    }
    else
    {
        KZB_OURS();
        cachebust();
        pace(cool_ms);
        mns = kzb_time_mkl(N, K, z0, total, 0);
        pace(g_trial_pace_ms);
        hns = kzb_time_mkl(N, K, z0, total, 1);
    }
#else
    (void)cool_ms;
    (void)flip;
    KZB_OURS();
#endif
#undef KZB_OURS
    double rmir = (vns > 0 && mns > 0) ? mns / vns : 0;
    double rhom = (vns > 0 && hns > 0) ? hns / vns : 0;
    /* THE number this arm exists for: our loop-over-K=1 on transform-
     * contiguous data vs MKL batched on the SAME layout. */
    double rloop = (lns > 0 && hns > 0) ? hns / lns : 0;
    printf("%-8d %-4d %-8s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %7.2fx %9.2e %9.2e  %s",
           N, K, "lm-oop", vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h));
    if (T > 1)
        printf("  T=%d engaged %ld%s", T, g_cell_engaged, g_cell_engaged == 0 ? " (SERIAL)" : "");
    printf("\n");
    if (out)
    {
        char eng[24] = "";
        if (T > 1)
            snprintf(eng, sizeof eng, ",%ld", g_cell_engaged);
        fprintf(out, "%d,%d,z:lm,lm-oop,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e,%.3e,%s%s\n",
                N, K, vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h), eng);
        fflush(out);
    }
    free_d(z0);
    free_d(z0h);
    free_d(S);
    if (h1)
        vfft_destroy(h1);
    vfft_destroy(h);
}

/* ── --lm2d / --lm3d (2026-10-15): the LANE-MAJOR 2D (3D) C2C BATCH cell, step 6 of the lane-major design
 * (docs/roadmap/lane_major_batch_design.md): K planes of N1 x N2 (volumes of N1 x N2 x N3) side by side,
 * element (r, c) of transform t at z[2 ((r N2 + c) K + t)] ((i, j, k) at z[2 ((i N2 N3 + j N3 + k) K + t)]).
 * The same four arms and protocol as --lmc2c: ours (the batch door of the rank: the first axis a column pass
 * over the full width, then the lane-major batch one rank down per block), the loop of K one-plane plans on
 * contiguous planes, MKL on our lane-major layout (DFTI of the rank, NUMBER_OF_TRANSFORMS = K, distance 1,
 * strides {0, N2 N3 K, N3 K, K}), MKL on contiguous planes (distance N1 N2 N3); --mt both sides at T with the
 * engagement counted. */
static size_t lm2d_volume(int nd, const int *n)
{
    size_t P = 1;
    for (int i = 0; i < nd; i++)
        P *= (size_t)n[i];
    return P;
}
static void lm2d_shape(char *buf, size_t sz, int nd, const int *n)
{
    if (nd > 2)
        snprintf(buf, sz, "%dx%dx%d", n[0], n[1], n[2]);
    else
        snprintf(buf, sz, "%dx%d", n[0], n[1]);
}
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE lm2d_mkl_desc(int nd, const int *n, int K, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG len[3], st[4], P = 1;
    for (int i = 0; i < nd; i++)
    {
        len[i] = (MKL_LONG)n[i];
        P *= (MKL_LONG)n[i];
    }
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, (MKL_LONG)nd, len) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    /* the strides: the last axis 1 (contiguous planes, plane t at complex offset t P) or K (OUR lane-major
     * contract: element e of lane t at complex e K + t), each earlier axis the later extent times the next */
    st[0] = 0;
    st[nd] = home ? 1 : (MKL_LONG)K;
    for (int i = nd - 1; i >= 1; i--)
        st[i] = st[i + 1] * (MKL_LONG)n[i];
    DftiSetValue(d, DFTI_INPUT_DISTANCE, home ? P : (MKL_LONG)1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, home ? P : (MKL_LONG)1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, st);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, st);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
static int lm2d_mkl_ref(int nd, const int *n, int K, const double *z0, size_t total, int home, double *zout)
{
    DFTI_DESCRIPTOR_HANDLE d = lm2d_mkl_desc(nd, n, K, home);
    double *zi;
    if (!d)
        return 0;
    zi = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    DftiComputeForward(d, zi, zout);
    free_d(zi);
    DftiFreeDescriptor(&d);
    return 1;
}
static double lm2d_time_mkl(int nd, const int *n, int K, const double *z0, size_t total, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = lm2d_mkl_desc(nd, n, K, home);
    double *zi, *zo, best = 1e18;
    int reps, win;
    if (!d)
        return 0;
    zi = alloc_d(2 * total);
    zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, zi, zo);
    reps = reps_for(total);
    for (win = 0; win < 2; win++)
    {   /* two windows, the shape of kzb_time_mkl: the same protocol for both engines */
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                DftiComputeForward(d, zi, zo);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            double t0;
            if (t)
                pace(g_trial_pace_ms);
            t0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                DftiComputeForward(d, zi, zo);
            {
                const double ns = (vfft_now_ns() - t0) / reps;
                if (ns < best)
                    best = ns;
            }
        }
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}
#endif
/* the loop arm: K sequential one-plane plans over contiguous planes (plane t at t P complexes) */
static double lm2d_time_loop(vfft_plan h1, size_t P, int K, double *z0h, double *S, size_t total)
{
    const size_t tn = 2 * P;
    int reps;
    double best = 1e18;
    for (int w = 0; w < 10; w++)
        for (int k = 0; k < K; k++)
            vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, S + (size_t)k * tn);
    reps = reps_for(total);
    for (int t = 0; t < 5; t++)
    {
        double t0;
        if (t)
            pace(g_trial_pace_ms);
        t0 = vfft_now_ns();
        for (int i = 0; i < reps; i++)
            for (int k = 0; k < K; k++)
                vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, S + (size_t)k * tn);
        {
            const double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    return best;
}
static void run_lm2d_cell(int nd, const int *n, int K, FILE *out, int cool_ms, int flip)
{
    const int T = g_k1noop_mt ? g_mt : 1;
    const char *W;
    vfft_config_t cfg, c1;
    vfft_plan h, h1;
    const size_t P = lm2d_volume(nd, n), total = P * (size_t)K;
    double *z0, *S, *z0h;
    double rel = -1.0, lrel = -1.0;
    double vns = 0, mns = 0, hns = 0, lns = 0, rmir, rhom, rloop;
    char shape[32], path[16], plan[16];
    if (T == 1)
        bench_pin_one_thread();   /* the one-thread protocol + the sibling guard */
    W = k1z_bundle();
    lm2d_shape(shape, sizeof shape, nd, n);
    snprintf(path, sizeof path, "lm%dd-oop", nd);
    snprintf(plan, sizeof plan, "z:lm%dd", nd);
    if (!W)
    {
        printf("%s K=%-3d   SKIP (front-door bundle unavailable)\n", shape, K);
        return;
    }
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = nd;
    for (int i = 0; i < nd; i++)
        cfg.n[i] = n[i];
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL;   /* MKL is natural; cross-engine gate */
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = VFFT_BATCH_LANE_MAJOR;   /* explicit: the batch door of the rank */
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-11s K=%-3d   vfft_create FAILED: %s\n", shape, K, vfft_error());
        return;
    }
    c1 = cfg;   /* the one-plane plan for the LOOP arm (the same front door, howmany = 1, one thread) */
    c1.howmany = 1;
    c1.nthreads = 1;
    c1.batch = VFFT_BATCH_DEFAULT;
    h1 = vfft_create(&c1);
    z0 = alloc_d(2 * total);
    S = alloc_d(2 * total);
    srand(42 + n[0] + n[1] + (nd > 2 ? n[2] : 0) + K);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;
    /* the SAME logical batch on contiguous planes: plane k at k P */
    z0h = alloc_d(2 * total);
    for (int k = 0; k < K; k++)
        for (size_t e = 0; e < P; e++)
        {
            const size_t im = 2 * (e * K + k), ih = 2 * ((size_t)k * P + e);
            z0h[ih] = z0[im];
            z0h[ih + 1] = z0[im + 1];
        }
#ifdef VFFT_HAS_MKL
    {   /* the cross-engine gate vs the MIRROR arm (the same layout, both natural), MKL's home-vs-mirror
         * self-check (re-indexed: guards the stride setup), and the LOOP arm vs MKL's home spectrum */
        double *zm = alloc_d(2 * total), *zh = alloc_d(2 * total);
        if (lm2d_mkl_ref(nd, n, K, z0, total, 0, zm))
        {
            double xe = 0.0, xm = 0.0;
            vfft_execute(h, VFFT_FORWARD, z0, S);
            for (size_t i = 0; i < 2 * total; i++)
            {
                const double e = fabs(S[i] - zm[i]), m = fabs(zm[i]);
                if (e > xe) xe = e;
                if (m > xm) xm = m;
            }
            rel = xm > 0 ? xe / xm : xe;
            if (lm2d_mkl_ref(nd, n, K, z0h, total, 1, zh))
            {
                double se = 0.0, sm = 0.0;
                for (int k = 0; k < K; k++)
                    for (size_t e = 0; e < P; e++)
                    {
                        const size_t im = 2 * (e * K + k), ih = 2 * ((size_t)k * P + e);
                        const double er = fabs(zm[im] - zh[ih]), ei = fabs(zm[im + 1] - zh[ih + 1]);
                        const double d = er > ei ? er : ei, m = fabs(zh[ih]);
                        if (d > se) se = d;
                        if (m > sm) sm = m;
                    }
                if (sm > 0 && se / sm > 1e-10)
                    fprintf(stderr, "lm%dd: MKL mirror/home self-check DIFF %.2e at %s K=%d -- mirror strides suspect\n",
                            nd, se / sm, shape, K);
                if (h1)
                {
                    const size_t tn = 2 * P;
                    double le = 0.0, lm = 0.0;
                    for (int k = 0; k < K; k++)
                        vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, S + (size_t)k * tn);
                    for (size_t i = 0; i < 2 * total; i++)
                    {
                        const double e = fabs(S[i] - zh[i]), m = fabs(zh[i]);
                        if (e > le) le = e;
                        if (m > lm) lm = m;
                    }
                    lrel = lm > 0 ? le / lm : le;
                }
            }
        }
        free_d(zm);
        free_d(zh);
    }
#endif
    g_cell_engaged = -1;
#define LM2D_OURS() do { \
        { const long e0 = vfft_il2d_col_mt_passes(); \
          vns = k1z_time_vfft(h, z0, S, total); \
          if (T > 1) g_cell_engaged = vfft_il2d_col_mt_passes() - e0; } \
        if (h1) { pace(g_trial_pace_ms); lns = lm2d_time_loop(h1, P, K, z0h, S, total); } \
    } while (0)
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = lm2d_time_mkl(nd, n, K, z0, total, 0);
        pace(g_trial_pace_ms);
        hns = lm2d_time_mkl(nd, n, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        LM2D_OURS();
    }
    else
    {
        LM2D_OURS();
        cachebust();
        pace(cool_ms);
        mns = lm2d_time_mkl(nd, n, K, z0, total, 0);
        pace(g_trial_pace_ms);
        hns = lm2d_time_mkl(nd, n, K, z0, total, 1);
    }
#else
    (void)cool_ms;
    (void)flip;
    LM2D_OURS();
#endif
#undef LM2D_OURS
    rmir = (vns > 0 && mns > 0) ? mns / vns : 0;
    rhom = (vns > 0 && hns > 0) ? hns / vns : 0;
    rloop = (lns > 0 && hns > 0) ? hns / lns : 0;
    printf("%-11s %-4d %-8s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %7.2fx %9.2e %9.2e  %s",
           shape, K, path, vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h));
    if (T > 1)
        printf("  T=%d engaged %ld%s", T, g_cell_engaged, g_cell_engaged == 0 ? " (SERIAL)" : "");
    printf("\n");
    if (out)
    {
        char eng[24] = "";
        if (T > 1)
            snprintf(eng, sizeof eng, ",%ld", g_cell_engaged);
        fprintf(out, "%s,%d,%s,%s,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e,%.3e,%s%s\n",
                shape, K, plan, path, vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h), eng);
        fflush(out);
    }
    free_d(z0);
    free_d(z0h);
    free_d(S);
    if (h1)
        vfft_destroy(h1);
    vfft_destroy(h);
}

/* ── --lm2dr2c / --lm2dc2r (2026-10-15): the LANE-MAJOR 2D REAL BATCH cells (step 6): K real planes of
 * N1 x N2 side by side, x[(r N2 + c) K + t], the spectrum z[2 ((r H + f) K + t)], H = N2 / 2 + 1. The same four
 * arms and protocol as --lmr2c / --lmc2r: ours (the 2D real batch door), the loop of K one-plane 2D real plans
 * on contiguous planes, MKL (DFTI REAL 2D, CCE storage) on our lane-major layout -- the reals at strides
 * {0, N2 K, K}, the bins at {0, H K, K}, distance 1 -- and MKL on contiguous planes; --mt both sides at T. */
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE lm2dr_dfti(int N1, int N2, int K, int c2r, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    const MKL_LONG H = (MKL_LONG)(N2 / 2 + 1);
    MKL_LONG len[2] = { (MKL_LONG)N1, (MKL_LONG)N2 };
    MKL_LONG sr[3], sb[3];   /* the reals' strides, the bins' strides */
    MKL_LONG dr, db;         /* their distances */
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 2, len) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    {   /* contiguous planes: the reals at distance N1 N2, the bins at N1 H */
        sr[0] = 0; sr[1] = (MKL_LONG)N2; sr[2] = 1;
        sb[0] = 0; sb[1] = H; sb[2] = 1;
        dr = (MKL_LONG)N1 * N2; db = (MKL_LONG)N1 * H;
    }
    else
    {   /* our lane-major contract: element (r, c) of lane t at (r N2 + c) K + t, bin (r, f) at (r H + f) K + t */
        sr[0] = 0; sr[1] = (MKL_LONG)N2 * K; sr[2] = (MKL_LONG)K;
        sb[0] = 0; sb[1] = H * K; sb[2] = (MKL_LONG)K;
        dr = 1; db = 1;
    }
    /* the backward's INPUT is the bins (the 1D cell's trap: a forward descriptor used backward faults) */
    DftiSetValue(d, DFTI_INPUT_DISTANCE, c2r ? db : dr);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, c2r ? dr : db);
    DftiSetValue(d, DFTI_INPUT_STRIDES, c2r ? sb : sr);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, c2r ? sr : sb);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
static void kzbr_mkl_run(DFTI_DESCRIPTOR_HANDLE d, int c2r, const double *in, double *out);   /* the 1D real cell's, below */
static int lm2dr_mkl_ref(int N1, int N2, int K, int c2r, const double *in, int home, double *out)
{
    DFTI_DESCRIPTOR_HANDLE d = lm2dr_dfti(N1, N2, K, c2r, home);
    if (!d)
        return 0;
    kzbr_mkl_run(d, c2r, in, out);
    DftiFreeDescriptor(&d);
    return 1;
}
static double lm2dr_time_mkl(int N1, int N2, int K, int c2r, const double *in, double *out, size_t total, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = lm2dr_dfti(N1, N2, K, c2r, home);
    int reps = reps_for(total), win, t, i, w;
    double best = 1e18;
    if (!d)
        return 0;
    for (w = 0; w < 10; w++)
        kzbr_mkl_run(d, c2r, in, out);
    for (win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                kzbr_mkl_run(d, c2r, in, out);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (t = 0; t < 5; t++)
        {
            double t0;
            if (t)
                pace(g_trial_pace_ms);
            t0 = vfft_now_ns();
            for (i = 0; i < reps; i++)
                kzbr_mkl_run(d, c2r, in, out);
            {
                const double ns = (vfft_now_ns() - t0) / reps;
                if (ns < best)
                    best = ns;
            }
        }
    }
    DftiFreeDescriptor(&d);
    return best;
}
#endif
/* the loop arm: K sequential one-plane 2D real plans over contiguous planes */
static double lm2dr_time_loop(vfft_plan h1, int N1, int N2, int K, int c2r, const double *in_h, double *S_h, size_t total)
{
    const size_t rp = (size_t)N1 * N2, cp = 2 * (size_t)N1 * (N2 / 2 + 1);
    const size_t sin = c2r ? cp : rp, sout = c2r ? rp : cp;
    const vfft_dir_t dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
    int reps = reps_for(total), t, i, k, w;
    double best = 1e18;
    for (w = 0; w < 10; w++)
        for (k = 0; k < K; k++)
            vfft_execute(h1, dir, in_h + (size_t)k * sin, S_h + (size_t)k * sout);
    for (t = 0; t < 5; t++)
    {
        double t0;
        if (t)
            pace(g_trial_pace_ms);
        t0 = vfft_now_ns();
        for (i = 0; i < reps; i++)
            for (k = 0; k < K; k++)
                vfft_execute(h1, dir, in_h + (size_t)k * sin, S_h + (size_t)k * sout);
        {
            const double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
    return best;
}
static void run_lm2dr_cell(int N1, int N2, int K, int c2r, FILE *out, int cool_ms, int flip)
{
    const int T = g_k1noop_mt ? g_mt : 1, H = N2 / 2 + 1;
    const size_t P = (size_t)N1 * N2, PH = (size_t)N1 * H;
    const size_t XN = P * (size_t)K, ZN = 2 * PH * (size_t)K;   /* the reals, the bins (doubles) */
    const size_t nin = c2r ? ZN : XN, nout = c2r ? XN : ZN, total = XN;
    const char *W, *plan_s = c2r ? "z:lm2dc2r" : "z:lm2dr2c";
    vfft_config_t cfg, c1;
    vfft_plan h, h1;
    double *in, *in_h, *S, *S_h, *zm, *zh;
    double rel = -1.0, lrel = -1.0, vns = 0, mns = 0, hns = 0, lns = 0, rmir, rhom, rloop;
    size_t i, e, t;
    char shape[32];
    if (T == 1)
        bench_pin_one_thread();
    W = k1z_bundle();
    if (!W)
    {
        printf("%dx%d K=%-3d   SKIP (front-door bundle unavailable)\n", N1, N2, K);
        return;
    }
    snprintf(shape, sizeof shape, "%dx%d", N1, N2);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = 2;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = VFFT_BATCH_LANE_MAJOR;   /* the named geometry: the 2D real batch door */
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-9s K=%-3d   vfft_create FAILED: %s\n", shape, K, vfft_error());
        return;
    }
    c1 = cfg;
    c1.howmany = 1;
    c1.nthreads = 1;
    c1.batch = VFFT_BATCH_DEFAULT;
    h1 = vfft_create(&c1);
    in = alloc_d(nin + 16); in_h = alloc_d(nin + 16); S = alloc_d(nout + 16); S_h = alloc_d(nout + 16);
    zm = alloc_d(nout + 16); zh = alloc_d(nout + 16);
    srand(42 + N1 + N2 + K + c2r);
    if (!c2r)
    {   /* the reals, lane-major, and the same planes contiguous */
        for (i = 0; i < XN; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        for (e = 0; e < P; e++)
            for (t = 0; t < (size_t)K; t++) in_h[t * P + e] = in[e * (size_t)K + t];
    }
    else
    {   /* the bins: a random real batch through MKL's lane-major forward (the DC and Nyquist bins real) */
        double *r = alloc_d(XN + 16);
        for (i = 0; i < XN; i++) r[i] = (double)rand() / RAND_MAX - 0.5;
#ifdef VFFT_HAS_MKL
        if (!lm2dr_mkl_ref(N1, N2, K, 0, r, 0, in))
#endif
            for (i = 0; i < ZN; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        free_d(r);
        for (e = 0; e < PH; e++)
            for (t = 0; t < (size_t)K; t++)
            {
                in_h[2 * (t * PH + e)] = in[2 * (e * (size_t)K + t)];
                in_h[2 * (t * PH + e) + 1] = in[2 * (e * (size_t)K + t) + 1];
            }
    }
#ifdef VFFT_HAS_MKL
    if (lm2dr_mkl_ref(N1, N2, K, c2r, in, 0, zm))
    {   /* ours vs MKL on the identical layout; the loop arm vs MKL's contiguous output */
        double xe = 0.0, xm = 0.0;
        vfft_execute(h, c2r ? VFFT_BACKWARD : VFFT_FORWARD, in, S);
        for (i = 0; i < nout; i++)
        {
            const double d = fabs(S[i] - zm[i]), m = fabs(zm[i]);
            if (d > xe) xe = d;
            if (m > xm) xm = m;
        }
        rel = xm > 0 ? xe / xm : xe;
        if (h1 && lm2dr_mkl_ref(N1, N2, K, c2r, in_h, 1, zh))
        {
            double le = 0.0, lm = 0.0;
            {
                const size_t rp = P, cp = 2 * PH;
                int k;
                for (k = 0; k < K; k++)
                    vfft_execute(h1, c2r ? VFFT_BACKWARD : VFFT_FORWARD, in_h + (size_t)k * (c2r ? cp : rp), S_h + (size_t)k * (c2r ? rp : cp));
            }
            for (i = 0; i < nout; i++)
            {
                const double d = fabs(S_h[i] - zh[i]), m = fabs(zh[i]);
                if (d > le) le = d;
                if (m > lm) lm = m;
            }
            lrel = lm > 0 ? le / lm : le;
        }
    }
#endif
    g_cell_engaged = -1;
#define LM2DR_OURS() do { \
        { const long e0 = vfft_il2d_col_mt_passes(); \
          vns = k1z_time_vfft_d(h, in, S, total, c2r ? VFFT_BACKWARD : VFFT_FORWARD); \
          if (T > 1) g_cell_engaged = vfft_il2d_col_mt_passes() - e0; } \
        if (h1) { pace(g_trial_pace_ms); lns = lm2dr_time_loop(h1, N1, N2, K, c2r, in_h, S_h, total); } \
    } while (0)
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = lm2dr_time_mkl(N1, N2, K, c2r, in, zm, total, 0);
        pace(g_trial_pace_ms);
        hns = lm2dr_time_mkl(N1, N2, K, c2r, in_h, zh, total, 1);
        cachebust();
        pace(cool_ms);
        LM2DR_OURS();
    }
    else
    {
        LM2DR_OURS();
        cachebust();
        pace(cool_ms);
        mns = lm2dr_time_mkl(N1, N2, K, c2r, in, zm, total, 0);
        pace(g_trial_pace_ms);
        hns = lm2dr_time_mkl(N1, N2, K, c2r, in_h, zh, total, 1);
    }
#else
    (void)cool_ms;
    (void)flip;
    LM2DR_OURS();
#endif
#undef LM2DR_OURS
    rmir = (vns > 0 && mns > 0) ? mns / vns : 0;
    rhom = (vns > 0 && hns > 0) ? hns / vns : 0;
    rloop = (lns > 0 && hns > 0) ? hns / lns : 0;
    printf("%-9s %-4d %-8s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %7.2fx %9.2e %9.2e  %s",
           shape, K, c2r ? "lm2dc2r" : "lm2dr2c", vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h));
    if (T > 1)
        printf("  T=%d engaged %ld%s", T, g_cell_engaged, g_cell_engaged == 0 ? " (SERIAL)" : "");
    printf("\n");
    if (out)
    {
        char eng[24] = "";
        if (T > 1)
            snprintf(eng, sizeof eng, ",%ld", g_cell_engaged);
        fprintf(out, "%s,%d,%s,lm2d-oop,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e,%.3e,%s%s\n",
                shape, K, plan_s, vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h), eng);
        fflush(out);
    }
    free_d(in); free_d(in_h); free_d(S); free_d(S_h); free_d(zm); free_d(zh);
    if (h1)
        vfft_destroy(h1);
    vfft_destroy(h);
}

/* ── --lm3dr2c / --lm3dc2r (2026-10-19): the LANE-MAJOR 3D REAL BATCH cells (step 6 one rank up): K real
 * volumes of N1 x N2 x N3 side by side, x[((i N2 + j) N3 + k) K + t], the spectrum
 * z[2 (((i N2 + j) H + f) K + t)], H = N3 / 2 + 1. Four arms on the same data, out of place: ours (the 3D real
 * batch door, VFFT_BATCH_LANE_MAJOR), ours on CONTIGUOUS volumes (the volume queue: one plan at howmany = K,
 * VFFT_BATCH_CONTIGUOUS, the served door and not a loop; volume t at t N1 N2 N3 reals, its bins at
 * t 2 N1 N2 H doubles), MKL (DFTI REAL 3D, CCE storage) on our lane-major layout -- the reals at strides
 * {0, N2 N3 K, N3 K, K}, the bins at {0, N2 H K, H K, K}, distance 1 -- and MKL on contiguous volumes
 * (distances N1 N2 N3 and N1 N2 H). Two verdicts: MKL lm / ours lm and MKL contig / ours contig. Every arm
 * against MKL's result: ours lm against MKL lm, ours contig against MKL contig, MKL lm against MKL contig
 * re-indexed (the stride setup's guard) and ours lm against MKL contig re-indexed. For C2R the spectrum is
 * MKL's lane-major R2C of random reals (a valid half spectrum, the DC and Nyquist bins real). --mt both sides
 * at T, the engagement counted on both of our doors: the batch door's column pass (vfft_il2d_col_mt_passes)
 * and the queue's dispatches (vfft_pq_mt_passes) with the volume plan's own threaded passes
 * (vfft_ilnd_mt_passes + the column counter) -- a threaded number without a moved counter is vacuous. */
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE lm3dr_dfti(const int *n, int K, int c2r, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    const MKL_LONG N1 = (MKL_LONG)n[0], N2 = (MKL_LONG)n[1], N3 = (MKL_LONG)n[2], H = (MKL_LONG)(n[2] / 2 + 1);
    MKL_LONG len[3] = { N1, N2, N3 };
    MKL_LONG sr[4], sb[4];   /* the reals' strides, the bins' strides */
    MKL_LONG dr, db;         /* their distances */
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 3, len) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    {   /* contiguous volumes: the reals at distance N1 N2 N3, the bins at N1 N2 H */
        sr[0] = 0; sr[1] = N2 * N3; sr[2] = N3; sr[3] = 1;
        sb[0] = 0; sb[1] = N2 * H; sb[2] = H; sb[3] = 1;
        dr = N1 * N2 * N3; db = N1 * N2 * H;
    }
    else
    {   /* our lane-major contract: real (i, j, k) of lane t at ((i N2 + j) N3 + k) K + t,
         * bin (i, j, f) at ((i N2 + j) H + f) K + t */
        sr[0] = 0; sr[1] = N2 * N3 * (MKL_LONG)K; sr[2] = N3 * (MKL_LONG)K; sr[3] = (MKL_LONG)K;
        sb[0] = 0; sb[1] = N2 * H * (MKL_LONG)K; sb[2] = H * (MKL_LONG)K; sb[3] = (MKL_LONG)K;
        dr = 1; db = 1;
    }
    /* the backward's INPUT is the bins (the 1D cell's trap: a forward descriptor used backward faults) */
    DftiSetValue(d, DFTI_INPUT_DISTANCE, c2r ? db : dr);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, c2r ? dr : db);
    DftiSetValue(d, DFTI_INPUT_STRIDES, c2r ? sb : sr);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, c2r ? sr : sb);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
static int lm3dr_mkl_ref(const int *n, int K, int c2r, const double *in, int home, double *out)
{
    DFTI_DESCRIPTOR_HANDLE d = lm3dr_dfti(n, K, c2r, home);
    if (!d)
        return 0;
    kzbr_mkl_run(d, c2r, in, out);
    DftiFreeDescriptor(&d);
    return 1;
}
static double lm3dr_time_mkl(const int *n, int K, int c2r, const double *in, double *out, size_t total, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = lm3dr_dfti(n, K, c2r, home);
    int reps = reps_for(total), win, t, i, w;
    double best = 1e18;
    if (!d)
        return 0;
    for (w = 0; w < 10; w++)
        kzbr_mkl_run(d, c2r, in, out);
    for (win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                kzbr_mkl_run(d, c2r, in, out);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (t = 0; t < 5; t++)
        {
            double t0;
            if (t)
                pace(g_trial_pace_ms);
            t0 = vfft_now_ns();
            for (i = 0; i < reps; i++)
                kzbr_mkl_run(d, c2r, in, out);
            {
                const double ns = (vfft_now_ns() - t0) / reps;
                if (ns < best)
                    best = ns;
            }
        }
    }
    DftiFreeDescriptor(&d);
    return best;
}
#endif
/* the largest elementwise difference of a against b over n doubles, relative to b's largest magnitude */
static double lm3dr_rel(const double *a, const double *b, size_t n)
{
    double xe = 0.0, xm = 0.0;
    size_t i;
    for (i = 0; i < n; i++)
    {
        const double d = fabs(a[i] - b[i]), m = fabs(b[i]);
        if (d > xe) xe = d;
        if (m > xm) xm = m;
    }
    return xm > 0 ? xe / xm : xe;
}
/* a lane-major buffer of E elements (w doubles each: 1 a real, 2 a bin) of K lanes, re-indexed onto
 * contiguous volumes: element e of lane t from lm[w (e K + t)] to contig[w (t E + e)] */
static void lm3dr_to_contig(const double *lm, double *contig, size_t E, int K, int w)
{
    size_t e, t;
    for (e = 0; e < E; e++)
        for (t = 0; t < (size_t)K; t++)
        {
            const double *s = lm + (size_t)w * (e * (size_t)K + t);
            double *dd = contig + (size_t)w * (t * E + e);
            dd[0] = s[0];
            if (w > 1)
                dd[1] = s[1];
        }
}
static void run_lm3dr_cell(const int *n, int K, int c2r, FILE *out, int cool_ms, int flip)
{
    const int T = g_k1noop_mt ? g_mt : 1, N1 = n[0], N2 = n[1], N3 = n[2], H = N3 / 2 + 1;
    const size_t P = (size_t)N1 * N2 * N3, PH = (size_t)N1 * N2 * H;
    const size_t XN = P * (size_t)K, ZN = 2 * PH * (size_t)K;   /* the reals, the bins (doubles) */
    const size_t nin = c2r ? ZN : XN, nout = c2r ? XN : ZN, total = XN;
    const size_t E_out = c2r ? P : PH;   /* the output's elements per volume, w_out doubles each */
    const int w_out = c2r ? 1 : 2;
    const vfft_dir_t dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
    const char *W, *plan_s = c2r ? "z:lm3dc2r" : "z:lm3dr2c";
    vfft_config_t cfg, cq;
    vfft_plan h, hq;
    double *in, *in_h, *S, *S_h, *zm, *zh, *tmp;
    double rel = -1.0, qrel = -1.0, mself = -1.0, xrel = -1.0;
    double vns = 0, qns = 0, mns = 0, hns = 0, rmir, rhom, rq;
    long eng_pq = -1, eng_plane = -1, eng_grp = -1;   /* the queue's three arms: queued dispatches, the one-volume
                                                      * plan's threaded passes (loop), group executes (groups) */
    size_t i;
    char shape[32];
    if (T == 1)
        bench_pin_one_thread();
    W = k1z_bundle();
    snprintf(shape, sizeof shape, "%dx%dx%d", N1, N2, N3);
    if (!W)
    {
        printf("%s K=%-3d   SKIP (front-door bundle unavailable)\n", shape, K);
        return;
    }
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = 3;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.n[2] = N3;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = VFFT_BATCH_LANE_MAJOR;   /* the named geometry: the 3D real batch door */
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-11s K=%-3d   vfft_create FAILED (lane-major): %s\n", shape, K, vfft_error());
        return;
    }
    cq = cfg;   /* the CONTIGUOUS arm: the volume queue, one plan at howmany = K, the same T */
    cq.batch = VFFT_BATCH_CONTIGUOUS;
    hq = vfft_create(&cq);
    if (!hq)
        printf("# %s K=%d: vfft_create FAILED (contiguous): %s -- the contiguous arm is skipped\n", shape, K, vfft_error());
    in = alloc_d(nin + 16); in_h = alloc_d(nin + 16); S = alloc_d(nout + 16); S_h = alloc_d(nout + 16);
    zm = alloc_d(nout + 16); zh = alloc_d(nout + 16); tmp = alloc_d(nout + 16);
    srand(42 + N1 + N2 + N3 + K + c2r);
    if (!c2r)
    {   /* the reals, lane-major, and the same volumes contiguous */
        for (i = 0; i < XN; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        lm3dr_to_contig(in, in_h, P, K, 1);
    }
    else
    {   /* the bins: a random real batch through MKL's lane-major forward (the DC and Nyquist bins real) */
        double *r = alloc_d(XN + 16);
        for (i = 0; i < XN; i++) r[i] = (double)rand() / RAND_MAX - 0.5;
#ifdef VFFT_HAS_MKL
        if (!lm3dr_mkl_ref(n, K, 0, r, 0, in))
#endif
            for (i = 0; i < ZN; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        free_d(r);
        lm3dr_to_contig(in, in_h, PH, K, 2);
    }
#ifdef VFFT_HAS_MKL
    if (lm3dr_mkl_ref(n, K, c2r, in, 0, zm))
    {   /* ours lm against MKL on the identical layout */
        vfft_execute(h, dir, in, S);
        rel = lm3dr_rel(S, zm, nout);
        if (lm3dr_mkl_ref(n, K, c2r, in_h, 1, zh))
        {   /* the queue against MKL's contiguous output; MKL's two layouts against each other and ours lm
             * against the contiguous result, both re-indexed (the stride setup's guard) */
            if (hq)
            {
                vfft_execute(hq, dir, in_h, S_h);
                qrel = lm3dr_rel(S_h, zh, nout);
            }
            lm3dr_to_contig(zm, tmp, E_out, K, w_out);
            mself = lm3dr_rel(tmp, zh, nout);
            lm3dr_to_contig(S, tmp, E_out, K, w_out);
            xrel = lm3dr_rel(tmp, zh, nout);
            if (mself > 1e-10)
                fprintf(stderr, "lm3dr: MKL mirror/home self-check DIFF %.2e at %s K=%d -- mirror strides suspect\n",
                        mself, shape, K);
        }
    }
#endif
    g_cell_engaged = -1;
#define LM3DR_OURS() do { \
        { const long e0 = vfft_il2d_col_mt_passes(); \
          vns = k1z_time_vfft_d(h, in, S, total, dir); \
          if (T > 1) g_cell_engaged = vfft_il2d_col_mt_passes() - e0; } \
        if (hq) { \
          const long p0 = vfft_pq_mt_passes(), i0 = vfft_ilnd_mt_passes(), c0 = vfft_il2d_col_mt_passes(); \
          const long g0 = vfft_pq_grp_passes(); \
          pace(g_trial_pace_ms); \
          qns = k1z_time_vfft_d(hq, in_h, S_h, total, dir); \
          if (T > 1) { eng_pq = vfft_pq_mt_passes() - p0; eng_grp = vfft_pq_grp_passes() - g0; \
                       eng_plane = (vfft_ilnd_mt_passes() - i0) + (vfft_il2d_col_mt_passes() - c0); } } \
    } while (0)
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = lm3dr_time_mkl(n, K, c2r, in, zm, total, 0);
        pace(g_trial_pace_ms);
        hns = lm3dr_time_mkl(n, K, c2r, in_h, zh, total, 1);
        cachebust();
        pace(cool_ms);
        LM3DR_OURS();
    }
    else
    {
        LM3DR_OURS();
        cachebust();
        pace(cool_ms);
        mns = lm3dr_time_mkl(n, K, c2r, in, zm, total, 0);
        pace(g_trial_pace_ms);
        hns = lm3dr_time_mkl(n, K, c2r, in_h, zh, total, 1);
    }
#else
    (void)cool_ms;
    (void)flip;
    LM3DR_OURS();
#endif
#undef LM3DR_OURS
    rmir = (vns > 0 && mns > 0) ? mns / vns : 0;
    rhom = (vns > 0 && hns > 0) ? hns / vns : 0;
    rq = (qns > 0 && hns > 0) ? hns / qns : 0;
    printf("%-11s %-4d %-8s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %8.2fx %9.2e %9.2e %9.2e %9.2e  %s/%s",
           shape, K, c2r ? "lm3dc2r" : "lm3dr2c", vns, qns, mns, hns, rmir, rhom, rq, rel, qrel, mself, xrel,
           vfft_plan_route(h), hq ? vfft_plan_route(hq) : "-");
    if (T > 1)
        printf("  T=%d engaged %ld%s contig pq=%ld plane=%ld grp=%ld%s", T, g_cell_engaged, g_cell_engaged == 0 ? " (SERIAL)" : "",
               eng_pq, eng_plane, eng_grp, (eng_pq == 0 && eng_plane == 0 && eng_grp == 0) ? " (SERIAL)" : "");
    printf("\n");
    if (out)
    {
        char eng[64] = "";
        if (T > 1)
            snprintf(eng, sizeof eng, ",%ld,%ld,%ld,%ld", g_cell_engaged, eng_pq, eng_plane, eng_grp);
        fprintf(out, "%s,%d,%s,lm3dr-oop,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e,%.3e,%s%s\n",
                shape, K, plan_s, vns, qns, mns, hns, rmir, rhom, rq, rel, qrel, vfft_plan_route(h), eng);
        fflush(out);
    }
    free_d(in); free_d(in_h); free_d(S); free_d(S_h); free_d(zm); free_d(zh); free_d(tmp);
    if (hq)
        vfft_destroy(hq);
    vfft_destroy(h);
}

/* ── --lmr2c / --lmc2r (2026-10-15): the LANE-MAJOR 1D REAL BATCH cells, the lane-major design's
 * step 4 (docs/roadmap/lane_major_batch_design.md). The real batch door (DEFAULT = lane-major for a
 * real interleaved batch: x[e K + t] -> z[2 (f K + t)], f < N/2 + 1, and the c2r twin back) against
 * MKL's DFTI REAL with CCE storage on the same layout (NUMBER_OF_TRANSFORMS = K, distance 1, strides
 * K in and out), the loop over K = 1 real plans on contiguous data and MKL's contiguous real batch
 * (distance N for the reals, N/2 + 1 for the bins; the backward takes the bins' distance as its
 * input's) as the contiguous reference. The c2r cell's bins are MKL's own lane-major forward of a
 * random real plane. Correctness: ours against MKL on the identical layout, elementwise. ── */
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE kzbr_dfti(int N, int K, int c2r, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG st[2] = { 0, 1 };
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    {   /* transform-contiguous: the reals at distance N, the bins at N/2 + 1; the backward's INPUT is the bins */
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)(c2r ? N / 2 + 1 : N));
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(c2r ? N : N / 2 + 1));
    }
    else
    {   /* our lane-major contract: sample e (or bin f) of transform t at index e K + t, both domains */
        st[1] = (MKL_LONG)K;
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
    }
    DftiSetValue(d, DFTI_INPUT_STRIDES, st);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, st);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
static void kzbr_mkl_run(DFTI_DESCRIPTOR_HANDLE d, int c2r, const double *in, double *out)
{
    if (c2r) DftiComputeBackward(d, (void *)in, out);
    else     DftiComputeForward(d, (void *)in, out);
}
/* the two-window protocol of kzb_time_mkl / k1z_time_vfft_d, on the real descriptor */
static double kzbr_time_mkl(int N, int K, int c2r, const double *in, double *out, size_t total, int home)
{
    DFTI_DESCRIPTOR_HANDLE d = kzbr_dfti(N, K, c2r, home);
    int reps = reps_for(total), win, t, i, w;
    double best = 1e18;
    if (!d)
        return 0;
    for (w = 0; w < 10; w++)
        kzbr_mkl_run(d, c2r, in, out);
    for (win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = vfft_now_ns();
            pace(K1Z_WINDOW_IDLE_MS);
            do
                kzbr_mkl_run(d, c2r, in, out);
            while (vfft_now_ns() - tw0 < K1Z_WINDOW_IDLE_MS * 1e6 + K1Z_WINDOW_WARM_NS);
        }
        for (t = 0; t < 5; t++)
        {
            double t0;
            if (t)
                pace(g_trial_pace_ms);
            t0 = vfft_now_ns();
            for (i = 0; i < reps; i++)
                kzbr_mkl_run(d, c2r, in, out);
            {
                const double ns = (vfft_now_ns() - t0) / reps;
                if (ns < best)
                    best = ns;
            }
        }
    }
    DftiFreeDescriptor(&d);
    return best;
}
static int kzbr_mkl_ref(int N, int K, int c2r, const double *in, int home, double *out)
{
    DFTI_DESCRIPTOR_HANDLE d = kzbr_dfti(N, K, c2r, home);
    if (!d)
        return 0;
    kzbr_mkl_run(d, c2r, in, out);
    DftiFreeDescriptor(&d);
    return 1;
}
#endif
/* the LOOP arm: K sequential K = 1 real transforms over contiguous data (transform k's reals at k N,
 * its bins at k (N/2 + 1) complex) */
static double kzbr_time_loop(vfft_plan h1, int N, int K, int c2r, double *in_h, double *out_h, size_t total)
{
    const size_t rp = (size_t)N, cp = 2 * ((size_t)N / 2 + 1);
    const vfft_dir_t dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
    int reps = reps_for(total), w, k, t, i;
    double best = 1e18;
#define KZBR_LOOP() do { for (k = 0; k < K; k++) \
        vfft_execute(h1, dir, in_h + (size_t)k * (c2r ? cp : rp), out_h + (size_t)k * (c2r ? rp : cp)); } while (0)
    for (w = 0; w < 10; w++)
        KZBR_LOOP();
    for (t = 0; t < 5; t++)
    {
        double t0;
        if (t)
            pace(g_trial_pace_ms);
        t0 = vfft_now_ns();
        for (i = 0; i < reps; i++)
            KZBR_LOOP();
        {
            const double ns = (vfft_now_ns() - t0) / reps;
            if (ns < best)
                best = ns;
        }
    }
#undef KZBR_LOOP
    return best;
}
static void run_kzbr_cell(int N, int K, int c2r, FILE *out, int cool_ms, int flip)
{
    const int T = g_k1noop_mt ? g_mt : 1, hp1 = N / 2 + 1;
    const char *plan_s = c2r ? "z:lmc2r" : "z:lmr2c";
    const size_t XN = (size_t)N * (size_t)K, ZN = 2 * (size_t)hp1 * (size_t)K;   /* the reals, the bins (doubles) */
    const size_t nin = c2r ? ZN : XN, nout = c2r ? XN : ZN, total = XN;
    const char *W;
    vfft_config_t cfg, c1;
    vfft_plan h, h1;
    double *in, *in_h, *S, *S_h, *zm, *zh;
    double rel = -1.0, lrel = -1.0, vns = 0, mns = 0, hns = 0, lns = 0, rmir, rhom, rloop;
    size_t i, e, t;
    if (N < 4 || (N & 1) || K < 2)
    {
        printf("%-8d K=%-3d   SKIP (the real lane-major batch: an even N >= 4, K >= 2)\n", N, K);
        return;
    }
    if (T == 1)
        bench_pin_one_thread();
    W = k1z_bundle();
    if (!W)
    {
        printf("%-8d K=%-3d   SKIP (front-door bundle unavailable)\n", N, K);
        return;
    }
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = 1;
    cfg.n[0] = N;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = VFFT_BATCH_LANE_MAJOR;   /* the real batch's DEFAULT is lane-major too; said explicitly here */
    cfg.nthreads = T;
    cfg.wisdom_dir = W;
    h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d K=%-3d   vfft_create FAILED\n", N, K);
        return;
    }
    c1 = cfg;
    c1.howmany = 1;
    c1.nthreads = 1;
    c1.batch = VFFT_BATCH_DEFAULT;
    h1 = vfft_create(&c1);
    in = alloc_d(nin + 16); in_h = alloc_d(nin + 16); S = alloc_d(nout + 16); S_h = alloc_d(nout + 16);
    zm = alloc_d(nout + 16); zh = alloc_d(nout + 16);
    srand(42 + N + K + c2r);
    if (!c2r)
    {   /* the reals, lane-major, and the same transforms contiguous */
        for (i = 0; i < XN; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        for (e = 0; e < (size_t)N; e++)
            for (t = 0; t < (size_t)K; t++) in_h[t * (size_t)N + e] = in[e * (size_t)K + t];
    }
    else
    {   /* the bins: a random real plane through MKL's lane-major forward (the DC and Nyquist bins real) */
        double *r = alloc_d(XN + 16);
        for (i = 0; i < XN; i++) r[i] = (double)rand() / RAND_MAX - 0.5;
#ifdef VFFT_HAS_MKL
        if (!kzbr_mkl_ref(N, K, 0, r, 0, in))
#endif
            for (i = 0; i < ZN; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        free_d(r);
        for (e = 0; e < (size_t)hp1; e++)
            for (t = 0; t < (size_t)K; t++)
            {
                in_h[2 * (t * (size_t)hp1 + e)] = in[2 * (e * (size_t)K + t)];
                in_h[2 * (t * (size_t)hp1 + e) + 1] = in[2 * (e * (size_t)K + t) + 1];
            }
    }
#ifdef VFFT_HAS_MKL
    if (kzbr_mkl_ref(N, K, c2r, in, 0, zm))
    {   /* ours vs MKL on the identical layout; the loop arm vs MKL's contiguous output */
        double xe = 0.0, xm = 0.0;
        vfft_execute(h, c2r ? VFFT_BACKWARD : VFFT_FORWARD, in, S);
        for (i = 0; i < nout; i++)
        {
            const double d = fabs(S[i] - zm[i]), m = fabs(zm[i]);
            if (d > xe) xe = d;
            if (m > xm) xm = m;
        }
        rel = xm > 0 ? xe / xm : xe;
        if (h1 && kzbr_mkl_ref(N, K, c2r, in_h, 1, zh))
        {
            double le = 0.0, lm = 0.0;
            {   /* one pass of the loop arm: its output */
                const size_t rp = (size_t)N, cp = 2 * (size_t)hp1;
                int k;
                for (k = 0; k < K; k++)
                    vfft_execute(h1, c2r ? VFFT_BACKWARD : VFFT_FORWARD, in_h + (size_t)k * (c2r ? cp : rp), S_h + (size_t)k * (c2r ? rp : cp));
            }
            for (i = 0; i < nout; i++)
            {
                const double d = fabs(S_h[i] - zh[i]), m = fabs(zh[i]);
                if (d > le) le = d;
                if (m > lm) lm = m;
            }
            lrel = lm > 0 ? le / lm : le;
        }
    }
#endif
    g_cell_engaged = -1;
#define KZBR_OURS() do { \
        \
        { const long e0 = vfft_il2d_col_mt_passes(); \
          vns = k1z_time_vfft_d(h, in, S, total, c2r ? VFFT_BACKWARD : VFFT_FORWARD); \
          if (T > 1) g_cell_engaged = vfft_il2d_col_mt_passes() - e0; } \
        if (h1) { pace(g_trial_pace_ms); lns = kzbr_time_loop(h1, N, K, c2r, in_h, S_h, total); } \
    } while (0)
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = kzbr_time_mkl(N, K, c2r, in, zm, total, 0);
        pace(g_trial_pace_ms);
        hns = kzbr_time_mkl(N, K, c2r, in_h, zh, total, 1);
        cachebust();
        pace(cool_ms);
        KZBR_OURS();
    }
    else
    {
        KZBR_OURS();
        cachebust();
        pace(cool_ms);
        mns = kzbr_time_mkl(N, K, c2r, in, zm, total, 0);
        pace(g_trial_pace_ms);
        hns = kzbr_time_mkl(N, K, c2r, in_h, zh, total, 1);
    }
#else
    (void)cool_ms;
    (void)flip;
    KZBR_OURS();
#endif
#undef KZBR_OURS
    rmir = (vns > 0 && mns > 0) ? mns / vns : 0;
    rhom = (vns > 0 && hns > 0) ? hns / vns : 0;
    rloop = (lns > 0 && hns > 0) ? hns / lns : 0;
    printf("%-8d %-4d %-8s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %7.2fx %9.2e %9.2e  %s",
           N, K, c2r ? "lmc2r" : "lmr2c", vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h));
    if (T > 1)
        printf("  T=%d engaged %ld%s", T, g_cell_engaged, g_cell_engaged == 0 ? " (SERIAL)" : "");
    printf("\n");
    if (out)
    {
        char eng[24] = "";
        if (T > 1)
            snprintf(eng, sizeof eng, ",%ld", g_cell_engaged);
        fprintf(out, "%d,%d,%s,lm-oop,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e,%.3e,%s%s\n",
                N, K, plan_s, vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel, vfft_plan_route(h), eng);
        fflush(out);
    }
    free_d(in); free_d(in_h); free_d(S); free_d(S_h); free_d(zm); free_d(zh);
    if (h1)
        vfft_destroy(h1);
    vfft_destroy(h);
}
/* ════════════════════════════════════════════════════════════════════════
 * --ilmt : TRANSFORM-CONTIGUOUS batch MULTITHREADED vs MKL batched MT.
 *
 * The apples-to-apples MT cell: DFTI with NUMBER_OF_TRANSFORMS=K and
 * DISTANCE=N *is* our transform-contiguous geometry, so both engines see
 * byte-identical memory and compute the same natural-order spectrum (the
 * correctness column is a cross-engine elementwise compare, not a proxy).
 *
 * FOUR arms per cell + a control:
 *   ours MT (nthreads=8) · ours ST (1) · MKL MT (8) · MKL ST (1) · ours MT again
 * so one line carries the headline ratio AND both engines' own scaling AND
 * the noise floor that says whether the headline is real.
 *
 * 🔴 THREAD HYGIENE — the two traps that would silently invalidate this:
 *  (a) OUR pool workers spin on _mm_pause FOREVER (threads.h has no blocktime),
 *      so 7 live workers would steal 7 P-cores from any MKL arm timed while
 *      they exist. Every MKL arm is therefore preceded by
 *      thread_pool_resize(1) — a real teardown, not a flag.
 *  (b) MKL's OpenMP threads spin for KMP_BLOCKTIME (default 200 ms) after a
 *      compute before parking, so our arms need >=300 ms of cool AFTER an MKL
 *      arm. cool_ms is floored at 300 in this mode for exactly that reason.
 * Both are silent-corruption hazards: they cost time, never correctness, so
 * nothing in the output would look wrong.
 *
 * PINNING: mask 0x5555 = logical 0,2,..,14 = the 8 DISTINCT P-cores of a
 * 14900KF (0-15 are 8 P-cores x 2 HT, 16-31 are E-cores). Applied
 * PROCESS-wide before any MKL/OpenMP init so BOTH engines get the same 8
 * cores: our pool already pins caller->0 and workers->2,4,..,14 (threads.h
 * stride 2), and Intel OpenMP respects the process mask at init. Without it
 * MKL would spread across 32 logical CPUs including E-cores and HT siblings
 * and no ratio here would mean anything.
 * ════════════════════════════════════════════════════════════════════════ */
static int g_ilmt = 0;
static int g_zr2c = 0;   /* --zr2c: D2 interleaved r2c/c2r vs MKL real-CCE in-place */

/* THE ONE-THREAD PROTOCOL: the library's measurement scope (vfft.h), entered
 * for the process -- the calling thread pinned to the library's measuring
 * core (the second P-core: logical 2, mask 0x4, on this host), its SMT
 * sibling held, the process at HIGH priority, the machine's measurement lock
 * held. Without the pin a paced sample wakes on whichever core the scheduler
 * picks (cold caches, that core's own frequency ramp) and the pace measures
 * the migration, not the transform; MKL at one thread runs on this same
 * calling thread, so both arms sit on the same core. Every one-thread cell
 * runner calls this once; the threaded runners use ilmt_pin_pcores() instead.
 * VFFT_BENCH_PIN=0 lifts it (the control for "did the pin itself move a
 * number?"): then the library's own races float too. */
#include "bench_scope.h"   /* the gauntlet's switches (VFFT_BENCH_GUARD, VFFT_PCORE_MASK) onto the library's scope */
static void bench_pin_one_thread(void)
{
    static int done = 0;
    const char *e = getenv("VFFT_BENCH_PIN");
    if (done) return;
    done = 1;
    if (e && !strcmp(e, "0")) {
        bench_scope_lifted();
        printf("# one-thread protocol: pin LIFTED (VFFT_BENCH_PIN=0) — the caller floats\n");
        return;
    }
    bench_scope(-1, 1, bench_guard_mode());
    printf("# one-thread protocol: the library's measurement scope at HIGH priority (VFFT_BENCH_PIN=0 lifts)\n");
}

/* one logical CPU per P-core; VFFT_PCORE_MASK overrides. */
static void ilmt_pin_pcores(void)
{
    bench_pin_pcores();   /* shared with the calibrate probe (bench_scope.h) */
}

/* ours: transform-contiguous batch through the FRONT DOOR (one handle, one
 * vfft_execute per rep — the MT slabbing is the library's, not the bench's). */
static double ilmt_time_ours(vfft_plan h, double *z0, double *S, size_t total)
{
    for (int w = 0; w < 10; w++)
        vfft_execute(h, VFFT_FORWARD, z0, S);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_execute(h, VFFT_FORWARD, z0, S);
        double ns = (vfft_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

#ifdef VFFT_HAS_MKL
/* MKL batched, transform-contiguous, at `threads` threads. The thread count
 * is set BEFORE create+commit because DFTI bakes its threading decision at
 * COMMIT time — setting it after would time a descriptor planned for the
 * previous count. Our pool is torn down by the caller first (trap (a)). */
static double ilmt_time_mkl(int N, int K, const double *z0, size_t total,
                            int threads)
{
    mkl_set_num_threads(threads);
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                             (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(d, zi, zo);
        double ns = (vfft_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}
#endif

static void run_ilmt_cell(int N, int K, FILE *out, int cool_ms, int flip)
{
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d K=%-3d   SKIP (front-door bundle unavailable)\n", N, K);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = 1;
    cfg.n[0] = N;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch = VFFT_BATCH_CONTIGUOUS; /* explicit: the axis
                                                       * under test */
    cfg.wisdom_dir = W;
    /* Create the MT handle FIRST (mt_c2c_gate discipline: an MT-first create
     * is what exposed the 2026-07-06 lazily-built-shared-state race). The ST
     * create below leaves the pool as the MT create grew it: a create only
     * grows the pool, and each plan runs at its own count. */
    cfg.nthreads = g_mt;
    vfft_plan hmt = vfft_create(&cfg);
    cfg.nthreads = 1;
    vfft_plan hst = vfft_create(&cfg);
    if (!hmt || !hst)
    {
        printf("%-8d K=%-3d   vfft_create FAILED\n", N, K);
        if (hmt) vfft_destroy(hmt);
        if (hst) vfft_destroy(hst);
        return;
    }

    size_t total = (size_t)N * (size_t)K;
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total);
    srand(1729 + N + K);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    double rel = -1.0, mtst = -1.0;
#ifdef VFFT_HAS_MKL
    { /* cross-engine correctness on the SAME layout, both natural. Also
       * MT-vs-ST bitwise on our side — the shipped gate's arm 7, re-checked
       * here on the exact buffers being timed. */
        double *zh = alloc_d(2 * total), *Sst = alloc_d(2 * total);
        mkl_set_num_threads(1);
        if (kzb_mkl_ref(N, K, z0, total, 1, zh))
        {
            vfft_execute(hmt, VFFT_FORWARD, z0, S);
            vfft_execute(hst, VFFT_FORWARD, z0, Sst);
            double xe = 0.0, xm = 0.0;
            for (size_t i = 0; i < 2 * total; i++)
            {
                double e = fabs(S[i] - zh[i]), m = fabs(zh[i]);
                if (e > xe) xe = e;
                if (m > xm) xm = m;
            }
            rel = xm > 0 ? xe / xm : xe;
            mtst = memcmp(S, Sst, 2 * total * sizeof(double)) == 0 ? 0.0 : 1.0;
            if (mtst != 0.0)
                fprintf(stderr, "ilmt: MT != ST BITWISE at N=%d K=%d — "
                                "timing columns are not comparable\n", N, K);
        }
        free_d(zh);
        free_d(Sst);
    }
#endif

    /* ARMS. Order flips per cell; our pool is town down before every MKL arm
     * (trap (a)) and MKL gets cool_ms>=300 to park before ours (trap (b)). */
    double omt = 0, ost = 0, mmt = 0, mst = 0, octl = 0;
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        thread_pool_resize(1);
        mmt = ilmt_time_mkl(N, K, z0, total, g_mt);
        pace(g_trial_pace_ms);
        mst = ilmt_time_mkl(N, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        omt = ilmt_time_ours(hmt, z0, S, total);
        pace(g_trial_pace_ms);
        thread_pool_resize(1); /* no spinners during the ST arm */
        ost = ilmt_time_ours(hst, z0, S, total);
        pace(g_trial_pace_ms);
        octl = ilmt_time_ours(hmt, z0, S, total); /* control: repeat arm 1 */
    }
    else
    {
        omt = ilmt_time_ours(hmt, z0, S, total);
        pace(g_trial_pace_ms);
        thread_pool_resize(1);
        ost = ilmt_time_ours(hst, z0, S, total);
        cachebust();
        pace(cool_ms);
        mmt = ilmt_time_mkl(N, K, z0, total, g_mt);
        pace(g_trial_pace_ms);
        mst = ilmt_time_mkl(N, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        octl = ilmt_time_ours(hmt, z0, S, total); /* control: repeat arm 1 */
    }
#else
    (void)cool_ms; (void)flip;
    omt = ilmt_time_ours(hmt, z0, S, total);
    thread_pool_resize(1);
    ost = ilmt_time_ours(hst, z0, S, total);
    octl = ilmt_time_ours(hmt, z0, S, total);
#endif
    double r_mkl = (omt > 0 && mmt > 0) ? mmt / omt : 0;   /* >1 = we win  */
    double sc_us = (omt > 0 && ost > 0) ? ost / omt : 0;   /* our scaling  */
    double sc_mk = (mmt > 0 && mst > 0) ? mst / mmt : 0;   /* MKL scaling  */
    /* 🔴 THE HONEST HEADLINE. MKL's MT is a NET LOSS at every cell measured
     * so far (sc_mkl < 1) — its per-call OpenMP fork/join costs ~20us while
     * our spin-pool wakes in ~10ns — so comparing our MT against MKL's MT
     * would be scoring against an option MKL's own users would not pick.
     * Compare against whichever MKL configuration is FASTER. */
    double mbest = (mmt > 0 && mst > 0) ? (mmt < mst ? mmt : mst)
                                        : (mmt > 0 ? mmt : mst);
    double r_best = (omt > 0 && mbest > 0) ? mbest / omt : 0;
    /* control spread: two identical arms, same cell, non-adjacent. A headline
     * delta SMALLER than this is not a result (thermal protocol). */
    double ctl = (omt > 0 && octl > 0)
                     ? fabs(octl - omt) / (omt < octl ? omt : octl) : 0;
    printf("%-7d %-4d %10.0f %10.0f %10.0f %10.0f | %7.2fx %6.2fx %6.2fx %6.2fx %5.1f%% %9.2e %s\n",
           N, K, omt, ost, mmt, mst, r_best, r_mkl, sc_us, sc_mk,
           100.0 * ctl, rel, mtst == 0.0 ? "" : "MT!=ST");
    if (out)
    {
        fprintf(out, "%d,%d,ilmt,%d,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3e,%.0f\n",
                N, K, g_mt, omt, ost, mmt, mst, r_best, r_mkl, sc_us, sc_mk,
                ctl, rel, mtst);
        fflush(out);
    }
    free_d(z0);
    free_d(S);
    vfft_destroy(hmt);
    vfft_destroy(hst);
}

/* ════════════════════════════════════════════════════════════════════════
 * --2dil : the THREE-ARM interleaved-2D scoping cell (M0a of
 * docs/roadmap/fft2d_il_c2c_design.md; mkl_2d_campaign IMPLICATIONS phase
 * 0a — the measurement that decides whether an interleaved 2D problem
 * exists). FRONT DOOR ONLY (vfft_create/vfft_execute) so O-inter pays the
 * real convert-around and wisdom serves what production serves. Arms:
 *   O-inter : VFFT_LAYOUT_INTERLEAVED, INPLACE   (the convert-around, Q2)
 *   M-inter : MKL rank-2 DFTI_COMPLEX (CCE storage default), DFTI_INPLACE
 *             — MKL's measured BEST 2D arm (campaign S3: D < A at 60/68)
 *   M-split : MKL DFTI_REAL_REAL NOT_INPLACE — reproduces the banked
 *             comparison config in the same run
 *   ctl     : memcpy of 2T doubles
 * Per arm: ROUNDS samples (reps_for(T) execs each), arm order REVERSED on
 * odd rounds, cachebust between arms; median + spread reported; a ratio
 * whose distance from 1 is below the ctl spread prints '~' = NOT A RESULT.
 * In-place timing saturates the data toward inf — full AVX2 speed (no
 * assists) — so correctness is gated BEFORE timing on fresh data
 * (roundtrip/T sanity; both O arms are shipped paths gated elsewhere).
 * MKL pinned to 1 thread (the auto-threading law). Wisdom: read-only from
 * $VFFT_WISDOM_DIR else "." (frontdoor-gate law: point it at a SCRATCH
 * copy). Prime dims excluded (fft2d prime = deferred, 127x100 defect).
 * ════════════════════════════════════════════════════════════════════════ */
static double il2d__med(double *v, int n)
{
    double t;
    int i, j;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t; }
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double il2d__spread(const double *v, int n, double med)
{
    double lo = v[0], hi = v[0];
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return med > 0 ? 100.0 * (hi - lo) / med : 0.0;
}

/* ── --2dreal M0 (2026-08-25): the 2D real DOOR comparison. Arms, one
 * process, alternate order per round, medians: O-z = the interleaved door
 * (front door, layout=IL — first-ever measurement of the _z veneer);
 * M-cce = MKL DFTI
 * REAL 2D with CONJUGATE_EVEN_STORAGE=COMPLEX_COMPLEX (the like-for-like
 * CCE arm the published §4 numbers never used). Forward (r2c) only —
 * the c2r mirror is the follow-up. ESTIMATE rigor: both doors serve the
 * same plan family, isolating the DOOR gap; calibrated-plan ratios are
 * a separate question. */
static void run_2dreal_cell(int N1, int N2, int rounds, const char *W)
{
    size_t RN = (size_t)N1 * N2, hp1 = (size_t)(N2 / 2 + 1);
    size_t CN = (size_t)N1 * hp1, i;
    double *x = alloc_d(RN);
    double *z = alloc_d(2 * CN), *ctl = alloc_d(RN);
    double *zn = alloc_d(2 * CN); /* O-nat spectrum (its own comb) */
    double tm[64], tc[64], tn[64];
    int r, nm = 0, nc = 0, nn = 0;
    vfft_plan pn = NULL;
    vfft_config_t c;
    if (!x || !z || !ctl || !zn)
        return;
    srand(17 + N1 + N2);
    for (i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C;
    c.placement = VFFT_PLACEMENT_OUT;
    /* VFFT_ESTIMATE is a planned tier only */
    c.rank = 2; c.n[0] = N1; c.n[1] = N2;
    c.howmany = 1; c.nthreads = 1; c.wisdom_dir = W;
    /* ── O-nat: the native IL tier — THE interleaved serving since M3
     * (the z-veneer door is deleted; an interleaved create IS native or
     * refuses). Math gated by il2d_real_gate; validated here by the
     * pair roundtrip (c2r section). Output scrambled along N1 at nst>1
     * (the tier's contract). */
    c.transform = VFFT_R2C;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    pn = vfft_create(&c);
    if (pn)
        vfft_execute(pn, VFFT_FORWARD, x, zn);
    else
        printf("  %4dx%-4d  O-nat create REFUSED — arm dropped\n",
               N1, N2);
#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE h = 0;
        MKL_LONG dims[2] = { N1, N2 };
        double *cce = alloc_d(RN * 2);
        int mok = 0;
        if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 2, dims) ==
            DFTI_NO_ERROR)
        {
            DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE,
                         DFTI_COMPLEX_COMPLEX);
            DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
        }
        for (r = 0; r < rounds && r < 64; r++)
        {
            int a, order[3] = { 1, 2, 3 };
            if (r & 1)
            {
                order[0] = 3; order[1] = 2; order[2] = 1;
            }
            for (a = 0; a < 3; a++)
            {
                const int arm = order[a];
                size_t reps = 1 + (size_t)(2e5 / (double)(RN + 1));
                double t0, dt;
                size_t k;
                cachebust();
                t0 = vfft_now_ns();
                for (k = 0; k < reps; k++)
                {
                    if (arm == 1 && mok)
                        DftiComputeForward(h, (void *)x, cce);
                    else if (arm == 2)
                        memcpy(ctl, x, RN * 8);
                    else if (arm == 3 && pn)
                        vfft_execute(pn, VFFT_FORWARD, x, zn);
                }
                dt = (vfft_now_ns() - t0) / (double)reps;
                if (arm == 1 && mok) tm[nm++] = dt;
                else if (arm == 2) tc[nc++] = dt;
                else if (arm == 3 && pn) tn[nn++] = dt;
            }
        }
        if (h)
            DftiFreeDescriptor(&h);
        free_d(cce);
    }
#else
    (void)r;
#endif
    {
        double mm = nm ? il2d__med(tm, nm) : 0, mc = il2d__med(tc, nc);
        double mn = nn ? il2d__med(tn, nn) : 0;
        printf("  %4dx%-4d  r2c  ctl %8.0f (%4.1f%%) | "
               "O-nat %9.0f (%4.1f%%) | "
               "M-cce %9.0f | nat xMKL %.2f\n",
               N1, N2, mc, il2d__spread(tc, nc, mc),
               mn, nn ? il2d__spread(tn, nn, mn) : 0, mm,
               (mm > 0 && mn > 0) ? mm / mn : 0);
    }
    /* ── the c2r MIRROR: backward through the interleaved door, input = the
     * fwd run's own spectrum. C2R plans are backward-only (VFFT_BACKWARD). */
    {
        double *xr = alloc_d(RN);
        vfft_plan pnc = NULL;
        double bm[64], bn[64];
        int bnm = 0, bnn = 0;
        if (!xr)
            goto done;
        /* O-nat c2r: consumes the native pair's own comb (zn); the pair
         * roundtrip IS the validation (fwd proven by il2d_real_gate). */
        if (pn)
        {
            c.transform = VFFT_C2R;
            c.layout = VFFT_LAYOUT_INTERLEAVED;
            pnc = vfft_create(&c);
            if (pnc)
            {
                double d3 = 0, sc3 = (double)N1 * N2;
                vfft_execute(pnc, VFFT_BACKWARD, zn, xr);
                for (i = 0; i < RN; i++)
                {
                    double a3 = fabs(xr[i] / sc3 - x[i]);
                    if (a3 > d3) d3 = a3;
                }
                if (d3 > 1e-9)
                {
                    printf("  %4dx%-4d  O-nat pair RT FAIL %.1e — arm "
                           "dropped\n", N1, N2, d3);
                    vfft_destroy(pnc);
                    pnc = NULL;
                }
            }
        }
#ifdef VFFT_HAS_MKL
        {
            DFTI_DESCRIPTOR_HANDLE hb = 0;
            MKL_LONG dims2[2] = { N1, N2 };
            double *cce2 = alloc_d(RN * 2);
            int mok2 = 0;
            if (DftiCreateDescriptor(&hb, DFTI_DOUBLE, DFTI_REAL, 2,
                                     dims2) == DFTI_NO_ERROR)
            {
                DftiSetValue(hb, DFTI_CONJUGATE_EVEN_STORAGE,
                             DFTI_COMPLEX_COMPLEX);
                DftiSetValue(hb, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                mok2 = (DftiCommitDescriptor(hb) == DFTI_NO_ERROR);
            }
            if (mok2 && cce2)
            {
                /* MKL bwd VALIDATION (the --2dc2r audit-gap class): its
                 * own fwd fills cce2, its bwd must invert to N1*N2*x. */
                double d = 0, sc2 = (double)N1 * N2;
                DftiComputeForward(hb, (void *)x, cce2);
                DftiComputeBackward(hb, cce2, xr);
                for (i = 0; i < RN; i++)
                {
                    double a2 = fabs(xr[i] / sc2 - x[i]);
                    if (a2 > d) d = a2;
                }
                if (d > 1e-9)
                {
                    printf("  %4dx%-4d  MKL bwd INVALID %.1e — arm "
                           "dropped\n", N1, N2, d);
                    mok2 = 0;
                }
            }
            for (r = 0; r < rounds && r < 64; r++)
            {
                int a2, order2[2] = { 1, 2 };
                if (r & 1)
                {
                    order2[0] = 2; order2[1] = 1;
                }
                for (a2 = 0; a2 < 2; a2++)
                {
                    const int arm = order2[a2];
                    size_t reps = 1 + (size_t)(2e5 / (double)(RN + 1));
                    double t0, dt;
                    size_t k;
                    cachebust();
                    t0 = vfft_now_ns();
                    for (k = 0; k < reps; k++)
                    {
                        if (arm == 1 && mok2)
                            DftiComputeBackward(hb, cce2, xr);
                        else if (arm == 2 && pnc)
                            vfft_execute(pnc, VFFT_BACKWARD, zn, xr);
                    }
                    dt = (vfft_now_ns() - t0) / (double)reps;
                    if (arm == 1 && mok2) bm[bnm++] = dt;
                    else if (arm == 2 && pnc) bn[bnn++] = dt;
                }
            }
            if (hb)
                DftiFreeDescriptor(&hb);
            free_d(cce2);
        }
#endif
        {
            double mm2 = bnm ? il2d__med(bm, bnm) : 0;
            double mn2 = bnn ? il2d__med(bn, bnn) : 0;
            printf("  %4dx%-4d  c2r  "
                   "O-nat %9.0f (%4.1f%%) | M-cce %9.0f | "
                   "nat xMKL %.2f\n",
                   N1, N2, mn2,
                   bnn ? il2d__spread(bn, bnn, mn2) : 0, mm2,
                   (mm2 > 0 && mn2 > 0) ? mm2 / mn2 : 0);
        }
        free_d(xr);
        if (pnc) vfft_destroy(pnc);
    }
done:
    if (pn) vfft_destroy(pn);
    free_d(x); free_d(z); free_d(ctl);
    free_d(zn);
}

static void run_2dil_cell(int N1, int N2, int rounds, const char *W)
{
    size_t T = (size_t)N1 * N2, i;
    double *z = alloc_d(2 * T);                     /* O-inter z        */
    double *mz = alloc_d(2 * T);                    /* M-inter z        */
    double *xr = alloc_d(T), *xi = alloc_d(T);      /* M-split input    */
    double *mr = alloc_d(T), *mi = alloc_d(T);      /* M-split output   */
    double *cs = alloc_d(2 * T), *cd = alloc_d(2 * T); /* ctl memcpy    */
    double *zn = alloc_d(2 * T);                    /* O-native z       */
    double smp[6][64];
    double med[6], spr[6];
    double rti = -1;
    int have[6] = { 0, 1, 0, 0, 0, 1 }; /* (slot 0: the deleted O-split arm), Oi, On, Mi, Ms, ctl */
    vfft_plan hi = NULL, hn = NULL;
    int r, a0, a, k;
    if (rounds > 64) rounds = 64;
    fprintf(stderr, "[2dil] %dx%d create (wisdom miss => calibrates here)...\n", N1, N2);
    srand(11 + N1 + N2);
    for (i = 0; i < T; i++) {
        xr[i] = (double)rand() / RAND_MAX - 0.5;
        xi[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (i = 0; i < 2 * T; i++) cs[i] = (double)rand() / RAND_MAX - 0.5;
    {
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_PLACEMENT_IN;
        cfg.rank = 2;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_NATURAL;
        cfg.nthreads = 1;
        cfg.wisdom_dir = W;
        /* benches never mutate the store */
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.order = VFFT_ORDER_NATURAL;     /* the interleaved tier's own */
        hi = vfft_create(&cfg);
        /* O-native: same caller contract, native tier opted in for THIS
         * create only. Engagement is verified below, not assumed. */
#ifdef _WIN32
        _putenv("VFFT_IL2D_NATIVE=1");
#else
        putenv("VFFT_IL2D_NATIVE=1");
#endif
        hn = vfft_create(&cfg);
#ifdef _WIN32
        _putenv("VFFT_IL2D_NATIVE=");
#else
        putenv("VFFT_IL2D_NATIVE=0");
#endif
    }
    if (!hi) {
        printf("  %5dx%-5d  create FAIL (hi=%p)\n", N1, N2,
               (void *)hi);
        return;
    }
    fprintf(stderr, "[2dil] %dx%d created; gating + timing %d rounds...\n", N1, N2, rounds);
    /* correctness pre-gate, fresh data: roundtrip/T (shipped paths;
     * forward elementwise gates live in the 2D gate battery) */
    for (i = 0; i < T; i++) { z[2 * i] = xr[i]; z[2 * i + 1] = xi[i]; }
    vfft_execute(hi, VFFT_FORWARD, z, z);
    vfft_execute(hi, VFFT_BACKWARD, z, z);
    rti = 0;
    for (i = 0; i < T; i++) {
        double a1 = fabs(z[2 * i] / (double)T - xr[i]);
        double b1 = fabs(z[2 * i + 1] / (double)T - xi[i]);
        if (a1 > rti) rti = a1;
        if (b1 > rti) rti = b1;
    }
    /* O-native engagement proof (mt_results_need_engagement_proof adapted):
     * the native tier serves NATURAL, the wrapper serves SCRAMBLED — one
     * fwd on each, identical outputs => NOT engaged => arm absent. */
    if (hn) {
        double dmax = 0.0;
        for (i = 0; i < T; i++) { zn[2 * i] = xr[i]; zn[2 * i + 1] = xi[i]; }
        vfft_execute(hn, VFFT_FORWARD, zn, zn);
        for (i = 0; i < T; i++) { z[2 * i] = xr[i]; z[2 * i + 1] = xi[i]; }
        vfft_execute(hi, VFFT_FORWARD, z, z);
        for (i = 0; i < 2 * T; i++) {
            double d = fabs(zn[i] - z[i]);
            if (d > dmax) dmax = d;
        }
        have[2] = (dmax > 0.0);
        if (!have[2]) { vfft_destroy(hn); hn = NULL; }
    }
#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE hMi = 0, hMs = 0;
        MKL_LONG dims[2];
        dims[0] = N1;
        dims[1] = N2;
        if (DftiCreateDescriptor(&hMi, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims)
                == DFTI_NO_ERROR) {
            DftiSetValue(hMi, DFTI_PLACEMENT, DFTI_INPLACE);
            have[3] = (DftiCommitDescriptor(hMi) == DFTI_NO_ERROR);
        }
        if (DftiCreateDescriptor(&hMs, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims)
                == DFTI_NO_ERROR) {
            DftiSetValue(hMs, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
            DftiSetValue(hMs, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            have[4] = (DftiCommitDescriptor(hMs) == DFTI_NO_ERROR);
        }
        for (i = 0; i < 2 * T; i++) mz[i] = cs[i];
        for (r = 0; r < rounds; r++) {
            for (a0 = 0; a0 < 6; a0++) {
                int reps = reps_for(T);
                double t0, ns;
                a = (r & 1) ? 5 - a0 : a0;
                if (!have[a]) continue;
                cachebust();
                t0 = vfft_now_ns();
                for (k = 0; k < reps; k++) {
                    switch (a) {
                    case 1: vfft_execute(hi, VFFT_FORWARD, z, z); break;
                    case 2: vfft_execute(hn, VFFT_FORWARD, zn, zn); break;
                    case 3: DftiComputeForward(hMi, mz); break;
                    case 4: DftiComputeForward(hMs, xr, xi, mr, mi); break;
                    case 5: memcpy(cd, cs, 2 * T * 8); break;
                    }
                }
                ns = (vfft_now_ns() - t0) / reps;
                smp[a][r] = ns;
            }
        }
        if (hMi) DftiFreeDescriptor(&hMi);
        if (hMs) DftiFreeDescriptor(&hMs);
    }
#else
    for (r = 0; r < rounds; r++) {
        for (a0 = 0; a0 < 6; a0++) {
            int reps = reps_for(T);
            double t0, ns;
            a = (r & 1) ? 5 - a0 : a0;
            if (!have[a]) continue;
            cachebust();
            t0 = vfft_now_ns();
            for (k = 0; k < reps; k++) {
                switch (a) {
                case 1: vfft_execute(hi, VFFT_FORWARD, z, z); break;
                case 2: vfft_execute(hn, VFFT_FORWARD, zn, zn); break;
                case 5: memcpy(cd, cs, 2 * T * 8); break;
                }
            }
            ns = (vfft_now_ns() - t0) / reps;
            smp[a][r] = ns;
        }
    }
#endif
    for (a = 0; a < 6; a++) {
        if (!have[a]) { med[a] = 0; spr[a] = 0; continue; }
        med[a] = il2d__med(smp[a], rounds); /* sorts in place */
        spr[a] = il2d__spread(smp[a], rounds, med[a]);
    }
    {
        const double cspr = spr[5]; /* ctl spread %, the noise floor */
        printf("  %5dx%-5d rt %.1e | ctl %9.0f (%4.1f%%) | "
               "O-inter %10.0f (%4.1f%%)",
               N1, N2, rti, med[5], spr[5],
               med[1], spr[1]);
        if (have[2])
            printf(" | O-NATIVE %10.0f (%4.1f%%)", med[2], spr[2]);
        else
            printf(" | O-NATIVE     absent");
#ifdef VFFT_HAS_MKL
        printf(" | M-inter %10.0f (%4.1f%%) | M-split %10.0f (%4.1f%%)\n",
               med[3], spr[3], med[4], spr[4]);
        if (med[1] > 0 && med[3] > 0) {
            double q1 = med[3] / med[1]; /* O-inter xMKLcce: >1 = we win */
            double q3 = med[4] / med[3]; /* banked-config vs MKL's best   */
            printf("        O-inter xMKLcce %.2f%s | M-split/M-inter %.2f",
                   q1, fabs(1 - q1) * 100 < cspr ? "~" : "", q3);
            if (have[2] && med[2] > 0)
                printf(" | O-NATIVE xMKLcce %.2f%s, native uplift %.2f%s",
                       med[3] / med[2],
                       fabs(1 - med[3] / med[2]) * 100 < cspr ? "~" : "",
                       med[1] / med[2],
                       fabs(1 - med[1] / med[2]) * 100 < cspr ? "~" : "");
            printf("\n");
        }
#else
        printf("  (no MKL)\n");
#endif
    }
    vfft_destroy(hi);
    if (hn) vfft_destroy(hn);
    free_d(z); free_d(mz); free_d(zn);
    free_d(xr); free_d(xi); free_d(mr); free_d(mi);
    free_d(cs); free_d(cd);
}

/* ════════════════════════════════════════════════════════════════════════
 * --3dil : the rank-3 INTERLEAVED c2c tier (transforms/fftnd/fftnd_il.h)
 * vs MKL DFTI 3D, OUT OF PLACE on both sides (the tier's contract). The
 * split rank-N tier is NOT an arm (owner, 2026-09-06: "split is not our
 * concern. IL is what matters"; its create calibrates, too). Arms:
 *   O-NATIVE = vfft 3D INTERLEAVED OOP DEFAULT — the per-plane structure
 *              (2D child vs flat axis-1 + rows) replays from wisdom; a
 *              cold store races it at create (stderr says so);
 *   M-inter  = DFTI 3D COMPLEX_COMPLEX NOT_INPLACE (MKL's CCE, its best);
 *   M-split  = DFTI 3D REAL_REAL NOT_INPLACE (shows CCE is MKL's best);
 *   ctl      = memcpy of the cube (the noise floor).
 * Protocol as --2dil: cachebust before every sample, reps_for(T)
 * executions per sample, arm order alternated on odd rounds, median +
 * spread; '~' = a delta below the ctl spread (NOT A RESULT). Correctness
 * pre-gate: roundtrip bwd(fwd(x))/T == x (the forward is gated elementwise
 * by ilnd_probe: naive-DFT spot bins found across the two digit-reversed
 * column axes).
 * ════════════════════════════════════════════════════════════════════════ */
long vfft_ilnd_mt_passes(void); /* vfft_diagnostics.h: the 3D IL MT engagement counter */
#ifdef VFFT_HAS_MKL
/* TRAP (c) of the two-team protocol (found 2026-09-07 on the 3D cells): our
 * pool pins the CALLER to logical core 0 when it arms, and Windows threads
 * inherit the creating thread's affinity — so an OpenMP team MKL creates
 * AFTER our first threaded create lands entirely on core 0 (measured: 3D
 * CCE 16^3 at T=8 = 34 ms instead of 3 us). MKL's team is therefore
 * created HERE, before any vfft plan pins anything; it persists for the
 * process (parked between arms by KMP_BLOCKTIME). */
static void mkl3d_warm_team(int T)
{
    DFTI_DESCRIPTOR_HANDLE h = 0;
    MKL_LONG dims[3] = { 16, 16, 16 };
    double *x = alloc_d(2 * 4096), *y = alloc_d(2 * 4096);
    size_t i;
    for (i = 0; i < 2 * 4096; i++) x[i] = 1.0;
    mkl_set_num_threads(T);
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims) == DFTI_NO_ERROR) {
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (DftiCommitDescriptor(h) == DFTI_NO_ERROR)
            for (i = 0; i < 4; i++) DftiComputeForward(h, x, y);
        DftiFreeDescriptor(&h);
    }
    free_d(x); free_d(y);
}
#endif
static void run_3dil_cell(int N1, int N2, int N3, int rounds, const char *W, int T)
{
    size_t TN = (size_t)N1 * N2 * N3, i;
    double *xr = alloc_d(TN), *xi = alloc_d(TN);         /* M-split input   */
    double *z = alloc_d(2 * TN), *zo = alloc_d(2 * TN);  /* O-NATIVE in/out */
    double *mz = alloc_d(2 * TN), *mo = alloc_d(2 * TN); /* M-inter in/out  */
    double *mr = alloc_d(TN), *mi = alloc_d(TN);         /* M-split output  */
    double *cs = alloc_d(2 * TN), *cd = alloc_d(2 * TN); /* ctl memcpy      */
    double smp[4][64];
    double med[4], spr[4];
    double rtn = -1;
    int have[4] = { 0, 0, 0, 1 }; /* On, Mi, Ms, ctl */
    vfft_plan hn = NULL;
    int r, a0, a, k;
#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE hMi = 0, hMs = 0;
#endif
    if (rounds > 64) rounds = 64;
    if (T <= 1) bench_pin_one_thread();   /* the one-thread protocol (once per process) */
    fprintf(stderr, "[3dil] %dx%dx%d create (wisdom miss => races here)...\n", N1, N2, N3);
    srand(17 + N1 + N2 + N3);
    for (i = 0; i < TN; i++) {
        xr[i] = (double)rand() / RAND_MAX - 0.5;
        xi[i] = (double)rand() / RAND_MAX - 0.5;
        z[2 * i] = xr[i];
        z[2 * i + 1] = xi[i];
    }
    for (i = 0; i < 2 * TN; i++) cs[i] = (double)rand() / RAND_MAX - 0.5;
    {
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_PLACEMENT_OUT;
        cfg.rank = 3;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.n[2] = N3;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_NATURAL; /* natural: the like-for-like order (MKL's output is natural) */
        cfg.nthreads = T; /* --mt: the plan's thread snapshot; MKL gets the same T */
        cfg.wisdom_dir = W;
        /* benches never mutate the store */
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        hn = vfft_create(&cfg);
    }
    have[0] = (hn != NULL);
    fprintf(stderr, "[3dil] %dx%dx%d created (%s); gating + timing %d rounds...\n",
            N1, N2, N3, hn ? "native" : "REFUSED", rounds);
    /* roundtrip pre-gate on fresh data (mo as the bwd scratch) */
    if (hn) {
        vfft_execute(hn, VFFT_FORWARD, z, zo);
        vfft_execute(hn, VFFT_BACKWARD, zo, mo);
        rtn = 0;
        for (i = 0; i < 2 * TN; i++) {
            double d = fabs(mo[i] / (double)TN - z[i]);
            if (d > rtn) rtn = d;
        }
    }
#ifdef VFFT_HAS_MKL
    {
        MKL_LONG dims[3];
        dims[0] = N1;
        dims[1] = N2;
        dims[2] = N3;
        if (DftiCreateDescriptor(&hMi, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims) == DFTI_NO_ERROR) {
            DftiSetValue(hMi, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            have[1] = (DftiCommitDescriptor(hMi) == DFTI_NO_ERROR);
        }
        if (DftiCreateDescriptor(&hMs, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims) == DFTI_NO_ERROR) {
            DftiSetValue(hMs, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
            DftiSetValue(hMs, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            have[2] = (DftiCommitDescriptor(hMs) == DFTI_NO_ERROR);
        }
        for (i = 0; i < 2 * TN; i++) mz[i] = z[i];
    }
#endif
    {
    const long eng0 = vfft_ilnd_mt_passes();
    long nexec = 0;
    int prev_mkl = 0;
    for (r = 0; r < rounds; r++) {
        for (a0 = 0; a0 < 4; a0++) {
            int reps = reps_for(TN);
            double t0, ns;
            a = (r & 1) ? 3 - a0 : a0;
            if (!have[a]) continue;
            if (g_trial_pace_ms > 0)
                pace(g_trial_pace_ms); /* the per-SAMPLE pace (env VFFT_TRIAL_PACE_MS): thermal
                                        * headroom between arms. BEFORE the cachebust and the
                                        * untimed warm-up, never right before the timed region:
                                        * whatever the sleep cost the core (frequency, caches,
                                        * a parked team) is paid in the warm-up, not the reps */
            cachebust();
            if (T <= 1 && g_trial_pace_ms > 0) {
                /* paced one-thread samples: the same >= 5 ms untimed warm-up
                 * on every arm, so the pace never lands in the timed reps */
                const double tw = vfft_now_ns();
                do {
                    switch (a) {
                    case 0: vfft_execute(hn, VFFT_FORWARD, z, zo); nexec++; break;
#ifdef VFFT_HAS_MKL
                    case 1: DftiComputeForward(hMi, mz, mo); break;
                    case 2: DftiComputeForward(hMs, xr, xi, mr, mi); break;
#endif
                    default: memcpy(cd, cs, 2 * TN * 8); break;
                    }
                } while (vfft_now_ns() - tw < 5e6);
            }
            if (T > 1) {
                /* THREAD HYGIENE (the --ilmt traps): MKL's OpenMP team spins
                 * KMP_BLOCKTIME (200 ms) after a compute, so our arm after an
                 * MKL arm gets 300 ms of cool first, outside the timed region.
                 * The library's pool is the plan's (config.nthreads): this TU
                 * includes threads.h and therefore owns a SECOND copy of the
                 * pool state, which it never sizes (trap (d), found 2026-09-07:
                 * the bench's own startup pool = 7 idle spinners on the plan's
                 * worker cores, every threaded arm in the process 100x slow). */
                if (a == 0 && prev_mkl)
                    pace(300);
                prev_mkl = (a == 1 || a == 2);
                /* UNTIMED warm-up of >= 5 ms per sample, both sides alike:
                 * the rebuilt pool's workers start up, MKL's parked team
                 * wakes, each worker's cache partition settles, and the
                 * caller core (asleep through pace) is back at full clock.
                 * Measured 2026-09-07: a threaded plan's first ~2-3 ms of
                 * executes after a pool rebuild run 1.5-5x slow (81x27x27:
                 * round-0 mean 57-223 us, steady 40 us), so one warm
                 * execute left the timed reps inside that transient. */
                {
                    const double tw = vfft_now_ns();
                    do {
                        switch (a) {
                        case 0: vfft_execute(hn, VFFT_FORWARD, z, zo); nexec++; break;
#ifdef VFFT_HAS_MKL
                        case 1: DftiComputeForward(hMi, mz, mo); break;
                        case 2: DftiComputeForward(hMs, xr, xi, mr, mi); break;
#endif
                        default: break;
                        }
                    } while (a != 3 && vfft_now_ns() - tw < 5e6);
                }
            }
            t0 = vfft_now_ns();
            for (k = 0; k < reps; k++) {
                switch (a) {
                case 0: vfft_execute(hn, VFFT_FORWARD, z, zo); nexec++; break;
#ifdef VFFT_HAS_MKL
                case 1: DftiComputeForward(hMi, mz, mo); break;
                case 2: DftiComputeForward(hMs, xr, xi, mr, mi); break;
#endif
                case 3: memcpy(cd, cs, 2 * TN * 8); break;
                }
            }
            ns = (vfft_now_ns() - t0) / reps;
            smp[a][r] = ns;
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[3dil] r%d arm%d reps=%d %.0f ns\n", r, a, reps, ns);
        }
    }
    /* the engagement proof (mt_results_need_engagement_proof): a threaded
     * plan whose counter did not move every execute ran SERIAL */
    if (TN > 1)
        printf("  [3dil] %dx%dx%d TN=%d: engaged %ld of %ld executes%s\n", N1, N2, N3, T,
               vfft_ilnd_mt_passes() - eng0, nexec,
               (vfft_ilnd_mt_passes() - eng0) == nexec ? "" :
               (vfft_ilnd_mt_passes() - eng0) == 0 ? " (SERIAL verdict or declined)" : " (PARTIAL)");
    }
#ifdef VFFT_HAS_MKL
    if (hMi) DftiFreeDescriptor(&hMi);
    if (hMs) DftiFreeDescriptor(&hMs);
#endif
    for (a = 0; a < 4; a++) {
        if (!have[a]) { med[a] = 0; spr[a] = 0; continue; }
        med[a] = il2d__med(smp[a], rounds); /* sorts in place */
        spr[a] = il2d__spread(smp[a], rounds, med[a]);
    }
    {
        const double cspr = spr[3]; /* ctl spread %, the noise floor */
        printf("  %4dx%-4dx%-4d rt %.1e | ctl %9.0f (%4.1f%%) |",
               N1, N2, N3, rtn, med[3], spr[3]);
        if (have[0]) printf(" O-NATIVE %10.0f (%4.1f%%)", med[0], spr[0]);
        else         printf(" O-NATIVE    REFUSED");
#ifdef VFFT_HAS_MKL
        printf(" | M-inter %10.0f (%4.1f%%) | M-split %10.0f (%4.1f%%)\n",
               med[1], spr[1], med[2], spr[2]);
        if (have[0] && med[1] > 0) {
            double q1 = med[1] / med[0]; /* O-NATIVE xMKLcce: >1 = we win */
            printf("        O-NATIVE xMKLcce %.2f%s", q1, fabs(1 - q1) * 100 < cspr ? "~" : "");
            if (med[2] > 0) printf(" | M-split/M-inter %.2f", med[2] / med[1]);
            printf("\n");
        }
#else
        printf("  (no MKL)\n");
#endif
    }
    if (hn) vfft_destroy(hn);
    free_d(xr); free_d(xi);
    free_d(z); free_d(zo); free_d(mz); free_d(mo);
    free_d(mr); free_d(mi); free_d(cs); free_d(cd);
}

/* ════════════════════════════════════════════════════════════════════════
 * --zr2c : Phase 1 of DESIGN_interleaved_r2c.md — the first honest ours/MKL
 * table for K=1 INTERLEAVED real transforms, both directions.
 *   OURS (D2): x[N] ==reinterpret(0 work)==> z[N/2] -> front-door IL c2c(N/2)
 *              NATURAL OOP fwd -> _zr2c_fold_fwd -> CCE.  c2r is the mirror:
 *              _zr2c_fold_bwd -> c2c(N/2) bwd -> N*x.
 *   MKL: DFTI_REAL, CCE=COMPLEX_COMPLEX, DFTI_INPLACE — its BEST arm
 *        (CONCLUSIONS V6, 8-21%% over its OOP), padded N+2 plane; the
 *        backward gets its own descriptor (same-handle direction reuse is a
 *        banned shape since the 2026-08-13 --c2r OOB).
 *   GATES (per direction, never a roundtrip): fwd = cross-engine elementwise
 *   (both engines emit the same natural CCE — the --k1noop precedent; a naive
 *   DFT is impractical at 65536); c2r = each engine's backward fed MKL's
 *   reference spectrum and checked vs N*x elementwise.
 *   MEDIANS of 5 trials (house law). MKL reps run in place on junk after the
 *   first rep — dense transforms are data-oblivious; FTZ/DAZ regime. ── */
static double _zr2c_med5(double t[5])
{
    for (int i = 1; i < 5; i++)
    {
        double v = t[i]; int j = i - 1;
        while (j >= 0 && t[j] > v) { t[j + 1] = t[j]; j--; }
        t[j + 1] = v;
    }
    return t[2];
}
#define ZR2C_TIME(dst, BODY) do {                                   \
        for (int w_ = 0; w_ < 3; w_++) { BODY; }                    \
        for (int t_ = 0; t_ < 5; t_++) {                            \
            if (t_) pace(g_trial_pace_ms);                          \
            double t0_ = vfft_now_ns();                       \
            for (int i_ = 0; i_ < reps; i_++) { BODY; }             \
            (dst)[t_] = (vfft_now_ns() - t0_) / reps;         \
        }                                                           \
    } while (0)

/* ══════════════════════════════════════════════════════════════════════════
 * FRONT-DOOR zr2c ARMS (audit finding G2, 2026-08-22)
 *
 * 🔴 Until now NO mode in this bench ever set cfg.transform to VFFT_R2C or
 * VFFT_C2R. run_zr2c_cell above hand-assembles the composite: it creates a
 * C2C plan at N/2 and calls _zr2c_fold_fwd/_bwd itself. Three consequences:
 * the route-1 memcpy and the route-0 scratch hop were never timed, a wrong
 * route verdict had no observable effect anywhere in the bench, and the
 * IN-PLACE real shapes -- the library's only in-place real FFT -- had never
 * been timed at all.
 *
 * This block is ADDITIVE on purpose. The hand-built arms above stay, because
 * running both in the same process is the sanity check: route-0 OOP c2r here
 * should land near the hand arm's c2r, since that hand shape already equals
 * what the executor does. If it does not, this code is wrong, not the old
 * number. Deleting the old arms would have thrown that check away.
 *
 * Arms: {r2c, c2r} x {OOP, IN-PLACE} x {route 0, route 1, wisdom} = 12.
 * Route is a CREATE-TIME decision (vfft.c reads VFFT_ZR2C_ROUTE at the top of
 * _zr2c_build), so the env is set before vfft_create, never inside a timed
 * body -- and no vfft_create is ever inside one.
 *
 * Own CSV, own filename: the columns differ from the 16-column zr2c schema,
 * and appending a different width into the banked file would corrupt it.
 * ═════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double ns;       /* median-of-5; 0 = NOT TIMED                          */
    double err;      /* gate rel err; -1 = ungated (no MKL reference)        */
    int    built;    /* vfft_create succeeded                               */
    int    refused;  /* execute left the destination untouched              */
} zr_arm_t;

/* route index: 0 = force child route 0, 1 = force route 1, 2 = wisdom picks.
 * 🔴 Static NON-const: POSIX putenv keeps the caller's pointer. And only
 * ever "0", "1" or the empty form -- _zr2c_build tests `e && e[0]` then
 * atoi(), so a word like "w" would silently read as route 0. */
static void zr_route_env(int ri)
{
    static char e0[] = "VFFT_ZR2C_ROUTE=0";
    static char e1[] = "VFFT_ZR2C_ROUTE=1";
    static char eW[] = "VFFT_ZR2C_ROUTE=";
    putenv(ri == 0 ? e0 : (ri == 1 ? e1 : eW));
}
static const char *ZR_RT[3] = { "r0", "r1", "W " };

/* One front-door arm, gated then timed.
 *
 * REFUSAL DETECTION. vfft_execute returns void, so there is no status to
 * check: a refused call is indistinguishable from a completed one except
 * that it leaves the destination UNTOUCHED. So seed the destination with a
 * known pattern and compare after -- an in-place plan called wrongly, or any
 * signature refusal, shows up as "destination never changed" instead of as a
 * suspiciously fast timing. Gate-before-time catches it a second way.
 *
 * The parameter MUST be named `reps`: ZR2C_TIME captures it by name. */
static void zr_fd_arm(zr_arm_t *a, int N, const char *W, int is_c2r, int ip,
                      int ri, const double *x, const double *bsrc,
                      int haveref, double *dst, int reps)
{
    const size_t xs = (size_t)N + 2;
    const int half = N / 2;
    const double *src = is_c2r ? bsrc : x;
    vfft_config_t cfg;
    vfft_plan p;
    size_t i;

    a->ns = 0; a->err = -1; a->built = 0; a->refused = 0;
    if (is_c2r && !bsrc) return;

    zr_route_env(ri);                      /* create-time, outside all timing */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = is_c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    cfg.rank      = 1;
    cfg.n[0]      = N;                     /* the REAL N, not half */
    cfg.howmany   = 1;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads  = 1;
    cfg.wisdom_dir = W;
    /* cfg.order left 0: real spectra are natural by definition.
     * cfg.wisdom_write left 0: A BENCH NEVER BANKS. */
    p = vfft_create(&cfg);
    if (!p) return;                        /* built stays 0 -> named sentinel */
    a->built = 1;

    /* ---- gate, one shot, BEFORE any timing ---- */
    if (ip) {
        memcpy(dst, src, 8 * xs);
        vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD, dst, dst);
        /* refusal shows as "never moved" */
        a->refused = (memcmp(dst, src, 8 * xs) == 0);
    } else {
        for (i = 0; i < xs; i++) dst[i] = -1.0e300;   /* poison */
        vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD, (double *)src, dst);
        a->refused = (dst[0] == -1.0e300);
    }
    if (a->refused) { vfft_destroy(p); return; }

    if (haveref) {
        double w = 0, m = 0;
        if (is_c2r) {                      /* backward vs N*x, never a roundtrip */
            for (i = 0; i < (size_t)N; i++) {
                double d = fabs(dst[i] - (double)N * x[i]);
                double q = fabs(x[i]);
                if (d > w) w = d;
                if (q > m) m = q;
            }
            a->err = m > 0 ? w / ((double)N * m) : w;
        } else {                           /* forward vs the CCE reference */
            for (i = 0; i < (size_t)(2 * (half + 1)); i++) {
                double d = fabs(dst[i] - bsrc[i]);
                double q = fabs(bsrc[i]);
                if (d > w) w = d;
                if (q > m) m = q;
            }
            a->err = m > 0 ? w / m : w;
        }
        if (!(a->err < 1e-9)) { vfft_destroy(p); return; }  /* gate fail: ns=0 */
    }

    /* ---- time. Seed ONCE, outside the macro: a per-rep memcpy would tax
     * this arm and nothing else in the file (the MKL arms seed outside too). */
    {
        double t[5];
        if (ip) {
            memcpy(dst, src, 8 * xs);
            ZR2C_TIME(t, vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD, dst, dst));
        } else {
            ZR2C_TIME(t, vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD, (double *)src, dst));
        }
        a->ns = _zr2c_med5(t);
    }
    vfft_destroy(p);
}

/* The 12-arm front-door cell. `rot` rotates the visiting order: with 12 arms
 * a fixed order always leaves the same arm hottest, and `flip` only rotates
 * ENGINES, not arms. rot is printed so the order is visible, not implicit. */
static void run_zr2c_fd_cell(int N, FILE *out, int cool_ms, int rot)
{
    const size_t xs = (size_t)N + 2;
    const int half = N / 2;
    const char *W = k1z_bundle();
    double *x, *dst[2][2];
    const double *bsrc = NULL;
    double *crefbuf = NULL;
    int reps = reps_for((size_t)N);
    zr_arm_t A[2][2][3];
    int c, ipx, ri, k;

    if (!W) { printf("%-8d zr2c-fd SKIP (front-door bundle unavailable)\n", N); return; }

    x = alloc_d(xs);
    dst[0][0] = alloc_d(xs); dst[0][1] = alloc_d(xs);   /* r2c oop / r2c ip */
    dst[1][0] = alloc_d(xs); dst[1][1] = alloc_d(xs);   /* c2r oop / c2r ip */
    /* the SAME seed as the hand arm, so the two are comparable in one run */
    srand(31 + N);
    for (int i = 0; i < N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    x[N] = x[N + 1] = 0.0;
    memset(A, 0, sizeof A);

#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE hF = NULL;
        crefbuf = alloc_d(xs);
        if (DftiCreateDescriptor(&hF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N)
                == DFTI_NO_ERROR) {
            DftiSetValue(hF, DFTI_PLACEMENT, DFTI_INPLACE);
            DftiSetValue(hF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            if (DftiCommitDescriptor(hF) == DFTI_NO_ERROR) {
                memcpy(crefbuf, x, 8 * xs);
                DftiComputeForward(hF, crefbuf);
                bsrc = crefbuf;            /* the CCE reference AND c2r input */
            }
            DftiFreeDescriptor(&hF);
        }
    }
#endif

    for (k = 0; k < 12; k++) {
        int idx = (k + rot) % 12;
        c   = idx / 6;               /* 0 = r2c, 1 = c2r */
        ipx = (idx / 3) % 2;         /* 0 = OOP, 1 = in-place */
        ri  = idx % 3;               /* route */
        zr_fd_arm(&A[c][ipx][ri], N, W, c, ipx, ri, x, bsrc,
                  bsrc != NULL, dst[c][ipx], reps);
        cachebust();
        pace(cool_ms);
    }

    for (c = 0; c < 2; c++)
        for (ipx = 0; ipx < 2; ipx++) {
            printf("%-7d | %s %-3s |", N, c ? "c2r" : "r2c", ipx ? "IP" : "OOP");
            for (ri = 0; ri < 3; ri++) {
                zr_arm_t *a = &A[c][ipx][ri];
                if (!a->built)        printf(" %s CREATE-FAIL |", ZR_RT[ri]);
                else if (a->refused)  printf(" %s REFUSED     |", ZR_RT[ri]);
                else if (a->ns == 0)  printf(" %s GATE-FAIL   |", ZR_RT[ri]);
                else                  printf(" %s %8.0f ns |", ZR_RT[ri], a->ns);
            }
            printf(" gate %s\n",
                   bsrc ? "cross-engine" : "UNGATED (no MKL reference)");
        }

    if (out)
        for (c = 0; c < 2; c++)
            for (ipx = 0; ipx < 2; ipx++)
                for (ri = 0; ri < 3; ri++)
                    fprintf(out, "%d,1,%s,%s,%s,%.0f,%.3e,%d,%d,%d\n",
                            N, c ? "c2r" : "r2c", ipx ? "ip" : "oop",
                            ri == 0 ? "r0" : (ri == 1 ? "r1" : "W"),
                            A[c][ipx][ri].ns, A[c][ipx][ri].err,
                            A[c][ipx][ri].built, A[c][ipx][ri].refused, rot);

    /* alloc_d is posix_memalign-backed: plain free() corrupts the heap here. */
    free_d(x);
    free_d(dst[0][0]); free_d(dst[0][1]);
    free_d(dst[1][0]); free_d(dst[1][1]);
    if (crefbuf) free_d(crefbuf);
}

static void run_zr2c_cell(int N, FILE *out, int cool_ms, int flip)
{
    const int half = N / 2, top = N / 4;
    const char *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d zr2c   SKIP (front-door bundle unavailable)\n", N);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_PLACEMENT_OUT;
    cfg.rank = 1;
    cfg.n[0] = half;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom_dir = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d zr2c   vfft_create(c2c %d natural OOP) FAILED\n", N, half);
        return;
    }

    double *x  = alloc_d((size_t)N + 2);   /* real input; its z view IS x    */
    double *Zc = alloc_d((size_t)N);       /* c2c(N/2) plane (half complex)  */
    double *XC = alloc_d((size_t)N + 2);   /* our CCE output                 */
    double *y  = alloc_d((size_t)N);       /* our c2r output (N reals)       */
    double *aS = alloc_d((size_t)top + 1);
    double *aC = alloc_d((size_t)top + 1);
    double *bS = alloc_d((size_t)top + 1);   /* raw sin/cos for the backward */
    double *bC = alloc_d((size_t)top + 1);
    _zr2c_init_aff(N, aS, aC, bS, bC);
    srand(31 + N);
    for (int i = 0; i < N; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    x[N] = x[N + 1] = 0.0;

    double vfw = 0, vbw = 0, mfw = 0, mbw = 0;
    double xerr = -1, gours = -1, gmkl = -1;
    /* cascade arm (owner directive #2: halves >= 2048 belong to the cascade) */
    double cfw = 0, cbw = 0, cxerr = -1, cgours = -1;
    int conv = -1; /* 0 = DIF digit-reversal matched, 1 = DIT, -1 = n/a */
    int c4row = 0; /* a kind-4 row exists for the half */
    /* NATORDER cascade arm: placement=INPLACE + order=NATURAL reaches the
     * stfn natural cascade through existing routing (the k1nat machinery) —
     * no order-tape API needed, and the comparison turns properly symmetric:
     * ours IN-PLACE vs MKL IN-PLACE, the full law-(f) D2 shape. */
    double nfw = 0, nbw = 0, nxerr = -1, ngours = -1;
    int natarm = 0;
    int reps = reps_for((size_t)N);
    double tf[5], tb[5];
    const double *bsrc = XC;   /* c2r-arm input spectrum (MKL ref if present) */

#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE hF = 0, hB = 0;
    int okF = 0, okB = 0;
    double *mip  = alloc_d((size_t)N + 2);
    double *mip2 = alloc_d((size_t)N + 2);
    double *cref = alloc_d((size_t)N + 2);
    if (DftiCreateDescriptor(&hF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(hF, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(hF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        okF = (DftiCommitDescriptor(hF) == DFTI_NO_ERROR);
    }
    if (DftiCreateDescriptor(&hB, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(hB, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(hB, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        okB = (DftiCommitDescriptor(hB) == DFTI_NO_ERROR);
    }
    if (okF)
    {
        memcpy(cref, x, ((size_t)N + 2) * 8);
        DftiComputeForward(hF, cref);          /* the CCE reference spectrum */
        bsrc = cref;
    }
#endif

    /* ── gates, one shot per direction ── */
    vfft_execute(h, VFFT_FORWARD, x, Zc);
    _zr2c_fold_fwd(Zc, XC, aS, aC, N, 1, (size_t)N, (size_t)N + 2);
#ifdef VFFT_HAS_MKL
    if (okF)
    {
        double w = 0, xm = 0;
        for (int i = 0; i < 2 * (half + 1); i++)
        {
            double d = fabs(XC[i] - cref[i]);
            if (d > w) w = d;
            double a = fabs(cref[i]);
            if (a > xm) xm = a;
        }
        xerr = xm > 0 ? w / xm : w;
    }
#endif
    {   /* ours c2r fed the reference spectrum: must return N*x */
        _zr2c_fold_bwd(bsrc, Zc, bS, bC, N, 1, (size_t)N + 2, (size_t)N);
        vfft_execute(h, VFFT_BACKWARD, Zc, y);
        double gw = 0, gm = 0;
        for (int i = 0; i < N; i++)
        {
            double d = fabs(y[i] - (double)N * x[i]);
            if (d > gw) gw = d;
            double a = fabs(x[i]);
            if (a > gm) gm = a;
        }
        gours = gm > 0 ? gw / ((double)N * gm) : gw;
#ifdef VFFT_HAS_MKL
        if (okF && okB)
        {
            memcpy(mip2, cref, ((size_t)N + 2) * 8);
            DftiComputeBackward(hB, mip2);
            gw = 0;
            for (int i = 0; i < N; i++)
            {
                double d = fabs(mip2[i] - (double)N * x[i]);
                if (d > gw) gw = d;
            }
            gmkl = gm > 0 ? gw / ((double)N * gm) : gw;
        }
#endif
    }

    /* ── timing (flip-ordered; cachebust + pace between engines) ── */
#ifdef VFFT_HAS_MKL
    double tmf[5], tmb[5];
    if (flip && okF && okB)
    {
        memcpy(mip, x, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmf, DftiComputeForward(hF, mip));
        memcpy(mip2, cref, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmb, DftiComputeBackward(hB, mip2));
        mfw = _zr2c_med5(tmf); mbw = _zr2c_med5(tmb);
        cachebust();
        pace(cool_ms);
    }
#endif
    ZR2C_TIME(tf, {
        vfft_execute(h, VFFT_FORWARD, x, Zc);
        _zr2c_fold_fwd(Zc, XC, aS, aC, N, 1, (size_t)N, (size_t)N + 2);
    });
    ZR2C_TIME(tb, {
        _zr2c_fold_bwd(bsrc, Zc, bS, bC, N, 1, (size_t)N + 2, (size_t)N);
        vfft_execute(h, VFFT_BACKWARD, Zc, y);
    });
    vfw = _zr2c_med5(tf); vbw = _zr2c_med5(tb);
#ifdef VFFT_HAS_MKL
    if (!flip && okF && okB)
    {
        cachebust();
        pace(cool_ms);
        memcpy(mip, x, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmf, DftiComputeForward(hF, mip));
        memcpy(mip2, cref, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmb, DftiComputeBackward(hB, mip2));
        mfw = _zr2c_med5(tmf); mbw = _zr2c_med5(tmb);
    }
    /* ── NATORDER-CASCADE IN-PLACE ARM (halves >= 1024): the whole D2 route
     * in one padded plane. fwd: x -> plane, c2c(half) natural IN-PLACE, fold
     * IN-PLACE. bwd: CCE -> plane, fold_bwd IN-PLACE, c2c bwd IN-PLACE ->
     * N*x. Timed reps run on junk after rep 1, same convention as the MKL
     * in-place arms (dense transforms are data-oblivious; FTZ/DAZ).
     * half >= 1024 per the RE: MKL's regime-S real path runs its CASCADE at
     * EVERY half-length (CONCLUSIONS §3, region-C at every half) — our worst
     * cell (2048, half 1024) is where we deviate from that; let the race see
     * the cascade there too. */
    if (half >= 1024 && okF)
    {
        vfft_config_t ncf;
        memset(&ncf, 0, sizeof ncf);
        ncf.transform = VFFT_C2C;
        ncf.placement = VFFT_PLACEMENT_IN;
        ncf.rank = 1;
        ncf.n[0] = half;
        ncf.howmany = 1;
        ncf.order = VFFT_ORDER_NATURAL;
        ncf.layout = VFFT_LAYOUT_INTERLEAVED;
        ncf.nthreads = 1;
        ncf.wisdom_dir = W;
        vfft_plan hn = vfft_create(&ncf);
        if (hn)
        {
            natarm = 1;
            /* fwd gate */
            memcpy(XC, x, ((size_t)N + 2) * 8);
            vfft_execute(hn, VFFT_FORWARD, XC, XC);
            _zr2c_fold_fwd(XC, XC, aS, aC, N, 1, (size_t)N + 2, (size_t)N + 2);
            {
                double w2 = 0, xm2 = 0;
                for (int i = 0; i < 2 * (half + 1); i++)
                {
                    double d = fabs(XC[i] - cref[i]);
                    if (d > w2) w2 = d;
                    double a = fabs(cref[i]);
                    if (a > xm2) xm2 = a;
                }
                nxerr = xm2 > 0 ? w2 / xm2 : w2;
            }
            /* bwd gate */
            memcpy(XC, cref, ((size_t)N + 2) * 8);
            _zr2c_fold_bwd(XC, XC, bS, bC, N, 1, (size_t)N + 2, (size_t)N + 2);
            vfft_execute(hn, VFFT_BACKWARD, XC, XC);
            {
                double gw = 0, gm2 = 0;
                for (int i = 0; i < N; i++)
                {
                    double d = fabs(XC[i] - (double)N * x[i]);
                    if (d > gw) gw = d;
                    double a = fabs(x[i]);
                    if (a > gm2) gm2 = a;
                }
                ngours = gm2 > 0 ? gw / ((double)N * gm2) : gw;
            }
            if (nxerr >= 0 && nxerr < 1e-9 && ngours < 1e-9)
            {
                double tnf[5], tnb[5];
                memcpy(XC, x, ((size_t)N + 2) * 8);
                ZR2C_TIME(tnf, {
                    vfft_execute(hn, VFFT_FORWARD, XC, XC);
                    _zr2c_fold_fwd(XC, XC, aS, aC, N, 1, (size_t)N + 2, (size_t)N + 2);
                });
                memcpy(XC, cref, ((size_t)N + 2) * 8);
                ZR2C_TIME(tnb, {
                    _zr2c_fold_bwd(XC, XC, bS, bC, N, 1, (size_t)N + 2, (size_t)N + 2);
                    vfft_execute(hn, VFFT_BACKWARD, XC, XC);
                });
                nfw = _zr2c_med5(tnf);
                nbw = _zr2c_med5(tnb);
            }
            vfft_destroy(hn);
        }
    }

    if (hF) DftiFreeDescriptor(&hF);
    if (hB) DftiFreeDescriptor(&hB);
    free_d(mip); free_d(mip2); free_d(cref);
#endif

    double rf = (vfw > 0 && mfw > 0) ? mfw / vfw : 0;
    double rb = (vbw > 0 && mbw > 0) ? mbw / vbw : 0;
    printf("%-7d | r2c ours %9.0f  mkl-ip %9.0f  %5.2fx | c2r ours %9.0f  "
           "mkl-ip %9.0f  %5.2fx | xerr %.1e gours %.1e gmkl %.1e\n",
           N, vfw, mfw, rf, vbw, mbw, rb, xerr, gours, gmkl);
    if (natarm)
    {
        if (nfw > 0)
            printf("        | nat-ip casc r2c %8.0f       %5.2fx | c2r %13.0f"
                   "       %5.2fx | xerr %.1e gours %.1e\n",
                   nfw, (mfw > 0 && nfw > 0) ? mfw / nfw : 0,
                   nbw, (mbw > 0 && nbw > 0) ? mbw / nbw : 0, nxerr, ngours);
        else
            printf("        | nat-ip casc: GATE FAIL (xerr %.1e gours %.1e) — not timed\n",
                   nxerr, ngours);
    }
    if (c4row)
    {
        if (conv >= 0 && cfw > 0)
            printf("        | casc(%s) r2c %8.0f          %5.2fx | casc c2r %8.0f"
                   "          %5.2fx | xerr %.1e gours %.1e\n",
                   conv == 0 ? "DIF" : "DIT",
                   cfw, (mfw > 0 && cfw > 0) ? mfw / cfw : 0,
                   cbw, (mbw > 0 && cbw > 0) ? mbw / cbw : 0, cxerr, cgours);
        else
            printf("        | casc: ORDER CONVENTION UNKNOWN (neither DIF nor DIT "
                   "chain digit-reversal matched — needs an order-tape API)\n");
    }
    fflush(stdout);
    if (out)
        fprintf(out, "%d,1,%.0f,%.0f,%.3f,%.0f,%.0f,%.3f,%.1e,%.1e,%.1e,%.0f,%.0f,%d,%.0f,%.0f\n",
                N, vfw, mfw, rf, vbw, mbw, rb, xerr, gours, gmkl, cfw, cbw, conv, nfw, nbw);
    free_d(x); free_d(Zc); free_d(XC); free_d(y); free_d(aS); free_d(aC);
    vfft_destroy(h);
}

int main(int argc, char **argv)
{
    static const char usage[] =
        "usage: bench_1d_vs_mkl [mode flags] [wisdom] [csv] [pace_ms] [N] [K] [cool_ms] [flip] [core]\n"
        "  1D c2c    --k1nat / --k1noop / --k1zip / --k1dir; --mt threaded; --cmp mkl|fftw|kfr\n"
        "  1D real   --realfwd / --realbwd; --zr2c\n"
        "  2D / 3D   --2dilnat / --3dilnat; --2drealnat / --3dreal (--realip / --realown in place); --2dil / --3dil / --2dreal\n"
        "  batches   --lmc2c / --lmr2c / --lmc2r; --lm2d / --lm3d; --lm2dr2c / --lm2dc2r / --lm3dr2c / --lm3dc2r; --ilmt\n"
#ifdef VFFT_HAS_FLOAT
        "  FP32      --f32 with --k1noop [--mt] / --k1nat: the cell on float data (K > 1: a contiguous batch), MKL DFTI_SINGLE\n"
#endif
        "  wisdom = a file inside the wisdom directory (names the directory); the header of bench_1d_vs_mkl.c has the rest\n";
    /* --mt: the threaded cells (both engines at T), pinned core 0. Detect + strip
     * argv[1] so the positional args below keep their meaning. Thread count =
     * $VFFT_MT (default 8). */
    int mt = 0, il2d = 0, il3d = 0, real2d = 0;
    /* leading flags, any order; no mode flag at all prints the usage (exit 2). */
    while (argc >= 2 && argv[1][0] == '-' && argv[1][1] == '-')
    {
        if (strcmp(argv[1], "--mt") == 0)
        {
            mt = 1;
            const char *e = getenv("VFFT_MT");
            g_mt = (e && atoi(e) > 0) ? atoi(e) : 8;
        }
        else if (strcmp(argv[1], "--2dil") == 0)
        {
            il2d = 1; /* three-arm interleaved-2D scoping cell (M0a) */
        }
        else if (strcmp(argv[1], "--3dilnat") == 0)
        {
            g_k3nat = 1; /* the 3D interleaved GAUNTLET cell (2026-09-24): N1xN2xN3 in the N argument */
        }
        else if (strcmp(argv[1], "--2dilnat") == 0)
        {
            g_k2nat = 1; /* the 2D interleaved GAUNTLET cell (2026-09-23): N1 = the N
                          * argument, N2 = the K argument; one shape per process */
        }
        else if (strcmp(argv[1], "--3dil") == 0)
        {
            il3d = 1; /* the rank-3 INTERLEAVED c2c tier vs MKL DFTI 3D */
        }
        else if (strcmp(argv[1], "--2dreal") == 0)
        {
            real2d = 1; /* 2D real DOOR race: z door vs split door vs MKL CCE */
        }
        else if (strcmp(argv[1], "--k1dir") == 0)
        {   /* K=1 IL in-place, natural, BOTH directions (see g_k1dir). */
            g_k1zip = 1;
            g_k1nat = 1;
            g_k1dir = 1;
        }
        else if (strcmp(argv[1], "--k1zip") == 0)
        {
            /* K=1 kind-4 cells IN-PLACE on BOTH engines — the true
             * apples-to-apples in-place interleaved cell (Phase A4,
             * docs/roadmap/cascade_natural_inplace_plan.md). Distinct default
             * CSV below so a probe can never overwrite a banked baseline. */
            g_k1zip = 1;
        }
        else if (strcmp(argv[1], "--k1nat") == 0)
        {
            /* B6: natural-vs-natural — the number the natural-order campaign
             * exists to produce. --k1zip's in-place discipline with
             * order=NATURAL on our side (ZCASC route, stfn terminator; the
             * @nat verdict races/banks at create through the front door).
             * MKL's native order IS natural, so for the first time both
             * engines compute the SAME spectrum and the correctness column
             * is a cross-engine elementwise compare. */
            g_k1zip = 1;
            g_k1nat = 1;
        }
        else if (strcmp(argv[1], "--k1noop") == 0)
        {
            /* Phase D1 (il_coverage_plan.md): natural-vs-natural
             * OUT-OF-PLACE — the measurement that sizes Phase D. Our side:
             * order=NATURAL, placement=OUTOFPLACE (today's route = K=1 IL
             * engines to their reach, convert fallback above — NOT the
             * cascade; that routing is exactly what Phase D would add).
             * MKL side: DFTI NOT_INPLACE. Same cross-engine elementwise
             * correctness column as --k1nat. */
            g_k1nat = 1; /* g_k1zip stays 0 -> OOP on both engines */
        }
        else if (strcmp(argv[1], "--zr2c") == 0)
        {
            /* Phase 1 (DESIGN_interleaved_r2c.md): the first honest ours/MKL
             * table for K=1 INTERLEAVED real transforms. Ours = the D2 route
             * (front-door IL c2c(N/2) + zr2c.h folds); MKL = REAL CCE
             * IN-PLACE, its best arm (CONCLUSIONS V6), bwd-twin descriptor. */
            g_zr2c = 1;
        }
        else if (strcmp(argv[1], "--kzb") == 0 || strcmp(argv[1], "--lmc2c") == 0)
        {
            /* the lane-major c2c batch cell (since 2026-10-15 the batch door's; --kzb was
             * Phase C1's gap map over the convert bridge), see the run_kzb_cell block. */
            g_kzb = 1;
        }
        else if (strcmp(argv[1], "--lmr2c") == 0)
            g_kzbr = 1;   /* the lane-major r2c batch cell (2026-10-15) */
        else if (strcmp(argv[1], "--lmc2r") == 0)
            g_kzbr = 2;   /* ... its c2r twin */
        else if (strcmp(argv[1], "--lm2d") == 0)
            g_lm2d = 2;   /* the lane-major 2D c2c batch cell (2026-10-15), the shape N1xN2 in the N argument */
        else if (strcmp(argv[1], "--lm3d") == 0)
            g_lm2d = 3;   /* the lane-major 3D c2c batch cell, the shape N1xN2xN3 */
        else if (strcmp(argv[1], "--lm2dr2c") == 0)
            g_lm2dr = 1;  /* the lane-major 2D real batch cell, r2c */
        else if (strcmp(argv[1], "--lm2dc2r") == 0)
            g_lm2dr = 2;  /* ... c2r */
        else if (strcmp(argv[1], "--lm3dr2c") == 0)
            g_lm3dr = 1;  /* the lane-major 3D real batch cell, r2c (2026-10-19) */
        else if (strcmp(argv[1], "--lm3dc2r") == 0)
            g_lm3dr = 2;  /* ... c2r */
        else if (strcmp(argv[1], "--cmp") == 0 && argc >= 3)
        {   /* the cell's comparator. kfr (2026-09-25) exists only in a
             * --kfr build; fftw (2026-09-28) needs no build flag -- it binds
             * at runtime from $VFFT_FFTW_DLL and is refused loudly there if
             * the library is absent or is MKL's wrapper layer. */
            g_cmp_fftw = (strcmp(argv[2], "fftw") == 0);
#ifdef VFFT_HAS_KFR
            g_cmp_kfr = (strcmp(argv[2], "kfr") == 0);
#else
            if (strcmp(argv[2], "kfr") == 0)
            {
                fprintf(stderr, "--cmp kfr: this bench was built without the KFR arm (build.py --kfr)\n");
                return 2;
            }
#endif
            if (!g_cmp_fftw && strcmp(argv[2], "kfr") && strcmp(argv[2], "mkl"))
            {
                fprintf(stderr, "--cmp: unknown comparator '%s' (mkl | fftw | kfr)\n", argv[2]);
                return 2;
            }
            argv++;
            argc--;
        }
        else if (strcmp(argv[1], "--realfwd") == 0)
        {
            g_real = 1;   /* the 1D real gauntlet cell, r2c (2026-09-29) */
        }
        else if (strcmp(argv[1], "--realbwd") == 0)
        {
            g_real = 2;   /* ... c2r */
        }
        else if (strcmp(argv[1], "--2drealnat") == 0)
        {
            g_k2real = 1; /* the 2D real gauntlet cell: N1 in the N slot, N2 in the K slot */
        }
        else if (strcmp(argv[1], "--3dreal") == 0)
        {
            g_k3real = 1; /* the 3D real gauntlet cell (2026-10-07): the shape N1xN2xN3 in the N slot */
        }
        else if (strcmp(argv[1], "--realip") == 0)
        {
            g_real_ip = 1; /* the real cell in place (2026-10-08): the 2D r2c cell on one padded plane */
        }
        else if (strcmp(argv[1], "--realown") == 0)
        {
            g_real_ip = 1;
            g_real_own = 1; /* door 2: the plan's own plane at its pitch (2026-10-08) */
        }
        else if (strcmp(argv[1], "--ilmt") == 0)
        {
            /* TC-batch MT vs MKL batched MT (2026-08-06). Implies the MT
             * thread count; see the run_ilmt_cell block for the arm shape,
             * the P-core mask and the two thread-hygiene traps. */
            g_ilmt = 1;
            const char *e = getenv("VFFT_MT");
            g_mt = (e && atoi(e) > 0) ? atoi(e) : 8;
        }
#ifdef VFFT_HAS_FLOAT
        else if (strcmp(argv[1], "--f32") == 0)
            g_f32 = 1;   /* the FP32 arm of --k1noop / --k1nat (the run_k1z_cell_f32 block) */
#endif
        else
            break;
        argv++;
        argc--;
    }
    if (!(il2d || il3d || real2d || g_k3nat || g_k2nat || g_k1zip || g_k1nat || g_zr2c || g_kzb || g_kzbr ||
          g_lm2d || g_lm2dr || g_lm3dr || g_real || g_k2real || g_k3real || g_real_ip || g_ilmt))
    {
        fputs(usage, stderr);
        return 2;
    }
#ifdef VFFT_HAS_FLOAT
    /* --f32 serves the K=1 natural cell and its contiguous batch, MKL as the comparator: --k1noop
     * (with --mt the threaded cell) or --k1nat, nothing else */
    if (g_f32 && (!g_k1nat || g_k1dir || (mt && g_k1zip) || g_cmp_fftw || il2d || il3d || real2d || g_k3nat ||
                  g_k2nat || g_zr2c || g_kzb || g_kzbr || g_lm2d || g_lm2dr || g_lm3dr || g_real || g_k2real ||
                  g_k3real || g_real_ip || g_ilmt
#ifdef VFFT_HAS_KFR
                  || g_cmp_kfr
#endif
                  ))
    {
        fprintf(stderr, "--f32: the FP32 arm is the K=1 natural cell against MKL: --k1noop (out of place; --mt threaded) "
                        "or --k1nat (in place); K > 1 in the K slot is the contiguous batch\n");
        return 2;
    }
#endif
#ifdef VFFT_HAS_KFR
    if (g_cmp_kfr && (mt || il2d || il3d || real2d || g_k2real || g_k3real || g_real_ip))
    {   /* the 1D c2c cell and, since 2026-10-10, the 1D real cells (--realfwd / --realbwd; run_real_cell admits an even N, K = 1) and the 2D/3D c2c cells (--2dilnat / --3dilnat) */
        fprintf(stderr, "--cmp kfr: the KFR arm is the 1D c2c cell, the 2D/3D c2c cells (--2dilnat / --3dilnat) and the 1D real cells (--realfwd / --realbwd), one thread\n");
        return 2;
    }
#endif
    /* the FFTW arm serves the gauntlet's cells: the K=1 1D c2c cell, the 2D/3D
     * c2c cells (--2dilnat / --3dilnat, 2026-10-19) and the real cells. The
     * earlier forms have their own comparator shapes and are not wired to it,
     * so refuse rather than silently report a zero ratio. */
    if (g_cmp_fftw && (il2d || il3d || real2d))
    {
        fprintf(stderr, "--cmp fftw: the FFTW arm serves the gauntlet's cells (--k1noop / --k1nat, --2dilnat / --3dilnat, "
                        "--realfwd / --realbwd with --2drealnat / --3dreal), not the earlier 2D / real forms\n");
        return 2;
    }
    g_k1noop_mt = ((g_k1nat && !g_k1zip) || g_k2nat || g_k3nat || g_real || g_kzb || g_kzbr || g_lm2d || g_lm2dr || g_lm3dr) && mt;   /* the lane-major batch cells thread too (2026-10-15) */   /* the 3D real cell threads too (2026-10-07); the 2D real cell since 2026-10-08 */   /* the 2D/3D cells share the
                                                                          * threaded-cell discipline (2026-09-24) */
    if (g_k1noop_mt)
    {
        /* the two-team protocol's setup (3D_mt_il_strategy.md §5): the 8
         * P-cores for both engines, and MKL's OpenMP team created BEFORE
         * this thread is pinned to core 0 below (trap c: threads inherit
         * the creator's affinity). */
        ilmt_pin_pcores();
        if (g_cmp_fftw && !fftw_arm_threads(g_mt))
            return 2;   /* FFTW's team: the threads API bound and initialised here, under the
                         * P-core mask and before the caller's pin; its workers are spawned at
                         * the first threaded execute, inside each arm's untimed warm. A DLL
                         * without the API cannot run the threaded cell: refused, not timed serial */
#ifdef VFFT_HAS_MKL
        {
            DFTI_DESCRIPTOR_HANDLE d = NULL;
            double *a = alloc_d(2 * 4096), *b = alloc_d(2 * 4096);
            size_t i;
            for (i = 0; i < 2 * 4096; i++) a[i] = 1.0;
            mkl_set_num_threads(g_mt);
            if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)4096) == DFTI_NO_ERROR)
            {
                DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                if (DftiCommitDescriptor(d) == DFTI_NO_ERROR)
                    for (i = 0; i < 4; i++) DftiComputeForward(d, a, b);
                DftiFreeDescriptor(&d);
            }
            free_d(a); free_d(b);
        }
#endif
    }
    const char *wpath = (argc >= 2) ? argv[1]
                                    : "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    const char *csv = (argc >= 3)         ? argv[2]
                      : g_ilmt            ? "vfft_perf_tuned_1d_ilmt.csv"
                      : g_kzb             ? "vfft_perf_tuned_1d_kzb.csv"
                      : g_lm2d == 3       ? "vfft_perf_tuned_lm3d.csv"
                      : g_lm2d            ? "vfft_perf_tuned_lm2d.csv"
                      : g_lm2dr           ? "vfft_perf_tuned_lm2dr.csv"
                      : g_lm3dr           ? "vfft_perf_tuned_lm3dr.csv"
                      : g_kzbr            ? "vfft_perf_tuned_1d_lmreal.csv"
                      : g_zr2c            ? "vfft_perf_tuned_1d_zr2c.csv"
                      : (g_k1nat && !g_k1zip) ? "vfft_perf_tuned_1d_k1noop.csv"
                      : g_k1nat           ? "vfft_perf_tuned_1d_k1nat.csv"
                      : g_k1zip           ? "vfft_perf_tuned_1d_k1zip.csv"
                      : mt                ? "vfft_perf_tuned_1d_mt.csv"
                                          : "vfft_perf_tuned_1d.csv";
    int pace_ms = (argc >= 4) ? atoi(argv[3]) : 300;
    /* ISOLATED single-cell mode: target_N>0 benches ONLY cell (target_N,target_K)
     * in this (fresh) process — run_bench.py drives one cell per process, killing
     * cross-cell cache/thermal carryover. A prime target_N rides the override path.
     * target_N==0 keeps the legacy full in-process loop (quick-look). */
    int target_N = (argc >= 5) ? atoi(argv[4]) : 0;
    long target_K = (argc >= 6) ? atol(argv[5]) : BENCH_K;
    int cool_ms = (argc >= 7) ? atoi(argv[6]) : 0; /* inter-engine idle (order-bias fix) */
    if (g_k1noop_mt && cool_ms < 300)
        cool_ms = 300; /* trap (b): MKL's OpenMP spins KMP_BLOCKTIME (200 ms) before parking */
    int flip = (argc >= 8) ? atoi(argv[7]) : 0;    /* 1 = MKL first (alternate per cell) */
    /* 🔴 g_zr2c belongs in this disjunction: a bare --zr2c ran UNPINNED
     * while docs/performance/v1_0_results.md described its numbers as
     * "pinned core 2". An explicit core argument always wins; this only
     * fixes the default. */
    int core = (argc >= 9) ? atoi(argv[8]) : (mt ? 0 : (il2d || il3d || g_zr2c) ? 2
                                                                                                           : -1); /* MT->0, 2D/zr2c->P-core 2 */
    {
        const char *tp = getenv("VFFT_TRIAL_PACE_MS");
        g_trial_pace_ms = tp ? atoi(tp) : 0;
    }
    if (mt && !g_k1noop_mt)
        target_N = 0; /* --k1noop --mt keeps the one-process-per-cell form */

    vfft_env_init();
    /* --ilmt: confine the PROCESS to the 8 distinct P-cores before any MKL /
     * OpenMP initialization (Intel OpenMP reads the mask at init), then pin
     * the caller to logical 0 — the core threads.h reserves for it. */
    if (g_ilmt)
    {
        ilmt_pin_pcores();
        if (core < 0)
            core = 0;
    }
    if (core >= 0)   /* the library's measurement scope on that core; every single-thread mode (the
                      * 2D/R2C/zr2c ones included) holds its sibling (2026-09-22), a threaded one does not */
        bench_scope(core, 0, mt ? VFFT_MEASURE_GUARD_OFF : bench_guard_mode());

#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(mt ? g_mt : 1); /* --ilmt sets it per arm instead */
#endif
    /* --2dil: the three-arm interleaved-2D scoping cell (M0a), then done. */
    if (il2d)
    {
        const char *wd = getenv("VFFT_WISDOM_DIR");
        const char *W;
        int rounds;
        const char *re_ = getenv("VFFT_2DIL_ROUNDS");
        rounds = (re_ && atoi(re_) > 0) ? atoi(re_) : 9;
        if (!wd) wd = ".";
        setvbuf(stdout, NULL, _IONBF, 0); /* live lines even when redirected */
        W = (wd);
#ifdef VFFT_HAS_MKL
        mkl_set_num_threads(1);
#endif
        printf("=== 2DIL three-arm scoping cell (front door; wisdom=%s %s; "
               "rounds=%d, core%d) ===\n",
               wd, W ? "loaded" : "MISSING", rounds, core);
        printf("# arms: O-inter = vfft INPLACE interleaved; M-inter = "
               "DFTI CCE INPLACE (MKL best); M-split = banked REAL_REAL "
               "config; ctl = memcpy. '~' = delta below ctl spread (NOT A "
               "RESULT).\n");
        {
            int cells[][2] = { { 64, 64 },   { 128, 128 }, { 256, 256 },
                               { 512, 512 }, { 100, 100 }, { 1024, 1024 },
                               { 16, 4096 }, { 4096, 16 }, { 32, 1024 },
                               { 64, 256 },
                               /* the L2 band-threshold ladder (2026-08-25):
                                * band residency needs wl <= L2/(16*N1) —
                                * on 2 MB P-core L2 that pins wl at the
                                * kernel minimum (8) by N1=16384 and goes
                                * infeasible at 32768. Long columns, cheap
                                * rows, to isolate the column pass. */
                               { 4096, 64 }, { 8192, 64 }, { 16384, 64 },
                               { 32768, 64 } };
            int nc = (int)(sizeof cells / sizeof cells[0]), ci;
            const char *cf = getenv("VFFT_2DIL_CELLS"); /* "64x64,256x256" filter */
            /* VFFT_2DIL_SHAPES="46x64,47x64,138x128": ANY shapes, replacing the
             * list above (2026-09-22; the column pool's new radices have no
             * cell in it) */
            int xcells[32][2], nx = 0;
            const char *sh = getenv("VFFT_2DIL_SHAPES");
            if (sh) {
                const char *q = sh;
                while (*q && nx < 32) {
                    int a = 0, b = 0;
                    if (sscanf(q, "%dx%d", &a, &b) == 2 && a > 0 && b > 0) { xcells[nx][0] = a; xcells[nx][1] = b; nx++; }
                    while (*q && *q != ',') q++;
                    if (*q == ',') q++;
                }
            }
            for (ci = 0; ci < (sh ? nx : nc); ci++) {
                const int N1c = sh ? xcells[ci][0] : cells[ci][0], N2c = sh ? xcells[ci][1] : cells[ci][1];
                if (cf && !sh) {
                    char tag[32];
                    snprintf(tag, sizeof tag, "%dx%d", N1c, N2c);
                    if (!strstr(cf, tag)) continue;
                }
                run_2dil_cell(N1c, N2c, rounds, W);
                pace(pace_ms);
            }
        }
        return 0;
    }

    /* --3dil: the rank-3 INTERLEAVED c2c tier vs MKL DFTI 3D, then done. */
    if (il3d)
    {
        const char *wd = getenv("VFFT_WISDOM_DIR");
        const char *W;
        int rounds;
        const char *re_ = getenv("VFFT_3DIL_ROUNDS");
        rounds = (re_ && atoi(re_) > 0) ? atoi(re_) : 9;
        if (!wd) wd = ".";
        setvbuf(stdout, NULL, _IONBF, 0); /* live lines even when redirected */
        W = (wd);
        if (mt)
            ilmt_pin_pcores(); /* the 8 distinct P-cores for BOTH engines, before any OpenMP init */
        /* trap (d): the bench's startup pool is THIS TU's copy of the pool
         * state (threads.h is static inline) — 7 idle spinners pinned to
         * the very cores the library's workers use. Torn down here; the
         * library's pool is the plans' own (config.nthreads). */
        thread_pool_resize(1);
#ifdef VFFT_HAS_MKL
        mkl_set_num_threads(mt ? g_mt : 1); /* like-for-like: MKL threads 3D at the same T */
        if (mt)
            mkl3d_warm_team(g_mt); /* trap (c): MKL's team born before our pool pins the caller */
#endif
        printf("=== 3DIL: the rank-3 INTERLEAVED c2c tier vs MKL DFTI 3D (front door; "
               "wisdom=%s %s; rounds=%d, T=%d, core%d) ===\n",
               wd, W ? "loaded" : "MISSING", rounds, mt ? g_mt : 1, core);
        printf("# order: NATURAL (the like-for-like order: MKL's output is natural)\n");
        if (mt)
            printf("# threaded protocol (T=%d): two-team hygiene per sample (pool torn down before "
                   "MKL arms, MKL's team born before our pool pins), >= 5 ms untimed warm-up, "
                   "steady-state reps; per-sample pace %d ms (VFFT_TRIAL_PACE_MS) — a pace PARKS "
                   "the teams, the warm-up re-wakes them, spreads still widen: quote the unpaced run.\n",
                   g_mt, g_trial_pace_ms);
        else
            printf("# one-thread samples: per-sample pace %d ms (VFFT_TRIAL_PACE_MS) before the "
                   "cachebust + >= 5 ms warm-up; the pin line follows at the first cell.\n",
                   g_trial_pace_ms);
        printf("# arms, all OUT OF PLACE: O-NATIVE = vfft 3D INTERLEAVED (fftnd_il.h, "
               "structure from wisdom); M-inter = DFTI 3D CCE NOT_INPLACE (MKL best); "
               "M-split = DFTI REAL_REAL NOT_INPLACE; ctl = memcpy. '~' = delta "
               "below ctl spread (NOT A RESULT).\n");
        {
            int cells[][3] = { { 16, 16, 16 },   { 32, 32, 32 },   { 64, 64, 64 },
                               { 128, 128, 128 }, { 32, 16, 64 },  { 64, 128, 32 },
                               { 256, 64, 16 },   { 27, 9, 15 },   { 36, 20, 28 },
                               { 45, 45, 45 },    { 81, 27, 27 },
                               /* long-N3 cells: the axis-2 row plan is a K=1 cell
                                * in ZTURN-T's band (pow2 4096; 2^a*odd 12288) */
                               { 16, 16, 4096 },  { 8, 16, 12288 }, { 32, 32, 4096 } };
            int nc = (int)(sizeof cells / sizeof cells[0]), ci;
            const char *cf = getenv("VFFT_3DIL_CELLS"); /* "16x16x16,64x64x64": the cells to run,
                                                         * any shapes, in this order (2026-09-24:
                                                         * a list, no longer a filter of the
                                                         * built-in cells above, which run when
                                                         * the variable is unset) */
            if (cf) {
                const char *p = cf;
                while (*p) {
                    int a, b, c;
                    if (sscanf(p, "%dx%dx%d", &a, &b, &c) == 3 && a > 0 && b > 0 && c > 0) {
                        run_3dil_cell(a, b, c, rounds, W, mt ? g_mt : 1);
                        pace(pace_ms);
                    } else
                        fprintf(stderr, "[3dil] VFFT_3DIL_CELLS: cannot read a shape at '%.20s'\n", p);
                    while (*p && *p != ',') p++;
                    while (*p == ',' || *p == ' ') p++;
                }
            } else
                for (ci = 0; ci < nc; ci++) {
                    run_3dil_cell(cells[ci][0], cells[ci][1], cells[ci][2], rounds, W, mt ? g_mt : 1);
                    pace(pace_ms);
                }
        }
        return 0;
    }

    /* --2dreal: the 2D real DOOR race (M0), then done. */
    if (real2d)
    {
        const char *wd = getenv("VFFT_WISDOM_DIR");
        const char *W;
        int rounds;
        const char *re_ = getenv("VFFT_2DIL_ROUNDS");
        rounds = (re_ && atoi(re_) > 0) ? atoi(re_) : 9;
        if (!wd) wd = ".";
        setvbuf(stdout, NULL, _IONBF, 0);
        W = (wd);
#ifdef VFFT_HAS_MKL
        mkl_set_num_threads(1);
#endif
        printf("=== 2DREAL door race (r2c fwd + c2r bwd; wisdom=%s %s; "
               "rounds=%d) ===\n",
               wd, W ? "loaded" : "MISSING", rounds);
        printf("# O-nat = the native IL tier "
               "(THE interleaved serving) | M-cce = MKL DFTI REAL 2D CCE | "
               "MEASURE-rigor creates\n");
        {
            /* squares AND the aspect set — the regimes live at the aspects
             * (owner 2026-08-25: never judge a door on squares alone;
             * long columns stress the un-banded column pass, long rows
             * stress the row pack). N2 must be even (r2c contract). */
            int cells[][2] = { { 64, 64 },   { 256, 256 }, { 512, 512 },
                               { 1024, 1024 },
                               { 16, 4096 }, { 4096, 16 }, { 32, 1024 },
                               { 64, 256 },  { 4096, 64 }, { 8192, 64 } };
            int nc2 = (int)(sizeof cells / sizeof cells[0]), ci;
            for (ci = 0; ci < nc2; ci++)
                run_2dreal_cell(cells[ci][0], cells[ci][1], rounds, W);
        }
        return 0;
    }

    /* Shipped-reader view of the SAME positional wisdom file (oop_wisdom.h is
     * the single source of format truth for OOP kind lines): serves the K=1
     * kind-4 SCRAMBLED-cascade cells, which route through the vfft front door
     * instead of the local c2c factor walk (see run_k1z_cell). */
    g_k1z_wpath = wpath;
    /* writable ONLY under an explicit VFFT_WISDOM_DIR: a bench never banks
     * by accident */
    vw2_open(&g_k1z_store, k1z_dir(), getenv("VFFT_WISDOM_DIR") != NULL);
    g_k1z_oopw_loaded = (g_k1z_store.nrec > 0);

    FILE *f = fopen(wpath, "r");
    if (!f)
    {
        fprintf(stderr, "cannot open wisdom %s\n", wpath);
        return 1;
    }
    FILE *out = fopen(csv, target_N ? "a" : "w");
    g_csv_path = csv;   /* the K=1 cell path replaces its own earlier row (2026-09-21) */
    /* THE HEADER GOES ON THE FILE, NOT ON THE MODE (2026-09-19): a gauntlet
     * runs one process per cell -- the isolation the protocol requires -- and
     * every one of them appends to the same CSV, so the header must be
     * written by whichever process CREATES the file. Keyed on the file being
     * empty, which is also true of the "w" sweep, so that path is unchanged. */
    long csv_pos = 0;
    if (out)
    {
        fseek(out, 0, SEEK_END);
        csv_pos = ftell(out);
    }
    if (out && csv_pos == 0)
    {
        if (g_k3nat)
            fprintf(out, "N1,N2,N3,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,%sflip\n", g_k1noop_mt ? "engaged," : "");
        else if (g_k3real)
            fprintf(out, "N1,N2,N3,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,%sflip\n", g_k1noop_mt ? "engaged," : "");
        else if (g_k2real)
            fprintf(out, "N1,N2,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,flip\n");
        else if (g_k2nat)
            fprintf(out, "N1,N2,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,%sflip\n", g_k1noop_mt ? "engaged," : "");
        else if (g_ilmt)
            fprintf(out, "N,K,path,threads,ours_mt_ns,ours_st_ns,mkl_mt_ns,mkl_st_ns,"
                         "ratio_vs_mkl_best,ratio_vs_mkl_mt,scale_ours,scale_mkl,"
                         "ctl_spread,xerr,mt_ne_st\n");
        else if (g_lm3dr)   /* the --lm3d columns with the loop replaced by the volume queue (ours contiguous); the
                             * second verdict ratio_contig = mkl_home / contig; threaded: both doors' engagement */
            fprintf(out, "N,K,plan,path,vfft_ns,contig_ns,mkl_mirror_ns,mkl_home_ns,"
                         "ratio_mirror,ratio_home,ratio_contig,xerr,contig_xerr,route%s\n",
                    g_k1noop_mt ? ",engaged,engaged_pq,engaged_plane,engaged_grp" : "");
        else if (g_kzb || g_kzbr || g_lm2d || g_lm2dr)
            fprintf(out, "N,K,plan,path,vfft_ns,loop_ns,mkl_mirror_ns,mkl_home_ns,"
                         "ratio_mirror,ratio_home,ratio_loop_vs_home,xerr,loop_xerr,route%s\n", g_k1noop_mt ? ",engaged" : "");
        else if (g_zr2c)
            fprintf(out, "N,K,ours_r2c_ns,mkl_r2c_ip_ns,r2c_ratio,ours_c2r_ns,"
                         "mkl_c2r_ip_ns,c2r_ratio,xerr_fwd,gate_ours_c2r,gate_mkl_c2r,"
                         "casc_r2c_ns,casc_c2r_ns,casc_conv,natip_r2c_ns,natip_c2r_ns\n");
        else
            /* route= the ENGINE the front door committed (vfft_plan_route),
             * flip= which engine ran first in the pair -- the two columns a
             * band-map gauntlet reads. APPENDED, so every column a consumer
             * already parses by position keeps its index. */
            fprintf(out, "N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,flip%s\n",
                    g_k1noop_mt ? ",engaged" : "");
    }
#ifdef VFFT_HAS_FLOAT
    if (g_f32)
    {   /* --f32: one isolated FP32 cell per process (the N slot; K > 1 in the K slot: the contiguous
         * batch); nothing else runs in this process */
        if (!target_N)
        {
            printf("--f32 needs the cell's N\n");
            if (out)
                fclose(out);
            fclose(f);
            return 2;
        }
        run_k1z_cell_f32(target_N, (argc >= 6 && target_K > 1) ? (size_t)target_K : 1u, out, cool_ms, flip);
        if (out)
            fclose(out);
        fclose(f);
        printf("\nbenched 1 cell.  CSV -> %s\n", csv);
        return 0;
    }
#endif
    if (g_k3nat)
    {   /* --3dilnat: the 3D interleaved gauntlet cell, the shape N1xN2xN3 in
         * the N argument; nothing else runs in this process (2026-09-24) */
        int n1 = 0, n2 = 0, n3 = 0;
        if (argc < 5 || sscanf(argv[4], "%dx%dx%d", &n1, &n2, &n3) != 3)
        {
            printf("--3dilnat needs the shape N1xN2xN3 in the N argument\n");
            return 2;
        }
        run_k2z_cell(n1, n2, n3, out, cool_ms, flip);
        if (out)
            fclose(out);
        printf("\nbenched 1 cell.  CSV -> %s\n", csv);
        return 0;
    }
    if (g_real || g_k2real || g_k3real)
    {   /* --realfwd / --realbwd / --2drealnat / --3dreal: one isolated real cell per
         * process (the N and K slots); nothing else runs in this process */
        if (!target_N)
        {
            printf("--realfwd / --realbwd need the cell's N\n");
            return 2;
        }
        {
            real_geo_t g;
            memset(&g, 0, sizeof g);
            if (g_k3real)
            {   /* the 3D real cell (2026-10-07): the shape N1xN2xN3 in the N slot, any N3 parity;
                 * the CCE volume N1 x N2 x (N3/2+1); the library's rank-3 real tier, FFTW and MKL plan it */
                int n1 = 0, n2 = 0, n3 = 0;
                if (argc < 5 || sscanf(argv[4], "%dx%dx%d", &n1, &n2, &n3) != 3 || n1 < 2 || n2 < 2 || n3 < 2)
                {
                    printf("--3dreal needs the shape N1xN2xN3 in the N argument\n");
                    return 2;
                }
                g.nd = 3; g.N1 = n1; g.N2 = n2; g.N3 = n3; g.K = 1;
                g.rp = (size_t)n1 * (size_t)n2 * (size_t)n3;
                g.cp = 2u * (size_t)n1 * (size_t)n2 * ((size_t)n3 / 2u + 1u);
                g.pts = g.rp;
            }
            else if (g_k2real)
            {   /* N1 in the N slot, N2 in the K slot, either parity (odd N2 since
                 * 2026-10-07: the CCE plane is N1 x (N2/2+1) pairs for both; the
                 * library serves it, FFTW and MKL plan it) */
                if (target_K < 2)
                {
                    printf("--2drealnat needs N1 in the N slot and N2 >= 2 in the K slot\n");
                    return 2;
                }
                g.nd = 2; g.N1 = target_N; g.N2 = (int)target_K; g.K = 1;
                g.rp = (size_t)g.N1 * (size_t)g.N2;
                g.cp = 2u * (size_t)g.N1 * ((size_t)g.N2 / 2u + 1u);
                g.pts = g.rp;
            }
            else
            {
                g.nd = 1; g.N1 = target_N; g.N2 = 0; g.K = target_K > 1 ? (size_t)target_K : 1u;
                g.rp = (size_t)g.N1;
                g.cp = 2u * ((size_t)g.N1 / 2u + 1u);
                g.pts = g.K * g.rp;
            }
            if (!g_real)
                g_real = 1;   /* --2drealnat alone: forward */
            run_real_cell(&g, out, cool_ms, flip);
        }
        if (out)
            fclose(out);
        printf("\nbenched 1 cell.  CSV -> %s\n", csv);
        return 0;
    }
    if (g_k2nat)
    {   /* --2dilnat: the 2D interleaved gauntlet cell (N1 = the N argument,
         * N2 = the K argument); nothing else runs in this process */
        run_k2z_cell(target_N, (int)target_K, 0, out, cool_ms, flip);
        if (out)
            fclose(out);
        printf("\nbenched 1 cell.  CSV -> %s\n", csv);
        return 0;
    }
    if (g_ilmt)
    {
        if (cool_ms < 300)
            cool_ms = 300; /* trap (b): MKL's OpenMP spins KMP_BLOCKTIME
                            * (default 200 ms) before parking */
#ifdef VFFT_HAS_MKL
        { /* does MKL actually ACCEPT the thread count under this affinity
           * mask? A sparse mask can defeat OpenMP topology detection, and a
           * silently-1-thread MKL would read as a huge fake win for us. */
            mkl_set_num_threads(g_mt);
            int got = mkl_get_max_threads();
            printf("# MKL: requested %d threads, mkl_get_max_threads()=%d%s\n",
                   g_mt, got,
                   got == g_mt ? "" : "   <-- MISMATCH, MKL arm is HANDICAPPED");
        }
#endif
        if (!target_N)
            printf("=== dag vs MKL — 1D C2C fwd, TRANSFORM-CONTIGUOUS BATCH, "
                   "MULTITHREADED (%d threads both engines; pace=%dms "
                   "cool=%dms) ===\n"
                   "# identical memory on both sides: DFTI howmany=K "
                   "distance=N IS our transform-contiguous geometry\n"
                   "# r_best = ours_mt vs MKL's FASTER config (its MT is a net "
                   "loss at most cells, so that is usually its ST) — the "
                   "honest headline\n"
                   "# scale = own ST/MT (8.00 = perfect). ctl = repeat-arm "
                   "spread; a delta under it is NOT a result\n"
                   "%-7s %-4s %10s %10s %10s %10s | %8s %7s %6s %6s %6s %9s\n",
                   g_mt, pace_ms, cool_ms, "N", "K", "ours_mt", "ours_st",
                   "mkl_mt", "mkl_st", "r_best", "r_mklmt", "sc_us", "sc_mkl",
                   "ctl", "xerr");
        if (target_N)
            run_ilmt_cell(target_N, (int)target_K, out, cool_ms, flip);
        else
        {
            static const int ILMT_N[] = { 256, 512, 1024, 4096, 16384, 65536 };
            static const int ILMT_K[] = { 4, 8, 32 };
            int cells = 0;
            for (size_t ki = 0; ki < sizeof ILMT_K / sizeof ILMT_K[0]; ki++)
                for (size_t ni = 0; ni < sizeof ILMT_N / sizeof ILMT_N[0]; ni++)
                {
                    run_ilmt_cell(ILMT_N[ni], ILMT_K[ki], out, cool_ms,
                                  flip ^ (cells & 1));
                    cells++;
                    pace(pace_ms);
                }
        }
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_zr2c)
    {
        /* Phase 1 dispatch (DESIGN_interleaved_r2c.md §6): fixed cell list —
         * no wisdom-walk (the cells are the spec's, not the banked rows).
         * target_N>0 = one isolated cell per process (the trusted mode). */
        if (!target_N)
            printf("=== dag(D2 zr2c) vs MKL — 1D REAL r2c/c2r, K=1, INTERLEAVED CCE (pace=%dms cool=%dms) ===\n"
                   "# ours = reinterpret + front-door IL c2c(N/2) NATURAL OOP + zr2c fold (OOP arm)\n"
                   "# MKL  = DFTI_REAL CCE DFTI_INPLACE — its best arm (V6); backward on its own twin descriptor\n"
                   "# gates: fwd = cross-engine elementwise | c2r = each engine's backward vs N*x. MEDIANS of 5.\n",
                   pace_ms, cool_ms);
        /* FRONT-DOOR arms get their OWN csv: the column set differs from the
         * 16-column zr2c schema above, and appending a different width into
         * the banked file would corrupt it. */
        FILE *fdout = fopen("vfft_perf_tuned_1d_zr2c_fd.csv", "w");
        if (fdout)
            fprintf(fdout, "N,K,transform,placement,route,ns,relerr,built,refused,rot\n");
        printf("\n# FRONT-DOOR arms: vfft_create(R2C/C2R) x {OOP,IP} x {route0,route1,wisdom}\n"
               "# these are the honest composite numbers -- the hand-built arms above\n"
               "# never timed the route-1 memcpy, the route-0 scratch hop, or ANY in-place shape.\n"
               "# both run in one process on purpose: route-0 OOP c2r here should land near\n"
               "# the hand arm's c2r, since that hand shape already equals what the executor does.\n");
        if (target_N)
        {
            run_zr2c_cell(target_N, out, cool_ms, flip);
            run_zr2c_fd_cell(target_N, fdout, cool_ms, 0);
        }
        else
        {
            static const int ZRN[] = { 512, 2048, 8192, 16384, 65536 };
            for (size_t ni = 0; ni < sizeof ZRN / sizeof ZRN[0]; ni++)
            {
                run_zr2c_cell(ZRN[ni], out, cool_ms, flip ^ ((int)ni & 1));
                pace(pace_ms);
                run_zr2c_fd_cell(ZRN[ni], fdout, cool_ms, (int)ni);
                pace(pace_ms);
            }
        }
        if (fdout)
            fclose(fdout);
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_kzbr)
    {   /* the lane-major real batch cell: one isolated cell per process (the N and K arguments) */
        if (!target_N || target_K < 2)
        {
            fprintf(stderr, "--lmr2c / --lmc2r need the cell's N and K (K >= 2)\n");
            if (out) fclose(out);
            fclose(f);
            return 2;
        }
        printf("=== dag vs MKL — 1D %s LANE-MAJOR batch, OOP (%s; pace=%dms) ===\n"
               "# vfft = the lane-major batch door | loop = K sequential K=1 real plans on contiguous data\n"
               "# mkl_mir = MKL REAL CCE on OUR lane-major | mkl_home = MKL transform-contiguous | r_loop = loop vs mkl_home\n"
               "%-8s %-4s %-8s %11s %11s %11s %11s %7s %7s %8s %9s %9s\n",
               g_kzbr == 2 ? "c2r" : "r2c", g_k1noop_mt ? "both sides threaded" : "one thread", pace_ms,
               "N", "K", "path", "vfft_ns", "loop_ns", "mkl_mir_ns", "mkl_home_ns", "r_mir", "r_home", "r_loop", "xerr", "loop_err");
        run_kzbr_cell(target_N, (int)target_K, g_kzbr == 2, out, cool_ms, flip);
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_lm2dr)
    {   /* the lane-major 2D real batch cell: the shape N1xN2 in the N argument, K in the K argument */
        int n1 = 0, n2 = 0;
        if (argc < 5 || sscanf(argv[4], "%dx%d", &n1, &n2) != 2 || target_K < 2)
        {
            printf("--lm2dr2c / --lm2dc2r need the shape N1xN2 in the N argument and K >= 2\n");
            return 2;
        }
        printf("=== dag vs MKL — 2D %s LANE-MAJOR batch, OOP (%s; pace=%dms) ===\n"
               "# vfft = the 2D real batch door | loop = K sequential one-plane 2D real plans on contiguous planes\n"
               "# mkl_mir = MKL REAL CCE 2D on OUR lane-major | mkl_home = MKL contiguous planes | r_loop = loop vs mkl_home\n"
               "%-9s %-4s %-8s %11s %11s %11s %11s %7s %7s %8s %9s %9s\n",
               g_lm2dr == 2 ? "c2r" : "r2c", g_k1noop_mt ? "both sides threaded" : "one thread", pace_ms,
               "N1xN2", "K", "path", "vfft_ns", "loop_ns", "mkl_mir_ns", "mkl_home_ns", "r_mir", "r_home", "r_loop", "xerr", "loop_err");
        run_lm2dr_cell(n1, n2, (int)target_K, g_lm2dr == 2, out, cool_ms, flip);
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_lm3dr)
    {   /* the lane-major 3D real batch cell: the shape N1xN2xN3 in the N argument, K in the K argument */
        int n[3] = { 0, 0, 0 };
        if (argc < 5 || sscanf(argv[4], "%dx%dx%d", &n[0], &n[1], &n[2]) != 3 || target_K < 2)
        {
            printf("--lm3dr2c / --lm3dc2r need the shape N1xN2xN3 in the N argument and K >= 2\n");
            return 2;
        }
        printf("=== dag vs MKL — 3D %s LANE-MAJOR batch, OOP (%s; pace=%dms) ===\n"
               "# vfft = the 3D real batch door | contig = the volume queue (one plan, K contiguous volumes)\n"
               "# mkl_mir = MKL REAL CCE 3D on OUR lane-major | mkl_home = MKL contiguous volumes | r_contig = mkl_home vs contig\n"
               "# xerr = vfft vs mkl_mir | contig_err = contig vs mkl_home | mkl_self = mkl_mir vs mkl_home | x_home = vfft vs mkl_home\n"
               "%-11s %-4s %-8s %11s %11s %11s %11s %7s %7s %9s %9s %9s %9s %9s\n",
               g_lm3dr == 2 ? "c2r" : "r2c", g_k1noop_mt ? "both sides threaded" : "one thread", pace_ms,
               "N1xN2xN3", "K", "path", "vfft_ns", "contig_ns", "mkl_mir_ns", "mkl_home_ns", "r_mir", "r_home", "r_contig",
               "xerr", "contig_err", "mkl_self", "x_home");
        run_lm3dr_cell(n, (int)target_K, g_lm3dr == 2, out, cool_ms, flip);
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_lm2d)
    {   /* the lane-major 2D / 3D c2c batch cell: the shape N1xN2 (N1xN2xN3) in the N argument, K in the K argument */
        const int nd = g_lm2d;
        int n[3] = { 0, 0, 0 };
        if (argc < 5 || sscanf(argv[4], "%dx%dx%d", &n[0], &n[1], &n[2]) != nd || target_K < 2)
        {
            printf("--lm%dd needs the shape %s in the N argument and K >= 2\n", nd, nd == 3 ? "N1xN2xN3" : "N1xN2");
            return 2;
        }
        printf("=== dag vs MKL — %dD C2C LANE-MAJOR batch, OOP natural (%s; pace=%dms) ===\n"
               "# vfft = the %dD batch door (every axis a column pass) | loop = K sequential one-plane %dD plans on contiguous planes\n"
               "# mkl_mir = MKL %dD on OUR lane-major | mkl_home = MKL contiguous planes | r_loop = loop vs mkl_home\n"
               "%-11s %-4s %-8s %11s %11s %11s %11s %7s %7s %8s %9s %9s\n",
               nd, g_k1noop_mt ? "both sides threaded" : "one thread", pace_ms, nd, nd, nd,
               "shape", "K", "path", "vfft_ns", "loop_ns", "mkl_mir_ns", "mkl_home_ns", "r_mir", "r_home", "r_loop", "xerr", "loop_err");
        run_lm2d_cell(nd, n, (int)target_K, out, cool_ms, flip);
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_kzb)
    {
        /* Phase C1 dispatch: direct-cell only — K∈{2,3} have NO wisdom
         * lines in either file, so the wisdom walk below can never reach
         * those cells (C0 finding). target_N>0 = one isolated cell per
         * process (the map protocol); no target = in-process quick-look
         * over the full K∈{2,3,4} × N grid. */
        if (!target_N)
            printf("=== dag vs MKL — 1D C2C fwd batched, OOP natural "
                   "(pace=%dms) ===\n"
                   "# vfft = the lane-major batch door (the column pass) | loop = K sequential "
                   "K=1 on transform-contiguous\n"
                   "# mkl_mir = MKL on OUR lane-major | mkl_home = MKL "
                   "transform-contiguous | r_loop = loop vs mkl_home\n"
                   "%-8s %-4s %-8s %11s %11s %11s %11s %7s %7s %8s %9s %9s\n",
                   pace_ms, "N", "K", "path", "vfft_ns", "loop_ns",
                   "mkl_mir_ns", "mkl_home_ns", "r_mir", "r_home", "r_loop",
                   "xerr", "loop_err");
        if (target_N)
            run_kzb_cell(target_N, (int)target_K, out, cool_ms, flip);
        else
        {
            static const int KZB_K[] = { 2, 3, 4 };
            static const int KZB_N[] = { 256, 512, 1024, 2048,
                                         4096, 8192, 16384, 32768 };
            int cells = 0;
            for (size_t ki = 0; ki < sizeof KZB_K / sizeof KZB_K[0]; ki++)
                for (size_t ni = 0; ni < sizeof KZB_N / sizeof KZB_N[0]; ni++)
                {
                    run_kzb_cell(KZB_N[ni], KZB_K[ki], out, cool_ms,
                                 flip ^ (cells & 1));
                    cells++;
                    pace(pace_ms);
                }
        }
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }

    if (!target_N)
    {
        printf("=== dag JIT vs MKL — 1D C2C fwd, K=%d (calibrated cells; pace=%dms) ===\n", BENCH_K, pace_ms);
        printf("%-8s %-16s %-7s %12s %12s %8s %7s %10s\n",
               "N", "plan", "path", "vfft_ns", "mkl_ns", "vGFLOP", "ratio", "rt_err");
        printf("---------+----------------+-------+------------+------------+--------+-------+----------\n");
    }

    int benched = 0, skipped = 0;

    /* ── K=1 kind-4 pass, enumerated FROM THE STORE ──────────────────────
     * The cells to visit come from the live store, not from a wisdom file:
     * the legacy enumeration source (oop_wisdom.txt) is deleted, and a
     * re-raced or newly-added cell has to be visible to the bench without
     * a file to re-parse. Each record is resolved through the PRODUCTION
     * twin, so what is measured and labelled is what the front door
     * serves. */
    if (g_k1z_oopw_loaded)
    {
        int cursor = 0, done[128], ndone = 0;
        const vw2_rec_t *r;
        while ((r = vw2_scan(&g_k1z_store, &cursor)) != NULL)
        {
            const char *eng = vw2_rec_get(r, "eng");
            vfft_oop_wisdom_entry_t ze;
            int N4 = r->key.n[0], d, dup = 0;
            if (r->key.t != VW2_T_C2C || r->key.rank != 1 || r->key.q != 1)
                continue;
            if (!eng || (strcmp(eng, "zturn") && strcmp(eng, "zsplit")))
                continue;                       /* kind-4 = the cascade */
            if (target_N && N4 != target_N)
                continue;
            for (d = 0; d < ndone; d++)
                if (done[d] == N4)
                    dup = 1;
            if (dup || ndone >= 128)
                continue;
            done[ndone++] = N4;
            if (!(0 /* no kind-4 cascade rows since 2026-09-15 */))
            {
                skipped++;
                continue;
            }
            run_k1z_cell(N4, &ze, out, cool_ms, flip ^ (benched & 1));
            benched++;
            pace(pace_ms);
        }
    }
    if (f)
        fclose(f);

    /* --k1nat sub-2048 direct cell (il_coverage_plan.md Phase B5): these
     * cells have NO kind-4 line (they are the K=1 IL-tier band, and no
     * kind-3 K=1 lines ship either), so file-driven enumeration can never
     * reach them. With an explicit [N] below 2048 the cell runs directly:
     * the front-door NATURAL in-place create serves the @nat ILP verdict
     * (racing + banking it on first touch, replaying after). */
    /* ... and (2026-09-05) any N above 2048 WITHOUT a factor of 4: the flat
     * mixed-radix DIT's cells, served natively by the same K=1 IL tier
     * (route=flat) — no kind-4 line exists there either. */
    /* ... and (2026-09-15) the 2^a * odd cells of ZTURN-T's odd band
     * (docs/design/ztt_odd_design.md): the K=1 IL tier serves them through
     * the same front door; their kind-4 cascade lines are purged. */
    {   /* the K=1 IL direct cell: asked, not re-derived (planning/policy.h).
         * NOT the same set as "the planner races here" — above the race
         * ceiling an odd N is still served by the prime engine. */
        vfft_cell_t bc;
        memset(&bc, 0, sizeof bc);
        bc.N = target_N; bc.K = 1; bc.rank = 1; bc.T = 1;
        bc.layout = VW2_LAY_IL; bc.ord = VW2_ORD_NAT;
        g_k1_direct_cell = vfft_policy_k1_direct_cell(&bc);
    }
    if (g_k1nat && target_N && g_k1_direct_cell && benched == 0)   /* the pow2 band too since the kind-4 lines left (2026-09-15) */
    {
        run_k1z_cell(target_N, NULL, out, cool_ms, flip);
        benched++;
        skipped = 0;
    }
    if (out)
        fclose(out);
    printf("\nbenched %d cells, skipped %d.  CSV -> %s\n", benched, skipped, csv);
    return 0;
}
