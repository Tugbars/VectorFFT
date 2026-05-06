/* bench_vtune.c — VTune-instrumented 1D C2C bench across selected cells.
 *
 * Goal: profile WHY each cell behaves the way it does. Cells are picked
 * from build_tuned/results/vfft_perf_tuned_1d.txt to span:
 *   CLOSE     — MKL came within 1.30× (we want to find ILP / store-bound
 *               bottlenecks that justify v1.1 codelet reschedules)
 *   MID       — typical wins (1.8–2.5×, validates codelets behave as
 *               cost model predicts)
 *   DECISIVE  — MKL <0.25× our throughput (validates per-radix codegen
 *               at the small-N batch regime where we dominate)
 *
 * Each cell runs ~2 seconds of FFT work bracketed by ITT API tasks so
 * VTune's sampling profile attributes time, retiring %, port pressure,
 * etc. to a named region.
 *
 * Usage:
 *   bench_vtune.exe              # VFFT only
 *   bench_vtune.exe --mkl        # VFFT + MKL on same cells, side-by-side
 *
 * Recommended VTune command (from build_tuned/dev/bench_vtune/):
 *   vtune -collect uarch-exploration -result-dir vt_uarch -- ^
 *         bench_vtune.exe --mkl
 *   vtune -report hotspots -result-dir vt_uarch
 *   vtune-gui vt_uarch    # for the full UI
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "vfft.h"
#include "ittnotify.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

/* ── ITT domain + per-category task handles ───────────────────── */
static __itt_domain *g_dom = NULL;

/* Cell descriptor */
typedef struct {
    int          N;
    size_t       K;
    const char  *category;   /* "CLOSE" / "MID" / "DECISIVE" */
    const char  *note;
    double       ratio_vs_mkl;
} cell_t;

/* Bench grid: hand-picked from vfft_perf_tuned_1d.txt.
 * Selection rationale per category in the comment above each row. */
static const cell_t CELLS[] = {
    /* CLOSE — large pow2 K=4 is ILP-bound; codelet stage-chain has
     * dependency latency that nvcc/icx can't hide at low batch */
    {131072, 4,  "CLOSE",    "1.17x — pow2 K=4 ILP weakness",       1.17},
    { 32768, 4,  "CLOSE",    "1.20x — pow2 K=4",                    1.20},
    {  8192, 4,  "CLOSE",    "1.32x — pow2 K=4 borderline",         1.32},
    {   243, 4,  "CLOSE",    "1.26x — 3^5 prime power, K=4",        1.26},

    /* MID — typical pow2 wins */
    {  1024, 256, "MID",     "1.94x — pow2 well-tuned baseline",    1.94},
    {  2048, 256, "MID",     "2.08x — pow2 mid",                    2.08},
    {  4096, 256, "MID",     "1.80x — pow2 large",                  1.80},

    /* DECISIVE — small N where every per-radix codegen detail counts */
    {     8, 256, "DECISIVE", "7.78x — small-N batch, MKL drowns",  7.78},
    {    16, 256, "DECISIVE", "4.80x — small-N batch",              4.80},
    {    60,  32, "DECISIVE", "5.15x — composite 12x5",             5.15},
    {   128, 256, "DECISIVE", "2.93x — radix-4x4x8",                2.93},
    {   243, 256, "DECISIVE", "2.69x — same N=243 as CLOSE, K=256", 2.69},

    /* BLUESTEIN — prime-N cells routing through Bluestein/Rader.
     * Targets: cells where MKL/FFTW3 are unusually competitive against
     * us. Goal is to identify whether non-codelet stages (chirp modulate,
     * pointwise multiply, demodulate at 16-23% of total per existing
     * src/core/bluestein.h:23-33 perf comment) are hot enough to justify
     * AVX-512 ports of _blue_cmul_sv/_blue_cmul_vv. */
    {    47, 256, "BLUESTEIN", "1.41x MKL / 0.99x FFTW3 — tied",         1.41},
    {    59, 256, "BLUESTEIN", "1.62x MKL / 0.93x FFTW3 — losing FFTW3", 1.62},
    {    83, 256, "BLUESTEIN", "2.83x MKL / 1.01x FFTW3 — barely beats", 2.83},
    {   107, 256, "BLUESTEIN", "1.06x MKL / 1.41x FFTW3 — close to MKL", 1.06},
    {   179, 256, "BLUESTEIN", "1.01x MKL / 0.92x FFTW3 — worst loss",   1.01},
    {   311, 256, "BLUESTEIN", "1.36x MKL / 1.61x FFTW3 — control",      1.36},
};
static const int N_CELLS = (int)(sizeof(CELLS) / sizeof(CELLS[0]));

/* ── Timer ─────────────────────────────────────────────────────── */
static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

/* ── Per-cell ITT task handle name builder ─────────────────────── */
static __itt_string_handle *make_task(const char *prefix, const cell_t *c) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_N%d_K%zu_%s", prefix, c->N, c->K, c->category);
    return __itt_string_handle_create(buf);
}

/* ── Decide rep count to hit ~2 seconds of work per cell ─────────
 * Quick warm + measurement; then size REPS to hit target. */
static int auto_reps(double ns_per_call, double target_seconds) {
    if (ns_per_call <= 0) return 1000;
    int reps = (int)(target_seconds * 1e9 / ns_per_call);
    if (reps < 50)        reps = 50;
    if (reps > 50000000)  reps = 50000000;
    return reps;
}


/* ═══════════════════════════════════════════════════════════════
 * VFFT bench path
 * ═══════════════════════════════════════════════════════════════ */
static double bench_vfft_cell(const cell_t *c, double *re, double *im) {
    vfft_plan p = vfft_plan_c2c(c->N, c->K, VFFT_MEASURE);
    if (!p) {
        fprintf(stderr, "  [vfft] plan_c2c(N=%d K=%zu) failed\n", c->N, c->K);
        return -1;
    }

    /* Warmup + sizing */
    for (int w = 0; w < 5; w++) vfft_execute_fwd(p, re, im);
    double t0 = now_ns();
    vfft_execute_fwd(p, re, im);
    double sample = now_ns() - t0;
    int reps = auto_reps(sample, 2.0);

    /* ITT-marked region — VTune attributes samples here */
    __itt_string_handle *task = make_task("VFFT", c);
    __itt_task_begin(g_dom, __itt_null, __itt_null, task);

    double t_start = now_ns();
    for (int r = 0; r < reps; r++) vfft_execute_fwd(p, re, im);
    double t_end = now_ns();

    __itt_task_end(g_dom);

    double avg_ns = (t_end - t_start) / reps;
    vfft_destroy(p);
    return avg_ns;
}


/* ═══════════════════════════════════════════════════════════════
 * MKL bench path (optional, --mkl flag)
 * ═══════════════════════════════════════════════════════════════ */
#ifdef VFFT_HAS_MKL
static double bench_mkl_cell(const cell_t *c, double *re, double *im) {
    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG strides[2] = {0, (MKL_LONG)c->K};
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)c->N);
    DftiSetValue(desc, DFTI_COMPLEX_STORAGE,    DFTI_REAL_REAL);
    DftiSetValue(desc, DFTI_PLACEMENT,          DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)c->K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE,     1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE,    1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES,      strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES,     strides);
    if (DftiCommitDescriptor(desc) != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        fprintf(stderr, "  [mkl] commit failed\n");
        return -1;
    }

    for (int w = 0; w < 5; w++) DftiComputeForward(desc, re, im);
    double t0 = now_ns();
    DftiComputeForward(desc, re, im);
    double sample = now_ns() - t0;
    int reps = auto_reps(sample, 2.0);

    __itt_string_handle *task = make_task("MKL", c);
    __itt_task_begin(g_dom, __itt_null, __itt_null, task);

    double t_start = now_ns();
    for (int r = 0; r < reps; r++) DftiComputeForward(desc, re, im);
    double t_end = now_ns();

    __itt_task_end(g_dom);

    double avg_ns = (t_end - t_start) / reps;
    DftiFreeDescriptor(&desc);
    return avg_ns;
}
#endif


/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    int do_mkl = 0;
    const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mkl") == 0) do_mkl = 1;
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        }
    }
    /* Open output file (always, even when VTune wraps us — VTune
     * captures stdout but file output goes through). */
    FILE *fout = NULL;
    if (out_path) {
        fout = fopen(out_path, "w");
        if (!fout) fprintf(stderr, "warn: cannot open --output %s\n", out_path);
    }

#ifndef VFFT_HAS_MKL
    if (do_mkl) {
        fprintf(stderr, "Built without VFFT_HAS_MKL — ignoring --mkl flag\n");
        do_mkl = 0;
    }
#endif

    /* ── Init VFFT, load wisdom for MEASURE-grade plans ────────── */
    vfft_init();
    vfft_pin_thread(0);
    vfft_set_num_threads(1);   /* Single-threaded — VTune profiles
                                  cleanest at T=1; threading artifacts
                                  obscure codelet behavior. */

    int wrc = vfft_load_wisdom(
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    fprintf(stderr, "[vfft] wisdom load rc=%d\n", wrc);

#ifdef VFFT_HAS_MKL
    if (do_mkl) mkl_set_num_threads(1);
#endif

    g_dom = __itt_domain_create("VectorFFT.VTuneBench");

    /* Allocate buffers big enough for the largest cell */
    size_t max_NK = 0;
    for (int i = 0; i < N_CELLS; i++) {
        size_t nk = (size_t)CELLS[i].N * CELLS[i].K;
        if (nk > max_NK) max_NK = nk;
    }
    double *re = (double *)vfft_alloc(max_NK * sizeof(double));
    double *im = (double *)vfft_alloc(max_NK * sizeof(double));
    if (!re || !im) { fprintf(stderr, "alloc failed\n"); return 1; }
    srand(42);
    for (size_t i = 0; i < max_NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* ── Print header (also to file if --output given) ─────────── */
    const char *hdr_fmt = "\n%-9s %-8s %-5s %-9s %-12s %-12s %-8s %-8s %-7s\n";
    const char *sep = "---------------------------------------------------------------------------------------\n";
    printf(hdr_fmt, "category", "N", "K", "factors",
           "vfft_ns", "mkl_ns", "vfft_GF", "mkl_GF", "ratio");
    printf("%s", sep);
    if (fout) {
        fprintf(fout, hdr_fmt, "category", "N", "K", "factors",
                "vfft_ns", "mkl_ns", "vfft_GF", "mkl_GF", "ratio");
        fprintf(fout, "%s", sep);
    }

    for (int i = 0; i < N_CELLS; i++) {
        const cell_t *c = &CELLS[i];

        double v_ns = bench_vfft_cell(c, re, im);
        double m_ns = -1.0;
#ifdef VFFT_HAS_MKL
        if (do_mkl) m_ns = bench_mkl_cell(c, re, im);
#endif

        double v_gf = (v_ns > 0) ? 5.0 * c->N * log2((double)c->N) * c->K / v_ns : 0;
        double m_gf = (m_ns > 0) ? 5.0 * c->N * log2((double)c->N) * c->K / m_ns : 0;
        double ratio = (m_ns > 0 && v_ns > 0) ? m_ns / v_ns : 0;

        const char *row_fmt = "%-9s %-8d %-5zu %-9s %12.0f %12.0f %8.2f %8.2f %6.2fx  %s\n";
        printf(row_fmt, c->category, c->N, c->K, "(see wisdom)",
               v_ns, m_ns, v_gf, m_gf, ratio, c->note);
        fflush(stdout);
        if (fout) {
            fprintf(fout, row_fmt, c->category, c->N, c->K, "(see wisdom)",
                    v_ns, m_ns, v_gf, m_gf, ratio, c->note);
            fflush(fout);
        }
    }

    if (fout) fclose(fout);
    vfft_free(re);
    vfft_free(im);
    return 0;
}
