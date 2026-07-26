/* bench_phase0b.c — Phase 0b paced isolated A/B (docs/roadmap/
 * cascade_load_path_restructure.md, Phase 0b).
 *
 * TWO ARM PAIRS, each with a control arm (3 arms per pair):
 *   PAIR A (INGEST):      arm0 = radix4_z_s0s_fwd_avx2 (incumbent, real kernel
 *                                object, 16 p5/16cx)
 *                         arm1 = s0t_fwd (challenger, #included VERBATIM from
 *                                spike_s0t_ingest.c — the Phase-0-proven body,
 *                                18 p5/16cx, deinterleave+turn fused)
 *                         arm2 = radix4_z_s0s_fwd_avx2 again (CONTROL)
 *     Shape: whole-plane pass, 4 interleaved leg streams at stride N/4 complex
 *     in (zin plane), block-split records out (shared out plane).
 *     GATE A: best(s0t)/best(s0s) <= 1.20 per cell.
 *   PAIR B (TERMINATOR):  arm0 = radix8_z_sterm_fwd_avx2 (incumbent, real
 *                                kernel object, 64 p5/iter, TR4 loads)
 *                         arm1 = radix8_z_stermt_fwd_avx2 (turn-free, TR4s
 *                                deleted -> 16 contiguous loads, 32 p5/iter;
 *                                scratchpad stermt.c, full-cascade-proven
 *                                bit-identical via the turned route)
 *                         arm2 = radix8_z_sterm_fwd_avx2 again (CONTROL)
 *     Shape: count = N/8 columns, input = the REAL cascade-front state
 *     (s0s + all mids run once via zsplit.h kernels, then copied into the
 *     arena plane T), twiddles = p->twq (packed per-column w^1) exactly as
 *     vfft_zsplit_create builds them. Re-runs [M2] under a passing control.
 *
 * PLAN / TWIDDLES: vfft_zsplit_create() from src/core/oop/zsplit.h — the real
 * production plan path (calibrated chains 2048:4.8.8.8, 4096:4.4.4.8.8,
 * 8192:4.4.8.8.8, 16384:4.8.8.8.8; all R0=4, terminator radix-8). p->t2q is
 * forced to 0: the timed incumbent is single-quad sterm by the brief.
 *
 * HARNESS DISCIPLINE (each item = a scar; see the Phase 0b brief):
 *  1. Pinned logical core 2: SetProcessAffinityMask AND SetThreadAffinityMask
 *     with mask 4, HIGH_PRIORITY_CLASS. FATAL exit(2) if either mask call
 *     fails. NO spinner on the SMT sibling — it must stay idle.
 *  2. Cachebust = 64 MB sweep (write+read every 64 B) — covers the 36 MB L3.
 *  3. Pacing: cachebust FIRST (it is work, it heats), THEN Sleep(pace_ms),
 *     THEN the timed region. Default pace_ms = 1000. Between cells (and
 *     between pairs of the same cell): Sleep(5*pace_ms) (default 5000 ms).
 *  4. Rotation: in rep r the execution order is arms (r%3, (r+1)%3, (r+2)%3),
 *     so every arm visits every thermal slot equally; reps is FORCED UP to a
 *     multiple of 3 (the arm count).
 *  5. ONE arena (_aligned_malloc, 4 KB base), planes at stride sz+64 so
 *     consecutive planes are 64-byte-skewed mod 4 KB (no 4 KB aliasing).
 *     ONE shared output plane: every arm of both pairs writes plane `out`.
 *  6. Controls (arm2 = same incumbent code, rotated slot) must read
 *     0.98-1.02 x incumbent; outside that band the cell is printed VOID.
 *  7. best-of over rotated reps; >= 12 samples per arm per cell (default
 *     reps = 12; one sample per arm per rep).
 *  8. Timer: QueryPerformanceCounter; reported ns are per whole-plane pass.
 *  9. Inner repeats per sample sized once per (cell,pair) from an incumbent
 *     estimate so each timed region >= ~1.2 ms; IDENTICAL inner for all arms.
 *
 * SMOKE (runs before any timing; --smoke = run only this):
 *   s1: s0t_fwd output vs the spike's scalar index-map reference, <= 1e-13.
 *   s2: full-cascade bit-identity: incumbent vfft_zsplit_execute_fwd (t2q=0)
 *       vs the turned route (s0s -> msg mids -> msgt_oop turn at stage nf-2
 *       -> stermt), memcmp == 0.  (msgt_oop comes from scratchpad
 *       gate1_kernels.h — the earlier agent's proven turn store.)
 *   s3: pair-B incumbent arm equivalence: sterm(T -> out) memcmp == 0 vs the
 *       execute_fwd output (T = copy of the front-run scratch state).
 *   s4: stermt(T -> out) single pass: all outputs finite (T's byte layout is
 *       the timing shape; stermt value-correctness is s2).
 *   s5: all outputs finite (s0s, s0t, both cascade routes).
 *   Any smoke failure => exit(1), no timing.
 *
 * USAGE: bench_phase0b [N|all] [A|B|both] [reps] [pace_ms] [--smoke]
 *   defaults:            all     both      12      1000
 *
 * BUILD (mandated compiler, see Phase 0b report for exact recorded commands):
 *   C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma
 *     -I <repo>/src/core/oop -I <scratchpad>
 *     bench_phase0b.c + radix4_z_s0s_avx2.c radix8_z_s0s_avx2.c
 *     radix4_z_msg_avx2.c radix8_z_msg_avx2.c radix8_z_sterm_avx2.c
 *     radix8_z_sterm2_avx2.c + <scratchpad>/stermt.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include "zsplit.h"            /* real plan path: src/core/oop */

/* the Phase-0-proven ingest body + its scalar reference, included VERBATIM
 * (its self-test main is renamed away). */
#define main spike_s0t_ingest_main
#include "spike_s0t_ingest.c"
#undef main

#include "gate1_kernels.h"     /* scratchpad: _b_turn/msgt_oop (proven turn) */

extern void radix8_z_stermt_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);

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

/* ---------------------------------------------------------------- cascades */
/* incumbent front: s0s + ALL mids, leaves p->sp = terminator input */
static void front_incumbent(const vfft_zsplit_plan_t *p, const double *zin)
{
    const int nf = p->nf;
    if (p->chain[0] == 8)
        radix8_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
            (unsigned long long)p->D[0], 0, (unsigned long long)p->D[0], 0,
            (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
            (unsigned long long)p->D[0], 0, (unsigned long long)p->D[0], 0,
            (unsigned long long)p->D[0]);
    for (int s = 1; s <= nf - 2; s++) {
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2 : radix4_z_msg_fwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twsp[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s], 0, 0,
          (unsigned long long)p->D[s]);
    }
}

/* turned route (identical to the earlier agent's proven turned_cascade.c):
 * s0s -> mids 1..nf-3 in place -> msgt_oop turn at stage nf-2 (sp -> B)
 * -> stermt (B -> zout) */
static void exec_fwd_turned(const vfft_zsplit_plan_t *p, const double *zin,
                            double *zout, double *B)
{
    const int nf = p->nf;
    if (p->chain[0] == 8)
        radix8_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
            (unsigned long long)p->D[0], 0, (unsigned long long)p->D[0], 0,
            (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
            (unsigned long long)p->D[0], 0, (unsigned long long)p->D[0], 0,
            (unsigned long long)p->D[0]);
    for (int s = 1; s <= nf - 3; s++) {
        if (p->chain[s] == 8)
            radix8_z_msg_fwd_avx2(p->sp, 0, p->sp, 0, p->twsp[s], 0,
                (unsigned long long)p->D[s], (unsigned long long)p->G[s], 0, 0,
                (unsigned long long)p->D[s]);
        else
            radix4_z_msg_fwd_avx2(p->sp, 0, p->sp, 0, p->twsp[s], 0,
                (unsigned long long)p->D[s], (unsigned long long)p->G[s], 0, 0,
                (unsigned long long)p->D[s]);
    }
    {   const int s = nf - 2;   /* radix-8 at every calibrated chain */
        msgt_oop(p->sp, B, p->twsp[s], (size_t)p->D[s], (size_t)p->G[s],
                 (size_t)p->D[s]);
    }
    radix8_z_stermt_fwd_avx2(B, 0, zout, 0, p->twq, 0, 0, 0,
        (size_t)(p->N / 8), 0, (size_t)(p->N / 8));
}

/* ------------------------------------------------------------------- cell */
typedef struct {
    int N;
    const vfft_zsplit_plan_t *p;
    const double *zin;   /* interleaved input plane                  */
    double *out;         /* ONE shared output plane, all arms        */
    const double *T;     /* pair-B input: copied cascade-front state */
} cell_t;

/* arm dispatch — arms 0 and 2 are the SAME incumbent code (control) */
static void call_arm_A(const cell_t *c, int arm)
{
    if (arm == 1)
        s0t_fwd(c->zin, c->out, (size_t)c->N);
    else
        radix4_z_s0s_fwd_avx2(c->zin, 0, c->out, 0, 0, 0,
            (unsigned long long)(c->N / 4), 0, (unsigned long long)(c->N / 4),
            0, (unsigned long long)(c->N / 4));
}
static void call_arm_B(const cell_t *c, int arm)
{
    if (arm == 1)
        radix8_z_stermt_fwd_avx2(c->T, 0, c->out, 0, c->p->twq, 0, 0, 0,
            (size_t)(c->N / 8), 0, (size_t)(c->N / 8));
    else
        radix8_z_sterm_fwd_avx2(c->T, 0, c->out, 0, c->p->twq, 0, 0, 0,
            (unsigned long long)(c->N / 8), 0, (unsigned long long)(c->N / 8));
}

#define NARMS 3

static int run_pair(const cell_t *c, char pair, int reps, int pace_ms)
{
    void (*call)(const cell_t *, int) = (pair == 'A') ? call_arm_A : call_arm_B;
    const char *nm_inc  = (pair == 'A') ? "s0s"  : "sterm";
    const char *nm_chal = (pair == 'A') ? "s0t"  : "stermt";

    /* inner repeats: sized ONCE from the incumbent, identical for all arms */
    call(c, 0); call(c, 0); call(c, 0);                     /* warm */
    double t0 = now_ns();
    for (int i = 0; i < 8; i++) call(c, 0);
    double est = (now_ns() - t0) / 8.0;
    if (est < 1.0) est = 1.0;
    long inner = (long)(1.2e6 / est) + 1;
    if (inner < 1) inner = 1;
    if (inner > 2000000) inner = 2000000;

    double best[NARMS] = { 1e30, 1e30, 1e30 };
    double *smp = (double *)malloc((size_t)reps * NARMS * sizeof(double));
    printf("  PAIR %c  N=%-6d arms: 0=%s(incumbent) 1=%s(challenger) "
           "2=%s(CONTROL)  inner=%ld  est=%.0fns\n",
           pair, c->N, nm_inc, nm_chal, nm_inc, inner, est);
    fflush(stdout);

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j < NARMS; j++) {
            int a = (j + r) % NARMS;              /* rotation, item 4 */
            cachebust();                          /* bust FIRST (it heats) */
            Sleep((DWORD)pace_ms);                /* THEN cool           */
            double s = now_ns();
            for (long i = 0; i < inner; i++) call(c, a);
            double e = (now_ns() - s) / (double)inner;
            smp[r * NARMS + a] = e;
            if (e < best[a]) best[a] = e;
        }
    }
    g_sink += c->out[0];

    for (int r = 0; r < reps; r++)
        printf("    rep %2d (order %d,%d,%d): inc=%9.1f  chal=%9.1f  ctl=%9.1f\n",
               r, r % NARMS, (r + 1) % NARMS, (r + 2) % NARMS,
               smp[r * NARMS + 0], smp[r * NARMS + 1], smp[r * NARMS + 2]);

    double ratio = best[1] / best[0], ctl = best[2] / best[0];
    int ctl_ok = (ctl >= 0.98 && ctl <= 1.02);
    printf("  PAIR %c  N=%-6d best: inc=%9.1f  chal=%9.1f  ctl=%9.1f | "
           "chal/inc=%.4f  ctl/inc=%.4f | %s",
           pair, c->N, best[0], best[1], best[2], ratio, ctl,
           ctl_ok ? "CTL PASS" : "CTL OUT OF BAND -- RUN VOID");
    if (pair == 'A' && ctl_ok)
        printf(" | GATE A (<=1.20): %s", ratio <= 1.20 ? "PASS" : "FAIL");
    printf("\n\n");
    fflush(stdout);
    free(smp);
    return ctl_ok;
}

/* ------------------------------------------------------------------ smoke */
static int smoke_cell(const cell_t *c, vfft_zsplit_plan_t *p,
                      double *zo_ref, double *zo2, double *B, double *T)
{
    const int N = c->N; const size_t nd = (size_t)2 * N; int ok = 1;

    /* s1: s0t vs the spike's scalar reference */
    s0t_fwd(c->zin, c->out, (size_t)N);
    s0t_ref(c->zin, zo2, (size_t)N);
    double mx = 0.0;
    for (size_t i = 0; i < nd; i++) {
        double d = fabs(c->out[i] - zo2[i]); if (d > mx) mx = d;
    }
    int s1 = (mx <= 1e-13) && all_finite(c->out, nd);
    printf("    s1 s0t vs scalar ref : maxabs=%.3e %s\n", mx, s1 ? "PASS" : "FAIL");

    /* s2: full-cascade bit-identity, incumbent (t2q=0) vs turned route */
    vfft_zsplit_execute_fwd(p, c->zin, zo_ref);
    exec_fwd_turned(p, c->zin, zo2, B);
    int s2 = (memcmp(zo_ref, zo2, nd * sizeof(double)) == 0)
             && all_finite(zo_ref, nd) && all_finite(zo2, nd);
    mx = 0.0;
    for (size_t i = 0; i < nd; i++) {
        double d = fabs(zo_ref[i] - zo2[i]); if (d > mx) mx = d;
    }
    printf("    s2 cascade bitcmp    : %s (maxabs %.3g)\n",
           s2 ? "IDENTICAL" : "**DIFFER**", mx);

    /* prepare T = real cascade-front state; s3: timed incumbent arm == plan */
    front_incumbent(p, c->zin);
    memcpy(T, p->sp, nd * sizeof(double));
    radix8_z_sterm_fwd_avx2(T, 0, c->out, 0, p->twq, 0, 0, 0,
        (unsigned long long)(N / 8), 0, (unsigned long long)(N / 8));
    int s3 = (memcmp(c->out, zo_ref, nd * sizeof(double)) == 0);
    printf("    s3 sterm(T) == plan  : %s\n", s3 ? "IDENTICAL" : "**DIFFER**");

    /* s4: stermt single pass on T — timing shape, outputs finite */
    radix8_z_stermt_fwd_avx2(T, 0, c->out, 0, p->twq, 0, 0, 0,
        (size_t)(N / 8), 0, (size_t)(N / 8));
    int s4 = all_finite(c->out, nd);
    printf("    s4 stermt(T) finite  : %s\n", s4 ? "PASS" : "FAIL");

    ok = s1 && s2 && s3 && s4;
    printf("    SMOKE N=%-6d %s\n", N, ok ? "PASS" : "**FAIL**");
    return ok;
}

/* ------------------------------------------------------------------- main */
int main(int argc, char **argv)
{
    int cells_all[4] = { 2048, 4096, 8192, 16384 };
    int cells[4], ncells = 0;
    char pair = '2';            /* '2' = both */
    int reps = 12, pace_ms = 1000, smoke_only = 0, pos = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--smoke") == 0) { smoke_only = 1; continue; }
        switch (pos++) {
        case 0:
            if (strcmp(argv[i], "all") == 0) break;
            cells[ncells++] = atoi(argv[i]); break;
        case 1:
            if (!strcmp(argv[i], "A") || !strcmp(argv[i], "a")) pair = 'A';
            else if (!strcmp(argv[i], "B") || !strcmp(argv[i], "b")) pair = 'B';
            else pair = '2';        /* "both" (and anything else) = both */
            break;
        case 2: reps = atoi(argv[i]); break;
        case 3: pace_ms = atoi(argv[i]); break;
        }
    }
    if (ncells == 0) { memcpy(cells, cells_all, sizeof(cells_all)); ncells = 4; }
    if (reps < 1) reps = 1;
    if (reps % NARMS) reps += NARMS - reps % NARMS;   /* item 4: multiple of 3 */

    /* item 1: pin logical core 2, both process and thread; FATAL on failure */
    if (!SetProcessAffinityMask(GetCurrentProcess(), 4)) {
        fprintf(stderr, "FATAL: SetProcessAffinityMask(4) failed\n"); return 2;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), 4)) {
        fprintf(stderr, "FATAL: SetThreadAffinityMask(4) failed\n"); return 2;
    }
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
        fprintf(stderr, "WARN: HIGH_PRIORITY_CLASS not set\n");
    _mm_setcsr(_mm_getcsr() | 0x8040);                /* FTZ|DAZ, both routes */

    g_bust = (char *)malloc(64u << 20);
    if (!g_bust) { fprintf(stderr, "FATAL: bust alloc\n"); return 2; }
    memset(g_bust, 1, 64u << 20);

    printf("bench_phase0b: pair=%s reps=%d (x3 arms) pace=%dms cellpace=%dms "
           "%s\n", pair == '2' ? "both" : (pair == 'A' ? "A" : "B"),
           reps, pace_ms, 5 * pace_ms, smoke_only ? "[SMOKE ONLY]" : "");

    int any_void = 0, any_smoke_fail = 0;
    for (int ci = 0; ci < ncells; ci++) {
        int N = cells[ci];
        int chain[VFFT_ZSPLIT_MAX_NF];
        int nf = vfft_zsplit_default_chain(N, chain);
        if (!nf) { printf("N=%d: no calibrated chain, skip\n", N); continue; }
        vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
        if (!p) { printf("N=%d: plan create failed\n", N); return 2; }
        p->t2q = 0;   /* timed incumbent = single-quad sterm (the brief) */

        /* item 5: ONE arena, planes 64-B skewed, one shared out plane */
        size_t sz = (size_t)2 * N * sizeof(double);
        size_t stride = sz + 64;
        char *arena = (char *)_aligned_malloc(6 * stride + 64, 4096);
        if (!arena) { fprintf(stderr, "FATAL: arena alloc\n"); return 2; }
        double *zin    = (double *)(arena + 0 * stride);
        double *out    = (double *)(arena + 1 * stride);   /* SHARED */
        double *T      = (double *)(arena + 2 * stride);
        double *zo_ref = (double *)(arena + 3 * stride);
        double *zo2    = (double *)(arena + 4 * stride);
        double *B      = (double *)(arena + 5 * stride);
        for (int i = 0; i < N; i++) {           /* deterministic, spike fill */
            zin[2 * i]     = sin(0.7 * (double)i) + (double)i * 1e-3;
            zin[2 * i + 1] = cos(1.3 * (double)i);
        }

        char cs[64]; int o = 0;
        for (int s = 0; s < nf; s++)
            o += sprintf(cs + o, "%d%s", chain[s], s < nf - 1 ? "." : "");
        printf("== N=%d chain=%s ==\n", N, cs);

        if (!smoke_cell((const cell_t[]){{ N, p, zin, out, T }}, p,
                        zo_ref, zo2, B, T)) {
            any_smoke_fail = 1;
            printf("  smoke FAILED -- no timing at this cell\n\n");
            _aligned_free(arena); vfft_zsplit_destroy(p);
            continue;
        }
        if (!smoke_only) {
            cell_t c = { N, p, zin, out, T };
            if (pair != 'B') { if (!run_pair(&c, 'A', reps, pace_ms)) any_void = 1;
                               Sleep((DWORD)(5 * pace_ms)); }
            if (pair != 'A') { if (!run_pair(&c, 'B', reps, pace_ms)) any_void = 1; }
        }
        _aligned_free(arena);
        vfft_zsplit_destroy(p);
        if (ci + 1 < ncells) Sleep((DWORD)(5 * pace_ms));  /* item 3 */
    }
    (void)g_sink;
    if (any_smoke_fail) return 1;
    if (any_void) {
        printf("NOTE: at least one control out of band -- those cells are "
               "VOID; re-run with more pacing/reps.\n");
        return 3;               /* exit 3 = completed but >=1 cell VOID */
    }
    return 0;
}
