/* k1_pow2_gate.c — the K=1 INTERLEAVED c2c tier at POWER-OF-TWO N through
 * the FRONT DOOR (2026-09-07, the sub-2048 campaign's gate): 128, 256, 512,
 * 1024, 2048, 4096 — the Bailey band, the first cascade cell and one cell
 * above it. No other gate in the suite covers a pow2 K=1 IL plan end to end
 * (flatdit_gate runs odd N; the k1scr / ilp_front gates assert the old
 * scrambled-equals-natural identity). This gate asserts the CONTRACT, prints
 * the route, and never asserts which engine won a cell.
 *
 * COLD on purpose (run_gates: ("flag", False)): the first OOP create of a
 * cell is a wisdom MISS on the scratch store, so the K=1 plan race runs and
 * banks; the second create must REPLAY bit-identically.
 *
 * NATURAL pass, per cell (ORDER_NATURAL, explicit — NOT the default order:
 * under VFFT_ORDER_DEFAULT the engine's own order is a legal answer, and at
 * N >= 2048 that is the cascade's comb, so a DFT check or an in-place-vs-OOP
 * comparison would be asserting a contract the caller never asked for):
 *   1. OOP forward against an independent DFT at sampled natural bins + DC;
 *   2. OOP backward as a roundtrip (unnormalized inverse: N * x);
 *   3. IN-PLACE forward + backward on the same handle (z -> z), the forward
 *      bitwise the OOP forward;
 *   4. a SECOND create replays the banked verdict bit-identically;
 *   5. T=8: the same cell created threaded reproduces the T=1 forward to
 *      1e-11 (BITWISE when the same engine serves — a create at T races its
 *      own verdicts) and roundtrips; engagement printed (serial is legal).
 * SCRAMBLED pass, per cell (ORDER_SCRAMBLED, explicit): forward + backward
 *   roundtrip N * x out of place and in place (the in-place cell's comb may
 *   differ from the OOP cell's — each is its own verdict; printed), replay
 *   bit-identical, T=8 same to 1e-11 + roundtrip; whether the served order is the natural spectrum
 *   (the identity, legal) or a comb is PRINTED, never asserted — the cell's
 *   comb is the served engine's own, which this gate does not decode.
 * PROPERTY: vfft_ilfd_race_short_samples() reads 0 at the end — every flat
 *   form / tile race of the run was decided above the clock's tick.
 *
 * Run:   k1_pow2_gate.exe --wisdir <scratch dir>
 * Build: python build_tuned/build.py --compile --src build_tuned/benches/k1_pow2_gate.c --vfft */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

static const int NS[] = { 128, 256, 512, 1024, 2048, 4096 };

static double now_ms(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return 1e3 * (double)t.QuadPart / (double)f.QuadPart;
}
/* max over 8 sampled natural bins (+ DC) of |X_k - DFT_k| / max|X| */
static double spot_err(const double *x, const double *X, int N)
{
    double mx = 0, e = 0;
    for (int j = 0; j < 2 * N; j++) if (fabs(X[j]) > mx) mx = fabs(X[j]);
    for (int t = 0; t < 9; t++)
    {
        const int k = t ? (t * 7919 + 3) % N : 0;
        double re = 0, im = 0;
        for (int n = 0; n < N; n++)
        {
            const double a = -2.0 * 3.14159265358979323846 * (double)((long long)k * n % N) / N;
            const double c = cos(a), s = sin(a);
            re += x[2 * n] * c - x[2 * n + 1] * s;
            im += x[2 * n] * s + x[2 * n + 1] * c;
        }
        {
            const double d = fabs(X[2 * k] - re) + fabs(X[2 * k + 1] - im);
            if (d > e) e = d;
        }
    }
    return mx > 0 ? e / mx : e;
}
static double relerr(const double *a, const double *b, int N, double scale)
{
    double m = 0, e = 0;
    for (int j = 0; j < 2 * N; j++)
    {
        if (fabs(b[j]) > m) m = fabs(b[j]);
        if (fabs(a[j] * scale - b[j]) > e) e = fabs(a[j] * scale - b[j]);
    }
    return m > 0 ? e / m : e;
}
static vfft_plan mk_t(vfft_wisdom *W, int N, int ip, int order, int T)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = order; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
static vfft_plan mk(vfft_wisdom *W, int N, int ip, int order) { return mk_t(W, N, ip, order, 1); }

/* the cell's served ROUTE read back from the scratch store's K=1 row of the
 * given order cell: the kind-3 IL token (il_route=) or the kind-4 cascade
 * (eng=zturn) — printed with its chain/pair token; "NOROW" when absent. */
static void route_of_store(const char *wisdir, int N, const char *ord, char *out, size_t n)
{
    static const char *files[] = { "wisdom2_oop.txt", "wisdom2_scr.txt" };
    char path[1024], line[4096], key[64], okey[32];
    snprintf(out, n, "NOROW");
    snprintf(key, sizeof key, "n=%d ", N);
    snprintf(okey, sizeof okey, "ord=%s ", ord);
    for (int fi = 0; fi < 2; fi++)
    {
        FILE *f;
        snprintf(path, sizeof path, "%s/%s", wisdir, files[fi]);
        f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f))
        {
            const char *r;
            char tok[64] = "";
            if (!strstr(line, "t=c2c") || !strstr(line, key) || !strstr(line, "q=1 ") || !strstr(line, okey)) continue;
            if (strstr(line, "dir=bwd")) continue;   /* the OOP verdict row may carry role=comp (the pair recipe IS the verdict) */
            if ((r = strstr(line, "il_route=")) != NULL)
            {
                char route[16] = "", pair[32] = "", chain[48] = "";
                const char *q;
                sscanf(r + 9, "%15s", route);
                if ((q = strstr(line, "il_pair=")) != NULL) sscanf(q + 8, "%31s", pair);
                if ((q = strstr(line, "il_flat=")) != NULL) sscanf(q + 8, "%47s", chain);
                if ((q = strstr(line, "il_chain=")) != NULL) sscanf(q + 9, "%47s", chain);
                snprintf(tok, sizeof tok, "%s %s", route, chain[0] ? chain : pair);
            }
            else if (strstr(line, "eng=zturn") || strstr(line, "mode=zcasc"))
            {
                char chain[48] = "";
                const char *q = strstr(line, "chain=");
                if (q) sscanf(q + 6, "%47s", chain);
                snprintf(tok, sizeof tok, "zcasc %s", chain[0] ? chain : "(ref)");
            }
            else if (strstr(line, "mode=ilp"))
                snprintf(tok, sizeof tok, "ilp");
            if (tok[0]) { snprintf(out, n, "%s", tok); fclose(f); return; }
        }
        fclose(f);
    }
}

typedef struct { double fwd, rt, ip, iprt; int replay, mt_bit, mt_ok; long mt_eng; double mt_rt, mt_err, t_race, t_replay; int ok; } cell_t;

/* one order class of one cell: OOP fwd/bwd, IP fwd/bwd, replay, T=8 */
static cell_t run_class(vfft_wisdom *W, int N, int order, const double *x, double *y_ref, int check_dft)
{
    cell_t c; memset(&c, 0, sizeof c); c.fwd = c.rt = c.ip = c.iprt = c.mt_rt = c.mt_err = 1;
    double *y = calloc(2 * (size_t)N, 8), *r = calloc(2 * (size_t)N, 8), *z = calloc(2 * (size_t)N, 8), *y2 = calloc(2 * (size_t)N, 8);
    double t0 = now_ms();
    vfft_plan ho = mk(W, N, 0, order);
    c.t_race = now_ms() - t0;
    if (ho)
    {
        vfft_plan ho2, hi;
        vfft_execute(ho, VFFT_FORWARD, (double *)x, NULL, y, NULL);
        vfft_execute(ho, VFFT_BACKWARD, y, NULL, r, NULL);
        c.fwd = check_dft ? spot_err(x, y, N) : 0.0;
        c.rt = relerr(r, x, N, 1.0 / N);
        t0 = now_ms();
        ho2 = mk(W, N, 0, order);
        c.t_replay = now_ms() - t0;
        if (ho2)
        {
            vfft_execute(ho2, VFFT_FORWARD, (double *)x, NULL, y2, NULL);
            c.replay = (memcmp(y, y2, 2 * (size_t)N * 8) == 0);
            vfft_destroy(ho2);
        }
        hi = mk(W, N, 1, order);
        if (hi)
        {
            memcpy(z, x, 2 * (size_t)N * 8);
            vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
            c.ip = relerr(z, y, N, 1.0);
            vfft_execute(hi, VFFT_BACKWARD, z, NULL, z, NULL);
            c.iprt = relerr(z, x, N, 1.0 / N);
            vfft_destroy(hi);
        }
        {   /* T=8: bitwise the T=1 forward, roundtrip; engagement printed */
            const long e0 = vfft_ilfd_mt_passes() + vfft_zt_mt_passes();
            vfft_plan hm = mk_t(W, N, 0, order, 8);
            const long e1 = vfft_ilfd_mt_passes() + vfft_zt_mt_passes();
            (void)e0;
            if (hm)
            {
                vfft_execute(hm, VFFT_FORWARD, (double *)x, NULL, y2, NULL);
                c.mt_bit = (memcmp(y, y2, 2 * (size_t)N * 8) == 0);
                c.mt_err = relerr(y2, y, N, 1.0);   /* a T-raced create may serve another engine */
                vfft_execute(hm, VFFT_BACKWARD, y2, NULL, r, NULL);
                c.mt_rt = relerr(r, x, N, 1.0 / N);
                c.mt_eng = (vfft_ilfd_mt_passes() + vfft_zt_mt_passes()) - e1;
                c.mt_ok = (c.mt_bit || c.mt_err < 1e-11) && c.mt_rt < 1e-11;
                vfft_destroy(hm);
            }
        }
        if (y_ref) memcpy(y_ref, y, 2 * (size_t)N * 8);
        vfft_destroy(ho);
        c.ok = c.fwd < 1e-11 && c.rt < 1e-11 && (c.ip < 1e-11 || !check_dft) && c.iprt < 1e-11 && c.replay && c.mt_ok;
    }
    free(y); free(r); free(z); free(y2);
    return c;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL; int fails = 0;
    for (int a = 1; a + 1 < argc; a++) if (!strcmp(argv[a], "--wisdir")) wisdir = argv[a + 1];
    if (!wisdir) { printf("usage: %s --wisdir <dir>\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("wisdom load FAILED\n"); return 2; }
    printf("=== K=1 IL c2c at POW2 N through the front door: OOP + IP, both directions, both order classes, replay bit-identical, T=8 bitwise ===\n");
    printf("%-5s | %-22s | %-8s %-8s | %-8s %-8s | %-7s %-7s | %s\n",
           "N", "route (store)", "oop fwd", "oop rt", "ip fwd", "ip rt", "race ms", "replay", "");
    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        double *x = calloc(2 * (size_t)N, 8), *yn = calloc(2 * (size_t)N, 8), *ys = calloc(2 * (size_t)N, 8);
        char route[64], sroute[64];
        cell_t nat, scr;
        srand(1000 + N);
        for (int j = 0; j < 2 * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        nat = run_class(W, N, VFFT_ORDER_NATURAL, x, yn, 1);
        route_of_store(wisdir, N, "nat", route, sizeof route);
        if (!nat.ok) fails++;
        printf("%-5d | %-22s | %.1e  %.1e | %.1e  %.1e | %7.0f %7.0f | %s%s\n", N, route,
               nat.fwd, nat.rt, nat.ip, nat.iprt, nat.t_race, nat.t_replay,
               nat.replay ? "replay bitwise" : "replay DIFFERS", nat.ok ? "" : "   *** FAIL ***");
        printf("   mt | T=8 %s rt %.1e engaged=%ld%s\n", nat.mt_bit ? "BITWISE" : nat.mt_err < 1e-11 ? "same to 1e-11" : "DIFFERS", nat.mt_rt, nat.mt_eng,
               nat.mt_eng > 0 ? " (threaded)" : " (serial verdict)");
        scr = run_class(W, N, VFFT_ORDER_SCRAMBLED, x, ys, 0);
        route_of_store(wisdir, N, "scr", sroute, sizeof sroute);
        if (!scr.ok) fails++;
        {
            const double eid = relerr(ys, yn, N, 1.0);
            printf("  scr | %-22s | %s   rt %.1e | ip %.1e  rt %.1e | %7.0f %7.0f | %s%s\n", sroute,
                   eid < 1e-11 ? "natural served" : "comb served  ", scr.rt, scr.ip, scr.iprt, scr.t_race, scr.t_replay,
                   /* ip column: the in-place cell's forward vs the OOP cell's — its OWN comb is legal */
                   scr.replay ? "replay bitwise" : "replay DIFFERS", scr.ok ? "" : "   *** FAIL ***");
            printf("   mt | T=8 %s rt %.1e engaged=%ld%s\n", scr.mt_bit ? "BITWISE" : scr.mt_err < 1e-11 ? "same to 1e-11" : "DIFFERS", scr.mt_rt, scr.mt_eng,
                   scr.mt_eng > 0 ? " (threaded)" : " (serial verdict)");
        }
        free(x); free(yn); free(ys);
    }
    {
        const long short_samples = vfft_ilfd_race_short_samples();
        printf("\nrace clock: short samples = %ld%s\n", short_samples, short_samples ? "   *** FAIL ***" : "");
        if (short_samples) fails++;
    }
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
