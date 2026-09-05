/* THE UNORDERED-LANE A/B (2026-09-05): the ZTURN-S fwd cascade with ordered
 * plane lanes (s0t + stf / stf2) vs the [0,2,1,3] twins (s0tu + stfu /
 * stf2u) — same chain, same t2q, ONE process, one input arena, one output
 * arena per arm. Correctness = the twins must be BIT-IDENTICAL to the
 * gate-proven ordered kernels (the per-lane arithmetic is the same, only
 * the lane positions differ). Race protocol (memory: thermally noisy box):
 * core 2, HIGH priority, alternated order per round, N rounds of a ~2 ms
 * batch each, medians, control = ordered vs ordered (A' vs A); a delta
 * inside the control's own delta or inside A's spread is NOT a result.
 *   python build_tuned/build.py --compile --src build_tuned/benches/lanesu_probe.c --vfft
 *   lanesu_probe.exe [rounds] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
#include "oop/zturn.h"

static double qpc_ns(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{ double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double med(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }
static double spr(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return (v[n - 1 - n / 10] - v[n / 10]) / v[n / 2]; }
static double batch_ns(const vfft_zturn2_plan_t *p, const double *zin, double *zout, int reps)
{
    double t0 = qpc_ns();
    for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd(p, zin, zout);
    return (qpc_ns() - t0) / reps;
}

int main(int argc, char **argv)
{
    static const int NS[] = { 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
    const int n = (int)(sizeof NS / sizeof NS[0]);
    const int rounds = (argc > 1) ? atoi(argv[1]) : 15;
    int bad = 0, cells = 0;
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    printf("%-7s %-16s %-5s | %-8s %-8s | %8s %8s | %6s | %7s | %7s | verdict\n",
           "N", "chain", "t2q", "output", "maxdiff", "A ns", "B ns", "B/A", "A'/A", "spreadA");
    for (int i = 0; i < n; i++) {
        const int N = NS[i];
        int chain[VFFT_ZSPLIT_MAX_NF];
        const int nf = vfft_zsplit_default_chain(N, chain);
        char cs[64];
        if (nf < 3) { printf("%-7d NO CHAIN\n", N); continue; }
        { int off = 0; for (int s = 0; s < nf; s++) off += snprintf(cs + off, sizeof cs - off, "%s%d", s ? "." : "", chain[s]); }
        for (int t2q = 0; t2q <= (chain[nf - 1] == 8); t2q++) {
            vfft_zturn2_plan_t *pa = vfft_zturn2_create_chain_u(N, chain, nf, 0);
            vfft_zturn2_plan_t *pb = vfft_zturn2_create_chain_u(N, chain, nf, 1);
            double *zin = (double *)_aligned_malloc((size_t)N * 16, 64);
            double *za  = (double *)_aligned_malloc((size_t)N * 16 + 64, 64);
            double *zb  = (double *)_aligned_malloc((size_t)N * 16 + 64, 64);
            double A[128], B[128], C[128];
            if (!pa || !pb || !zin || !za || !zb) { printf("%-7d %-16s t2q=%d NO PLAN\n", N, cs, t2q); bad++; continue; }
            pa->t2q = t2q; pb->t2q = t2q;
            for (int j = 0; j < 2 * N; j++) zin[j] = (double)rand() / RAND_MAX - 0.5;
            /* correctness: bit-identical spectra (scrambled order, same in both) */
            vfft_zturn2_execute_fwd(pa, zin, za);
            vfft_zturn2_execute_fwd(pb, zin, zb);
            const int ident = (memcmp(za, zb, (size_t)N * 16) == 0);
            double mx = 0;
            for (int j = 0; j < 2 * N; j++) { double d = fabs(za[j] - zb[j]); if (d > mx) mx = d; }
            if (!ident) bad++;
            cells++;
            /* race: ~2 ms batches, alternated order, control = A again */
            int reps;
            { double t1 = batch_ns(pa, zin, za, 4); reps = (int)(2.0e6 / t1); if (reps < 1) reps = 1; if (reps > 4000) reps = 4000; }
            for (int r = 0; r < rounds; r++) {
                if (r & 1) {
                    C[r] = batch_ns(pa, zin, za, reps);
                    B[r] = batch_ns(pb, zin, zb, reps);
                    A[r] = batch_ns(pa, zin, za, reps);
                } else {
                    A[r] = batch_ns(pa, zin, za, reps);
                    B[r] = batch_ns(pb, zin, zb, reps);
                    C[r] = batch_ns(pa, zin, za, reps);
                }
            }
            {
                double ta[128], tb[128], tc[128];
                memcpy(ta, A, sizeof(double) * rounds); memcpy(tb, B, sizeof(double) * rounds); memcpy(tc, C, sizeof(double) * rounds);
                const double ma = med(ta, rounds), mb = med(tb, rounds), mc = med(tc, rounds);
                memcpy(ta, A, sizeof(double) * rounds);
                const double sa = spr(ta, rounds);
                const double d = mb / ma - 1.0, dc = fabs(mc / ma - 1.0);
                const char *verdict = (fabs(d) > sa && fabs(d) > 2.0 * dc) ? (d < 0 ? "B FASTER" : "B SLOWER") : "no result (inside noise)";
                printf("%-7d %-16s t2q=%d | %-8s %-8.1e | %8.0f %8.0f | %6.3f | %7.3f | %6.1f%% | %s\n",
                       N, cs, t2q, ident ? "BITWISE" : "DIFF", mx, ma, mb, mb / ma, mc / ma, 100.0 * sa, verdict);
                fflush(stdout);
            }
            /* STAGE-LEVEL: the terminator alone and the ingest alone (the only
             * kernels whose code differs), same protocol, on each plan's own
             * plane/tables after one full execute. */
            {
                const size_t Nn = (size_t)N;
                const int r4 = (chain[nf - 1] == 4);
                double TA[128], TB[128], IA[128], IB[128];
                vfft_zturn2_execute_fwd(pa, zin, za); vfft_zturn2_execute_fwd(pb, zin, zb);
                for (int r = 0; r < rounds; r++) {
                    for (int arm = 0; arm < 2; arm++) {
                        const int useb = (r & 1) ? !arm : arm;
                        const vfft_zturn2_plan_t *p = useb ? pb : pa;
                        double *zo = useb ? zb : za;
                        _vfft_zt_msg_fn tf = r4 ? _vfft_zt_stf4_fwd_pick(p->lanes_u) : _vfft_zt_stf8_fwd_pick(p->lanes_u, p->t2q);
                        _vfft_zt_msg_fn sf = _vfft_zt_s0t_fwd_pick(p->lanes_u);
                        const size_t cnt = r4 ? Nn / 4 : Nn / 8;
                        double t0 = qpc_ns();
                        for (int q = 0; q < reps; q++) tf(p->plane, 0, zo, 0, p->tzq, 0, 0, 0, cnt, 0, cnt);
                        double tt = (qpc_ns() - t0) / reps;
                        t0 = qpc_ns();
                        for (int q = 0; q < reps; q++) sf(zin, 0, p->plane, 0, 0, 0, Nn / 4, 0, 0, 0, Nn / 4);
                        double ti = (qpc_ns() - t0) / reps;
                        if (useb) { TB[r] = tt; IB[r] = ti; } else { TA[r] = tt; IA[r] = ti; }
                    }
                }
                {
                    double v[128];
                    memcpy(v, TA, sizeof(double) * rounds); const double mta = med(v, rounds);
                    memcpy(v, TB, sizeof(double) * rounds); const double mtb = med(v, rounds);
                    memcpy(v, IA, sizeof(double) * rounds); const double mia = med(v, rounds);
                    memcpy(v, IB, sizeof(double) * rounds); const double mib = med(v, rounds);
                    memcpy(v, TA, sizeof(double) * rounds); const double sta = spr(v, rounds);
                    printf("        stages: terminator A %7.0f  B %7.0f  B/A %.3f (spread %.1f%%) | ingest A %7.0f  B %7.0f  B/A %.3f\n",
                           mta, mtb, mtb / mta, 100.0 * sta, mia, mib, mib / mia);
                    fflush(stdout);
                }
            }
            vfft_zturn2_destroy(pa); vfft_zturn2_destroy(pb);
            _aligned_free(zin); _aligned_free(za); _aligned_free(zb);
        }
    }
    printf(bad ? "=== *** %d BAD *** ===\n" : "=== ALL BITWISE (%d cells) ===\n", bad ? bad : cells);
    return bad ? 1 : 0;
}
