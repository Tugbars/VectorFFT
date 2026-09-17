/* ilprime_chain_probe.c — race named INNER verdicts of one prime cell
 * SAME-RUN through the front door (2026-09-18).
 *
 * The prime cell's cold race is one sample per arm; this probe re-races a
 * few named verdicts with the house protocol (core 2, HIGH, alternating
 * order, min and median of R rounds, pacing BETWEEN rounds) so a banked
 * verdict can be checked against the chain the K=1 tier used to lend.
 * Each arm is a real front-door plan: the scratch store's prime row is
 * rewritten to the arm's tokens, a fresh wisdom handle loads it, and the
 * create REPLAYS it (VFFT_ILPR_LOG shows the replay line).
 *
 * Run:   ilprime_chain_probe.exe <scratch store> <N> [rounds] kind:shape:tw ...
 *        e.g. ... 131071 15 ztt:8.4.8.4.8.8.4:0 ztt:8.8.8.8.8.8:2048
 * Build: python build.py --compile --vfft --src benches/ilprime_chain_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

#define MAXA 12
static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
/* rewrite n=<N>'s prime row: eng= and the inner tokens; 1 if a row was rewritten */
static int set_row(const char *dir, int N, const char *eng, const char *kind, const char *shape, int tw)
{
    char path[700], tmp[700], line[4096], key[32];
    FILE *f, *g;
    int hit = 0;
    snprintf(path, sizeof path, "%s/wisdom2_prime.txt", dir);
    snprintf(tmp, sizeof tmp, "%s/wisdom2_prime.tmp", dir);
    snprintf(key, sizeof key, "n=%d ", N);
    f = fopen(path, "rb");
    if (!f) return 0;
    g = fopen(tmp, "wb");
    if (!g) { fclose(f); return 0; }
    while (fgets(line, sizeof line, f))
    {
        if (!strncmp(line, "@cell", 5) && strstr(line, key) && strstr(line, "t=c2c"))
        {
            char *bar = strchr(line, '|');
            if (bar)
            {
                *bar = 0;
                fprintf(g, "%s| eng=%s in=%s in_sh=%s in_tw=%d | ran=1 src=probe date=2026-09-18\n",
                        line, eng, kind, shape, tw);
                hit = 1;
                continue;
            }
        }
        fputs(line, g);
    }
    if (!hit)
        fprintf(g, "@cell t=c2c n=%d q=1 ord=scr place=ip role=comp lay=il | eng=%s in=%s in_sh=%s in_tw=%d | ran=1 src=probe date=2026-09-18\n",
                N, eng, kind, shape, tw);
    fclose(f); fclose(g);
    return MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING) ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const int N = argc > 2 ? atoi(argv[2]) : 131071;
    const int R = argc > 3 ? atoi(argv[3]) : 15;
    int na = 0, a, r, i;
    vfft_plan P[MAXA];
    vfft_wisdom *H[MAXA];
    char name[MAXA][64];
    double *x, *y, best[MAXA], med[MAXA], samp[MAXA][256];
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    _putenv("VFFT_ILPR_LOG=1");
    for (i = 4; i < argc && na < MAXA; i++)
    {
        char kind[8], shape[64];
        int tw = 0;
        const char *eng;
        if (sscanf(argv[i], "%7[^:]:%63[^:]:%d", kind, shape, &tw) < 2) { printf("bad arm %s\n", argv[i]); return 2; }
        /* the method follows M: Rader iff the shape's product is N-1 is not knowable here,
         * so the caller names it through the kind prefix "r-" (Rader) else Bluestein */
        eng = "bluestein";
        if (kind[0] == 'r' && kind[1] == '-') { eng = "rader"; memmove(kind, kind + 2, strlen(kind + 2) + 1); }
        if (!set_row(dir, N, eng, kind, shape, tw)) { printf("cannot write the row\n"); return 2; }
        H[na] = vfft_wisdom_load(dir);
        {
            vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_PATIENT;
            cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_NATURAL;
            cfg.nthreads = 1; cfg.wisdom = H[na]; cfg.wisdom_write = 0;
            P[na] = vfft_create(&cfg);
        }
        snprintf(name[na], sizeof name[na], "%s %s %s tw=%d", eng, kind, shape, tw);
        if (!P[na]) { printf("arm %s: create REFUSED\n", name[na]); vfft_wisdom_free(H[na]); continue; }
        na++;
    }
    if (na == 0) { printf("no arms\n"); return 2; }
    x = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
    y = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
    for (i = 0; i < 2 * N; i++) x[i] = 1.0 + 1e-6 * (double)(i & 255);
    for (a = 0; a < na; a++) { vfft_execute(P[a], VFFT_FORWARD, x, NULL, y, NULL); best[a] = 1e300; }
    for (r = 0; r < R && r < 256; r++)
    {
        for (i = 0; i < na; i++)
        {
            double t0, t1;
            a = (r & 1) ? na - 1 - i : i;   /* alternate the order */
            t0 = now_ns();
            vfft_execute(P[a], VFFT_FORWARD, x, NULL, y, NULL);
            t1 = now_ns();
            samp[a][r] = t1 - t0;
            if (t1 - t0 < best[a]) best[a] = t1 - t0;
        }
        Sleep(200);   /* pace BETWEEN rounds */
    }
    printf("N=%d, %d rounds, min / median ns per arm:\n", N, R);
    for (a = 0; a < na; a++)
    {
        qsort(samp[a], (size_t)R, sizeof(double), cmpd);
        med[a] = samp[a][R / 2];
        printf("  %-44s min %9.0f   med %9.0f\n", name[a], best[a], med[a]);
    }
    for (a = 0; a < na; a++) { vfft_destroy(P[a]); vfft_wisdom_free(H[a]); }
    _aligned_free(x); _aligned_free(y);
    return 0;
}
