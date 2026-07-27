/* zsplit_wis_gate.c — calibrate→wisdom→create round-trip gate for the K=1
 * SCRAMBLED cascade terminator pick (kind-4 oop_wisdom line, §4.9993).
 * Everything through vfft.h ONLY. Per cell:
 *   create #1 in a scratch VFFT_WISDOM_DIR (MISS -> in-create t2q race,
 *             line banked) -> fwd out1 + roundtrip gate;
 *   file gate: oop_wisdom.txt in the scratch dir now has a "N 1 4 ..." line;
 *   create #2 (HIT -> pure read, no race; create must be faster) -> fwd out2;
 *             memcmp(out1,out2)==0 (the two terminators are bit-identical, so
 *             the route's output must not depend on which was picked);
 *   create #3 with recalibrate=1 (forced re-race) -> fwd out3, memcmp again.
 *
 * Usage: zsplit_wis_gate.exe <scratch-wisdom-dir>
 *   (caller pre-populates the dir with the canonical wisdom set so the
 *    CLASSIC path hits and only the zsplit race runs on create #1)
 * Build: python build.py --src benches/zsplit_wis_gate.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include "vfft.h"

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}

static int find_kind4_line(const char *dir, int N, char *out, size_t outsz)
{
    char path[600];
    snprintf(path, sizeof path, "%s\\oop_wisdom.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[512];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        int n_, k_, kind_;
        if (sscanf(line, "%d %d %d", &n_, &k_, &kind_) == 3 &&
            n_ == N && k_ == 1 && kind_ == 4) {
            snprintf(out, outsz, "%s", line);
            char *nl = strchr(out, '\n');
            if (nl) *nl = 0;
            found = 1;
            break;
        }
    }
    fclose(f);
    return found;
}

static vfft_plan mk(int N, int recal)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.nthreads = 1;
    cfg.recalibrate = recal;
    return vfft_create(&cfg);
}

static int run_cell(int N, const char *wisdir)
{
    int rc = 0;
    double *in  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *o1  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *o2  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *rt  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    srand(N + 77);
    for (int i = 0; i < 2 * N; i++) in[i] = (double)rand() / RAND_MAX - 0.5;

    /* #1: miss -> race + bank */
    double t0 = now_ms();
    vfft_plan h1 = mk(N, 0);
    double c1 = now_ms() - t0;
    if (!h1) { printf("N=%d create#1 FAILED\n", N); return 1; }
    vfft_execute(h1, VFFT_FORWARD, in, NULL, o1, NULL);
    vfft_execute(h1, VFFT_BACKWARD, o1, NULL, rt, NULL);
    double err = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(rt[i] / N - in[i]);
        if (d > err) err = d;
    }
    printf("N=%-6d GATE rtrip#1 relerr=%.3e %s   (create#1 %.1f ms)\n",
           N, err, err < 1e-13 ? "PASS" : "FAIL", c1);
    if (err >= 1e-13) rc = 1;
    vfft_destroy(h1);

    char line[512];
    int have = find_kind4_line(wisdir, N, line, sizeof line);
    printf("N=%-6d GATE kind-4 banked: %s   %s\n",
           N, have ? "PASS" : "FAIL", have ? line : "(missing)");
    if (!have) rc = 1;

    /* #2: hit -> pure read (no race); output must be bit-identical */
    t0 = now_ms();
    vfft_plan h2 = mk(N, 0);
    double c2 = now_ms() - t0;
    if (!h2) { printf("N=%d create#2 FAILED\n", N); return 1; }
    vfft_execute(h2, VFFT_FORWARD, in, NULL, o2, NULL);
    int same = memcmp(o1, o2, (size_t)2 * N * 8) == 0;
    printf("N=%-6d GATE wis-hit bit-identical: %s   (create#2 %.1f ms)\n",
           N, same ? "PASS" : "FAIL", c2);
    if (!same) rc = 1;
    vfft_destroy(h2);

    /* #3: forced recalibrate -> re-race, re-bank. Since the route axis
     * (Phase 5 tranche 2) the re-race may LEGALLY flip the ROUTE on a
     * joint-marginal cell — legacy and ZTURN emit DIFFERENT (both
     * SCRAMBLED-legal) permutations, and recalibrate=1 re-measures by
     * design — so the bit-compare against o1 only applies when the
     * re-banked route matches create#1's; on a flip, gate the roundtrip
     * (each route's bwd must invert its own fwd comb). */
    int r1 = 0, r3 = 0;
    {   int t_, c_; double n_;
        if (sscanf(line, "%*d %*d %*d %d %d %lf %d", &t_, &c_, &n_, &r1) < 4)
            r1 = 0;  /* old-format line = legacy route */
    }
    vfft_plan h3 = mk(N, 1);
    if (!h3) { printf("N=%d create#3 FAILED\n", N); return 1; }
    vfft_execute(h3, VFFT_FORWARD, in, NULL, o2, NULL);
    have = find_kind4_line(wisdir, N, line, sizeof line);
    {   int t_, c_; double n_;
        if (!have ||
            sscanf(line, "%*d %*d %*d %d %d %lf %d", &t_, &c_, &n_, &r3) < 4)
            r3 = 0;
    }
    if (r1 == r3)
        same = memcmp(o1, o2, (size_t)2 * N * 8) == 0;
    else {
        vfft_execute(h3, VFFT_BACKWARD, o2, NULL, rt, NULL);
        err = 0;
        for (int i = 0; i < 2 * N; i++) {
            double d = fabs(rt[i] / N - in[i]);
            if (d > err) err = d;
        }
        same = err < 1e-13;
        printf("N=%-6d NOTE  re-race flipped route %d->%d (joint-marginal "
               "cell); rtrip relerr=%.3e\n", N, r1, r3, err);
    }
    printf("N=%-6d GATE recal re-race: %s   %s\n",
           N, (same && have) ? "PASS" : "FAIL", have ? line : "(missing)");
    if (!same || !have) rc = 1;
    vfft_destroy(h3);

    _mm_free(in); _mm_free(o1); _mm_free(o2); _mm_free(rt);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 2) { printf("usage: zsplit_wis_gate <scratch-wisdom-dir>\n"); return 2; }
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    static char envbuf[640];
    snprintf(envbuf, sizeof envbuf, "VFFT_WISDOM_DIR=%s", argv[1]);
    _putenv(envbuf);
    int rc = 0;
    const int cells[] = { 2048, 4096, 8192, 16384 };
    for (int i = 0; i < 4; i++) rc |= run_cell(cells[i], argv[1]);
    printf(rc ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return rc;
}
