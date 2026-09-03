/* zen4_il_race.c — race the K=1 INTERLEAVED c2c cells at one N through the
 * PUBLIC front door at VFFT_PATIENT and PERSIST the verdicts into the store
 * named by $VFFT_WISDOM_DIR (the per-host folder; wisdom2/README §2.2).
 *
 * The three cells are the ones bench_1d_vs_fftw.c consumes:
 *   oop/scr  — the kind-4 ZTURN/zsplit cascade   (the bench's default mode)
 *   ip/nat   — natural in-place, k1nat            (--k1nat)
 *   oop/nat  — natural out-of-place, k1noop       (--k1noop)
 * A store MISS races at cfg.rigor and banks; a HIT is served untouched, so
 * re-running is safe and cheap. Set VFFT_RECALIBRATE=1 to force re-racing.
 *
 * Correctness: every plan is roundtripped (bwd(fwd(x)) == N*x) before it is
 * reported — a plan that banks but cannot round-trip is a bug, not a result.
 *
 * Build: python build.py --src benches/zen4_il_race.c --vfft --compile
 * Run  : VFFT_WISDOM_DIR=<…>/generated/wisdom/Zen4 zen4_il_race.exe [N] [cells]
 *        cells = comma list of oop/scr, ip/nat, oop/nat (default: all three) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft.h"

static double now_s(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double roundtrip(vfft_plan p, int N, int inplace)
{
    double *z0 = malloc(2u * N * sizeof *z0), *a = malloc(2u * N * sizeof *a),
           *b  = malloc(2u * N * sizeof *b);
    double maxe = 0, maxm = 0;
    int i;
    srand(42 + N);
    for (i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
    memcpy(a, z0, 2u * N * sizeof *a);
    if (inplace) {
        vfft_execute(p, VFFT_FORWARD,  a, NULL, a, NULL);
        vfft_execute(p, VFFT_BACKWARD, a, NULL, a, NULL);
        memcpy(b, a, 2u * N * sizeof *b);
    } else {
        vfft_execute(p, VFFT_FORWARD,  a, NULL, b, NULL);
        vfft_execute(p, VFFT_BACKWARD, b, NULL, a, NULL);
        memcpy(b, a, 2u * N * sizeof *b);
    }
    for (i = 0; i < 2 * N; i++) {
        double e = fabs(b[i] / N - z0[i]), m = fabs(z0[i]);
        if (e > maxe) maxe = e;
        if (m > maxm) maxm = m;
    }
    free(z0); free(a); free(b);
    return maxm > 0 ? maxe / maxm : maxe;
}

static int race_cell(int N, int inplace, int order, const char *label)
{
    vfft_config_t cfg;
    vfft_plan p;
    double t0, dt, err;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform    = VFFT_C2C;
    cfg.placement    = inplace ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.layout       = VFFT_LAYOUT_INTERLEAVED;
    cfg.order        = order;
    cfg.dims         = 1;
    cfg.n[0]         = N;
    cfg.howmany      = 1;
    cfg.rigor        = VFFT_PATIENT;
    cfg.wisdom_write = 1;                       /* persist: the guard */
    cfg.recalibrate  = getenv("VFFT_RECALIBRATE") != NULL;
    printf("── %-8s N=%d  create(PATIENT) …\n", label, N);
    fflush(stdout);
    t0 = now_s();
    p  = vfft_create(&cfg);
    dt = now_s() - t0;
    if (!p) {
        printf("   %-8s N=%d  REFUSED (see stderr)  [%.1fs]\n", label, N, dt);
        return 1;
    }
    err = roundtrip(p, N, inplace);
    printf("   %-8s N=%d  plan ok  create=%.1fs  roundtrip=%.2e %s\n",
           label, N, dt, err, err < 1e-10 ? "OK" : "*** BAD ***");
    vfft_destroy(p);
    return err < 1e-10 ? 0 : 1;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    const char *cells = (argc > 2) ? argv[2] : "oop/scr,ip/nat,oop/nat";
    const char *wd = getenv("VFFT_WISDOM_DIR");
    int fails = 0;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("zen4_il_race  vfft %s  isa=%s  wisdom_dir=%s\n",
           vfft_version(), vfft_isa(), wd ? wd : "(UNSET — nothing will persist)");
    if (strstr(cells, "oop/scr")) fails += race_cell(N, 0, VFFT_ORDER_SCRAMBLED, "oop/scr");
    if (strstr(cells, "ip/nat"))  fails += race_cell(N, 1, VFFT_ORDER_NATURAL,   "ip/nat");
    if (strstr(cells, "oop/nat")) fails += race_cell(N, 0, VFFT_ORDER_NATURAL,   "oop/nat");
    /* ip/scr = measurement_arms.md B2, the in-place IL engine-attach race
     * (CONVERT incumbent vs native il2p): banks wisdom2_scr lay=il mode=ilp|conv */
    if (strstr(cells, "ip/scr"))  fails += race_cell(N, 1, VFFT_ORDER_SCRAMBLED, "ip/scr");
    printf("done: %d cell(s) failed\n", fails);
    return fails ? 1 : 0;
}
