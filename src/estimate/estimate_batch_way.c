/* estimate_batch_way.c -- the way of a contiguous 2D or 3D real batch on T threads, timed: the evidence behind the
 * estimate's rule (est__real_nd_batch_whole, src/core/il/estimate/real_estimate_nd_il.h).
 *
 * Per cell (shape, K, direction, placement), the estimate's own batch built by est_build_real_ndb three ways, every
 * other choice the same:
 *   w0  all threads on each transform in turn (the queue's loop, pq=0)
 *   w1  whole transforms per thread (the queue, pq=1, T one-thread copies)
 *   w2  the 3D groups arm (pq=2, K copies at G = T / K threads), where vfft_policy_pq_groups_ok admits it
 * and the way the estimate's rule picks (rule 1 = w1). Nothing is measured at create, nothing is banked.
 * Timing, the real batch studies' convention: R2C out of place a block of reps of >= 10 ms after one untimed run; in
 * place and C2R one execute a sample, the input re-laid untimed, 3 samples a round (median); R rounds, the arms' order
 * rotated each round; per arm the median; w1/w0 and w2/w0 as the median of the per-round ratios [the 20th..80th
 * percentile]. In the library's measurement scope (CPU 0, the guard), the process raised.
 *
 *   estimate_batch_way <T> <2d|3d|all|val> [rounds=15]
 *     all: the fit's grid (13 planes 16 x 16 - 1024 x 1024, 11 volumes 8^3 - 96^3, K 2 - 32); val: the validation grid
 *     (9 planes, 7 volumes, K 2 - 28, none of the fit's). A batch whose buffers pass 300 MB is skipped.
 * The rule's loss is the time of its pick over the faster of w0 / w1. On a quiet machine only.
 * Build: python gauntlet/build.py --compile --src src/estimate/estimate_batch_way.c   (no --vfft) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.c"

static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static est_machine_t g_m;

static vfft_plan build(const int *n, int rank, size_t K, int T, int c2r, int ip, int way, int *groups_ok)
{
    vfft_config_t pc;
    vfft_cfg_t c;
    char why[640];
    struct vfft_plan_s *h = NULL;
    memset(&pc, 0, sizeof pc);
    pc.transform = c2r ? VFFT_C2R : VFFT_R2C; pc.rank = rank; pc.n[0] = n[0]; pc.n[1] = n[1]; pc.n[2] = rank == 3 ? n[2] : 0;
    pc.howmany = K; pc.nthreads = T; pc.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT; pc.flags = VFFT_FLAG_ESTIMATE;
    if (!_vfft_api_translate(&pc, &c)) return NULL;
    _vfft_create_lock();
    if (rank == 2)
    {
        est_plan2dr_t *p = (est_plan2dr_t *)malloc(sizeof *p), *q = (est_plan2dr_t *)malloc(sizeof *q);
        *groups_ok = 0;
        if (way < 2 && p && q && !est_plan_2d_real(&g_m, n[0], n[1], c2r, ip, T, p, why, sizeof why) &&
            !est_plan_2d_real(&g_m, n[0], n[1], c2r, ip, 1, q, why, sizeof why))
            h = est_build_real_ndb(p, NULL, q, NULL, NULL, way, &c, NULL, why, sizeof why);
        free(p); free(q);
    }
    else
    {
        est_plan3dr_t *p = (est_plan3dr_t *)malloc(sizeof *p), *q = (est_plan3dr_t *)malloc(sizeof *q),
                      *g = (est_plan3dr_t *)malloc(sizeof *g);
        const int groups = vfft_policy_pq_groups_ok(&c, K, T);
        *groups_ok = groups;
        if ((way < 2 || groups) && p && q && g && !est_plan_3d_real(&g_m, n[0], n[1], n[2], c2r, ip, T, p, why, sizeof why) &&
            !est_plan_3d_real(&g_m, n[0], n[1], n[2], c2r, ip, 1, q, why, sizeof why) &&
            (!groups || !est_plan_3d_real(&g_m, n[0], n[1], n[2], c2r, ip, T / (int)K, g, why, sizeof why)))
            h = est_build_real_ndb(NULL, p, NULL, q, groups ? g : NULL, way, &c, NULL, why, sizeof why);
        free(p); free(q); free(g);
    }
    _vfft_create_unlock();
    if (h && h->nthreads > 1)
        for (int d = VFFT_FORWARD; d <= VFFT_BACKWARD; d++)
            if (h->k1_go[d]) { h->k1_go_mt[d] = h->k1_go[d]; h->k1_go[d] = _vfft_k1_go_mt; }
    if (!h && way < 2) fprintf(stderr, "build refused (way %d): %s\n", way, why);
    return (vfft_plan)h;
}

static void cell(const int *n, int rank, size_t K, int T, int c2r, int ip, int R)
{
    const int last = n[rank - 1];
    const size_t A = rank == 2 ? (size_t)n[0] : (size_t)n[0] * n[1], hp1 = (size_t)last / 2 + 1;
    const size_t bytes = 16 * A * hp1;   /* a transform's spectrum, the rule's measure */
    vfft_plan p[3] = { 0, 0, 0 };
    int gok = 0, ok[3];
    for (int w = 0; w < 3; w++) { int g; p[w] = build(n, rank, K, T, c2r, ip, w, &g); if (w == 0) gok = g; ok[w] = p[w] != NULL; }
    if (!ok[0] || !ok[1]) { for (int w = 0; w < 3; w++) if (p[w]) vfft_destroy(p[w]); printf("# refused %dx%d%s K=%zu\n", n[0], n[1], rank == 3 ? "x.." : "", K); return; }
    vfft_info_t inf;
    vfft_info(p[0], &inf);
    const size_t nin = inf.in_doubles + 64, nout = inf.out_doubles + 64, nx = ip ? (nin > nout ? nin : nout) : nin;
    double *x = vfft_malloc(nx * 8), *y = vfft_malloc(nout * 8), *s = vfft_malloc(nx * 8);
    if (!x || !y || !s) { printf("# out of memory at K=%zu\n", K); goto done; }
    for (size_t i = 0; i < nx; i++) s[i] = (double)((i * 2654435761u) & 1023) / 1024.0 - 0.5;
    memcpy(x, s, nx * 8);
    const vfft_dir_t dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
    const int blocks = !c2r && !ip;
    int reps[3] = { 1, 1, 1 };
    double t[3][64], rat[3][64];
    for (int w = 0; w < 3; w++)
        if (ok[w])
        {   /* warm >= 5 ms; the R2C out-of-place block's reps */
            double t0 = vfft_now_ns(), e;
            int k = 0;
            do { if (!blocks) memcpy(x, s, nx * 8); vfft_execute(p[w], dir, x, ip ? x : y); k++; } while ((e = vfft_now_ns() - t0) < 5e6 && k < 100000);
            if (blocks) { reps[w] = (int)(1e7 / (e / k)); if (reps[w] < 1) reps[w] = 1; if (reps[w] > 100000) reps[w] = 100000; }
        }
    for (int r = 0; r < R; r++)
        for (int j = 0; j < 3; j++)
        {
            const int w = (r + j) % 3;
            if (!ok[w]) continue;
            if (blocks)
            {
                vfft_execute(p[w], dir, x, y);
                const double t0 = vfft_now_ns();
                for (int k = 0; k < reps[w]; k++) vfft_execute(p[w], dir, x, y);
                t[w][r] = (vfft_now_ns() - t0) / reps[w];
            }
            else
            {
                double v[3];
                for (int k = 0; k < 3; k++)
                {
                    memcpy(x, s, nx * 8);
                    const double t0 = vfft_now_ns();
                    vfft_execute(p[w], dir, x, ip ? x : y);
                    v[k] = vfft_now_ns() - t0;
                }
                qsort(v, 3, sizeof v[0], cmpd);
                t[w][r] = v[1];
            }
        }
    double med[3] = { 0, 0, 0 }, rm[3] = { 0, 0, 0 }, rl[3] = { 0, 0, 0 }, rh[3] = { 0, 0, 0 };
    for (int w = 0; w < 3; w++)
        if (ok[w])
        {
            double u[64];
            for (int r = 0; r < R; r++) { u[r] = t[w][r]; rat[w][r] = t[w][r] / t[0][r]; }
            qsort(u, R, sizeof u[0], cmpd);
            qsort(rat[w], R, sizeof rat[w][0], cmpd);
            med[w] = u[R / 2]; rm[w] = rat[w][R / 2]; rl[w] = rat[w][R / 5]; rh[w] = rat[w][R - 1 - R / 5];
        }
    {
        char name[40];
        if (rank == 3) snprintf(name, sizeof name, "%dx%dx%d", n[0], n[1], n[2]); else snprintf(name, sizeof name, "%dx%d", n[0], n[1]);
        const int rule = est__real_nd_batch_whole(&g_m, bytes, K, T);
        printf("%s %-12s K=%-3zu %-3s T=%d bytes %9zu  w0 %11.0f  w1 %11.0f  w2 %11s  | w1/w0 %.3f [%.3f..%.3f]", c2r ? "c2r" : "r2c",
               name, K, ip ? "ip" : "oop", T, bytes, med[0], med[1], "", rm[1], rl[1], rh[1]);
        if (ok[2]) printf("  w2/w0 %.3f [%.3f..%.3f] (%.0f)", rm[2], rl[2], rh[2], med[2]);
        printf("  | rule %d\n", rule);
        fflush(stdout);
    }
done:
    vfft_free(x); vfft_free(y); vfft_free(s);
    for (int w = 0; w < 3; w++) if (p[w]) vfft_destroy(p[w]);
}

int main(int argc, char **argv)
{
    static const int S2f[][2] = { { 16, 16 }, { 32, 32 }, { 64, 64 }, { 128, 128 }, { 256, 256 }, { 384, 384 }, { 512, 512 },
                                 { 1024, 1024 }, { 64, 1024 }, { 1024, 64 }, { 360, 360 }, { 100, 500 }, { 200, 63 } };
    static const int S3f[][3] = { { 8, 8, 8 }, { 16, 16, 16 }, { 24, 24, 24 }, { 32, 32, 32 }, { 48, 48, 48 }, { 64, 64, 64 },
                                 { 96, 96, 96 }, { 9, 186, 46 }, { 32, 64, 15 }, { 20, 30, 40 }, { 128, 128, 32 } };
    static const size_t KS0[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32 };
    /* the validation grid ("val"): shapes and K none of the fit's */
    static const int V2[][2] = { { 48, 48 }, { 96, 96 }, { 200, 200 }, { 300, 500 }, { 128, 1024 }, { 2048, 256 }, { 50, 50 },
                                 { 80, 90 }, { 640, 480 } };
    static const int V3[][3] = { { 12, 12, 12 }, { 40, 40, 40 }, { 64, 32, 16 }, { 20, 20, 100 }, { 80, 80, 80 }, { 16, 128, 64 },
                                 { 10, 50, 7 } };
    static const size_t KV[] = { 2, 3, 5, 6, 8, 11, 14, 20, 28 };
    const int T = argc > 1 ? atoi(argv[1]) : 8;
    const char *which = argc > 2 ? argv[2] : "all";
    const int val = !strcmp(which, "val");
    const size_t *KS = val ? KV : KS0;
    const size_t nK = val ? sizeof KV / sizeof KV[0] : sizeof KS0 / sizeof KS0[0];
    const int (*S2)[2] = val ? V2 : S2f;
    const int (*S3)[3] = val ? V3 : S3f;
    const size_t n2 = val ? sizeof V2 / sizeof V2[0] : sizeof S2f / sizeof S2f[0], n3 = val ? sizeof V3 / sizeof V3[0] : sizeof S3f / sizeof S3f[0];
    const int R = argc > 3 ? atoi(argv[3]) : 15;
    memset(&g_m, 0, sizeof g_m);
    g_m.lanes = VFFT_IL_VW; g_m.l1d = vfft_cpu_l1d_bytes(); g_m.l2 = vfft_cpu_l2_bytes(); g_m.l3 = vfft_cpu_l3_bytes();
    g_m.cores = vfft_topology()->ncores; g_m.l1d_ways = (int)vfft_cpu_l1d_ways();
    {
        vfft_measure_config_t mc;
        char d[320];
        memset(&mc, 0, sizeof mc);
        mc.pin = VFFT_MEASURE_PIN_CORE; mc.pin_core = 0; mc.guard = VFFT_MEASURE_GUARD_DEFAULT;
        mc.priority = VFFT_MEASURE_PRIORITY_LEAVE;
        vfft_measure_configure(&mc);
        const int f = vfft_measure_begin();
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        printf("# the way of a contiguous 2D/3D real batch, T=%d, %d rounds; machine L1d %ld L2 %ld L3 %ld; %s%s\n", T, R,
               g_m.l1d, g_m.l2, g_m.l3, vfft_measure_describe(d, sizeof d), f ? " (not clean)" : "");
    }
    for (int c2r = 0; c2r < 2; c2r++)
        for (int ip = 0; ip < 2; ip++)
        {
            if (strcmp(which, "3d"))
                for (size_t i = 0; i < n2; i++)
                    for (size_t k = 0; k < nK; k++)
                    {
                        const size_t b = 16 * (size_t)S2[i][0] * (S2[i][1] / 2 + 1);
                        if ((double)KS[k] * b * (ip ? 1 : 2) > 300e6) continue;   /* the batch's buffers past 300 MB */
                        const int n[3] = { S2[i][0], S2[i][1], 0 };
                        cell(n, 2, KS[k], T, c2r, ip, R);
                    }
            if (strcmp(which, "2d"))
                for (size_t i = 0; i < n3; i++)
                    for (size_t k = 0; k < nK; k++)
                    {
                        const size_t b = 16 * (size_t)S3[i][0] * S3[i][1] * (S3[i][2] / 2 + 1);
                        if ((double)KS[k] * b * (ip ? 1 : 2) > 300e6) continue;
                        cell(S3[i], 3, KS[k], T, c2r, ip, R);
                    }
        }
    return 0;
}
