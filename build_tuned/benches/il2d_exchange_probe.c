/* il2d_exchange_probe.c — INC-2 of docs/design/il2d_real_mt.md.
 * MEASUREMENT ONLY, no library change. Prices the two constants the
 * column-MT plan is costed against:
 *
 *  (a) THE ROW->COLUMN EXCHANGE. In pass-parallel 2D MT the row pass
 *      leaves worker t's rows in t's PRIVATE L2, and the column pass
 *      then reads columns — so every worker touches every other
 *      worker's dirty lines. Arms, same run, same buffer:
 *        W  : T workers each WRITE their own contiguous row range.
 *        R1 : T workers each READ BACK THEIR OWN rows   (no exchange)
 *        R2 : T workers each read a COLUMN STRIP        (full exchange)
 *        R0 : one core writes all, one core reads all   (control)
 *      R2 - R1 is the exchange cost. If it is small relative to a
 *      cell's single-thread transform time, INC-4 (matching the row
 *      partition to the column partition to drive the exchange to
 *      zero) is NOT WORTH BUILDING — that is the decision this makes.
 *
 *  (b) THE BARRIER CONSTANT: dispatch T no-op workers + wait_all.
 *      Every candidate column partition pays 1-2 of these per
 *      transform, and the engage decision is priced against it.
 *
 * The pool here is this TU's own (threads.h is header-only) — the
 * point is to measure the machine, not the library.
 * Build: python build.py --src benches/il2d_exchange_probe.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "env.h"      /* stride_pin_thread — the caller MUST sit on core 0 */
#include "threads.h"

static double now_ns(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)f.QuadPart;
}

typedef struct {
    double *p;          /* the complex plane, 2*N1*hp1 doubles */
    int N1, hp1;
    int lo, hi;         /* row range (W, R1) or column range (R2)  */
    int mode;           /* 0 = write rows, 1 = read rows, 2 = read cols */
    volatile double sink;
} arg_t;

static void work_tramp(void *v)
{
    arg_t *a = (arg_t *)v;
    const int hp1 = a->hp1;
    double s = 0.0;
    int i, k;
    if (a->mode == 0) {
        for (i = a->lo; i < a->hi; i++) {
            double *row = a->p + (size_t)i * 2 * hp1;
            for (k = 0; k < 2 * hp1; k++)
                row[k] = (double)(i + k);
        }
    } else if (a->mode == 1) {
        for (i = a->lo; i < a->hi; i++) {
            const double *row = a->p + (size_t)i * 2 * hp1;
            for (k = 0; k < 2 * hp1; k++)
                s += row[k];
        }
    } else {
        for (i = 0; i < a->N1; i++) {
            const double *row = a->p + (size_t)i * 2 * hp1;
            for (k = 2 * a->lo; k < 2 * a->hi; k++)
                s += row[k];
        }
    }
    a->sink = s;
}

/* run one phase across T threads (caller = thread 0), return ns */
static double phase(arg_t *args, int T, double *p, int N1, int hp1,
                    int mode)
{
    int t, nd = 0;
    double t0;
    for (t = 0; t < T; t++) {
        args[t].p = p; args[t].N1 = N1; args[t].hp1 = hp1;
        args[t].mode = mode;
        if (mode == 2) {                    /* column strips */
            args[t].lo = (int)((long)hp1 * t / T);
            args[t].hi = (int)((long)hp1 * (t + 1) / T);
        } else {                            /* row ranges    */
            args[t].lo = (int)((long)N1 * t / T);
            args[t].hi = (int)((long)N1 * (t + 1) / T);
        }
    }
    t0 = now_ns();
    for (t = 1; t < T && t <= _stride_pool_size; t++)
        _stride_pool_dispatch(&_stride_workers[nd++], work_tramp, &args[t]);
    work_tramp(&args[0]);
    if (nd) _stride_pool_wait_all();
    return now_ns() - t0;
}

static void noop_tramp(void *v) { (void)v; }

int main(void)
{
    static const int CELLS[][2] = {
        { 256, 256 }, { 512, 512 }, { 1024, 1024 },
        { 4096, 64 }, { 8192, 64 }, { 4096, 16 },
    };
    const int NC = (int)(sizeof CELLS / sizeof CELLS[0]);
    const int T = 8;
    int ci, r;
    arg_t args[16];
    setvbuf(stdout, NULL, _IONBF, 0);
    stride_set_num_threads(T);
    stride_pin_thread(0);
    printf("=== INC-2 exchange + barrier probe (T=%d, pool=%d, min-of-15) ===\n",
           T, _stride_pool_size + 1);

    /* (b) the barrier constant: dispatch T-1 no-ops + wait_all */
    {
        double best = 1e300;
        for (r = 0; r < 200; r++) {
            int t, nd = 0;
            double t0 = now_ns(), d;
            for (t = 1; t < T && t <= _stride_pool_size; t++)
                _stride_pool_dispatch(&_stride_workers[nd++], noop_tramp,
                                      NULL);
            if (nd) _stride_pool_wait_all();
            d = now_ns() - t0;
            if (d < best) best = d;
        }
        printf("barrier constant (dispatch %d + wait_all): %.0f ns\n\n",
               T - 1, best);
    }

    printf("  cell        plane KB   W(ns)    R1 own    R2 cols    R0 1core"
           "   |  exchange = R2-R1   R2 GB/s\n");
    for (ci = 0; ci < NC; ci++) {
        const int N1 = CELLS[ci][0], N2 = CELLS[ci][1];
        const int hp1 = N2 / 2 + 1;
        const size_t bytes = (size_t)N1 * hp1 * 16;
        double *p = (double *)_aligned_malloc(bytes, 64);
        double w = 1e300, r1 = 1e300, r2 = 1e300, r0 = 1e300;
        if (!p) { printf("OOM\n"); return 2; }
        memset(p, 0, bytes);
        for (r = 0; r < 15; r++) {
            double d;
            /* W then R1: same partition, data stays in each core's L2 */
            d = phase(args, T, p, N1, hp1, 0); if (d < w) w = d;
            d = phase(args, T, p, N1, hp1, 1); if (d < r1) r1 = d;
            /* W then R2: orthogonal partition => full exchange */
            phase(args, T, p, N1, hp1, 0);
            d = phase(args, T, p, N1, hp1, 2); if (d < r2) r2 = d;
            /* control: single core writes then reads */
            {
                arg_t a1;
                a1.p = p; a1.N1 = N1; a1.hp1 = hp1; a1.lo = 0;
                a1.hi = N1; a1.mode = 0;
                work_tramp(&a1);
                a1.mode = 1;
                {
                    double t0 = now_ns();
                    work_tramp(&a1);
                    d = now_ns() - t0;
                }
                if (d < r0) r0 = d;
            }
        }
        printf("  %4dx%-4d  %8zu  %7.0f  %8.0f  %9.0f  %10.0f   |"
               "  %10.0f  %9.1f\n",
               N1, N2, bytes / 1024, w, r1, r2, r0, r2 - r1,
               (double)bytes / r2);
        _aligned_free(p);
    }
    stride_set_num_threads(1);
    return 0;
}
