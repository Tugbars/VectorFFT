/* pool_preserve_gate.c — a plan create must NEVER shrink the process thread pool.
 *
 * THE LAW (vfft.c, _vfft_pool_arm): a plan may GROW the pool and never shrinks
 * it. Shrinking stays available to the CALLER through the public
 * vfft_set_num_threads(); a plan never needs an empty pool to run serially,
 * because every engine clamps its worker count by its own plan-time snapshot.
 *
 * WHY A GATE AND NOT A COMMENT. The law has been broken once already (2D real IL
 * create tore the pool from 8 to 1 for the whole process — that is why
 * _vfft_pool_arm exists) and, at the time this gate was written, was broken
 * again in a second place: the OOP order=NATURAL K=1 race in oop/c2c_oop_create.h
 * called the public shrinking setter with the child's nthreads=1, while its five
 * sibling copies of the same race body used the grow-only helper. A correctness
 * test cannot see this — the plan is right, the output is right, and every OTHER
 * plan in the process silently loses its workers. Only an assertion on the pool
 * count itself can fail here.
 *
 * PLATFORM-AGNOSTIC BY DESIGN. Nothing here assumes a core count. The gate asks
 * for a modest pool (T_REQ), takes vfft_get_num_threads()'s READ-BACK as the
 * truth for this host, and asserts INVARIANCE of that number across each
 * create. Any host that grants >= 2 workers exercises the property.
 *
 * WHY THE CELL IS WHAT IT IS. The OOP natural race runs only when
 *   order == NATURAL && N >= 2048 && layout == INTERLEAVED && K == 1
 *   && (no banked natural-order verdict OR cfg.recalibrate)
 *   && a K=1 cascade candidate REPLAYS from the store (kind-4 verdict).
 * So the store must be SEEDED (the runner passes a seeded scratch copy: this gate
 * is registered ("bare", True) in run_gates.py) and recalibrate=1 forces the
 * race regardless of what @natoop holds. n=16384 q=1 place=oop eng=zturn is a
 * banked cell in the shipped store. Deterministic: the pool-shrinking call sits
 * BEFORE the timing loop, so which arm wins is irrelevant to this assertion.
 *
 * Public API only. No timings, no clock. Exit 0 = PASS, 1 = FAIL, 2 = setup.
 *
 * Build: python build.py --src benches/pool_preserve_gate.c --vfft --compile
 * Run  : pool_preserve_gate.exe <seeded-scratch-wisdom-dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vfft.h"

#define T_REQ 4   /* modest: any dev box grants it; the READ-BACK is the truth */

static int g_fail = 0;

static void check(const char *what, int before, int after)
{
    if (after == before)
        printf("  ok    %-44s pool %d -> %d\n", what, before, after);
    else
    {
        printf("  FAIL  %-44s pool %d -> %d   *** A PLAN SHRANK THE POOL ***\n",
               what, before, after);
        g_fail++;
    }
}

int main(int argc, char **argv)
{
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    vfft_config_t c;
    vfft_plan p;
    int T;

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("pool_preserve_gate: wisdom dir %s%s\n", wisdir, W ? "" : "  (load failed - cells may miss)");

    vfft_set_num_threads(T_REQ);
    T = vfft_get_num_threads();
    printf("  requested %d workers, host granted %d (this number must not move)\n", T_REQ, T);
    if (T < 2)
    {
        printf("  SKIP: host granted < 2 workers; the property cannot be exercised here\n");
        return 0;
    }

    /* ── 1. OOP, order=NATURAL, K=1, interleaved, N above the cascade floor,
     *       recalibrate=1 -> the natord-cascade race runs. The child asks for 1
     *       thread: "the house spelling of 'this child is serial'". ── */
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = VFFT_OUTOFPLACE;
    c.layout    = VFFT_LAYOUT_INTERLEAVED;
    c.order     = VFFT_ORDER_NATURAL;
    c.rigor     = VFFT_MEASURE;
    c.dims = 1; c.n[0] = 16384; c.howmany = 1;
    c.nthreads = 1;
    c.recalibrate = 1;            /* force the race even if @natoop is banked */
    c.wisdom = W; c.wisdom_write = 0;
    p = vfft_create(&c);
    if (!p) { printf("  setup: OOP natural K=1 create failed\n"); return 2; }
    check("OOP c2c NATURAL K=1 il N=16384 nthreads=1 create", T, vfft_get_num_threads());
    vfft_destroy(p);
    check("... and its destroy", T, vfft_get_num_threads());

    /* ── 2. The original incident: a 2D real IL plan with nthreads=1. ──
     * Re-arm first, so a failure here is THIS section's and not a cascade
     * from section 1 having already emptied the pool. */
    vfft_set_num_threads(T_REQ);
    T = vfft_get_num_threads();
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C;
    c.placement = VFFT_OUTOFPLACE;
    c.layout    = VFFT_LAYOUT_INTERLEAVED;
    c.rigor     = VFFT_MEASURE;
    c.dims = 2; c.n[0] = 256; c.n[1] = 256; c.howmany = 1;
    c.nthreads = 1;
    c.wisdom = W; c.wisdom_write = 0;
    p = vfft_create(&c);
    if (!p) { printf("  setup: 2D real IL create failed\n"); return 2; }
    check("2D r2c IL 256x256 nthreads=1 create", T, vfft_get_num_threads());
    {
        /* execute once: the incident tore the pool down on EVERY execute too */
        size_t RN = 256u * 256u, CN = 256u * (256u / 2 + 1);
        double *x = (double *)calloc(RN, sizeof(double));
        double *z = (double *)calloc(2 * CN, sizeof(double));
        if (x && z)
        {
            vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
            check("... and its execute", T, vfft_get_num_threads());
        }
        free(x); free(z);
    }
    vfft_destroy(p);

    /* ── 3. Control: a plan that asks for MORE than the pool may grow it, and the
     *       grown pool must persist (grow-only means grow is allowed). ── */
    vfft_set_num_threads(T_REQ);
    T = vfft_get_num_threads();
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = VFFT_INPLACE;
    c.layout    = VFFT_LAYOUT_INTERLEAVED;
    c.rigor     = VFFT_MEASURE;
    c.dims = 1; c.n[0] = 1024; c.howmany = 8;
    c.nthreads = T;
    c.wisdom = W; c.wisdom_write = 0;
    p = vfft_create(&c);
    if (p)
    {
        int after = vfft_get_num_threads();
        if (after >= T) printf("  ok    %-44s pool %d -> %d\n", "control: plan asking for T (may grow, never shrink)", T, after);
        else { printf("  FAIL  %-44s pool %d -> %d\n", "control: plan asking for T", T, after); g_fail++; }
        vfft_destroy(p);
    }

    if (W) vfft_wisdom_free(W);
    if (g_fail) { printf("=== *** FAIL *** === (%d violation%s)\n", g_fail, g_fail == 1 ? "" : "s"); return 1; }
    printf("=== ALL PASS === (pool never shrank across create/execute/destroy)\n");
    return 0;
}
