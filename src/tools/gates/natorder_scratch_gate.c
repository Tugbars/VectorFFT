/* natorder_scratch_gate.c — a plan's per-worker scratch must be sized and
 * indexed by the SAME number, and that number must be the plan's own.
 *
 * THE LAW (vfft.c, _vfft_pool_arm's contract): every engine clamps its worker
 * count by its OWN plan-time snapshot (h->nthreads). The pool is GROW-ONLY,
 * so "the pool is bigger at execute than it was at create" is not an edge case
 * — it is the designed behaviour, and any engine that reads the LIVE pool at
 * execute while having sized a buffer at create will eventually index past it.
 *
 * THE BUG THIS GATE WAS WRITTEN FOR. The natural-order reorder sized its
 * per-worker cycle scratch at create from the live pool (`_thread_pool_nworkers+1`
 * slots) and, at execute, read T from the live pool again and sliced
 * `tmp + slot*2*K` per worker. Create a natural-order plan while the pool is 1,
 * let anything grow the pool, execute the first plan: T workers slice a
 * 1-slot buffer. Both the 1D (`nat_tmp`) and 2D (`nat2d_tmp`) scratches had it.
 *
 * WHAT IS ASSERTED, and why it is honest about its limits. Output after the
 * pool grew must be BITWISE identical to the output computed before it grew
 * (memcmp-exact, the house MT==ST rule), and the process must survive. A heap
 * overrun is not GUARANTEED to be visible without a sanitizer: it may corrupt
 * memory nobody reads. On this host, before the fix, the output WAS bitwise
 * equal and the process then died in a later free/create — at a different
 * point on each run (exit 127, then 139). That nondeterminism is the signature
 * of heap corruption, and it is why the gate keeps the [trace] markers: they
 * tell you WHERE it died, which "exit 139" alone does not. Under ASan the
 * overrun itself is reported at the write. What the gate guarantees after the
 * fix is the invariant the fix establishes: the plan's execute never uses more
 * scratch slots than the plan allocated, because both come from h->nthreads.
 *
 * THE POOL GROWS THROUGH A PLAN (no process-wide thread setting). Each cell
 * creates its one-thread plan and executes it, then creates the same cell at
 * nthreads = T_REQ and executes that: the pool is now T_REQ (vfft_pool_threads,
 * the diagnostics header), and the one-thread plan runs again on the grown
 * pool. The first cell's one-thread plan is created while the pool is 1; the
 * second cell's while it is already grown: the same law from both sides.
 * A host with fewer than T_REQ logical CPUs refuses the threaded create; the
 * gate fails at setup then, printing the library's reason.
 *
 * THE CELLS reach the scratch path deterministically only from a SEEDED store:
 *   1D  n=1024 q=32 ord=nat place=ip banks mode=pcyc (cycle reorder — the
 *       path that USES the per-worker slots; pairs mode never touches tmp).
 *       N*K = 32768 >= 8192, so the MT branch engages.
 *   2D  256x256 order=NATURAL: the dim1 row tape exists (not FREE), and the
 *       whole-row reorder runs as N1 rows x K=N2 lanes through the same worker.
 * Registered ("bare", True) in run_gates.py so it receives a seeded scratch dir.
 *
 * Public API and the diagnostics header only. No timings.
 * Exit 0 = PASS, 1 = FAIL, 2 = setup.
 *
 * Build: python build.py --src benches/natorder_scratch_gate.c --vfft --compile
 * Run  : natorder_scratch_gate.exe <seeded-scratch-wisdom-dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vfft.h"
#include "vfft_diagnostics.h" /* vfft_pool_threads */

#define T_REQ 4

static int g_fail = 0;

static void fill(double *re, double *im, size_t n, unsigned seed)
{
    size_t i;
    srand(seed);
    for (i = 0; i < n; i++)
    {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static void verdict(const char *what, int ok)
{
    printf("  %-4s  %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) g_fail++;
}

/* Build a natural-order in-place split c2c plan of the given shape at one
 * thread and execute it forward once; create the same cell at `pool_grown`
 * threads and execute it (the pool grows); execute the one-thread plan again
 * on identical input. All three outputs must be bitwise equal. The [trace]
 * lines exist to localize a crash: a corrupted heap surfaces at a later
 * free/create, not at the overrunning write. */
static void run_cell(const char *name, int rank, int n0, int n1, size_t K,
                     const char *W, int pool_grown)
{
    vfft_config_t c;
    vfft_plan p_small, p_big;
    size_t tot = (size_t)n0 * (rank == 2 ? (size_t)n1 : K);
    double *re0, *im0, *reA, *imA, *reB, *imB, *reC, *imC;

    printf("  [trace] %s: creating at nthreads=1 (pool %d)\n", name, vfft_pool_threads());
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = VFFT_PLACEMENT_IN;
    c.layout    = VFFT_LAYOUT_SPLIT;
    c.order     = VFFT_ORDER_NATURAL;
    c.rank = rank; c.n[0] = n0; if (rank == 2) c.n[1] = n1;
    c.howmany = (rank == 2) ? 1 : K;
    c.nthreads = 1;               /* the plan's snapshot is 1, whatever the pool is */
    c.wisdom_dir = W;

    p_small = vfft_create(&c);
    if (!p_small) { printf("  setup: %s create at nthreads=1 failed: %s\n", name, vfft_error()); g_fail++; return; }

    re0 = (double *)malloc(tot * sizeof(double)); im0 = (double *)malloc(tot * sizeof(double));
    reA = (double *)malloc(tot * sizeof(double)); imA = (double *)malloc(tot * sizeof(double));
    reB = (double *)malloc(tot * sizeof(double)); imB = (double *)malloc(tot * sizeof(double));
    reC = (double *)malloc(tot * sizeof(double)); imC = (double *)malloc(tot * sizeof(double));
    if (!re0 || !im0 || !reA || !imA || !reB || !imB || !reC || !imC)
    { printf("  setup: alloc\n"); g_fail++; vfft_destroy(p_small); return; }
    fill(re0, im0, tot, 515u + (unsigned)n0 + (unsigned)K);

    /* reference: the one-thread plan (serial by construction) */
    memcpy(reA, re0, tot * 8); memcpy(imA, im0, tot * 8);
    vfft_execute_split(p_small, VFFT_FORWARD, reA, imA, reA, imA);
    printf("  [trace] reference execute at nthreads=1 done\n");

    /* ── grow the pool: the same cell at pool_grown threads, created and executed ── */
    printf("  [trace] creating the threaded plan (nthreads=%d)\n", pool_grown);
    c.nthreads = pool_grown;
    p_big = vfft_create(&c);
    if (p_big)
    {
        memcpy(reC, re0, tot * 8); memcpy(imC, im0, tot * 8);
        vfft_execute_split(p_big, VFFT_FORWARD, reC, imC, reC, imC);
        verdict("control: the threaded plan's output bitwise == the one-thread reference",
                memcmp(reA, reC, tot * 8) == 0 && memcmp(imA, imC, tot * 8) == 0);
    }
    else { printf("  setup: %s create at nthreads=%d failed: %s\n", name, pool_grown, vfft_error()); g_fail++; }
    printf("  %s: pool %d after the threaded plan; re-executing the nthreads=1 plan\n", name, vfft_pool_threads());

    /* the SAME one-thread plan, now with a bigger live pool than it was sized for */
    memcpy(reB, re0, tot * 8); memcpy(imB, im0, tot * 8);
    vfft_execute_split(p_small, VFFT_FORWARD, reB, imB, reB, imB);
    printf("  [trace] execute-after-growth returned\n");
    verdict("plan created at nthreads=1, executed on the grown pool: output bitwise == reference",
            memcmp(reA, reB, tot * 8) == 0 && memcmp(imA, imB, tot * 8) == 0);

    if (p_big)
    {
        vfft_destroy(p_big);
        printf("  [trace] destroy(p_big) returned\n");
    }
    printf("  [trace] destroying p_small (the plan sized at one thread)\n");
    vfft_destroy(p_small);
    printf("  [trace] destroy(p_small) returned; freeing buffers\n");
    free(re0); free(im0); free(reA); free(imA); free(reB); free(imB); free(reC); free(imC);
    printf("  [trace] buffers freed\n");
}

int main(int argc, char **argv)
{
    const char *W = argc > 1 ? argv[1] : ".";

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("natorder_scratch_gate: wisdom dir %s; the pool grows to nthreads=%d through a plan\n", W, T_REQ);

    run_cell("1D c2c ip split NATURAL n=1024 K=32 (pcyc)", 1, 1024, 0, 32, W, T_REQ);
    run_cell("2D c2c ip split NATURAL 256x256",          2,  256, 256, 1, W, T_REQ);

    if (vfft_pool_threads() < 2)
        printf("  note: the pool never grew (%d): the grown-pool half of the law was not exercised\n", vfft_pool_threads());
    if (g_fail) { printf("=== *** FAIL *** === (%d)\n", g_fail); return 1; }
    printf("=== ALL PASS === (scratch sized and indexed by the plan's own snapshot)\n");
    return 0;
}
