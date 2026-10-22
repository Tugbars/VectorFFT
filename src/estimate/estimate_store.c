/* estimate_store.c -- write the estimate method's plans as a wisdom store on disk (tracked, not wired into the library).
 *
 * For every N in [lo, hi]: est_plan_1d, then est_put_rows -- the rows a race would bank, through the
 * library's own writers (the same call the library's est_build_c2c makes into its private store,
 * il/estimate/c2c_estimate_il.h) --
 * into an EMPTY store directory, saved once at the end. A canonical tool then replays exactly these
 * plans: python gauntlet/gauntlet.py run --cells lo..hi --store <dir>  (never --merge).
 * One cell: 1D c2c, natural order, out of place, one thread (the gauntlet's default contract).
 *
 *   estimate_store <empty store dir> <lo> <hi>
 * Build: python gauntlet/build.py --compile --src src/estimate/estimate_store.c   (no --vfft) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long est_clock_reads;
static void est_clock_probe(void);
#define VFFT_CLOCK_PROBE est_clock_probe()
#include "vfft.c"   /* carries the library's estimate (il/estimate/estimate_il.h): est_plan_1d, est_put_rows */
static void est_clock_probe(void) { est_clock_reads++; }

int main(int argc, char **argv)
{
    if (argc < 4) { printf("usage: estimate_store <empty store dir> <lo> <hi>\n"); return 2; }
    const char *dir = argv[1];
    const int lo = atoi(argv[2]), hi = atoi(argv[3]);
    est_machine_t m;
    memset(&m, 0, sizeof m);
    m.lanes = VFFT_IL_VW;
    m.l1d = vfft_cpu_l1d_bytes();
    m.l2 = vfft_cpu_l2_bytes();
    m.l3 = vfft_cpu_l3_bytes();
    m.cores = vfft_topology()->ncores;
    printf("# machine: lanes %d, L1d %ld, L2 %ld, L3 %ld, cores %d\n", m.lanes, m.l1d, m.l2, m.l3, m.cores);

    vw2_store_t st;
    if (vw2_open(&st, dir, 1) != VW2_OK && st.nrec != 0) { printf("cannot open %s as an empty store\n", dir); return 1; }
    if (st.nrec) { printf("%s is not empty (%d rows): refusing to mix rows\n", dir, (int)st.nrec); return 1; }
    int ok = 0, refused = 0;
    for (int N = lo; N <= hi; N++)
    {
        est_request_t rq = { N, VFFT_C2C, VFFT_ORDER_NATURAL, VFFT_PLACEMENT_OUT, 1 };
        est_recipe_t rc;
        char why[768];
        if (est_plan_1d(&m, &rq, &rc, why, sizeof why)) { printf("N=%d refused (formula): %s\n", N, why); refused++; continue; }
        if (est_put_rows(&st, &rc, &rq, 1, why, sizeof why)) { printf("N=%d refused (rows): %s\n", N, why); refused++; continue; }
        ok++;
    }
    const int rows = (int)st.nrec;
    const int rc = vw2_save(&st);
    vw2_close(&st);
    printf("# %d lengths written (%d rows), %d refused; save %s; clock reads %ld (must be 0)\n", ok, rows, refused,
           rc == VW2_OK ? "ok" : "FAILED", est_clock_reads);
    return (rc == VW2_OK && est_clock_reads == 0) ? 0 : 1;
}
