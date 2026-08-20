/* sp_ccol_decode_gate.c — B5: the CCOL decode gate (THIN DRIVER).
 *
 * Closes the loop "calibrator banked it -> the STORE holds it -> create
 * serves it": for every kind-3 CCOL record in <wisdir>'s wisdom2 store
 * (enumerated via vw2_scan, each cell resolved through the PRODUCTION
 * twin vw2_oop_lookup_k1 — so the expectation IS what create will serve,
 * exact-beats-wildcard included), drive the REAL front door (vfft_create,
 * layout=SPLIT, placement=OOP, order=NATURAL), then (a) assert the served
 * route is CCOL, (b) assert the built plan matches the banked record via
 * the PRODUCTION comparator (vfft_sp_ccol_line_served,
 * dp_planner_split_oop.h — gates hold no logic), and (c) execute forward
 * and check against the scalar reference to 1e-9. Exit 0 = ALL PASS.
 * (Re-hosted on wisdom2 at the wave-3 close — the legacy oop_wisdom.txt
 * is FROZEN and no longer what the front door serves.)
 *
 * Usage: sp_ccol_decode_gate.exe <wisdir> [N...]
 *   cells default to every kind-3 CCOL record found in the store.
 * Build: python build.py --src benches/sp_ccol_decode_gate.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The front door, compiled in, so struct vfft_plan_s is a complete type
 * (the cil_ab.c pattern). */
#include "vfft.c"
#include "dp_planner_split_oop.h"

static int gate_cell(const vfft_oop_wisdom_entry_t *ke)
{
    const int N = ke->N;
    printf("[%d] banked: route CCOL pair %dx%d chain %d vars %d ns %.1f\n",
           N, ke->R1, ke->R2, ke->cc_chain, ke->cc_vars, ke->ns);

    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor     = VFFT_MEASURE;      /* wisdom HIT: no calibration runs */
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = 1;
    cfg.order     = VFFT_ORDER_NATURAL;
    cfg.layout    = VFFT_LAYOUT_SPLIT;

    vfft_plan p = vfft_create(&cfg);
    if (!p) { printf("[%d] FAIL: create returned NULL\n", N); return 1; }
    struct vfft_plan_s *h = (struct vfft_plan_s *)p;

    if (h->k1_sp_route != VFFT_K1_SP_CCOL) {
        printf("[%d] FAIL: served sp_route %d, banked CCOL\n",
               N, h->k1_sp_route);
        vfft_destroy(p);
        return 1;
    }
    if (!vfft_sp_ccol_line_served(ke, h->k1sp)) {
        printf("[%d] FAIL: built plan does not match the banked line\n", N);
        vfft_destroy(p);
        return 1;
    }

    /* execute + correctness vs the scalar reference */
    double *sre = _sp_ad(N), *sim = _sp_ad(N);
    double *dre = _sp_ad(N), *dim = _sp_ad(N);
    double *Rr = _sp_ad(N), *Ri = _sp_ad(N);
    if (!sre || !sim || !dre || !dim || !Rr || !Ri) {
        printf("[%d] FAIL: OOM\n", N);
        vfft_destroy(p);
        return 1;
    }
    srand(1234 + N);
    for (int n = 0; n < N; n++) {
        sre[n] = (double)rand() / RAND_MAX - 0.5;
        sim[n] = (double)rand() / RAND_MAX - 0.5;
    }
    if (_sp_reference(sre, sim, Rr, Ri, N) != 0) {
        printf("[%d] FAIL: reference self-check\n", N);
        vfft_destroy(p);
        return 1;
    }
    vfft_execute(p, VFFT_FORWARD, sre, sim, dre, dim);
    double e = 0, m = 0;
    for (int n = 0; n < N; n++) {
        double d1 = fabs(dre[n] - Rr[n]), d2 = fabs(dim[n] - Ri[n]);
        if (d1 > e) e = d1;
        if (d2 > e) e = d2;
        double mm = fabs(Rr[n]) > fabs(Ri[n]) ? fabs(Rr[n]) : fabs(Ri[n]);
        if (mm > m) m = mm;
    }
    int ok = (m > 0 && e / m < 1e-9);
    printf("[%d] %s: served chain matches, relerr %.2e\n",
           N, ok ? "PASS" : "FAIL", m > 0 ? e / m : e);
    vfft_proto_aligned_free(sre); vfft_proto_aligned_free(sim);
    vfft_proto_aligned_free(dre); vfft_proto_aligned_free(dim);
    vfft_proto_aligned_free(Rr);  vfft_proto_aligned_free(Ri);
    vfft_destroy(p);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: sp_ccol_decode_gate.exe <wisdir> [N...]\n");
        return 2;
    }
    char env[700];
    snprintf(env, sizeof env, "VFFT_WISDOM_DIR=%s", argv[1]);
    _putenv(env); /* before the first create — the bundle loads from here */

    /* enumerate the LIVE store; resolve each CCOL cell through the
     * production twin so the expectation is exactly what create serves */
    vw2_store_t st;
    vw2_open(&st, argv[1], 0);
    int fails = 0, cells = 0;
    int done[128], ndone = 0;
    int cursor = 0;
    const vw2_rec_t *r;
    while ((r = vw2_scan(&st, &cursor)) != NULL) {
        const char *spr = vw2_rec_get(r, "sp_route");
        vfft_oop_wisdom_entry_t ke;
        int N, d, dup = 0;
        if (r->key.t != VW2_T_C2C || r->key.rank != 1) continue;
        if (!spr || strcmp(spr, "ccol")) continue;
        N = r->key.n[0];
        for (d = 0; d < ndone; d++) if (done[d] == N) dup = 1;
        if (dup) continue;
        if (argc > 2) {
            int want = 0;
            for (int a = 2; a < argc; a++)
                if (atoi(argv[a]) == N) want = 1;
            if (!want) continue;
        }
        if (!vw2_oop_lookup_k1(&st, N, &ke)) continue; /* seed/unservable */
        if (ke.k1_sp_route != VFFT_K1_SP_CCOL) continue; /* served row != ccol */
        if (ndone < 128) done[ndone++] = N;
        cells++;
        fails += gate_cell(&ke);
    }
    vw2_close(&st);
    if (cells == 0) {
        printf("DECODE GATE VACUOUS: no CCOL cells in the store — FAIL\n");
        return 1;
    }
    printf("%s: %d CCOL cell(s), %d failure(s)\n",
           fails ? "DECODE GATE FAIL" : "DECODE GATE ALL PASS", cells, fails);
    return fails ? 1 : 0;
}
