/* wisdom2_real_gate.h — the wave-2 flip gate for the r2c/c2r ROUTE family.
 *
 * Acceptance is plan-equivalence, ZERO timing: every check below is a store
 * operation, so the gate is deterministic and safe on a noisy machine.
 *
 * What it proves:
 *   1 CODEC IDENTITY   bank -> save -> reopen -> read back the same route.
 *   2 WILDCARD LAW     a migrated ord=* place=* row serves; a concrete
 *                      verdict at the same cell BEATS it (exact-beats-
 *                      wildcard), and the migrated row survives underneath.
 *   3 ENGINE ISOLATION a zr2c-owned cell is neither read as a route verdict
 *                      nor overwritten by one.
 *   4 SIBLING ISOLATION a dir=bwd / role=comp record in the same shard is
 *                      never matched by the route lookup — the hazard that
 *                      hand-rolled field-by-field scanners carry.
 *   5 REFUSALS         junk cells and cross-transform routes are refused at
 *                      the codec, not silently accepted.
 *
 * 🔴 Point at a SCRATCH dir. The gate BANKS and SAVES.
 */
#ifndef VFFT_WISDOM2_REAL_GATE_H
#define VFFT_WISDOM2_REAL_GATE_H

#include <stdio.h>
#include <string.h>
#include "wisdom2.h"
#include "wisdom2_real_reader.h"

#define VW2RG_CHECK(cond, ...)                                   \
    do {                                                         \
        if (!(cond)) {                                           \
            printf("  *** FAIL *** ");                           \
            printf(__VA_ARGS__);                                 \
            printf("\n");                                        \
            fails++;                                             \
        }                                                        \
    } while (0)

/* Bank one route record, reporting a refusal as a failure. */
static int vw2__rg_bank(vw2_store_t *st, int t, int N, size_t K,
                        int ord, int pl, int route,
                        double win, double lose,
                        const char *src, const char *from)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_real_rec_from_route(&rec, t, N, K, ord, pl, route,
                                win, lose, src, from, &why)) {
        printf("  *** FAIL *** bank refused (%s) t=%d N=%d K=%zu\n",
               why ? why : "?", t, N, K);
        return -1;
    }
    if (vw2_bank(st, &rec) != VW2_OK) {
        vw2_rec_free(&rec);
        printf("  *** FAIL *** vw2_bank rejected t=%d N=%d K=%zu\n", t, N, K);
        return -1;
    }
    return 0;
}

static int vfft_wisdom2_real_gate_run(const char *dir)
{
    vw2_store_t st;
    int fails = 0;

    printf("\n=== wisdom2 REAL-ROUTE gate (wave 2) — dir=%s ===\n", dir);

    /* ---- 1. codec identity: bank, save, reopen, read back --------------- */
    if (vw2_open(&st, dir, 1) != VW2_OK) {
        printf("  *** FAIL *** vw2_open(%s) failed\n", dir);
        return -1;
    }
    vw2__rg_bank(&st, VW2_T_R2C, 512, 4, VW2_ORD_NAT, VW2_PL_OOP,
                 VW2_RROUTE_STRIDE, 812.5, 901.0, "race", NULL);
    vw2__rg_bank(&st, VW2_T_C2R, 512, 4, VW2_ORD_NAT, VW2_PL_OOP,
                 VW2_RROUTE_NATURAL, 774.25, 830.0, "race", NULL);
    if (vw2_save(&st) != VW2_OK) {
        printf("  *** FAIL *** vw2_save failed\n");
        fails++;
    }
    vw2_close(&st);

    if (vw2_open(&st, dir, 1) != VW2_OK) {
        printf("  *** FAIL *** reopen failed\n");
        return -1;
    }
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_R2C, 512, 4, VW2_PL_OOP)
                    == VW2_RROUTE_STRIDE,
                "r2c 512x4 did not read back as stride");
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2R, 512, 4, VW2_PL_OOP)
                    == VW2_RROUTE_NATURAL,
                "c2r 512x4 did not read back as natural");
    /* placement is a KEY axis: the ip cell is a different, empty cell */
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_R2C, 512, 4, VW2_PL_IP)
                    == VW2_RROUTE_NONE,
                "r2c 512x4 ip must miss — placement is a key axis");
    printf("  [1] codec identity + placement axis\n");

    /* ---- 2. wildcard law ------------------------------------------------ */
    /* a migration row with no ord/place columns, exactly as c2r_path.txt
     * rows arrive (wildcards are legal only with a from=). */
    vw2__rg_bank(&st, VW2_T_C2R, 256, 8, VW2_ORD_ANY, VW2_PL_ANY,
                 VW2_RROUTE_SPLIT, 0.0, 0.0, "migrated", "c2r_path.txt");
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2R, 256, 8, VW2_PL_OOP)
                    == VW2_RROUTE_SPLIT,
                "migrated wildcard row must serve a concrete request");
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2R, 256, 8, VW2_PL_IP)
                    == VW2_RROUTE_SPLIT,
                "wildcard must serve BOTH placements");
    /* now a real race lands at one concrete placement — it must win there,
     * while the wildcard still serves the placement it has not raced. */
    vw2__rg_bank(&st, VW2_T_C2R, 256, 8, VW2_ORD_NAT, VW2_PL_OOP,
                 VW2_RROUTE_NATURAL, 401.0, 455.5, "race", NULL);
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2R, 256, 8, VW2_PL_OOP)
                    == VW2_RROUTE_NATURAL,
                "concrete verdict must BEAT the migrated wildcard");
    VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2R, 256, 8, VW2_PL_IP)
                    == VW2_RROUTE_SPLIT,
                "wildcard must survive under the concrete row");
    printf("  [2] wildcard law (exact beats wildcard, wildcard survives)\n");

    /* ---- 3. engine isolation: a zr2c cell is not ours ------------------- */
    {
        vw2_rec_t z;
        const char *why = NULL;
        memset(&z, 0, sizeof z);
        z.key.t = VW2_T_R2C; z.key.rank = 1; z.key.n[0] = 1024;
        z.key.q = 1; z.key.ord = VW2_ORD_NAT; z.key.pl = VW2_PL_OOP;
        if (vw2_rec_set(&z, 1, "eng", "zr2c") != VW2_OK ||
            vw2_rec_set(&z, 1, "zr_kv", "5") != VW2_OK ||
            vw2_bank(&st, &z) != VW2_OK) {
            printf("  *** FAIL *** could not stage a zr2c cell\n");
            fails++;
        }
        (void)why;
        VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_R2C, 1024, 1, VW2_PL_OOP)
                        == VW2_RROUTE_NONE,
                    "a zr2c cell must NOT read as a route verdict");
        VW2RG_CHECK(vw2_real_cell_taken(&st, VW2_T_R2C, 1024, 1, VW2_PL_OOP) == 1,
                    "a zr2c cell must report as taken");
        /* the banker must decline rather than clobber it */
        VW2RG_CHECK(vw2_real_route_bank(&st, VW2_T_R2C, 1024, 1, VW2_PL_OOP,
                                        VW2_RROUTE_RFFT, 100.0, 200.0) == 0,
                    "bank into a zr2c cell must decline cleanly");
        {
            vw2_key_t k;
            const vw2_rec_t *r;
            memset(&k, 0, sizeof k);
            k.t = VW2_T_R2C; k.rank = 1; k.n[0] = 1024;
            k.q = 1; k.ord = VW2_ORD_NAT; k.pl = VW2_PL_OOP;
            r = vw2_lookup(&st, &k);
            VW2RG_CHECK(r && vw2_rec_get(r, "eng") &&
                            !strcmp(vw2_rec_get(r, "eng"), "zr2c"),
                        "the zr2c cell was CLOBBERED by the route banker");
        }
        printf("  [3] engine isolation (zr2c cell intact, not misread)\n");
    }

    /* ---- 4. sibling isolation: dir / role must never be matched --------- */
    {
        vw2_rec_t d;
        memset(&d, 0, sizeof d);
        d.key.t = VW2_T_R2C; d.key.rank = 1; d.key.n[0] = 2048;
        d.key.q = 2; d.key.ord = VW2_ORD_NAT; d.key.pl = VW2_PL_OOP;
        d.key.dir = VW2_DIR_BWD;
        if (vw2_rec_set(&d, 1, "eng", "route") != VW2_OK ||
            vw2_rec_set(&d, 1, "route", "stride") != VW2_OK ||
            vw2_bank(&st, &d) != VW2_OK) {
            printf("  *** FAIL *** could not stage a dir=bwd sibling\n");
            fails++;
        }
        VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_R2C, 2048, 2, VW2_PL_OOP)
                        == VW2_RROUTE_NONE,
                    "a dir=bwd sibling must NOT be matched by a dir-absent request");
    }
    {
        vw2_rec_t c;
        memset(&c, 0, sizeof c);
        c.key.t = VW2_T_C2R; c.key.rank = 1; c.key.n[0] = 2048;
        c.key.q = 2; c.key.ord = VW2_ORD_NAT; c.key.pl = VW2_PL_OOP;
        c.key.role = VW2_ROLE_COMP;
        if (vw2_rec_set(&c, 1, "eng", "route") != VW2_OK ||
            vw2_rec_set(&c, 1, "route", "split") != VW2_OK ||
            vw2_bank(&st, &c) != VW2_OK) {
            printf("  *** FAIL *** could not stage a role=comp sibling\n");
            fails++;
        }
        VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2R, 2048, 2, VW2_PL_OOP)
                        == VW2_RROUTE_NONE,
                    "a role=comp sibling must NOT be matched by a problem request");
        printf("  [4] sibling isolation (dir=bwd, role=comp both invisible)\n");
    }

    /* ---- 5. refusals ---------------------------------------------------- */
    {
        vw2_rec_t r;
        const char *why = NULL;
        VW2RG_CHECK(vw2_real_rec_from_route(&r, VW2_T_R2C, 0, 4, VW2_ORD_NAT,
                                            VW2_PL_OOP, VW2_RROUTE_RFFT,
                                            0, 0, "race", NULL, &why) != 0,
                    "N=0 junk cell must be refused");
        why = NULL;
        /* a c2r route on an r2c tag is a category error, not a near-miss */
        VW2RG_CHECK(vw2_real_rec_from_route(&r, VW2_T_R2C, 512, 4, VW2_ORD_NAT,
                                            VW2_PL_OOP, VW2_RROUTE_SPLIT,
                                            0, 0, "race", NULL, &why) != 0,
                    "c2r route on an r2c tag must be refused");
        VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_C2C, 512, 4, VW2_PL_OOP)
                        == VW2_RROUTE_NONE,
                    "a non-real transform tag must never resolve a route");
        printf("  [5] refusals (junk cell, cross-transform route, wrong tag)\n");
    }

    /* the record set must survive one more save/load cycle unchanged */
    if (vw2_save(&st) != VW2_OK) {
        printf("  *** FAIL *** second vw2_save failed\n");
        fails++;
    }
    {
        int n_before = st.nrec;
        vw2_close(&st);
        if (vw2_open(&st, dir, 1) != VW2_OK) {
            printf("  *** FAIL *** second reopen failed\n");
            return -1;
        }
        VW2RG_CHECK(st.nrec == n_before,
                    "record count changed across save/load (%d -> %d)",
                    n_before, st.nrec);
        VW2RG_CHECK(vw2_real_route_lookup(&st, VW2_T_R2C, 512, 4, VW2_PL_OOP)
                        == VW2_RROUTE_STRIDE,
                    "verdict lost across the second roundtrip");
        printf("  [6] save/load roundtrip stable (%d records)\n", st.nrec);
    }
    vw2_close(&st);

    printf("\n  === %s (%d fail) ===\n\n", fails ? "*** FAIL ***" : "ALL PASS", fails);
    return fails;
}

#undef VW2RG_CHECK
#endif /* VFFT_WISDOM2_REAL_GATE_H */
