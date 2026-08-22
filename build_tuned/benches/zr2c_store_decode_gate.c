/* zr2c_store_decode_gate.c — every shipped kind-5 row must decode through the
 * PRODUCTION lookup, and say the same thing the row's own text says.
 *
 * WHY THIS EXISTS. The route bit is the only verdict kind-5 wisdom owns, and
 * until 2026-08-22 nothing in the tree read it outside vfft.c and the
 * migrator's verify leg. The codec reassembles one packed kv from FOUR
 * independent per-slot records keyed {t, n, q=1, ord=nat, place}, so a row
 * that is mis-keyed, carries an unknown route token, or lands in the wrong
 * slot serves the WRONG route with no symptom — both routes are
 * correctness-gated, so the only consequence is speed, which no bench
 * measures through the front door either.
 *
 * WHAT IT PROVES, per shipped eng=zr2c row:
 *   1 DECODES        the production lookup returns a verdict for that cell
 *   2 AGREES         the decoded slot matches the row's own route= token
 *   3 SLOT IDENTITY  the row lands in the slot its (t, place) key implies,
 *                    not a neighbour's
 *
 * WHAT IT DELIBERATELY DOES NOT PROVE. That vfft_create actually BUILT that
 * route: there is no public accessor for a plan's route, and both routes
 * produce identical output by construction, so no black-box check can tell
 * them apart. That half needs plan introspection that does not exist yet —
 * recorded rather than faked.
 *
 * READ-ONLY. Opens the store, never banks, never saves. Safe to point at the
 * shipped generated/ directory, and ZERO timing, so it is valid on a noisy
 * machine.
 *
 * Build: python build.py --src benches/zr2c_store_decode_gate.c --compile
 * Run  : zr2c_store_decode_gate.exe [wisdom dir]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wisdom2.h"
#include "wisdom2_oop_reader.h"

static int g_fail = 0;
static int g_rows = 0;

static const char *slot_name(int slot)
{
    static const char *n[4] = { "r2c/oop", "r2c/ip", "c2r/oop", "c2r/ip" };
    return (slot >= 0 && slot < 4) ? n[slot] : "?";
}

int main(int argc, char **argv)
{
    const char *cand[3] = {
        (argc >= 2) ? argv[1] : NULL,
        "../src/dag-fft-compiler/generator/generated",
        "../../src/dag-fft-compiler/generator/generated",
    };
    const char *wdir = NULL;
    vw2_store_t st;
    int i;

    for (i = 0; i < 3 && !wdir; i++) {
        char pp[512];
        FILE *pf;
        if (!cand[i]) continue;
        snprintf(pp, sizeof pp, "%s/wisdom2_real.txt", cand[i]);
        pf = fopen(pp, "r");
        if (pf) { fclose(pf); wdir = cand[i]; }
    }
    printf("zr2c STORE-DECODE gate — every shipped kind-5 row, through the "
           "production lookup\n");
    if (!wdir) {
        printf("wisdom dir: NOT FOUND — refusing to gate against nothing\n");
        return 1;
    }
    if (vw2_open(&st, wdir, 0) != VW2_OK) {   /* 0 = READ-ONLY */
        printf("could not open %s\n", wdir);
        return 1;
    }
    printf("wisdom dir: %s (%d records, read-only)\n\n", wdir, st.nrec);

    for (i = 0; i < st.nrec; i++) {
        const vw2_rec_t *r = &st.rec[i];
        const char *eng = vw2_rec_get(r, "eng");
        const char *route = vw2_rec_get(r, "route");
        int is_c2r, is_ip, slot, want, kv = 0, got;

        if (!eng || strcmp(eng, "zr2c")) continue;
        if (r->key.rank != 1) continue;
        if (r->key.t != VW2_T_R2C && r->key.t != VW2_T_C2R) continue;
        g_rows++;

        is_c2r = (r->key.t == VW2_T_C2R);
        is_ip  = (r->key.pl == VW2_PL_IP);
        slot   = vfft_zr2c_kv_slot(is_c2r, is_ip);

        /* 2 — what the row's own text says */
        if (!route) {
            printf("  n=%-6d %-8s  *** FAIL *** row carries no route= token\n",
                   (int)r->key.n[0], slot_name(slot));
            g_fail = 1;
            continue;
        }
        if (!strcmp(route, "child_oop_il"))      want = 1;   /* encoded, not raw */
        else if (!strcmp(route, "child_nat_ip")) want = 2;
        else {
            printf("  n=%-6d %-8s  *** FAIL *** unknown route token \"%s\"\n",
                   (int)r->key.n[0], slot_name(slot), route);
            g_fail = 1;
            continue;
        }

        /* 1 + 3 — the production lookup, and the slot it lands in */
        if (!vw2_oop_lookup_zr2c(&st, (int)r->key.n[0], &kv)) {
            printf("  n=%-6d %-8s  *** FAIL *** row does not DECODE (lookup "
                   "returned no verdict)\n", (int)r->key.n[0], slot_name(slot));
            g_fail = 1;
            continue;
        }
        got = vfft_zr2c_kv_get(kv, slot);
        if (got != want) {
            printf("  n=%-6d %-8s  *** FAIL *** slot decodes %d, row says %d (%s)\n",
                   (int)r->key.n[0], slot_name(slot), got, want, route);
            g_fail = 1;
            continue;
        }
    }

    printf("  %d shipped eng=zr2c row(s) checked\n", g_rows);
    if (!g_rows) {
        printf("  *** FAIL *** no zr2c rows found — the gate proved nothing\n");
        g_fail = 1;
    }
    vw2_close(&st);
    printf("\n%s\n", g_fail ? "ZR2C STORE-DECODE GATE: FAILURE"
                            : "ZR2C STORE-DECODE GATE: ALL DECODE");
    return g_fail;
}
