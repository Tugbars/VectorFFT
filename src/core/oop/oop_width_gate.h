/* oop_width_gate.h — does a BANKED tcut width replay, and does a width
 * banked for a DIFFERENT cache refuse to replay?
 *
 * The width axis is only shippable if a wisdom record can carry it safely.
 * Two properties have to hold, and neither is checkable by staring at the
 * code:
 *
 *   B0 BACK-COMPAT. A record banked before this axis existed has no width
 *      field, and MUST replay as UNTILED — byte-for-byte today's driver.
 *      There is no sentinel to forget: absent parses as 0 and 0 means
 *      untiled.
 *   B1 CACHE FENCE. A width is a property of one machine's L1, not of the
 *      transform. Replaying a 48 KB-tuned width on a 32 KB cache overshoots
 *      by 50%, and overshoot loses the whole benefit at once rather than
 *      degrading. So a stamped-L1 mismatch must fall back to UNTILED and
 *      SAY SO — never "use it anyway".
 *
 * Also checked: an illegal width for the banked chain is refused (not
 * force-fit), and a legal one actually engages with the geometry that was
 * banked.
 *
 * Everything goes through the PUBLIC front door (vfft_create), and the
 * verdict is read from the create-time [tcut] line — an arm that silently
 * fell back would otherwise be indistinguishable from one that worked.
 *
 * MECHANISM: each case banks a synthetic kind-4 verdict into the wisdom2
 * store through the shipped constructor (vw2_oop_bank_entry — the one
 * definition of the format), then creates through the front door and reads
 * the engagement. The store file is byte-snapshotted before the first
 * injection and byte-restored after the last, so a scratch dir stays
 * reusable with its provenance intact.
 *
 * 🔴 Point wisdir at a SCRATCH copy. This gate REWRITES wisdom2_oop.txt
 *    while it runs (restored on exit, but a crash mid-gate leaves the last
 *    injected case in place).
 */
#ifndef VFFT_OOP_WIDTH_GATE_H
#define VFFT_OOP_WIDTH_GATE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vfft.h"
#include "../wisdom2/wisdom2_oop_reader.h"
#include "cpu_cache.h" /* vfft_cpu_l1d_bytes: the CASES table derives its
                        * native/foreign stamps from the live sizing value
                        * instead of hardcoding one host's caches */

/* ── stderr tap ────────────────────────────────────────────────────────── */
static char _wg_errpath[1024];
static long _wg_errpos = 0;
static int _wg_tap_open(const char *dir)
{
    snprintf(_wg_errpath, sizeof _wg_errpath, "build_tuned/benches/_wisdom_width_gate.log" /* cwd, 0.12: never inside a wisdom dir */ );
    if (!freopen(_wg_errpath, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    _wg_errpos = 0;
    return 1;
}
static const char *_wg_tap_read(void)
{
    static char buf[8192];
    buf[0] = 0;
    fflush(stderr);
    {
        FILE *f = fopen(_wg_errpath, "rb");
        if (!f) return buf;
        if (fseek(f, _wg_errpos, SEEK_SET) == 0) {
            size_t n = fread(buf, 1, sizeof buf - 1, f);
            buf[n] = 0;
            _wg_errpos += (long)n;
        }
        fclose(f);
    }
    return buf;
}

/* ── store snapshot/restore (bytes, not records — provenance survives) ─── */
static char *_wg_snap = NULL;
static long  _wg_snap_len = 0;
static int _wg_snapshot(const char *dir)
{
    char path[1024];
    FILE *f;
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", dir);
    f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    _wg_snap_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    _wg_snap = (char *)malloc((size_t)_wg_snap_len + 1);
    if (!_wg_snap || (long)fread(_wg_snap, 1, (size_t)_wg_snap_len, f) != _wg_snap_len) {
        fclose(f); free(_wg_snap); _wg_snap = NULL;
        return 0;
    }
    fclose(f);
    return 1;
}
static int _wg_restore(const char *dir)
{
    char path[1024];
    FILE *f;
    int ok;
    if (!_wg_snap) return 0;
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", dir);
    f = fopen(path, "wb");
    if (!f) return 0;
    ok = (long)fwrite(_wg_snap, 1, (size_t)_wg_snap_len, f) == _wg_snap_len;
    fclose(f);
    free(_wg_snap); _wg_snap = NULL;
    return ok;
}

/* ── inject one synthetic kind-4 verdict (shipped constructor, no fprintf) */
static int _wg_inject(const char *dir, int N, int cc_chain, double ns,
                      int zroute, int zt_t2q, int zt_tw, int zt_l1)
{
    vw2_store_t st;
    vfft_oop_wisdom_entry_t e;
    int rc;
    memset(&e, 0, sizeof e);
    e.N = N;
    e.K = 1;
    e.kind = VFFT_OOP_KIND_ZSPLIT;
    e.cc_chain = cc_chain;
    e.ns = ns;
    e.zs_route = zroute;
    e.zs_t2q = 0;
    e.zt_t2q = zroute ? zt_t2q : 0;
    e.zt_tw = zroute ? zt_tw : 0;
    e.zt_l1 = zroute ? zt_l1 : 0;
    vw2_open(&st, dir, 1);
    rc = vw2_oop_bank_entry(&st, &e);
    if (rc == VW2_OK) rc = vw2_save(&st);
    vw2_close(&st);
    return rc == VW2_OK;
}

/* ── create one plan and report what tiling it engaged ─────────────────── */
typedef struct { int ok, tiled, tcut, refused_or_absent; long w, NT; char note[160]; } _wg_res_t;

static _wg_res_t _wg_make(const char *wisdir, int N)
{
    _wg_res_t r; memset(&r, 0, sizeof r);
    /* the env axis must be OFF: we are testing the WISDOM path */
    putenv((char *)"VFFT_TCUT=");
    putenv((char *)"VFFT_TCUT_W=");
    putenv((char *)"VFFT_TCUT_VERBOSE=1");
    putenv((char *)"VFFT_FORCE_ZROUTE=zturn");

    {
        vfft_wisdom *W = vfft_wisdom_load(wisdir);
        vfft_config_t cfg;
        vfft_plan h;
        const char *log, *q;
        if (!W) { snprintf(r.note, sizeof r.note, "wisdom load failed"); return r; }
        (void)_wg_tap_read();

        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.order = VFFT_ORDER_SCRAMBLED; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1; cfg.wisdom = W;
        h = vfft_create(&cfg);
        log = _wg_tap_read();
        r.ok = h != NULL;

        q = strstr(log, "[tcut]");
        if (q) {
            char form[16];
            if (strstr(q, "tuned for L1d") || strstr(q, "ILLEGAL")) {
                r.refused_or_absent = 1;
                {
                    const char *nl = strchr(q, '\n');
                    size_t len = nl ? (size_t)(nl - q) : strlen(q);
                    if (len > sizeof r.note - 1) len = sizeof r.note - 1;
                    memcpy(r.note, q, len); r.note[len] = 0;
                }
            } else {
                sscanf(q, "[tcut] N=%*d nf=%*d tiled=%d tcut=%d tfuse=%*d tw=%15s "
                          "w=%ld NT=%ld", &r.tiled, &r.tcut, form, &r.w, &r.NT);
            }
        } else {
            r.refused_or_absent = 1;
            snprintf(r.note, sizeof r.note, "(no [tcut] line -> untiled)");
        }
        if (h) vfft_destroy(h);
        vfft_wisdom_free(W);
    }
    return r;
}

/* Run the gate. Returns the FAIL count; -1 = setup failed (bad dir / no
 * kind-4 record at N to anchor on / snapshot failed). */
static int vfft_oop_width_gate_run(const char *wisdir, int N)
{
    const int CC = 232223;              /* 4.8.4.4.4.8, the banked 16384 chain */
    int fails = 0, c;

    /* Cache stamps are DERIVED, not literal (2026-09-03). The old table
     * hardcoded 49152/32768, which baked "the host is the 14900KF" into the
     * EXPECTATIONS: on a 32 KB-L1 host with VFFT_L1D_DISCOVER=1 the fence
     * answered the other way and three rows failed while the library behaved
     * correctly — and the illegal-width row stopped testing legality at all
     * (the fence in front of it fired first). native = this build's live
     * sizing value; foreign = a real size guaranteed to differ. The gate
     * asserts BEHAVIOUR (native engages, foreign refused), never a machine. */
    const int L1_NATIVE  = (int)vfft_cpu_l1d_bytes();
    const int L1_FOREIGN = (L1_NATIVE == 49152) ? 32768 : 49152;
    const struct {
        const char *desc;               /* printed row label                 */
        int zroute, zt_t2q, zt_tw, zt_l1;
        const char *want; int want_tiled; long want_w;
    } CASES[] = {
        /* B0: pre-width verdict, zsplit route, no width fields -> untiled  */
        { "zsplit, no width fields", 0, 0, 0,    0,
          "UNTILED (legacy record)",           0, 0 },
        /* zturn route, still no width -> untiled                           */
        { "zturn, no width fields",  1, 0, 0,    0,
          "UNTILED (route, no width)",         0, 0 },
        /* B1a: width + MATCHING (native) L1 -> engages                     */
        { "zturn tw=1024 l1=native", 1, 0, 1024, L1_NATIVE,
          "TILED w=1024",                      1, 1024 },
        { "zturn tw=2048 l1=native", 1, 0, 2048, L1_NATIVE,
          "TILED w=2048",                      1, 2048 },
        /* B1b: width tuned for a DIFFERENT cache -> untiled + loud         */
        { "zturn tw=1024 l1=foreign", 1, 0, 1024, L1_FOREIGN,
          "UNTILED (L1 mismatch)",             0, 0 },
        /* illegal width for this chain (1536 does not divide SEC=4096) —
         * NATIVE L1 on purpose, so the LEGALITY check is the one that
         * fires rather than the fence that sits in front of it            */
        { "zturn tw=1536 l1=native", 1, 0, 1536, L1_NATIVE,
          "UNTILED (illegal width)",           0, 0 },
    };
    const int nc = (int)(sizeof CASES / sizeof CASES[0]);

    if (!_wg_tap_open(wisdir)) { printf("stderr tap failed\n"); return -1; }
    {
        /* anchor: a kind-4 verdict must exist at N (same precondition the
         * legacy line-surgery gate had — nothing to test otherwise) */
        vw2_store_t st;
        vfft_oop_wisdom_entry_t e;
        int hit;
        vw2_open(&st, wisdir, 0);
        hit = vw2_oop_lookup_zsplit(&st, N, &e);
        vw2_close(&st);
        if (!hit) {
            printf("no kind-4 (N=%d K=1) record in %s/wisdom2_oop.txt — nothing to test\n",
                   N, wisdir);
            return -1;
        }
    }
    if (!_wg_snapshot(wisdir)) {
        printf("could not snapshot %s/wisdom2_oop.txt\n", wisdir);
        return -1;
    }

    printf("\n=== banked tcut WIDTH replay gate  N=%d ===\n", N);
    printf("  (env axis disabled; every verdict comes from wisdom2_oop.txt)\n\n");
    printf("  %-28s %-26s %s\n", "banked verdict", "engaged", "verdict");
    printf("  --------------------------------------------------------------"
           "------------------\n");

    for (c = 0; c < nc; c++) {
        _wg_res_t r;
        int got_tiled, ok;
        char eng[40];
        if (!_wg_inject(wisdir, N, CC, 49306.2, CASES[c].zroute,
                        CASES[c].zt_t2q, CASES[c].zt_tw, CASES[c].zt_l1)) {
            printf("  %-28s %-26s *** FAIL *** (inject refused)\n",
                   CASES[c].desc, "-");
            fails++;
            continue;
        }
        r = _wg_make(wisdir, N);
        got_tiled = (r.tiled == 1 && !r.refused_or_absent);
        ok = r.ok
          && got_tiled == (CASES[c].want_tiled != 0)
          && (!CASES[c].want_tiled || r.w == CASES[c].want_w);
        if (!ok) fails++;
        if (got_tiled) snprintf(eng, sizeof eng, "tiled t=%d w=%ld NT=%ld",
                                r.tcut, r.w, r.NT);
        else           snprintf(eng, sizeof eng, "untiled");
        printf("  %-28s %-26s %s\n", CASES[c].desc, eng,
               ok ? "PASS" : "*** FAIL ***");
        if (!ok || r.note[0])
            printf("      expected %s%s%s\n", CASES[c].want,
                   r.note[0] ? "  |  " : "", r.note[0] ? r.note : "");
    }

    if (!_wg_restore(wisdir))
        printf("  ⚠ restore of wisdom2_oop.txt FAILED — scratch dir is dirty\n");

    printf("\n  === %d PASS, %d FAIL ===\n", nc - fails, fails);
    printf("  B0 back-compat: a pre-width record replays UNTILED, no sentinel needed.\n"
           "  B1 cache fence: a width tuned for another L1 is REFUSED, not reused.\n");
    printf("  (original wisdom2_oop.txt restored byte-for-byte)\n");
    return fails;
}

#endif /* VFFT_OOP_WIDTH_GATE_H */
