/* oop_wisdom_roundtrip_gate.c — does every field of a wisdom entry survive
 * being written and read back?
 *
 * WHY THIS EXISTS. On 2026-08-02 a tile-width field was added to the kind-4
 * record. TWO places emit that record — the planner's wisdom emitter and the
 * calibration driver, which banks through the shipped writer directly. Only one
 * learned about the new field, so a real calibration run recorded a TILED
 * winner as UNTILED and **nothing complained**. The bug was invisible: the file
 * was well-formed, the reader accepted it, and the plan it produced was a valid
 * plan — just not the one that was measured.
 *
 * A point fix ("teach the other writer too") does not stop the next field from
 * repeating it. This gate does, by testing the PROPERTY instead:
 *
 *   fill every field -> write -> read back -> compare.
 *
 * A writer that forgets a field, or a reader that stops parsing one, fails here
 * regardless of which field it is or who forgot it.
 *
 * PLUS A SIZE CANARY. A round-trip test can only check the fields it knows
 * about, so it cannot notice a field added *after* it was written. The static
 * assertion below fails the build when the struct grows, which forces whoever
 * grew it to come here — the one thing a test cannot do for itself.
 *
 * Build: python build.py --src benches/oop_wisdom_roundtrip_gate.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oop_wisdom.h"

/* ── SIZE CANARY ────────────────────────────────────────────────────────────
 * If this fails, you added or removed a field in vfft_oop_wisdom_entry_t.
 * Before updating the number:
 *   1. add the field to fill_all() and compare_all() below,
 *   2. make sure EVERY writer emits it — there is more than one,
 *   3. make sure the reader parses it, and that its ABSENCE still means
 *      whatever the old format meant (back-compat is not optional here).
 * Only then change this constant. */
#define VFFT_OOP_ENTRY_SIZE_EXPECTED  ((size_t)216)

typedef char _entry_size_canary[
    (sizeof(vfft_oop_wisdom_entry_t) == VFFT_OOP_ENTRY_SIZE_EXPECTED) ? 1 : -1];

/* ── the property ───────────────────────────────────────────────────────── */
static void fill_all(vfft_oop_wisdom_entry_t *e, int kind)
{
    memset(e, 0, sizeof *e);
    e->N = 16384;
    e->K = 1;
    e->kind = kind;
    e->R1 = 128; e->R2 = 64;
    e->t1p_variant = 1;
    e->nf = 4;
    for (int i = 0; i < e->nf; i++) { e->factors[i] = 4 + i; e->variants[i] = i % 3; }
    e->k1_sp_route = 2; e->k1_il_route = 3;
    e->il_R1 = 32; e->il_R2 = 16;
    e->cc_chain = 2222222;
    e->zs_t2q = 1;
    e->zs_route = 1;
    e->zt_t2q = 1;
    e->zt_tw = 1024;                  /* the field that was silently dropped */
    e->zt_l1 = 49152;                 /* and its companion                   */
    e->ns = 38426.6;
}

typedef struct { const char *name; long got, want; } diff_t;

static int compare_all(const vfft_oop_wisdom_entry_t *w,
                       const vfft_oop_wisdom_entry_t *r,
                       int kind, diff_t *d, int maxd)
{
    int n = 0;
#define CHK(f) do { if (n < maxd && (long)(w->f) != (long)(r->f)) {           \
                        d[n].name = #f; d[n].got = (long)(r->f);              \
                        d[n].want = (long)(w->f); n++; } } while (0)
    CHK(N); CHK(K); CHK(kind);
    if (kind == VFFT_OOP_KIND_ZSPLIT) {
        /* kind-4 carries: zs_t2q, cc_chain, ns, [zs_route zt_t2q [zt_tw zt_l1]] */
        CHK(zs_t2q); CHK(cc_chain); CHK(zs_route); CHK(zt_t2q);
        CHK(zt_tw);  CHK(zt_l1);
    } else if (kind == VFFT_OOP_KIND_BAILEY2V) {
        CHK(k1_sp_route); CHK(R1); CHK(R2);
        CHK(k1_il_route); CHK(il_R1); CHK(il_R2);
    } else if (kind == VFFT_OOP_KIND_BAILEY2) {
        CHK(R1); CHK(R2); CHK(t1p_variant);
    } else if (kind == VFFT_OOP_KIND_MODEB) {
        CHK(nf);
        for (int i = 0; i < w->nf; i++) { CHK(factors[i]); CHK(variants[i]); }
    }
    if (n < maxd) {
        double dn = w->ns - r->ns;
        if (dn < 0) dn = -dn;
        if (dn > 0.05) { d[n].name = "ns"; d[n].got = (long)r->ns;
                         d[n].want = (long)w->ns; n++; }
    }
#undef CHK
    return n;
}

static const char *kind_name(int k)
{
    switch (k) {
        case VFFT_OOP_KIND_BAILEY2:  return "BAILEY2";
        case VFFT_OOP_KIND_MODEB:    return "MODEB";
        case VFFT_OOP_KIND_BAILEY2V: return "BAILEY2V (kind 3)";
        case VFFT_OOP_KIND_ZSPLIT:   return "ZSPLIT (kind 4)";
        default:                     return "?";
    }
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[1024];
    snprintf(path, sizeof path, "%s/_roundtrip_oop_wisdom.txt", dir);

    printf("\n=== oop_wisdom round-trip gate ===\n");
    printf("  sizeof(vfft_oop_wisdom_entry_t) = %zu bytes (canary expects %zu)\n\n",
           sizeof(vfft_oop_wisdom_entry_t), (size_t)VFFT_OOP_ENTRY_SIZE_EXPECTED);

    const int KINDS[] = { VFFT_OOP_KIND_ZSPLIT, VFFT_OOP_KIND_BAILEY2V,
                          VFFT_OOP_KIND_BAILEY2, VFFT_OOP_KIND_MODEB };
    int fails = 0;

    for (size_t ki = 0; ki < sizeof KINDS / sizeof KINDS[0]; ki++) {
        const int kind = KINDS[ki];
        vfft_oop_wisdom_entry_t w;
        fill_all(&w, kind);

        FILE *f = fopen(path, "w");
        if (!f) { printf("  cannot write %s\n", path); return 2; }
        vfft_oop_wisdom_write_entry(f, &w);
        fclose(f);

        /* what the writer actually produced — printed so a failure is readable */
        char line[512] = {0};
        f = fopen(path, "r");
        if (f) { if (!fgets(line, sizeof line, f)) line[0] = 0; fclose(f); }
        line[strcspn(line, "\r\n")] = 0;

        vfft_oop_wisdom_t wis;
        memset(&wis, 0, sizeof wis);
        int loaded = (vfft_oop_wisdom_load(&wis, path) == 0) && wis.count == 1;

        printf("  %-20s %s\n", kind_name(kind), line);
        if (!loaded) {
            printf("      *** FAIL — the reader did not accept what the writer "
                   "emitted (count=%d) ***\n", loaded ? wis.count : 0);
            fails++;
            continue;
        }

        diff_t d[24];
        int nd = compare_all(&w, &wis.e[0], kind, d, 24);
        if (nd == 0) {
            printf("      every field survived the round trip\n");
        } else {
            fails++;
            printf("      *** FAIL — %d field(s) did not survive ***\n", nd);
            for (int i = 0; i < nd; i++)
                printf("        %-12s wrote %ld, read back %ld\n",
                       d[i].name, d[i].want, d[i].got);
        }
    }
    remove(path);

    printf("\n  === %d kind(s) PASS, %d FAIL ===\n",
           (int)(sizeof KINDS / sizeof KINDS[0]) - fails, fails);
    if (!fails)
        printf("  A writer that forgets a field, or a reader that stops parsing\n"
               "  one, fails here — whichever field it is and whoever forgot it.\n");
    return fails ? 1 : 0;
}
