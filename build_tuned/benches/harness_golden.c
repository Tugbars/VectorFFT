/* harness_golden.c — the golden-artifact driver for the refactor harness.
 *
 * Produces two of the three artifacts docs/design/refactor_safety_harness.md
 * asks for, both clock-free and both diffed byte-for-byte across every
 * migration step:
 *
 *   REFUSAL MATRIX (section 2.10) — vfft_create's accept/refuse decision for
 *     every cell of the declared config space, legal and illegal alike.
 *     Emitted as NAME-KEYED sorted lines, never a positional bitmap: a bitmap
 *     shifts every row when one cell is inserted, so the diff stops being
 *     readable exactly when it matters. Both directions matter — over-refusal
 *     is caught by the legal twin, under-refusal by the illegal cell.
 *
 *   GOLDEN OUTPUT BITS (section 2.7) — a digest of every output plane, per
 *     direction, compared BITWISE. This is what catches the failures the plan
 *     fingerprint structurally cannot: a selector-to-pointer mis-wiring during
 *     a MERGE (the il_kv blocked variants differ at ~1e-16), a shared-plan arm
 *     swap, an inner-size transposition. Note a roundtrip check cannot see any
 *     of those — a roundtrip through a swapped fwd/bwd pair still round-trips.
 *
 * NOT YET COVERED, deliberately: r2c/c2r/trig golden bits and the naive
 * O(N^2) reference. Their plane-role contract is not stated plainly enough in
 * include/vfft.h to encode without guessing, and a WRONG golden artifact is
 * worse than a missing one — it bakes an incorrect expectation into the
 * baseline permanently and every later step then "passes". Adding them is the
 * next increment, and it matters: the wisdom store holds ZERO trig cells, so
 * the naive reference is the only protection that family will ever get.
 *
 * Public API only. No library internals, no timings, no clock.
 *
 * Build: python build.py --src benches/harness_golden.c --vfft --compile
 * Run  : harness_golden.exe [--out FILE]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

/* Built with -DVFFT_FINGERPRINT the harness can SELF-VALIDATE: it reads the
 * create-race counter and refuses to emit a golden digest for any cell whose
 * create raced. Without the flag it still works, it just cannot police itself.
 * Prefer the instrumented build for a baseline capture. */
#ifdef VFFT_FINGERPRINT
#include "vfft_fingerprint.h"
static long races_now(void)
{
    long c[VFFT__FP_NCOUNTERS];
    vfft__fp_counters(c);
    return c[5];
}
#else
static long races_now(void) { return -1; }   /* -1 = cannot tell */
#endif

/* FNV-1a over raw bytes: the CELL is the triage unit, so a digest is the right
 * instrument here (unlike the plan trace, where a hash would destroy triage). */
static unsigned long long digest(const double *p, size_t n)
{
    unsigned long long h = 1469598103934665603ULL;
    const unsigned char *b = (const unsigned char *)p;
    size_t i, bytes = n * sizeof(double);
    if (!p) return 0ULL;
    for (i = 0; i < bytes; i++) {
        h ^= (unsigned long long)b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void fill(double *p, size_t n, unsigned seed)
{
    size_t i;
    unsigned s = seed * 2654435761u + 1u;
    for (i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (double)(s >> 8) / (double)(1u << 24) - 0.5;
    }
}

/* ---------------------------------------------------------------- refusals */

typedef struct {
    const char *name;
    vfft_transform_t xf;
    vfft_placement_t place;
    int layout, order, dims, n0, n1;
    size_t K;
    int expect_ok;          /* 1 = must build, 0 = must be refused */
} cell_t;

static const cell_t CELLS[] = {
    /* ---- legal, one per axis value -------------------------------------- */
    {"c2c.split.oop.default.1d",   VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 1,  1},
    {"c2c.split.ip.default.1d",    VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 1,  1},
    {"c2c.split.ip.scrambled",     VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_SCRAMBLED, 1, 256, 0, 1,  1},
    {"c2c.split.ip.natural",       VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_NATURAL,   1, 256, 0, 1,  1},
    {"c2c.il.ip.default",          VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT,   1, 256, 0, 1,  1},
    {"c2c.il.oop.default",         VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT,   1, 256, 0, 1,  1},
    /* K=4, not K=8. A golden cell MUST be banked in the wisdom store, or create
     * misses, calls _calibrate_c2c, and RACES - and a race puts the clock inside
     * the baseline. c2c n=256 has q= 1,4,7,11,15,16,19,23,27,31,32,64,128,256
     * banked; there is no q=8. That single omission produced four different
     * digests from four runs, which read as "the transform is nondeterministic"
     * and cost a long hunt. The transform is bit-exact: repeating the SAME plan
     * in-process gave 0 of 2048 doubles differing. What varied was which plan
     * got built. Pick banked cells, and let the purity assert below catch it if
     * you get one wrong. */
    {"c2c.split.ip.K4",            VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 4,  1},
    {"c2c.split.ip.K32",           VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 32, 1},
    {"c2c.split.ip.N64",           VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 64,  0, 1,  1},
    {"c2c.split.ip.N1024",         VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 1024,0, 1,  1},

    /* ---- illegal: each names the rule it is testing ---------------------- */
    {"REFUSE.dct2.interleaved",    VFFT_DCT2, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT,   1, 256, 0, 1,  0},
    {"REFUSE.dct2.2d",             VFFT_DCT2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   2, 64,  64,1,  0},
    {"REFUSE.r2c.scrambled",       VFFT_R2C,  VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_SCRAMBLED, 1, 256, 0, 1,  0},
    {"REFUSE.c2r.natural.order",   VFFT_C2R,  VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_NATURAL,   1, 256, 0, 1,  0},
    {"REFUSE.dims5",               VFFT_C2C,  VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   5, 64,  64,1,  0},
    {"REFUSE.N0",                  VFFT_C2C,  VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 0,   0, 1,  0},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

static void cfg_of(const cell_t *c, vfft_config_t *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    cfg->transform = c->xf;
    cfg->placement = c->place;
    cfg->layout    = (vfft_layout_t)c->layout;
    cfg->order     = c->order;
    cfg->dims      = c->dims;
    cfg->n[0]      = c->n0;
    cfg->n[1]      = c->n1;
    cfg->howmany   = c->K;
    cfg->rigor     = VFFT_MEASURE;
    cfg->wisdom_write = 0;          /* serving mode: never write the store */
}

/* ------------------------------------------------------------ golden bits */

/* c2c only, where the plane contract is unambiguous:
 *   SPLIT       -> (sre, sim, dre, dim), four real planes of N*K
 *   INTERLEAVED -> (z, NULL, zout, NULL), one plane of 2*N*K
 *   in-place    -> destination aliases source, per the documented
 *                  (z, NULL, z, NULL) spelling. */
static int golden_c2c(const cell_t *c, FILE *out, long races_at_entry)
{
    vfft_config_t cfg;
    vfft_plan p;
    size_t n = (size_t)c->n0 * c->K;
    double *a, *b, *da, *db;
    int il = (c->layout == VFFT_LAYOUT_INTERLEAVED);
    int ip = (c->place == VFFT_INPLACE);
    size_t span = il ? 2 * n : n;

    long r1;

    if (c->xf != VFFT_C2C || !c->expect_ok) return 0;
    cfg_of(c, &cfg);
    p = vfft_create(&cfg);
    r1 = races_now();
    if (!p) { fprintf(out, "golden %-28s CREATE_FAILED\n", c->name); return 1; }

    /* PURITY ASSERT. A cell whose create raced is not a baseline: the clock
     * chose its plan, so its digest is a coin flip and diffing it measures
     * thermal noise. Emit the violation INSTEAD of the digest - a loud refusal
     * beats a plausible-looking number that changes on its own.
     *
     * The baseline `races_at_entry` is taken by the CALLER, before any create
     * for this cell. It cannot be taken here: the refusal check already created
     * this same config once, that first create absorbed the race, and banking is
     * always IN-MEMORY first (even at wisdom_write=0), so the create below is a
     * pure replay. Measuring across only the second create made this assert
     * inert - it passed a cold-dir negative control that should have failed it. */
    if (races_at_entry >= 0 && r1 > races_at_entry) {
        fprintf(out, "golden %-28s NOT_BANKED_RACED races=%ld"
                     "  (bank this cell or drop it)\n",
                c->name, r1 - races_at_entry);
        vfft_destroy(p);
        return 1;
    }

    a  = (double *)calloc(span, sizeof(double));
    b  = (double *)calloc(il ? 1 : span, sizeof(double));
    da = (double *)calloc(span, sizeof(double));
    db = (double *)calloc(il ? 1 : span, sizeof(double));
    if (!a || !b || !da || !db) { vfft_destroy(p); free(a); free(b); free(da); free(db); return 1; }

    fill(a, span, (unsigned)c->n0 + (unsigned)c->K);
    if (!il) fill(b, span, (unsigned)c->n0 + 7u);

    vfft_execute(p, VFFT_FORWARD, a, il ? NULL : b,
                 ip ? a : da, il ? NULL : (ip ? b : db));
    fprintf(out, "golden %-28s fwd=%016llx %016llx\n", c->name,
            digest(ip ? a : da, span), il ? 0ULL : digest(ip ? b : db, span));

    /* Backward must NOT alias on an out-of-place plan: feeding (da,db)->(da,db)
     * to an OOP handle is the documented refusal case, and the digest then just
     * re-reads the forward output and reports fwd==bwd. In-place is the one
     * shape where source and destination legitimately coincide. */
    vfft_execute(p, VFFT_BACKWARD, ip ? a : da, il ? NULL : (ip ? b : db),
                 a, il ? NULL : b);
    fprintf(out, "golden %-28s bwd=%016llx %016llx\n", c->name,
            digest(a, span), il ? 0ULL : digest(b, span));

    vfft_destroy(p);
    free(a); free(b); free(da); free(db);
    return 0;
}

int main(int argc, char **argv)
{
    FILE *out = stdout;
    int i, bad = 0, only = -1;
    if (argc > 2 && !strcmp(argv[1], "--out")) {
        out = fopen(argv[2], "w");
        if (!out) { printf("cannot open %s\n", argv[2]); return 2; }
    }

    /* ONE PROCESS PER CELL is the only diffable mode. Several things in the
     * library are process-lifetime, not per-plan - above all the K=1 pair-order
     * memo, keyed by N and consulted BEFORE the race. Run every cell in one
     * process and a later cell inherits whatever an earlier one memoized, so the
     * artifact becomes order- and history-dependent: reproducible within one
     * binary, DIFFERENT across builds, for no reason connected to the change
     * under test. Not hypothetical - it produced a false "step 4 changed output
     * bits" on exactly two cells and cost a revert scare. A baseline capture MUST
     * use --cell; the all-cells path below is triage only. */
    for (i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--cell")) only = atoi(argv[i + 1]);
    if (only >= 0 && only < NCELLS) {
        vfft_config_t cfg; vfft_plan p;
        /* Taken BEFORE the refusal create. That first create is what races, and
         * it banks in memory, so anything measured after it sees a replay. */
        long r_entry = races_now();
        cfg_of(&CELLS[only], &cfg);
        p = vfft_create(&cfg);
        fprintf(out, "refuse %-28s %s%s\n", CELLS[only].name,
                p ? "ACCEPT" : "REFUSE",
                ((p != NULL) == (CELLS[only].expect_ok != 0)) ? "" : "  <<< UNEXPECTED");
        if ((p != NULL) != (CELLS[only].expect_ok != 0)) bad++;
        if (p) vfft_destroy(p);
        bad += golden_c2c(&CELLS[only], out, r_entry);
        if (out != stdout) fclose(out);
        return bad ? 1 : 0;
    }

    fprintf(out, "# golden artifacts - refusal decisions and output-bit digests.\n");
    fprintf(out, "# No timings. Name-keyed and sorted, so a diff names the cell.\n");
    fprintf(out, "# Any digest change is a REVERT, on every step class, always.\n#\n");

    for (i = 0; i < NCELLS; i++) {
        vfft_config_t cfg;
        vfft_plan p;
        cfg_of(&CELLS[i], &cfg);
        p = vfft_create(&cfg);
        fprintf(out, "refuse %-28s %s%s\n", CELLS[i].name,
                p ? "ACCEPT" : "REFUSE",
                ((p != NULL) == (CELLS[i].expect_ok != 0)) ? "" : "  <<< UNEXPECTED");
        if ((p != NULL) != (CELLS[i].expect_ok != 0)) bad++;
        if (p) vfft_destroy(p);
    }

    fprintf(out, "#\n");
    /* -1 disables the purity assert on the triage path: every cell above has
     * already created once, so a per-cell entry baseline is meaningless here.
     * Triage output is not diffable anyway - use --cell for a real capture. */
    for (i = 0; i < NCELLS; i++) bad += golden_c2c(&CELLS[i], out, -1);

    fprintf(out, "#\n# cells=%d unexpected=%d\n", NCELLS, bad);
    if (out != stdout) fclose(out);
    printf("harness_golden: cells=%d unexpected=%d\n", NCELLS, bad);
    return bad ? 1 : 0;
}
