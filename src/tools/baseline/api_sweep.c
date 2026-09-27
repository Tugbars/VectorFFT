/* api_sweep.c — the public-API sweep for the layout separation gate.
 *
 * WHY THIS EXISTS
 * ---------------
 * harness_golden covers 1D C2C at a dozen cells and fp_sweep a dozen plan
 * fingerprints. The layout separation (docs/roadmap/layout_separation_plan.md)
 * rewires every create tier and both execute paths: 1D/2D/3D/4D, C2C/R2C/C2R
 * and the real-to-real transforms, SPLIT and INTERLEAVED, every order, in place
 * and out of place, batched and threaded. This sweep crosses those axes through
 * the PUBLIC API only and states, per cell:
 *
 *   cell  NAME ACCEPT|REFUSE            the create decision (the refusal matrix)
 *   fp    NAME <fingerprint line>       what was built (the @fp tree)
 *   bits  NAME fwd=... bwd=...          FNV-1a of every buffer after each
 *                                       direction, BITS, zero tolerance
 *   races NAME n                        creates that raced (must be 0 on replay)
 *
 * The buffers are oversized and every one of them is filled and digested, so
 * the digest does not depend on this file knowing each transform's exact
 * extent: whatever the library writes, and whatever it leaves alone, is in the
 * bits. A pointer signature this file gets wrong is refused by vfft_execute
 * (printed, nothing computed) - also deterministic, also in the bits.
 *
 * ONE PROCESS PER CELL (process-lifetime memos, see fp_sweep.c). The driver is
 * api_sweep.py: it banks the sweep's cells into a scratch store once (--write),
 * then replays each cell against a fresh copy of that store, repeated.
 *
 * Build: VFFT_FINGERPRINT=1 python gauntlet/build.py --src src/tools/baseline/api_sweep.c --vfft --compile
 * Run  : api_sweep --list | --cell I [--write]
 *        api_sweep --spec "t=c2c n=64x64 q=1 ord=scr place=oop lay=il"
 *            one wisdom2 @cell key as a config (wisdom_replay.py drives it):
 *            create against the store, report races and the fingerprint, and
 *            the bits only when nothing raced (a raced plan is a coin flip)
 *        api_sweep --roundtrip SRCDIR DSTDIR
 *            vfft_wisdom_load(SRCDIR) then vfft_wisdom_save(DSTDIR)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef VFFT_FINGERPRINT
#error "api_sweep requires -DVFFT_FINGERPRINT (the race counter and the plan fingerprint): VFFT_FINGERPRINT=1 python gauntlet/build.py --src src/tools/baseline/api_sweep.c --vfft --compile"
#endif
#include "vfft_fingerprint.h"

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

typedef struct {
    char name[96];
    vfft_transform_t xf;
    vfft_placement_t place;
    vfft_layout_t layout;
    int order, dims, n[4];
    size_t K;
    int threads, owned, geom;
} cell_t;

#define MAXC 1024
static cell_t C[MAXC];
static int NC;

static const char *XN(vfft_transform_t x)
{
    static const char *n[] = {"c2c", "r2c", "c2r", "dct1", "dct2", "dct3",
                              "dct4", "dst1", "dst2", "dst3", "dht"};
    return n[x];
}

static void add(vfft_transform_t xf, vfft_placement_t pl, vfft_layout_t lay,
                int order, int dims, int n0, int n1, int n2, int n3, size_t K,
                int threads, int owned, int geom)
{
    cell_t *c;
    char dimtxt[48];
    if (NC >= MAXC) return;
    c = &C[NC++];
    memset(c, 0, sizeof *c);
    c->xf = xf; c->place = pl; c->layout = lay; c->order = order;
    c->dims = dims; c->n[0] = n0; c->n[1] = n1; c->n[2] = n2; c->n[3] = n3;
    c->K = K; c->threads = threads; c->owned = owned; c->geom = geom;
    if (dims == 1) snprintf(dimtxt, sizeof dimtxt, "%d", n0);
    else if (dims == 2) snprintf(dimtxt, sizeof dimtxt, "%dx%d", n0, n1);
    else if (dims == 3) snprintf(dimtxt, sizeof dimtxt, "%dx%dx%d", n0, n1, n2);
    else snprintf(dimtxt, sizeof dimtxt, "%dx%dx%dx%d", n0, n1, n2, n3);
    snprintf(c->name, sizeof c->name, "%s.%s.%s.%s.%s.K%zu.T%d%s%s",
             XN(xf), lay == VFFT_LAYOUT_SPLIT ? "sp" : "il",
             pl == VFFT_INPLACE ? "ip" : "oop",
             order == VFFT_ORDER_DEFAULT ? "def" : order == VFFT_ORDER_NATURAL ? "nat" : "scr",
             dimtxt, K, threads, owned ? ".owned" : "",
             geom == VFFT_BATCH_TRANSFORM_CONTIGUOUS ? ".tc" :
             geom == VFFT_BATCH_LANE_MAJOR ? ".lm" : "");
}

/* The cell set. Every line is a family; the sizes are chosen to reach the
 * engines the separation moves (pow2 pair/ZTT, odd chain3/flat, primes, the
 * 2D tier's axes, the 3D tiers), kept small enough that the whole sweep banks
 * and replays in minutes. Refusals are cells too: the matrix is part of the
 * contract. */
static void build_cells(void)
{
    static const int n1d[] = {64, 256, 1024, 4096, 45, 1125, 97, 2401};
    static const int nreal[] = {256, 1024, 1000, 45, 97};
    const vfft_layout_t SP = VFFT_LAYOUT_SPLIT, IL = VFFT_LAYOUT_INTERLEAVED;
    const vfft_placement_t IP = VFFT_INPLACE, OP = VFFT_OUTOFPLACE;
    const int D = VFFT_ORDER_DEFAULT, NA = VFFT_ORDER_NATURAL, SC = VFFT_ORDER_SCRAMBLED;
    size_t i;

    /* 1D C2C */
    for (i = 0; i < sizeof n1d / sizeof n1d[0]; i++) {
        int n = n1d[i];
        add(VFFT_C2C, IP, SP, D,  1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, IP, SP, NA, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, IP, SP, SC, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, OP, SP, D,  1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, IP, SP, D,  1, n, 0, 0, 0, 4, 1, 0, 0);
        add(VFFT_C2C, OP, SP, D,  1, n, 0, 0, 0, 4, 1, 0, 0);
        add(VFFT_C2C, IP, IL, D,  1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, IP, IL, SC, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, OP, IL, D,  1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, OP, IL, NA, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, OP, IL, SC, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2C, OP, IL, D,  1, n, 0, 0, 0, 3, 1, 0, 0);
        add(VFFT_C2C, IP, IL, D,  1, n, 0, 0, 0, 3, 1, 0, 0);
    }
    /* threads: the MT tiers of both layouts */
    add(VFFT_C2C, IP, SP, D, 1, 4096, 0, 0, 0, 8, 2, 0, 0);
    add(VFFT_C2C, OP, SP, D, 1, 4096, 0, 0, 0, 8, 2, 0, 0);
    add(VFFT_C2C, OP, IL, D, 1, 4096, 0, 0, 0, 1, 2, 0, 0);
    add(VFFT_C2C, OP, IL, D, 1, 2401, 0, 0, 0, 1, 2, 0, 0);
    add(VFFT_C2C, OP, IL, D, 1, 1024, 0, 0, 0, 8, 2, 0, 0);
    /* batch geometry and owned buffers */
    add(VFFT_C2C, OP, IL, D, 1, 256, 0, 0, 0, 4, 1, 0, VFFT_BATCH_LANE_MAJOR);
    add(VFFT_C2C, OP, IL, D, 1, 256, 0, 0, 0, 4, 1, 0, VFFT_BATCH_TRANSFORM_CONTIGUOUS);
    add(VFFT_C2C, IP, SP, D, 1, 256, 0, 0, 0, 4, 1, 0, VFFT_BATCH_TRANSFORM_CONTIGUOUS);
    add(VFFT_C2C, IP, SP, D, 1, 256, 0, 0, 0, 6, 1, 1, 0);
    add(VFFT_C2C, IP, IL, D, 1, 256, 0, 0, 0, 4, 1, 1, 0);

    /* 1D real */
    for (i = 0; i < sizeof nreal / sizeof nreal[0]; i++) {
        int n = nreal[i];
        add(VFFT_R2C, OP, SP, D, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2R, OP, SP, D, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_R2C, OP, SP, D, 1, n, 0, 0, 0, 4, 1, 0, 0);
        add(VFFT_R2C, OP, IL, D, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_C2R, OP, IL, D, 1, n, 0, 0, 0, 1, 1, 0, 0);
        add(VFFT_R2C, OP, IL, D, 1, n, 0, 0, 0, 4, 1, 0, 0);
        add(VFFT_C2R, OP, IL, D, 1, n, 0, 0, 0, 4, 1, 0, 0);
        add(VFFT_R2C, IP, IL, D, 1, n, 0, 0, 0, 1, 1, 0, 0);
    }
    add(VFFT_R2C, OP, SP, SC, 1, 256, 0, 0, 0, 1, 1, 0, 0);        /* refused */

    /* real-to-real */
    add(VFFT_DCT1, OP, SP, D, 1, 65, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DCT2, OP, SP, D, 1, 256, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DCT2, OP, SP, D, 1, 60, 0, 0, 0, 4, 1, 0, 0);
    add(VFFT_DCT3, OP, SP, D, 1, 256, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DCT4, OP, SP, D, 1, 256, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DST1, OP, SP, D, 1, 63, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DST2, OP, SP, D, 1, 128, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DST3, OP, SP, D, 1, 128, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DHT,  OP, SP, D, 1, 256, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_DCT2, OP, IL, D, 1, 256, 0, 0, 0, 1, 1, 0, 0);        /* refused */
    add(VFFT_DCT2, OP, SP, D, 2, 64, 64, 0, 0, 1, 1, 0, 0);        /* refused */

    /* 2D C2C */
    {
        static const int p2[][2] = {{64, 64}, {256, 256}, {48, 80}, {45, 64}, {16, 1024}, {97, 32}};
        for (i = 0; i < sizeof p2 / sizeof p2[0]; i++) {
            int a = p2[i][0], b = p2[i][1];
            add(VFFT_C2C, IP, SP, D,  2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2C, IP, SP, NA, 2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, SP, D,  2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2C, IP, IL, D,  2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, IL, D,  2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, IL, SC, 2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, IL, D,  2, a, b, 0, 0, 3, 1, 0, 0);
        }
        add(VFFT_C2C, OP, IL, D, 2, 256, 256, 0, 0, 1, 2, 0, 0);
        add(VFFT_C2C, IP, SP, D, 2, 256, 256, 0, 0, 1, 2, 0, 0);
    }
    /* 2D real */
    {
        static const int r2[][2] = {{64, 64}, {48, 80}, {45, 64}, {64, 45}};
        for (i = 0; i < sizeof r2 / sizeof r2[0]; i++) {
            int a = r2[i][0], b = r2[i][1];
            add(VFFT_R2C, OP, SP, D, 2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2R, OP, SP, D, 2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_R2C, OP, IL, D, 2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_C2R, OP, IL, D, 2, a, b, 0, 0, 1, 1, 0, 0);
            add(VFFT_R2C, OP, IL, NA, 2, a, b, 0, 0, 1, 1, 0, 0);
        }
    }
    /* 3D and 4D */
    {
        static const int p3[][3] = {{16, 16, 16}, {32, 16, 8}, {15, 16, 12}, {64, 64, 64}};
        for (i = 0; i < sizeof p3 / sizeof p3[0]; i++) {
            int a = p3[i][0], b = p3[i][1], c = p3[i][2];
            add(VFFT_C2C, IP, SP, D,  3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, SP, D,  3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_C2C, IP, IL, D,  3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, IL, D,  3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_C2C, OP, IL, SC, 3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_R2C, OP, SP, D,  3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_C2R, OP, SP, D,  3, a, b, c, 0, 1, 1, 0, 0);
            add(VFFT_R2C, OP, IL, D,  3, a, b, c, 0, 1, 1, 0, 0);   /* refused */
        }
        add(VFFT_C2C, OP, IL, D, 3, 64, 64, 64, 0, 1, 2, 0, 0);
        add(VFFT_C2C, OP, SP, D, 4, 8, 8, 8, 8, 1, 1, 0, 0);
        add(VFFT_C2C, IP, SP, D, 4, 8, 8, 8, 8, 1, 1, 0, 0);
        add(VFFT_R2C, OP, SP, D, 4, 8, 8, 8, 8, 1, 1, 0, 0);
        add(VFFT_C2C, OP, IL, D, 4, 8, 8, 8, 8, 1, 1, 0, 0);       /* refused */
    }
    /* plain refusals */
    add(VFFT_C2C, OP, SP, D, 5, 64, 64, 0, 0, 1, 1, 0, 0);
    add(VFFT_C2C, OP, SP, D, 1, 0, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_C2R, OP, SP, NA, 1, 256, 0, 0, 0, 1, 1, 0, 0);
    add(VFFT_C2C, OP, IL, D, 1, 256, 0, 0, 0, 4, 1, 1, 0);
}

/* ------------------------------------------------------------------ bits */

static unsigned long long digest(const double *p, size_t n)
{
    unsigned long long h = 1469598103934665603ULL;
    const unsigned char *b = (const unsigned char *)p;
    size_t i, bytes = n * sizeof(double);
    if (!p) return 0ULL;
    for (i = 0; i < bytes; i++) { h ^= b[i]; h *= 1099511628211ULL; }
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

static long races_now(void)
{
    long c[VFFT__FP_NCOUNTERS];
    vfft__fp_counters(c);
    return c[5];
}

static int is_real2real(vfft_transform_t x) { return x >= VFFT_DCT1; }

/* the pointer roles of vfft_execute, per include/vfft.h's table */
static void run_dir(vfft_plan p, const cell_t *c, vfft_dir_t dir,
                    double *a, double *b, double *d, double *e)
{
    int il = c->layout == VFFT_LAYOUT_INTERLEAVED;
    int ip = c->place == VFFT_INPLACE;
    vfft_transform_t x = c->xf;
    if (is_real2real(x)) { vfft_execute(p, dir, a, NULL, d, NULL); return; }
    if (x == VFFT_C2C) {
        if (il) vfft_execute(p, dir, a, NULL, ip ? a : d, NULL);
        else    vfft_execute(p, dir, a, b, ip ? a : d, ip ? b : e);
        return;
    }
    /* R2C forward = real -> spectrum; its backward runs the C2R roles, and the
     * other way round for a C2R plan */
    {
        int to_spec = (x == VFFT_R2C) == (dir == VFFT_FORWARD);
        if (ip) { vfft_execute(p, dir, a, NULL, a, NULL); return; }
        if (to_spec) vfft_execute(p, dir, a, NULL, d, il ? NULL : e);
        else         vfft_execute(p, dir, a, il ? NULL : b, d, NULL);
    }
}

static int one_cell(const cell_t *c, int i, int write, int bits_only_if_pure);

static int one(int i, int write) { return one_cell(&C[i], i, write, 0); }

static int one_cell(const cell_t *c, int i, int write, int bits_only_if_pure)
{
    static char fp[1 << 16];
    long raced;
    vfft_config_t cfg;
    vfft_plan p;
    size_t ntot = 1, span, k;
    long r0;
    double *buf[4];
    int j;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c->xf;
    cfg.placement = c->place;
    cfg.layout = c->layout;
    cfg.order = c->order;
    cfg.dims = c->dims;
    for (j = 0; j < 4; j++) cfg.n[j] = c->n[j];
    cfg.howmany = c->K;
    cfg.nthreads = c->threads;
    cfg.owned_buffers = c->owned;
    cfg.batch_geom = c->geom;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = write;

    if (c->threads > 1) vfft_set_num_threads(c->threads);
    r0 = races_now();
    p = vfft_create(&cfg);
    printf("cell  %s %s\n", c->name, p ? "ACCEPT" : "REFUSE");
    if (!p) return 0;
    raced = races_now() - r0;
    printf("races %s %ld\n", c->name, raced);
    if (bits_only_if_pure && raced) {
        /* the plan came from a race, not the store: its shape is a coin flip */
        vfft_destroy(p);
        return 0;
    }
    {
        size_t len = vfft__fingerprint(p, fp, sizeof fp);
        char *line, *save = NULL;
        if (len >= sizeof fp) printf("fp    %s TRUNCATED %zu\n", c->name, len);
        for (line = strtok_r(fp, "\n", &save); line; line = strtok_r(NULL, "\n", &save))
            printf("fp    %s %s\n", c->name, line);
    }

    for (j = 0; j < c->dims && j < 4; j++) ntot *= (size_t)(c->n[j] > 0 ? c->n[j] : 1);
    span = (4 * ntot + 64) * (c->K ? c->K : 1);
    if (bits_only_if_pure && ntot * (c->K ? c->K : 1) > ((size_t)1 << 22)) {
        /* a store row can name a 2048x512 plane or a 256^3 cube; its create
         * and fingerprint are the replay's point, four oversized buffers are
         * not (they would not fit four jobs in memory) */
        printf("bits  %s skipped-large\n", c->name);
        vfft_destroy(p);
        return 0;
    }
    if (c->owned) {
        /* the plan's own planes: fill and digest what it hands back */
        double *sre = NULL, *sim = NULL, *dre = NULL, *dim = NULL;
        size_t st = vfft_plan_stride(p);
        size_t n = (size_t)c->n[0] * st;
        vfft_plan_planes(p, &sre, &sim, &dre, &dim);
        if (sre) fill(sre, n, 11u);
        if (sim) fill(sim, n, 12u);
        vfft_execute(p, VFFT_FORWARD, sre, sim, dre, dim);
        printf("bits  %s stride=%zu fwd=%016llx %016llx %016llx %016llx\n", c->name, st,
               digest(sre, sre ? n : 0), digest(sim, sim ? n : 0),
               digest(dre != sre ? dre : NULL, dre && dre != sre ? n : 0),
               digest(dim != sim ? dim : NULL, dim && dim != sim ? n : 0));
        vfft_execute(p, VFFT_BACKWARD, dre ? dre : sre, dim ? dim : sim, sre, sim);
        printf("bits  %s bwd=%016llx %016llx\n", c->name,
               digest(sre, sre ? n : 0), digest(sim, sim ? n : 0));
        vfft_destroy(p);
        return 0;
    }
    for (j = 0; j < 4; j++) {
        buf[j] = (double *)malloc(span * sizeof(double));
        if (!buf[j]) { printf("bits  %s OOM\n", c->name); vfft_destroy(p); return 1; }
        fill(buf[j], span, (unsigned)(i * 4 + j + 1));
    }
    run_dir(p, c, VFFT_FORWARD, buf[0], buf[1], buf[2], buf[3]);
    printf("bits  %s fwd=", c->name);
    for (k = 0; k < 4; k++) printf("%016llx%s", digest(buf[k], span), k < 3 ? " " : "\n");
    run_dir(p, c, VFFT_BACKWARD, buf[2], buf[3], buf[0], buf[1]);
    printf("bits  %s bwd=", c->name);
    for (k = 0; k < 4; k++) printf("%016llx%s", digest(buf[k], span), k < 3 ? " " : "\n");
    for (j = 0; j < 4; j++) free(buf[j]);
    vfft_destroy(p);
    return 0;
}

/* "t=c2c n=64x64 q=1 ord=scr place=oop role=comp lay=il" -> a cell. Absent
 * lay= is a legacy (lay=ANY) row: served to either layout, replayed as SPLIT.
 * place=* rows are replayed out of place. Returns 0 on a key it cannot map. */
static int parse_spec(const char *spec, cell_t *c)
{
    char buf[512], *tok, *save = NULL;
    static const char *xn[] = {"c2c", "r2c", "c2r", "dct1", "dct2", "dct3",
                               "dct4", "dst1", "dst2", "dst3", "dht"};
    int j;
    memset(c, 0, sizeof *c);
    c->K = 1; c->threads = 1; c->place = VFFT_OUTOFPLACE; c->dims = 1;
    c->xf = (vfft_transform_t)-1;
    snprintf(buf, sizeof buf, "%s", spec);
    snprintf(c->name, sizeof c->name, "spec");
    for (tok = strtok_r(buf, " ", &save); tok; tok = strtok_r(NULL, " ", &save)) {
        if (!strncmp(tok, "t=", 2)) {
            for (j = 0; j < 11; j++) if (!strcmp(tok + 2, xn[j])) c->xf = (vfft_transform_t)j;
        } else if (!strncmp(tok, "n=", 2)) {
            char *p = tok + 2;
            c->dims = 0;
            while (*p && c->dims < 4) {
                c->n[c->dims++] = (int)strtol(p, &p, 10);
                if (*p == 'x') p++; else break;
            }
        } else if (!strncmp(tok, "q=", 2)) {
            c->K = (size_t)strtoul(tok + 2, NULL, 10);
        } else if (!strncmp(tok, "ord=", 4)) {
            c->order = !strcmp(tok + 4, "nat") ? VFFT_ORDER_NATURAL :
                       !strcmp(tok + 4, "scr") ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_DEFAULT;
        } else if (!strncmp(tok, "place=", 6)) {
            c->place = !strcmp(tok + 6, "ip") ? VFFT_INPLACE : VFFT_OUTOFPLACE;
        } else if (!strncmp(tok, "lay=", 4)) {
            c->layout = !strcmp(tok + 4, "il") ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
        } else if (!strncmp(tok, "thr=", 4) || !strncmp(tok, "nthr=", 5)) {
            c->threads = atoi(strchr(tok, '=') + 1);
        }
    }
    if (c->xf == (vfft_transform_t)-1 || c->dims < 1) return 0;
    /* real-to-real transforms and 1D real take no order */
    if (c->xf >= VFFT_DCT1 || (c->xf != VFFT_C2C && c->dims == 1)) c->order = VFFT_ORDER_DEFAULT;
    return 1;
}

int main(int argc, char **argv)
{
    int i, write = 0;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    build_cells();
    for (i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--write")) write = 1;
    if (argc > 2 && !strcmp(argv[1], "--spec")) {
        cell_t c;
        setvbuf(stdout, NULL, _IOFBF, 1 << 16);
        if (!parse_spec(argv[2], &c)) { printf("cell  spec UNMAPPED\n"); return 0; }
        return one_cell(&c, 0, 0, 1);
    }
    if (argc > 3 && !strcmp(argv[1], "--roundtrip")) {
        vfft_wisdom *w = vfft_wisdom_load(argv[2]);
        int rc;
        if (!w) { printf("roundtrip LOAD_FAILED\n"); return 1; }
        rc = vfft_wisdom_save(w, argv[3]);
        vfft_wisdom_free(w);
        printf("roundtrip save=%d\n", rc);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--list")) {
        for (i = 0; i < NC; i++) printf("%3d %s\n", i, C[i].name);
        return 0;
    }
    for (i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--cell")) {
            int k = atoi(argv[i + 1]);
            if (k < 0 || k >= NC) { fprintf(stderr, "cell %d out of range\n", k); return 2; }
            setvbuf(stdout, NULL, _IOFBF, 1 << 16);
            return one(k, write);
        }
    fprintf(stderr, "usage: api_sweep --list | --cell I [--write]\n");
    return 2;
}
