/* trig_digest_probe.c — output digests for the trig family (DCT/DST/DHT).
 *
 * WHY THIS EXISTS
 * ---------------
 * The migration plan calls step 27 (the trig create tier) its least-protected
 * step, and it is right: the wisdom store holds ZERO trig cells, so the
 * fingerprint replay has almost nothing to replay, and harness_golden covers
 * trig only as two REFUSE decisions -- no output bits at all. Moving that tier
 * with no output check would be moving it on hope.
 *
 * WHAT THIS IS, AND WHAT IT IS DELIBERATELY NOT
 * --------------------------------------------
 * This is a REGRESSION check, not a correctness check. It does not know what
 * a DCT-II of this input should be; it records what the library produces today
 * so the same question can be asked after the move. That is exactly the risk a
 * pure refactor carries, and it is all this needs to cover.
 *
 * The naive O(N^2) reference -- the check that would prove trig CORRECT rather
 * than UNCHANGED -- is deliberately still absent. harness_golden.c states the
 * reason and it holds here too: the plane-role contract for these families is
 * not stated plainly enough in include/vfft.h to encode without guessing, and
 * a WRONG expectation baked into a baseline is worse than a missing one,
 * because every later step then "passes". A digest of today's output cannot be
 * wrong in that way -- it asserts only equality with itself.
 *
 * THE SPAN QUESTION, handled by construction
 * ------------------------------------------
 * Rather than guess each family's output length, every buffer is allocated at
 * 4*N and zeroed, and the digest covers a fixed N doubles. A span guessed too
 * short still digests initialized memory; one guessed too long would digest
 * zeros, not garbage. Either way the value is deterministic, which is the only
 * property a regression digest needs.
 *
 * SERVING MODE. wisdom_write = 0: this probe must never write the store.
 *
 * Public API only. No library internals, no timings, no clock.
 *
 * Build: python build.py --src benches/trig_digest_probe.c --vfft --compile
 * Run  : trig_digest_probe.exe [--out FILE]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vfft.h"

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

/* Deterministic filler, same shape as harness_golden's: the input must not
 * depend on anything that can vary between the before and after run. */
static void fill(double *p, size_t n, unsigned seed)
{
    size_t i;
    unsigned s = seed * 2654435761u + 1u;
    for (i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (double)((int)(s >> 8) % 2000 - 1000) / 512.0;
    }
}

struct cell {
    const char *name;
    int xf;
    int n;
    size_t K;
};

static const struct cell CELLS[] = {
    {"dct1.N64",   VFFT_DCT1,  64, 1},
    {"dct2.N64",   VFFT_DCT2,  64, 1},
    {"dct3.N64",   VFFT_DCT3,  64, 1},
    {"dct4.N64",   VFFT_DCT4,  64, 1},
    {"dst1.N64",   VFFT_DST1,  64, 1},
    {"dst2.N64",   VFFT_DST2,  64, 1},
    {"dst3.N64",   VFFT_DST3,  64, 1},
    {"dht.N64",    VFFT_DHT,   64, 1},
    {"dct2.N256",  VFFT_DCT2, 256, 1},
    {"dct4.N256",  VFFT_DCT4, 256, 1},
    {"dht.N256",   VFFT_DHT,  256, 1},
    {"dct2.N256.K4", VFFT_DCT2, 256, 4},
    {"dct2.N128.K8", VFFT_DCT2, 128, 8},
    {"dht.N128.K4",  VFFT_DHT,  128, 4},
};

int main(int argc, char **argv)
{
    FILE *out = stdout;
    size_t ci;
    int bank = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out = fopen(argv[++i], "wb"); /* wb: LF, so the artifact diffs on
                                           * content, not on line endings */
            if (!out) { fprintf(stderr, "cannot open %s\n", argv[i]); return 2; }
        } else if (!strcmp(argv[i], "--bank")) {
            /* WARM RUN. Trig has no banked wisdom, so every create RACES and
             * the clock picks the plan -- which makes the output digest a coin
             * flip. One --bank run against a SCRATCH VFFT_WISDOM_DIR banks the
             * verdicts; the runs that follow replay them and are deterministic.
             *
             * This is only ever safe because the library refuses to bank when
             * VFFT_WISDOM_DIR is unset (the store then opens read-only). The
             * shipped tree is never a legal target for this flag. */
            bank = 1;
        }
    }

    fprintf(out, "# trig output digests - REGRESSION, not correctness.\n");
    fprintf(out, "# A digest asserts only that the output did not change.\n");
    fprintf(out, "# See the file header for why the naive reference is absent.\n");
    fprintf(out, "#\n");

    for (ci = 0; ci < sizeof(CELLS) / sizeof(CELLS[0]); ci++) {
        const struct cell *c = &CELLS[ci];
        vfft_config_t cfg;
        vfft_plan p;
        size_t span = (size_t)c->n * c->K;
        size_t alloc = span * 4;
        double *a, *d;

        memset(&cfg, 0, sizeof cfg);
        cfg.transform     = (vfft_transform_t)c->xf;
        cfg.dims          = 1;
        cfg.n[0]          = c->n;
        cfg.howmany       = c->K;
        cfg.layout        = VFFT_LAYOUT_SPLIT;
        cfg.placement     = VFFT_OUTOFPLACE;
        cfg.order         = VFFT_ORDER_DEFAULT;
        cfg.rigor         = VFFT_MEASURE;
        cfg.wisdom_write  = bank;      /* 0 = serving, memory-only. 1 only on a
                                        * --bank warm run into a scratch dir. */

        p = vfft_create(&cfg);
        if (!p) { fprintf(out, "trig %-16s CREATE_FAILED\n", c->name); continue; }

        a = (double *)calloc(alloc, sizeof(double));
        d = (double *)calloc(alloc, sizeof(double));
        if (!a || !d) { vfft_destroy(p); free(a); free(d); return 1; }

        fill(a, span, (unsigned)c->n + (unsigned)c->K);
        vfft_execute(p, VFFT_FORWARD, a, NULL, d, NULL);
        fprintf(out, "trig %-16s fwd=%016llx\n", c->name, digest(d, span));

        vfft_destroy(p);
        free(a);
        free(d);
    }

    if (out != stdout) fclose(out);
    return 0;
}
