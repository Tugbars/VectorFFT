/**
 * bluestein_wisdom.h -- Per-(N, K) wisdom for Bluestein/Rader prime cells.
 *
 * The standard stride wisdom (planner.h) records the best plan for cells
 * that factor smoothly into available radix codelets. Prime N's that
 * don't factor route through Bluestein's algorithm, which embeds the
 * N-point DFT into a length-M circular convolution computed via two
 * inner FFTs at length M >= 2N-1.
 *
 * The two key parameters for Bluestein are:
 *   M  -- inner FFT length (must factor into available radixes)
 *   B  -- block size for the K-axis sweep (cache-friendly chunks)
 *
 * The current heuristic _bluestein_choose_m(N) picks M by minimizing
 * stage count, but that's blind to per-codelet quality. Empirical
 * sweep on N=179 K=256 found the heuristic picks M=361 = 19^2 (2
 * stages of radix-19) when M=384 = 64*6 is 4.65x faster (same 2
 * stages, but radix-64 vastly outperforms radix-19 due to register
 * pressure and retiring efficiency). For N=107 the gap is 1.14x.
 *
 * This header adds wisdom-driven (M, B) selection: at plan time, look
 * up the cell in a separately-loaded wisdom table and use the recorded
 * (M, B) instead of the heuristic. The inner M-point FFT itself is
 * still wisdom-tuned via the existing stride_wisdom_t entries for
 * (M, B) -- so a hit here only fixes the M, B choice and leaves the
 * inner factorization / variants to existing infrastructure.
 *
 * Wisdom file format (separate from stride wisdom):
 *
 *   @bluestein_version 1
 *   # N K M B best_ns
 *   47  256  95  64  53234.0
 *   107 256  240 16  122698.0
 *   179 256  384 16  209911.0
 *   ...
 *
 * Lookup miss = use existing heuristic (zero risk to non-bench cells).
 */
#ifndef STRIDE_BLUESTEIN_WISDOM_H
#define STRIDE_BLUESTEIN_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLUESTEIN_WISDOM_MAX 256
#define BLUESTEIN_WISDOM_VERSION 1

typedef struct {
    int    N;
    size_t K;
    int    M;
    size_t B;
    double best_ns;
} bluestein_wisdom_entry_t;

typedef struct {
    bluestein_wisdom_entry_t entries[BLUESTEIN_WISDOM_MAX];
    int count;
} bluestein_wisdom_t;

static void bluestein_wisdom_init(bluestein_wisdom_t *bw) {
    bw->count = 0;
}

/* Lookup (N, K). Returns NULL on miss or if bw is NULL. */
static const bluestein_wisdom_entry_t *bluestein_wisdom_lookup(
    const bluestein_wisdom_t *bw, int N, size_t K)
{
    if (!bw) return NULL;
    for (int i = 0; i < bw->count; i++) {
        if (bw->entries[i].N == N && bw->entries[i].K == K)
            return &bw->entries[i];
    }
    return NULL;
}

/* Add or update (N, K) entry. If a faster best_ns arrives later, it
 * overrides the prior entry. */
static void bluestein_wisdom_add(bluestein_wisdom_t *bw, int N, size_t K,
                                 int M, size_t B, double best_ns)
{
    for (int i = 0; i < bw->count; i++) {
        if (bw->entries[i].N == N && bw->entries[i].K == K) {
            if (best_ns < bw->entries[i].best_ns) {
                bw->entries[i].M = M;
                bw->entries[i].B = B;
                bw->entries[i].best_ns = best_ns;
            }
            return;
        }
    }
    if (bw->count >= BLUESTEIN_WISDOM_MAX) return;
    bw->entries[bw->count].N = N;
    bw->entries[bw->count].K = K;
    bw->entries[bw->count].M = M;
    bw->entries[bw->count].B = B;
    bw->entries[bw->count].best_ns = best_ns;
    bw->count++;
}

/* Save wisdom file. Returns 0 on success, -1 on failure. */
static int bluestein_wisdom_save(const bluestein_wisdom_t *bw, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@bluestein_version %d\n", BLUESTEIN_WISDOM_VERSION);
    fprintf(f, "# VectorFFT Bluestein wisdom -- %d entries\n", bw->count);
    fprintf(f, "# N K M B best_ns\n");
    for (int i = 0; i < bw->count; i++) {
        fprintf(f, "%d %zu %d %zu %.2f\n",
                bw->entries[i].N, bw->entries[i].K,
                bw->entries[i].M, bw->entries[i].B,
                bw->entries[i].best_ns);
    }
    fclose(f);
    return 0;
}

/* Load wisdom file. Returns number of entries loaded, or -1 on
 * file-open failure. Missing file is not an error -- it's the
 * normal case when the user hasn't run the calibrator. */
static int bluestein_wisdom_load(bluestein_wisdom_t *bw, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int loaded = 0;
    bw->count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] == '@') {
            int ver;
            if (sscanf(line, "@bluestein_version %d", &ver) == 1
                && ver != BLUESTEIN_WISDOM_VERSION) {
                fprintf(stderr, "warn: bluestein wisdom version %d != expected %d\n",
                        ver, BLUESTEIN_WISDOM_VERSION);
            }
            continue;
        }
        bluestein_wisdom_entry_t e;
        if (sscanf(line, "%d %zu %d %zu %lf",
                   &e.N, &e.K, &e.M, &e.B, &e.best_ns) == 5) {
            if (bw->count < BLUESTEIN_WISDOM_MAX) {
                bw->entries[bw->count++] = e;
                loaded++;
            }
        }
    }
    fclose(f);
    return loaded;
}

#endif /* STRIDE_BLUESTEIN_WISDOM_H */
