/* demo_cost_model_rank.c — smoke test: does prototype's cost model
 * (src/prototype/cost_model/factorizer.h) rank [4,4,4,16] above
 * [4,4,4,4,4] for N=1024 K=128?
 *
 * If yes, we know the cost model captures the stride context that DP's
 * recursive-bench misses, and the hybrid (cost-model rank → measure
 * top-K) approach is justified.
 *
 * NOTE: this driver uses the cost model ONLY — no plan creation, no
 * measurement, no prototype-core executor. The cost model is fully
 * standalone (its own stride_registry_t stub at factorizer.h:67).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../prototype/cost_model/factorizer.h"

/* Walk every factorization of N (recursively), score each via V1 and V2,
 * print all of them sorted by V2 score. */

#define MAX_DECOMPOSITIONS_DUMP 2048

typedef struct {
    int factors[FACT_MAX_STAGES];
    int nf;
    double v1_score;
    double v2_score;
} _cand_t;

typedef struct {
    _cand_t cands[MAX_DECOMPOSITIONS_DUMP];
    int n;
    int N;
    size_t K;
    const stride_cpu_info_t *cpu;
    int current[FACT_MAX_STAGES];
} _enum_t;

static void _enum_recurse(_enum_t *e, int remaining, int nf) {
    if (remaining == 1) {
        if (nf == 0) return;
        if (e->n >= MAX_DECOMPOSITIONS_DUMP) return;
        _cand_t *c = &e->cands[e->n++];
        c->nf = nf;
        memcpy(c->factors, e->current, (size_t)nf * sizeof(int));
        c->v1_score = stride_score_factorization(c->factors, nf, e->K, e->N, e->cpu);
        c->v2_score = stride_score_factorization_v2(c->factors, nf, e->K, e->N, e->cpu);
        return;
    }
    if (nf >= FACT_MAX_STAGES) return;
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (remaining % R != 0) continue;
        if (!_radix_has_codelet(R)) continue;
        e->current[nf] = R;
        _enum_recurse(e, remaining / R, nf + 1);
    }
}

static int _cmp_v2(const void *a, const void *b) {
    double xa = ((const _cand_t *)a)->v2_score;
    double xb = ((const _cand_t *)b)->v2_score;
    return (xa < xb) ? -1 : (xa > xb) ? 1 : 0;
}

int main(int argc, char **argv) {
    int N = 1024;
    size_t K = 128;
    if (argc >= 3) { N = atoi(argv[1]); K = (size_t)atoll(argv[2]); }

    stride_cpu_info_t cpu = stride_detect_cpu();
    printf("[cost-model-rank] N=%d K=%zu\n", N, (size_t)K);
    printf("  CPU: L1=%zu L2=%zu L3=%zu  DTLB=%d entries (%d cyc/miss)\n\n",
           cpu.l1d_bytes, cpu.l2_bytes, cpu.l3_bytes,
           cpu.dtlb_entries, cpu.dtlb_miss_cycles);

    _enum_t e;
    memset(&e, 0, sizeof(e));
    e.N = N; e.K = K; e.cpu = &cpu;
    _enum_recurse(&e, N, 0);

    qsort(e.cands, e.n, sizeof(_cand_t), _cmp_v2);

    printf("  rank  factors                  V1 score        V2 score\n");
    printf("  ────────────────────────────────────────────────────────\n");
    int show = (e.n < 20) ? e.n : 20;
    for (int i = 0; i < show; i++) {
        char buf[64] = {0};
        int pos = 0;
        for (int s = 0; s < e.cands[i].nf; s++) {
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%s%d",
                            s ? "x" : "", e.cands[i].factors[s]);
        }
        printf("  %3d   %-24s %12.0f    %12.0f\n",
               i + 1, buf, e.cands[i].v1_score, e.cands[i].v2_score);
    }
    if (e.n > show) printf("  ... and %d more\n", e.n - show);
    printf("\n  total candidates enumerated: %d\n", e.n);
    return 0;
}
