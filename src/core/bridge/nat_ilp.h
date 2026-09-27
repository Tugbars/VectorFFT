/* nat_ilp.h — B4: the @nat -> IL recipe signpost (bridge, TEMPORARY).
 *
 * A split stride @nat row (the in-place natural cell) can name the IL recipe
 * (mode VFFT_NAT_ILP): its ref_ilp token points at the kind-3 IL row, or the
 * prime shard, that serves the cell. Filling that token reads IL wisdom from
 * the split in-place create, so it lives here, the one place allowed to see
 * both. The owner's D2 answer retires it: the IL in-place door gets its own
 * lay=il row and VFFT_NAT_ILP / ZCASC / CONV leave the split enum (a wisdom
 * format change, its own step with a migration).
 *
 * Caller: split/rank1/c2c_ip_create_split.h (_bank_nat_1d).
 */
#ifndef VFFT_BRIDGE_NAT_ILP_H
#define VFFT_BRIDGE_NAT_ILP_H

#include "il/wisdom/wisdom2_oop_il.h"          /* vw2_prime_method_lookup */
#include "common/wisdom/wisdom2_oop_rows.h"    /* vw2_oop_k1_row_lay(_ord) */
#include "split/wisdom/wisdom2_stride_reader.h" /* vw2_stride_bank_nat */

/* the mode-row RECIPE rule: fac/var are the split plan's chain — the served
 * recipe only for the tape modes. A mode=ilp row carries no chain; its
 * signpost names the IL recipe row instead. */
/* the ILP recipe row, AS KEYED: the kind-3 row at N (lay=il / split /
 * lay-less, exact keys) else the PRIME shard row; 0 = none (mono) */
static int _ilp_ref_of(struct vfft_wisdom_s *W, int N, int mode, int scr_req)
{
    int lay;
    if (mode != VFFT_NAT_ILP || W->vw2_off_oop) return 0;
    /* an explicit SCRAMBLED request is served from its own order cell:
     * the signpost names the ord=scr kind-3 IL row */
    if (scr_req && vw2_oop_k1_row_lay_ord(&W->vw2, N, 1) == VW2_LAY_IL) return 5;
    lay = vw2_oop_k1_row_lay(&W->vw2, N);
    if (lay == VW2_LAY_IL) return 1;
    if (lay == VW2_LAY_SPLIT) return 2;
    if (lay == VW2_LAY_ANY) return 3;
    if (vw2_prime_method_lookup(&W->vw2, N)) return 4;
    return 0;
}

static void _bank_nat_1d(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                         int N, size_t K, int mode, double ns,
                         const int *fac, const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = nf;
    nn.use_dif = use_dif;
    nn.ref_comp = 0 /* no kind-4 recipe rows are written */;
    nn.ref_ilp = _ilp_ref_of(W, N, mode, 0);   /* the @nat cell: the ord=nat recipe */
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++)
    {
        nn.factors[s] = fac[s];
        nn.variants[s] = var[s];
    }
    /* @nat verdicts bank into the wisdom2 store (memory; persistence behind
     * config.wisdom_write). */
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/0, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

#endif /* VFFT_BRIDGE_NAT_ILP_H */
