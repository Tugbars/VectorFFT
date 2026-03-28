/**
 * @file vfft_execute.h
 * @brief VectorFFT execution layer — DIF/DIT fused, buffered, sub-problem completion
 *
 * The three optimizations are ONE mechanism:
 *
 *   1. OUTER stages run DIF (outer→inner), with BUFFERED execution
 *      when K exceeds L1. Gather K_BLOCK k-values into contiguous
 *      scratch, run codelet at L1 speed, scatter back.
 *
 *   2. At the DIF→DIT boundary, the outer DIF has placed each
 *      inner sub-problem contiguously in the buffer.
 *
 *   3. INNER stages run DIT (inner→outer) with SUB-PROBLEM COMPLETION.
 *      Each L1-sized sub-problem runs through ALL inner stages
 *      before moving to the next. Data stays L1-hot across stages.
 *
 * DIF reverses outer digits. DIT reverses inner digits.
 * Together they reverse all digits = natural order.
 * ZERO explicit permutation.
 *
 * Requires: vfft_planner.h (types, plan struct, vfft_apply_twiddles_dispatch)
 */

#ifndef VFFT_EXECUTE_H
#define VFFT_EXECUTE_H

#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * L1 PARAMETERS
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_L1_BYTES
#define VFFT_L1_BYTES 49152  /* 48KB — Raptor Lake P-core */
#endif

/* K_BLOCK for buffered execution.
 * Working set per block = (2*R - 1) * K_BLOCK * 16 bytes.
 * For R=16: (31) * 16 * 16 = 7.9KB → fits L1 with room to spare.
 * For R=64: (127) * 8 * 16 = 16.3KB → fits.
 * K_BLOCK must be a multiple of 4 (AVX2 VL). */
#ifndef VFFT_K_BLOCK
#define VFFT_K_BLOCK 16
#endif

/* Sub-problem completion threshold: max inner sub-problem size (doubles)
 * that fits in L1. Working set = inner_size * 32 bytes (src+dst, re+im).
 * 48KB / 32 = 1536. Conservative: 1024. */
#ifndef VFFT_TILE_MAX
#define VFFT_TILE_MAX 1024
#endif

/* Buffered execution threshold: if (2*R-1)*K*16 > this, use buffered */
#define VFFT_BUFFERED_THRESHOLD(R, K) \
    (((2*(R)-1) * (K) * 16) > VFFT_L1_BYTES)


/* ═══════════════════════════════════════════════════════════════
 * SPLIT POINT — where DIF outer meets DIT inner
 *
 * Inner stages: 0 .. split-1 (DIT, sub-problem completion)
 * Outer stages: split .. S-1 (DIF, buffered)
 *
 * Split where cumulative inner sub-problem first exceeds L1.
 * ═══════════════════════════════════════════════════════════════ */

static int vfft_find_split(const vfft_plan *plan)
{
    size_t inner_size = 1;
    for (int s = 0; s < (int)plan->nstages; s++)
    {
        size_t next = inner_size * plan->stages[s].radix;
        if (next > VFFT_TILE_MAX)
            return s;  /* stages 0..s-1 are inner, s..S-1 are outer */
        inner_size = next;
    }
    /* Everything fits L1 — all stages are inner DIT */
    return (int)plan->nstages;
}


/* ═══════════════════════════════════════════════════════════════
 * BUFFERED STAGE — generic gather/codelet/scatter
 *
 * Works for any codelet (DIF fwd, DIF bwd, DIT fwd, DIT bwd).
 * Caller passes the appropriate codelet function pointer.
 *
 * For stages without fused tw codelet, caller passes NULL for
 * tw_codelet and provides notw_codelet + handles twiddle separately.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_buffered_stage(
    const double *__restrict__ src_re, const double *__restrict__ src_im,
    double *__restrict__ dst_re, double *__restrict__ dst_im,
    const vfft_stage *st, size_t group_off,
    vfft_tw_codelet_fn tw_codelet,    /* fused tw codelet (or NULL) */
    vfft_codelet_fn notw_codelet,     /* fallback notw (used if tw_codelet=NULL) */
    int conjugate_twiddle)            /* 0=fwd, 1=bwd twiddle application */
{
    const size_t R = st->radix;
    const size_t K = st->K;
    const size_t kb = VFFT_K_BLOCK;

    /* Stack scratch — sized for max radix 64, K_BLOCK 16 */
    double __attribute__((aligned(32))) s_re[64 * VFFT_K_BLOCK];
    double __attribute__((aligned(32))) s_im[64 * VFFT_K_BLOCK];
    double __attribute__((aligned(32))) d_re[64 * VFFT_K_BLOCK];
    double __attribute__((aligned(32))) d_im[64 * VFFT_K_BLOCK];
    double __attribute__((aligned(32))) tw_r[(64-1) * VFFT_K_BLOCK];
    double __attribute__((aligned(32))) tw_i[(64-1) * VFFT_K_BLOCK];

    const double *in_re = src_re + group_off;
    const double *in_im = src_im + group_off;
    double *out_re = dst_re + group_off;
    double *out_im = dst_im + group_off;

    for (size_t k_base = 0; k_base < K; k_base += kb)
    {
        size_t k_actual = kb;
        if (k_base + k_actual > K) k_actual = K - k_base;

        /* GATHER data: in[n*K + k_base + j] -> s[n*k_actual + j] */
        for (size_t n = 0; n < R; n++) {
            memcpy(s_re + n * k_actual, in_re + n * K + k_base, k_actual * sizeof(double));
            memcpy(s_im + n * k_actual, in_im + n * K + k_base, k_actual * sizeof(double));
        }

        /* GATHER twiddles */
        if (st->tw_re) {
            for (size_t n = 0; n < R - 1; n++) {
                memcpy(tw_r + n * k_actual, st->tw_re + n * K + k_base, k_actual * sizeof(double));
                memcpy(tw_i + n * k_actual, st->tw_im + n * K + k_base, k_actual * sizeof(double));
            }
        }

        /* CODELET at stride K_BLOCK — all L1-hot */
        if (tw_codelet && st->tw_re) {
            tw_codelet(s_re, s_im, d_re, d_im, tw_r, tw_i, k_actual);
        } else if (st->tw_re && notw_codelet) {
            /* DIF fallback: notw then twiddle on output
             * DIT fallback: twiddle on input then notw
             * Caller controls order via conjugate_twiddle being set appropriately.
             * For simplicity, we always do notw then twiddle (DIF style).
             * DIT caller should pre-apply twiddle before calling, or use fused codelet. */
            notw_codelet(s_re, s_im, d_re, d_im, k_actual);
            vfft_apply_twiddles_dispatch(d_re, d_im, tw_r, tw_i, R, k_actual, conjugate_twiddle);
        } else if (notw_codelet) {
            notw_codelet(s_re, s_im, d_re, d_im, k_actual);
        }

        /* SCATTER results */
        for (size_t n = 0; n < R; n++) {
            memcpy(out_re + n * K + k_base, d_re + n * k_actual, k_actual * sizeof(double));
            memcpy(out_im + n * K + k_base, d_im + n * k_actual, k_actual * sizeof(double));
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * FORWARD TRANSFORM — DIF outer + DIT inner
 *
 * Phase 1: DIF outer stages (S-1 down to split)
 *   Natural input → sequential read. Buffered when K > L1.
 *
 * Phase 2: DIT inner stages (0 up to split-1), sub-problem completion
 *   Each L1-sized tile runs through ALL inner stages before next tile.
 *
 * Output: natural order. Zero permutation.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_fwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    const size_t N = plan->N;
    const size_t S = plan->nstages;

    if (N <= 1) {
        if (N == 1) { out_re[0] = in_re[0]; out_im[0] = in_im[0]; }
        return;
    }
    if (S == 1) {
        plan->stages[0].fwd(in_re, in_im, out_re, out_im, 1);
        return;
    }

    const int split = vfft_find_split(plan);

    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* ── Phase 1: Outer DIF stages (outer→inner) ────────────── */

    int first_stage = 1;

    for (int s = (int)S - 1; s >= split; s--)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);
        const int needs_buffer = VFFT_BUFFERED_THRESHOLD(R, K);

        const double *rd_re = first_stage ? in_re : src_re;
        const double *rd_im = first_stage ? in_im : src_im;

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (needs_buffer) {
                vfft_buffered_stage(rd_re, rd_im, dst_re, dst_im,
                    st, off, st->tw_dif_fwd, st->fwd, 0);
            }
            else if (K > 1 && st->tw_re && st->tw_dif_fwd) {
                /* Direct DIF: fused butterfly + output twiddle */
                st->tw_dif_fwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re) {
                /* Fallback: notw then separate twiddle */
                st->fwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
                vfft_apply_twiddles_dispatch(
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, R, K, 0);
            }
            else {
                st->fwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        /* Pointer management */
        if (first_stage) {
            src_re = dst_re; src_im = dst_im;
            dst_re = plan->buf_a_re; dst_im = plan->buf_a_im;
            first_stage = 0;
        } else {
            double *t;
            t = src_re; src_re = dst_re; dst_re = t;
            t = src_im; src_im = dst_im; dst_im = t;
        }
    }

    /* If no outer stages (everything fits L1), copy input to src buffer */
    if (split == (int)S) {
        memcpy(src_re, in_re, N * sizeof(double));
        memcpy(src_im, in_im, N * sizeof(double));
    }

    /* ── Phase 2: Inner DIT stages (sub-problem completion) ─── */

    if (split > 0)
    {
        /* Inner sub-problem size = product of inner radixes */
        size_t inner_size = 1;
        for (int s = 0; s < split; s++)
            inner_size *= plan->stages[s].radix;

        const size_t n_subs = N / inner_size;

        /* Save buffer state — restore for each sub-problem so
         * every sub-problem starts from the same src (DIF output) */
        double *base_src_re = src_re, *base_src_im = src_im;
        double *base_dst_re = dst_re, *base_dst_im = dst_im;

        for (size_t p = 0; p < n_subs; p++)
        {
            size_t sub_off = p * inner_size;

            /* Reset to DIF output buffer for each sub-problem */
            src_re = base_src_re; src_im = base_src_im;
            dst_re = base_dst_re; dst_im = base_dst_im;

            /* Process ALL inner DIT stages on this sub-problem.
             * Stage order: inner→outer (s=0 to split-1).
             * Data stays L1-hot across all inner stages. */
            for (int s = 0; s < split; s++)
            {
                const vfft_stage *st = &plan->stages[s];
                const size_t R = st->radix;
                const size_t K = st->K;
                const size_t groups_in_sub = inner_size / (R * K);

                for (size_t g = 0; g < groups_in_sub; g++)
                {
                    size_t off = sub_off + g * R * K;

                    if (K > 1 && st->tw_re && st->tw_fwd) {
                        /* Fused DIT: input twiddle + butterfly */
                        st->tw_fwd(
                            src_re + off, src_im + off,
                            dst_re + off, dst_im + off,
                            st->tw_re, st->tw_im, K);
                    }
                    else if (K > 1 && st->tw_re) {
                        /* Fallback: separate twiddle then butterfly */
                        vfft_apply_twiddles_dispatch(
                            src_re + off, src_im + off,
                            st->tw_re, st->tw_im, R, K, 0);
                        st->fwd(
                            src_re + off, src_im + off,
                            dst_re + off, dst_im + off, K);
                    }
                    else {
                        /* K=1 (innermost): no twiddle */
                        st->fwd(
                            src_re + off, src_im + off,
                            dst_re + off, dst_im + off, K);
                    }
                }

                /* Swap src/dst for next inner stage */
                double *t;
                t = src_re; src_re = dst_re; dst_re = t;
                t = src_im; src_im = dst_im; dst_im = t;
            }
        }

        /* After all sub-problems: result is in src (after split swaps).
         * If split is odd, result is in base_dst. If even, in base_src. */
    }

    /* Copy result to output */
    if (src_re != out_re) {
        memcpy(out_re, src_re, N * sizeof(double));
        memcpy(out_im, src_im, N * sizeof(double));
    }
}


/* ═══════════════════════════════════════════════════════════════
 * BACKWARD TRANSFORM — DIF outer + DIT inner (backward codelets)
 *
 * Same structure as forward but uses bwd codelets + conjugate twiddles.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_bwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    const size_t N = plan->N;
    const size_t S = plan->nstages;

    if (N <= 1) {
        if (N == 1) { out_re[0] = in_re[0]; out_im[0] = in_im[0]; }
        return;
    }
    if (S == 1) {
        plan->stages[0].bwd(in_re, in_im, out_re, out_im, 1);
        return;
    }

    const int split = vfft_find_split(plan);

    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* ── Phase 1: Outer DIF stages (backward DIF = DIF with conjugate twiddles) ── */

    int first_stage = 1;

    for (int s = (int)S - 1; s >= split; s--)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);
        const int needs_buffer = VFFT_BUFFERED_THRESHOLD(R, K);

        const double *rd_re = first_stage ? in_re : src_re;
        const double *rd_im = first_stage ? in_im : src_im;

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (needs_buffer) {
                vfft_buffered_stage(rd_re, rd_im, dst_re, dst_im,
                    st, off, st->tw_dif_bwd, st->bwd, 1);
            }
            else if (K > 1 && st->tw_re && st->tw_dif_bwd) {
                st->tw_dif_bwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re) {
                st->bwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
                vfft_apply_twiddles_dispatch(
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, R, K, 1);
            }
            else {
                st->bwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        if (first_stage) {
            src_re = dst_re; src_im = dst_im;
            dst_re = plan->buf_a_re; dst_im = plan->buf_a_im;
            first_stage = 0;
        } else {
            double *t;
            t = src_re; src_re = dst_re; dst_re = t;
            t = src_im; src_im = dst_im; dst_im = t;
        }
    }

    if (split == (int)S) {
        memcpy(src_re, in_re, N * sizeof(double));
        memcpy(src_im, in_im, N * sizeof(double));
    }

    /* ── Phase 2: Inner DIT stages (sub-problem completion, bwd codelets) ── */

    if (split > 0)
    {
        size_t inner_size = 1;
        for (int s = 0; s < split; s++)
            inner_size *= plan->stages[s].radix;

        const size_t n_subs = N / inner_size;

        double *base_src_re = src_re, *base_src_im = src_im;
        double *base_dst_re = dst_re, *base_dst_im = dst_im;

        for (size_t p = 0; p < n_subs; p++)
        {
            size_t sub_off = p * inner_size;

            src_re = base_src_re; src_im = base_src_im;
            dst_re = base_dst_re; dst_im = base_dst_im;

            for (int s = 0; s < split; s++)
            {
                const vfft_stage *st = &plan->stages[s];
                const size_t R = st->radix;
                const size_t K = st->K;
                const size_t groups_in_sub = inner_size / (R * K);

                for (size_t g = 0; g < groups_in_sub; g++)
                {
                    size_t off = sub_off + g * R * K;

                    if (K > 1 && st->tw_re && st->tw_bwd) {
                        st->tw_bwd(
                            src_re + off, src_im + off,
                            dst_re + off, dst_im + off,
                            st->tw_re, st->tw_im, K);
                    }
                    else if (K > 1 && st->tw_re) {
                        vfft_apply_twiddles_dispatch(
                            src_re + off, src_im + off,
                            st->tw_re, st->tw_im, R, K, 1);
                        st->bwd(
                            src_re + off, src_im + off,
                            dst_re + off, dst_im + off, K);
                    }
                    else {
                        st->bwd(
                            src_re + off, src_im + off,
                            dst_re + off, dst_im + off, K);
                    }
                }

                double *t;
                t = src_re; src_re = dst_re; dst_re = t;
                t = src_im; src_im = dst_im; dst_im = t;
            }
        }
    }

    if (src_re != out_re) {
        memcpy(out_re, src_re, N * sizeof(double));
        memcpy(out_im, src_im, N * sizeof(double));
    }
}


#endif /* VFFT_EXECUTE_H */
