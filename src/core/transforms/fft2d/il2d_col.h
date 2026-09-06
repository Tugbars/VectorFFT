/* il2d_col.h — THE COLUMN-AXIS PASS DESCRIPTOR (2026-09-06, phase 1 of
 * the rank-N interleaved tier, docs: memory fftnd_il_campaign).
 *
 * Everything the interleaved column pass over an N-row plane of rn complex
 * per row needs: the chain, its kernels and tables, the banded walk, the
 * natural leaf redirection, the column-axis Bluestein and the column-MT
 * verdict. The 2D plan owns one instance (il2d_col: axis N1 over rows of
 * N2, or hp1 for the real tier); a rank-N IL plan owns one per column axis
 * (fftnd_il.h: axis 0 over N1 rows of N2*N3, axis 1 per plane over N2 rows
 * of N3). The ROW pass is not here: rows belong to the plan that owns them.
 * The column-pass functions (il2d_cols.h, il2d_tier.h) take a pointer to
 * this; nothing in it depends on the plan struct. */
#ifndef VFFT_TRANSFORMS_FFT2D_IL2D_COL_H
#define VFFT_TRANSFORMS_FFT2D_IL2D_COL_H

#include <stddef.h>
#include "il2p.h"   /* vfft_il2p_fn */

typedef struct {
    int N;                    /* rows: the axis length */
    size_t rn;                /* row length in complex: N2 (c2c), hp1 = N2/2+1 (real) */
    /* the column chain: stage s has radix R[s] over sub-length L[s] (D =
     * L/R); stages 0..nst-2 are t2c with driver-built d-major record tables
     * (tf fwd / tb conjugated bwd), the leaf n1c; kernels f fwd / b bwd */
    int nst;
    int R[8], L[8];
    vfft_il2p_fn f[8], b[8];
    double *tf[8], *tb[8];
    int wc;                   /* column-tile width (complex); 0 = full rn (untiled) */
    /* the BANDED walk (the cascade's tcut in 2D form): wl = band width in
     * ROWS (0 = unbanded); cut = DERIVED from wl (the first stage with
     * L_s | wl; wide prefix stages run first); tfuse folds the ROW pass per
     * band (c2c only — the owning plan runs the rows) */
    int wl, cut, tfuse;
    /* the staged band route (§10b): copy each band into scratch at pitch
     * (a skew that kills the 4KB set-group aliasing), suffix + rows there,
     * copy back. Requires wl > 0. c2c only. */
    int staged, pitch;
    double *bandscr;          /* 2 * wl * pitch doubles */
    int colmt;                /* the RACED column-MT verdict for this cell at the plan's T */
    /* NATURAL n1 (M4-lite): the leaf call for block b writes its R rows at
     * out-base natperm[b*R] with OLs = (N/R)*rn — natural order as a driver
     * redirection, any chain; bwd gathers from the natural positions.
     * natperm is block-affine by construction (asserted at create). */
    int nat;
    int natarm;               /* natural x MT partition, RACED at create: 0 = block, 1 = strips */
    int *natperm;             /* N entries, scr row -> natural row */
    double *natscr;           /* 2*N*rn: the pre-leaf plane */
    /* column-axis BLUESTEIN: chirp convolution at M = next pow2 >= 2N-1 over
     * an M x rn scratch plane; blu = M (0 = off); R/L/f/b/tf/tb hold the
     * M-chain */
    int blu;
    double *bluchf, *bluchb;  /* chirp, 2*N each, fwd/bwd */
    double *blukf, *blukb;    /* comb-order kernels, 2*M */
    double *bluscr;           /* the M x rn plane, 2*M*rn */
} vfft_ilcol_t;

#endif /* VFFT_TRANSFORMS_FFT2D_IL2D_COL_H */
