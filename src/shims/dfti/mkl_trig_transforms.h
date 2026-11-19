/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/*
 * mkl_trig_transforms.h -- MKL's Trigonometric Transform interface on VectorFFT.
 *
 * A program written for MKL's Trigonometric Transform routines builds against this header in place of
 * MKL's: put this directory before MKL's include directory and compile vfft_tt.c beside the VectorFFT
 * library. Each committed handle becomes one VectorFFT plan, run in place on the data vector f, with
 * MKL's scale applied. Double precision. Served:
 *
 *   MKL_SINE_TRANSFORM            VectorFFT's DST-I of the n - 1 values f[1..n-1]
 *   MKL_COSINE_TRANSFORM          its DCT-I of the n + 1 values f[0..n]
 *   MKL_STAGGERED_SINE_TRANSFORM  its DST-III (forward) and DST-II (backward) of the n values f[1..n]
 *
 * The staggered cosine and staggered2 transforms and the single-precision routines are refused with a
 * status and a printed reason.
 *
 * The names are MKL's and so are the numeric values of the constants. The functions behind the names
 * are vfft_tt_*, so MKL can stay linked beside this interface without a symbol in common.
 */
#ifndef VFFT_DFTI_MKL_TRIG_TRANSFORMS_H
#define VFFT_DFTI_MKL_TRIG_TRANSFORMS_H

#include "mkl_dfti.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* MKL_INT: int, 64 bits under MKL_ILP64; MKL's own definition when its headers came first */
#ifndef MKL_INT
#ifdef MKL_ILP64
#define MKL_INT long long int
#else
#define MKL_INT int
#endif
#endif

/* ── the transforms ────────────────────────────────────────────────────── */
#define MKL_SINE_TRANSFORM 0
#define MKL_COSINE_TRANSFORM 1
#define MKL_STAGGERED_COSINE_TRANSFORM 2  /* refused: VectorFFT does not serve DCT-II / DCT-III */
#define MKL_STAGGERED_SINE_TRANSFORM 3
#define MKL_STAGGERED2_COSINE_TRANSFORM 4 /* refused: VectorFFT does not serve DCT-IV */
#define MKL_STAGGERED2_SINE_TRANSFORM 5   /* refused: VectorFFT does not serve DST-IV */

/* ── the functions ─────────────────────────────────────────────────────── */
void vfft_tt_d_init(MKL_INT *n, MKL_INT *tt_type, MKL_INT ipar[], double dpar[], MKL_INT *stat);
void vfft_tt_d_commit(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat);
void vfft_tt_d_forward(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat);
void vfft_tt_d_backward(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat);
/* single precision: refused */
void vfft_tt_s_init(MKL_INT *n, MKL_INT *tt_type, MKL_INT ipar[], float spar[], MKL_INT *stat);
void vfft_tt_s_commit(float f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], float spar[], MKL_INT *stat);
void vfft_tt_s_forward(float f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], float spar[], MKL_INT *stat);
void vfft_tt_s_backward(float f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], float spar[], MKL_INT *stat);
void vfft_tt_free(DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], MKL_INT *stat);

/* ── MKL's names ───────────────────────────────────────────────────────── */
#define d_init_trig_transform vfft_tt_d_init
#define d_commit_trig_transform vfft_tt_d_commit
#define d_forward_trig_transform vfft_tt_d_forward
#define d_backward_trig_transform vfft_tt_d_backward
#define s_init_trig_transform vfft_tt_s_init
#define s_commit_trig_transform vfft_tt_s_commit
#define s_forward_trig_transform vfft_tt_s_forward
#define s_backward_trig_transform vfft_tt_s_backward
#define free_trig_transform vfft_tt_free

#ifdef __cplusplus
}
#endif
#endif /* VFFT_DFTI_MKL_TRIG_TRANSFORMS_H */
