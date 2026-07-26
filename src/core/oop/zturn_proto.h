/* zturn_proto.h — ZTURN-S full-cascade C PROTOTYPE (Phase 2 of
 * docs/roadmap/cascade_load_path_restructure.md, Amendment ZTURN-S).
 *
 * Moves the K=1 z-cascade's corner-turn from the terminator's LOADS into the
 * ingest's STORES using MKL's observed SECTIONED record geometry:
 *   plane = 2N doubles, 4 sections at byte offsets {0, 4N, 8N, 12N};
 *   ingest position p (p = 0..N/4-1) emits ONE 64-B record
 *   [re(lane0..3)][im(lane0..3)] at section bitrev2(p mod 4), granule p div 4;
 *   lanes = the radix-4 butterfly OUTPUT digit (the turn, paid store-side).
 *
 * Pipeline:
 *   fwd: s0t  (fused turn ingest, PROVEN body from bench_ingest_diag.c arm c)
 *        -> PRODUCTION radix{4,8}_z_msg_fwd_avx2 BYTE-FOR-BYTE, production
 *           arg tuple (Ls=D_s, Gs=G_s, count=D_s), in-place on the plane;
 *           ONLY the table contents repack (lane-varying, x4 section-tiled)
 *        -> stf  (NEW terminator: 4 section taps, 128 B contiguous per tap,
 *           radix8_z_sterm_fwd's EXACT op graph, REINT stores; the TR4 load
 *           edge is GONE).
 *   bwd: stfb (DEINT loads, radix8_z_sterm_bwd's EXACT op graph, direct
 *        record stores) -> msg_bwd byte-for-byte -> s0tb (un-turn in the
 *        LOAD network: 4 section records + 4x4 transpose, IDFT-4, REINT).
 *
 * Output order (SCRAMBLED class, fwd/bwd MATCHED):
 *   out_z[l*(N/8) + 4*k' + j] = X[l*(N/8) + 4*rho(k') + j],
 *   rho = digit reversal over the MIDDLE radices (chain[1..nf-2]).
 *   Differs from legacy zsplit by a pure per-row (N/32 x 4) transpose
 *   (Gamma): out_z[l*(N/8)+4k'+j] = out_legacy[l*(N/8)+j*(N/32)+k'].
 *
 * All four NEW bodies are 1:1 intrinsic transcriptions of simulator-proven
 * op graphs (scratchpad p2/derive_a/zturn_sim.c, memcmp-EXACT vs production
 * at N=2048/4096/8192/16384 fwd+bwd); arithmetic order and FMA placement
 * preserved, so the gates in build_tuned/benches/zturn_proto_gate.c are
 * memcmp-EXACT. Scope fence: chain[0] == 4 REQUIRED (4-section geometry).
 * Roundtrip = N*x (no 1/N in-kernel), matching zsplit. zin == zout OK.
 */
#ifndef VFFT_ZTURN_PROTO_H
#define VFFT_ZTURN_PROTO_H

#include <immintrin.h>
#include "zsplit.h"   /* chains, _vfft_zs_brev, msg kernel decls, allocator */

typedef struct {
    int N, nf;
    int chain[VFFT_ZSPLIT_MAX_NF];
    long D[VFFT_ZSPLIT_MAX_NF], G[VFFT_ZSPLIT_MAX_NF];
    double *twz[VFFT_ZSPLIT_MAX_NF];   /* mid tables, lane-varying, fwd     */
    double *twzb[VFFT_ZSPLIT_MAX_NF];  /* mid tables, bwd (sin negated)     */
    double *tzq, *tzqb;                /* terminator per-(k',lane) w^1      */
    double *plane;                     /* sectioned plane, 2N doubles, 64B  */
} vfft_zturn_plan_t;

/* (im,re)-swap sign mask for x * (-i): xor lane 1,3 after permute.
 * File-scope const so gcc hoists ONE load out of the ingest loop. */
static const __m256d _vzt_mim = { 0.0, -0.0, 0.0, -0.0 };

/* ====================================================================== */
/* s0t — fused-turn SECTIONED ingest (fwd). VERBATIM transcription of the */
/* proven arm c 's0t_sect_fwd' (build_tuned/benches/bench_ingest_diag.c): */
/* 18-op turn lattice, 4 rate-matched section cursors, +64 B/iter each.   */
/* Position k -> sec0, k+1 -> sec2, k+2 -> sec1, k+3 -> sec3 (bitrev2).   */
/* ====================================================================== */
__attribute__((noinline, used, target("avx2,fma")))
static void _vfft_zturn_s0t_fwd(const double * __restrict zin,
                                double * __restrict sp, size_t N)
{
    const size_t Ls = N / 4;
    double * __restrict s0 = sp;                /* byte 0   = section 0 */
    double * __restrict s1 = sp + (N >> 1);     /* byte 4N  = section 1 */
    double * __restrict s2 = sp + N;            /* byte 8N  = section 2 */
    double * __restrict s3 = sp + N + (N >> 1); /* byte 12N = section 3 */
    for (size_t k = 0; k + 4 <= Ls; k += 4) {
        /* ---- half A: positions k, k+1 ---- */
        const __m256d a0 = _mm256_loadu_pd(&zin[2*(0*Ls + k)]);
        const __m256d a1 = _mm256_loadu_pd(&zin[2*(1*Ls + k)]);
        const __m256d a2 = _mm256_loadu_pd(&zin[2*(2*Ls + k)]);
        const __m256d a3 = _mm256_loadu_pd(&zin[2*(3*Ls + k)]);
        const __m256d at0 = _mm256_add_pd(a0, a2);
        const __m256d at1 = _mm256_sub_pd(a0, a2);
        const __m256d at2 = _mm256_add_pd(a1, a3);
        const __m256d at3 = _mm256_sub_pd(a1, a3);
        const __m256d ar  = _mm256_xor_pd(_mm256_permute_pd(at3, 0x5), _vzt_mim);
        const __m256d aY0 = _mm256_add_pd(at0, at2);
        const __m256d aY2 = _mm256_sub_pd(at0, at2);
        const __m256d aY1 = _mm256_add_pd(at1, ar);
        const __m256d aY3 = _mm256_sub_pd(at1, ar);
        const __m256d au0 = _mm256_unpacklo_pd(aY0, aY1);
        const __m256d au1 = _mm256_unpackhi_pd(aY0, aY1);
        const __m256d au2 = _mm256_unpacklo_pd(aY2, aY3);
        const __m256d au3 = _mm256_unpackhi_pd(aY2, aY3);
        _mm256_storeu_pd(&s0[2*k + 0], _mm256_permute2f128_pd(au0, au2, 0x20)); /* re(k)   -> sec0 */
        _mm256_storeu_pd(&s0[2*k + 4], _mm256_permute2f128_pd(au1, au3, 0x20)); /* im(k)   -> sec0 */
        _mm256_storeu_pd(&s2[2*k + 0], _mm256_permute2f128_pd(au0, au2, 0x31)); /* re(k+1) -> sec2 */
        _mm256_storeu_pd(&s2[2*k + 4], _mm256_permute2f128_pd(au1, au3, 0x31)); /* im(k+1) -> sec2 */
        /* ---- half B: positions k+2, k+3 ---- */
        const __m256d b0 = _mm256_loadu_pd(&zin[2*(0*Ls + k) + 4]);
        const __m256d b1 = _mm256_loadu_pd(&zin[2*(1*Ls + k) + 4]);
        const __m256d b2 = _mm256_loadu_pd(&zin[2*(2*Ls + k) + 4]);
        const __m256d b3 = _mm256_loadu_pd(&zin[2*(3*Ls + k) + 4]);
        const __m256d bt0 = _mm256_add_pd(b0, b2);
        const __m256d bt1 = _mm256_sub_pd(b0, b2);
        const __m256d bt2 = _mm256_add_pd(b1, b3);
        const __m256d bt3 = _mm256_sub_pd(b1, b3);
        const __m256d br  = _mm256_xor_pd(_mm256_permute_pd(bt3, 0x5), _vzt_mim);
        const __m256d bY0 = _mm256_add_pd(bt0, bt2);
        const __m256d bY2 = _mm256_sub_pd(bt0, bt2);
        const __m256d bY1 = _mm256_add_pd(bt1, br);
        const __m256d bY3 = _mm256_sub_pd(bt1, br);
        const __m256d bu0 = _mm256_unpacklo_pd(bY0, bY1);
        const __m256d bu1 = _mm256_unpackhi_pd(bY0, bY1);
        const __m256d bu2 = _mm256_unpacklo_pd(bY2, bY3);
        const __m256d bu3 = _mm256_unpackhi_pd(bY2, bY3);
        _mm256_storeu_pd(&s1[2*k + 0], _mm256_permute2f128_pd(bu0, bu2, 0x20)); /* re(k+2) -> sec1 */
        _mm256_storeu_pd(&s1[2*k + 4], _mm256_permute2f128_pd(bu1, bu3, 0x20)); /* im(k+2) -> sec1 */
        _mm256_storeu_pd(&s3[2*k + 0], _mm256_permute2f128_pd(bu0, bu2, 0x31)); /* re(k+3) -> sec3 */
        _mm256_storeu_pd(&s3[2*k + 4], _mm256_permute2f128_pd(bu1, bu3, 0x31)); /* im(k+3) -> sec3 */
    }
}

/* ====================================================================== */
/* stf — NEW terminator (fwd). Loads: leg q at record                     */
/* SEC[q mod 4]*S + 2*k' + (q div 4), SEC = {0,2,1,3}: two consecutive    */
/* records (128 B contiguous) per section, lanes pass straight through — */
/* NO shuffles on the load edge. Math + REINT store edge: VERBATIM        */
/* radix8_z_sterm_fwd_avx2 op graph (t44..t138), w^1 per (k',lane) from  */
/* tzq[8k'..], powers by the exact squaring tree. Column base = 4*k'.    */
/* ====================================================================== */
__attribute__((noinline, used, target("avx2,fma")))
static void _vfft_zturn_stf_fwd(const double * __restrict plane,
                                double * __restrict zout,
                                const double * __restrict tzq, size_t N)
{
    const size_t S = N >> 4, OLs = N >> 3, K2 = N >> 5;
    const double * __restrict p0 = plane;           /* section 0 */
    const double * __restrict p1 = plane + 8*S;     /* section 1 */
    const double * __restrict p2 = plane + 16*S;    /* section 2 */
    const double * __restrict p3 = plane + 24*S;    /* section 3 */
    for (size_t k2 = 0; k2 < K2; k2++) {
        const size_t rb = 16*k2;              /* 2 records per section/group */
        const size_t cb = 4*k2;               /* output column base          */
        /* section-tap load edge (leg q & 3 -> SEC: 0,2,1,3) */
        const __m256d lane_re_0 = _mm256_loadu_pd(&p0[rb + 0]);
        const __m256d lane_im_0 = _mm256_loadu_pd(&p0[rb + 4]);
        const __m256d lane_re_1 = _mm256_loadu_pd(&p2[rb + 0]);
        const __m256d lane_im_1 = _mm256_loadu_pd(&p2[rb + 4]);
        const __m256d lane_re_2 = _mm256_loadu_pd(&p1[rb + 0]);
        const __m256d lane_im_2 = _mm256_loadu_pd(&p1[rb + 4]);
        const __m256d lane_re_3 = _mm256_loadu_pd(&p3[rb + 0]);
        const __m256d lane_im_3 = _mm256_loadu_pd(&p3[rb + 4]);
        const __m256d lane_re_4 = _mm256_loadu_pd(&p0[rb + 8]);
        const __m256d lane_im_4 = _mm256_loadu_pd(&p0[rb + 12]);
        const __m256d lane_re_5 = _mm256_loadu_pd(&p2[rb + 8]);
        const __m256d lane_im_5 = _mm256_loadu_pd(&p2[rb + 12]);
        const __m256d lane_re_6 = _mm256_loadu_pd(&p1[rb + 8]);
        const __m256d lane_im_6 = _mm256_loadu_pd(&p1[rb + 12]);
        const __m256d lane_re_7 = _mm256_loadu_pd(&p3[rb + 8]);
        const __m256d lane_im_7 = _mm256_loadu_pd(&p3[rb + 12]);
        /* SU-scheduled body — VERBATIM radix8_z_sterm_fwd_avx2 */
        const __m256d t44 = _mm256_set1_pd(0.70710678118654757);
        const __m256d t0 = lane_re_7;
        const __m256d t1 = _mm256_loadu_pd(&tzq[8*k2 + 0]);
        const __m256d t2 = _mm256_loadu_pd(&tzq[8*k2 + 4]);
        const __m256d t3 = _mm256_fnmadd_pd(t2, t2, _mm256_mul_pd(t1, t1));
        const __m256d t4 = _mm256_fmadd_pd(t1, t2, _mm256_mul_pd(t2, t1));
        const __m256d t7 = _mm256_fnmadd_pd(t4, t2, _mm256_mul_pd(t3, t1));
        const __m256d t8 = _mm256_fmadd_pd(t3, t2, _mm256_mul_pd(t4, t1));
        const __m256d t5 = _mm256_fnmadd_pd(t4, t4, _mm256_mul_pd(t3, t3));
        const __m256d t6 = _mm256_fmadd_pd(t3, t4, _mm256_mul_pd(t4, t3));
        const __m256d t9 = _mm256_fnmadd_pd(t6, t8, _mm256_mul_pd(t5, t7));
        const __m256d t10 = _mm256_fmadd_pd(t5, t8, _mm256_mul_pd(t6, t7));
        const __m256d t28 = _mm256_fnmadd_pd(t6, t2, _mm256_mul_pd(t5, t1));
        const __m256d t29 = _mm256_fmadd_pd(t5, t2, _mm256_mul_pd(t6, t1));
        const __m256d t52 = _mm256_fnmadd_pd(t6, t4, _mm256_mul_pd(t5, t3));
        const __m256d t53 = _mm256_fmadd_pd(t5, t4, _mm256_mul_pd(t6, t3));
        const __m256d t11 = lane_im_7;
        const __m256d t12 = _mm256_fnmadd_pd(t11, t10, _mm256_mul_pd(t0, t9));
        const __m256d t13 = _mm256_fmadd_pd(t0, t10, _mm256_mul_pd(t11, t9));
        const __m256d t17 = lane_re_3;
        const __m256d t18 = lane_im_3;
        const __m256d t19 = _mm256_fnmadd_pd(t18, t8, _mm256_mul_pd(t17, t7));
        const __m256d t20 = _mm256_fmadd_pd(t17, t8, _mm256_mul_pd(t18, t7));
        const __m256d t22 = _mm256_sub_pd(t20, t13);
        const __m256d t24 = _mm256_sub_pd(t19, t12);
        const __m256d t81 = _mm256_add_pd(t13, t20);
        const __m256d t82 = _mm256_add_pd(t12, t19);
        const __m256d t27 = lane_re_5;
        const __m256d t30 = lane_im_5;
        const __m256d t31 = _mm256_fnmadd_pd(t30, t29, _mm256_mul_pd(t27, t28));
        const __m256d t32 = _mm256_fmadd_pd(t27, t29, _mm256_mul_pd(t30, t28));
        const __m256d t34 = lane_re_1;
        const __m256d t35 = lane_im_1;
        const __m256d t36 = _mm256_fnmadd_pd(t35, t2, _mm256_mul_pd(t34, t1));
        const __m256d t37 = _mm256_fmadd_pd(t34, t2, _mm256_mul_pd(t35, t1));
        const __m256d t38 = _mm256_sub_pd(t37, t32);
        const __m256d t40 = _mm256_sub_pd(t36, t31);
        const __m256d t84 = _mm256_add_pd(t32, t37);
        const __m256d t85 = _mm256_add_pd(t31, t36);
        const __m256d t41 = _mm256_add_pd(t24, t38);
        const __m256d t42 = _mm256_sub_pd(t40, t22);
        const __m256d t101 = _mm256_sub_pd(t38, t24);
        const __m256d t103 = _mm256_add_pd(t22, t40);
        const __m256d t86 = _mm256_sub_pd(t84, t81);
        const __m256d t88 = _mm256_sub_pd(t85, t82);
        const __m256d t115 = _mm256_add_pd(t81, t84);
        const __m256d t116 = _mm256_add_pd(t82, t85);
        const __m256d t43 = _mm256_sub_pd(t42, t41);
        const __m256d t48 = _mm256_add_pd(t41, t42);
        const __m256d t104 = _mm256_add_pd(t101, t103);
        const __m256d t107 = _mm256_sub_pd(t101, t103);
        const __m256d t51 = lane_re_6;
        const __m256d t54 = lane_im_6;
        const __m256d t55 = _mm256_fnmadd_pd(t54, t53, _mm256_mul_pd(t51, t52));
        const __m256d t56 = _mm256_fmadd_pd(t51, t53, _mm256_mul_pd(t54, t52));
        const __m256d t58 = lane_re_2;
        const __m256d t59 = lane_im_2;
        const __m256d t60 = _mm256_fnmadd_pd(t59, t4, _mm256_mul_pd(t58, t3));
        const __m256d t61 = _mm256_fmadd_pd(t58, t4, _mm256_mul_pd(t59, t3));
        const __m256d t62 = _mm256_sub_pd(t61, t56);
        const __m256d t64 = _mm256_sub_pd(t60, t55);
        const __m256d t91 = _mm256_add_pd(t56, t61);
        const __m256d t92 = _mm256_add_pd(t55, t60);
        const __m256d t67 = lane_re_4;
        const __m256d t68 = lane_im_4;
        const __m256d t69 = _mm256_fnmadd_pd(t68, t6, _mm256_mul_pd(t67, t5));
        const __m256d t70 = _mm256_fmadd_pd(t67, t6, _mm256_mul_pd(t68, t5));
        const __m256d t72 = lane_re_0;
        const __m256d t76 = _mm256_sub_pd(t72, t69);
        const __m256d t95 = _mm256_add_pd(t69, t72);
        const __m256d t78 = _mm256_sub_pd(t76, t62);
        const __m256d t131 = _mm256_fmadd_pd(t43, t44, t78);
        const __m256d t135 = _mm256_fnmadd_pd(t43, t44, t78);
        const __m256d t98 = _mm256_sub_pd(t95, t92);
        const __m256d t99 = _mm256_sub_pd(t98, t86);
        const __m256d t125 = _mm256_add_pd(t86, t98);
        const __m256d t111 = _mm256_add_pd(t62, t76);
        const __m256d t133 = _mm256_fnmadd_pd(t44, t104, t111);
        const __m256d t137 = _mm256_fmadd_pd(t44, t104, t111);
        const __m256d t119 = _mm256_add_pd(t92, t95);
        const __m256d t120 = _mm256_sub_pd(t119, t116);
        const __m256d t129 = _mm256_add_pd(t116, t119);
        const __m256d t73 = lane_im_0;
        const __m256d t74 = _mm256_sub_pd(t73, t70);
        const __m256d t94 = _mm256_add_pd(t70, t73);
        const __m256d t77 = _mm256_add_pd(t64, t74);
        const __m256d t132 = _mm256_fmadd_pd(t44, t48, t77);
        const __m256d t136 = _mm256_fnmadd_pd(t44, t48, t77);
        const __m256d t96 = _mm256_sub_pd(t94, t91);
        const __m256d t100 = _mm256_add_pd(t88, t96);
        const __m256d t126 = _mm256_sub_pd(t96, t88);
        const __m256d t110 = _mm256_sub_pd(t74, t64);
        const __m256d t134 = _mm256_fnmadd_pd(t44, t107, t110);
        const __m256d t138 = _mm256_fmadd_pd(t44, t107, t110);
        const __m256d t118 = _mm256_add_pd(t91, t94);
        const __m256d t122 = _mm256_sub_pd(t118, t115);
        const __m256d t130 = _mm256_add_pd(t115, t118);
        /* Z store edge (REINT) — VERBATIM, column base cb = 4*k' */
        const __m256d _pr_0 = _mm256_permute4x64_pd(t129, 0xD8);
        const __m256d _qi_0 = _mm256_permute4x64_pd(t130, 0xD8);
        _mm256_storeu_pd(&zout[2*cb], _mm256_unpacklo_pd(_pr_0, _qi_0));
        _mm256_storeu_pd(&zout[2*cb + 4], _mm256_unpackhi_pd(_pr_0, _qi_0));
        const __m256d _pr_1 = _mm256_permute4x64_pd(t137, 0xD8);
        const __m256d _qi_1 = _mm256_permute4x64_pd(t138, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)1*OLs + cb)], _mm256_unpacklo_pd(_pr_1, _qi_1));
        _mm256_storeu_pd(&zout[2*((size_t)1*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_1, _qi_1));
        const __m256d _pr_2 = _mm256_permute4x64_pd(t125, 0xD8);
        const __m256d _qi_2 = _mm256_permute4x64_pd(t126, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)2*OLs + cb)], _mm256_unpacklo_pd(_pr_2, _qi_2));
        _mm256_storeu_pd(&zout[2*((size_t)2*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_2, _qi_2));
        const __m256d _pr_3 = _mm256_permute4x64_pd(t135, 0xD8);
        const __m256d _qi_3 = _mm256_permute4x64_pd(t136, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)3*OLs + cb)], _mm256_unpacklo_pd(_pr_3, _qi_3));
        _mm256_storeu_pd(&zout[2*((size_t)3*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_3, _qi_3));
        const __m256d _pr_4 = _mm256_permute4x64_pd(t120, 0xD8);
        const __m256d _qi_4 = _mm256_permute4x64_pd(t122, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)4*OLs + cb)], _mm256_unpacklo_pd(_pr_4, _qi_4));
        _mm256_storeu_pd(&zout[2*((size_t)4*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_4, _qi_4));
        const __m256d _pr_5 = _mm256_permute4x64_pd(t133, 0xD8);
        const __m256d _qi_5 = _mm256_permute4x64_pd(t134, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)5*OLs + cb)], _mm256_unpacklo_pd(_pr_5, _qi_5));
        _mm256_storeu_pd(&zout[2*((size_t)5*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_5, _qi_5));
        const __m256d _pr_6 = _mm256_permute4x64_pd(t99, 0xD8);
        const __m256d _qi_6 = _mm256_permute4x64_pd(t100, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)6*OLs + cb)], _mm256_unpacklo_pd(_pr_6, _qi_6));
        _mm256_storeu_pd(&zout[2*((size_t)6*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_6, _qi_6));
        const __m256d _pr_7 = _mm256_permute4x64_pd(t131, 0xD8);
        const __m256d _qi_7 = _mm256_permute4x64_pd(t132, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)7*OLs + cb)], _mm256_unpacklo_pd(_pr_7, _qi_7));
        _mm256_storeu_pd(&zout[2*((size_t)7*OLs + cb) + 4], _mm256_unpackhi_pd(_pr_7, _qi_7));
    }
}

/* ====================================================================== */
/* stfb — NEW backward terminator. DEINT loads from the ZTURN-S comb      */
/* (column base 4*k'), radix8_z_sterm_bwd_avx2's EXACT op graph           */
/* (t45..t145, IDFT-8 then POST conj-w^q), then DIRECT record stores:     */
/* leg q -> record SEC[q mod 4]*S + 2*k' + (q div 4), lanes straight —   */
/* the TR4 store edge is GONE (exact address-mirror of stf's loads).      */
/* Leg map (from sterm_bwd's TR4 stores): 0=(t30,t31) 1=(t132,t133)       */
/* 2=(t79,t80) 3=(t136,t137) 4=(t105,t106) 5=(t140,t141) 6=(t120,t121)    */
/* 7=(t144,t145).                                                         */
/* ====================================================================== */
__attribute__((noinline, used, target("avx2,fma")))
static void _vfft_zturn_stfb_bwd(const double * __restrict zin,
                                 double * __restrict plane,
                                 const double * __restrict tzqb, size_t N)
{
    const size_t S = N >> 4, OLs = N >> 3, K2 = N >> 5;
    double * __restrict p0 = plane;           /* section 0 */
    double * __restrict p1 = plane + 8*S;     /* section 1 */
    double * __restrict p2 = plane + 16*S;    /* section 2 */
    double * __restrict p3 = plane + 24*S;    /* section 3 */
    for (size_t k2 = 0; k2 < K2; k2++) {
        const size_t rb = 16*k2;
        const size_t cb = 4*k2;
        /* Z load edge (DEINT) — VERBATIM radix8_z_sterm_bwd_avx2, k -> cb */
        const __m256d _zl_0 = _mm256_loadu_pd(&zin[2*cb]);
        const __m256d _zh_0 = _mm256_loadu_pd(&zin[2*cb + 4]);
        const __m256d lane_re_0 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_0, _zh_0), 0xD8);
        const __m256d lane_im_0 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_0, _zh_0), 0xD8);
        const __m256d _zl_1 = _mm256_loadu_pd(&zin[2*((size_t)1*OLs + cb)]);
        const __m256d _zh_1 = _mm256_loadu_pd(&zin[2*((size_t)1*OLs + cb) + 4]);
        const __m256d lane_re_1 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_1, _zh_1), 0xD8);
        const __m256d lane_im_1 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_1, _zh_1), 0xD8);
        const __m256d _zl_2 = _mm256_loadu_pd(&zin[2*((size_t)2*OLs + cb)]);
        const __m256d _zh_2 = _mm256_loadu_pd(&zin[2*((size_t)2*OLs + cb) + 4]);
        const __m256d lane_re_2 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_2, _zh_2), 0xD8);
        const __m256d lane_im_2 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_2, _zh_2), 0xD8);
        const __m256d _zl_3 = _mm256_loadu_pd(&zin[2*((size_t)3*OLs + cb)]);
        const __m256d _zh_3 = _mm256_loadu_pd(&zin[2*((size_t)3*OLs + cb) + 4]);
        const __m256d lane_re_3 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_3, _zh_3), 0xD8);
        const __m256d lane_im_3 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_3, _zh_3), 0xD8);
        const __m256d _zl_4 = _mm256_loadu_pd(&zin[2*((size_t)4*OLs + cb)]);
        const __m256d _zh_4 = _mm256_loadu_pd(&zin[2*((size_t)4*OLs + cb) + 4]);
        const __m256d lane_re_4 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_4, _zh_4), 0xD8);
        const __m256d lane_im_4 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_4, _zh_4), 0xD8);
        const __m256d _zl_5 = _mm256_loadu_pd(&zin[2*((size_t)5*OLs + cb)]);
        const __m256d _zh_5 = _mm256_loadu_pd(&zin[2*((size_t)5*OLs + cb) + 4]);
        const __m256d lane_re_5 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_5, _zh_5), 0xD8);
        const __m256d lane_im_5 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_5, _zh_5), 0xD8);
        const __m256d _zl_6 = _mm256_loadu_pd(&zin[2*((size_t)6*OLs + cb)]);
        const __m256d _zh_6 = _mm256_loadu_pd(&zin[2*((size_t)6*OLs + cb) + 4]);
        const __m256d lane_re_6 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_6, _zh_6), 0xD8);
        const __m256d lane_im_6 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_6, _zh_6), 0xD8);
        const __m256d _zl_7 = _mm256_loadu_pd(&zin[2*((size_t)7*OLs + cb)]);
        const __m256d _zh_7 = _mm256_loadu_pd(&zin[2*((size_t)7*OLs + cb) + 4]);
        const __m256d lane_re_7 = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zl_7, _zh_7), 0xD8);
        const __m256d lane_im_7 = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zl_7, _zh_7), 0xD8);
        /* SU-scheduled body — VERBATIM radix8_z_sterm_bwd_avx2 */
        const __m256d t45 = _mm256_set1_pd(0.70710678118654757);
        const __m256d t0 = lane_re_7;
        const __m256d t2 = lane_im_7;
        const __m256d t4 = lane_re_3;
        const __m256d t36 = _mm256_sub_pd(t4, t0);
        const __m256d t7 = _mm256_add_pd(t0, t4);
        const __m256d t5 = lane_im_3;
        const __m256d t34 = _mm256_sub_pd(t5, t2);
        const __m256d t6 = _mm256_add_pd(t2, t5);
        const __m256d t8 = lane_re_5;
        const __m256d t9 = lane_im_5;
        const __m256d t10 = lane_re_1;
        const __m256d t41 = _mm256_sub_pd(t10, t8);
        const __m256d t83 = _mm256_add_pd(t34, t41);
        const __m256d t13 = _mm256_add_pd(t8, t10);
        const __m256d t43 = _mm256_sub_pd(t41, t34);
        const __m256d t15 = _mm256_add_pd(t7, t13);
        const __m256d t69 = _mm256_sub_pd(t13, t7);
        const __m256d t11 = lane_im_1;
        const __m256d t39 = _mm256_sub_pd(t11, t9);
        const __m256d t12 = _mm256_add_pd(t9, t11);
        const __m256d t42 = _mm256_add_pd(t36, t39);
        const __m256d t82 = _mm256_sub_pd(t39, t36);
        const __m256d t14 = _mm256_add_pd(t6, t12);
        const __m256d t67 = _mm256_sub_pd(t12, t6);
        const __m256d t44 = _mm256_sub_pd(t43, t42);
        const __m256d t47 = _mm256_add_pd(t42, t43);
        const __m256d t85 = _mm256_sub_pd(_mm256_xor_pd(t83, _mm256_set1_pd(-0.0)), t82);
        const __m256d t88 = _mm256_sub_pd(t83, t82);
        const __m256d t16 = lane_re_6;
        const __m256d t17 = lane_im_6;
        const __m256d t18 = lane_re_2;
        const __m256d t21 = _mm256_add_pd(t16, t18);
        const __m256d t52 = _mm256_sub_pd(t18, t16);
        const __m256d t19 = lane_im_2;
        const __m256d t20 = _mm256_add_pd(t17, t19);
        const __m256d t50 = _mm256_sub_pd(t19, t17);
        const __m256d t22 = lane_re_4;
        const __m256d t23 = lane_im_4;
        const __m256d t24 = lane_re_0;
        const __m256d t27 = _mm256_add_pd(t22, t24);
        const __m256d t57 = _mm256_sub_pd(t24, t22);
        const __m256d t29 = _mm256_add_pd(t21, t27);
        const __m256d t30 = _mm256_add_pd(t15, t29);
        const __m256d t59 = _mm256_sub_pd(t57, t50);
        const __m256d t74 = _mm256_sub_pd(t27, t21);
        const __m256d t92 = _mm256_add_pd(t50, t57);
        const __m256d t75 = _mm256_sub_pd(t74, t67);
        const __m256d t100 = _mm256_sub_pd(t29, t15);
        const __m256d t115 = _mm256_add_pd(t67, t74);
        const __m256d t130 = _mm256_fmadd_pd(t44, t45, t59);
        const __m256d t134 = _mm256_fmadd_pd(t45, t85, t92);
        const __m256d t138 = _mm256_fnmadd_pd(t44, t45, t59);
        const __m256d t142 = _mm256_fnmadd_pd(t45, t85, t92);
        const __m256d t25 = lane_im_0;
        const __m256d t26 = _mm256_add_pd(t23, t25);
        const __m256d t55 = _mm256_sub_pd(t25, t23);
        const __m256d t28 = _mm256_add_pd(t20, t26);
        const __m256d t31 = _mm256_add_pd(t14, t28);
        const __m256d t58 = _mm256_add_pd(t52, t55);
        const __m256d t72 = _mm256_sub_pd(t26, t20);
        const __m256d t91 = _mm256_sub_pd(t55, t52);
        const __m256d t78 = _mm256_add_pd(t69, t72);
        const __m256d t104 = _mm256_sub_pd(t28, t14);
        const __m256d t119 = _mm256_sub_pd(t72, t69);
        const __m256d t131 = _mm256_fmadd_pd(t45, t47, t58);
        const __m256d t135 = _mm256_fmadd_pd(t45, t88, t91);
        const __m256d t139 = _mm256_fnmadd_pd(t45, t47, t58);
        const __m256d t143 = _mm256_fnmadd_pd(t45, t88, t91);
        const __m256d t61 = _mm256_loadu_pd(&tzqb[8*k2 + 0]);
        const __m256d t63 = _mm256_loadu_pd(&tzqb[8*k2 + 4]);
        const __m256d t132 = _mm256_fnmadd_pd(t131, t63, _mm256_mul_pd(t130, t61));
        const __m256d t133 = _mm256_fmadd_pd(t130, t63, _mm256_mul_pd(t131, t61));
        const __m256d t76 = _mm256_fnmadd_pd(t63, t63, _mm256_mul_pd(t61, t61));
        const __m256d t77 = _mm256_fmadd_pd(t61, t63, _mm256_mul_pd(t63, t61));
        const __m256d t79 = _mm256_fnmadd_pd(t78, t77, _mm256_mul_pd(t75, t76));
        const __m256d t80 = _mm256_fmadd_pd(t75, t77, _mm256_mul_pd(t78, t76));
        const __m256d t94 = _mm256_fnmadd_pd(t77, t63, _mm256_mul_pd(t76, t61));
        const __m256d t95 = _mm256_fmadd_pd(t76, t63, _mm256_mul_pd(t77, t61));
        const __m256d t136 = _mm256_fnmadd_pd(t135, t95, _mm256_mul_pd(t134, t94));
        const __m256d t137 = _mm256_fmadd_pd(t134, t95, _mm256_mul_pd(t135, t94));
        const __m256d t101 = _mm256_fnmadd_pd(t77, t77, _mm256_mul_pd(t76, t76));
        const __m256d t102 = _mm256_fmadd_pd(t76, t77, _mm256_mul_pd(t77, t76));
        const __m256d t105 = _mm256_fnmadd_pd(t104, t102, _mm256_mul_pd(t100, t101));
        const __m256d t106 = _mm256_fmadd_pd(t100, t102, _mm256_mul_pd(t104, t101));
        const __m256d t109 = _mm256_fnmadd_pd(t102, t63, _mm256_mul_pd(t101, t61));
        const __m256d t110 = _mm256_fmadd_pd(t101, t63, _mm256_mul_pd(t102, t61));
        const __m256d t140 = _mm256_fnmadd_pd(t139, t110, _mm256_mul_pd(t138, t109));
        const __m256d t141 = _mm256_fmadd_pd(t138, t110, _mm256_mul_pd(t139, t109));
        const __m256d t116 = _mm256_fnmadd_pd(t102, t77, _mm256_mul_pd(t101, t76));
        const __m256d t117 = _mm256_fmadd_pd(t101, t77, _mm256_mul_pd(t102, t76));
        const __m256d t120 = _mm256_fnmadd_pd(t119, t117, _mm256_mul_pd(t115, t116));
        const __m256d t121 = _mm256_fmadd_pd(t115, t117, _mm256_mul_pd(t119, t116));
        const __m256d t124 = _mm256_fnmadd_pd(t102, t95, _mm256_mul_pd(t101, t94));
        const __m256d t125 = _mm256_fmadd_pd(t101, t95, _mm256_mul_pd(t102, t94));
        const __m256d t144 = _mm256_fnmadd_pd(t143, t125, _mm256_mul_pd(t142, t124));
        const __m256d t145 = _mm256_fmadd_pd(t142, t125, _mm256_mul_pd(t143, t124));
        /* DIRECT record stores (NO TR4): leg q -> SEC[q&3], record 2k'+(q>>2) */
        _mm256_storeu_pd(&p0[rb + 0],  t30);  _mm256_storeu_pd(&p0[rb + 4],  t31);   /* leg 0 */
        _mm256_storeu_pd(&p2[rb + 0],  t132); _mm256_storeu_pd(&p2[rb + 4],  t133);  /* leg 1 */
        _mm256_storeu_pd(&p1[rb + 0],  t79);  _mm256_storeu_pd(&p1[rb + 4],  t80);   /* leg 2 */
        _mm256_storeu_pd(&p3[rb + 0],  t136); _mm256_storeu_pd(&p3[rb + 4],  t137);  /* leg 3 */
        _mm256_storeu_pd(&p0[rb + 8],  t105); _mm256_storeu_pd(&p0[rb + 12], t106);  /* leg 4 */
        _mm256_storeu_pd(&p2[rb + 8],  t140); _mm256_storeu_pd(&p2[rb + 12], t141);  /* leg 5 */
        _mm256_storeu_pd(&p1[rb + 8],  t120); _mm256_storeu_pd(&p1[rb + 12], t121);  /* leg 6 */
        _mm256_storeu_pd(&p3[rb + 8],  t144); _mm256_storeu_pd(&p3[rb + 12], t145);  /* leg 7 */
    }
}

/* ====================================================================== */
/* s0tb — NEW backward ingest: un-turn in the LOAD network. Granule t     */
/* holds positions 4t+m at section SEC[m] (SEC is an involution, so       */
/* loading sections in order 0,2,1,3 yields natural position order);      */
/* 4x4 transpose records -> leg vectors (lanes = radix-4 output digit),   */
/* then radix4_z_s0s_bwd_avx2's EXACT IDFT-4 body + REINT store edge.     */
/* ====================================================================== */
__attribute__((noinline, used, target("avx2,fma")))
static void _vfft_zturn_s0tb_bwd(const double * __restrict plane,
                                 double * __restrict zout, size_t N)
{
    const size_t S = N >> 4, Ls = N >> 2;
    const double * __restrict p0 = plane;           /* section 0 */
    const double * __restrict p1 = plane + 8*S;     /* section 1 */
    const double * __restrict p2 = plane + 16*S;    /* section 2 */
    const double * __restrict p3 = plane + 24*S;    /* section 3 */
    for (size_t t = 0; t < S; t++) {
        const size_t rb = 8*t;
        const size_t cb = 4*t;                      /* position base */
        /* record loads: position 4t+0 <- sec0, +1 <- sec2, +2 <- sec1, +3 <- sec3 */
        const __m256d R0 = _mm256_loadu_pd(&p0[rb + 0]);
        const __m256d I0 = _mm256_loadu_pd(&p0[rb + 4]);
        const __m256d R1 = _mm256_loadu_pd(&p2[rb + 0]);
        const __m256d I1 = _mm256_loadu_pd(&p2[rb + 4]);
        const __m256d R2 = _mm256_loadu_pd(&p1[rb + 0]);
        const __m256d I2 = _mm256_loadu_pd(&p1[rb + 4]);
        const __m256d R3 = _mm256_loadu_pd(&p3[rb + 0]);
        const __m256d I3 = _mm256_loadu_pd(&p3[rb + 4]);
        /* 4x4 transpose: record lanes (output digit) -> leg vectors over positions */
        const __m256d ur0 = _mm256_unpacklo_pd(R0, R1);
        const __m256d ur1 = _mm256_unpackhi_pd(R0, R1);
        const __m256d ur2 = _mm256_unpacklo_pd(R2, R3);
        const __m256d ur3 = _mm256_unpackhi_pd(R2, R3);
        const __m256d lane_re_0 = _mm256_permute2f128_pd(ur0, ur2, 0x20);
        const __m256d lane_re_1 = _mm256_permute2f128_pd(ur1, ur3, 0x20);
        const __m256d lane_re_2 = _mm256_permute2f128_pd(ur0, ur2, 0x31);
        const __m256d lane_re_3 = _mm256_permute2f128_pd(ur1, ur3, 0x31);
        const __m256d ui0 = _mm256_unpacklo_pd(I0, I1);
        const __m256d ui1 = _mm256_unpackhi_pd(I0, I1);
        const __m256d ui2 = _mm256_unpacklo_pd(I2, I3);
        const __m256d ui3 = _mm256_unpackhi_pd(I2, I3);
        const __m256d lane_im_0 = _mm256_permute2f128_pd(ui0, ui2, 0x20);
        const __m256d lane_im_1 = _mm256_permute2f128_pd(ui1, ui3, 0x20);
        const __m256d lane_im_2 = _mm256_permute2f128_pd(ui0, ui2, 0x31);
        const __m256d lane_im_3 = _mm256_permute2f128_pd(ui1, ui3, 0x31);
        /* SU-scheduled body — VERBATIM radix4_z_s0s_bwd_avx2 (IDFT-4 leaf) */
        const __m256d t0 = lane_re_3;
        const __m256d t2 = lane_im_3;
        const __m256d t5 = lane_re_1;
        const __m256d t10 = _mm256_sub_pd(t5, t0);
        const __m256d t24 = _mm256_add_pd(t0, t5);
        const __m256d t6 = lane_im_1;
        const __m256d t8 = _mm256_sub_pd(t6, t2);
        const __m256d t23 = _mm256_add_pd(t2, t6);
        const __m256d t12 = lane_re_2;
        const __m256d t13 = lane_im_2;
        const __m256d t15 = lane_re_0;
        const __m256d t19 = _mm256_sub_pd(t15, t12);
        const __m256d t20 = _mm256_add_pd(t8, t19);
        const __m256d t31 = _mm256_sub_pd(t19, t8);
        const __m256d t27 = _mm256_add_pd(t12, t15);
        const __m256d t28 = _mm256_sub_pd(t27, t24);
        const __m256d t33 = _mm256_add_pd(t24, t27);
        const __m256d t16 = lane_im_0;
        const __m256d t17 = _mm256_sub_pd(t16, t13);
        const __m256d t22 = _mm256_sub_pd(t17, t10);
        const __m256d t32 = _mm256_add_pd(t10, t17);
        const __m256d t26 = _mm256_add_pd(t13, t16);
        const __m256d t30 = _mm256_sub_pd(t26, t23);
        const __m256d t34 = _mm256_add_pd(t23, t26);
        /* Z store edge (REINT) — VERBATIM, column base cb = 4*t */
        const __m256d _pr_0 = _mm256_permute4x64_pd(t33, 0xD8);
        const __m256d _qi_0 = _mm256_permute4x64_pd(t34, 0xD8);
        _mm256_storeu_pd(&zout[2*cb], _mm256_unpacklo_pd(_pr_0, _qi_0));
        _mm256_storeu_pd(&zout[2*cb + 4], _mm256_unpackhi_pd(_pr_0, _qi_0));
        const __m256d _pr_1 = _mm256_permute4x64_pd(t31, 0xD8);
        const __m256d _qi_1 = _mm256_permute4x64_pd(t32, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)1*Ls + cb)], _mm256_unpacklo_pd(_pr_1, _qi_1));
        _mm256_storeu_pd(&zout[2*((size_t)1*Ls + cb) + 4], _mm256_unpackhi_pd(_pr_1, _qi_1));
        const __m256d _pr_2 = _mm256_permute4x64_pd(t28, 0xD8);
        const __m256d _qi_2 = _mm256_permute4x64_pd(t30, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)2*Ls + cb)], _mm256_unpacklo_pd(_pr_2, _qi_2));
        _mm256_storeu_pd(&zout[2*((size_t)2*Ls + cb) + 4], _mm256_unpackhi_pd(_pr_2, _qi_2));
        const __m256d _pr_3 = _mm256_permute4x64_pd(t20, 0xD8);
        const __m256d _qi_3 = _mm256_permute4x64_pd(t22, 0xD8);
        _mm256_storeu_pd(&zout[2*((size_t)3*Ls + cb)], _mm256_unpacklo_pd(_pr_3, _qi_3));
        _mm256_storeu_pd(&zout[2*((size_t)3*Ls + cb) + 4], _mm256_unpackhi_pd(_pr_3, _qi_3));
    }
}

/* ============================ plan ==================================== */

static inline void vfft_zturn_destroy(vfft_zturn_plan_t *p)
{
    if (!p) return;
    for (int s = 0; s < VFFT_ZSPLIT_MAX_NF; s++) {
        VFFT_ZS_FREE(p->twz[s]);
        VFFT_ZS_FREE(p->twzb[s]);
    }
    VFFT_ZS_FREE(p->tzq);
    VFFT_ZS_FREE(p->tzqb);
    VFFT_ZS_FREE(p->plane);
    free(p);
}

/* Chains = the SAME calibrated per-cell winners as production zsplit
 * (vfft_zsplit_default_chain — no invented chains). Tables are the
 * CANONICAL plan-time repack (spec section 3/4; proven table builder,
 * zturn_sim.c lines 308-341): mid tables lane-varying + x4 section-tiled,
 * angle -2*pi*((l*(j+4*brev'(g'))) mod M)/M, M = N/D_s; terminator
 * per-(k',lane) w^1 at angle -2*pi*((j+4*rho(k')) mod N)/N. */
static inline vfft_zturn_plan_t *vfft_zturn_create(int N)
{
    int chain[VFFT_ZSPLIT_MAX_NF];
    const int nf = vfft_zsplit_default_chain(N, chain);
    if (nf < 3 || nf > VFFT_ZSPLIT_MAX_NF) return NULL;
    if (chain[0] != 4) return NULL;    /* ZTURN-S fence: 4-section geometry */
    if (chain[nf - 1] != 8) return NULL;
    long prod = 1;
    for (int s = 0; s < nf; s++) {
        if (s >= 1 && s <= nf - 2 && chain[s] != 4 && chain[s] != 8) return NULL;
        prod *= chain[s];
    }
    if (prod != N || (N / 8) % 4) return NULL;

    vfft_zturn_plan_t *p = (vfft_zturn_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N; p->nf = nf;
    for (int s = 0; s < nf; s++) p->chain[s] = chain[s];
    p->D[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) p->D[i] = p->D[i + 1] * chain[i + 1];
    p->G[0] = 1;
    for (int i = 1; i < nf; i++) p->G[i] = p->G[i - 1] * chain[i - 1];
    if (p->D[nf - 2] % 4) goto fail;   /* asserted, not assumed (spec risk) */

    {
        const double TAU = 2.0 * M_PI;
        for (int s = 1; s <= nf - 2; s++) {
            const int R = chain[s], Rm1 = R - 1;
            const long M = (long)N / p->D[s];          /* = G_{s+1}       */
            const long Gp = p->G[s] / 4;               /* section period  */
            p->twz[s]  = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
            p->twzb[s] = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
            if (!p->twz[s] || !p->twzb[s]) goto fail;
            for (long g = 0; g < p->G[s]; g++) {
                const long g2 = g % Gp;                /* x4 section tiling */
                const long br = _vfft_zs_brev(g2, s - 1, chain + 1);
                for (int l = 1; l < R; l++)
                    for (int j = 0; j < 4; j++) {
                        const double a = -TAU
                            * (double)(((long)l * (j + 4 * br)) % M) / (double)M;
                        double *f = p->twz[s]  + ((size_t)g * Rm1 + (l - 1)) * 8;
                        double *b = p->twzb[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                        f[j] = cos(a); f[4 + j] = sin(a);
                        b[j] = cos(a); b[4 + j] = -sin(a);
                    }
            }
        }
        {
            const long K2 = (long)N / 32;
            p->tzq   = (double *)VFFT_ZS_ALLOC((size_t)K2 * 8 * 8);
            p->tzqb  = (double *)VFFT_ZS_ALLOC((size_t)K2 * 8 * 8);
            p->plane = (double *)VFFT_ZS_ALLOC((size_t)2 * N * 8);
            if (!p->tzq || !p->tzqb || !p->plane) goto fail;
            for (long k2 = 0; k2 < K2; k2++) {
                const long br = _vfft_zs_brev(k2, nf - 2, chain + 1);
                for (int j = 0; j < 4; j++) {
                    const double a = -TAU
                        * (double)((j + 4 * br) % (long)N) / (double)N;
                    p->tzq[8 * k2 + j] = cos(a);  p->tzq[8 * k2 + 4 + j] = sin(a);
                    p->tzqb[8 * k2 + j] = cos(a); p->tzqb[8 * k2 + 4 + j] = -sin(a);
                }
            }
        }
    }
    return p;
fail:
    vfft_zturn_destroy(p);
    return NULL;
}

/* natural z in -> ZTURN-S scrambled comb out (P_fwd, spec section 5).
 * zin == zout OK (terminator is the only writer of zout, reads plane). */
static inline void vfft_zturn_execute_fwd(const vfft_zturn_plan_t *p,
                                          const double *zin, double *zout)
{
    _vfft_zturn_s0t_fwd(zin, p->plane, (size_t)p->N);
    for (int s = 1; s <= p->nf - 2; s++) {
        /* PRODUCTION msg kernels BYTE-FOR-BYTE, production arg tuple */
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2 : radix4_z_msg_fwd_avx2;
        f(p->plane, 0, p->plane, 0, p->twz[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    _vfft_zturn_stf_fwd(p->plane, zout, p->tzq, (size_t)p->N);
}

/* ZTURN-S scrambled comb in -> N * natural z out. zin == zout OK. */
static inline void vfft_zturn_execute_bwd(const vfft_zturn_plan_t *p,
                                          const double *zin, double *zout)
{
    _vfft_zturn_stfb_bwd(zin, p->plane, p->tzqb, (size_t)p->N);
    for (int s = p->nf - 2; s >= 1; s--) {
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_bwd_avx2 : radix4_z_msg_bwd_avx2;
        f(p->plane, 0, p->plane, 0, p->twzb[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    _vfft_zturn_s0tb_bwd(p->plane, zout, (size_t)p->N);
}

#endif /* VFFT_ZTURN_PROTO_H */
