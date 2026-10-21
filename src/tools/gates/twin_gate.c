/* twin_gate.c — the AVX-512 TWIN GATE: does every interleaved (zil) kernel and
 * every ZTURN-T fused driver that both ISAs carry compute at avx512 what its
 * avx2 twin computes, at every column count, and store nothing else?
 *
 * WHAT IT RUNS
 *   - every (kind, radix, direction) that BOTH il_registry_avx2.h and
 *     il_registry_avx512.h list: the avx2 kernel and the avx512 kernel on the
 *     same logical input, with the same logical twiddles, at column counts that
 *     reach the wide body and every tail arm: 1..17, 31, 32, 33, 63, 64, 65, 66,
 *     67 (the per-column-group kinds up to the columns their table holds); the
 *     ZTURN-T stage kernels, whose contract is whole 8-column iterations, at
 *     8, 16, 24, 32, 40, 56, 64, 72;
 *   - the solo tier's mono-64 IL twins (il_solo_registry.h:
 *     vfft_k1_mono64_8x8_il_fwd / _bwd, kind mono64_8x8_il), which no radix
 *     list names: the whole 64-point transform, one call;
 *   - the 14 pair2p/tangent kernels (t2tan, t2tanm128, t2bw32, t2bw32m128,
 *     n1ttan, n1tbw32, n1tbw32t256, t2ttan, n1tan, n1btan216) and the 6
 *     radix-16/32 forward blocked pair kernels (t2b, t2b48, n1tb, n1tb44,
 *     n1tb48), built outside the corpus at both ISAs (recipes.tsv's ship and
 *     replay rows), which no radix list names either: each in the class of
 *     its classic twin;
 *   - every ZTURN-T fused driver BOTH ztt registries list (the same N and
 *     chain, the slot non-null at both): natural dest and plane (in place),
 *     plain scrambled, forward and backward, untiled and at the smallest tile
 *     both ISAs admit.
 * A kind one ISA lists and the other does not is reported, not gated (an
 * empty avx512 list is the registry declaring the kind absent there). A kind
 * both list that has no case here is a FAIL, and so is a radix of such a kind
 * that avx2 lists and avx512 does not, unless il_registry_avx512.h declares
 * it absent (ABSENT512): the gate never skips silently. Each side's kernel
 * table is checked at compile time against the kernels its registry's lists
 * hold (TWIN_REG_NKERN), so a new kind cannot go ungated either.
 *
 * ONE SOURCE, THREE TRANSLATION UNITS. Compiled with -DVFFT_TWIN_SIDE at the
 * avx2 flags and again at the avx512 flags, this file is one ISA's SIDE: it
 * includes il_isa.h and the library's own headers at that ISA and exports
 * VFFT_IL_SYM(twin_side) (twin_side_avx2, twin_side_avx512): its kernel
 * table, read from its registry's radix lists; its ZTURN-T cells; and its
 * twiddle builders. Compiled plainly (the avx2 flags, no AVX-512 code) it is
 * the gate. Each side builds ITS OWN ISA's twiddle records with the
 * library's builders, which are the same calls at both ISAs (il_isa.h):
 *   pair records (VTW2: t2, t2t, t2tg, the blocked and _ct forms, the rows'
 *       t2r/t2tr, t2cs)            _vfft_il3p_vtw2 (il2p.h; vtw_avx512.h at avx512)
 *   per-block broadcast records (t2cp, t2c, t2cb*), splat records (msz,
 *       mszt) and the gen2 bases (t2csg, t2csgt, t2csgn, t2csgnt, and the
 *       Hermitian last stage's t2csgh, t2csght)
 *                                  vfft_ilfd_create_chain (il_flatdit.h): the
 *       tables of a flat DIT plan built at that ISA — chain {4, R, 5} (stage 1
 *       is t2cp and msz), {64, R} (stage 1 is t2csg over 64-column groups, one
 *       group) and {4, 47, R} (t2csg over 47-column groups, four groups)
 *   lane records (tmg, tmgd, tlf, tlfi, t0d)       _ztt_fill_stage (ztt.h)
 *   the drivers' streams, run-base table and the plain class's output
 *       permutation, and tld's output positions     _ztt_create (ztt.h)
 * A broadcast record holds one twiddle in every slot, so a table is a pool
 * of records: radices 32 and 64 have no t2cp stage, and their t2c/t2cb*
 * kernels read the records of the {8, 47, 5} plan's radix-47 stage — the
 * same twiddle in record i at both ISAs.
 *
 * THE LAYOUTS. A buffer is LOGICAL complex slots, and each ISA lays them out
 * its way: natural interleaved; the ZTURN-T plane's block split, 64-B blocks
 * of [re x VW][im x VW] (VW = 4 avx2, 8 avx512: the codelet headers'
 * contract); or a position map (the plain class's scrambled output and tld's:
 * the ISA's own permutation, from its plan). The real mono's real side (rn1,
 * rn1b48, rn1b444: the forward's input, the backward's output) is LOGICAL real
 * slots, one double each, natural at both ISAs; so are both sides of the real
 * flat leaf r1c (its real legs, and its digit runs: digit 0 a real run, digits
 * 1..h complex, every double a slot). The Hermitian last stage (t2csgh,
 * t2csght) puts a leg above R/2 conjugated at the mirror NM - bin and the
 * middle leg by its bin: its spectrum holds the direct bins below U and the
 * mirrored ones above them (NM = 2U), the mirror base passed at the call.
 * Input values go in by the ISA's layout, outputs come out by it, and the
 * comparison is slot by slot.
 *
 * WHAT IT CHECKS, per case and per ISA:
 *   1. every output slot in the kernel's footprint is written and finite —
 *      the output starts as a canary (a quiet NaN no arithmetic makes), the
 *      input slots outside the footprint and the bands around every input
 *      and twiddle table hold a second NaN, so a missed store reads the
 *      canary and a read outside the input or a table yields a NaN;
 *   2. nothing outside the footprint is stored: the gaps the strides leave
 *      between columns, legs, rows and groups, and the bands on both sides
 *      of the buffer (as long as the buffer, at least 4 KB: t_band), still
 *      hold the canary;
 *   3. an out-of-place kernel leaves its input as it was;
 *   4. a ZTURN-T driver's plane is the gate's own (2N doubles, banded the
 *      same way, POISON): a plane driver may use those 2N doubles and leaves
 *      the bands as they were (ztt.h's 4 KB of slack is where the plan
 *      places the plane, not room to store); a dest driver leaves all of it;
 *   5. a case that works in the buffer it stores into — in place, and every
 *      ZTURN-T driver (dest and scrambled run their pipeline in zout) — runs
 *      a second time with every double of that buffer outside the footprint,
 *      its bands, and the plane holding a finite value of its own (t_pat):
 *      there a store of arithmetic on a NaN would carry the NaN through bit
 *      for bit and pass check 2, here it changes the value. That run's
 *      output must equal the first's bitwise, and those doubles must hold;
 *   6. avx512 == avx2: max |y512 - y2| / max |y2| over the footprint, at most
 *      TWIN_TOL. The comparison is bitwise first; a case that is not bitwise
 *      counts toward the kind's worst error.
 *
 * TWIN_TOL (8 * DBL_EPSILON, 1.8e-15). Both ISAs evaluate the same DAG on the
 * same twiddles (vtw_avx512.h: every angle is the AVX2 builder's), so 19624
 * of the 20739 kernel cases match bitwise, the ZTURN-T stage kernels and
 * fused drivers all of theirs. The other kernel cases differ by at most
 * 4.7e-16 (2.1 DBL_EPSILON), and they are exactly two groups: radix 5 and
 * 10 at every count (n1, n1c, n1t, t2, t2t, t2tg, t2c, t2cp, t2cs: the avx2
 * files carry the older radix-5 FMA arithmetic, decision D0), and the
 * blocked kernels (n1b*, n1cb*, n1tb*, t2b*, t2bt*, t2cb*) at the counts that
 * end in a tail (the avx2 tail re-runs the blocked passes, the avx512 tail is
 * the monolithic DAG). A wrong twiddle, leg, lane, layout or permutation
 * moves the result by 0.7 and more.
 *
 * RUNS ONLY WHERE AVX-512 RUNS: a host with avx512f, avx512dq and avx512vl,
 * or Intel SDE emulating one (exact AVX-512 semantics, 10-100x slower; for
 * correctness only, its timings mean nothing):
 *   sde.exe -spr -- src/tools/gates/twin_gate.exe
 * Without AVX-512 it says so and exits 2 before any avx512 code runs.
 *
 * Options: --kind k1,k2,..   only these kinds ("ztt" = the fused drivers)
 *          --ztt-max N       the fused drivers of N <= this only
 * No wisdom store: the gate creates no plan through the front door.
 *
 * Build (from the repo root; run_gates.py runs the same script):
 *   python src/tools/gates/twin_gate_build.py
 * which is, with gauntlet/build.py's driver flags and -I set, and n2 / n5 the
 * kernels each registry's FWD and BWD lists hold (the script counts them):
 *   gcc <avx2 flags>   -c twin_gate.c -o main.o              (the gate)
 *   gcc <avx2 flags>   -DVFFT_TWIN_SIDE -DTWIN_REG_NKERN=n2 -c twin_gate.c -o side_avx2.o
 *   gcc <avx512 flags> -DVFFT_TWIN_SIDE -DTWIN_REG_NKERN=n5 -c twin_gate.c -o side_avx512.o
 *   gcc main.o side_avx2.o side_avx512.o \
 *       src/dag-fft-compiler/.obj/avx2/libdagcodelets.a \
 *       src/dag-fft-compiler/.obj/avx512/libdagcodelets.a -lm -o twin_gate.exe
 * the two codelet libraries being build.py's own (VFFT_ISA=avx2 and
 * VFFT_ISA=avx512). */
#include <stddef.h>
#include <stdint.h>

/* ── the surface both sides export ─────────────────────────────────────── */
typedef void (*twin_kfn)(const double *, const double *, double *, double *,
                         const double *, const double *,
                         size_t, size_t, size_t, size_t, size_t);
typedef void (*twin_ztt_fn)(const double *, double *, double *, const double *,
                            const size_t *, size_t);
typedef void (*twin_zttp_fn)(const double *, double *, const double *, size_t);

typedef struct { const char *kind; int R, bwd; twin_kfn fn; } twin_kernel_t;

typedef struct {
    int n, nf, chain[7];
    twin_ztt_fn fwd_dest, fwd_plane, bwd_dest, bwd_plane;
    twin_zttp_fn fwd_scr, bwd_scr;
} twin_cell_t;

/* the tables of one stage of a flat DIT plan, and their sizes in doubles */
typedef struct {
    void *plan;
    const double *tf, *tfb;   /* t2cp: per-block broadcast records; t2csg: T1 */
    const double *t2, *t2b;   /* t2csg: T2, nbase records per group */
    const double *tz, *tzb;   /* msz: per-block splat records */
    size_t ntf, nt2, ntz;
    int nb, G, ngrp, nbase;
} twin_flat_t;

/* a ZTURN-T plan's tables and its cell's drivers (the plane is the gate's own) */
typedef struct {
    void *plan;
    const double *tw, *twb;
    size_t ntw;
    const size_t *rb, *perm;
    twin_ztt_fn fwd_dest, fwd_plane, bwd_dest, bwd_plane;
    twin_zttp_fn fwd_scr, bwd_scr;
} twin_ztt_t;

typedef struct {
    const char *isa;
    int vw;
    int nkern;
    const twin_kernel_t *kern;
    int ncell;
    const twin_cell_t *cell;
    double *(*pair)(int R, int blocks, int cols, int modulus, int conj, size_t *nd);
    double *(*lane)(long L, int R, long RL, int bwd, size_t *nd);
    int (*flat)(const int *R, int K, int s, twin_flat_t *t);
    void (*flat_free)(twin_flat_t *t);
    int (*ztt)(int N, const int *chain, int nf, int scr, int staged, twin_ztt_t *z);
    void (*ztt_free)(twin_ztt_t *z);
    int (*tile_ok)(int N, const int *chain, int nf, size_t tile, int scr);
    size_t *(*scr_pos)(int N, const int *chain, int nf);
    void (*free_table)(void *p);
} twin_side_t;

#if defined(VFFT_TWIN_SIDE)
/* ═════════════════════ ONE ISA'S SIDE ═════════════════════════════════════ */
#include <stdlib.h>
#include <string.h>
#include "il_isa.h"
#include "il_solo_registry.h"
#include "il2p.h"
#include "il_flatdit.h"
#include "ztt.h"

/* every kernel the registry lists, from its radix lists: the emitter defines
 * the same lists at every ISA, empty where the ISA has no kernel
 * (emit_il_registry.ml), so this table is the registry's, whole; then the
 * pair2p/tangent kernels and the radix-16/32 forward blocked pair kernels
 * (il2p.h declares them at this ISA) and the solo tier's mono-64 IL twins
 * (il_solo_registry.h), which no list names.
 * TWIN_REG_NKERN, the kernels the registry's FWD and BWD lists hold at this
 * side's ISA, comes from twin_gate_build.py: the table must expand exactly
 * those, so a kind the registry gains and the table does not name stops the
 * compile instead of going ungated. */
#ifndef TWIN_REG_NKERN
#error "TWIN_REG_NKERN (the kernels the registry's FWD and BWD lists hold) is unset: build with twin_gate_build.py"
#endif
#define TWIN_NTAN 14
#define TWIN_NBLK 6
#define TWIN_NSOLO 2
#define _TWIN_K2(kind, dir, bwd, R) \
    { #kind, R, bwd, (twin_kfn)VFFT_IL_SYM(radix##R##_z_##kind##_##dir) },
#define _TWIN_K(kind, dir, bwd, R) _TWIN_K2(kind, dir, bwd, R)
#define TWIN_FX(R) _TWIN_K(TWIN_KIND, fwd, 0, R)
#define TWIN_BX(R) _TWIN_K(TWIN_KIND, bwd, 1, R)
static const twin_kernel_t _twin_kern[] = {
#define TWIN_KIND msz
    VFFT_IL_MSZ_FWD_RADICES(TWIN_FX) VFFT_IL_MSZ_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND mszt
    VFFT_IL_MSZT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1
    VFFT_IL_N1_FWD_RADICES(TWIN_FX) VFFT_IL_N1_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1b216
    VFFT_IL_N1B216_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1b416
    VFFT_IL_N1B416_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1b48
    VFFT_IL_N1B48_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1b88
    VFFT_IL_N1B88_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1c
    VFFT_IL_N1C_FWD_RADICES(TWIN_FX) VFFT_IL_N1C_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1cb448
    VFFT_IL_N1CB448_FWD_RADICES(TWIN_FX) VFFT_IL_N1CB448_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1cb48
    VFFT_IL_N1CB48_FWD_RADICES(TWIN_FX) VFFT_IL_N1CB48_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1cb48h
    VFFT_IL_N1CB48H_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1cb816
    VFFT_IL_N1CB816_FWD_RADICES(TWIN_FX) VFFT_IL_N1CB816_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1cb84
    VFFT_IL_N1CB84_FWD_RADICES(TWIN_FX) VFFT_IL_N1CB84_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1cb84h
    VFFT_IL_N1CB84H_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1cb88
    VFFT_IL_N1CB88_FWD_RADICES(TWIN_FX) VFFT_IL_N1CB88_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1cb88h
    VFFT_IL_N1CB88H_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1ccs
    VFFT_IL_N1CCS_FWD_RADICES(TWIN_FX) VFFT_IL_N1CCS_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1ch
    VFFT_IL_N1CH_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1r
    VFFT_IL_N1R_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1rtan
    VFFT_IL_N1RTAN_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1t
    VFFT_IL_N1T_FWD_RADICES(TWIN_FX) VFFT_IL_N1T_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1tb416
    VFFT_IL_N1TB416_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1tb88
    VFFT_IL_N1TB88_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1tr
    VFFT_IL_N1TR_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1trtan
    VFFT_IL_N1TRTAN_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND n1t_ct
    VFFT_IL_N1T_CT_FWD_RADICES(TWIN_FX) VFFT_IL_N1T_CT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND n1_ct
    VFFT_IL_N1_CT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND r1c
    VFFT_IL_R1C_FWD_RADICES(TWIN_FX) VFFT_IL_R1C_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND r2z
    VFFT_IL_R2Z_FWD_RADICES(TWIN_FX) VFFT_IL_R2Z_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND r2zr
    VFFT_IL_R2ZR_FWD_RADICES(TWIN_FX) VFFT_IL_R2ZR_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND rn1
    VFFT_IL_RN1_FWD_RADICES(TWIN_FX) VFFT_IL_RN1_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND rn1b444
    VFFT_IL_RN1B444_FWD_RADICES(TWIN_FX) VFFT_IL_RN1B444_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND rn1b48
    VFFT_IL_RN1B48_FWD_RADICES(TWIN_FX) VFFT_IL_RN1B48_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t0d
    VFFT_IL_T0D_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t0tp
    VFFT_IL_T0TP_FWD_RADICES(TWIN_FX) VFFT_IL_T0TP_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2
    VFFT_IL_T2_FWD_RADICES(TWIN_FX) VFFT_IL_T2_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2b416
    VFFT_IL_T2B416_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t2b88
    VFFT_IL_T2B88_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t2bt216
    VFFT_IL_T2BT216_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2bt416
    VFFT_IL_T2BT416_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2bt48
    VFFT_IL_T2BT48_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2bt88
    VFFT_IL_T2BT88_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2c
    VFFT_IL_T2C_FWD_RADICES(TWIN_FX) VFFT_IL_T2C_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2cb44
    VFFT_IL_T2CB44_FWD_RADICES(TWIN_FX) VFFT_IL_T2CB44_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2cb48
    VFFT_IL_T2CB48_FWD_RADICES(TWIN_FX) VFFT_IL_T2CB48_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2cb84
    VFFT_IL_T2CB84_FWD_RADICES(TWIN_FX) VFFT_IL_T2CB84_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2cb88
    VFFT_IL_T2CB88_FWD_RADICES(TWIN_FX) VFFT_IL_T2CB88_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2cp
    VFFT_IL_T2CP_FWD_RADICES(TWIN_FX) VFFT_IL_T2CP_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2cs
    VFFT_IL_T2CS_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t2csg
    VFFT_IL_T2CSG_FWD_RADICES(TWIN_FX) VFFT_IL_T2CSG_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2csgh
    VFFT_IL_T2CSGH_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t2csght
    VFFT_IL_T2CSGHT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2csgn
    VFFT_IL_T2CSGN_FWD_RADICES(TWIN_FX) VFFT_IL_T2CSGN_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2csgnt
    VFFT_IL_T2CSGNT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2csgt
    VFFT_IL_T2CSGT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2h
    VFFT_IL_T2H_FWD_RADICES(TWIN_FX) VFFT_IL_T2H_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2m
    VFFT_IL_T2M_FWD_RADICES(TWIN_FX) VFFT_IL_T2M_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2r
    VFFT_IL_T2R_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t2rtan
    VFFT_IL_T2RTAN_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
#define TWIN_KIND t2t
    VFFT_IL_T2T_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2tg
    VFFT_IL_T2TG_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2tr
    VFFT_IL_T2TR_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2trtan
    VFFT_IL_T2TRTAN_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2t_ct
    VFFT_IL_T2T_CT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND t2_ct
    VFFT_IL_T2_CT_FWD_RADICES(TWIN_FX) VFFT_IL_T2_CT_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND tld
    VFFT_IL_TLD_FWD_RADICES(TWIN_FX) VFFT_IL_TLD_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND tlf
    VFFT_IL_TLF_FWD_RADICES(TWIN_FX) VFFT_IL_TLF_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND tlfi
    VFFT_IL_TLFI_FWD_RADICES(TWIN_FX) VFFT_IL_TLFI_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND tmg
    VFFT_IL_TMG_FWD_RADICES(TWIN_FX) VFFT_IL_TMG_BWD_RADICES(TWIN_BX)
#undef TWIN_KIND
#define TWIN_KIND tmgd
    VFFT_IL_TMGD_FWD_RADICES(TWIN_FX)
#undef TWIN_KIND
    _TWIN_K(t2tan, fwd, 0, 8) _TWIN_K(t2tan, fwd, 0, 16) _TWIN_K(t2tanm128, fwd, 0, 16)
    _TWIN_K(t2bw32, fwd, 0, 32) _TWIN_K(t2bw32m128, fwd, 0, 32)
    _TWIN_K(n1ttan, fwd, 0, 8) _TWIN_K(n1ttan, fwd, 0, 16)
    _TWIN_K(n1tbw32, fwd, 0, 32) _TWIN_K(n1tbw32t256, fwd, 0, 32)
    _TWIN_K(t2ttan, bwd, 1, 8) _TWIN_K(t2ttan, bwd, 1, 16)
    _TWIN_K(n1tan, bwd, 1, 8) _TWIN_K(n1tan, bwd, 1, 16) _TWIN_K(n1btan216, bwd, 1, 32)
    _TWIN_K(t2b, fwd, 0, 16) _TWIN_K(t2b, fwd, 0, 32) _TWIN_K(t2b48, fwd, 0, 32)
    _TWIN_K(n1tb, fwd, 0, 32) _TWIN_K(n1tb44, fwd, 0, 16) _TWIN_K(n1tb48, fwd, 0, 32)
    { "mono64_8x8_il", 64, 0, (twin_kfn)VFFT_IL_SYM(vfft_k1_mono64_8x8_il_fwd) },
    { "mono64_8x8_il", 64, 1, (twin_kfn)VFFT_IL_SYM(vfft_k1_mono64_8x8_il_bwd) },
    { NULL, 0, 0, NULL }
};
_Static_assert(sizeof _twin_kern / sizeof _twin_kern[0] == (size_t)TWIN_REG_NKERN + TWIN_NTAN + TWIN_NBLK + TWIN_NSOLO + 1,
               "the kernel table does not expand every kernel the registry's FWD and BWD lists hold: "
               "a kind is missing from it (or listed twice)");

static twin_cell_t _twin_cells[VFFT_ZTT_NCELLS];

static double *_twin_pair(int R, int blocks, int cols, int modulus, int conj, size_t *nd)
{
    *nd = (size_t)blocks * _vfft_il3p_vtw2_recs(cols) * (size_t)(R - 1) * VFFT_IL_TWREC;
    return _vfft_il3p_vtw2(R, blocks, cols, modulus, conj);
}

static double *_twin_lane(long L, int R, long RL, int bwd, size_t *nd)
{
    double *t;
    *nd = (size_t)2 * (size_t)(R - 1) * (size_t)L;
    t = (double *)vfft_aligned_alloc(*nd * sizeof(double));
    if (t) _ztt_fill_stage(t, L, R, RL, bwd);
    return t;
}

/* stage s of the plan over chain R: which tables it carries is the create's
 * choice (t2cp and msz on a run above VFFT_ILFD_TAIL_D, t2csg at or below) */
static int _twin_flat(const int *R, int K, int s, twin_flat_t *t)
{
    vfft_ilfd_plan_t *p;
    int N = 1, k;
    memset(t, 0, sizeof *t);
    for (k = 0; k < K; k++) N *= R[k];
    p = vfft_ilfd_create_chain(N, R, K);
    if (!p) return 0;
    t->plan = p;
    t->nb = (int)p->nblk[s];
    if (p->tz[s]) {
        t->tz = p->tz[s];
        t->tzb = p->tzb[s];
        t->ntz = (size_t)t->nb * (size_t)(R[s] - 1) * VFFT_IL_TWREC;
    }
    if (p->tail[s] == 2) {
        t->G = R[s - 1];
        t->ngrp = t->nb / t->G;
        t->nbase = p->t2nb[s];
        t->tf = p->tf[s];
        t->tfb = p->tfb[s];
        t->ntf = (size_t)((t->G + VFFT_IL_TWPER - 1) / VFFT_IL_TWPER) * (size_t)t->nbase * VFFT_IL_TWREC;
        t->t2 = p->t2g[s];
        t->t2b = p->t2gb[s];
        t->nt2 = (size_t)t->ngrp * (size_t)t->nbase * VFFT_IL_TWREC;
    } else if (p->tail[s] == 0 && p->tf[s]) {
        t->tf = p->tf[s];
        t->tfb = p->tfb[s];
        t->ntf = (size_t)t->nb * (size_t)(R[s] - 1) * VFFT_IL_TWREC;
    }
    return 1;
}

static void _twin_flat_free(twin_flat_t *t)
{
    vfft_ilfd_destroy((vfft_ilfd_plan_t *)t->plan);
    t->plan = NULL;
}

static int _twin_ztt(int N, const int *chain, int nf, int scr, int staged, twin_ztt_t *z)
{
    vfft_ztt_plan_t *p = _ztt_create(N, chain, nf, scr, staged);
    memset(z, 0, sizeof *z);
    if (!p) return 0;
    z->plan = p;
    z->tw = p->tw;
    z->twb = p->twb;
    z->ntw = p->twdoubles;
    z->rb = p->rb;
    z->perm = p->perm;
    if (p->cell) {
        z->fwd_dest = (twin_ztt_fn)p->cell->fwd_dest;
        z->fwd_plane = (twin_ztt_fn)p->cell->fwd_plane;
        z->bwd_dest = (twin_ztt_fn)p->cell->bwd_dest;
        z->bwd_plane = (twin_ztt_fn)p->cell->bwd_plane;
        z->fwd_scr = (twin_zttp_fn)p->cell->fwd_scr;
        z->bwd_scr = (twin_zttp_fn)p->cell->bwd_scr;
    }
    return 1;
}

static void _twin_ztt_free(twin_ztt_t *z)
{
    vfft_ztt_destroy((vfft_ztt_plan_t *)z->plan);
    z->plan = NULL;
}

static int _twin_tile_ok(int N, const int *chain, int nf, size_t tile, int scr)
{
    return vfft_ztt_tile_legal_ord(N, chain, nf, tile, scr);
}

/* the plain class's output position of DIF slot ic = col * R_last + leg (the
 * slot tld writes), from the plan's frequency -> position table */
static size_t *_twin_scr_pos(int N, const int *chain, int nf)
{
    vfft_ztt_plan_t *p = _ztt_create(N, chain, nf, 1, 1);
    size_t *pos;
    long j;
    if (!p) return NULL;
    pos = (size_t *)malloc((size_t)N * sizeof *pos);
    if (pos)
        for (j = 0; j < (long)N; j++) pos[_ztt_digitrev(j, chain, nf)] = p->perm[j];
    vfft_ztt_destroy(p);
    return pos;
}

static void _twin_free_table(void *p) { vfft_aligned_free(p); }

const twin_side_t *VFFT_IL_SYM(twin_side)(void)
{
    static twin_side_t s;
    static int init;
    if (!init) {
        int i, j;
        for (i = 0; i < VFFT_ZTT_NCELLS; i++) {
            const vfft_ztt_cell_t *c = &vfft_ztt_cells[i];
            twin_cell_t *d = &_twin_cells[i];
            d->n = c->n;
            d->nf = c->nf;
            for (j = 0; j < 7; j++) d->chain[j] = c->chain[j];
            d->fwd_dest = (twin_ztt_fn)c->fwd_dest;
            d->fwd_plane = (twin_ztt_fn)c->fwd_plane;
            d->bwd_dest = (twin_ztt_fn)c->bwd_dest;
            d->bwd_plane = (twin_ztt_fn)c->bwd_plane;
            d->fwd_scr = (twin_zttp_fn)c->fwd_scr;
            d->bwd_scr = (twin_zttp_fn)c->bwd_scr;
        }
        s.isa = VFFT_IL_ISA_NAME;
        s.vw = VFFT_IL_VW;
        s.nkern = (int)(sizeof _twin_kern / sizeof _twin_kern[0]) - 1;
        s.kern = _twin_kern;
        s.ncell = VFFT_ZTT_NCELLS;
        s.cell = _twin_cells;
        s.pair = _twin_pair;
        s.lane = _twin_lane;
        s.flat = _twin_flat;
        s.flat_free = _twin_flat_free;
        s.ztt = _twin_ztt;
        s.ztt_free = _twin_ztt_free;
        s.tile_ok = _twin_tile_ok;
        s.scr_pos = _twin_scr_pos;
        s.free_table = _twin_free_table;
        init = 1;
    }
    return &s;
}

#else
/* ═════════════════════ THE GATE ═══════════════════════════════════════════ */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const twin_side_t *twin_side_avx2(void);
extern const twin_side_t *twin_side_avx512(void);

#define TWIN_TOL (8.0 * DBL_EPSILON)
#define GUARD 512                     /* doubles (4 KB): the least band around a buffer (t_band), and
                                       * the band before a table */
#define TGUARD 2048                   /* doubles of POISON after a table: a whole record set past
                                       * its end (radix 64: 63 x 16) reads NaN, not unmapped memory */
#define CANARY_BITS 0x7ff8dead5eedbeefULL   /* output prefill */
#define POISON_BITS 0x7ff80bad0bad0badULL   /* outside an input's footprint, around a table */
#define TWIN_MAXFAIL 6                /* FAIL lines printed per kind */

static const twin_side_t *S[2];       /* 0 = avx2, 1 = avx512 */

/* ── buffers ───────────────────────────────────────────────────────────── */
static double *t_alloc(size_t nd, double **base)
{   /* 64-B aligned, freed through *base */
    char *p = (char *)malloc(nd * sizeof(double) + 64 + sizeof(void *));
    double *a;
    if (!p) { fprintf(stderr, "twin_gate: out of memory\n"); exit(3); }
    a = (double *)(((uintptr_t)(p + sizeof(void *)) + 63) & ~(uintptr_t)63);
    ((void **)a)[-1] = p;
    *base = a;
    return a;
}
static void t_free(double *base) { if (base) free(((void **)base)[-1]); }
static void t_fill(double *p, size_t n, uint64_t bits)
{
    size_t i;
    for (i = 0; i < n; i++) memcpy(&p[i], &bits, 8);
}
static int t_is(double v, uint64_t bits) { uint64_t u; memcpy(&u, &v, 8); return u == bits; }

/* the band on each side of a buffer the kernel reads or stores (and of a
 * driver's plane): as long as the buffer, at least GUARD, whole 64-B lines.
 * GUARD is the 4 KB of slack ztt.h gives the plane: a store into slack
 * corrupts nothing a test through the plan sees, so the band must hold it;
 * and a store past either end by up to the buffer's own length (one more
 * column, leg, row, group, digit or run, a doubled stride) lands in it */
static size_t t_band(size_t nd)
{
    const size_t b = (nd + 7) & ~(size_t)7;
    return b > GUARD ? b : GUARD;
}

/* the second fill (run_case): a finite value of its own at every double of
 * the store buffer outside the footprint, of its bands and of the plane. A
 * store there of arithmetic on what it held cannot leave it as it was, where
 * a NaN carries through bit for bit; a read of it moves the output. */
static double t_pat(size_t j) { return 1.25 + (double)((j * 2654435761u) % 65521u) / 65521.0; }
static void t_prefill(double *p, size_t n, int fill, uint64_t bits)
{
    size_t j;
    if (fill) for (j = 0; j < n; j++) p[j] = t_pat(j);
    else t_fill(p, n, bits);
}
static int t_kept(const double *p, size_t j, int fill, uint64_t bits)
{
    double v;
    if (!fill) return t_is(p[j], bits);
    v = t_pat(j);
    return !memcmp(&p[j], &v, 8);
}

/* a table copied between two POISON bands: a read past either end is a NaN */
typedef struct { double *base; const double *p; } t_table_t;
static t_table_t t_guarded(const double *src, size_t nd)
{
    t_table_t t;
    double *a = t_alloc(GUARD + nd + TGUARD, &t.base);
    t_fill(a, GUARD + nd + TGUARD, POISON_BITS);
    if (src && nd) memcpy(a + GUARD, src, nd * sizeof(double));
    t.p = a + GUARD;
    return t;
}

/* ── logical buffers and each ISA's layout of them ───────────────────── */
enum { L_NAT, L_BSPLIT, L_MAP, L_REAL };   /* L_REAL: a slot is one double (the real mono's real side) */
typedef struct {
    size_t n;                 /* logical complex slots (L_REAL: real slots) */
    int lay;
    unsigned char *fp;        /* the kernel's footprint */
    size_t *map[2];           /* L_MAP: slot -> complex position, per ISA */
} t_buf_t;

static void b_init(t_buf_t *b, size_t n, int lay)
{
    b->n = n;
    b->lay = lay;
    b->fp = (unsigned char *)calloc(n ? n : 1, 1);
    b->map[0] = b->map[1] = NULL;
}
static void b_free(t_buf_t *b)
{
    free(b->fp);
    free(b->map[0]);
    free(b->map[1]);
    memset(b, 0, sizeof *b);
}
static void b_all(t_buf_t *b) { memset(b->fp, 1, b->n); }
static size_t b_re(const t_buf_t *b, int isa, size_t i)
{
    const size_t vw = (size_t)S[isa]->vw;
    if (b->lay == L_BSPLIT) return 2 * (i - i % vw) + i % vw;
    if (b->lay == L_MAP) return 2 * b->map[isa][i];
    if (b->lay == L_REAL) return i;
    return 2 * i;
}
static size_t b_im(const t_buf_t *b, int isa) { return b->lay == L_BSPLIT ? (size_t)S[isa]->vw : 1; }
/* the doubles a buffer's slots span */
static size_t b_nd(const t_buf_t *b) { return b->lay == L_REAL ? b->n : 2 * b->n; }

/* ── one case ──────────────────────────────────────────────────────────── */
enum { CALL_K, CALL_ZTT, CALL_ZTTP };
typedef struct {
    t_buf_t in, out;
    int inplace, call;
    twin_kfn k[2];
    twin_ztt_fn zf[2];
    twin_zttp_fn zp[2];
    const double *tw[2], *t2[2];
    const void *a1, *a3;          /* t2csgn's base table and group order */
    size_t mirror;                /* the Hermitian kinds: the mirror NM in complex slots, its base
                                   * (zout_unused) NM past the spectrum: out (fwd), in (mirror_in) */
    int mirror_in;
    const size_t *rb[2];          /* a driver's run-base table (t0tp's rides in t2) */
    size_t plane;                 /* CALL_ZTT: the plane's complex slots (the gate's own, banded) */
    int inplane;                  /* 1: the driver works in the plane (the plane slots); 0: it
                                   * leaves all of it as it was (dest: the pipeline runs in zout) */
    size_t Ls, Gs, OLs, OGs, count, tile;
} t_case_t;

/* per ISA: run, check, extract the footprint's values into y (2 per slot).
 * fill 0, every case's run: the bands and the slots outside the footprint
 * hold NaNs (the canary where the kernel stores, POISON where it only reads;
 * the plane POISON). fill 1, the second run of a case that works in its
 * output (run_case): the buffer it stores into, its bands and the plane hold
 * t_pat; an out-of-place input stays POISON. Returns 0, or 1 with the first
 * problem in why. */
static int run_isa(const t_case_t *c, int isa, int fill, const double *x, double *y, char *why, size_t whyn)
{
    const size_t nin2 = b_nd(&c->in), nout2 = b_nd(&c->out), np2 = 2 * c->plane;
    const size_t len = c->inplace && nin2 > nout2 ? nin2 : nout2;   /* the buffer it stores into */
    const size_t gin = t_band(nin2), gout = t_band(len), gp = t_band(np2);
    const size_t ntot = gout + len + gout, itot = gin + nin2 + gin, ptot = gp + np2 + gp;
    double *ob, *ib = NULL, *pb = NULL, *icopy = NULL, *obase, *ibase = NULL, *pbase = NULL, *cbase = NULL;
    double *in, *out;
    unsigned char *om = (unsigned char *)calloc(len ? len : 1, 1);   /* the output footprint, per double */
    size_t i, j, nunw = 0, nnan = 0, nstray = 0, nguard = 0, nplane = 0, nwrote = 0;
    const size_t imi = b_im(&c->in, isa), imo = b_im(&c->out, isa);
    int bad = 0;

    ob = t_alloc(ntot, &obase);
    t_prefill(ob, ntot, fill, CANARY_BITS);
    out = ob + gout;
    if (c->inplace) {
        in = out;
    } else {
        ib = t_alloc(itot, &ibase);
        t_fill(ib, itot, POISON_BITS);
        in = ib + gin;
    }
    for (i = 0; i < c->in.n; i++)
        if (c->in.fp[i]) {
            const size_t r = b_re(&c->in, isa, i);
            in[r] = x[2 * i];
            if (c->in.lay != L_REAL) in[r + imi] = x[2 * i + 1];
        }
    if (!c->inplace) {
        icopy = t_alloc(itot, &cbase);
        memcpy(icopy, ib, itot * sizeof(double));
    }
    if (np2) {
        pb = t_alloc(ptot, &pbase);
        t_prefill(pb, ptot, fill, POISON_BITS);
    }
    for (i = 0; i < c->out.n; i++)
        if (c->out.fp[i]) {
            const size_t r = b_re(&c->out, isa, i);
            om[r] = 1;
            if (c->out.lay != L_REAL) om[r + imo] = 1;
        }

    if (c->call == CALL_K)
        c->k[isa](in, (const double *)c->a1, out,
                  c->mirror ? (c->mirror_in ? in : out) + 2 * c->mirror : (double *)c->a3, c->tw[isa], c->t2[isa],
                  c->Ls, c->Gs, c->OLs, c->OGs, c->count);
    else if (c->call == CALL_ZTT)
        c->zf[isa](in, out, pb + gp, c->tw[isa], c->rb[isa], c->tile);
    else
        c->zp[isa](in, out, c->tw[isa], c->tile);

    /* every double of the buffer it stores into, and of its bands, that is not
     * an output still holds its fill; the plane's bands (a plane driver) or all
     * of it (a dest driver) too; an out-of-place call's input is as it was */
    for (j = 0; j < ntot; j++) {
        const int band = j < gout || j >= gout + len;
        if ((band || !om[j - gout]) && !t_kept(ob, j, fill, CANARY_BITS)) {
            if (band) nguard++;
            else nstray++;
        }
    }
    for (j = 0; j < (np2 ? ptot : 0); j++)
        if ((!c->inplane || j < gp || j >= gp + np2) && !t_kept(pb, j, fill, POISON_BITS)) nplane++;
    if (!c->inplace)
        nwrote = memcmp(icopy, ib, itot * sizeof(double)) != 0;
    for (i = 0; i < c->out.n; i++)
        if (c->out.fp[i]) {
            const size_t r = b_re(&c->out, isa, i);
            const double re = out[r], im = c->out.lay == L_REAL ? 0.0 : out[r + imo];
            if (t_is(re, CANARY_BITS) || t_is(im, CANARY_BITS)) nunw++;
            else if (!isfinite(re) || !isfinite(im)) nnan++;
            y[2 * i] = re;
            y[2 * i + 1] = im;
        }
    if (nunw || nnan || nstray || nguard || nplane || nwrote) {
        snprintf(why, whyn, "%s%s: %zu unwritten, %zu non-finite, %zu doubles stored outside the footprint, "
                 "%zu in its bands, %zu in the plane%s", S[isa]->isa, fill ? " (finite fill)" : "",
                 nunw, nnan, nstray, nguard, nplane, nwrote ? ", wrote its input" : "");
        bad = 1;
    }
    free(om);
    t_free(obase);
    t_free(ibase);
    t_free(pbase);
    t_free(cbase);
    return bad;
}

/* ── the bookkeeping ───────────────────────────────────────────────────── */
typedef struct {
    int kernels, cases, bitwise, fails, printed;
    double worst;
    char where[96];
} t_stat_t;

static int g_fail_kernels, g_fail_drivers, g_kernels, g_cases, g_drivers, g_dcases, g_bitwise;
static double g_worst;

static uint64_t g_rs;
static double rnd(void)
{
    g_rs ^= g_rs << 13;
    g_rs ^= g_rs >> 7;
    g_rs ^= g_rs << 17;
    return (double)(g_rs >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}
static void seed(const char *s, int a, int b, int c)
{
    uint64_t h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    h ^= (uint64_t)a * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)(b + 1) * 0xC2B2AE3D27D4EB4FULL;
    h ^= (uint64_t)(c + 7) * 0x165667B19E3779F9ULL;
    g_rs = h ? h : 88172645463325252ULL;
}

/* run one case at both ISAs and judge it; label names it in a FAIL line. A
 * case that works in the buffer it stores into (in place, and every ZTURN-T
 * driver: dest and scrambled run their pipeline in zout) runs twice per ISA,
 * the second time on the finite fill: its output must not move (it reads
 * nothing outside its footprint) */
static int run_case(const t_case_t *c, t_stat_t *st, const char *label)
{
    double *x = (double *)malloc((2 * c->in.n + 1) * sizeof(double));
    double *y[2];
    char why[2][256];
    int bad[2], isa, fail = 0, bitwise = 1;
    double emax = 0, amax = 0, rel;
    size_t i;
    for (i = 0; i < 2 * c->in.n; i++) x[i] = c->in.fp[i / 2] ? rnd() : 0.0;
    for (isa = 0; isa < 2; isa++) {
        y[isa] = (double *)calloc(2 * c->out.n + 1, sizeof(double));
        why[isa][0] = 0;
        bad[isa] = run_isa(c, isa, 0, x, y[isa], why[isa], sizeof why[isa]);
        if (!bad[isa] && (c->inplace || c->call != CALL_K)) {
            double *y1 = (double *)calloc(2 * c->out.n + 1, sizeof(double));
            bad[isa] = run_isa(c, isa, 1, x, y1, why[isa], sizeof why[isa]);
            if (!bad[isa] && memcmp(y1, y[isa], 2 * c->out.n * sizeof(double))) {
                snprintf(why[isa], sizeof why[isa], "%s (finite fill): its output moves with what the doubles "
                         "outside its footprint hold (it reads them)", S[isa]->isa);
                bad[isa] = 1;
            }
            free(y1);
        }
    }
    for (i = 0; i < c->out.n; i++)
        if (c->out.fp[i]) {
            int part;
            for (part = 0; part < 2; part++) {
                const double a = y[0][2 * i + part], b = y[1][2 * i + part];
                if (memcmp(&a, &b, 8)) bitwise = 0;
                if (fabs(a) > amax) amax = fabs(a);
                if (fabs(b - a) > emax || b != b) emax = (b != b) ? INFINITY : fabs(b - a);
            }
        }
    rel = emax / (amax > 0 ? amax : 1.0);
    st->cases++;
    if (bad[0] || bad[1] || !(rel <= TWIN_TOL)) {
        fail = 1;
        st->fails++;
        if (st->printed++ < TWIN_MAXFAIL) {
            if (bad[0] || bad[1])
                printf("  FAIL %s: %s%s%s\n", label, why[0], bad[0] && bad[1] ? "; " : "", why[1]);
            else
                printf("  FAIL %s: avx512 != avx2, rel %.2e > %.1e\n", label, rel, TWIN_TOL);
        }
    } else if (bitwise) {
        st->bitwise++;
    }
    if (!bitwise && rel > st->worst) {
        st->worst = rel;
        snprintf(st->where, sizeof st->where, "%s", label);
    }
    free(x);
    free(y[0]);
    free(y[1]);
    return fail;
}

/* ── the kinds ─────────────────────────────────────────────────────────── */
enum {
    C_COL,     /* legs l*Ls + k in, m*OLs + k out (n1, n1c, t2, the blocked forms) */
    C_TRANS,   /* m*... in, the corner turn k*OLs + m out (n1t, t2t) */
    C_TRANSG,  /* the corner turn at a leg stride: k*OLs + m*OGs (t2tg) */
    C_CS,      /* column stride: l*Ls + k*Gs in, m*OLs + k*OGs out (n1ccs, t2cs) */
    C_GEN2,    /* C_CS with the gen2 T1 / T2 bases (t2csg, t2csgt) */
    C_GEN2N,   /* the gen2 group loop: base table, group order (t2csgn, t2csgnt) */
    C_ROWS,    /* the row loop, rows at pitch Gs / OGs, legs at Ls = the lanes */
    C_ROWST,   /* the row loop with the corner turn */
    C_DIGIT,   /* the digit loop: OGs digits at pitch Gs, broadcast records (t2c, t2cp) */
    C_SPLAT,   /* in place, Gs groups of R legs, splat records (msz, mszt) */
    C_MONO,    /* the whole R-point transform, natural in and out, one call (mono64_8x8_il) */
    C_RMONO,   /* the real mono: real legs l*Ls + k in, the CCE half's bins m*OLs + k out,
                * the backward the other way (rn1, rn1b48, rn1b444) */
    C_RLEAF,   /* the real flat leaf: real legs l*Ls + k in, digit 0 real at k and digit p
                * complex at doubles 2p*OLs + 2k out, the backward the other way (r1c) */
    C_HERM,    /* the gen2 group loop with the Hermitian edge on the spectrum side: base
                * table, legs above R/2 at the mirror (t2csgh, t2csght) */
    C_T0TP,    /* ZTURN-T ingest: natural legs in, runs of R0 at rb[c] out (block split) */
    C_TMG,     /* ZTURN-T mid, in place on the block split plane */
    C_TLF,     /* ZTURN-T last: block split in, natural out */
    C_T0D,     /* ZTURN-T plain ingest: natural in, leg-major block split out */
    C_TLD      /* ZTURN-T plain last: block split <-> the scrambled positions */
};
enum { TW_NONE, TW_PAIR, TW_BCAST, TW_GEN2, TW_SPLAT, TW_LANE };
static const struct { const char *kind; int cls, tw; } KINDS[] = {
    { "n1", C_COL, TW_NONE },        { "n1c", C_COL, TW_NONE },       { "n1_ct", C_COL, TW_NONE },
    { "n1b216", C_COL, TW_NONE },    { "n1b416", C_COL, TW_NONE },    { "n1b48", C_COL, TW_NONE },
    { "n1b88", C_COL, TW_NONE },     { "n1cb48", C_COL, TW_NONE },    { "n1cb84", C_COL, TW_NONE },
    { "n1cb88", C_COL, TW_NONE },    { "n1cb448", C_COL, TW_NONE },   { "n1cb816", C_COL, TW_NONE },
    { "t2", C_COL, TW_PAIR },        { "t2_ct", C_COL, TW_PAIR },     { "t2b416", C_COL, TW_PAIR },
    { "t2b88", C_COL, TW_PAIR },
    { "t2b", C_COL, TW_PAIR },       { "t2b48", C_COL, TW_PAIR },
    { "t2tan", C_COL, TW_PAIR },     { "t2tanm128", C_COL, TW_PAIR }, { "t2bw32", C_COL, TW_PAIR },
    { "t2bw32m128", C_COL, TW_PAIR }, { "n1tan", C_COL, TW_NONE },    { "n1btan216", C_COL, TW_NONE },
    { "n1t", C_TRANS, TW_NONE },     { "n1t_ct", C_TRANS, TW_NONE },  { "n1tb416", C_TRANS, TW_NONE },
    { "n1tb88", C_TRANS, TW_NONE },  { "t2t", C_TRANS, TW_PAIR },     { "t2t_ct", C_TRANS, TW_PAIR },
    { "t2bt216", C_TRANS, TW_PAIR }, { "t2bt416", C_TRANS, TW_PAIR }, { "t2bt48", C_TRANS, TW_PAIR },
    { "t2bt88", C_TRANS, TW_PAIR },  { "t2tg", C_TRANSG, TW_PAIR },
    { "n1tb", C_TRANS, TW_NONE },    { "n1tb44", C_TRANS, TW_NONE },  { "n1tb48", C_TRANS, TW_NONE },
    { "n1ttan", C_TRANS, TW_NONE },  { "n1tbw32", C_TRANS, TW_NONE }, { "n1tbw32t256", C_TRANS, TW_NONE },
    { "t2ttan", C_TRANS, TW_PAIR },
    { "n1ccs", C_CS, TW_NONE },      { "t2cs", C_CS, TW_PAIR },
    { "t2csg", C_GEN2, TW_GEN2 },    { "t2csgt", C_GEN2, TW_GEN2 },
    { "t2csgn", C_GEN2N, TW_GEN2 },  { "t2csgnt", C_GEN2N, TW_GEN2 },
    { "t2csgh", C_HERM, TW_GEN2 },   { "t2csght", C_HERM, TW_GEN2 },  { "r1c", C_RLEAF, TW_NONE },
    { "n1r", C_ROWS, TW_NONE },      { "n1rtan", C_ROWS, TW_NONE },   { "t2r", C_ROWS, TW_PAIR },
    { "t2rtan", C_ROWS, TW_PAIR },   { "n1tr", C_ROWST, TW_NONE },    { "n1trtan", C_ROWST, TW_NONE },
    { "t2tr", C_ROWST, TW_PAIR },    { "t2trtan", C_ROWST, TW_PAIR },
    { "t2c", C_DIGIT, TW_BCAST },    { "t2cp", C_DIGIT, TW_BCAST },   { "t2cb44", C_DIGIT, TW_BCAST },
    { "t2cb48", C_DIGIT, TW_BCAST }, { "t2cb84", C_DIGIT, TW_BCAST }, { "t2cb88", C_DIGIT, TW_BCAST },
    { "msz", C_SPLAT, TW_SPLAT },    { "mszt", C_SPLAT, TW_SPLAT },
    { "t0tp", C_T0TP, TW_NONE },     { "tmg", C_TMG, TW_LANE },       { "tmgd", C_TMG, TW_LANE },
    { "tlf", C_TLF, TW_LANE },       { "tlfi", C_TLF, TW_LANE },      { "t0d", C_T0D, TW_LANE },
    { "tld", C_TLD, TW_NONE },       { "mono64_8x8_il", C_MONO, TW_NONE },
    { "rn1", C_RMONO, TW_NONE },     { "rn1b48", C_RMONO, TW_NONE },  { "rn1b444", C_RMONO, TW_NONE },
};
#define NKINDS ((int)(sizeof KINDS / sizeof KINDS[0]))

/* the cells il_registry_avx512.h DECLARES ABSENT inside a kind both ISAs list
 * (its DECLARED ABSENT block, corpus.ml zil_isa_gap): the codelet_zsplit law,
 * t0tp/tld write or read R-runs as whole 8-lane blocks (R % VW == 0). A radix
 * the avx2 list of such a kind carries and the avx512 list does not is a FAIL
 * unless it is here; an entry the registries contradict is a FAIL too. A kind
 * whose avx512 lists are empty is declared absent whole. */
static const struct { const char *kind; int R, bwd; } ABSENT512[] = {
    { "t0tp", 4, 0 }, { "t0tp", 4, 1 }, { "tld", 4, 0 }, { "tld", 4, 1 },
};
#define NABSENT512 ((int)(sizeof ABSENT512 / sizeof ABSENT512[0]))

static const int COUNTS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                              31, 32, 33, 63, 64, 65, 66, 67 };
#define NCOUNTS ((int)(sizeof COUNTS / sizeof COUNTS[0]))
/* the ZTURN-T stage kernels: whole 8-column iterations; tld's chain per count
 * (R0 . mid . 8), the plain plan whose permutation places its output */
static const struct { int count, nf, chain[3]; } ZCOUNTS[] = {
    { 8, 1, { 8 } },        { 16, 2, { 4, 4 } },   { 24, 2, { 8, 3 } },   { 32, 2, { 8, 4 } },
    { 40, 2, { 8, 5 } },    { 56, 2, { 8, 7 } },   { 64, 2, { 8, 8 } },   { 72, 2, { 8, 9 } },
};
#define NZCOUNTS ((int)(sizeof ZCOUNTS / sizeof ZCOUNTS[0]))

/* the per-kernel tables of one ISA that do not depend on the count */
typedef struct { twin_flat_t f; t_table_t tf, t2, tz; int ok; } t_ktab_t;

static int ktab_build(int isa, int cls, int tw, int R, int bwd, t_ktab_t *kt)
{
    const twin_side_t *s = S[isa];
    memset(kt, 0, sizeof *kt);
    if (tw == TW_BCAST || tw == TW_SPLAT) {
        const int own[3] = { 4, R, 5 }, pool[3] = { 8, 47, 5 };
        if (!(s->flat(own, 3, 1, &kt->f) && (tw == TW_SPLAT ? kt->f.tz != NULL : kt->f.tf != NULL))) {
            if (kt->f.plan) s->flat_free(&kt->f);
            if (tw == TW_SPLAT || !s->flat(pool, 3, 1, &kt->f) || !kt->f.tf) return 0;
        }
        if (tw == TW_SPLAT) kt->tz = t_guarded(bwd ? kt->f.tzb : kt->f.tz, kt->f.ntz);
        else kt->tf = t_guarded(bwd ? kt->f.tfb : kt->f.tf, kt->f.ntf);
    } else if (tw == TW_GEN2) {
        const int one[2] = { 64, R }, four[3] = { 4, 47, R };
        if (cls == C_GEN2 ? !s->flat(one, 2, 1, &kt->f) : !s->flat(four, 3, 2, &kt->f)) return 0;
        if (!kt->f.tf || !kt->f.t2) return 0;
        kt->tf = t_guarded(bwd ? kt->f.tfb : kt->f.tf, kt->f.ntf);
        kt->t2 = t_guarded(bwd ? kt->f.t2b : kt->f.t2, kt->f.nt2);
    }
    if (kt->f.plan) s->flat_free(&kt->f);
    kt->ok = 1;
    return 1;
}
static void ktab_free(t_ktab_t *kt)
{
    t_free(kt->tf.base);
    t_free(kt->t2.base);
    t_free(kt->tz.base);
}

/* build the geometry of (class, R, count, direction) into c; aux tables into *aux */
static int case_geom(int cls, int R, int C, int bwd, t_case_t *c, size_t **a1, size_t **a3, int ngrp)
{
    size_t l, m, k, r, g, d, i;
    const size_t RR = (size_t)R, CC = (size_t)C;
    memset(c, 0, sizeof *c);
    c->call = CALL_K;
    c->count = CC;
    *a1 = *a3 = NULL;
    switch (cls) {
    case C_COL:
    case C_TRANS:
    case C_TRANSG:
        c->Ls = CC + 3;
        b_init(&c->in, (RR - 1) * c->Ls + CC, L_NAT);
        for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) c->in.fp[l * c->Ls + k] = 1;
        if (cls == C_COL) {
            c->OLs = CC + 5;
            b_init(&c->out, (RR - 1) * c->OLs + CC, L_NAT);
            for (m = 0; m < RR; m++) for (k = 0; k < CC; k++) c->out.fp[m * c->OLs + k] = 1;
        } else if (cls == C_TRANS) {
            c->OLs = RR + 2;
            b_init(&c->out, (CC - 1) * c->OLs + RR, L_NAT);
            for (k = 0; k < CC; k++) for (m = 0; m < RR; m++) c->out.fp[k * c->OLs + m] = 1;
        } else {
            c->OLs = 2 * RR + 1;
            c->OGs = 2;
            b_init(&c->out, (CC - 1) * c->OLs + 2 * (RR - 1) + 1, L_NAT);
            for (k = 0; k < CC; k++) for (m = 0; m < RR; m++) c->out.fp[k * c->OLs + 2 * m] = 1;
        }
        return 1;
    case C_CS:
    case C_GEN2:
        c->Ls = 2;
        c->Gs = 2 * RR + 1;
        b_init(&c->in, (CC - 1) * c->Gs + 2 * (RR - 1) + 1, L_NAT);
        for (k = 0; k < CC; k++) for (l = 0; l < RR; l++) c->in.fp[k * c->Gs + 2 * l] = 1;
        c->OLs = 3;
        c->OGs = 3 * RR + 2;
        b_init(&c->out, (CC - 1) * c->OGs + 3 * (RR - 1) + 1, L_NAT);
        for (k = 0; k < CC; k++) for (m = 0; m < RR; m++) c->out.fp[k * c->OGs + 3 * m] = 1;
        return 1;
    case C_GEN2N: {
        const size_t D = 1 + CC % 3, ng = (size_t)ngrp, OGs = D + 1, OLs = CC * OGs + 1;
        const size_t step = RR * OLs + 2;
        size_t *ob = (size_t *)malloc(ng * CC * sizeof(size_t));
        size_t *go = (CC & 1) ? (size_t *)malloc(ng * sizeof(size_t)) : NULL;
        c->Ls = D;
        c->Gs = ng;
        c->OLs = OLs;
        c->OGs = OGs;
        b_init(&c->in, ng * CC * RR * D, L_NAT);
        b_all(&c->in);
        b_init(&c->out, (ng - 1) * step + (RR - 1) * OLs + (CC - 1) * OGs + D, L_NAT);
        for (g = 0; g < ng; g++) {
            for (k = 0; k < CC; k++) ob[g * CC + k] = g * step;
            if (go) go[g] = ng - 1 - g;
            for (m = 0; m < RR; m++) for (k = 0; k < CC; k++) for (d = 0; d < D; d++)
                c->out.fp[g * step + m * OLs + k * OGs + d] = 1;
        }
        *a1 = ob;
        *a3 = go;
        return 1;
    }
    case C_ROWS:
    case C_ROWST: {
        const size_t nr = 1 + CC % 3, L = CC;
        c->Ls = L;
        c->count = nr * L;
        c->Gs = RR * L + 3;
        b_init(&c->in, (nr - 1) * c->Gs + RR * L, L_NAT);
        for (r = 0; r < nr; r++) for (l = 0; l < RR; l++) for (k = 0; k < L; k++)
            c->in.fp[r * c->Gs + l * L + k] = 1;
        if (cls == C_ROWS) {
            c->OLs = L + 2;
            c->OGs = RR * c->OLs + 1;
            b_init(&c->out, (nr - 1) * c->OGs + (RR - 1) * c->OLs + L, L_NAT);
            for (r = 0; r < nr; r++) for (m = 0; m < RR; m++) for (k = 0; k < L; k++)
                c->out.fp[r * c->OGs + m * c->OLs + k] = 1;
        } else {
            c->OLs = RR + 1;
            c->OGs = L * c->OLs + 2;
            b_init(&c->out, (nr - 1) * c->OGs + (L - 1) * c->OLs + RR, L_NAT);
            for (r = 0; r < nr; r++) for (k = 0; k < L; k++) for (m = 0; m < RR; m++)
                c->out.fp[r * c->OGs + k * c->OLs + m] = 1;
        }
        return 1;
    }
    case C_DIGIT: {
        const size_t D = 3;
        c->OGs = D;
        c->Ls = CC + 1;
        c->OLs = CC + 2;
        c->Gs = RR * (CC + 2) + 1;
        b_init(&c->in, (D - 1) * c->Gs + (RR - 1) * c->Ls + CC, L_NAT);
        b_init(&c->out, (D - 1) * c->Gs + (RR - 1) * c->OLs + CC, L_NAT);
        for (d = 0; d < D; d++) for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) {
            c->in.fp[d * c->Gs + l * c->Ls + k] = 1;
            c->out.fp[d * c->Gs + l * c->OLs + k] = 1;
        }
        return 1;
    }
    case C_SPLAT: {
        const size_t ng = 3;
        c->inplace = 1;
        c->Ls = CC + 2;
        c->Gs = ng;
        b_init(&c->in, ng * RR * c->Ls, L_NAT);
        b_init(&c->out, ng * RR * c->Ls, L_NAT);
        for (g = 0; g < ng; g++) for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) {
            c->in.fp[g * RR * c->Ls + l * c->Ls + k] = 1;
            c->out.fp[g * RR * c->Ls + l * c->Ls + k] = 1;
        }
        return 1;
    }
    case C_MONO:
        b_init(&c->in, RR, L_NAT);
        b_init(&c->out, RR, L_NAT);
        b_all(&c->in);
        b_all(&c->out);
        return 1;
    case C_RMONO: {
        /* the R reals of a column at pitch rp, its R/2 + 1 bins at pitch cp: the
         * input's legs at Ls = count + 3, the output's at OLs = count + 5 */
        const size_t H = RR / 2 + 1;
        t_buf_t *rs = bwd ? &c->out : &c->in, *cs = bwd ? &c->in : &c->out;
        size_t rp, cp;
        c->Ls = CC + 3;
        c->OLs = CC + 5;
        rp = bwd ? c->OLs : c->Ls;
        cp = bwd ? c->Ls : c->OLs;
        b_init(rs, (RR - 1) * rp + CC, L_REAL);
        b_init(cs, (H - 1) * cp + CC, L_NAT);
        for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) rs->fp[l * rp + k] = 1;
        for (m = 0; m < H; m++) for (k = 0; k < CC; k++) cs->fp[m * cp + k] = 1;
        return 1;
    }
    case C_RLEAF: {
        /* the R real legs at pitch lp, the digit runs at pitch dp (complex): digit 0
         * the real run at doubles 0..count-1, digit p (1..h) at doubles 2p*dp + 2k,
         * 2p*dp + 2k + 1; the rest of block 0 is a gap. The input's pitch is Ls =
         * count + 3, the output's OLs = count + 5 */
        const size_t h = RR / 2;
        t_buf_t *rs = bwd ? &c->out : &c->in, *ds = bwd ? &c->in : &c->out;
        size_t lp, dp;
        c->Ls = CC + 3;
        c->OLs = CC + 5;
        lp = bwd ? c->OLs : c->Ls;
        dp = bwd ? c->Ls : c->OLs;
        b_init(rs, (RR - 1) * lp + CC, L_REAL);
        b_init(ds, 2 * h * dp + 2 * CC, L_REAL);
        for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) rs->fp[l * lp + k] = 1;
        for (k = 0; k < CC; k++) ds->fp[k] = 1;
        for (m = 1; m <= h; m++) for (k = 0; k < 2 * CC; k++) ds->fp[2 * m * dp + k] = 1;
        return 1;
    }
    case C_HERM: {
        /* ngrp groups of count columns x R legs on the plane (the forward D columns
         * per block, the backward one), the spectrum at the groups' bases g*step
         * (the base table, stride count). Leg l of column k at the bin u = base +
         * d + k*P + l*S (column pitch P, leg stride S): a leg below R/2 there, one
         * above conjugated at the mirror NM - u, the middle leg by its bin (direct
         * when 2*(base + d + k*P) < S: the first columns of group 0). Every u stays
         * below U and NM = 2U, so the mirrored slots lie above every direct one */
        const size_t ng = (size_t)ngrp, D = bwd ? 1 : 1 + CC % 3, P = D + 1, S = CC * P + 1;
        const size_t step = RR * S + 2, U = (ng - 1) * step + RR * S, NM = 2 * U, h = RR / 2;
        size_t *ob = (size_t *)malloc(ng * CC * sizeof(size_t));
        t_buf_t *pl = bwd ? &c->out : &c->in, *sp = bwd ? &c->in : &c->out;
        b_init(pl, ng * CC * RR * D, L_NAT);
        b_all(pl);
        b_init(sp, NM + 1, L_NAT);
        for (g = 0; g < ng; g++) {
            for (k = 0; k < CC; k++) ob[g * CC + k] = g * step;
            for (d = 0; d < D; d++) for (k = 0; k < CC; k++) for (l = 0; l < RR; l++) {
                const size_t dk = g * step + d + k * P, u = dk + l * S;
                sp->fp[(l < h || (l == h && 2 * dk < S)) ? u : NM - u] = 1;
            }
        }
        c->Gs = ng;
        if (bwd) {   /* the spectrum's leg stride and column pitch; the plane block by block */
            c->Ls = S;
            c->OLs = P;
        } else {     /* D columns per block; the spectrum's leg stride and column pitch */
            c->Ls = D;
            c->OLs = S;
            c->OGs = P;
        }
        c->mirror = NM;
        c->mirror_in = bwd;
        *a1 = ob;
        return 1;
    }
    case C_T0TP: {
        /* the run-base table (tw_im): column c's run at rb[c] * R0, a bijection */
        size_t *rbt = (size_t *)malloc(CC * sizeof(size_t));
        c->Ls = CC + 5;
        b_init(&c->in, (RR - 1) * c->Ls + CC, L_NAT);
        for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) c->in.fp[l * c->Ls + k] = 1;
        b_init(&c->out, RR * CC, L_BSPLIT);
        b_all(&c->out);
        for (k = 0; k < CC; k++) rbt[k] = (CC - 1 - k + CC / 2) % CC;
        *a1 = rbt;
        return 1;
    }
    case C_TMG:
    case C_TLF:
    case C_T0D: {
        const size_t ng = 2;
        c->Ls = CC + 8;
        c->Gs = ng;
        if (cls == C_TMG) {
            c->inplace = 1;
            b_init(&c->in, ng * RR * c->Ls, L_BSPLIT);
            b_init(&c->out, ng * RR * c->Ls, L_BSPLIT);
        } else if (cls == C_TLF) {
            c->OLs = CC + 3;
            b_init(&c->in, ng * RR * c->Ls, L_BSPLIT);
            b_init(&c->out, ng * RR * c->OLs, L_NAT);
        } else {
            b_init(&c->in, ng * RR * c->Ls, L_NAT);
            b_init(&c->out, ng * RR * c->Ls, L_BSPLIT);
        }
        for (g = 0; g < ng; g++) for (l = 0; l < RR; l++) for (k = 0; k < CC; k++) {
            c->in.fp[g * RR * c->Ls + l * c->Ls + k] = 1;
            c->out.fp[g * RR * (cls == C_TLF ? c->OLs : c->Ls) + l * (cls == C_TLF ? c->OLs : c->Ls) + k] = 1;
        }
        return 1;
    }
    default:
        (void)i;
        return 0;
    }
}

/* tld: Gs groups of count columns x R legs; block split on one side, the
 * ISA's scrambled positions (its plain plan's) on the other */
static int case_tld(int R, int zc, int bwd, t_case_t *c)
{
    const size_t ng = 2, CC = (size_t)ZCOUNTS[zc].count, n = ng * (size_t)R * CC;
    int chain[4], nf = ZCOUNTS[zc].nf + 1, j, isa;
    size_t g, i;
    t_buf_t *mapped;
    memset(c, 0, sizeof *c);
    for (j = 0; j < ZCOUNTS[zc].nf; j++) chain[j] = ZCOUNTS[zc].chain[j];
    chain[nf - 1] = R;
    c->call = CALL_K;
    c->Gs = ng;
    c->count = CC;
    c->inplace = !bwd;
    b_init(&c->in, n, bwd ? L_MAP : L_BSPLIT);
    b_init(&c->out, n, bwd ? L_BSPLIT : L_MAP);
    b_all(&c->in);
    b_all(&c->out);
    mapped = bwd ? &c->in : &c->out;
    for (isa = 0; isa < 2; isa++) {
        size_t *pos = S[isa]->scr_pos((int)((size_t)R * CC), chain, nf);
        if (!pos) return 0;
        mapped->map[isa] = (size_t *)malloc(n * sizeof(size_t));
        for (g = 0; g < ng; g++)
            for (i = 0; i < (size_t)R * CC; i++)
                mapped->map[isa][g * R * CC + i] = g * R * CC + pos[i];
        free(pos);
    }
    return 1;
}

static void case_free(t_case_t *c)
{
    b_free(&c->in);
    b_free(&c->out);
}

/* ── the kernel pass ───────────────────────────────────────────────────── */
static int want(const char *kind, const char *only)
{
    const char *p = only;
    size_t n = strlen(kind);
    if (!only) return 1;
    while (*p) {
        const char *e = strchr(p, ',');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (len == n && !strncmp(p, kind, n)) return 1;
        if (!e) break;
        p = e + 1;
    }
    return 0;
}

static const twin_kernel_t *find_kernel(int isa, const char *kind, int R, int bwd)
{
    int i;
    for (i = 0; i < S[isa]->nkern; i++) {
        const twin_kernel_t *k = &S[isa]->kern[i];
        if (k->R == R && k->bwd == bwd && !strcmp(k->kind, kind)) return k;
    }
    return NULL;
}

/* one kernel at both ISAs over its counts */
static void gate_kernel(int ki, int R, int bwd, twin_kfn f2, twin_kfn f5, t_stat_t *st)
{
    const char *kind = KINDS[ki].kind;
    const int cls = KINDS[ki].cls, tw = KINDS[ki].tw;
    t_ktab_t kt[2];
    int isa, ci, ncount, kfail = 0;
    char label[96];
    for (isa = 0; isa < 2; isa++)
        if (!ktab_build(isa, cls, tw, R, bwd, &kt[isa])) {
            st->cases++;
            st->fails++;
            if (st->printed++ < TWIN_MAXFAIL)
                printf("  FAIL %s radix %d %s: %s builds no table for it\n", kind, R,
                       bwd ? "bwd" : "fwd", S[isa]->isa);
            if (isa) ktab_free(&kt[0]);
            g_fail_kernels++;
            return;
        }
    ncount = cls == C_MONO ? 1 : (cls >= C_T0TP) ? NZCOUNTS : NCOUNTS;
    for (ci = 0; ci < ncount; ci++) {
        const int C = cls == C_MONO ? 1 : (cls >= C_T0TP) ? ZCOUNTS[ci].count : COUNTS[ci];
        t_case_t c;
        size_t *a1 = NULL, *a3 = NULL;
        t_table_t ptab[2];
        memset(ptab, 0, sizeof ptab);
        if (tw == TW_GEN2 && C > kt[0].f.G) continue;   /* the table holds G columns */
        if (cls == C_TLD) {
            if (!case_tld(R, ci, bwd, &c)) {
                st->cases++;
                st->fails++;
                kfail = 1;
                if (st->printed++ < TWIN_MAXFAIL)
                    printf("  FAIL %s radix %d: no plain plan places count %d\n", kind, R, C);
                case_free(&c);
                continue;
            }
        } else {
            case_geom(cls, R, C, bwd, &c, &a1, &a3, kt[0].f.ngrp < 3 ? kt[0].f.ngrp : 3);
        }
        c.k[0] = f2;
        c.k[1] = f5;
        if (cls == C_GEN2N || cls == C_HERM) {
            c.a1 = a1;
            c.a3 = a3;
        }
        if (cls == C_T0TP) c.t2[0] = c.t2[1] = (const double *)a1;
        for (isa = 0; isa < 2; isa++) {
            size_t nd = 0;
            double *t = NULL;
            if (tw == TW_PAIR) {
                const int cols = (cls == C_ROWS || cls == C_ROWST) ? (int)c.Ls : C;
                t = S[isa]->pair(R, 1, cols, R * (cols + 1), bwd, &nd);
            } else if (tw == TW_LANE) {
                t = S[isa]->lane((long)C, R, (long)R * C, bwd, &nd);
            }
            if (t) {
                ptab[isa] = t_guarded(t, nd);
                S[isa]->free_table(t);
                c.tw[isa] = ptab[isa].p;
            }
            if (tw == TW_BCAST || tw == TW_GEN2) c.tw[isa] = kt[isa].tf.p;
            if (tw == TW_GEN2) c.t2[isa] = kt[isa].t2.p;
            if (tw == TW_SPLAT) c.tw[isa] = kt[isa].tz.p;
        }
        seed(kind, R, bwd, C);
        snprintf(label, sizeof label, "%s radix %d %s count %d", kind, R, bwd ? "bwd" : "fwd", C);
        kfail |= run_case(&c, st, label);
        for (isa = 0; isa < 2; isa++) t_free(ptab[isa].base);
        case_free(&c);
        free(a1);
        free(a3);
    }
    ktab_free(&kt[0]);
    ktab_free(&kt[1]);
    if (kfail) g_fail_kernels++;
}

static int count_kind(int isa, const char *kind)
{
    int i, n = 0;
    for (i = 0; i < S[isa]->nkern; i++) n += !strcmp(S[isa]->kern[i].kind, kind);
    return n;
}

static int declared_absent512(const char *kind, int R, int bwd)
{
    int i;
    for (i = 0; i < NABSENT512; i++)
        if (ABSENT512[i].R == R && ABSENT512[i].bwd == bwd && !strcmp(ABSENT512[i].kind, kind)) return 1;
    return 0;
}

static void kernel_pass(const char *only)
{
    int ki, isa, i, j;
    /* a declared absence the registries contradict */
    for (i = 0; i < NABSENT512; i++) {
        const char *kind = ABSENT512[i].kind;
        const int R = ABSENT512[i].R, bwd = ABSENT512[i].bwd;
        if (!want(kind, only)) continue;
        if (find_kernel(1, kind, R, bwd) || !find_kernel(0, kind, R, bwd)) {
            printf("%-8s FAIL: radix %d %s is declared absent at avx512 (ABSENT512), but %s\n", kind, R,
                   bwd ? "bwd" : "fwd", find_kernel(1, kind, R, bwd) ? "avx512 lists it" : "avx2 does not list it");
            g_fail_kernels++;
        }
    }
    /* the kinds the gate has no case for: listed by one ISA, reported; by both, a FAIL */
    for (isa = 0; isa < 2; isa++)
        for (i = 0; i < S[isa]->nkern; i++) {
            const char *kind = S[isa]->kern[i].kind;
            int known = 0, seen = 0, both = 0;
            for (ki = 0; ki < NKINDS; ki++) known |= !strcmp(KINDS[ki].kind, kind);
            for (j = 0; j < i && !seen; j++) seen = !strcmp(S[isa]->kern[j].kind, kind);
            if (known || seen || !want(kind, only)) continue;
            if (isa == 1 && count_kind(0, kind)) continue;   /* reported with avx2 */
            for (j = 0; j < S[0]->nkern; j++)
                if (!strcmp(S[0]->kern[j].kind, kind) &&
                    find_kernel(1, kind, S[0]->kern[j].R, S[0]->kern[j].bwd)) both++;
            if (both) {
                printf("%-8s FAIL: both ISAs list %d of its kernels and the gate has no case for the kind\n",
                       kind, both);
                g_fail_kernels += both;
            } else if (count_kind(1 - isa, kind)) {
                printf("%-8s %s %d, %s %d kernels, none at both: not gated\n", kind, S[0]->isa,
                       count_kind(0, kind), S[1]->isa, count_kind(1, kind));
            } else {
                printf("%-8s %s only (%d kernels): not gated\n", kind, S[isa]->isa, count_kind(isa, kind));
            }
        }
    for (ki = 0; ki < NKINDS; ki++) {
        t_stat_t st;
        int n2 = 0, n5 = 0, nabs = 0;
        if (!want(KINDS[ki].kind, only)) continue;
        memset(&st, 0, sizeof st);
        for (i = 0; i < S[1]->nkern; i++) n5 += !strcmp(S[1]->kern[i].kind, KINDS[ki].kind);
        for (i = 0; i < S[0]->nkern; i++) {
            const twin_kernel_t *k = &S[0]->kern[i];
            const twin_kernel_t *t;
            if (strcmp(k->kind, KINDS[ki].kind)) continue;
            n2++;
            t = find_kernel(1, k->kind, k->R, k->bwd);
            if (!t) {
                /* a kind avx512 lists: a radix it lacks is declared absent, or a FAIL */
                if (!n5) continue;
                if (declared_absent512(k->kind, k->R, k->bwd)) {
                    nabs++;
                    continue;
                }
                st.fails++;
                g_fail_kernels++;
                if (st.printed++ < TWIN_MAXFAIL)
                    printf("  FAIL %s radix %d %s: avx2 lists it, avx512 does not, and il_registry_avx512.h "
                           "does not declare it absent\n", k->kind, k->R, k->bwd ? "bwd" : "fwd");
                continue;
            }
            st.kernels++;
            gate_kernel(ki, k->R, k->bwd, k->fn, t->fn, &st);
        }
        g_kernels += st.kernels;
        g_cases += st.cases;
        g_bitwise += st.bitwise;
        if (st.worst > g_worst) g_worst = st.worst;
        if (!n2 && !n5) continue;
        printf("%-8s %3d kernels %5d cases  bitwise %5d  worst %.2e%s%s", KINDS[ki].kind, st.kernels,
               st.cases, st.bitwise, st.worst, st.worst > 0 ? " at " : "", st.worst > 0 ? st.where : "");
        if (n2 != st.kernels || n5 != st.kernels) {
            printf("  (avx2 lists %d, avx512 %d", n2, n5);
            if (nabs) printf("; %d declared absent at avx512", nabs);
            printf(")");
        }
        printf("  %s\n", st.fails ? "FAIL" : "ok");
        fflush(stdout);
    }
}

/* ── the ZTURN-T fused drivers ─────────────────────────────────────────── */
static const twin_cell_t *find_cell(int isa, const twin_cell_t *c)
{
    int i, s;
    for (i = 0; i < S[isa]->ncell; i++) {
        const twin_cell_t *d = &S[isa]->cell[i];
        if (d->n != c->n || d->nf != c->nf) continue;
        for (s = 0; s < c->nf && d->chain[s] == c->chain[s]; s++) {}
        if (s == c->nf) return d;
    }
    return NULL;
}

static const char *SLOT[6] = { "fwd dest", "fwd plane", "bwd dest", "bwd plane", "fwd scr", "bwd scr" };

static void driver_pass(int ztt_max)
{
    t_stat_t st[6];
    int i, slot, isa, ncells = 0;
    memset(st, 0, sizeof st);
    for (i = 0; i < S[1]->ncell; i++) {
        const twin_cell_t *c5 = &S[1]->cell[i], *c2 = find_cell(0, c5);
        const twin_cell_t *cc[2];
        int scr, cellfail = 0;
        if (!c2 || c5->n > ztt_max) continue;
        cc[0] = c2;
        cc[1] = c5;
        ncells++;
        for (scr = 0; scr < 2; scr++) {
            twin_ztt_t z[2];
            t_table_t tw[2][2];
            size_t tiles[2] = { 0, 0 }, t;
            int ok = 1, ti, ntile = 1;
            const int s0 = scr ? 4 : 0, s1 = scr ? 6 : 4;
            /* the class is gated where both rows carry it: an avx512 row may hold one
             * class only (the 8-lane laws, ztt.h) */
            int have = 1;
            for (isa = 0; isa < 2; isa++)
                have = have && (scr ? (cc[isa]->fwd_scr && cc[isa]->bwd_scr)
                                    : (cc[isa]->fwd_dest && cc[isa]->fwd_plane &&
                                       cc[isa]->bwd_dest && cc[isa]->bwd_plane));
            if (!have) continue;
            memset(tw, 0, sizeof tw);
            for (isa = 0; isa < 2; isa++) {
                if (!S[isa]->ztt(c5->n, c5->chain, c5->nf, scr, 0, &z[isa])) {
                    ok = 0;
                    if (isa) {
                        t_free(tw[0][0].base);
                        t_free(tw[0][1].base);
                        S[0]->ztt_free(&z[0]);
                    }
                    break;
                }
                tw[isa][0] = t_guarded(z[isa].tw, z[isa].ntw);
                tw[isa][1] = t_guarded(z[isa].twb, z[isa].ntw);
            }
            if (!ok) {
                st[s0].cases++;
                st[s0].fails++;
                printf("  FAIL ztt %d %s: a create refused a cell both registries list\n", c5->n,
                       scr ? "plain" : "natural");
                cellfail = 1;
                continue;
            }
            for (t = 2; t < (size_t)c5->n; t *= 2)
                if (S[0]->tile_ok(c5->n, c5->chain, c5->nf, t, scr) &&
                    S[1]->tile_ok(c5->n, c5->chain, c5->nf, t, scr)) {
                    tiles[1] = t;
                    ntile = 2;
                    break;
                }
            for (slot = s0; slot < s1; slot++)
                for (ti = 0; ti < ntile; ti++) {
                    t_case_t c;
                    char label[96];
                    const size_t N = (size_t)c5->n;
                    const int bwd = slot == 2 || slot == 3 || slot == 5;
                    int ok2 = 1;
                    memset(&c, 0, sizeof c);
                    for (isa = 0; isa < 2; isa++) {
                        const twin_cell_t *e = cc[isa];
                        switch (slot) {
                        case 0: c.zf[isa] = e->fwd_dest; break;
                        case 1: c.zf[isa] = e->fwd_plane; break;
                        case 2: c.zf[isa] = e->bwd_dest; break;
                        case 3: c.zf[isa] = e->bwd_plane; break;
                        case 4: c.zp[isa] = e->fwd_scr; break;
                        default: c.zp[isa] = e->bwd_scr; break;
                        }
                        if (slot < 4 ? !c.zf[isa] : !c.zp[isa]) ok2 = 0;
                        c.tw[isa] = tw[isa][bwd].p;
                        c.rb[isa] = z[isa].rb;
                    }
                    if (!ok2) continue;
                    c.call = slot < 4 ? CALL_ZTT : CALL_ZTTP;
                    c.tile = tiles[ti];
                    c.inplace = (slot == 1 || slot == 3);
                    c.plane = slot < 4 ? N : 0;
                    c.inplane = c.inplace;
                    b_init(&c.in, N, slot == 5 ? L_MAP : L_NAT);
                    b_init(&c.out, N, slot == 4 ? L_MAP : L_NAT);
                    b_all(&c.in);
                    b_all(&c.out);
                    if (slot >= 4) {
                        t_buf_t *b = slot == 4 ? &c.out : &c.in;
                        for (isa = 0; isa < 2; isa++) {
                            b->map[isa] = (size_t *)malloc(N * sizeof(size_t));
                            memcpy(b->map[isa], z[isa].perm, N * sizeof(size_t));
                        }
                    }
                    seed("ztt", c5->n, slot, (int)c.tile);
                    {
                        int s, off;
                        off = snprintf(label, sizeof label, "ztt %d ", c5->n);
                        for (s = 0; s < c5->nf; s++)
                            off += snprintf(label + off, sizeof label - (size_t)off, "%s%d", s ? "." : "",
                                            c5->chain[s]);
                        snprintf(label + off, sizeof label - (size_t)off, " %s tile %zu", SLOT[slot], c.tile);
                    }
                    st[slot].kernels += (ti == 0);
                    cellfail |= run_case(&c, &st[slot], label);
                    case_free(&c);
                }
            for (isa = 0; isa < 2; isa++) {
                t_free(tw[isa][0].base);
                t_free(tw[isa][1].base);
                S[isa]->ztt_free(&z[isa]);
            }
        }
        if (cellfail) g_fail_drivers++;
    }
    for (slot = 0; slot < 6; slot++) {
        g_drivers += st[slot].kernels;
        g_dcases += st[slot].cases;
        g_bitwise += st[slot].bitwise;
        if (st[slot].worst > g_worst) g_worst = st[slot].worst;
        printf("ztt %-9s %3d drivers %4d cases  bitwise %4d  worst %.2e%s%s  %s\n", SLOT[slot],
               st[slot].kernels, st[slot].cases, st[slot].bitwise, st[slot].worst,
               st[slot].worst > 0 ? " at " : "", st[slot].worst > 0 ? st[slot].where : "",
               st[slot].fails ? "FAIL" : "ok");
    }
    printf("ztt: %d cells both registries list (of %d avx2, %d avx512)\n", ncells, S[0]->ncell, S[1]->ncell);
}

int main(int argc, char **argv)
{
    const char *only = NULL;
    int ztt_max = 262144, i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--kind") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--ztt-max") && i + 1 < argc) ztt_max = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: twin_gate [--kind k1,k2,..] [--ztt-max N]\n");
            return 2;
        }
    }
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512dq") ||
        !__builtin_cpu_supports("avx512vl")) {
        printf("twin_gate: this CPU has no AVX-512 (avx512f, avx512dq, avx512vl): run it on an AVX-512 "
               "host or under Intel SDE (sde -spr -- twin_gate.exe)\n");
        return 2;
    }
    S[0] = twin_side_avx2();
    S[1] = twin_side_avx512();
    printf("twin gate: %s %d kernels, %s %d kernels; ztt %d / %d cells; tolerance %.1e\n",
           S[0]->isa, S[0]->nkern, S[1]->isa, S[1]->nkern, S[0]->ncell, S[1]->ncell, TWIN_TOL);
    fflush(stdout);
    kernel_pass(only);
    if (want("ztt", only)) driver_pass(ztt_max);
    if (g_fail_kernels || g_fail_drivers) {
        printf("TWIN GATE *** FAIL ***: %d kernels, %d driver cells (the FAIL lines above)\n",
               g_fail_kernels, g_fail_drivers);
        return 1;
    }
    printf("TWIN GATE ALL PASS: avx512 == avx2, %d kernels (%d cases), %d drivers (%d cases), "
           "worst %.1e\n", g_kernels, g_cases, g_drivers, g_dcases, g_worst);
    printf("  %d of the %d cases bitwise\n", g_bitwise, g_cases + g_dcases);
    return 0;
}
#endif /* VFFT_TWIN_SIDE */
