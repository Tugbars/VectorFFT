/* zil_pipe_gate.c — kernel-level R2 gate for the zil->pipeline port
 * (docs/roadmap/zil_pipeline_port.md, §7/§9). Races every pipeline-hosted
 * zsplit kernel (codelet_zsplit.ml) against its legacy codelet_zil twin on
 * identical inputs/tables, plus a scalar reference and the production
 * roundtrip contract bwd(fwd) = R*x with table-side-conjugated tables.
 * 31 checks; expected verdict as of 2026-07-25: ALL BIT-IDENTICAL (0.0).
 *
 * The 30 kernel .c arms are NOT committed — regenerate them (WSL, from
 * src/dag-fft-compiler/generator, after `DUNE_CACHE=disabled
 * /home/tugbars/.opam/5.2.0/bin/dune build --root $PWD bin/gen_radix.exe`):
 *
 *   G=./_build/default/bin/gen_radix.exe ; ZP="--isa avx2 --uarch raptor_lake_avx2"
 *   legacy arms (sed-rename so both link side by side):
 *     $G 4 --z-ms   --emit-c | sed 's/_z_ms_/_z_msref_/g'      > legacy_ms4f.c
 *     $G 4 --z-msb  --emit-c | sed 's/_z_ms_/_z_msref_/g'      > legacy_ms4b.c
 *     (same for radix 8; msg/msgb -> s/_z_msg_/_z_msgref_/;
 *      s0s/s0sb -> s/_z_s0s_/_z_s0sref_/; sterm/stermb ->
 *      s/_z_sterm_/_z_stermref_/; sterm2 -> s/_z_sterm2_/_z_sterm2ref_/)
 *   pipeline arms (SAME production names -> also drop-in for R2-cascade):
 *     $G 4 --zp-ms $ZP --emit-c > zp_ms4f.c   ... and --zp-msb/--zp-msg/
 *     --zp-msgb/--zp-s0s/--zp-s0sb (r4+r8), --zp-sterm/--zp-stermb/
 *     --zp-sterm2 (r8 only)
 *   build (MUST have -mavx2 -mfma: msg _zsg bodies carry no target attr):
 *     gcc -O2 -mavx2 -mfma -o zil_pipe_gate.exe zil_pipe_gate.c legacy_*.c zp_*.c
 *
 * Layout notes: block-split planes, leg l's re half at 2*(l*Ls + kq) + j,
 * im half +4 (kq = column quad base, j = lane). Ls = count here.
 * tw: splat-pair records, legs 1..R-1, 8 doubles/leg [c x4][s x4];
 * bwd tables = conjugated (sin negated) — kernels must NOT re-conjugate
 * (the double-conj trap, zil_pipeline_port.md §6.1).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DECL(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long, unsigned long long);
DECL(radix8_z_msref_fwd_avx2) DECL(radix8_z_msref_bwd_avx2)
DECL(radix4_z_msref_fwd_avx2) DECL(radix4_z_msref_bwd_avx2)
DECL(radix8_z_ms_fwd_avx2)    DECL(radix8_z_ms_bwd_avx2)
DECL(radix4_z_ms_fwd_avx2)    DECL(radix4_z_ms_bwd_avx2)
DECL(radix8_z_msgref_fwd_avx2) DECL(radix8_z_msgref_bwd_avx2)
DECL(radix4_z_msgref_fwd_avx2) DECL(radix4_z_msgref_bwd_avx2)
DECL(radix8_z_msg_fwd_avx2)    DECL(radix8_z_msg_bwd_avx2)
DECL(radix4_z_msg_fwd_avx2)    DECL(radix4_z_msg_bwd_avx2)
DECL(radix8_z_s0sref_fwd_avx2) DECL(radix8_z_s0sref_bwd_avx2)
DECL(radix4_z_s0sref_fwd_avx2) DECL(radix4_z_s0sref_bwd_avx2)
DECL(radix8_z_s0s_fwd_avx2)    DECL(radix8_z_s0s_bwd_avx2)
DECL(radix4_z_s0s_fwd_avx2)    DECL(radix4_z_s0s_bwd_avx2)
DECL(radix8_z_stermref_fwd_avx2) DECL(radix8_z_stermref_bwd_avx2)
DECL(radix8_z_sterm_fwd_avx2)    DECL(radix8_z_sterm_bwd_avx2)
DECL(radix8_z_sterm2ref_fwd_avx2) DECL(radix8_z_sterm2_fwd_avx2)

typedef void (*msfn)(const double *, const double *, double *, double *,
                     const double *, const double *, unsigned long long,
                     unsigned long long, unsigned long long, unsigned long long,
                     unsigned long long);

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}

/* block-split element address (doubles offset), re half; im = +4 */
static size_t addr(int l, size_t k, size_t Ls)
{
    return 2 * ((size_t)l * Ls + (k & ~(size_t)3)) + (k & 3);
}

/* scalar reference for ONE column: fwd = DFT_R(x .* w), w_0 = 1;
 * bwd = conj(w) .* IDFT_R(x) (unscaled, +theta kernel). */
static void ref_col(int R, int bwd, const double *ang, const double *xr,
                    const double *xi, double *yr, double *yi)
{
    double txr[8], txi[8];
    int l, k;
    if (!bwd) {
        for (l = 0; l < R; l++) {
            double c = cos(l ? ang[l] : 0.0), s = sin(l ? ang[l] : 0.0);
            txr[l] = xr[l] * c - xi[l] * s;
            txi[l] = xr[l] * s + xi[l] * c;
        }
        for (k = 0; k < R; k++) {
            double sr = 0.0, si = 0.0;
            for (l = 0; l < R; l++) {
                double th = -2.0 * 3.14159265358979323846 * l * k / R;
                double c = cos(th), s = sin(th);
                sr += txr[l] * c - txi[l] * s;
                si += txr[l] * s + txi[l] * c;
            }
            yr[k] = sr; yi[k] = si;
        }
    } else {
        for (k = 0; k < R; k++) {
            double sr = 0.0, si = 0.0;
            for (l = 0; l < R; l++) {
                double th = 2.0 * 3.14159265358979323846 * l * k / R;
                double c = cos(th), s = sin(th);
                sr += xr[l] * c - xi[l] * s;
                si += xr[l] * s + xi[l] * c;
            }
            /* post-multiply by conj(w_k) */
            {
                double c = cos(k ? ang[k] : 0.0), s = -sin(k ? ang[k] : 0.0);
                yr[k] = sr * c - si * s;
                yi[k] = sr * s + si * c;
            }
        }
    }
}

static int run_case(int R, int bwd)
{
    const size_t count = 64, Ls = count;
    const size_t nd = 2 * (size_t)R * Ls;
    double *in  = (double *)_aligned_malloc(nd * 8, 64);
    double *o_l = (double *)_aligned_malloc(nd * 8, 64);
    double *o_p = (double *)_aligned_malloc(nd * 8, 64);
    double *rt  = (double *)_aligned_malloc(nd * 8, 64);
    double tw[7 * 8], twb[7 * 8], ang[8];
    unsigned seed = 42 + R + bwd;
    size_t i, k;
    int l, ok = 1;
    msfn f_l, f_p, f_pb;

    for (i = 0; i < nd; i++) in[i] = urand(&seed);
    ang[0] = 0.0;
    for (l = 1; l < R; l++) ang[l] = -2.0 * 3.14159265358979323846 * l * 3.0 / 117.0;
    for (l = 1; l < R; l++) {
        int j;
        for (j = 0; j < 4; j++) {
            tw[(l - 1) * 8 + j]      = cos(ang[l]);
            tw[(l - 1) * 8 + 4 + j]  = sin(ang[l]);
            twb[(l - 1) * 8 + j]     = cos(ang[l]);
            twb[(l - 1) * 8 + 4 + j] = -sin(ang[l]);
        }
    }
    if (R == 8) {
        f_l = bwd ? radix8_z_msref_bwd_avx2 : radix8_z_msref_fwd_avx2;
        f_p = bwd ? radix8_z_ms_bwd_avx2 : radix8_z_ms_fwd_avx2;
        f_pb = radix8_z_ms_bwd_avx2;
    } else {
        f_l = bwd ? radix4_z_msref_bwd_avx2 : radix4_z_msref_fwd_avx2;
        f_p = bwd ? radix4_z_ms_bwd_avx2 : radix4_z_ms_fwd_avx2;
        f_pb = radix4_z_ms_bwd_avx2;
    }

    /* A/B: legacy vs pipeline on identical inputs */
    f_l(in, 0, o_l, 0, bwd ? twb : tw, 0, Ls, 0, 0, 0, count);
    f_p(in, 0, o_p, 0, bwd ? twb : tw, 0, Ls, 0, 0, 0, count);
    {
        double dmax = 0.0, amax = 0.0;
        for (i = 0; i < nd; i++) {
            double d = fabs(o_l[i] - o_p[i]);
            double a = fabs(o_l[i]);
            if (d > dmax) dmax = d;
            if (a > amax) amax = a;
        }
        printf("R=%d %s  legacy-vs-pipeline  max|d|=%.3e (scale %.3e)  %s\n",
               R, bwd ? "bwd" : "fwd", dmax, amax,
               dmax <= 1e-14 * amax ? "PASS" : "FAIL");
        if (dmax > 1e-14 * amax) ok = 0;
    }
    /* pipeline vs scalar reference, per column */
    {
        double dmax = 0.0;
        for (k = 0; k < count; k++) {
            double xr[8], xi[8], yr[8], yi[8];
            for (l = 0; l < R; l++) {
                xr[l] = in[addr(l, k, Ls)];
                xi[l] = in[addr(l, k, Ls) + 4];
            }
            ref_col(R, bwd, ang, xr, xi, yr, yi);
            for (l = 0; l < R; l++) {
                double dr = fabs(o_p[addr(l, k, Ls)] - yr[l]);
                double di = fabs(o_p[addr(l, k, Ls) + 4] - yi[l]);
                if (dr > dmax) dmax = dr;
                if (di > dmax) dmax = di;
            }
        }
        printf("R=%d %s  pipeline-vs-scalar  max|d|=%.3e  %s\n",
               R, bwd ? "bwd" : "fwd", dmax, dmax <= 1e-13 ? "PASS" : "FAIL");
        if (dmax > 1e-13) ok = 0;
    }
    /* roundtrip (fwd case only): bwd(fwd(x)) = R*x through the pipeline pair */
    if (!bwd) {
        double dmax = 0.0;
        f_pb(o_p, 0, rt, 0, twb, 0, Ls, 0, 0, 0, count);
        for (i = 0; i < nd; i++) {
            double d = fabs(rt[i] - (double)R * in[i]);
            if (d > dmax) dmax = d;
        }
        printf("R=%d roundtrip bwd(fwd)=R*x  max|d|=%.3e  %s\n",
               R, dmax, dmax <= 1e-12 ? "PASS" : "FAIL");
        if (dmax > 1e-12) ok = 0;
    }
    _aligned_free(in); _aligned_free(o_l); _aligned_free(o_p); _aligned_free(rt);
    return ok;
}

/* P2: msg group-loop wrapper — Gs groups in ONE call, in-place on the same
 * plane (contiguous group bases g*2*R*Ls), per-group splat-pair sets at
 * twg = tw + g*(R-1)*8. Gate: legacy-vs-pipeline on identical buffers +
 * per-(group,column) scalar reference + roundtrip. */
static int run_msg_case(int R, int bwd)
{
    const size_t count = 32, Ls = count, Gs = 4;
    const size_t nd = 2 * Gs * (size_t)R * Ls;
    double *b_l = (double *)_aligned_malloc(nd * 8, 64);
    double *b_p = (double *)_aligned_malloc(nd * 8, 64);
    double *b_rt = (double *)_aligned_malloc(nd * 8, 64);
    double *tw  = (double *)_aligned_malloc(Gs * 7 * 8 * 8, 64);
    double *twb = (double *)_aligned_malloc(Gs * 7 * 8 * 8, 64);
    double ang[4][8];
    unsigned seed = 1234 + R + bwd;
    size_t i, g, k;
    int l, ok = 1;
    msfn f_l, f_p, f_pb;

    for (i = 0; i < nd; i++) b_l[i] = urand(&seed);
    memcpy(b_p, b_l, nd * 8);
    memcpy(b_rt, b_l, nd * 8);
    for (g = 0; g < Gs; g++) {
        ang[g][0] = 0.0;
        for (l = 1; l < R; l++)
            ang[g][l] = -2.0 * 3.14159265358979323846
                        * ((double)l * (double)(3 + 5 * g)) / 231.0;
        for (l = 1; l < R; l++) {
            int j;
            for (j = 0; j < 4; j++) {
                tw[(g * (R - 1) + (l - 1)) * 8 + j]      = cos(ang[g][l]);
                tw[(g * (R - 1) + (l - 1)) * 8 + 4 + j]  = sin(ang[g][l]);
                twb[(g * (R - 1) + (l - 1)) * 8 + j]     = cos(ang[g][l]);
                twb[(g * (R - 1) + (l - 1)) * 8 + 4 + j] = -sin(ang[g][l]);
            }
        }
    }
    if (R == 8) {
        f_l = bwd ? radix8_z_msgref_bwd_avx2 : radix8_z_msgref_fwd_avx2;
        f_p = bwd ? radix8_z_msg_bwd_avx2 : radix8_z_msg_fwd_avx2;
        f_pb = radix8_z_msg_bwd_avx2;
    } else {
        f_l = bwd ? radix4_z_msgref_bwd_avx2 : radix4_z_msgref_fwd_avx2;
        f_p = bwd ? radix4_z_msg_bwd_avx2 : radix4_z_msg_fwd_avx2;
        f_pb = radix4_z_msg_bwd_avx2;
    }

    /* in-place, one call for all Gs groups */
    f_l(b_l, 0, b_l, 0, bwd ? twb : tw, 0, Ls, Gs, 0, 0, count);
    f_p(b_p, 0, b_p, 0, bwd ? twb : tw, 0, Ls, Gs, 0, 0, count);
    {
        double dmax = 0.0, amax = 0.0;
        for (i = 0; i < nd; i++) {
            double d = fabs(b_l[i] - b_p[i]);
            double a = fabs(b_l[i]);
            if (d > dmax) dmax = d;
            if (a > amax) amax = a;
        }
        printf("R=%d %s msg legacy-vs-pipeline  max|d|=%.3e (scale %.3e)  %s\n",
               R, bwd ? "bwd" : "fwd", dmax, amax,
               dmax <= 1e-14 * amax ? "PASS" : "FAIL");
        if (dmax > 1e-14 * amax) ok = 0;
    }
    /* pipeline vs scalar reference, per (group, column) — inputs from b_rt */
    {
        double dmax = 0.0;
        for (g = 0; g < Gs; g++) {
            const double *gin = b_rt + g * 2 * (size_t)R * Ls;
            const double *gout = b_p + g * 2 * (size_t)R * Ls;
            for (k = 0; k < count; k++) {
                double xr[8], xi[8], yr[8], yi[8];
                for (l = 0; l < R; l++) {
                    xr[l] = gin[addr(l, k, Ls)];
                    xi[l] = gin[addr(l, k, Ls) + 4];
                }
                ref_col(R, bwd, ang[g], xr, xi, yr, yi);
                for (l = 0; l < R; l++) {
                    double dr = fabs(gout[addr(l, k, Ls)] - yr[l]);
                    double di = fabs(gout[addr(l, k, Ls) + 4] - yi[l]);
                    if (dr > dmax) dmax = dr;
                    if (di > dmax) dmax = di;
                }
            }
        }
        printf("R=%d %s msg pipeline-vs-scalar  max|d|=%.3e  %s\n",
               R, bwd ? "bwd" : "fwd", dmax, dmax <= 1e-13 ? "PASS" : "FAIL");
        if (dmax > 1e-13) ok = 0;
    }
    /* roundtrip (fwd only): bwd(fwd(x)) = R*x, in place on b_p */
    if (!bwd) {
        double dmax = 0.0;
        f_pb(b_p, 0, b_p, 0, twb, 0, Ls, Gs, 0, 0, count);
        for (i = 0; i < nd; i++) {
            double d = fabs(b_p[i] - (double)R * b_rt[i]);
            if (d > dmax) dmax = d;
        }
        printf("R=%d msg roundtrip bwd(fwd)=R*x  max|d|=%.3e  %s\n",
               R, dmax, dmax <= 1e-12 ? "PASS" : "FAIL");
        if (dmax > 1e-12) ok = 0;
    }
    _aligned_free(b_l); _aligned_free(b_p); _aligned_free(b_rt);
    _aligned_free(tw); _aligned_free(twb);
    return ok;
}

/* P3: s0s leaf — fwd: natural z in (element (l,k) = complex l*Ls+k) ->
 * block-split planes out; bwd: planes in -> natural z out. Twiddle-free.
 * Reference reuses ref_col with all-zero angles (identity twiddles). */
static int run_s0s_case(int R, int bwd)
{
    const size_t count = 64, Ls = count;
    const size_t nd = 2 * (size_t)R * Ls;
    double *in  = (double *)_aligned_malloc(nd * 8, 64);
    double *o_l = (double *)_aligned_malloc(nd * 8, 64);
    double *o_p = (double *)_aligned_malloc(nd * 8, 64);
    double *rt  = (double *)_aligned_malloc(nd * 8, 64);
    double ang[8] = { 0 };
    unsigned seed = 777 + R + bwd;
    size_t i, k;
    int l, ok = 1;
    msfn f_l, f_p, f_pb;

    for (i = 0; i < nd; i++) in[i] = urand(&seed);
    if (R == 8) {
        f_l = bwd ? radix8_z_s0sref_bwd_avx2 : radix8_z_s0sref_fwd_avx2;
        f_p = bwd ? radix8_z_s0s_bwd_avx2 : radix8_z_s0s_fwd_avx2;
        f_pb = radix8_z_s0s_bwd_avx2;
    } else {
        f_l = bwd ? radix4_z_s0sref_bwd_avx2 : radix4_z_s0sref_fwd_avx2;
        f_p = bwd ? radix4_z_s0s_bwd_avx2 : radix4_z_s0s_fwd_avx2;
        f_pb = radix4_z_s0s_bwd_avx2;
    }
    f_l(in, 0, o_l, 0, 0, 0, Ls, 0, 0, 0, count);
    f_p(in, 0, o_p, 0, 0, 0, Ls, 0, 0, 0, count);
    {
        double dmax = 0.0, amax = 0.0;
        for (i = 0; i < nd; i++) {
            double d = fabs(o_l[i] - o_p[i]);
            double a = fabs(o_l[i]);
            if (d > dmax) dmax = d;
            if (a > amax) amax = a;
        }
        printf("R=%d %s s0s legacy-vs-pipeline  max|d|=%.3e (scale %.3e)  %s\n",
               R, bwd ? "bwd" : "fwd", dmax, amax,
               dmax <= 1e-14 * amax ? "PASS" : "FAIL");
        if (dmax > 1e-14 * amax) ok = 0;
    }
    /* pipeline vs scalar reference */
    {
        double dmax = 0.0;
        for (k = 0; k < count; k++) {
            double xr[8], xi[8], yr[8], yi[8];
            for (l = 0; l < R; l++) {
                if (!bwd) { /* z in */
                    xr[l] = in[2 * ((size_t)l * Ls + k)];
                    xi[l] = in[2 * ((size_t)l * Ls + k) + 1];
                } else { /* planes in */
                    xr[l] = in[addr(l, k, Ls)];
                    xi[l] = in[addr(l, k, Ls) + 4];
                }
            }
            ref_col(R, bwd, ang, xr, xi, yr, yi);
            for (l = 0; l < R; l++) {
                double gr, gi;
                if (!bwd) { /* planes out */
                    gr = o_p[addr(l, k, Ls)];
                    gi = o_p[addr(l, k, Ls) + 4];
                } else { /* z out */
                    gr = o_p[2 * ((size_t)l * Ls + k)];
                    gi = o_p[2 * ((size_t)l * Ls + k) + 1];
                }
                if (fabs(gr - yr[l]) > dmax) dmax = fabs(gr - yr[l]);
                if (fabs(gi - yi[l]) > dmax) dmax = fabs(gi - yi[l]);
            }
        }
        printf("R=%d %s s0s pipeline-vs-scalar  max|d|=%.3e  %s\n",
               R, bwd ? "bwd" : "fwd", dmax, dmax <= 1e-13 ? "PASS" : "FAIL");
        if (dmax > 1e-13) ok = 0;
    }
    /* roundtrip (fwd only): s0sb(s0s(x)) = R*x in the z domain */
    if (!bwd) {
        double dmax = 0.0;
        f_pb(o_p, 0, rt, 0, 0, 0, Ls, 0, 0, 0, count);
        for (i = 0; i < nd; i++) {
            double d = fabs(rt[i] - (double)R * in[i]);
            if (d > dmax) dmax = d;
        }
        printf("R=%d s0s roundtrip bwd(fwd)=R*x  max|d|=%.3e  %s\n",
               R, dmax, dmax <= 1e-12 ? "PASS" : "FAIL");
        if (dmax > 1e-12) ok = 0;
    }
    _aligned_free(in); _aligned_free(o_l); _aligned_free(o_p); _aligned_free(rt);
    return ok;
}

/* P4: sterm terminator — fwd: block-split col-blocks in (column k's legs =
 * complexes 8k+l), per-column packed w^1 (squaring tree in-register),
 * radix-8 DIT, REINT stores to the leg-major z comb (OLs). bwd mirrors.
 * Reference: per-column ref_col with ang[l] = l*a_k (w^l = w1^l). */
static int run_sterm_case(int bwd)
{
    const int R = 8;
    const size_t count = 64, OLs = count;
    const size_t nd = 2 * (size_t)R * count;
    double *in  = (double *)_aligned_malloc(nd * 8, 64);
    double *o_l = (double *)_aligned_malloc(nd * 8, 64);
    double *o_p = (double *)_aligned_malloc(nd * 8, 64);
    double *rt  = (double *)_aligned_malloc(nd * 8, 64);
    double *twq  = (double *)_aligned_malloc(count * 16 * 8, 64);
    double *twqb = (double *)_aligned_malloc(count * 16 * 8, 64);
    unsigned seed = 999 + bwd;
    size_t i, k;
    int l, ok = 1;
    msfn f_l, f_p, f_pb;

    for (i = 0; i < nd; i++) in[i] = urand(&seed);
    for (k = 0; k < count; k++) {
        double a = -2.0 * 3.14159265358979323846 * (double)((7 * k + 3) % 97) / 97.0;
        twq[2 * (k & ~(size_t)3) + (k & 3)]      = cos(a);
        twq[2 * (k & ~(size_t)3) + 4 + (k & 3)]  = sin(a);
        twqb[2 * (k & ~(size_t)3) + (k & 3)]     = cos(a);
        twqb[2 * (k & ~(size_t)3) + 4 + (k & 3)] = -sin(a);
    }
    f_l = bwd ? radix8_z_stermref_bwd_avx2 : radix8_z_stermref_fwd_avx2;
    f_p = bwd ? radix8_z_sterm_bwd_avx2 : radix8_z_sterm_fwd_avx2;
    f_pb = radix8_z_sterm_bwd_avx2;

    f_l(in, 0, o_l, 0, bwd ? twqb : twq, 0, 0, 0, OLs, 0, count);
    f_p(in, 0, o_p, 0, bwd ? twqb : twq, 0, 0, 0, OLs, 0, count);
    {
        double dmax = 0.0, amax = 0.0;
        for (i = 0; i < nd; i++) {
            double d = fabs(o_l[i] - o_p[i]);
            double a = fabs(o_l[i]);
            if (d > dmax) dmax = d;
            if (a > amax) amax = a;
        }
        printf("%s sterm legacy-vs-pipeline  max|d|=%.3e (scale %.3e)  %s\n",
               bwd ? "bwd" : "fwd", dmax, amax,
               dmax <= 1e-14 * amax ? "PASS" : "FAIL");
        if (dmax > 1e-14 * amax) ok = 0;
    }
    /* pipeline vs scalar reference */
    {
        double dmax = 0.0;
        for (k = 0; k < count; k++) {
            double a1 = -2.0 * 3.14159265358979323846 * (double)((7 * k + 3) % 97) / 97.0;
            double ang[8], xr[8], xi[8], yr[8], yi[8];
            for (l = 0; l < R; l++) ang[l] = (double)l * a1;
            for (l = 0; l < R; l++) {
                if (!bwd) { /* block-split col-blocks in: complex j = 8k+l */
                    size_t j = 8 * k + (size_t)l;
                    xr[l] = in[2 * (j & ~(size_t)3) + (j & 3)];
                    xi[l] = in[2 * (j & ~(size_t)3) + 4 + (j & 3)];
                } else { /* z comb in */
                    xr[l] = in[2 * ((size_t)l * OLs + k)];
                    xi[l] = in[2 * ((size_t)l * OLs + k) + 1];
                }
            }
            ref_col(R, bwd, ang, xr, xi, yr, yi);
            for (l = 0; l < R; l++) {
                double gr, gi;
                if (!bwd) { /* z comb out */
                    gr = o_p[2 * ((size_t)l * OLs + k)];
                    gi = o_p[2 * ((size_t)l * OLs + k) + 1];
                } else { /* col-blocks out */
                    size_t j = 8 * k + (size_t)l;
                    gr = o_p[2 * (j & ~(size_t)3) + (j & 3)];
                    gi = o_p[2 * (j & ~(size_t)3) + 4 + (j & 3)];
                }
                if (fabs(gr - yr[l]) > dmax) dmax = fabs(gr - yr[l]);
                if (fabs(gi - yi[l]) > dmax) dmax = fabs(gi - yi[l]);
            }
        }
        printf("%s sterm pipeline-vs-scalar  max|d|=%.3e  %s\n",
               bwd ? "bwd" : "fwd", dmax, dmax <= 1e-13 ? "PASS" : "FAIL");
        if (dmax > 1e-13) ok = 0;
    }
    /* roundtrip (fwd only): stermb(sterm(x)) = R*x in the block domain */
    if (!bwd) {
        double dmax = 0.0;
        f_pb(o_p, 0, rt, 0, twqb, 0, 0, 0, OLs, 0, count);
        for (i = 0; i < nd; i++) {
            double d = fabs(rt[i] - (double)R * in[i]);
            if (d > dmax) dmax = d;
        }
        printf("sterm roundtrip bwd(fwd)=R*x  max|d|=%.3e  %s\n",
               dmax, dmax <= 1e-12 ? "PASS" : "FAIL");
        if (dmax > 1e-12) ok = 0;
    }
    _aligned_free(in); _aligned_free(o_l); _aligned_free(o_p); _aligned_free(rt);
    _aligned_free(twq); _aligned_free(twqb);
    return ok;
}

/* P5: sterm2 — the 2-quad unroll-and-jam twin. Contract: bit-identical to
 * sterm (scheduling only). count = 60 = 7*8 + 4 exercises BOTH the 2-quad
 * main loop and the baseline-shaped 4-column tail. */
static int run_sterm2_case(void)
{
    const int R = 8;
    const size_t count = 60, OLs = count;
    const size_t nd = 2 * (size_t)R * count;
    double *in   = (double *)_aligned_malloc(nd * 8, 64);
    double *o_s  = (double *)_aligned_malloc(nd * 8, 64);
    double *o_l2 = (double *)_aligned_malloc(nd * 8, 64);
    double *o_p2 = (double *)_aligned_malloc(nd * 8, 64);
    double *twq  = (double *)_aligned_malloc(count * 16 * 8, 64);
    unsigned seed = 4242;
    size_t i, k;
    int ok = 1;

    for (i = 0; i < nd; i++) in[i] = urand(&seed);
    for (k = 0; k < count; k++) {
        double a = -2.0 * 3.14159265358979323846 * (double)((11 * k + 5) % 89) / 89.0;
        twq[2 * (k & ~(size_t)3) + (k & 3)]     = cos(a);
        twq[2 * (k & ~(size_t)3) + 4 + (k & 3)] = sin(a);
    }
    radix8_z_sterm_fwd_avx2(in, 0, o_s, 0, twq, 0, 0, 0, OLs, 0, count);
    radix8_z_sterm2ref_fwd_avx2(in, 0, o_l2, 0, twq, 0, 0, 0, OLs, 0, count);
    radix8_z_sterm2_fwd_avx2(in, 0, o_p2, 0, twq, 0, 0, 0, OLs, 0, count);
    {
        double d1 = 0.0, d2 = 0.0;
        for (i = 0; i < nd; i++) {
            if (fabs(o_p2[i] - o_s[i]) > d1) d1 = fabs(o_p2[i] - o_s[i]);
            if (fabs(o_p2[i] - o_l2[i]) > d2) d2 = fabs(o_p2[i] - o_l2[i]);
        }
        printf("sterm2 pipeline-vs-pipeline-sterm  max|d|=%.3e  %s\n",
               d1, d1 == 0.0 ? "PASS (bit)" : "FAIL");
        printf("sterm2 pipeline-vs-legacy-sterm2   max|d|=%.3e  %s\n",
               d2, d2 == 0.0 ? "PASS (bit)" : "FAIL");
        if (d1 != 0.0 || d2 != 0.0) ok = 0;
    }
    _aligned_free(in); _aligned_free(o_s); _aligned_free(o_l2); _aligned_free(o_p2);
    _aligned_free(twq);
    return ok;
}

int main(void)
{
    int ok = 1;
    ok &= run_case(4, 0);
    ok &= run_case(4, 1);
    ok &= run_case(8, 0);
    ok &= run_case(8, 1);
    ok &= run_msg_case(4, 0);
    ok &= run_msg_case(4, 1);
    ok &= run_msg_case(8, 0);
    ok &= run_msg_case(8, 1);
    ok &= run_s0s_case(4, 0);
    ok &= run_s0s_case(4, 1);
    ok &= run_s0s_case(8, 0);
    ok &= run_s0s_case(8, 1);
    ok &= run_sterm_case(0);
    ok &= run_sterm_case(1);
    ok &= run_sterm2_case();
    printf("\n%s\n", ok ? "GATE: OVERALL PASS" : "GATE: OVERALL FAIL");
    return ok ? 0 : 1;
}
