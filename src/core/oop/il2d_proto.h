/* il2d_proto.h — SCALAR SIMULATOR for the native IL 2D c2c tier's stage
 * maps (M0 of docs/roadmap/fft2d_il_c2c_design.md).
 *
 * LAW (the il2p scar, eight guessed stride maps falsified in one session):
 * maps are derived with the algebra shown, then proven by a running
 * simulator, BEFORE any SIMD code uses them. This header is that simulator;
 * build_tuned/benches/il2d_proto_gate.c is its gate vs a naive separable
 * DFT, elementwise, per direction (roundtrip never gates a permuted
 * transform).
 *
 * ── THE ALGEBRA (column pass: same-slot in-place DIF along axis i) ──
 *
 * One column of length L = R*D (R = this stage's radix, D = remaining
 * length). Split the output index k = R*u + r:
 *
 *   X[R*u + r] = sum_{d=0}^{D-1} [ ( sum_{t=0}^{R-1} x[d + t*D] * W_R^{r t} )
 *                                  * W_L^{d r} ] * W_D^{d u}
 *
 * so stage work at slot pair (d, r):
 *   legs      x[d + t*D],  t in [0,R)          leg stride  = D
 *   butterfly y_r = sum_t x[d+t*D] * W_R^{rt}
 *   twiddle   y_r *= W_L^{d*r}                 (W_L = e^{sgn*2*pi*i/L})
 *   store     slot d + r*D  (same slot set the legs came from: in-place)
 * then recurse on each length-D block. The output lands digit-reversed:
 * decomposing k least-significant-first along the chain (r0 = k mod R0,
 * u = k/R0, r1 = u mod R1, ...), the FIRST stage's digit is the MOST
 * significant position digit:
 *
 *   pos(k) = r0*(N1/R0) + r1*(N1/(R0*R1)) + ... + r_{m-1}*1
 *
 * On the 2D plane z[2*(i*N2 + j)] every i-unit scales by N2 complex:
 * leg stride D*N2, block stride L*N2, d-groups at stride N2, and the
 * column index j is the vector/count axis (unit complex stride).
 *
 * ── THE TWO FACTS THE SIMULATOR EXISTS TO PIN ──
 * 1. The twiddle W_L^{d r} depends on (r, d) ONLY — never on the column j
 *    and never on the block base. This is the broadcast-record (z-T1S)
 *    admissibility fact the t2c kernels are built on: per (d, leg) records
 *    hoisted out of the column loop.
 * 2. The last stage has D = 1 ⇒ W_L^{d r} = 1: it is the twiddle-free
 *    n1 shape. A single-stage chain (N1 <= codelet radix) is served by the
 *    SHIPPED n1 kernels with Ls = N2 and no new emission (M1).
 *
 * Row pass in this simulator: a naive N2-point DFT per row, NATURAL output
 * — it isolates the column-pass algebra, which is the new machinery. The
 * production row pass is the K=1 IL engine and carries its own per-plan
 * contract; the M1 driver gate runs end-to-end through the front door.
 *
 * Direction: bwd = conjugated twiddles (sign +1), unscaled, own map, gated
 * against the naive inverse independently (per-direction law).
 *
 * Scope v1: chain radices are the cil codelet set {4,8,16,32,64}; N1 =
 * product of the chain; N2 any >= 1 (the kernels' count contract is
 * ANY >= 1 via the VEX-128 odd tail).
 */
#ifndef VFFT_IL2D_PROTO_H
#define VFFT_IL2D_PROTO_H

#include <math.h>
#include <stddef.h>

#define IL2D_MAX_STAGES 8

/* one in-place DIF stage over every length-L block of every column.
 * z = interleaved plane, N1 x N2 complex; L = current block length,
 * R = stage radix; sgn = -1.0 fwd, +1.0 bwd. Scalar on purpose. */
static void il2d_sim_col_stage(double *z, int N1, int N2,
                               int L, int R, double sgn)
{
    const int D = L / R;
    const double pi = 3.14159265358979323846;
    /* scratch for one butterfly's R legs (R <= 64) */
    double yr[64], yi[64];
    int b, d, j, t, r;
    for (b = 0; b < N1; b += L) {                 /* block base (i units)  */
        for (d = 0; d < D; d++) {                 /* twiddle-bearing digit */
            for (j = 0; j < N2; j++) {            /* column = count axis   */
                /* butterfly: y_r = sum_t x[d + t*D] * W_R^{r t} */
                for (r = 0; r < R; r++) {
                    double sr = 0.0, si = 0.0;
                    for (t = 0; t < R; t++) {
                        const size_t idx =
                            2 * ((size_t)(b + d + t * D) * N2 + j);
                        const double a = sgn * 2.0 * pi * (double)(r * t)
                                         / (double)R;
                        const double c = cos(a), s = sin(a);
                        sr += z[idx] * c - z[idx + 1] * s;
                        si += z[idx] * s + z[idx + 1] * c;
                    }
                    /* stage twiddle W_L^{d r} — depends on (d, r) only */
                    {
                        const double a = sgn * 2.0 * pi * (double)(d * r)
                                         / (double)L;
                        const double c = cos(a), s = sin(a);
                        yr[r] = sr * c - si * s;
                        yi[r] = sr * s + si * c;
                    }
                }
                /* same-slot stores: slot d + r*D of this block */
                for (r = 0; r < R; r++) {
                    const size_t idx =
                        2 * ((size_t)(b + d + r * D) * N2 + j);
                    z[idx] = yr[r];
                    z[idx + 1] = yi[r];
                }
            }
        }
    }
}

/* full column pass: chain[0..nf) applied at shrinking L; returns 0, or -1
 * if the chain does not multiply to N1. */
static int il2d_sim_col_pass(double *z, int N1, int N2,
                             const int *chain, int nf, double sgn)
{
    int L = N1, s;
    long prod = 1;
    for (s = 0; s < nf; s++) prod *= chain[s];
    if (prod != N1 || nf < 1 || nf > IL2D_MAX_STAGES) return -1;
    for (s = 0; s < nf; s++) {
        il2d_sim_col_stage(z, N1, N2, L, chain[s], sgn);
        L /= chain[s];
    }
    return 0;
}

/* output row map: k (natural frequency) -> plane row (scrambled position).
 * First stage's digit = most significant position digit (derivation above). */
static int il2d_sim_row_pos(int k, int N1, const int *chain, int nf)
{
    int pos = 0, w = N1, s;
    for (s = 0; s < nf; s++) {
        const int r = k % chain[s];
        w /= chain[s];
        pos += r * w;
        k /= chain[s];
    }
    return pos;
}

/* naive N2-point DFT per row, natural output, out-of-place row buffer. */
static void il2d_sim_row_pass(double *z, int N1, int N2, double sgn)
{
    const double pi = 3.14159265358979323846;
    double tmp[2 * 4096];
    int i, k, j;
    for (i = 0; i < N1; i++) {
        double *row = z + 2 * (size_t)i * N2;
        for (k = 0; k < N2; k++) {
            double sr = 0.0, si = 0.0;
            for (j = 0; j < N2; j++) {
                const double a = sgn * 2.0 * pi * (double)(k * j)
                                 / (double)N2;
                const double c = cos(a), s = sin(a);
                sr += row[2 * j] * c - row[2 * j + 1] * s;
                si += row[2 * j] * s + row[2 * j + 1] * c;
            }
            tmp[2 * k] = sr;
            tmp[2 * k + 1] = si;
        }
        for (k = 0; k < 2 * N2; k++) row[k] = tmp[k];
    }
}

/* the composed simulator: column DIF pass (scrambled-i) + naive rows
 * (natural-j). Output element (ki, kj) of the DFT sits at plane position
 * (il2d_sim_row_pos(ki), kj). sgn = -1 fwd, +1 bwd (unscaled). */
static int il2d_sim_2d(double *z, int N1, int N2,
                       const int *chain, int nf, double sgn)
{
    if (N2 > 4096) return -1;                 /* row scratch bound */
    if (il2d_sim_col_pass(z, N1, N2, chain, nf, sgn)) return -1;
    il2d_sim_row_pass(z, N1, N2, sgn);
    return 0;
}

#endif /* VFFT_IL2D_PROTO_H */
