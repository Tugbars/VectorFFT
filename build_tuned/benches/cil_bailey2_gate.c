/* CIL-3/4 gate: bailey2 pieces through the shared scheduler.
 *   n1t — stage-1 leaf, four-step transpose fused into the stores
 *   t2  — stage-2 mid, streamed VTW2 twiddles applied with BYTW2
 * Each vs the legacy hand-scheduled kernel AND vs a scalar reference.
 *
 * VTW2 record (per column-pair p, per leg l=1..R-1, 8 doubles at
 * tw + (p*(R-1) + (l-1))*8):
 *   [ c(k), c(k), c(k+1), c(k+1) ][ -s(k), +s(k), -s(k+1), +s(k+1) ]
 * so BYTW2 = fmadd(c, x, mul(s, cflip x)) yields x * (c + i s).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DECL(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long, unsigned long long);
DECL(radix8_z_n1tref_fwd_avx2) DECL(radix8_z_n1t_fwd_avx2)
DECL(radix8_z_t2ref_fwd_avx2)  DECL(radix8_z_t2_fwd_avx2)

#define PI 3.14159265358979323846
#define R 8

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}

/* twiddle angle used by both the table builder and the reference */
static double ang(int l, size_t k, size_t M) { return -2.0 * PI * (double)l * (double)k / (double)M; }

int main(void)
{
    const size_t count = 64, Ls = count, OLs = count, M = R * count;
    const size_t nd = 2 * (size_t)R * count;
    double *in = (double *)_aligned_malloc(nd * 8, 64);
    double *o_l = (double *)_aligned_malloc(nd * 8, 64);
    double *o_p = (double *)_aligned_malloc(nd * 8, 64);
    double *tw = (double *)_aligned_malloc((count / 2) * (R - 1) * 8 * 8, 64);
    unsigned seed = 99;
    int ok = 1;

    for (size_t i = 0; i < nd; i++) in[i] = urand(&seed);
    for (size_t p = 0; p < count / 2; p++)
        for (int l = 1; l < R; l++) {
            double *rec = tw + (p * (R - 1) + (size_t)(l - 1)) * 8;
            for (int j = 0; j < 2; j++) {
                double a = ang(l, 2 * p + (size_t)j, M);
                rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
            }
        }

    /* ---- n1t: leaf with corner-turn stores ---- */
    radix8_z_n1tref_fwd_avx2(in, 0, o_l, 0, 0, 0, Ls, 0, (unsigned long long)R, 0, count);
    radix8_z_n1t_fwd_avx2(in, 0, o_p, 0, 0, 0, Ls, 0, (unsigned long long)R, 0, count);
    {
        double dm = 0.0, am = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double d = fabs(o_l[i] - o_p[i]), a = fabs(o_l[i]);
            if (d > dm) dm = d;
            if (a > am) am = a;
        }
        printf("n1t  pipeline-vs-legacy  max|d|=%.3e  %s%s\n", dm,
               dm <= 1e-14 * am ? "PASS" : "FAIL", dm == 0.0 ? "  (BIT-IDENTICAL)" : "");
        if (dm > 1e-14 * am) ok = 0;
    }
    {   /* scalar ref, CORNER-TURNED addressing: (leg p, col k) -> 2*(k*OLs + p) */
        double dm = 0.0;
        for (size_t k = 0; k < count; k++)
            for (int p = 0; p < R; p++) {
                double sr = 0.0, si = 0.0;
                for (int m = 0; m < R; m++) {
                    double th = -2.0 * PI * (double)(m * p) / (double)R;
                    double xr = in[2 * ((size_t)m * Ls + k)], xi = in[2 * ((size_t)m * Ls + k) + 1];
                    sr += xr * cos(th) - xi * sin(th);
                    si += xr * sin(th) + xi * cos(th);
                }
                double gr = o_p[2 * (k * R + (size_t)p)], gi = o_p[2 * (k * R + (size_t)p) + 1];
                if (fabs(gr - sr) > dm) dm = fabs(gr - sr);
                if (fabs(gi - si) > dm) dm = fabs(gi - si);
            }
        printf("n1t  pipeline-vs-scalar  max|d|=%.3e  %s\n", dm, dm <= 1e-12 ? "PASS" : "FAIL");
        if (dm > 1e-12) ok = 0;
    }

    /* ---- t2: mid with streamed VTW2 twiddles ---- */
    radix8_z_t2ref_fwd_avx2(in, 0, o_l, 0, tw, 0, Ls, 0, OLs, 0, count);
    radix8_z_t2_fwd_avx2(in, 0, o_p, 0, tw, 0, Ls, 0, OLs, 0, count);
    {
        double dm = 0.0, am = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double d = fabs(o_l[i] - o_p[i]), a = fabs(o_l[i]);
            if (d > dm) dm = d;
            if (a > am) am = a;
        }
        printf("t2   pipeline-vs-legacy  max|d|=%.3e  %s%s\n", dm,
               dm <= 1e-14 * am ? "PASS" : "FAIL", dm == 0.0 ? "  (BIT-IDENTICAL)" : "");
        if (dm > 1e-14 * am) ok = 0;
    }
    {   /* scalar ref: y[p] = sum_l (x[l] * w(l,k)) e^{-2pi i l p / R} */
        double dm = 0.0;
        for (size_t k = 0; k < count; k++)
            for (int p = 0; p < R; p++) {
                double sr = 0.0, si = 0.0;
                for (int l = 0; l < R; l++) {
                    double xr = in[2 * ((size_t)l * Ls + k)], xi = in[2 * ((size_t)l * Ls + k) + 1];
                    double a = (l == 0) ? 0.0 : ang(l, k, M);
                    double tr = xr * cos(a) - xi * sin(a);
                    double ti = xr * sin(a) + xi * cos(a);
                    double th = -2.0 * PI * (double)(l * p) / (double)R;
                    sr += tr * cos(th) - ti * sin(th);
                    si += tr * sin(th) + ti * cos(th);
                }
                double gr = o_p[2 * ((size_t)p * OLs + k)], gi = o_p[2 * ((size_t)p * OLs + k) + 1];
                if (fabs(gr - sr) > dm) dm = fabs(gr - sr);
                if (fabs(gi - si) > dm) dm = fabs(gi - si);
            }
        printf("t2   pipeline-vs-scalar  max|d|=%.3e  %s\n", dm, dm <= 1e-12 ? "PASS" : "FAIL");
        if (dm > 1e-12) ok = 0;
    }
    printf("\n%s\n", ok ? "CIL-3/4 GATE: PASS" : "CIL-3/4 GATE: FAIL");
    return ok ? 0 : 1;
}
