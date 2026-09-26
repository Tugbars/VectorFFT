/* audit_b differential driver: AVX2 original (SYM_A) vs AVX-512 twin (SYM_B)
 * vs a long-double naive DFT reference, identical inputs, per-width VTW2
 * tables built from the SAME per-(leg,column) complex twiddles.
 *
 * -DSYM_A=<avx2 symbol> -DSYM_B=<avx512 symbol> -DRADIX=R
 * -DKIND: 0 = T2 pre-twiddle (fwd t2), 1 = T2 post-twiddle (bwd t2),
 *         2 = N1 (no twiddle), 3 = N1C colstride (n1ccs)
 * -DDIRS: -1 fwd (exp(-2 pi i ..)), +1 bwd
 * -DROWLOOP: 0/1 (the --cil-rowloop ABI: count = rows*Ls, Gs/OGs row pitches)
 *
 * Buffers end exactly at a PROT_NONE guard page (over-reads/over-writes past
 * the logical end fault and are reported as OOB). Output gaps are prefilled
 * with a sentinel and checked for stray writes.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <unistd.h>

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
extern void SYM_A(const double *, const double *, double *, double *,
                  const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void SYM_B(const double *, const double *, double *, double *,
                  const double *, const double *, size_t, size_t, size_t, size_t, size_t);

#define R RADIX
#define STR2(x) #x
#define STR(x) STR2(x)
static sigjmp_buf jb;
static void segv(int s) { (void)s; siglongjmp(jb, 1); }

/* allocate n doubles whose END abuts a PROT_NONE page */
typedef struct { void *base; size_t len; double *p; } gbuf;
static gbuf galloc(size_t n)
{
    size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    size_t bytes = n * sizeof(double);
    size_t data_pages = (bytes + pg - 1) / pg + 1;
    size_t len = (data_pages + 1) * pg;
    char *b = mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) { perror("mmap"); exit(2); }
    mprotect(b + data_pages * pg, pg, PROT_NONE);
    gbuf g = { b, len, (double *)(b + data_pages * pg - bytes) };
    return g;
}
static void gfree(gbuf g) { munmap(g.base, g.len); }

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double rnd(void)
{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return ((double)(rs >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}
static const uint64_t SENT_BITS = 0x7ff8dead0000beefull;
static int is_sent(double d) { uint64_t u; memcpy(&u, &d, 8); return u == SENT_BITS; }
static double sent(void) { double d; memcpy(&d, &SENT_BITS, 8); return d; }

/* VTW2 table at width vw (doubles/vector): per group g, per leg l, one record
 * [c x vw][-s,+s ...], lane j of group g = column g*per + j. */
static gbuf build_tab(int vw, int nk, const double *tre, const double *tim)
{
    int per = vw / 2;
    int ng = (nk + per - 1) / per;
    size_t n = (size_t)ng * (R - 1) * 2 * vw;
    gbuf g = galloc(n ? n : 1);
    for (int gi = 0; gi < ng; gi++)
        for (int l = 1; l < R; l++) {
            double *rec = g.p + ((size_t)gi * (R - 1) + (l - 1)) * 2 * vw;
            for (int j = 0; j < per; j++) {
                int k = gi * per + j;
                double c = 1234.5, s = -777.25; /* pad lanes: finite garbage */
                if (k < nk) { c = tre[(size_t)l * nk + k]; s = tim[(size_t)l * nk + k]; }
                rec[2 * j] = c; rec[2 * j + 1] = c;
                rec[vw + 2 * j] = -s; rec[vw + 2 * j + 1] = s;
            }
        }
    return g;
}

/* geometry */
typedef struct { size_t Ls, Gs, OLs, OGs, count, rows, lanes; } geo;

static size_t in_idx(const geo *g, int row, int l, size_t k)
{
    if (KIND == 3) return 2 * ((size_t)l * g->Ls + k * g->Gs);
    return 2 * ((size_t)row * g->Gs + (size_t)l * g->Ls + k);
}
static size_t out_idx(const geo *g, int row, int l, size_t k)
{
    if (KIND == 3) return 2 * ((size_t)l * g->OLs + k * g->OGs);
    return 2 * ((size_t)row * g->OGs + (size_t)l * g->OLs + k);
}

static void reference(const geo *g, const double *in, long double *ref,
                      const double *tre, const double *tim)
{
    const long double PI = 3.141592653589793238462643383279502884L;
    for (size_t row = 0; row < g->rows; row++)
        for (size_t k = 0; k < g->lanes; k++) {
            long double xr[R], xi[R];
            for (int l = 0; l < R; l++) {
                size_t a = in_idx(g, (int)row, l, k);
                xr[l] = in[a]; xi[l] = in[a + 1];
                if (KIND == 0 && l > 0) {
                    long double c = tre[(size_t)l * g->lanes + k], s = tim[(size_t)l * g->lanes + k];
                    long double r = c * xr[l] - s * xi[l], i = c * xi[l] + s * xr[l];
                    xr[l] = r; xi[l] = i;
                }
            }
            for (int m = 0; m < R; m++) {
                long double ar = 0, ai = 0;
                for (int l = 0; l < R; l++) {
                    long double ang = (long double)DIRS * 2.0L * PI * (long double)((l * m) % R) / (long double)R;
                    long double c = cosl(ang), s = sinl(ang);
                    ar += xr[l] * c - xi[l] * s;
                    ai += xr[l] * s + xi[l] * c;
                }
                if (KIND == 1 && m > 0) {
                    long double c = tre[(size_t)m * g->lanes + k], s = tim[(size_t)m * g->lanes + k];
                    long double r = c * ar - s * ai, i = c * ai + s * ar;
                    ar = r; ai = i;
                }
                size_t o = (size_t)row * (2 * R * g->lanes) + ((size_t)m * g->lanes + k) * 2;
                ref[o] = ar; ref[o + 1] = ai;
            }
        }
}

typedef struct { int oob; int stray; int unwritten; double relerr; long nbitdiff_ref; } res;

static res run_one(zfn f, int vw, const geo *g, const double *in_src, size_t nin, size_t nout,
                   const long double *ref, const double *tre, const double *tim, int inplace,
                   double *out_copy)
{
    res r = { 0, 0, 0, 0, 0 };
    gbuf tb = (KIND <= 1) ? build_tab(vw, (int)g->lanes, tre, tim) : galloc(1);
    size_t nbuf = inplace ? (nin > nout ? nin : nout) : nout;
    gbuf ib = galloc(nin), ob = galloc(nbuf);
    memcpy(ib.p, in_src, nin * sizeof(double));
    for (size_t i = 0; i < nbuf; i++) ob.p[i] = sent();
    if (inplace) memcpy(ob.p + (nbuf - nin), in_src, nin * sizeof(double)); /* same end */
    double *zi = inplace ? ob.p + (nbuf - nin) : ib.p;
    double *zo = inplace ? zi : ob.p + (nbuf - nout);
    if (sigsetjmp(jb, 1) == 0)
        f(zi, 0, zo, 0, KIND <= 1 ? tb.p : 0, 0, g->Ls, g->Gs, g->OLs, g->OGs, g->count);
    else
        r.oob = 1;
    if (!r.oob) {
        /* written set */
        size_t nb = inplace ? nin : nout;
        char *w = calloc(nb, 1);
        long double num = 0, den = 0;
        for (size_t row = 0; row < g->rows; row++)
            for (int m = 0; m < R; m++)
                for (size_t k = 0; k < g->lanes; k++) {
                    size_t a = out_idx(g, (int)row, m, k);
                    w[a] = w[a + 1] = 1;
                    double vr = zo[a], vi = zo[a + 1];
                    size_t o = row * (2 * R * g->lanes) + ((size_t)m * g->lanes + k) * 2;
                    if (out_copy) { out_copy[o] = vr; out_copy[o + 1] = vi; }
                    if (is_sent(vr) || is_sent(vi)) r.unwritten++;
                    long double dr = vr - ref[o], di = vi - ref[o + 1];
                    num = fmaxl(num, fabsl(dr)); num = fmaxl(num, fabsl(di));
                    den = fmaxl(den, fabsl(ref[o])); den = fmaxl(den, fabsl(ref[o + 1]));
                    if (!(fabsl(dr) <= 1e-9L * (fabsl(ref[o]) + 1)) || !(fabsl(di) <= 1e-9L * (fabsl(ref[o + 1]) + 1)))
                        r.nbitdiff_ref++;
                }
        r.relerr = (double)(num / (den > 0 ? den : 1));
        if (!inplace)
            for (size_t i = 0; i < nb; i++)
                if (!w[i] && !is_sent(zo[i])) r.stray++;
        free(w);
    }
    gfree(tb); gfree(ib); gfree(ob);
    return r;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = segv; sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, 0); sigaction(SIGBUS, &sa, 0);
    static const int counts[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17 };
    int ncounts = (int)(sizeof counts / sizeof counts[0]);
    int fails_a = 0, fails_b = 0, bitdiff_cells = 0, total = 0;
    double worst_a = 0, worst_b = 0;
    char detail[4096]; detail[0] = 0; size_t dl = 0;
    for (int geo_v = 0; geo_v < (KIND == 3 ? 3 : 1); geo_v++)
    for (int inplace = 0; inplace < 2; inplace++)
    for (int ci = 0; ci < ncounts; ci++) {
        geo g;
        size_t lanes = (size_t)counts[ci];
        if (ROWLOOP) {
            g.rows = 3; g.lanes = lanes; g.Ls = lanes; g.OLs = lanes;
            g.Gs = (size_t)R * lanes + (inplace ? 0 : 5); g.OGs = inplace ? g.Gs : (size_t)R * lanes + 3;
            g.count = g.rows * lanes;
        } else if (KIND == 3) {
            g.rows = 1; g.lanes = lanes; g.count = lanes;
            g.Ls = 1; g.OLs = 1;
            g.Gs = (size_t)R + (geo_v ? 3 : 0); g.OGs = inplace ? g.Gs : (size_t)R + (geo_v ? 1 : 0);
            if (geo_v == 2) { /* the il2d TURN route: (Ls,Gs,OLs,OGs) = (1, rn, P, 1), out transposed */
                if (inplace) continue;
                g.Gs = (size_t)R + 2; g.OLs = lanes + 3; g.OGs = 1;
            }
        } else {
            g.rows = 1; g.lanes = lanes; g.count = lanes;
            g.Ls = lanes + 3; g.OLs = inplace ? g.Ls : lanes + 5; g.Gs = 0; g.OGs = 0;
        }
        size_t nin = 0, nout = 0;
        for (size_t row = 0; row < g.rows; row++)
            for (int l = 0; l < R; l++)
                for (size_t k = 0; k < g.lanes; k++) {
                    size_t a = in_idx(&g, (int)row, l, k) + 2, b = out_idx(&g, (int)row, l, k) + 2;
                    if (a > nin) nin = a;
                    if (b > nout) nout = b;
                }
        if (inplace && nin != nout) continue;
        double *in = malloc(nin * sizeof(double));
        for (size_t i = 0; i < nin; i++) in[i] = rnd();
        double *tre = malloc((size_t)R * lanes * sizeof(double)), *tim = malloc((size_t)R * lanes * sizeof(double));
        for (size_t i = 0; i < (size_t)R * lanes; i++) { tre[i] = rnd(); tim[i] = rnd(); }
        size_t nref = g.rows * 2 * R * g.lanes;
        long double *ref = malloc(nref * sizeof(long double));
        reference(&g, in, ref, tre, tim);
        double *oa = malloc(nref * sizeof(double)), *ob = malloc(nref * sizeof(double));
        res ra = run_one((zfn)SYM_A, 4, &g, in, nin, nout, ref, tre, tim, inplace, oa);
        res rb = run_one((zfn)SYM_B, 8, &g, in, nin, nout, ref, tre, tim, inplace, ob);
        long nbd = 0;
        if (!ra.oob && !rb.oob)
            for (size_t i = 0; i < nref; i++) if (memcmp(&oa[i], &ob[i], 8)) nbd++;
        total++;
        int bad_a = ra.oob || ra.stray || ra.unwritten || ra.relerr > 1e-12;
        int bad_b = rb.oob || rb.stray || rb.unwritten || rb.relerr > 1e-12;
        fails_a += bad_a; fails_b += bad_b; bitdiff_cells += nbd > 0;
        if (ra.relerr > worst_a) worst_a = ra.relerr;
        if (rb.relerr > worst_b) worst_b = rb.relerr;
        if ((bad_a || bad_b || nbd) && dl < sizeof(detail) - 200)
            dl += (size_t)snprintf(detail + dl, sizeof(detail) - dl,
                   " [g%d %s cnt=%zu A:%s%s%s err=%.1e B:%s%s%s err=%.1e badcols=%ld bitdiff=%ld/%zu]",
                   geo_v, inplace ? "inpl" : "oop", lanes,
                   ra.oob ? "OOB " : "", ra.stray ? "STRAY " : "", ra.unwritten ? "UNWR " : "", ra.relerr,
                   rb.oob ? "OOB " : "", rb.stray ? "STRAY " : "", rb.unwritten ? "UNWR " : "", rb.relerr,
                   rb.nbitdiff_ref / 2, nbd, nref);
        free(in); free(tre); free(tim); free(ref); free(oa); free(ob);
    }
    printf("%s\t%s\tA_fail=%d/%d\tB_fail=%d/%d\tbitdiff_cells=%d\tworstA=%.2e\tworstB=%.2e\t%s\n",
           fails_b ? "B_WRONG" : (fails_a ? "A_WRONG" : (bitdiff_cells ? "OK_ULP" : "OK_BITEXACT")),
           STR(SYM_B), fails_a, total, fails_b, total, bitdiff_cells, worst_a, worst_b, detail);
    return 0;
}
