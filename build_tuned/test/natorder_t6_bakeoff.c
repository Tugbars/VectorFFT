/* natorder_t6_bakeoff.c — T6: UNIFIED natural-order bakeoff. Race every surviving candidate on the
 * same cells, same harness, same clock -> per-cell winner map (the profile Phase-1's calibrator will
 * later regenerate). Chains/variants/DIF hardcoded VERBATIM from spike_wisdom.txt (v6); chain B =
 * hand-injected PALINDROMIC factorization (digit reversal = involution -> pair-swap), raced under 3
 * uniform variant profiles (T1S/FLAT/LOG3, keep best) since wisdom has no calibrated variants for it.
 * Candidates per cell:
 *   FREE      nf==1 chains: already natural (baseline = answer)
 *   LEAF-IP   N<=128: aliased no-restrict n1_oop leaf (natural, 1 buffer)
 *   PURE-cyc  wisdom FFT + in-place cycle-following row perm      (exact bolt-on)
 *   PURE-gat  wisdom FFT + gather-to-scratch + copyback           (exact bolt-on)
 *   SCR-t1    wisdom FFT with last stage -> scatter terminator, t1_oop-style: j-outer comb writes +
 *             FULL K-replicated twiddle table streamed + per-row cmul (buildable TODAY)     [EST]
 *   SCR-t1-T  same but L1-TILED (COBRA): TQ blocks staged through a <=32KB buffer           [EST]
 *   SCR-t1s   same scatter, TINY scalar table (2 doubles/row-group) = future t1s_oop        [EST]
 *   PSWAP     injected palindromic FFT (best of 3 profiles) + involution pair-swap pass (exact)
 * [EST] composition: total = t_fftA + t_kernel - t_ippass (terminator REPLACES the last stage's
 * memory pass; P0 convention). Cells whose calibrated last stage is already FLAT double-count the
 * table slightly (conservative) — flagged 'F' in the table. Correctness: lane-0 vs naive DFT for
 * A-order (auto-detected perm), pair-swap output, LEAF-IP output.
 * Build: python build.py --src test/natorder_t6_bakeoff.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include <stddef.h>
#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* no-restrict leaf copies for LEAF-IP (T3-validated trick) */
#define __restrict__
#define __restrict
#define radix16_n1_oop_fwd_avx2_UG_UG nr_radix16
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix16_n1_oop_avx2.c"
#undef radix16_n1_oop_fwd_avx2_UG_UG
#define radix32_n1_oop_fwd_avx2_UG_UG nr_radix32
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix32_n1_oop_avx2.c"
#undef radix32_n1_oop_fwd_avx2_UG_UG
#define radix64_n1_oop_fwd_avx2_UG_UG nr_radix64
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix64_n1_oop_avx2.c"
#undef radix64_n1_oop_fwd_avx2_UG_UG
#define radix128_n1_oop_fwd_avx2_UG_UG nr_radix128
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix128_n1_oop_avx2.c"
#undef radix128_n1_oop_fwd_avx2_UG_UG
typedef void (*leaf_fn)(const double *, const double *, double *, double *, const double *, const double *,
                        size_t, size_t, size_t, size_t, size_t);
static leaf_fn NRLEAF(int N) { return N == 16 ? nr_radix16 : N == 32 ? nr_radix32
                                                         : N == 64   ? nr_radix64
                                                         : N == 128  ? nr_radix128
                                                                     : NULL; }

static vfft_proto_registry_t REG;
static double qpc_ns(void)
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static void refill(double *re, double *im, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        re[i] = (double)((i * 2654435761u) & 1023) / 1024.0 - 0.5;
        im[i] = (double)((i * 40503u) & 1023) / 1024.0 - 0.5;
    }
}
static void rescale(double *re, double *im, size_t n)
{
    double mx = 0;
    for (size_t i = 0; i < n; i += 13)
    {
        double a = fabs(re[i]);
        if (a > mx)
            mx = a;
    }
    if (mx > 1e80 || mx < 1e-80)
    {
        double s = mx > 0 ? 1.0 / mx : 1.0;
        for (size_t i = 0; i < n; i++)
        {
            re[i] *= s;
            im[i] *= s;
        }
    }
}

/* mixed-radix digit reversal (little-endian digits -> big-endian slots, same factor order) */
static void mk_perm(int N, const int *f, int nf, int *perm)
{
    for (int n = 0; n < N; n++)
    {
        int t = n, slot = 0, rem = N;
        for (int s = 0; s < nf; s++)
        {
            int d = t % f[s];
            t /= f[s];
            rem /= f[s];
            slot += d * rem;
        }
        perm[n] = slot;
    }
}
static void inv_perm(int N, const int *p, int *ip)
{
    for (int n = 0; n < N; n++)
        ip[p[n]] = n;
}
static void naive_dft_lane0(const double *re, const double *im, int N, size_t K, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++)
    {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++)
        {
            double a = -2.0 * M_PI * (double)k * n / N, c = cos(a), s = sin(a);
            double xr = re[(size_t)n * K], xi = im[(size_t)n * K];
            sr += xr * c - xi * s;
            si += xr * s + xi * c;
        }
        Xr[k] = sr;
        Xi[k] = si;
    }
}

/* ─── reorder kernels (rows = K doubles, split planes) ─── */
typedef struct
{
    double *dr, *di;
    const double *sr, *si, *twr, *twi;
    const int *map;
    size_t K;
    int N, R, P;
} kctx_t;
/* cmul a K-row by a K-wide table slice (FLAT/t1_oop model) */
static inline void row_cmul_tab(double *dr, double *di, const double *sr, const double *si,
                                const double *twr, const double *twi, size_t K)
{
    for (size_t c = 0; c < K; c += 4)
    {
        __m256d xr = _mm256_loadu_pd(sr + c), xi = _mm256_loadu_pd(si + c);
        __m256d wr = _mm256_loadu_pd(twr + c), wi = _mm256_loadu_pd(twi + c);
        _mm256_storeu_pd(dr + c, _mm256_sub_pd(_mm256_mul_pd(xr, wr), _mm256_mul_pd(xi, wi)));
        _mm256_storeu_pd(di + c, _mm256_add_pd(_mm256_mul_pd(xr, wi), _mm256_mul_pd(xi, wr)));
    }
}
/* cmul a K-row by ONE scalar twiddle (T1S/t1s_oop model) */
static inline void row_cmul_scal(double *dr, double *di, const double *sr, const double *si,
                                 double wr_, double wi_, size_t K)
{
    __m256d wr = _mm256_set1_pd(wr_), wi = _mm256_set1_pd(wi_);
    for (size_t c = 0; c < K; c += 4)
    {
        __m256d xr = _mm256_loadu_pd(sr + c), xi = _mm256_loadu_pd(si + c);
        _mm256_storeu_pd(dr + c, _mm256_sub_pd(_mm256_mul_pd(xr, wr), _mm256_mul_pd(xi, wi)));
        _mm256_storeu_pd(di + c, _mm256_add_pd(_mm256_mul_pd(xr, wi), _mm256_mul_pd(xi, wr)));
    }
}
/* in-place pass proxy (what the replaced last stage costs memory-wise) */
static void k_ippass(void *v)
{
    kctx_t *p = v;
    size_t n = (size_t)p->N * p->K;
    for (size_t i = 0; i < n; i++)
    {
        p->dr[i] = p->dr[i] * 1.000000001 + 1e-30;
        p->di[i] = p->di[i] * 1.000000001 + 1e-30;
    }
}
/* SCR-t1: j-outer scatter + FULL table stream + cmul. table indexed in write order (sequential). */
static void k_scat_t1(void *v)
{
    kctx_t *p = v;
    size_t K = p->K;
    int P = p->P, R = p->R;
    for (int j = 0; j < R; j++)
    {
        double *wr = p->dr + (size_t)j * P * K, *wi = p->di + (size_t)j * P * K;
        const double *tr = p->twr + (size_t)j * P * K, *ti = p->twi + (size_t)j * P * K;
        for (int q = 0; q < P; q++)
        {
            int m = p->map[q + j * P];
            row_cmul_tab(wr + (size_t)q * K, wi + (size_t)q * K, p->sr + (size_t)m * K, p->si + (size_t)m * K,
                         tr + (size_t)q * K, ti + (size_t)q * K, K);
        }
    }
}
/* SCR-t1s: same scatter, scalar twiddle per row (tiny table N*2 doubles) */
static void k_scat_t1s(void *v)
{
    kctx_t *p = v;
    size_t K = p->K;
    int P = p->P, R = p->R;
    for (int j = 0; j < R; j++)
    {
        double *wr = p->dr + (size_t)j * P * K, *wi = p->di + (size_t)j * P * K;
        const double *tr = p->twr + (size_t)j * P, *ti = p->twi + (size_t)j * P;
        for (int q = 0; q < P; q++)
        {
            int m = p->map[q + j * P];
            row_cmul_scal(wr + (size_t)q * K, wi + (size_t)q * K, p->sr + (size_t)m * K, p->si + (size_t)m * K,
                          tr[q], ti[q], K);
        }
    }
}
/* SCR-t1-TILED: stage TQ scattered blocks through an L1 buffer, then write contiguous TQ-row runs
 * per comb stream (with full-table cmul on the way out). buf <= 32KB. */
#define TQ 16
static double g_bufr[TQ * 64 * 64], g_bufi[TQ * 64 * 64]; /* worst case TQ*R*K = 16*64*... cap: use heap if needed */
static void k_scat_t1_tiled(void *v)
{
    kctx_t *p = v;
    size_t K = p->K;
    int P = p->P, R = p->R;
    /* map natural comb q -> source block base b(q) = map[q + 0*P] (block rows are contiguous) */
    for (int q0 = 0; q0 < P; q0 += TQ)
    {
        int tq = (q0 + TQ <= P) ? TQ : (P - q0);
        for (int t = 0; t < tq; t++)
        {
            int b = p->map[q0 + t]; /* block base row of q0+t (j=0 leg) */
            memcpy(g_bufr + (size_t)t * R * K, p->sr + (size_t)b * K, (size_t)R * K * 8);
            memcpy(g_bufi + (size_t)t * R * K, p->si + (size_t)b * K, (size_t)R * K * 8);
        }
        for (int j = 0; j < R; j++)
        {
            double *wr = p->dr + ((size_t)j * P + q0) * K, *wi = p->di + ((size_t)j * P + q0) * K;
            const double *tr = p->twr + ((size_t)j * P + q0) * K, *ti = p->twi + ((size_t)j * P + q0) * K;
            for (int t = 0; t < tq; t++)
                row_cmul_tab(wr + (size_t)t * K, wi + (size_t)t * K,
                             g_bufr + ((size_t)t * R + j) * K, g_bufi + ((size_t)t * R + j) * K,
                             tr + (size_t)t * K, ti + (size_t)t * K, K);
        }
    }
}
/* PURE cycle-following (min-of-cycle rule), one K-row temp */
typedef struct
{
    double *dr, *di, *tr, *ti;
    const int *map;
    size_t K;
    int N;
} cyc_t;
static void k_cycle(void *v)
{
    cyc_t *c = v;
    size_t K = c->K;
    const int *M = c->map;
    for (int start = 0; start < c->N; start++)
    {
        int m = M[start], mn = start;
        while (m != start)
        {
            if (m < mn)
            {
                mn = -1;
                break;
            }
            m = M[m];
        }
        if (mn < 0 || M[start] == start)
            continue;
        memcpy(c->tr, c->dr + (size_t)start * K, K * 8);
        memcpy(c->ti, c->di + (size_t)start * K, K * 8);
        int cur = start;
        for (;;)
        {
            int nxt = M[cur];
            if (nxt == start)
                break;
            memcpy(c->dr + (size_t)cur * K, c->dr + (size_t)nxt * K, K * 8);
            memcpy(c->di + (size_t)cur * K, c->di + (size_t)nxt * K, K * 8);
            cur = nxt;
        }
        memcpy(c->dr + (size_t)cur * K, c->tr, K * 8);
        memcpy(c->di + (size_t)cur * K, c->ti, K * 8);
    }
}
/* ── UPPER-BOUND kernels (T7): plan-time move lists + software prefetch + AVX row ops ── */
static inline void row_mov(double *dr, double *di, const double *sr, const double *si, size_t K)
{
    for (size_t c = 0; c < K; c += 4)
    {
        _mm256_storeu_pd(dr + c, _mm256_loadu_pd(sr + c));
        _mm256_storeu_pd(di + c, _mm256_loadu_pd(si + c));
    }
}
static inline void row_swap(double *ar, double *ai, double *br, double *bi, size_t K)
{
    for (size_t c = 0; c < K; c += 4)
    {
        __m256d xr = _mm256_loadu_pd(ar + c), xi = _mm256_loadu_pd(ai + c);
        __m256d yr = _mm256_loadu_pd(br + c), yi = _mm256_loadu_pd(bi + c);
        _mm256_storeu_pd(ar + c, yr);
        _mm256_storeu_pd(ai + c, yi);
        _mm256_storeu_pd(br + c, xr);
        _mm256_storeu_pd(bi + c, xi);
    }
}
/* cycles flattened at plan time: each cycle = idx list, -1 terminator, -2 end */
typedef struct
{
    double *dr, *di, *tr, *ti;
    const int *list;
    size_t K;
} cub_t;
static void k_cycle_ub(void *v)
{
    cub_t *c = v;
    size_t K = c->K;
    const int *L = c->list;
    while (*L != -2)
    {
        const int *s = L;
        int len = 0;
        while (L[len] != -1)
            len++;
        memcpy(c->tr, c->dr + (size_t)s[0] * K, K * 8);
        memcpy(c->ti, c->di + (size_t)s[0] * K, K * 8);
        for (int i = 0; i < len - 1; i++)
        {
            if (i + 8 < len)
                _mm_prefetch((const char *)(c->dr + (size_t)s[i + 8] * K), _MM_HINT_T0);
            row_mov(c->dr + (size_t)s[i] * K, c->di + (size_t)s[i] * K,
                    c->dr + (size_t)s[i + 1] * K, c->di + (size_t)s[i + 1] * K, K);
        }
        memcpy(c->dr + (size_t)s[len - 1] * K, c->tr, K * 8);
        memcpy(c->di + (size_t)s[len - 1] * K, c->ti, K * 8);
        L += len + 1;
    }
}
static int *mk_cycle_list(int N, const int *M)
{
    int *lst = malloc((size_t)(2 * N + 8) * 4), *w = lst;
    char *vis = calloc(N, 1);
    for (int st = 0; st < N; st++)
    {
        if (vis[st] || M[st] == st)
        {
            vis[st] = 1;
            continue;
        }
        int cur = st;
        while (!vis[cur])
        {
            vis[cur] = 1;
            *w++ = cur;
            cur = M[cur];
        }
        *w++ = -1;
    }
    *w = -2;
    return lst;
}
/* cycle-UB2: INTERLEAVE up to 8 independent cycles (8 dependency chains in flight) — attacks the
 * serial-chain latency that keeps cycle-UB at ~1.45x the in-place-pass floor. 8 temp row-pairs. */
typedef struct
{
    double *dr, *di, *tmp; /* tmp = 8*2*K doubles */
    const int *list;
    size_t K;
} cub2_t;
static void k_cycle_ub2(void *v)
{
    cub2_t *c = v;
    size_t K = c->K;
    const int *L = c->list;
    const int *seg[8];
    int len[8], pos[8];
    for (;;)
    {
        int na = 0;
        while (na < 8 && *L != -2)
        { /* grab up to 8 cycles */
            seg[na] = L;
            int l = 0;
            while (L[l] != -1)
                l++;
            len[na] = l;
            pos[na] = 0;
            L += l + 1;
            na++;
        }
        if (!na)
            break;
        for (int t = 0; t < na; t++)
        { /* save heads */
            memcpy(c->tmp + (size_t)(2 * t) * K, c->dr + (size_t)seg[t][0] * K, K * 8);
            memcpy(c->tmp + (size_t)(2 * t + 1) * K, c->di + (size_t)seg[t][0] * K, K * 8);
        }
        int live = 0;
        for (int t = 0; t < na; t++)
            if (len[t] > 1)
                live++;
        while (live)
        { /* round-robin one step per active cycle */
            for (int t = 0; t < na; t++)
            {
                if (pos[t] >= len[t] - 1)
                    continue;
                int i = pos[t];
                const int *s = seg[t];
                if (i + 4 < len[t])
                {
                    _mm_prefetch((const char *)(c->dr + (size_t)s[i + 4] * K), _MM_HINT_T0);
                    _mm_prefetch((const char *)(c->di + (size_t)s[i + 4] * K), _MM_HINT_T0);
                }
                row_mov(c->dr + (size_t)s[i] * K, c->di + (size_t)s[i] * K,
                        c->dr + (size_t)s[i + 1] * K, c->di + (size_t)s[i + 1] * K, K);
                if (++pos[t] >= len[t] - 1)
                    live--;
            }
        }
        for (int t = 0; t < na; t++)
        { /* close cycles from temps */
            memcpy(c->dr + (size_t)seg[t][len[t] - 1] * K, c->tmp + (size_t)(2 * t) * K, K * 8);
            memcpy(c->di + (size_t)seg[t][len[t] - 1] * K, c->tmp + (size_t)(2 * t + 1) * K, K * 8);
        }
    }
}
/* pair list for involutions: flat [a0,b0,a1,b1,...], -2 end */
typedef struct
{
    double *dr, *di;
    const int *list;
    size_t K;
} pub_t;
static void k_pswap_ub(void *v)
{
    pub_t *p = v;
    size_t K = p->K;
    const int *L = p->list;
    int np = 0;
    while (L[2 * np] != -2)
        np++;
    for (int i = 0; i < np; i++)
    {
        if (i + 4 < np)
        {
            _mm_prefetch((const char *)(p->dr + (size_t)L[2 * (i + 4) + 1] * K), _MM_HINT_T0);
            _mm_prefetch((const char *)(p->di + (size_t)L[2 * (i + 4) + 1] * K), _MM_HINT_T0);
        }
        row_swap(p->dr + (size_t)L[2 * i] * K, p->di + (size_t)L[2 * i] * K,
                 p->dr + (size_t)L[2 * i + 1] * K, p->di + (size_t)L[2 * i + 1] * K, K);
    }
}
static int *mk_pair_list(int N, const int *M)
{
    int *lst = malloc((size_t)(2 * N + 8) * 4), *w = lst;
    for (int n = 0; n < N; n++)
        if (M[n] > n)
        {
            *w++ = n;
            *w++ = M[n];
        }
    *w = -2;
    return lst;
}
/* CELL-TRANSPOSE (transpose.h-style cache-oblivious recursion, cell = K-double row): for nf==2
 * chains the digit reversal IS the f0 x f1 cell-grid transpose. OOP into scratch + copyback. */
typedef struct
{
    double *dr, *di, *gr, *gi;
    size_t K;
    int f0, f1, N;
} ct2_t;
static void _ct2_rec(ct2_t *c, int r0, int r1, int c0, int c1)
{
    if ((r1 - r0) <= 8 && (c1 - c0) <= 8)
    {
        size_t K = c->K;
        for (int i = r0; i < r1; i++)
            for (int j = c0; j < c1; j++)
                row_mov(c->gr + ((size_t)j * c->f0 + i) * K, c->gi + ((size_t)j * c->f0 + i) * K,
                        c->dr + ((size_t)i * c->f1 + j) * K, c->di + ((size_t)i * c->f1 + j) * K, K);
        return;
    }
    if ((r1 - r0) >= (c1 - c0))
    {
        int m = (r0 + r1) / 2;
        _ct2_rec(c, r0, m, c0, c1);
        _ct2_rec(c, m, r1, c0, c1);
    }
    else
    {
        int m = (c0 + c1) / 2;
        _ct2_rec(c, r0, r1, c0, m);
        _ct2_rec(c, r0, r1, m, c1);
    }
}
static void k_ct2(void *v)
{
    ct2_t *c = v;
    _ct2_rec(c, 0, c->f0, 0, c->f1);
    memcpy(c->dr, c->gr, (size_t)c->N * c->K * 8);
    memcpy(c->di, c->gi, (size_t)c->N * c->K * 8);
}
/* PURE gather + copyback */
typedef struct
{
    double *dr, *di, *gr, *gi;
    const int *map;
    size_t K;
    int N;
} gat_t;
static void k_gather(void *v)
{
    gat_t *g = v;
    size_t K = g->K;
    const int *M = g->map;
    size_t n = (size_t)g->N * K;
    for (int k = 0; k < g->N; k++)
    {
        memcpy(g->gr + (size_t)k * K, g->dr + (size_t)M[k] * K, K * 8);
        memcpy(g->gi + (size_t)k * K, g->di + (size_t)M[k] * K, K * 8);
    }
    memcpy(g->dr, g->gr, n * 8);
    memcpy(g->di, g->gi, n * 8);
}
/* PAIR-SWAP (involution perm): swap rows n <-> perm[n] once each */
typedef struct
{
    double *dr, *di, *tr, *ti;
    const int *map;
    size_t K;
    int N;
} psw_t;
static void k_pswap(void *v)
{
    psw_t *p = v;
    size_t K = p->K;
    const int *M = p->map;
    for (int n = 0; n < p->N; n++)
    {
        int m = M[n];
        if (m <= n)
            continue;
        memcpy(p->tr, p->dr + (size_t)n * K, K * 8);
        memcpy(p->dr + (size_t)n * K, p->dr + (size_t)m * K, K * 8);
        memcpy(p->dr + (size_t)m * K, p->tr, K * 8);
        memcpy(p->ti, p->di + (size_t)n * K, K * 8);
        memcpy(p->di + (size_t)n * K, p->di + (size_t)m * K, K * 8);
        memcpy(p->di + (size_t)m * K, p->ti, K * 8);
    }
}

/* ─── timing helpers ─── */
typedef void (*kfn)(void *);
static double time_kernel(kfn fn, void *ctx, double bytes)
{
    int inner = (int)(2.0e7 / (bytes * 0.15)) + 1;
    if (inner < 3)
        inner = 3;
    if (inner > 4000)
        inner = 4000;
    {
        double t0 = qpc_ns();
        for (int i = 0; i < inner; i++)
            fn(ctx);
        (void)t0;
    } /* warm-up round */
    double sum = 0;
    for (int o = 0; o < 5; o++)
    {
        Sleep(150); /* pacing (cool) */
        double t0 = qpc_ns();
        for (int i = 0; i < inner; i++)
            fn(ctx);
        sum += (qpc_ns() - t0) / inner;
    }
    return sum / 5.0;
}
static double time_fft(stride_plan_t *pl, double *re, double *im, size_t n, size_t K)
{
    int inner = (int)(1.6e7 / ((double)n * 4.0)) + 8;
    inner &= ~7;
    if (inner < 8)
        inner = 8;
    if (inner > 60000)
        inner = 60000;
    refill(re, im, n);
    for (int r = 0; r < 8; r++)
        vfft_proto_execute_fwd(pl, re, im, K); /* warm-up */
    double sum = 0;
    for (int o = 0; o < 5; o++)
    {
        Sleep(150);
        refill(re, im, n);
        double acc = 0;
        int done = 0;
        while (done < inner)
        {
            double t0 = qpc_ns();
            for (int r = 0; r < 8; r++)
                vfft_proto_execute_fwd(pl, re, im, K);
            acc += qpc_ns() - t0;
            done += 8;
            rescale(re, im, n);
        }
        sum += acc / done;
    }
    return sum / 5.0;
}
static double time_leafip(leaf_fn f, double *re, double *im, size_t n, size_t K)
{
    int inner = (int)(1.6e7 / ((double)n * 2.0)) + 8;
    inner &= ~7;
    if (inner < 8)
        inner = 8;
    if (inner > 200000)
        inner = 200000;
    refill(re, im, n);
    for (int r = 0; r < 8; r++)
        f(re, im, re, im, NULL, NULL, K, 1, K, 1, K); /* warm-up */
    double sum = 0;
    for (int o = 0; o < 5; o++)
    {
        Sleep(150);
        refill(re, im, n);
        double acc = 0;
        int done = 0;
        while (done < inner)
        {
            double t0 = qpc_ns();
            for (int r = 0; r < 8; r++)
                f(re, im, re, im, NULL, NULL, K, 1, K, 1, K);
            acc += qpc_ns() - t0;
            done += 8;
            rescale(re, im, n);
        }
        sum += acc / done;
    }
    return sum / 5.0;
}

typedef struct
{
    int N;
    size_t K;
    int nfA, fA[8], vA[8], difA;
    int nfB, fB[8];
    const char *lastv;
} cell_t;
static const cell_t CELLS[] = {
    {16, 4, 1, {16}, {0}, 0, 0, {0}, "-"},
    {32, 4, 1, {32}, {0}, 0, 0, {0}, "-"},
    {64, 4, 1, {64}, {0}, 0, 0, {0}, "-"},
    {128, 4, 2, {8, 16}, {0, 2}, 0, 3, {4, 8, 4}, "T1S"},
    {64, 64, 2, {4, 16}, {1, 0}, 1, 2, {8, 8}, "F"},
    {128, 64, 2, {8, 16}, {0, 2}, 0, 3, {4, 8, 4}, "T1S"},
    {1024, 4, 2, {64, 16}, {0, 2}, 0, 3, {8, 16, 8}, "T1S"},
    {1024, 32, 4, {4, 4, 8, 8}, {0, 2, 2, 2}, 0, 3, {8, 16, 8}, "T1S"},
    {256, 256, 3, {4, 4, 16}, {0, 2, 2}, 0, 2, {16, 16}, "T1S"},
    {4096, 4, 4, {4, 4, 8, 32}, {0, 0, 1, 0}, 1, 4, {8, 8, 8, 8}, "F"},
    {4096, 32, 5, {4, 4, 4, 8, 8}, {0, 2, 2, 2, 2}, 0, 4, {8, 8, 8, 8}, "T1S"},
    {4096, 256, 4, {4, 4, 4, 64}, {0, 2, 2, 2}, 0, 4, {8, 8, 8, 8}, "T1S"},
};
#define NCELLS (sizeof CELLS / sizeof CELLS[0])

static void run_cell(const cell_t *C)
{
    int N = C->N;
    size_t K = C->K, n = (size_t)N * K;
    double bytes = (double)n * 16.0 * 2.0;
    printf("\n== N=%-4d K=%-3zu  A=", N, K);
    for (int i = 0; i < C->nfA; i++)
        printf("%d%s", C->fA[i], i < C->nfA - 1 ? "." : "");
    printf("%s lastv=%s", C->difA ? "(DIF)" : "", C->lastv);
    if (C->nfB)
    {
        printf("  B=");
        for (int i = 0; i < C->nfB; i++)
            printf("%d%s", C->fB[i], i < C->nfB - 1 ? "." : "");
    }
    printf(" ==\n");

    stride_plan_t *pA = vfft_proto_plan_create_ex(N, K, C->fA, C->vA, C->nfA, C->difA, &REG);
    if (!pA)
    {
        printf("  plan A NULL!\n");
        return;
    }
    double *re = _aligned_malloc(n * 8, 64), *im = _aligned_malloc(n * 8, 64);
    double *sr = _aligned_malloc(n * 8, 64), *si = _aligned_malloc(n * 8, 64);
    double *dr = _aligned_malloc(n * 8, 64), *di = _aligned_malloc(n * 8, 64);
    double *Xr = malloc((size_t)N * 8), *Xi = malloc((size_t)N * 8);

    /* correctness: A order (auto-detect among 4 perm candidates), lane 0 vs naive */
    srand(66 + N + (int)K);
    for (size_t i = 0; i < n; i++)
    {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(sr, re, n * 8);
    memcpy(si, im, n * 8);
    naive_dft_lane0(re, im, N, K, Xr, Xi);
    vfft_proto_execute_fwd(pA, re, im, K);
    int *pf = malloc(N * 4), *ipf = malloc(N * 4), *pr = malloc(N * 4), *ipr = malloc(N * 4);
    int frev[8];
    for (int i = 0; i < C->nfA; i++)
        frev[i] = C->fA[C->nfA - 1 - i];
    mk_perm(N, C->fA, C->nfA, pf);
    inv_perm(N, pf, ipf);
    mk_perm(N, frev, C->nfA, pr);
    inv_perm(N, pr, ipr);
    const int *cand[4] = {pf, ipf, pr, ipr};
    int bi = -1;
    double be = 1e30;
    for (int ci = 0; ci < 4; ci++)
    {
        double e = 0;
        for (int k = 0; k < N; k++)
        {
            double d1 = fabs(re[(size_t)cand[ci][k] * K] - Xr[k]), d2 = fabs(im[(size_t)cand[ci][k] * K] - Xi[k]);
            if (d1 > e)
                e = d1;
            if (d2 > e)
                e = d2;
        }
        if (e < be)
        {
            be = e;
            bi = ci;
        }
    }
    double sc = 0;
    for (int k = 0; k < N; k++)
        if (fabs(Xr[k]) > sc)
            sc = fabs(Xr[k]);
    int okA = (be / (sc > 0 ? sc : 1)) < 1e-8;
    int *MA = (int *)cand[bi], *IMA = malloc(N * 4);
    inv_perm(N, MA, IMA);
    (void)IMA;

    /* baseline A + kernels */
    double tA = time_fft(pA, re, im, n, K);
    int R = C->fA[C->nfA - 1], P = N / R;
    for (size_t i = 0; i < n; i++)
    {
        sr[i] = 1.0 + (double)(i & 255);
        si[i] = 2.0 + (double)(i & 127);
        dr[i] = 0;
        di[i] = 0;
    }
    double *twr = _aligned_malloc(n * 8, 64), *twi = _aligned_malloc(n * 8, 64); /* FLAT table (N*K per plane) */
    for (size_t i = 0; i < n; i++)
    {
        twr[i] = 0.8;
        twi[i] = 0.6;
    }
    double *tsr = _aligned_malloc((size_t)N * 8, 64), *tsi = _aligned_malloc((size_t)N * 8, 64); /* tiny table */
    for (int i = 0; i < N; i++)
    {
        tsr[i] = 0.8;
        tsi[i] = 0.6;
    }
    kctx_t kc = {dr, di, sr, si, twr, twi, MA, K, N, R, P};
    double t_ip = time_kernel(k_ippass, &kc, bytes * 0.5);
    double t_s1 = time_kernel(k_scat_t1, &kc, bytes * 1.5);
    kctx_t kts = kc;
    kts.twr = tsr;
    kts.twi = tsi;
    double t_s1s = time_kernel(k_scat_t1s, &kts, bytes);
    double t_s1t = ((size_t)TQ * R * K <= sizeof(g_bufr) / 8) ? time_kernel(k_scat_t1_tiled, &kc, bytes * 1.5) : -1;
    double *tr = _aligned_malloc(K * 8, 64), *ti = _aligned_malloc(K * 8, 64);
    cyc_t cy = {dr, di, tr, ti, MA, K, N};
    double t_cy = time_kernel(k_cycle, &cy, bytes);
    int *clist = mk_cycle_list(N, MA);
    cub_t cu = {dr, di, tr, ti, clist, K};
    double t_cyu = time_kernel(k_cycle_ub, &cu, bytes);
    /* ub2: 8-way interleaved cycles; one-shot correctness vs plain cycle first */
    double *tmp8 = _aligned_malloc((size_t)16 * K * 8, 64);
    double t_cy2 = -1;
    {
        size_t nn = (size_t)N * K;
        double *a1 = malloc(nn * 8), *a2 = malloc(nn * 8), *b1 = malloc(nn * 8), *b2 = malloc(nn * 8);
        for (size_t i = 0; i < nn; i++)
        {
            a1[i] = b1[i] = (double)(i % 977);
            a2[i] = b2[i] = (double)(i % 991);
        }
        cyc_t cv = {a1, a2, tr, ti, MA, K, N};
        k_cycle(&cv);
        cub2_t c2v = {b1, b2, tmp8, clist, K};
        k_cycle_ub2(&c2v);
        int same = !memcmp(a1, b1, nn * 8) && !memcmp(a2, b2, nn * 8);
        free(a1);
        free(a2);
        free(b1);
        free(b2);
        if (same)
        {
            cub2_t c2 = {dr, di, tmp8, clist, K};
            t_cy2 = time_kernel(k_cycle_ub2, &c2, bytes);
        }
        else
            printf("  [cycle-UB2 CORRECTNESS FAIL — skipped]\n");
    }
    _aligned_free(tmp8);
    free(clist);
    double *gr = _aligned_malloc(n * 8, 64), *gi = _aligned_malloc(n * 8, 64);
    gat_t ga = {dr, di, gr, gi, MA, K, N};
    double t_ga = time_kernel(k_gather, &ga, bytes * 1.5);
    double t_ct = -1;
    if (C->nfA == 2)
    {
        ct2_t ct = {dr, di, gr, gi, K, C->fA[0], C->fA[1], N};
        /* sanity: transpose map == MA? natural[d0*f1+d1]=scrambled[d0+d1*f0]: check few */
        int okT = 1;
        for (int i = 0; i < C->fA[0] && okT; i++)      /* MA[d0+d1*f0] == d0*f1+d1 */
            for (int j = 0; j < C->fA[1]; j++)
                if (MA[i + j * C->fA[0]] != (i * C->fA[1] + j))
                {
                    okT = 0;
                    break;
                }
        if (okT)
            t_ct = time_kernel(k_ct2, &ct, bytes * 1.5);
    }

    /* chain B: 3 uniform profiles, keep best; then pair-swap (+ correctness) */
    double tB = -1, t_ps = -1;
    int bprof = -1, okB = 1;
    if (C->nfB)
    {
        int prof[3] = {2, 0, 1};
        const char *pn[3] = {"T1S", "FLAT", "LOG3"};
        for (int pi = 0; pi < 3; pi++)
        {
            int vb[8];
            for (int s = 0; s < C->nfB; s++)
                vb[s] = prof[pi];
            vb[0] = 0;
            stride_plan_t *pB = vfft_proto_plan_create_ex(N, K, C->fB, vb, C->nfB, 0, &REG);
            if (!pB)
                continue;
            double t = time_fft(pB, re, im, n, K);
            if (tB < 0 || t < tB)
            {
                tB = t;
                bprof = pi;
            }
            vfft_proto_plan_destroy(pB);
        }
        int *pB_ = malloc(N * 4);
        mk_perm(N, C->fB, C->nfB, pB_);
        int invol = 1;
        for (int m = 0; m < N; m++)
            if (pB_[pB_[m]] != m)
            {
                invol = 0;
                break;
            }
        if (invol && tB > 0)
        {
            /* correctness: run B fwd once, pair-swap, compare natural vs naive */
            int vb[8];
            for (int s = 0; s < C->nfB; s++)
                vb[s] = prof[bprof];
            vb[0] = 0;
            stride_plan_t *pB = vfft_proto_plan_create_ex(N, K, C->fB, vb, C->nfB, 0, &REG);
            memcpy(re, sr, 0); /* noop */
            srand(66 + N + (int)K);
            for (size_t i = 0; i < n; i++)
            {
                re[i] = (double)rand() / RAND_MAX - 0.5;
                im[i] = (double)rand() / RAND_MAX - 0.5;
            }
            vfft_proto_execute_fwd(pB, re, im, K);
            psw_t pw = {re, im, tr, ti, pB_, K, N};
            k_pswap(&pw);
            double e = 0;
            for (int k = 0; k < N; k++)
            {
                double d1 = fabs(re[(size_t)k * K] - Xr[k]), d2 = fabs(im[(size_t)k * K] - Xi[k]);
                if (d1 > e)
                    e = d1;
                if (d2 > e)
                    e = d2;
            }
            okB = (e / (sc > 0 ? sc : 1)) < 1e-8;
            psw_t pw2 = {dr, di, tr, ti, pB_, K, N};
            t_ps = time_kernel(k_pswap, &pw2, bytes);
            int *plist = mk_pair_list(N, pB_);
            pub_t pu = {dr, di, plist, K};
            double t_psu = time_kernel(k_pswap_ub, &pu, bytes);
            free(plist);
            if (t_psu < t_ps)
                t_ps = t_psu; /* keep UB if better */
            vfft_proto_plan_destroy(pB);
        }
        else
        {
            okB = invol;
        }
        free(pB_);
        if (!okB)
            printf("  [B corr/involution FAIL]\n");
    }

    /* LEAF-IP */
    double t_leaf = -1;
    int okL = 1;
    leaf_fn lf = NRLEAF(N);
    if (lf)
    {
        srand(66 + N + (int)K);
        for (size_t i = 0; i < n; i++)
        {
            re[i] = (double)rand() / RAND_MAX - 0.5;
            im[i] = (double)rand() / RAND_MAX - 0.5;
        }
        lf(re, im, re, im, NULL, NULL, K, 1, K, 1, K);
        double e = 0;
        for (int k = 0; k < N; k++)
        {
            double d1 = fabs(re[(size_t)k * K] - Xr[k]), d2 = fabs(im[(size_t)k * K] - Xi[k]);
            if (d1 > e)
                e = d1;
            if (d2 > e)
                e = d2;
        }
        okL = (e / (sc > 0 ? sc : 1)) < 1e-8;
        t_leaf = time_leafip(lf, re, im, n, K);
    }

    /* compose + report */
    printf("  baseline A (scrambled)      %10.0f ns   orderA=%s %s\n", tA,
           bi == 0 ? "fwd-perm" : bi == 1 ? "fwd-iperm"
                              : bi == 2   ? "rev-perm"
                                          : "rev-iperm",
           okA ? "ok" : "<CORR FAIL>");
    typedef struct
    {
        const char *name;
        double t;
    } row_t;
    row_t rows[12];
    int nr = 0;
    if (C->nfA == 1)
        rows[nr++] = (row_t){"FREE (already natural)", tA};
    if (t_leaf > 0 && okL)
        rows[nr++] = (row_t){"LEAF-IP (aliased leaf)", t_leaf};
    rows[nr++] = (row_t){"PURE-cycle", tA + t_cy};
    rows[nr++] = (row_t){"PURE-cycle-UB", tA + t_cyu};
    if (t_cy2 > 0)
        rows[nr++] = (row_t){"PURE-cycle-UB2 (8-way)", tA + t_cy2};
    rows[nr++] = (row_t){"PURE-gather", tA + t_ga};
    if (t_ct > 0)
        rows[nr++] = (row_t){"CELL-TRANSPOSE (nf=2)", tA + t_ct};
    rows[nr++] = (row_t){"SCR-t1  [EST]", tA + t_s1 - t_ip};
    if (t_s1t > 0)
        rows[nr++] = (row_t){"SCR-t1-TILED [EST]", tA + t_s1t - t_ip};
    rows[nr++] = (row_t){"SCR-t1s(needs P3) [EST]", tA + t_s1s - t_ip};
    if (t_ps > 0 && okB)
        rows[nr++] = (row_t){"PSWAP (inj chain+swap)", tB + t_ps};
    int win = 0;
    for (int i = 1; i < nr; i++)
        if (rows[i].t < rows[win].t)
            win = i;
    for (int i = 0; i < nr; i++)
        printf("  %-26s %10.0f ns   %+6.1f%%%s\n", rows[i].name, rows[i].t,
               100.0 * (rows[i].t - tA) / tA, i == win ? "   << WINNER" : "");
    if (tB > 0)
        printf("  (chain B fft alone: %.0f ns = %.2fx of A, best profile %s; pswap pass %.0f ns)\n",
               tB, tB / tA, bprof == 0 ? "T1S" : bprof == 1 ? "FLAT"
                                                            : "LOG3",
               t_ps);
    printf("  (kernels: ippass %.0f  scat-t1 %.0f  tiled %.0f  scat-t1s %.0f  cyc %.0f  gat %.0f)\n",
           t_ip, t_s1, t_s1t, t_s1s, t_cy, t_ga);

    vfft_proto_plan_destroy(pA);
    _aligned_free(re);
    _aligned_free(im);
    _aligned_free(sr);
    _aligned_free(si);
    _aligned_free(dr);
    _aligned_free(di);
    _aligned_free(twr);
    _aligned_free(twi);
    _aligned_free(tsr);
    _aligned_free(tsi);
    _aligned_free(tr);
    _aligned_free(ti);
    _aligned_free(gr);
    _aligned_free(gi);
    free(Xr);
    free(Xi);
    free(pf);
    free(ipf);
    free(pr);
    free(ipr);
    free(IMA);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_init(&REG);
    printf("# T6 BAKEOFF: all natural-order candidates, same harness. baseline = wisdom chain+variants\n");
    printf("# via low-level executor (no Tier1/JIT; uniform across candidates). [EST] = terminator\n");
    printf("# composition tA + kernel - ippass. lastv=F cells double-count the FLAT table (conservative).\n");
    for (size_t i = 0; i < NCELLS; i++)
    {
        Sleep(400);
        run_cell(&CELLS[i]);
    }
    return 0;
}
