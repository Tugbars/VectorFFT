/* ═══════════════════════════════════════════════════════════════
 * rfft.h — native real mixed-radix FFT (r2hc), forward, packed output
 *
 * Sections 60-63 of the lab notebook; design: docs/native_rfft_design.md.
 * This is the transcription of the PROVEN reference loop in
 * benchmarks/gate_rfft_compose_L.c (12 cells, zero fix iterations).
 *
 * Geometry (DIT, factors f[0..nf-1], f[0] = outermost combine,
 * f[nf-1] = leaf radix):
 *   - The plane is N rows x K lanes, FFTW halfcomplex packing at every
 *     recursion level: subproblem at offset q (stride Q) of size np
 *     holds re X[p] at position p <= np/2 and im X[np-p] above.
 *   - LEAF: one batched r2cf call, vl = S*K (S = N/leaf), im stream
 *     reversed via os_im < 0 with base one-past the plane.
 *   - STAGE d (radix r, child m, Q = prod of radices above): per
 *     column k, batched over subproblems via the Q-fold (vl = Q*K):
 *       k = 0        -> r2cf call (children's DC bins are real)
 *       0 < k < m/2  -> hc2hc call (tw[j] = W_np^{jk}, slot 0 unused)
 *       k = m/2      -> small direct loop (self-mirror column)
 *   - Ping-pong planes; the d = 0 stage writes the caller's output.
 *
 * Output format v1: PACKED halfcomplex plane (N x K). The hc2c
 * natural-split terminator (design D2) replaces the d = 0 stage in a
 * later phase; everything else is unchanged by that swap.
 *
 * Constraints v1: K % 8 == 0 (vl must be a vector-width multiple);
 * single-threaded; radices limited to the rfft codelet quadrant.
 * ═══════════════════════════════════════════════════════════════ */
#ifndef VFFT_RFFT_H
#define VFFT_RFFT_H

#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>

/* Portable 64-byte aligned alloc/free. mingw/MSVC lack C11 aligned_alloc and
 * REQUIRE _aligned_free for _aligned_malloc memory (plain free corrupts the heap).
 * (Windows portability shim over the bundle's Linux-only aligned_alloc.) */
#if defined(_WIN32) || defined(_MSC_VER)
#  include <malloc.h>
#  define RFFT_ALIGNED_ALLOC(a, sz) _aligned_malloc((sz), (a))
#  define RFFT_ALIGNED_FREE(p)      _aligned_free(p)
#else
#  define RFFT_ALIGNED_ALLOC(a, sz) aligned_alloc((a), (sz))
#  define RFFT_ALIGNED_FREE(p)      free(p)
#endif

/* Optional 2MB LARGE-PAGE allocation for the plane buffers (env VFFT_RFFT_HUGE=1).
 * VTune showed high-K rfft is store-DTLB-bound (41% DTLB-store at N=256 K=256): the
 * ~0.5-1.5MB plane spans 128-384 4KB pages and the output stride scatters stores
 * across them, thrashing the ~96-entry DTLB. A 2MB page collapses the plane to ~1
 * page, removing the TLB tax. Needs the "Lock pages in memory" privilege
 * (SeLockMemoryPrivilege, enabled by the harness) + elevation; ALWAYS falls back to
 * RFFT_ALIGNED_ALLOC if large pages are unavailable, so it is safe by default.
 * Manual Win32 decls avoid pulling <windows.h> into this widely-included header. */
#if defined(_WIN32)
#  if !defined(_WINDOWS_)   /* <windows.h> not included by this TU: declare what we use */
extern void  *__stdcall VirtualAlloc(void *, size_t, unsigned long, unsigned long);
extern int    __stdcall VirtualFree(void *, size_t, unsigned long);
extern size_t __stdcall GetLargePageMinimum(void);
#  endif
#  define RFFT_MEM_COMMIT       0x00001000UL
#  define RFFT_MEM_RESERVE      0x00002000UL
#  define RFFT_MEM_LARGE_PAGES  0x20000000UL
#  define RFFT_MEM_RELEASE      0x00008000UL
#  define RFFT_PAGE_RW          0x04UL
static int _rfft_huge_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("VFFT_RFFT_HUGE"); v = (e && atoi(e) > 0) ? 1 : 0; }
    return v;
}
#endif

/* Allocate `bytes`; sets *huge=1 iff backed by large pages (else 0 + normal alloc). */
static inline void *rfft_buf_alloc(size_t bytes, int *huge) {
    *huge = 0;
#if defined(_WIN32)
    if (_rfft_huge_on()) {
        size_t lp = GetLargePageMinimum();
        if (lp) {
            size_t r = ((bytes + lp - 1) / lp) * lp;
            void *p = VirtualAlloc((void*)0, r,
                                   RFFT_MEM_RESERVE | RFFT_MEM_COMMIT | RFFT_MEM_LARGE_PAGES,
                                   RFFT_PAGE_RW);
            if (p) { *huge = 1; return p; }
        }
    }
#endif
    return RFFT_ALIGNED_ALLOC(64, bytes);
}
static inline void rfft_buf_free(void *p, int huge) {
    if (!p) return;
#if defined(_WIN32)
    if (huge) { VirtualFree(p, 0, RFFT_MEM_RELEASE); return; }
#endif
    RFFT_ALIGNED_FREE(p);
}
#endif
/* E1 (section 67): software prefetch of the NEXT column's rows.
 * MEASURED NEGATIVE on the dev container (1-vCPU KVM Cascade Lake):
 * 3-5% slower across plans — the prefetches contend for the same
 * fill-buffer/miss resources the demand stream needs, and the L2
 * streamer was already covering most of the warm-up. Default OFF;
 * -DVFFT_RFFT_PREFETCH=1 enables (worth re-measuring on real
 * hardware, where the verdict may invert). */
#if defined(VFFT_RFFT_PREFETCH) && (defined(__AVX512F__) || defined(__AVX2__))
#define VFFT_RFFT_PF(addr) do {     _mm_prefetch((const char *)(addr), _MM_HINT_T0);     _mm_prefetch((const char *)(addr) + 512, _MM_HINT_T0); } while (0)
#else
#define VFFT_RFFT_PF(addr) do { (void)(addr); } while (0)
#endif

#ifdef VFFT_RFFT_PROFILE
#include <time.h>
static inline double _rfft_now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
#endif

#ifndef VFFT_RFFT_MAX_STAGES
#define VFFT_RFFT_MAX_STAGES 14
#endif
#ifndef VFFT_RFFT_MAX_RADIX
#define VFFT_RFFT_MAX_RADIX 16
#endif

typedef void (*rfft_r2cf_fn)(const double *in_re,
                             double *out_re, double *out_im,
                             ptrdiff_t is, ptrdiff_t os_re, ptrdiff_t os_im,
                             size_t vl);
typedef void (*rfft_hc_fn)(const double *in_re, const double *in_im,
                           double *out_re, double *out_im,
                           const double *tw_re, const double *tw_im,
                           ptrdiff_t is, ptrdiff_t os, size_t vl);

/* r2cb backward leaf (section 62): halfcomplex in (in_re,in_im) -> real out.
 * Distinct ABI from r2cf (which is real in -> split complex out). */
/* Split input strides (c2r cascade): the backward leaf reads the layout
 * r2cf WRITES — re ascending from its base, im DESCENDING from a
 * one-past (+NK) base. A single shared `is` cannot express the sign
 * split, which is exactly what broke the first two cascade attempts. */
typedef void (*rfft_r2cb_fn)(const double *in_re, const double *in_im,
                             double *out_re,
                             ptrdiff_t is_re, ptrdiff_t is_im,
                             ptrdiff_t os_re, size_t vl);

/* D2 natural terminator codelet (section 69), FFTW khc2c-shaped:
 * low slots s <= s* go to Rp/Ip + s*osp; upper slots (conjugated by
 * the codelet) go to Rm/Im + (r-1-s)*osm. */
typedef void (*rfft_hc2c_nat_fn)(
    const double *in_re, const double *in_im,
    double *Rp, double *Ip, double *Rm, double *Im,
    const double *tw_re, const double *tw_im,
    ptrdiff_t is, ptrdiff_t osp, ptrdiff_t osm, size_t vl);

/* T1 ranged variants (section 70): one call walks kcount columns,
 * pointers and twiddles advancing inside the codelet. */
typedef void (*rfft_hc_rng_fn)(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    ptrdiff_t is, ptrdiff_t os,
    ptrdiff_t cs_in, ptrdiff_t cs_out, int kcount, size_t vl);
typedef void (*rfft_hc2c_nat_rng_fn)(
    const double *in_re, const double *in_im,
    double *Rp, double *Ip, double *Rm, double *Im,
    const double *tw_re, const double *tw_im,
    ptrdiff_t is, ptrdiff_t osp, ptrdiff_t osm,
    ptrdiff_t cs_in, ptrdiff_t cs_out, int kcount, size_t vl);

/* radix-indexed codelet registry; 0 entries = radix unavailable */
typedef struct {
    rfft_r2cf_fn r2cf[VFFT_RFFT_MAX_RADIX + 1];
    rfft_hc_fn hc2hc[VFFT_RFFT_MAX_RADIX + 1];
    rfft_hc2c_nat_fn hc2c[VFFT_RFFT_MAX_RADIX + 1]; /* D2; optional */
    rfft_hc_rng_fn hc2hc_rng[VFFT_RFFT_MAX_RADIX + 1];      /* T1 */
    rfft_hc2c_nat_rng_fn hc2c_rng[VFFT_RFFT_MAX_RADIX + 1]; /* T1 */
    /* log3 twiddle-policy variants (section 62: hc2cf2 = hc2c + log3).
     * Same ABI as their flat counterparts; the planner prefers these when
     * present (fewer twiddle loads: 7->3 slots at r8). 0 = fall back to flat. */
    rfft_hc_fn hc2hc_log3[VFFT_RFFT_MAX_RADIX + 1];
    rfft_hc2c_nat_fn hc2c_log3[VFFT_RFFT_MAX_RADIX + 1];
    /* DIF orientation slots (section 63): distinct identity from DIT, so the
     * auto-emitted registrar does not clobber DIT with DIF. Same ABI as the
     * DIT twiddle stage. No executor calls these yet (populated-but-unused
     * until a DIF-using executor/planner path lands). */
    rfft_hc_fn hc2hc_dif[VFFT_RFFT_MAX_RADIX + 1];
    rfft_hc_fn hc2hc_dif_log3[VFFT_RFFT_MAX_RADIX + 1];
    /* c2r (backward real) slots (section 62): the r2cb leaf
     * (halfcomplex -> real) and the DIF BACKWARD twiddle stages. FFTW runs
     * hc2r as apply_dif with sign-flipped twiddles; these slots hold the
     * --hc2hc --dif --bwd --t1s codelets. The r2cb leaf has the same ABI as
     * r2cf at the call site (in_re/in_im -> out_re via the executor's strides;
     * the codelet's own signature is in_re,in_im,out_re,is,os_re,vl). */
    rfft_r2cb_fn r2cb[VFFT_RFFT_MAX_RADIX + 1];
    rfft_hc_fn hc2hc_dif_bwd[VFFT_RFFT_MAX_RADIX + 1];
    rfft_hc_fn hc2hc_dif_bwd_log3[VFFT_RFFT_MAX_RADIX + 1];
    /* ranged DIF backward (section 70 mirror): one call walks kcount interior
     * columns, advancing re-streams up / im-streams down by cs per column.
     * Collapses the c2r interior per-k loop to a single call. */
    rfft_hc_rng_fn hc2hc_dif_rng_bwd[VFFT_RFFT_MAX_RADIX + 1];
} rfft_codelets_t;

typedef struct {
    int radix, m, np;
    size_t Q, vl;
    rfft_r2cf_fn k0;
    rfft_hc_fn hc;
    rfft_hc_rng_fn hcr; /* T1 ranged, or NULL */
    /* precomputed twiddles for interior columns, packed from k=1:
     * tw_re[((k-1)*r + j)*vl + v], j=0 slot present but never loaded */
    double *tw_re, *tw_im;
    int kmax;
    /* mid (k=m/2) column coefficients when m even:
     * mid_c/mid_s[s*r + j] = cos/sin(-2*pi*j*(m/2 + s*m)/np) */
    double *mid_c, *mid_s;
    int has_mid;
} rfft_stage_t;

typedef struct {
    int N;
    size_t K;
    int nf;
    int factors[VFFT_RFFT_MAX_STAGES];
    rfft_r2cf_fn leaf;
    int leaf_r;
    size_t S;
    rfft_stage_t st[VFFT_RFFT_MAX_STAGES];
    rfft_hc2c_nat_fn hcn; /* stage-0 natural terminator (D2), or NULL */
    rfft_hc2c_nat_rng_fn hcnr; /* T1 ranged terminator, or NULL */
    double *nat_k0;       /* k0 scratch column, MAX_RADIX*K doubles */
#ifdef VFFT_RFFT_PROFILE
    /* per-phase accumulators (ns), reset by caller: [0]=leaf;
     * per stage d: k0 / interior columns / mid */
    double prof_leaf;
    double prof_k0[VFFT_RFFT_MAX_STAGES];
    double prof_cols[VFFT_RFFT_MAX_STAGES];
    double prof_mid[VFFT_RFFT_MAX_STAGES];
#endif
    size_t Kb; /* lane-block width (section 65): the cascade runs per
                * Kb-lane slab so both planes stay cache-resident
                * across ALL stages. Multiple of 8. */
    double *planeA, *planeB; /* scratch, N*K each, 64B aligned */
    int planeA_huge, planeB_huge; /* 1 iff large-page backed (VFFT_RFFT_HUGE) */
} rfft_plan_t;

static inline void rfft_plan_destroy(rfft_plan_t *p)
{
    if (!p) return;
    for (int d = 0; d < p->nf - 1; d++) {
        free(p->st[d].tw_re); free(p->st[d].tw_im);
        free(p->st[d].mid_c); free(p->st[d].mid_s);
    }
    rfft_buf_free(p->planeA, p->planeA_huge); rfft_buf_free(p->planeB, p->planeB_huge);
    RFFT_ALIGNED_FREE(p->nat_k0);
    free(p);
}

/* factors: f[0] = outermost combine, f[nf-1] = leaf.
 * Explicit per-stage variant: variant[d] for combine stage d (0..nf-2) is
 * 0=FLAT, 1=LOG3, 2=T1S(ranged); the leaf (factors[nf-1]) has no variant.
 * variant=NULL means the default policy (LOG3-preferred + ranged-wired), so the
 * plain rfft_plan_create wrapper below is byte-identical to the original. */
static inline rfft_plan_t *rfft_plan_create_ex(int N, size_t K,
                                               const int *factors, int nf,
                                               const int *variant,
                                               const rfft_codelets_t *reg)
{
    if (nf < 1 || nf > VFFT_RFFT_MAX_STAGES) return NULL;
    if (K == 0 || (K % 8) != 0) return NULL; /* vl must be 8-multiple */
    {
        long acc = 1;
        for (int i = 0; i < nf; i++) acc *= factors[i];
        if (acc != N) return NULL;
    }
    rfft_plan_t *p = (rfft_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N; p->K = K; p->nf = nf;
    memcpy(p->factors, factors, (size_t)nf * sizeof(int));

    p->leaf_r = factors[nf - 1];
    p->S = (size_t)(N / p->leaf_r);
    if (p->leaf_r > VFFT_RFFT_MAX_RADIX || !reg->r2cf[p->leaf_r])
        goto fail;
    p->leaf = reg->r2cf[p->leaf_r];

    for (int d = nf - 2; d >= 0; d--) {
        rfft_stage_t *st = &p->st[d];
        int r = factors[d];
        size_t Q = 1;
        for (int i = 0; i < d; i++) Q *= (size_t)factors[i];
        st->radix = r;
        st->Q = Q;
        st->np = (int)((size_t)N / Q);
        st->m = st->np / r;
        st->vl = Q * K;
        if (r > VFFT_RFFT_MAX_RADIX || !reg->r2cf[r] || !reg->hc2hc[r])
            goto fail;
        st->k0 = reg->r2cf[r];
        /* Per-stage variant. The PACKED executor runs st->hc/st->hcr for ALL
         * combine stages (incl d=0), so this is the only variant knob that
         * matters for the calibrated path. CRITICAL: hcr MUST be NULL for
         * FLAT/LOG3 or the executor's `if(st->hcr)` ranged branch fires anyway
         * and mis-attributes the measurement. */
        {
            int v = variant ? variant[d] : -1;   /* -1 = default policy */
            if (v == 0) {                          /* FLAT */
                st->hc = reg->hc2hc[r];
                st->hcr = NULL;
            } else if (v == 1) {                   /* LOG3 (flat fallback) */
                st->hc = reg->hc2hc_log3[r] ? reg->hc2hc_log3[r] : reg->hc2hc[r];
                st->hcr = NULL;
            } else if (v == 2) {                   /* T1S ranged (flat base) */
                st->hc = reg->hc2hc[r];
#ifdef VFFT_RFFT_RANGED
                st->hcr = reg->hc2hc_rng[r];
#else
                st->hcr = NULL;                    /* ranged off in this build -> FLAT */
#endif
            } else {                               /* default: log3-preferred + ranged */
                st->hc = reg->hc2hc_log3[r] ? reg->hc2hc_log3[r] : reg->hc2hc[r];
                st->hcr = reg->hc2hc_rng[r];
            }
        }
        st->kmax = (st->m % 2 == 0) ? st->m / 2 - 1 : (st->m - 1) / 2;
        st->has_mid = (st->m % 2 == 0);

        if (st->kmax >= 1) {
            /* broadcast-twiddle codelets (--t1s): r SCALARS per column */
            size_t sz = (size_t)st->kmax * (size_t)r;
            st->tw_re = (double *)malloc(sz * 8);
            st->tw_im = (double *)malloc(sz * 8);
            if (!st->tw_re || !st->tw_im) goto fail;
            /* unified slot convention (section 66): leg j loads
             * Twiddle slot j-1; slot r-1 in each column block is dead. */
            for (int k = 1; k <= st->kmax; k++) {
                for (int j = 1; j < r; j++) {
                    double th = 2.0 * M_PI * (double)j * (double)k /
                                (double)st->np;
                    st->tw_re[(size_t)(k - 1) * r + (j - 1)] = cos(th);
                    st->tw_im[(size_t)(k - 1) * r + (j - 1)] = -sin(th);
                }
                st->tw_re[(size_t)(k - 1) * r + (r - 1)] = 0.0;
                st->tw_im[(size_t)(k - 1) * r + (r - 1)] = 0.0;
            }
        }
        if (st->has_mid) {
            st->mid_c = (double *)malloc((size_t)r * (size_t)r * 8);
            st->mid_s = (double *)malloc((size_t)r * (size_t)r * 8);
            if (!st->mid_c || !st->mid_s) goto fail;
            for (int s = 0; s < r; s++)
                for (int j = 0; j < r; j++) {
                    double th = -2.0 * M_PI * (double)j *
                        ((double)st->m / 2.0 + (double)s * st->m) /
                        (double)st->np;
                    st->mid_c[s * r + j] = cos(th);
                    st->mid_s[s * r + j] = sin(th);
                }
        }
    }

    /* 2*N*K, not N*K: the packed leaf writes the IM stream at planeA + NK (descending),
     * so the top im address reaches NK + (S-1)*K < 2*NK. N*K under-allocates and the
     * packed forward overflows at high K (N>=512) — latent because the packed/split rfft
     * forward is only driven at K<=16 in the r2c dispatcher; the packed c2r input gen
     * exercises it at K>=32 (planeB likewise for the nf>=3 ping-pong). */
    p->planeA = (double *)rfft_buf_alloc((size_t)2 * N * K * 8, &p->planeA_huge);
    p->planeB_huge = 0;
    p->planeB = (nf >= 3)
        ? (double *)rfft_buf_alloc((size_t)2 * N * K * 8, &p->planeB_huge) : NULL;
    if (!p->planeA || (nf >= 3 && !p->planeB)) goto fail;
    p->hcn = (nf >= 2)
        ? (reg->hc2c_log3[p->st[0].radix] ? reg->hc2c_log3[p->st[0].radix]
                                          : reg->hc2c[p->st[0].radix])
        : NULL;
    p->hcnr = (nf >= 2) ? reg->hc2c_rng[p->st[0].radix] : NULL;
    p->nat_k0 = (double *)RFFT_ALIGNED_ALLOC(64,
        (size_t)VFFT_RFFT_MAX_RADIX * K * 8);
    if (!p->nat_k0) goto fail;
    /* Lane-blocking default: OFF (Kb = K). Section 65 measured the
     * L2-slab heuristic as NEGATIVE (-22% at (4,4,16)): Kb=96 cuts
     * each stream burst to ~768B, defeating the prefetchers, and the
     * cascade was never capacity-bound to begin with. The mechanism
     * stays (callers may tune p->Kb, multiple of 8) because slab
     * residency may pay on other hardware; it does not pay here. */
    p->Kb = K;
    return p;
fail:
    rfft_plan_destroy(p);
    return NULL;
}

/* Default-policy plan create (variant=NULL): byte-identical to the original. */
static inline rfft_plan_t *rfft_plan_create(int N, size_t K,
                                            const int *factors, int nf,
                                            const rfft_codelets_t *reg)
{
    return rfft_plan_create_ex(N, K, factors, nf, /*variant=*/NULL, reg);
}

/* Shared mid-column (k = m/2) kernel, s-blocked (SB = 4) per
 * section 66. mode 0: packed store (row Q*pp of dst). mode 1:
 * natural store (rows pp <= nh of dst_re/dst_im; im 0 at nh;
 * uppers skipped — self-paired column). slot stride = Q*K. */
static inline void rfft_mid_column(int r, int m, int np, size_t Q,
                                   size_t K, size_t vl,
                                   const double *mid_in,
                                   const double *mc, const double *ms,
                                   int natural, size_t nh,
                                   double *dst, double *dst_im)
{
    size_t v = 0;
#if defined(__AVX512F__)
    for (; v + 8 <= vl; v += 8) {
        for (int s0 = 0; s0 < r; s0 += 4) {
            int sn = (s0 + 4 <= r) ? 4 : (r - s0);
            __m512d acc_r[4], acc_i[4];
            for (int t = 0; t < sn; t++) {
                acc_r[t] = _mm512_setzero_pd();
                acc_i[t] = _mm512_setzero_pd();
            }
            for (int j = 0; j < r; j++) {
                __m512d c = _mm512_loadu_pd(mid_in + (size_t)j * (Q * K) + v);
                for (int t = 0; t < sn; t++) {
                    acc_r[t] = _mm512_fmadd_pd(c,
                        _mm512_set1_pd(mc[(s0 + t) * r + j]), acc_r[t]);
                    acc_i[t] = _mm512_fmadd_pd(c,
                        _mm512_set1_pd(ms[(s0 + t) * r + j]), acc_i[t]);
                }
            }
            for (int t = 0; t < sn; t++) {
                int sI = s0 + t;
                int pp = m / 2 + sI * m;
                if (!natural) {
                    if (pp <= np / 2)
                        _mm512_storeu_pd(dst + (Q * (size_t)pp) * K + v,
                                         acc_r[t]);
                    int sm = r - 1 - sI;
                    int pm = m / 2 + sm * m;
                    if (pm > np / 2)
                        _mm512_storeu_pd(dst + (Q * (size_t)pm) * K + v,
                                         acc_i[t]);
                } else if ((size_t)pp <= nh) {
                    _mm512_storeu_pd(dst + (size_t)pp * K + v, acc_r[t]);
                    _mm512_storeu_pd(dst_im + (size_t)pp * K + v,
                        ((size_t)pp == nh) ? _mm512_setzero_pd()
                                           : acc_i[t]);
                }
            }
        }
    }
#elif defined(__AVX2__)
    for (; v + 4 <= vl; v += 4) {
        for (int s0 = 0; s0 < r; s0 += 4) {
            int sn = (s0 + 4 <= r) ? 4 : (r - s0);
            __m256d acc_r[4], acc_i[4];
            for (int t = 0; t < sn; t++) {
                acc_r[t] = _mm256_setzero_pd();
                acc_i[t] = _mm256_setzero_pd();
            }
            for (int j = 0; j < r; j++) {
                __m256d c = _mm256_loadu_pd(mid_in + (size_t)j * (Q * K) + v);
                for (int t = 0; t < sn; t++) {
                    acc_r[t] = _mm256_fmadd_pd(c,
                        _mm256_set1_pd(mc[(s0 + t) * r + j]), acc_r[t]);
                    acc_i[t] = _mm256_fmadd_pd(c,
                        _mm256_set1_pd(ms[(s0 + t) * r + j]), acc_i[t]);
                }
            }
            for (int t = 0; t < sn; t++) {
                int sI = s0 + t;
                int pp = m / 2 + sI * m;
                if (!natural) {
                    if (pp <= np / 2)
                        _mm256_storeu_pd(dst + (Q * (size_t)pp) * K + v,
                                         acc_r[t]);
                    int sm = r - 1 - sI;
                    int pm = m / 2 + sm * m;
                    if (pm > np / 2)
                        _mm256_storeu_pd(dst + (Q * (size_t)pm) * K + v,
                                         acc_i[t]);
                } else if ((size_t)pp <= nh) {
                    _mm256_storeu_pd(dst + (size_t)pp * K + v, acc_r[t]);
                    _mm256_storeu_pd(dst_im + (size_t)pp * K + v,
                        ((size_t)pp == nh) ? _mm256_setzero_pd()
                                           : acc_i[t]);
                }
            }
        }
    }
#endif
    for (; v < vl; v++) {
        for (int sI = 0; sI < r; sI++) {
            int pp = m / 2 + sI * m;
            double sr = 0, si = 0;
            for (int j = 0; j < r; j++) {
                double c = mid_in[(size_t)j * (Q * K) + v];
                sr += c * mc[sI * r + j];
                si += c * ms[sI * r + j];
            }
            if (!natural) {
                if (pp <= np / 2)
                    dst[(Q * (size_t)pp) * K + v] = sr;
                /* upper rows of the packed map come from the mirror
                 * slot's imag: handled when sI reaches the mirror */
                int sm = r - 1 - sI;
                int pm = m / 2 + sm * m;
                if (pm > np / 2)
                    dst[(Q * (size_t)pm) * K + v] = si;
            } else if ((size_t)pp <= nh) {
                dst[(size_t)pp * K + v] = sr;
                dst_im[(size_t)pp * K + v] = ((size_t)pp == nh) ? 0.0 : si;
            }
        }
    }
}

/* forward, packed halfcomplex output into out (N*K doubles).
 * x is read-only; x != out required.
 * Section 65: lane-blocked schedule. The outer loop walks Kb-lane
 * slabs; the entire cascade runs per slab, so the two ping-pong slabs
 * (2*N*Kb*8B) stay cache-resident across all stages. Blocking breaks
 * the (q,lane) fold, so stage calls loop q explicitly (vl = bw). */
static inline void rfft_execute_fwd_packed(const rfft_plan_t *p,
                                           const double *x, double *out)
{
    const int N = p->N;
    const size_t K = p->K;
    const size_t NK = (size_t)N * K;

    for (size_t b = 0; b < K; b += p->Kb) {
        const size_t bw = (b + p->Kb <= K) ? p->Kb : (K - b);

        /* LEAF: per-group calls on this slab */
        {
#ifdef VFFT_RFFT_PROFILE
            double _t0 = _rfft_now();
#endif
            double *dst = (p->nf == 1) ? out : p->planeA;
            const ptrdiff_t SK = (ptrdiff_t)(p->S * K);
            for (size_t g = 0; g < p->S; g++)
                p->leaf(x + g * K + b,
                        dst + g * K + b,
                        dst + g * K + b + NK,
                        SK, SK, -SK, bw);
#ifdef VFFT_RFFT_PROFILE
            ((rfft_plan_t *)p)->prof_leaf += _rfft_now() - _t0;
#endif
        }
        if (p->nf == 1) continue;

        double *cur = p->planeA, *nxt;
        for (int d = p->nf - 2; d >= 0; d--) {
            const rfft_stage_t *st = &p->st[d];
            const int r = st->radix, m = st->m, np = st->np;
            const size_t Q = st->Q;
            const ptrdiff_t QK = (ptrdiff_t)(Q * K);
            const ptrdiff_t QmK = (ptrdiff_t)(Q * (size_t)m * K);
            nxt = (d == 0) ? out : ((cur == p->planeA) ? p->planeB
                                                       : p->planeA);
            /* E2 (section 67): when the lane dim is unblocked
             * (single full-width slab), the (q,lane) Q-FOLD is legal
             * and turns Q separate 2KB rows into one contiguous
             * Q*2KB stream per slot — section 62's geometry,
             * restored. Per-q is the general path for Kb < K. */
            const int folded = (bw == K && b == 0 && Q > 1);
            const size_t Qfold = folded ? 1 : Q;
            const size_t vlf = folded ? (Q * K) : bw;
            for (size_t q = 0; q < Qfold; q++) {
                double *curq = cur + q * K + b;
                double *nxtq = nxt + q * K + b;

                /* k = 0 (prefetch column 1's rows first) */
#ifdef VFFT_RFFT_PROFILE
                double _t0 = _rfft_now();
#endif
                if (st->kmax >= 1)
                    for (int j = 0; j < r; j++) {
                        VFFT_RFFT_PF(curq + (Q * (size_t)(r + j)) * K);
                        VFFT_RFFT_PF(curq +
                            (Q * (size_t)(r * (m - 1) + j)) * K);
                    }
                st->k0(curq, nxtq, nxtq + NK, QK, QmK, -QmK, vlf);
#ifdef VFFT_RFFT_PROFILE
                double _t1 = _rfft_now();
                ((rfft_plan_t *)p)->prof_k0[d] += _t1 - _t0;
#endif

                /* interior columns, prefetching column k+1 */
#ifdef VFFT_RFFT_RANGED
                if (st->hcr && st->kmax >= 1) {
                    st->hcr(curq + (Q * (size_t)r) * K,
                            curq + (Q * (size_t)(r * (m - 1))) * K,
                            nxtq + Q * K,
                            nxtq + (Q * (size_t)(m - 1)) * K,
                            st->tw_re, st->tw_im,
                            QK, QmK,
                            (ptrdiff_t)(Q * (size_t)r * K),
                            (ptrdiff_t)(Q * K),
                            st->kmax, vlf);
                } else
#endif
                for (int k = 1; k <= st->kmax; k++) {
                    if (k < st->kmax)
                        for (int j = 0; j < r; j++) {
                            VFFT_RFFT_PF(curq +
                                (Q * (size_t)(r * (k + 1) + j)) * K);
                            VFFT_RFFT_PF(curq +
                                (Q * (size_t)(r * (m - k - 1) + j)) * K);
                            VFFT_RFFT_PF(nxtq +
                                (Q * (size_t)(k + 1 + j * m)) * K);
                            VFFT_RFFT_PF(nxtq +
                                (Q * (size_t)(m - k - 1 + j * m)) * K);
                        }
                    st->hc(curq + (Q * (size_t)(r * k)) * K,
                           curq + (Q * (size_t)(r * (m - k))) * K,
                           nxtq + (Q * (size_t)k) * K,
                           nxtq + (Q * (size_t)(m - k)) * K,
                           st->tw_re + (size_t)(k - 1) * r,
                           st->tw_im + (size_t)(k - 1) * r,
                           QK, QmK, vlf);
                }
#ifdef VFFT_RFFT_PROFILE
                double _t2 = _rfft_now();
                ((rfft_plan_t *)p)->prof_cols[d] += _t2 - _t1;
#endif

                /* k = m/2 (m even): shared s-blocked mid kernel */
#ifdef VFFT_RFFT_PROFILE
                double _t3 = _rfft_now();
#endif
                if (st->has_mid)
                    rfft_mid_column(r, m, np, Q, K, vlf,
                        curq + (Q * (size_t)(r * (m / 2))) * K,
                        st->mid_c, st->mid_s, 0, 0, nxtq, NULL);
#ifdef VFFT_RFFT_PROFILE
                ((rfft_plan_t *)p)->prof_mid[d] += _rfft_now() - _t3;
#endif
            }
            cur = nxt;
        }
    }
}

/* ===== D2 (section 69): forward, NATURAL split-complex output =====
 * out_re/out_im are (N/2+1) x K planes, row f = frequency f.
 * Stages nf-2..1 run the packed cascade unchanged (full-width,
 * folded); stage 0 is the natural terminator: k = 0 via r2cf into a
 * scratch column + row scatter; interior columns via hc2c_nat (one
 * call covers residues k and m-k; constant-boundary lemma in
 * docs/native_rfft_design.md); the m/2 mid stores (Re, Im) at its
 * low rows directly. v1 runs full-width (plan Kb ignored).
 * Requires reg->hc2c[stage-0 radix] (p->hcn != NULL) for nf >= 2. */
static inline void rfft_execute_fwd_natural(const rfft_plan_t *p,
                                            const double *x,
                                            double *out_re,
                                            double *out_im)
{
    const int N = p->N;
    const size_t K = p->K;
    const size_t NK = (size_t)N * K;
    const size_t nh = (size_t)(N / 2);

    if (p->nf == 1) {
        p->leaf(x, p->planeA, p->planeA + NK,
                (ptrdiff_t)K, (ptrdiff_t)K, -(ptrdiff_t)K, K);
        memcpy(out_re, p->planeA, K * 8);
        memset(out_im, 0, K * 8);
        for (size_t f = 1; f < (size_t)((N + 1) / 2); f++) {
            memcpy(out_re + f * K, p->planeA + f * K, K * 8);
            memcpy(out_im + f * K, p->planeA + ((size_t)N - f) * K, K * 8);
        }
        if (N % 2 == 0) {
            memcpy(out_re + nh * K, p->planeA + nh * K, K * 8);
            memset(out_im + nh * K, 0, K * 8);
        }
        return;
    }

    {
        /* LEAF FOLD (restored per header design): in the unblocked case the
         * per-g calls cover exactly contiguous vl windows [g*K,(g+1)*K), so
         * ONE call at vl = S*K is address-identical — same fold the stage
         * loop's E2 logic applies. Was 16 calls at vl=K for N=256,(16,16). */
        const ptrdiff_t SK = (ptrdiff_t)(p->S * K);
        p->leaf(x, p->planeA, p->planeA + NK, SK, SK, -SK, p->S * K);
    }
    double *cur = p->planeA, *nxt;
    for (int d = p->nf - 2; d >= 1; d--) {
        const rfft_stage_t *st = &p->st[d];
        const int r = st->radix, m = st->m, np = st->np;
        const size_t Q = st->Q;
        const ptrdiff_t QK = (ptrdiff_t)(Q * K);
        const ptrdiff_t QmK = (ptrdiff_t)(Q * (size_t)m * K);
        const size_t vlf = Q * K; /* full-width fold, always legal here */
        nxt = (cur == p->planeA) ? p->planeB : p->planeA;
        st->k0(cur, nxt, nxt + NK, QK, QmK, -QmK, vlf);
#ifdef VFFT_RFFT_RANGED
        if (st->hcr && st->kmax >= 1) {
            st->hcr(cur + (Q * (size_t)r) * K,
                    cur + (Q * (size_t)(r * (m - 1))) * K,
                    nxt + Q * K,
                    nxt + (Q * (size_t)(m - 1)) * K,
                    st->tw_re, st->tw_im, QK, QmK,
                    (ptrdiff_t)(Q * (size_t)r * K),
                    (ptrdiff_t)(Q * K), st->kmax, vlf);
        } else
#endif
        for (int k = 1; k <= st->kmax; k++)
            st->hc(cur + (Q * (size_t)(r * k)) * K,
                   cur + (Q * (size_t)(r * (m - k))) * K,
                   nxt + (Q * (size_t)k) * K,
                   nxt + (Q * (size_t)(m - k)) * K,
                   st->tw_re + (size_t)(k - 1) * r,
                   st->tw_im + (size_t)(k - 1) * r,
                   QK, QmK, vlf);
        if (st->has_mid)
            rfft_mid_column(r, m, np, Q, K, vlf,
                cur + (Q * (size_t)(r * (m / 2))) * K,
                st->mid_c, st->mid_s, 0, 0, nxt, NULL);
        cur = nxt;
    }

    /* stage 0: natural terminator (Q = 1) */
    {
        const rfft_stage_t *st = &p->st[0];
        const int r = st->radix, m = st->m;
        const ptrdiff_t mK = (ptrdiff_t)((size_t)m * K);

        st->k0(cur, p->nat_k0, p->nat_k0 + (size_t)r * K,
               (ptrdiff_t)K, (ptrdiff_t)K, -(ptrdiff_t)K, K);
        memcpy(out_re, p->nat_k0, K * 8);
        memset(out_im, 0, K * 8);
        for (int sI = 1; sI < (r + 1) / 2; sI++) {
            memcpy(out_re + (size_t)sI * (size_t)m * K,
                   p->nat_k0 + (size_t)sI * K, K * 8);
            memcpy(out_im + (size_t)sI * (size_t)m * K,
                   p->nat_k0 + (size_t)(r - sI) * K, K * 8);
        }
        if (r % 2 == 0) {
            memcpy(out_re + nh * K, p->nat_k0 + (size_t)(r / 2) * K, K * 8);
            memset(out_im + nh * K, 0, K * 8);
        }

#ifdef VFFT_RFFT_RANGED
        if (p->hcnr && (m - 1) / 2 >= 1) {
            p->hcnr(cur + ((size_t)r) * K,
                    cur + ((size_t)(r * (m - 1))) * K,
                    out_re + K, out_im + K,
                    out_re + (size_t)(m - 1) * K,
                    out_im + (size_t)(m - 1) * K,
                    st->tw_re, st->tw_im,
                    (ptrdiff_t)K, mK, mK,
                    (ptrdiff_t)((size_t)r * K), (ptrdiff_t)K,
                    (m - 1) / 2, K);
        } else
#endif
        for (int k = 1; k <= (m - 1) / 2; k++)
            p->hcn(cur + ((size_t)(r * k)) * K,
                   cur + ((size_t)(r * (m - k))) * K,
                   out_re + (size_t)k * K, out_im + (size_t)k * K,
                   out_re + (size_t)(m - k) * K,
                   out_im + (size_t)(m - k) * K,
                   st->tw_re + (size_t)(k - 1) * r,
                   st->tw_im + (size_t)(k - 1) * r,
                   (ptrdiff_t)K, mK, mK, K);

        if (st->has_mid)
            rfft_mid_column(r, m, st->np, 1, K, K,
                cur + ((size_t)(r * (m / 2))) * K,
                st->mid_c, st->mid_s, 1, nh, out_re, out_im);
    }
}

/* ===== rfft NATURAL forward, LANE-RANGE [k0,k0+kw) =====================
 * MT building block: process only lanes [k0, k0+kw) of the K batch (per-q, NOT
 * folded — the fold needs the full width). Lanes are independent and the shared
 * planeA/planeB/nat_k0 are LANE-INDEXED, so concurrent calls on disjoint [k0,kw)
 * ranges never collide. Output is the natural split half-spectrum for those lanes.
 * (kw need not be 8-aligned; the codelets' scalar tail covers the remainder.) */
static inline void rfft_execute_fwd_natural_range(const rfft_plan_t *p, const double *x,
                                                  double *out_re, double *out_im,
                                                  size_t k0, size_t kw)
{
    const int N = p->N; const size_t K = p->K; const size_t NK = (size_t)N * K;
    const size_t nh = (size_t)(N / 2);
    if (p->nf == 1) {
        p->leaf(x + k0, p->planeA + k0, p->planeA + k0 + NK,
                (ptrdiff_t)K, (ptrdiff_t)K, -(ptrdiff_t)K, kw);
        memcpy(out_re + k0, p->planeA + k0, kw * 8); memset(out_im + k0, 0, kw * 8);
        for (size_t f = 1; f < (size_t)((N + 1) / 2); f++) {
            memcpy(out_re + f * K + k0, p->planeA + f * K + k0, kw * 8);
            memcpy(out_im + f * K + k0, p->planeA + ((size_t)N - f) * K + k0, kw * 8);
        }
        if (N % 2 == 0) { memcpy(out_re + nh * K + k0, p->planeA + nh * K + k0, kw * 8);
                          memset(out_im + nh * K + k0, 0, kw * 8); }
        return;
    }
    /* LEAF (per-g over this lane range) */
    { const ptrdiff_t SK = (ptrdiff_t)(p->S * K);
      for (size_t g = 0; g < p->S; g++)
          p->leaf(x + g * K + k0, p->planeA + g * K + k0, p->planeA + g * K + k0 + NK,
                  SK, SK, -SK, kw); }
    /* intermediate stages d = nf-2 .. 1 (per-q, lane range) */
    double *cur = p->planeA, *nxt;
    for (int d = p->nf - 2; d >= 1; d--) {
        const rfft_stage_t *st = &p->st[d]; const int r = st->radix, m = st->m;
        const size_t Q = st->Q;
        const ptrdiff_t QK = (ptrdiff_t)(Q * K), QmK = (ptrdiff_t)(Q * (size_t)m * K);
        nxt = (cur == p->planeA) ? p->planeB : p->planeA;
        for (size_t q = 0; q < Q; q++) {
            double *curq = cur + q * K + k0, *nxtq = nxt + q * K + k0;
            st->k0(curq, nxtq, nxtq + NK, QK, QmK, -QmK, kw);
            for (int k = 1; k <= st->kmax; k++)
                st->hc(curq + (Q * (size_t)(r * k)) * K, curq + (Q * (size_t)(r * (m - k))) * K,
                       nxtq + (Q * (size_t)k) * K, nxtq + (Q * (size_t)(m - k)) * K,
                       st->tw_re + (size_t)(k - 1) * r, st->tw_im + (size_t)(k - 1) * r,
                       QK, QmK, kw);
            if (st->has_mid)
                rfft_mid_column(r, m, st->np, Q, K, kw, curq + (Q * (size_t)(r * (m / 2))) * K,
                                st->mid_c, st->mid_s, 0, 0, nxtq, NULL);
        }
        cur = nxt;
    }
    /* stage 0: natural terminator (lane range) */
    { const rfft_stage_t *st = &p->st[0]; const int r = st->radix, m = st->m;
      const ptrdiff_t mK = (ptrdiff_t)((size_t)m * K);
      st->k0(cur + k0, p->nat_k0 + k0, p->nat_k0 + (size_t)r * K + k0,
             (ptrdiff_t)K, (ptrdiff_t)K, -(ptrdiff_t)K, kw);
      memcpy(out_re + k0, p->nat_k0 + k0, kw * 8); memset(out_im + k0, 0, kw * 8);
      for (int sI = 1; sI < (r + 1) / 2; sI++) {
          memcpy(out_re + (size_t)sI * (size_t)m * K + k0, p->nat_k0 + (size_t)sI * K + k0, kw * 8);
          memcpy(out_im + (size_t)sI * (size_t)m * K + k0, p->nat_k0 + (size_t)(r - sI) * K + k0, kw * 8);
      }
      if (r % 2 == 0) { memcpy(out_re + nh * K + k0, p->nat_k0 + (size_t)(r / 2) * K + k0, kw * 8);
                        memset(out_im + nh * K + k0, 0, kw * 8); }
      for (int k = 1; k <= (m - 1) / 2; k++)
          p->hcn(cur + (size_t)(r * k) * K + k0, cur + (size_t)(r * (m - k)) * K + k0,
                 out_re + (size_t)k * K + k0, out_im + (size_t)k * K + k0,
                 out_re + (size_t)(m - k) * K + k0, out_im + (size_t)(m - k) * K + k0,
                 st->tw_re + (size_t)(k - 1) * r, st->tw_im + (size_t)(k - 1) * r,
                 (ptrdiff_t)K, mK, mK, kw);
      if (st->has_mid)
          rfft_mid_column(r, m, st->np, 1, K, kw, cur + (size_t)(r * (m / 2)) * K + k0,
                          st->mid_c, st->mid_s, 1, nh, out_re + k0, out_im + k0);
    }
}

#endif /* VFFT_RFFT_H */
