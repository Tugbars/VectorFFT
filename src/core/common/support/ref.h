/* ref.h — the benchmark reference-backend interface.
 *
 * Declarations only: the vtable slot type and the reference-independent
 * helpers, all static inline. The backends that fill ref_vtable_t (e.g.
 * gauntlet/ref_fftw.h) live with the benches, because they bind external
 * libraries and core/ must not. Design: docs/roadmap/fftw_bench_design.md.
 *
 * A backend for FFTW must bind it at runtime (LoadLibraryA/GetProcAddress):
 * MKL exports fftw_* wrapper symbols, so a link that sees mkl_rt first
 * silently benchmarks MKL's wrappers under the FFTW label.
 */
#ifndef VFFT_PROTO_CORE_REF_H
#define VFFT_PROTO_CORE_REF_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ asserts */

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define REF_STATIC_ASSERT(c, m) _Static_assert((c), m)
#else
#  define REF_STATIC_ASSERT(c, m) \
     typedef char ref_sa_##__LINE__[(c) ? 1 : -1]
#endif

#define REF_DIE(...) do { \
    fprintf(stderr, "[ref] FATAL %s:%d — ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); abort(); } while (0)

/* Every plan must exist BEFORE the timed region. FFTW_MEASURE overwrites the
 * arrays while planning, so a plan built after the input is filled destroys it
 * and every subsequent number is garbage. Assert boundness at the top of any
 * timed body. */
#define REF_ASSERT_BOUND(arm) do { \
    if (!(arm)->plan) REF_DIE("arm '%s' timed with no plan — planning must " \
        "precede the timed region (FFTW_MEASURE overwrites its arrays)", (arm)->name); \
  } while (0)

/* The pacing/cachebust barrier must have run since the last timed sample. */
#define REF_ASSERT_PREBARRIER(st) do { \
    if (!(st)->barrier_done) REF_DIE("timed sample with no preceding barrier"); \
  } while (0)

/* ------------------------------------------------------------------- direction
 * `dir` is a PLAN parameter, never a per-execute argument. DFTI distances are
 * argument-anchored, not direction-anchored, so one descriptor reused across
 * directions without swapping them reads out of bounds. One plan object per
 * direction, always. */
typedef enum { REF_FWD = +1, REF_BWD = -1 } ref_dir_t;

/* ------------------------------------------------------------------- roles
 * The verdict role is PER-MODE, and matches what the MKL arm of that mode is:
 *   - mirror-regime modes (default 1a, --oop, --pad, --kzb, --k1*, --2d):
 *       verdict = MIRROR (our layout), HOME is the mandatory diagnostic.
 *   - home-regime modes (--r2c, --c2r, --padr2c, --2dr2c, --2dc2r, --zr2c):
 *       the MKL arm is a home arm with an untimed adapter and is the verdict,
 *       so FFTW gets the same; MIRROR is the diagnostic.
 * Never publish a mirror ratio beside a home ratio in one row. */
typedef enum {
    REF_ROLE_MIRROR = 0,  /* reference forced into OUR layout */
    REF_ROLE_HOME,        /* reference in ITS best layout, adapter untimed */
    REF_ROLE_LOOP,        /* our K=1 engine looped on the reference's memory */
    REF_ROLE_OURS,        /* the vfft arm */
    REF_ROLE_CONTROL      /* memcpy / no-op noise floor — MANDATORY per cell */
} ref_role_t;

static inline const char *ref_role_name(ref_role_t r)
{
    switch (r) {
    case REF_ROLE_MIRROR:  return "mir";
    case REF_ROLE_HOME:    return "home";
    case REF_ROLE_LOOP:    return "loop";
    case REF_ROLE_OURS:    return "ours";
    case REF_ROLE_CONTROL: return "ctrl";
    }
    return "?";
}

/* Which regime a MODE is in. Set once per run_*_cell. */
typedef enum { REF_REGIME_MIRROR = 0, REF_REGIME_HOME } ref_regime_t;

/* ------------------------------------------------------------------- flags
 * MEASURE is the only verdict rigour. ESTIMATE is deliberately unrepresentable:
 * it plans a structurally different library (at N=512 it picks the 128-bit
 * codelet family where MEASURE picks 256-bit), and its N=1000 guru split_dft
 * plan returned errors of 60+ where MEASURE gates at 1.96e-11. PATIENT and
 * EXHAUSTIVE are diagnostic only (2.2x spread between their plans at one
 * cell) and always run with a time limit. */
typedef enum {
    REF_PLAN_MEASURE = 0,   /* the only verdict-legal rigour */
    REF_PLAN_PATIENT,       /* diagnostic only */
    REF_PLAN_EXHAUSTIVE     /* diagnostic only */
} ref_rigour_t;

typedef struct {
    ref_rigour_t rigour;
    unsigned preserve_input : 1;  /* c2r: PRESERVE_INPUT is the VERDICT config */
    unsigned unaligned      : 1;  /* diagnostic; the arena is class 0 anyway */
    unsigned wisdom_only    : 1;  /* phase R/M of the two-phase wisdom runner */
    unsigned diagnostic     : 1;  /* row is role=DIAGNOSTIC, never a verdict */
} ref_flags_t;

/* ------------------------------------------------------------------- shape
 * ref_shape_t deliberately cannot express r2r: the trig transforms have never
 * been measured against fftw_plan_r2r_1d, and an r2r arm would put an
 * unmeasured comparison into a verdict column. */
typedef enum {
    REF_C2C = 0,   /* complex -> complex */
    REF_R2C,       /* real -> N/2+1 interleaved complex (== MKL CCE shape) */
    REF_C2R        /* N/2+1 interleaved complex -> real */
} ref_kind_t;

typedef enum {
    REF_LAYOUT_SPLIT_LM = 0,  /* our lane-major split: re[e*K+lane] */
    REF_LAYOUT_SPLIT_TC,      /* transform-contiguous split */
    REF_LAYOUT_IL_TC          /* interleaved transform-contiguous (FFTW's home) */
} ref_layout_t;

typedef struct {
    ref_kind_t   kind;
    ref_layout_t layout;
    int          rank;        /* 1 or 2 */
    size_t       n0, n1;      /* n1 == 0 for rank 1 */
    size_t       K;           /* howmany */
    ptrdiff_t    istride, idist;
    ptrdiff_t    ostride, odist;
    unsigned     inplace : 1;
} ref_shape_t;

/* A wrong stride triple is an OUT-OF-BOUNDS WRITE AT PLAN TIME, not a wrong
 * answer at gate time — the correctness gate runs far too late to catch it.
 * Every caller constructing a shape from compile-time constants must sit
 * behind this. Runtime shapes go through ref_shape_check(). */
#define REF_ASSERT_STRIDES(sh) \
    REF_STATIC_ASSERT((sh).istride > 0 && (sh).ostride > 0, \
                      "stride triple must be positive — a wrong triple is an " \
                      "OOB WRITE at plan time")

static inline int ref_shape_check(const ref_shape_t *s)
{
    if (!s || s->rank < 1 || s->rank > 2)          return 0;
    if (s->n0 == 0 || (s->rank == 2 && s->n1 == 0)) return 0;
    if (s->K == 0)                                  return 0;
    if (s->istride <= 0 || s->ostride <= 0)         return 0;
    if (s->K > 1 && (s->idist == 0 || s->odist == 0)) return 0;
    return 1;
}

/* ------------------------------------------------------------------- planes
 * FFTW hashes the split-plane deltas (ii - ri) and (io - ro) into its wisdom
 * key (dft/problem.c). Independently allocated planes miss wisdom across
 * launches, and the plan then drifts between runs (5 distinct plans in 8
 * launches, 9.4% spread): numbers stop being comparable between launches.
 * One block holding both planes at a size-derived offset makes the delta a
 * function of (N,K) alone, so wisdom hits with one plan_id across processes
 * (CSV column ref_planes=contiguous). */
static inline size_t ref_plane_stride(size_t bytes)
{
    return ((bytes + 4095u) & ~(size_t)4095u) + 64u; /* 4KB pitch + 64B skew */
}

typedef struct { double *re, *im; void *blk; size_t stride; } ref_planes_t;

/* ------------------------------------------------------------------- caps */
typedef struct {
    unsigned destroys_input : 1;  /* plan consumes its input (FFTW c2r default) */
    unsigned preserve_avail : 1;  /* PRESERVE_INPUT plannable for this shape */
    unsigned inplace_avail  : 1;
    unsigned split_avail    : 1;
    unsigned threads_avail  : 1;  /* 0 here: no fftw3_threads/fftw3_omp on host */
    unsigned natural_order  : 1;  /* output in natural order (FFTW/MKL: always) */
} ref_caps_t;

/* ------------------------------------------------------------------- vtable */
struct ref_plan;
typedef struct ref_plan ref_plan_t;

typedef struct {
    const char *name;                          /* "mkl" | "fftw" */
    const char *(*version)(void);              /* banner: proves WHAT is bound */
    int         (*available)(void);            /* 0 ⇒ mode must print n/a */

    ref_plan_t *(*plan)(ref_dir_t dir, const ref_shape_t *shape,
                        const ref_planes_t *planes, ref_flags_t flags);

    void        (*execute)(ref_plan_t *plan, void *in, void *out);

    void        (*destroy)(ref_plan_t *plan);

    /* Free-form, printed into the CSV: FFTW's sprint_plan, MKL's descriptor
     * summary. The FFTW string comes from the C heap: free() it, NEVER
     * fftw_free(); mixing them corrupts the heap at loop scale. */
    char       *(*describe)(const ref_plan_t *plan);

    ref_caps_t  (*caps)(const ref_plan_t *plan);
    uint64_t    (*plan_id)(const ref_plan_t *plan);   /* wisdom-hit witness */
    double      (*plan_ms)(const ref_plan_t *plan);
} ref_vtable_t;

/* ------------------------------------------------------------------- arms */
typedef struct {
    const char        *name;
    ref_role_t         role;
    const ref_vtable_t *vt;      /* NULL for the vfft and control arms */
    ref_plan_t        *plan;     /* NULL for vfft/control */
    void             (*run)(void *ctx);  /* the timed body */
    void             (*refill)(void *ctx);/* restores a destroyed input, UNTIMED */
    void              *ctx;
    double             ns_min, ns_med;   /* REF_STAT_MIN5 emits BOTH */
} ref_arm_t;

/* Five samples yield both statistics from the same data. The median is the
 * verdict; the min is carried for continuity with banked tables. */
#define REF_STAT_MIN5 5

/* ------------------------------------------------------------------- race
 * Order-neutralisation over N arms:
 *   n <= 4  : exhaustive rotation through all n! permutations (rot % n!)
 *   n >= 5  : seeded random permutation, sched=rand:<seed>, seed printed
 * A 3-cycle is NOT sufficient: it preserves every adjacency pair, which is
 * exactly the residual order bias to remove. */
typedef struct { unsigned n_arms; unsigned round; uint64_t seed; } ref_sched_t;

static inline unsigned ref_factorial(unsigned n)
{ unsigned f = 1u; while (n > 1u) f *= n--; return f; }

/* Fill perm[0..n-1] with the permutation for this round. */
static inline void ref_sched_perm(const ref_sched_t *sc, unsigned *perm)
{
    unsigned n = sc->n_arms, i;
    for (i = 0; i < n; i++) perm[i] = i;
    if (n <= 4u) {
        unsigned rank = sc->round % ref_factorial(n);      /* Lehmer decode */
        for (i = 0; i < n; i++) {
            unsigned f = ref_factorial(n - 1u - i), j = rank / f, k;
            rank %= f;
            { unsigned t = perm[i + j]; for (k = i + j; k > i; k--) perm[k] = perm[k-1]; perm[i] = t; }
        }
    } else {
        uint64_t x = sc->seed ^ (0x9E3779B97F4A7C15ull * (sc->round + 1u));
        for (i = n; i > 1u; i--) {                          /* Fisher-Yates */
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;        /* xorshift64 */
            { unsigned j = (unsigned)(x % i), t = perm[i-1u]; perm[i-1u] = perm[j]; perm[j] = t; }
        }
    }
}

/* Refuses to time a destructive arm that has no refill(): such a loop
 * transforms its own previous output from round 2 on. */
static inline int ref_race_check(const ref_arm_t *arms, unsigned n, int allow_unsound)
{
    unsigned i;
    for (i = 0; i < n; i++) {
        const ref_arm_t *a = &arms[i];
        if (!a->vt || !a->plan) continue;
        if (a->vt->caps(a->plan).destroys_input && !a->refill) {
            if (!allow_unsound)
                REF_DIE("arm '%s' destroys its input and has no refill — a "
                        "destructive timing loop transforms garbage from round 2. "
                        "Provide refill(), or pass --unsound-destroy to stamp the "
                        "row UNSOUND.", a->name);
            return 0;                                     /* caller stamps UNSOUND */
        }
    }
    return 1;
}

/* ------------------------------------------------------------------- ratio
 * Aborts when asked to make a HOME arm the verdict in a MIRROR-regime mode,
 * where HOME is only the diagnostic. */
static inline double ref_ratio(ref_regime_t regime, const ref_arm_t *ours,
                        const ref_arm_t *ref)
{
    if (regime == REF_REGIME_MIRROR && ref->role == REF_ROLE_HOME)
        REF_DIE("HOME arm '%s' used as the verdict in a MIRROR-regime mode — "
                "publish it as the diagnostic and race the mirror", ref->name);
    if (ours->ns_med <= 0.0) REF_DIE("ours ns_med not populated");
    return ref->ns_med / ours->ns_med;
}

/* ------------------------------------------------------------------- csv
 * Every mode routes its CSV path through csv_for(ref), so a run against one
 * reference never overwrites another reference's banked table. */
static inline const char *csv_for(const char *base, const char *refname,
                           char *buf, size_t cap)
{
    const char *dot = strrchr(base, '.');
    size_t stem = dot ? (size_t)(dot - base) : strlen(base);
    snprintf(buf, cap, "%.*s__%s%s", (int)stem, base, refname, dot ? dot : "");
    return buf;
}

#endif /* VFFT_PROTO_CORE_REF_H */
