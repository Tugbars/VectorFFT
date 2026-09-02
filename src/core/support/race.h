/* race.h — the one race body shared by the in-process racers.
 *
 * docs/design/planning_model.md §4 declares that every race has the same
 * five parts — ARMS → PROTOCOL → VERDICT → KEY → BANK — and
 * docs/design/vfft_front_door.md §5 states the house protocol once. This
 * header is the executable form of the first two parts and nothing else:
 * the caller still owns the arms it builds, the verdict rule it applies to
 * the aggregates this returns, the key, the bank and the log.
 *
 * WHAT IS SHARED
 * --------------
 *   for each warm-up pass:       run every arm once, untimed
 *   for each round:              (odd rounds in reverse arm order when
 *                                 p->alternate — A,B / B,A / A,B ...)
 *       for each arm:            reset(); t0; reps × run(); sample = dt/reps
 *   per arm:                     aggregate the samples (median / min / mean)
 *   return                       the index of the smallest aggregate,
 *                                first arm keeping ties
 *
 * WHAT IS DELIBERATELY NOT SHARED — the protocol CONSTANTS
 * --------------------------------------------------------
 * Round count, reps, aggregate, alternation, warm-up and the per-sample
 * reset are PARAMETERS. build_tuned/race_census.py records 14 distinct
 * protocols across the racers, and no check in the harness can tell
 * whether a unified protocol still picks the same winner; finding out means
 * re-racing, which is forbidden during development (memory: racing budget).
 * So a site migrated onto this body keeps its exact constants and its
 * verdict is unchanged by construction; collapsing the constants is the
 * pre-release sweep's decision, made with the clock, not here.
 *
 * The verdict rule stays at the site for the same reason: eight sites use a
 * 3% hysteresis toward an incumbent, five a 5% margin, the rest a bare
 * "<". vfft_race_beats() spells the hysteresis form once so a site does not
 * retype the multiply, but which arm is the incumbent is the site's
 * knowledge.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Depends on support/race_timing.h and <string.h> only: no plan type, no
 * wisdom type, no engine header. Arms are opaque (fn, ctx) pairs. No
 * mutable file-scope state — a static in a header is one copy per includer.
 *
 * ON THE CLOCK
 * ------------
 * _il_ab_now (clock_gettime(CLOCK_MONOTONIC)) — race_timing.h records that
 * on this toolchain it is the same 100 ns QPC tick as vfft_proto_now_ns, so
 * the two spellings are interchangeable for an interval; on a POSIX host it
 * is the monotonic clock the sites written with clock_gettime already use.
 */
#ifndef VFFT_SUPPORT_RACE_H
#define VFFT_SUPPORT_RACE_H
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include "support/race_timing.h" /* _il_ab_now: the shared monotonic clock */

#define VFFT_RACE_MAX_ARMS 32   /* _il2d_axis_race runs up to 28 */
#define VFFT_RACE_MAX_ROUNDS 96 /* _calibrate_pad runs RR=81 at PATIENT */

typedef struct
{
    const char *name;        /* for the site's log line; may be NULL */
    void (*run)(void *ctx);  /* one execution of this arm */
    void *ctx;
} vfft_race_arm_t;

typedef enum
{
    VFFT_RACE_MEDIAN = 0, /* reject one outlier either way (rounds odd) */
    VFFT_RACE_MIN = 1,    /* the least-disturbed sample */
    VFFT_RACE_MEAN = 2    /* vfft_natorder_race averages */
} vfft_race_agg_t;

typedef struct
{
    int rounds;              /* samples per arm, 1..VFFT_RACE_MAX_ROUNDS */
    int reps;                /* executions per sample, >= 1 */
    vfft_race_agg_t agg;
    int alternate;           /* 1: odd rounds run the arms in reverse order */
    int warm;                /* untimed passes per arm before round 0 */
    void (*reset)(void *ctx); /* before EVERY timed sample: e.g. re-seed an
                              * aliased in-place buffer (repeated in-place
                              * fwd walks into inf); NULL = no reset */
    void *reset_ctx;
} vfft_race_proto_t;

/* median of n in place; n odd returns the middle element, which is what
 * _il_ab_med9 (v[4] of 9) and the inline median-of-5 (v[2]) return. */
static inline double vfft_race_median(double *v, int n)
{
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && v[j] < v[j - 1]; j--)
        {
            double t = v[j];
            v[j] = v[j - 1];
            v[j - 1] = t;
        }
    return v[n / 2];
}

static inline double vfft_race_aggregate(vfft_race_agg_t agg, double *v, int n)
{
    if (agg == VFFT_RACE_MIN)
    {
        double m = v[0];
        for (int i = 1; i < n; i++)
            if (v[i] < m)
                m = v[i];
        return m;
    }
    if (agg == VFFT_RACE_MEAN)
    {
        double s = 0.0;
        for (int i = 0; i < n; i++)
            s += v[i];
        return s / (double)n;
    }
    return vfft_race_median(v, n);
}

/* Time n arms under p. ns[i] receives arm i's aggregate (ns per execution).
 * Returns the index of the smallest aggregate, the FIRST arm keeping ties —
 * the bare "<" verdict; a site with an incumbent applies vfft_race_beats()
 * to ns[] instead of using the return value. Returns -1 on a malformed
 * protocol (nothing timed, ns[] untouched). */
/* THE create-race counter (vfft.c; fingerprint field races=). Every race
 * that runs through this body counts, by construction — until 2026-09-02
 * only three hand-placed bumps in vfft.c counted, so every extracted race
 * (cascade, IL attach, natural order, 2D chain/axis/column-MT, zt_mt, plane
 * queue, prime method, pair order, odd-real bridge) reported races=0: a
 * false zero that made the harness's replay-purity check blind. */
extern long _vfft_create_race_count;

static int vfft_race_run(const vfft_race_proto_t *p, const vfft_race_arm_t *arms,
                         int n, double *ns)
{
    double s[VFFT_RACE_MAX_ARMS][VFFT_RACE_MAX_ROUNDS];
    int reps = p->reps < 1 ? 1 : p->reps;
    if (n < 1 || n > VFFT_RACE_MAX_ARMS || p->rounds < 1 ||
        p->rounds > VFFT_RACE_MAX_ROUNDS)
        return -1;
    _vfft_create_race_count++;              /* past a wisdom hit: a clock decides */
    if (getenv("VFFT_RACE_LOG"))            /* name every race that runs */
    {
        fprintf(stderr, "[race]");
        for (int a = 0; a < n; a++)
            fprintf(stderr, " %s", arms[a].name ? arms[a].name : "?");
        fprintf(stderr, " (rounds=%d reps=%d)\n", p->rounds, reps);
    }
    for (int w = 0; w < p->warm; w++)
        for (int a = 0; a < n; a++)
            arms[a].run(arms[a].ctx);
    for (int r = 0; r < p->rounds; r++)
        for (int k = 0; k < n; k++)
        {
            const int a = (p->alternate && (r & 1)) ? n - 1 - k : k;
            if (p->reset)
                p->reset(p->reset_ctx);
            const double t0 = _il_ab_now();
            for (int i = 0; i < reps; i++)
                arms[a].run(arms[a].ctx);
            s[a][r] = (_il_ab_now() - t0) / reps;
        }
    int best = 0;
    for (int a = 0; a < n; a++)
    {
        ns[a] = vfft_race_aggregate(p->agg, s[a], p->rounds);
        if (ns[a] < ns[best])
            best = a;
    }
    return best;
}

/* The hysteresis verdict: the challenger displaces the incumbent only when
 * it is faster by more than the margin (hyst = 0.97 is the house 3%). */
static inline int vfft_race_beats(double challenger_ns, double incumbent_ns,
                                  double hyst)
{
    return challenger_ns < incumbent_ns * hyst;
}

#endif /* VFFT_SUPPORT_RACE_H */
