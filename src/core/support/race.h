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
 * The protocol constants (rounds, reps, aggregate, alternation, warm-up, the
 * per-sample reset) are parameters: each site keeps its own. So is the verdict
 * rule; vfft_race_beats() spells the hysteresis form for a site that has an
 * incumbent.
 *
 * Depends on support/race_timing.h and libc only; arms are opaque (fn, ctx)
 * pairs. No mutable file-scope state: a static in a header is one copy per
 * includer.
 */
#ifndef VFFT_SUPPORT_RACE_H
#define VFFT_SUPPORT_RACE_H
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include "support/race_timing.h" /* _il_ab_now: the shared monotonic clock */

#define VFFT_RACE_MAX_ARMS 160  /* _il2d_axis_race runs up to 140: (3 row routes + the two-pass route x 4 tile steps) x (14 band widths + 6 column tiles) */
#define VFFT_RACE_MAX_ROUNDS 96 /* _calibrate_pad runs RR=81 at PATIENT */

/* The pause BETWEEN races, never inside one: thermal drift re-ranks plans
 * (+/-5% swings flip verdicts). It runs once, before the warm-up, by the
 * class a site declares in vfft_race_proto_t.pace:
 *   1 = single-thread arms: pause, then at least one untimed pass, because
 *       a core runs 1.5-5x slow for the first milliseconds after a sleep and
 *       a race without warm-up would hand that to arm 0 of round 0;
 *   0 = threaded arms, or a site not yet classified: no pause. A 200 ms pause
 *       parks the worker team and the timed block then pays the wake (a
 *       0.2x-vs-4x artifact). Classify by reading the arm function, not the
 *       proto's shape.
 * The DP planner's VFFT_IL_DP_PACE_MS aliases this constant. */
#define VFFT_RACE_PACE_MS 200
#if defined(_WIN32)
extern __declspec(dllimport) void __stdcall Sleep(unsigned long ms);
static inline void vfft_race_sleep_ms(int ms) { Sleep((unsigned long)ms); }
#else
#include <time.h>
static inline void vfft_race_sleep_ms(int ms)
{
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}
#endif

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
    int pace;                /* 1 = single-thread arms: VFFT_RACE_PACE_MS before the
                              * race + at least one untimed pass; 0 = threaded arms or
                              * unclassified: no pause (see VFFT_RACE_PACE_MS) */
} vfft_race_proto_t;

/* median of n, sorting v in place; the middle element for odd n */
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

/* The create-race counter (defined in vfft.c; the fingerprint's races=
 * field). Every race that runs through this body counts. */
extern long _vfft_create_race_count;

/* Time n arms under p. ns[i] receives arm i's aggregate (ns per execution).
 * Returns the index of the smallest aggregate, the FIRST arm keeping ties —
 * the bare "<" verdict; a site with an incumbent applies vfft_race_beats()
 * to ns[] instead of using the return value. Returns -1 on a malformed
 * protocol (nothing timed, ns[] untouched). */
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
        fprintf(stderr, " (rounds=%d reps=%d%s)\n", p->rounds, reps,
                p->pace ? " paced" : "");
    }
    {   /* the pause BETWEEN races, and the untimed pass that absorbs the
         * cold core it leaves behind (VFFT_RACE_PACE_MS) */
        int warm = p->warm;
        if (p->pace)
        {
            vfft_race_sleep_ms(VFFT_RACE_PACE_MS);
            if (warm < 1) warm = 1;
        }
        for (int w = 0; w < warm; w++)
            for (int a = 0; a < n; a++)
                arms[a].run(arms[a].ctx);
    }
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
 * it is faster by more than the margin (hyst = 0.97 for a 3% margin). */
static inline int vfft_race_beats(double challenger_ns, double incumbent_ns,
                                  double hyst)
{
    return challenger_ns < incumbent_ns * hyst;
}

#endif /* VFFT_SUPPORT_RACE_H */
