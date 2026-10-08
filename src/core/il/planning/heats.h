/* heats.h - the measured TOURNAMENT IN HEATS (owner, 2026-10-04): many
 * candidates ranked by same-run races, never by numbers taken one after
 * another.
 *
 * A candidate timed on its own carries the machine's state at its own moment
 * (a burst, a boost-clock step, a background wake-up lands on it and on none
 * of its rivals); a pause before it evens out the heat it starts from, not
 * what happens while it runs. In ONE race whose arms alternate round by
 * round, whatever happens lands on every arm, and an arm's aggregate (the
 * fastest of its rounds) drops a round a burst hit. So candidates are ranked
 * in HEATS: balanced groups of at most `cap`, each raced by the caller's heat
 * function; each heat's best `carry` go on to the next round, until one heat
 * stands, and that heat's ranking is the result.
 *
 * Users: the flat DITs' chain search (chain_search.h) and the IL planner's
 * screening (dp_planner_il.h).
 */
#ifndef VFFT_IL_HEATS_H
#define VFFT_IL_HEATS_H

#include <stdlib.h>
#include <string.h>

#define VFFT_HEATS_MAX 32 /* candidates per heat, at most */

/* the caller's heat: race candidates idx[0..n) (n <= VFFT_HEATS_MAX) in one
 * same-run race; ns[k] = idx[k]'s aggregate, 1e18 = refused or wrong */
typedef void (*vfft_heat_fn)(void *hctx, const int *idx, int n, double *ns);

/* THE TOURNAMENT over idx[0..n): the last heat's ranking (fastest first)
 * fills top[0..keep) and, when topns is not NULL, their times in that heat.
 * Returns how many it filled. *heats counts the heats run. */
static int vfft_heats_run(vfft_heat_fn heat, void *hctx, const int *idx, int n, int cap, int keep, int carry,
                          int *top, double *topns, int *heats)
{
    int *surv = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    int *next = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    double hns[VFFT_HEATS_MAX];
    int ns = n, got = 0, h, k;
    if (cap > VFFT_HEATS_MAX)
        cap = VFFT_HEATS_MAX;
    if (cap < 2)
        cap = 2;
    if (carry < 1)
        carry = 1;
    if (!surv || !next)
    {
        free(surv);
        free(next);
        return 0;
    }
    memcpy(surv, idx, (size_t)n * sizeof(int));
    while (ns > 0)
    {
        const int nh = (ns + cap - 1) / cap;
        int nn = 0;
        for (h = 0; h < nh; h++)
        {
            const int lo = (int)((long)ns * h / nh), hi = (int)((long)ns * (h + 1) / nh);
            heat(hctx, surv + lo, hi - lo, hns);
            (*heats)++;
            if (nh == 1)
            { /* the last heat: its ranking */
                for (got = 0; got < keep; got++)
                {
                    int b = -1;
                    for (k = 0; k < hi - lo; k++)
                        if (hns[k] < 1e17 && (b < 0 || hns[k] < hns[b]))
                            b = k;
                    if (b < 0)
                        break;
                    top[got] = surv[lo + b];
                    if (topns)
                        topns[got] = hns[b];
                    hns[b] = 1e18;
                }
                free(surv);
                free(next);
                return got;
            }
            for (int c = 0; c < carry; c++)
            {
                int b = -1;
                for (k = 0; k < hi - lo; k++)
                    if (hns[k] < 1e17 && (b < 0 || hns[k] < hns[b]))
                        b = k;
                if (b < 0)
                    break;
                next[nn++] = surv[lo + b];
                hns[b] = 1e18;
            }
        }
        memcpy(surv, next, (size_t)nn * sizeof(int));
        ns = nn;
    }
    free(surv);
    free(next);
    return 0;
}

#endif /* VFFT_IL_HEATS_H */
