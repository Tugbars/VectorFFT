/* slot_check.h — the one SLOT-INVARIANT walk, shared by the form checkers.
 *
 * THE INVARIANT: every kernel a form resolver hands back for a slot must be
 * CORRECT in that slot. A resolver is a lookup table keyed by (radix,
 * variant, role); an entry is reachable the moment any plan puts that radix
 * in that role, whether or not today's banked plan happens to.
 *
 * The body owns the walk, as race.h owns the race protocol; the caller owns
 * the slots it enumerates and the probe that builds and checks one (see
 * planning/il_slot_probe.h for the interleaved pair tier). Drivers only parse
 * arguments and call in.
 *
 *   for each slot:   announce it on stderr, unbuffered, BEFORE probing
 *                    probe it -> OK / ABSENT / WRONG (+ a reason)
 *                    tally, and report every WRONG with its reason
 *   return           the number of WRONG slots (0 = the invariant holds)
 *
 * The announce comes first because a wrong-kind kernel is MEMORY-UNSAFE: a
 * plain-store kernel indexes zout[o*OLs+k] while a turned-store slot passes
 * OLs = R, so it writes past the plan's buffer and the process can die before
 * any check runs. The last stderr marker then names the slot that killed it.
 */
#ifndef VFFT_SLOT_CHECK_H
#define VFFT_SLOT_CHECK_H

#include <stdio.h>

/* a probe verdict */
enum {
    VFFT_SLOT_OK     =  0,   /* built, ran, and passed an independent check  */
    VFFT_SLOT_ABSENT =  1,   /* no kernel for this (radix, variant, role)    */
    VFFT_SLOT_WRONG  = -1    /* a kernel EXISTS and is wrong here — a defect */
};

/* one slot to check: an arrangement, the packed form code to install in it,
 * and the direction. (R1, R2) is what puts a radix in a given role, so a
 * caller that enumerates arrangements gates every role of every radix. */
typedef struct
{
    int N;
    int R1, R2;
    int form;        /* the packed form/variant code (kv forward, bkv backward) */
    int bwd;         /* 0 = the forward slots, 1 = the backward slots           */
} vfft_slot_t;

/* build the plan with `form` installed, run it, check it against something
 * independent. Sets *why to a short reason on ABSENT/WRONG (may be NULL). */
typedef struct
{
    int (*probe)(void *ctx, const vfft_slot_t *s, const char **why);
    void *ctx;
} vfft_slot_probe_t;

typedef struct { int live, absent, wrong; } vfft_slot_tally_t;

static void _vfft_slot_name(const vfft_slot_t *s, char *buf, size_t n)
{
    snprintf(buf, n, "N=%d %dx%d %s form=0x%02x",
             s->N, s->R1, s->R2, s->bwd ? "bwd" : "fwd", s->form);
}

/* Walk every slot. Returns the number of WRONG ones; `t` and `rep` optional. */
static int vfft_slot_check(const vfft_slot_probe_t *p,
                           const vfft_slot_t *slots, int nslot,
                           vfft_slot_tally_t *t, FILE *rep, int verbose)
{
    vfft_slot_tally_t tally;
    char name[64];
    int wrong = 0;
    tally.live = tally.absent = tally.wrong = 0;
    if (!p || !p->probe || !slots) return 0;
    for (int i = 0; i < nslot; i++)
    {
        const char *why = NULL;
        int v;
        _vfft_slot_name(&slots[i], name, sizeof name);
        /* unbuffered, before the probe: see THE ANNOUNCE above */
        fprintf(stderr, "\r[slot] %-34s", name);
        v = p->probe(p->ctx, &slots[i], &why);
        if (v == VFFT_SLOT_OK)
        {
            tally.live++;
            if (verbose && rep) fprintf(rep, "    ok      %s\n", name);
        }
        else if (v == VFFT_SLOT_ABSENT)
        {
            tally.absent++;
            if (verbose && rep)
                fprintf(rep, "    absent  %s   %s\n", name, why ? why : "");
        }
        else
        {
            tally.wrong++;
            wrong++;
            if (rep)
                fprintf(rep, "  *** WRONG *** %s: %s\n",
                        name, why ? why : "refused, no reason given");
        }
    }
    fprintf(stderr, "\r%50s\r", "");
    if (t) *t = tally;
    return wrong;
}

#endif /* VFFT_SLOT_CHECK_H */
