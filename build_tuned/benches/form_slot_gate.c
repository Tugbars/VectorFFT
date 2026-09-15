/* form_slot_gate.c — THIN DRIVER for the slot invariant: every kernel a form
 * resolver hands back for a slot must be CORRECT in that slot.
 *
 * No logic here (owner directive 2026-08-18, restated 2026-09-11 "it's not a
 * bench"): the walk is support/slot_check.h, the slot list and the probe are
 * planning/il_slot_probe.h. This file parses arguments, builds the planner
 * context, calls in, and prints. Nothing else.
 *
 * Takes no wisdom directory and writes nothing: it builds and checks, it
 * never races and never banks. A wrong-kind kernel can be memory-unsafe, so
 * the walk announces each slot on stderr before probing it — if this gate
 * dies instead of printing a verdict, the last stderr line names the slot.
 *
 * Usage: form_slot_gate.exe [--verbose]
 * Build: python build.py --src benches/form_slot_gate.c --vfft --compile */
#include <stdio.h>
#include <string.h>
#include "dp_planner_il.h"      /* the planner the probe drives, first */
#include "il_slot_probe.h"      /* the pair tier's slot list + probe    */

#define MAX_SLOTS 4096

int main(int argc, char **argv)
{
    static const int NS[] = { 16, 32, 64, 128, 256, 512, 1024 };
    static vfft_slot_t slots[MAX_SLOTS];
    static vfft_il_dp_context_t ctx;
    vfft_slot_tally_t t;
    vfft_slot_probe_t probe;
    int verbose = 0, nslot, wrong, maxN = 0;

    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--verbose")) verbose = 1;

    for (int i = 0; i < (int)(sizeof NS / sizeof NS[0]); i++)
        if (NS[i] > maxN) maxN = NS[i];

    nslot = vfft_il_slot_list(NS, (int)(sizeof NS / sizeof NS[0]), slots, MAX_SLOTS);
    if (nslot < 0)
    {
        printf("slot list exceeds %d — raise MAX_SLOTS rather than cover less\n", MAX_SLOTS);
        return 2;
    }

    printf("form slot gate: every kernel a resolver offers must be CORRECT in its slot\n");
    printf("%d slots = every legal arrangement at N=16..1024 x every form offered\n\n", nslot);

    vfft_il_dp_init(&ctx, maxN);
    probe.probe = vfft_il_slot_probe;
    probe.ctx = &ctx;
    wrong = vfft_slot_check(&probe, slots, nslot, &t, stdout, verbose);
    vfft_il_dp_destroy(&ctx);

    printf("\n%d ran and passed, %d absent (no kernel), %d WRONG\n",
           t.live, t.absent, t.wrong);
    printf("=== %s ===\n", wrong ? "*** FAIL ***" : "ALL PASS");
    return wrong != 0;
}
