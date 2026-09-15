/* bwd_forms_race.c — the BACKWARD forms race for the SHIPPED forward pairs.
 *
 * The full calibrator re-races the forward pools too, and on the planner's
 * clock near-equal arrangements trade places run to run (4x16 <-> 8x8 at
 * 64, 16x16 <-> 8x32 at 256, pair <-> ZTURN-T at 512); its backward rows then
 * sit on arrangements the paced forward verdicts never picked. This driver
 * takes each cell's SHIPPED natural row, and when it is a pair, runs the
 * planner's own backward race (_il_dp_race_bwd: every backward variant of
 * each slot the resolvers offer — since 2026-09-11 the tangent twins at
 * radix 8 and 16) for that exact arrangement, REPEATS it (cooldown between
 * repeats), prints every arm's clock, and banks the dir=bwd row on the
 * scratch store only when the winner is the same in every repeat.
 *
 * No planning logic here: the race is the planner's, the row writer is the
 * store's. Usage: bwd_forms_race.exe <scratch wisdir> <repeats> [N...]
 * Build: python build.py --src benches/bwd_forms_race.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "dp_planner_il.h"

static void cooldown_ms(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#endif
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* core 2 */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 3) { printf("usage: bwd_forms_race.exe <scratch wisdir> <repeats> [N...]\n"); return 2; }
    const char *wisdir = argv[1];
    const int reps = atoi(argv[2]) > 0 ? atoi(argv[2]) : 3;
    static const int DEF[] = { 32, 64, 128, 256, 512 };
    int cells[64], ncell = 0;
    if (argc > 3) for (int i = 3; i < argc && ncell < 64; i++) cells[ncell++] = atoi(argv[i]);
    else for (int i = 0; i < 5; i++) cells[ncell++] = DEF[i];
    int maxN = 0;
    for (int i = 0; i < ncell; i++) if (cells[i] > maxN) maxN = cells[i];

    static vfft_il_dp_context_t ctx;
    vfft_il_dp_init(&ctx, maxN);
    vfft_il_dp_set_patient(&ctx);

    vw2_store_t st;
    vw2_open(&st, wisdir, 1);
    int banked = 0;
    for (int ci = 0; ci < ncell; ci++)
    {
        const int N = cells[ci];
        vfft_oop_wisdom_entry_t e;
        memset(&e, 0, sizeof e);
        if (!vw2_oop_lookup_k1(&st, N, &e) || e.k1_il_route != VFFT_K1_IL_2P_PURE)
        {
            printf("N=%-5d not a shipped pair (route %d) — nothing to race\n", N, e.k1_il_route);
            continue;
        }
        printf("N=%-5d shipped pair %dx%d kv=0x%02x: backward race x%d\n", N, e.il_R1, e.il_R2, e.il_kv, reps);
        int win[16], nwin = 0;
        double win_ns[16];
        for (int r = 0; r < reps; r++)
        {
            vfft_il_cand_t c;
            memset(&c, 0, sizeof c);
            c.route = VFFT_K1_IL_2P_PURE;
            c.R1 = e.il_R1; c.R2 = e.il_R2;
            c.il_kv = e.il_kv;
            double ns = _il_dp_race_bwd(&ctx, N, &c, /*verbose=*/1);
            if (ns > 1e17) { printf("   repeat %d: no backward arm ran\n", r); continue; }
            win[nwin] = c.il_bkv; win_ns[nwin] = ns; nwin++;
            if (r + 1 < reps) cooldown_ms(3000);
        }
        int stable = nwin == reps;
        for (int i = 1; i < nwin; i++) if (win[i] != win[0]) stable = 0;
        if (!stable)
        {
            printf("   verdict: UNSTABLE across repeats (");
            for (int i = 0; i < nwin; i++) printf("%s0x%02x", i ? "," : "", win[i]);
            printf(") — not banked\n");
            continue;
        }
        double best = win_ns[0];
        for (int i = 1; i < nwin; i++) if (win_ns[i] < best) best = win_ns[i];
        {
            vw2_rec_t br;
            const char *why = NULL;
            if (vw2_oop_rec_k1_bwd(&br, N, VFFT_K1_IL_2P_PURE, e.il_R1, e.il_R2,
                                   win[0], best, "race", &why) == VW2_OK &&
                vw2_bank(&st, &br) == VW2_OK)
            {
                banked++;
                printf("   verdict: bkv=0x%02x %.1f ns, stable x%d — banked\n", win[0], best, reps);
            }
            else
                printf("   verdict: bkv=0x%02x stable but bank REFUSED: %s\n", win[0], why ? why : "?");
        }
    }
    if (banked && vw2_save(&st) != VW2_OK) printf("store did NOT save\n");
    vw2_close(&st);
    vfft_il_dp_destroy(&ctx);
    printf("# done: %d backward row(s) banked\n", banked);
    return 0;
}
