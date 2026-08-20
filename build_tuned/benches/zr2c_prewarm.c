/* zr2c_prewarm.c — bank kind-5 zr2c route verdicts for the standard cells.
 *
 * Create-only driver: vfft_create through the WISDOM path (no VFFT_ZR2C_ROUTE
 * env) makes _zr2c_build race route 0 vs route 1 in-context on every
 * unmeasured (N, transform, placement) slot and bank the winner into
 * oop_wisdom.txt (kind 5). Correctness is zr2c_fd_gate.c's job — this only
 * populates verdicts, so big N stays cheap (no O(N^2) reference).
 *
 * Run PINNED per the measurement protocol (core 2, HIGH), with output
 * REDIRECTED TO FILES — never .NET pipes + ReadToEnd (VFFT_ZRACE_VERBOSE
 * fills the 4 KB stderr pipe and deadlocks both processes; hit 2026-08-13):
 *   powershell: $p = Start-Process .\benches\zr2c_prewarm.exe -PassThru `
 *     -NoNewWindow -RedirectStandardOutput out.txt -RedirectStandardError err.txt
 *   Start-Sleep -m 80; $p.ProcessorAffinity=0x4; $p.PriorityClass='High'
 * (needs C:\mingw152\mingw64\bin on PATH for the runtime DLLs under
 * PowerShell). Cells already banked replay instantly (rerun-safe); the
 * process AUTOSAVES per bank, so a killed run keeps its progress. Quiet-day
 * re-race = delete the kind-5 lines (PowerShell, never Git-Bash sed — CRLF)
 * and rerun; note N=510 is banked by zr2c_fd_gate.c, not this cell list.
 *
 * Build (from build_tuned/): python build.py --src benches/zr2c_prewarm.c \
 *   --vfft --compile   (run it pinned yourself; build.py's [run] is unpinned)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vfft.h"

static vfft_wisdom *g_W;

static void prewarm(int N)
{
    static const int tf[2] = { VFFT_R2C, VFFT_C2R };
    static const int pl[2] = { VFFT_OUTOFPLACE, VFFT_INPLACE };
    for (int t = 0; t < 2; t++)
        for (int p = 0; p < 2; p++) {
            vfft_config_t cfg;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = (vfft_transform_t)tf[t];
            cfg.placement = (vfft_placement_t)pl[p];
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.nthreads = 1;
            cfg.wisdom = g_W;
            cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
            vfft_plan h = vfft_create(&cfg);
            printf("  N=%-6d %s %-3s : %s\n", N,
                   t ? "c2r" : "r2c", p ? "ip" : "oop",
                   h ? "created (verdict banked/served)" : "CREATE FAILED");
            fflush(stdout);
            if (h) vfft_destroy(h);
        }
}

int main(int argc, char **argv)
{
    /* CWD-proof wisdom resolution (same probe as zr2c_fd_gate.c). */
    const char *cand[3] = {
        (argc >= 2) ? argv[1] : NULL,
        "../src/dag-fft-compiler/generator/generated",
        "../../src/dag-fft-compiler/generator/generated",
    };
    const char *wdir = NULL;
    for (int i = 0; i < 3 && !wdir; i++) {
        if (!cand[i]) continue;
        char pp[512];
        snprintf(pp, sizeof pp, "%s/wisdom2_oop.txt", cand[i]); /* dir marker = the LIVE store (legacy file is frozen) */
        FILE *pf = fopen(pp, "r");
        if (pf) { fclose(pf); wdir = cand[i]; }
    }
    if (!wdir) {
        printf("zr2c_prewarm: wisdom dir NOT FOUND — nothing to bank into\n");
        return 1;
    }
    g_W = vfft_wisdom_load(wdir);
    printf("zr2c prewarm — banking kind-5 route verdicts, wisdom: %s\n", wdir);
    static const int Ns[] = { 256, 512, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536 };
    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++)
        prewarm(Ns[i]);
    printf("done.\n");
    if (g_W) vfft_wisdom_free(g_W);
    return 0;
}
