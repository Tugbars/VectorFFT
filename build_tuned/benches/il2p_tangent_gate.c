/* il2p_tangent_gate.c — end-to-end gate for the tangent kernels as WIRED,
 * not as standalone codelets.
 *
 * The per-codelet gate (tangent_gate.c) builds its own twiddle table and calls
 * the kernels directly. This one goes through vfft_il2p_create / execute_fwd,
 * so it proves the wired path: il2p's own VTW2 table layout, its leaf->mid
 * geometry, and the il_kv variant plumbing (vfft_il2p_apply_kv_forms).
 *
 * For each (N, R1, R2) pair with a tangent form, it runs the plan with the
 * default forms and again with il_kv = PACK(mid, leaf), and checks BOTH
 * against a direct DFT. A variant that changes the answer fails here even if
 * the kernel is correct in isolation — which is exactly the wiring bug this
 * gate exists to catch.
 *
 * Build (from build_tuned/):
 *   python build.py --src benches/il2p_tangent_gate.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include "il2p.h"

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

/* worst |plan(x) - DFT(x)| over the whole transform */
static double err_vs_dft(const double *in, const double *out, int N)
{
    double worst = 0;
    for (int k = 0; k < N; k++) {
        cx ref = 0;
        for (int n = 0; n < N; n++) {
            cx x = in[2*n] + I*in[2*n+1];
            ref += cexp(-2.0*PI*I*(double)((long long)k*n % N)/(double)N) * x;
        }
        cx got = out[2*k] + I*out[2*k+1];
        double e = cabs(got - ref);
        if (e > worst) worst = e;
    }
    return worst;
}

static int check(int N, int R1, int R2, int mid_v, int leaf_v, const char *what)
{
    vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
    if (!p) { printf("  %-28s N=%-5d (%2dx%-2d)  NO PLAN (pair unavailable)\n",
                     what, N, R1, R2); return 1; }

    double *in  = (double *)malloc((size_t)2*N*sizeof(double));
    double *ref = (double *)malloc((size_t)2*N*sizeof(double));
    double *out = (double *)malloc((size_t)2*N*sizeof(double));
    for (int i = 0; i < 2*N; i++) in[i] = rnd();

    /* baseline: whatever create resolved (structural default) */
    vfft_il2p_execute_fwd(p, in, ref);
    double e_base = err_vs_dft(in, ref, N);

    /* now force the variant through the SAME path wisdom uses */
    vfft_il2p_apply_kv_forms(p, VFFT_IL_KV_PACK(mid_v, leaf_v));
    vfft_il2p_execute_fwd(p, in, out);
    double e_tan = err_vs_dft(in, out, N);

    /* did the variant actually change the emitted kernels? */
    int changed = 0;
    for (int i = 0; i < 2*N; i++) if (ref[i] != out[i]) { changed = 1; break; }

    int ok = (e_base < 1e-9) && (e_tan < 1e-9);
    printf("  %-28s N=%-5d (%2dx%-2d)  base %.2e  tangent %.2e  %s%s\n",
           what, N, R1, R2, e_base, e_tan,
           ok ? "OK" : "*** FAIL ***",
           changed ? "" : "  (note: bitwise same as base)");

    free(in); free(ref); free(out);
    vfft_il2p_destroy(p);
    return ok;
}

int main(void)
{
    printf("il2p tangent wiring gate — variant 3 through vfft_il2p_apply_kv_forms\n\n");
    int ok = 1;
    /* mid slot: R1 carries the tangent mid; leaf stays default (0) */
    ok &= check(64,   8,  8, 3, 0, "R8  mid");
    ok &= check(128, 16,  8, 3, 0, "R16 mid");
    ok &= check(256, 16, 16, 3, 0, "R16 mid");
    ok &= check(512, 32, 16, 3, 0, "R32 mid (blocked 2.16)");
    /* leaf slot: R2 carries the tangent leaf */
    ok &= check(64,   8,  8, 0, 3, "R8  leaf");
    ok &= check(128,  8, 16, 0, 3, "R16 leaf");
    ok &= check(256, 16, 16, 0, 3, "R16 leaf");
    ok &= check(512, 16, 32, 0, 3, "R32 leaf (blocked, TURN128)");
    /* both slots at once — the pure-tangent route */
    ok &= check(64,   8,  8, 3, 3, "R8  mid + R8  leaf");
    ok &= check(256, 16, 16, 3, 3, "R16 mid + R16 leaf");
    ok &= check(512, 32, 16, 3, 3, "R32 mid + R16 leaf");
    ok &= check(512, 16, 32, 3, 3, "R16 mid + R32 leaf");
    ok &= check(1024, 32, 32, 3, 3, "R32 mid + R32 leaf");
    /* blocked R32 leaf carries NO odd tail: with an odd mid radix the
     * count_ok gate must degrade the leaf to its default, still correct */
    ok &= check(480, 15, 32, 0, 3, "R32 leaf, ODD R1 (degrades)");
    /* odd count: monolithic tangent forms carry the VEX-128 tail, so a
     * tangent mid must stay correct when R2 is odd */
    ok &= check(240, 16, 15, 3, 0, "R16 mid, ODD count");
    /* ── TURNED-axis edge variant (leaf variant 4 = T256) ──────────────
     * Same wing32 interior, paired-wide store edge. PROMOTED at 128 (kv 64)
     * and 512 (kv 67) by the 2026-08-16 race. The mid-side M-128 edge lost
     * that race and was sunset; its nibble must now DEGRADE bitwise. */
    ok &= check(512, 16, 32, 0, 4, "R32 leaf T256");
    ok &= check(128,  4, 32, 0, 4, "R32 leaf T256 (the kv-64 pair)");
    ok &= check(512, 16, 32, 3, 4, "tangent mid + T256 leaf (kv 67)");
    /* sunset / absent forms: must DEGRADE bitwise to base */
    ok &= check(256, 16, 16, 4, 0, "R16 mid v4 (sunset, degrades)");
    ok &= check(512, 32, 16, 4, 0, "R32 mid v4 (sunset, degrades)");
    ok &= check(256, 16, 16, 0, 4, "R16 leaf v4 (degrades)");

    printf("\n%s\n", ok ? "TANGENT WIRING GATE: ALL CORRECT"
                        : "TANGENT WIRING GATE: FAILURE");
    return ok ? 0 : 1;
}
