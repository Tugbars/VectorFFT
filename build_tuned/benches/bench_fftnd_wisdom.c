/* bench_fftnd_wisdom.c — validate the fftnd calibrate/persist/rebuild flow.
 *
 * Per cell:
 *   COLD: stride_plan_nd_wise on an empty wisdom file -> calibrates
 *         (candidate table printed), appends, builds. Timed.
 *   WARM: same call again -> file hit, rebuild without measuring. Timed
 *         (must be orders of magnitude below cold).
 *   Both plans roundtrip-verified; warm plan's fwd output memcmp'd against
 *   cold plan's (same recipe -> bit-identical).
 *   The banked line is printed for eyeballing the format.
 *
 * Build: python build.py --src benches/bench_fftnd_wisdom.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftnd_planner.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

static int g_fail = 0;
#define WIS_PATH "fftnd_wisdom_test.txt"

static void run_cell(int rank, const int *N, const vfft_proto_registry_t *reg) {
    size_t total = 1;
    for (int m = 0; m < rank; m++) total *= (size_t)N[m];
    printf("cell r=%d n=", rank);
    for (int m = 0; m < rank; m++) printf("%d%s", N[m], m+1<rank?"x":"");
    printf(" (%zu pts)\n", total);

    double t0 = vfft_proto_now_ns();
    stride_plan_t *pc = stride_plan_nd_wise(rank, N, reg, WIS_PATH, 1, 1);
    double cold_ms = (vfft_proto_now_ns() - t0) * 1e-6;

    t0 = vfft_proto_now_ns();
    stride_plan_t *pw = stride_plan_nd_wise(rank, N, reg, WIS_PATH, 1, 1);
    double warm_ms = (vfft_proto_now_ns() - t0) * 1e-6;

    if (!pc || !pw) {
        printf("  BUILD FAIL (cold=%p warm=%p)\n", (void*)pc, (void*)pw);
        g_fail++;
        if (pc) stride_plan_destroy(pc);
        if (pw) stride_plan_destroy(pw);
        return;
    }

    /* roundtrip + cold/warm bit-identity */
    double *xr=AALLOC(total*8),*xi=AALLOC(total*8);
    double *ar=AALLOC(total*8),*ai=AALLOC(total*8);
    double *br=AALLOC(total*8),*bi=AALLOC(total*8);
    srand(9);
    for (size_t i=0;i<total;i++){ xr[i]=2.0*rand()/RAND_MAX-1;
                                  xi[i]=2.0*rand()/RAND_MAX-1; }
    memcpy(ar,xr,total*8); memcpy(ai,xi,total*8);
    stride_execute_fwd(pc,ar,ai);
    memcpy(br,xr,total*8); memcpy(bi,xi,total*8);
    stride_execute_fwd(pw,br,bi);
    int bit = !memcmp(ar,br,total*8) && !memcmp(ai,bi,total*8);

    stride_execute_bwd(pw,br,bi);
    double sc=(double)total, rt=0;
    for (size_t i=0;i<total;i++){
        double rel=(fabs(br[i]-sc*xr[i])+fabs(bi[i]-sc*xi[i]))
                  /(fabs(sc*xr[i])+fabs(sc*xi[i])+1e-300);
        if (rel>rt) rt=rel;
    }
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)pw->override_data;
    int ok = bit && rt < 1e-11 && warm_ms < cold_ms * 0.25;
    if (!ok) g_fail++;
    printf("  verdict: s=%d blk=[", d->split);
    for (int m=0;m<rank-1;m++) printf("%zu%s", d->lane_block[m], m+2<rank?",":"");
    printf("] | cold %.0f ms -> warm %.2f ms | rt=%.1e | cold/warm bit=%s | %s\n",
           cold_ms, warm_ms, rt, bit?"EXACT":"**MISMATCH**", ok?"OK":"**FAIL**");

    AFREE(xr);AFREE(xi);AFREE(ar);AFREE(ai);AFREE(br);AFREE(bi);
    stride_plan_destroy(pc);
    stride_plan_destroy(pw);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    remove(WIS_PATH);

    int c3[3] = { 64, 64, 64 };
    run_cell(3, c3, &reg);
    int c4[4] = { 16, 16, 16, 16 };
    run_cell(4, c4, &reg);
    int c4b[4] = { 32, 32, 32, 64 };
    run_cell(4, c4b, &reg);

    printf("\nbanked wisdom (%s):\n", WIS_PATH);
    FILE *f = fopen(WIS_PATH, "r");
    if (f) { char l[1024]; while (fgets(l,sizeof l,f)) printf("  %s", l); fclose(f); }

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
