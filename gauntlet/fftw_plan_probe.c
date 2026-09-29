/* fftw_plan_probe.c -- what FFTW plans for a 1D r2c / c2r on THIS machine
 * (fftw_sprint_plan of the FFTW_MEASURE plan, out of place, one thread), so
 * a design discussion rests on the routes FFTW actually takes here.
 * Usage: fftw_plan_probe <N> [N ...]
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/fftw_plan_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include "ref_fftw.h"

int main(int argc, char **argv)
{
    fftwx_api_t api;
    char err[512];
    if (!fftwx_bind(&api, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    printf("%s (%s)\n", api.version, api.dll_path);
    for (int a = 1; a < argc; a++)
    {
        const int N = atoi(argv[a]);
        double *x = (double *)_aligned_malloc(sizeof(double) * ((size_t)N + 2), 64);
        double *z = (double *)_aligned_malloc(sizeof(double) * ((size_t)N + 2), 64);
        fftwx_plan pf = api.plan_dft_r2c_1d(N, x, (fftwx_complex *)z, FFTWX_MEASURE);
        fftwx_plan pb = api.plan_dft_c2r_1d(N, (fftwx_complex *)z, x, FFTWX_MEASURE);
        char *sf = pf ? api.sprint_plan(pf) : NULL, *sb = pb ? api.sprint_plan(pb) : NULL;
        printf("\n== N=%d r2c: %s\n== N=%d c2r: %s\n", N, sf ? sf : "(no plan)", N, sb ? sb : "(no plan)");
        free(sf); free(sb);   /* free(), never fftw_free (ref_fftw.h) */
        if (pf) api.destroy_plan(pf);
        if (pb) api.destroy_plan(pb);
        _aligned_free(x); _aligned_free(z);
    }
    return 0;
}
