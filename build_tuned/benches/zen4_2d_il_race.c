/* zen4_2d_il_race.c — 2D C2C, K=1, INTERLEAVED: front door vs genuine FFTW.
 *
 * OUR ARM  vfft_create(C2C, dims=2, IN-PLACE, layout=INTERLEAVED, order=DEFAULT)
 *          at VFFT_PATIENT with wisdom_write=1. The il2d tier races its axes
 *          AT CREATE (measurement_arms.md E1: column chain, band width wl, row
 *          route, column kernel forms) and banks into $VFFT_WISDOM_DIR; the
 *          K=1 IL row cells at N2 must already be calibrated (calibrate_k1).
 *          One z buffer, z -> z (the --2dil bench's contract).
 * FFTW ARM plan_dft_2d(N1, N2, z, z, FORWARD, PATIENT) in-place, plus the
 *          out-of-place plan as a second FFTW number, both via ref_fftw.h
 *          (runtime-bound, genuineness asserted).
 * ORDER    the IL tier serves ord=scr on n1 (rows permuted); FFTW is natural.
 *          A direct elementwise compare is therefore impossible, and a naive
 *          2D DFT at this size is impractical, so BOTH arms are gated by
 *          roundtrip: bwd(fwd(z)) == N1*N2*z (the --2dil bench's gate).
 * PROTOCOL warmup 10, best-of-5, reps=2e6/(T+1) clamped, 400 ms clock-ramp
 *          spin, cachebust + cool_ms between engines, flip, core pin.
 *
 * Build: python build.py --src benches/zen4_2d_il_race.c --vfft --compile
 * Run  : zen4_2d_il_race.exe [N1] [N2] [cool_ms] [flip] [core]              */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
#include "ref_fftw.h"

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int reps_for(size_t T) { int r = (int)(2e6 / (double)(T + 1)); return r < 8 ? 8 : r > 100000 ? 100000 : r; }
static void pace(int ms) { if (ms > 0) Sleep((DWORD)ms); }
static void cachebust(void)
{
    size_t s = 32u * 1024 * 1024 / sizeof(double);
    double *j = _aligned_malloc(s * sizeof *j, 64); volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; _aligned_free(j);
}
static void fill(double *z, size_t T) { srand(42u + (unsigned)T); for (size_t i = 0; i < 2 * T; i++) z[i] = (double)rand() / RAND_MAX - 0.5; }
static double rt_err(const double *z0, const double *rt, size_t T)
{
    double me = 0, mm = 0, inv = 1.0 / (double)T;
    for (size_t i = 0; i < 2 * T; i++) {
        double e = fabs(rt[i] * inv - z0[i]), m = fabs(z0[i]);
        if (e > me) me = e; if (m > mm) mm = m;
    }
    return mm > 0 ? me / mm : me;
}
#define TIME(best, CALL) do { int _r = reps_for(T); (best) = 1e18;            \
    for (int _w = 0; _w < 10; _w++) { CALL; }                                 \
    for (int _t = 0; _t < 5; _t++) { double _t0 = now_ns();                   \
        for (int _i = 0; _i < _r; _i++) { CALL; }                              \
        double _ns = (now_ns() - _t0) / _r; if (_ns < (best)) (best) = _ns; } } while (0)

int main(int argc, char **argv)
{
    int N1 = argc > 1 ? atoi(argv[1]) : 512, N2 = argc > 2 ? atoi(argv[2]) : N1;
    int cool_ms = argc > 3 ? atoi(argv[3]) : 200, flip = argc > 4 ? atoi(argv[4]) : 0;
    int core = argc > 5 ? atoi(argv[5]) : 2;
    const size_t T = (size_t)N1 * (size_t)N2;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (core >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    /* ---- FFTW: bind, plan PATIENT (in-place fwd/bwd + oop fwd), gate ------ */
    fftwx_api_t fx; char err[512];
    if (!fftwx_bind(&fx, err, sizeof err)) { fprintf(stderr, "FFTW bind failed: %s\n", err); return 2; }
    const char *wis = getenv("VFFT_FFTW_WIS");
    int wis_loaded = wis && fx.import_wisdom_from_filename(wis);
    double *fz = fx.fmalloc(16 * T), *fo = fx.fmalloc(16 * T), *f0 = _aligned_malloc(16 * T, 64);
    double t0 = now_ns();
    fftwx_plan fpi = fx.plan_dft_2d(N1, N2, (fftwx_complex *)fz, (fftwx_complex *)fz, FFTWX_FORWARD,  FFTWX_PATIENT);
    fftwx_plan fbi = fx.plan_dft_2d(N1, N2, (fftwx_complex *)fz, (fftwx_complex *)fz, FFTWX_BACKWARD, FFTWX_PATIENT);
    fftwx_plan fpo = fx.plan_dft_2d(N1, N2, (fftwx_complex *)fz, (fftwx_complex *)fo, FFTWX_FORWARD,  FFTWX_PATIENT);
    double fplan_ms = (now_ns() - t0) / 1e6;
    if (wis) fx.export_wisdom_to_filename(wis);
    fill(fz, T); memcpy(f0, fz, 16 * T);              /* AFTER planning */
    fx.execute(fpi); fx.execute(fbi);
    double fgate = rt_err(f0, fz, T);

    /* ---- ours: front door, PATIENT, races + banks the il2d axes on a miss - */
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_INPLACE; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_DEFAULT; cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1; cfg.rigor = VFFT_PATIENT; cfg.nthreads = 1; cfg.wisdom_write = 1;
    double *vz = _aligned_malloc(16 * T, 64), *v0 = _aligned_malloc(16 * T, 64);
    t0 = now_ns();
    vfft_plan vp = vfft_create(&cfg);
    double vplan_ms = (now_ns() - t0) / 1e6;
    if (!vp) { fprintf(stderr, "vfft_create REFUSED\n"); return 2; }
    fill(vz, T); memcpy(v0, vz, 16 * T);
    vfft_execute(vp, VFFT_FORWARD, vz, NULL, vz, NULL);
    vfft_execute(vp, VFFT_BACKWARD, vz, NULL, vz, NULL);
    double vgate = rt_err(v0, vz, T);

    printf("zen4_2d_il_race  %dx%d K=1 c2c INTERLEAVED in-place  reps=%d x best-of-5  core=%d cool=%dms flip=%d\n"
           "  vfft %s isa=%s wisdom_dir=%s  (create %.0f ms)\n  %s  dll=%s  (plans %.0f ms, wisdom %s)\n",
           N1, N2, reps_for(T), core, cool_ms, flip, vfft_version(), vfft_isa(),
           getenv("VFFT_WISDOM_DIR") ? getenv("VFFT_WISDOM_DIR") : "(UNSET)", vplan_ms,
           fx.version, fx.dll_path, fplan_ms, wis ? (wis_loaded ? "loaded" : "cold") : "none");

    { volatile double acc = 1.0; double t1 = now_ns();               /* clock ramp */
      while (now_ns() - t1 < 400e6) { for (int i = 0; i < 4096; i++) acc = acc * 1.0000001 + 1e-9; } (void)acc; }
    double vns = 0, fns = 0, fons = 0;
    if (flip) {
        TIME(fns, fx.execute(fpi)); TIME(fons, fx.execute(fpo)); cachebust(); pace(cool_ms);
        TIME(vns, vfft_execute(vp, VFFT_FORWARD, vz, NULL, vz, NULL));
    } else {
        TIME(vns, vfft_execute(vp, VFFT_FORWARD, vz, NULL, vz, NULL)); cachebust(); pace(cool_ms);
        TIME(fns, fx.execute(fpi)); TIME(fons, fx.execute(fpo));
    }
    double gf = 5.0 * (double)T * log2((double)T);
    printf("%dx%d  2d-il  v(ip)=%10.1f  f(ip)=%10.1f  f(oop)=%10.1f  ratio(ip)=%5.2f ratio(oop)=%5.2f  "
           "vGFLOPS=%5.2f fGFLOPS=%5.2f  vgate=%.2e fgate=%.2e %s\n",
           N1, N2, vns, fns, fons, fns / vns, fons / vns, gf / vns, gf / fns, vgate, fgate,
           (vgate < 1e-10 && fgate < 1e-10) ? "OK" : "*** GATE FAIL ***");

    vfft_destroy(vp); fx.destroy_plan(fpi); fx.destroy_plan(fbi); fx.destroy_plan(fpo);
    fx.ffree(fz); fx.ffree(fo); _aligned_free(f0); _aligned_free(vz); _aligned_free(v0);
    return (vgate < 1e-10 && fgate < 1e-10) ? 0 : 1;
}
