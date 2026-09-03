/* zen4_r2c_il_race.c — K=1 INTERLEAVED r2c: front door vs genuine FFTW.
 *
 * OUR ARM  vfft_create(R2C, OUT-OF-PLACE, layout=INTERLEAVED, howmany=1) at
 *          VFFT_PATIENT with wisdom_write=1: a store MISS races the real route
 *          (measurement_arms.md R1.3, child_oop_il vs child_nat_ip) and BANKS
 *          it into $VFFT_WISDOM_DIR; a hit replays. N reals in -> N/2+1 CCE
 *          pairs out (vfft.h signature table: sre=real_in, dre=z_CCE_out).
 * FFTW ARM plan_dft_r2c_1d(N, x, z, PATIENT) through ref_fftw.h — bound at
 *          runtime from $VFFT_FFTW_DLL, genuineness asserted (never MKL's
 *          wrappers). Same contract: N reals -> N/2+1 interleaved bins.
 * PROTOCOL the canonical bench shape: warmup 10, best-of-5, reps=2e6/(N+1),
 *          cachebust + cool_ms between engines, flip = FFTW first, core pin.
 *          Correctness for BOTH arms is ELEMENTWISE vs a naive real DFT (never
 *          a roundtrip). FFTW's PATIENT planning scribbles its arrays, so the
 *          input is filled AFTER planning (the ordering law from the gate).
 *
 * Build: python build.py --src benches/zen4_r2c_il_race.c --vfft --compile
 * Run  : zen4_r2c_il_race.exe [N] [cool_ms] [flip] [core]     (env above) */
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
static int reps_for(int N) { int r = (int)(2e6 / (N + 1)); return r < 8 ? 8 : r > 100000 ? 100000 : r; }
static void pace(int ms) { if (ms > 0) Sleep((DWORD)ms); }
static void cachebust(void)
{
    size_t s = 32u * 1024 * 1024 / sizeof(double);
    double *j = _aligned_malloc(s * sizeof *j, 64); volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; _aligned_free(j);
}
static void fill(double *x, int N) { srand(42u + N); for (int i = 0; i < N; i++) x[i] = (double)rand() / RAND_MAX - 0.5; }

/* max |Z[k] - naive(x)[k]| / max|naive| over the N/2+1 CCE bins */
static double gate(const double *x, const double *z, int N)
{
    double me = 0, mm = 0;
    for (int k = 0; k <= N / 2; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)k * (double)n / (double)N;
            re += x[n] * cos(a); im += x[n] * sin(a);
        }
        double er = z[2 * k] - re, ei = z[2 * k + 1] - im;
        double e = sqrt(er * er + ei * ei), m = sqrt(re * re + im * im);
        if (e > me) me = e;
        if (m > mm) mm = m;
    }
    return mm > 0 ? me / mm : me;
}

#define TIME(best, CALL) do { int _r = reps_for(N); (best) = 1e18;            \
    for (int _w = 0; _w < 10; _w++) { CALL; }                                 \
    for (int _t = 0; _t < 5; _t++) { double _t0 = now_ns();                   \
        for (int _i = 0; _i < _r; _i++) { CALL; }                              \
        double _ns = (now_ns() - _t0) / _r; if (_ns < (best)) (best) = _ns; } } while (0)

int main(int argc, char **argv)
{
    int N       = argc > 1 ? atoi(argv[1]) : 4096;
    int cool_ms = argc > 2 ? atoi(argv[2]) : 200;
    int flip    = argc > 3 ? atoi(argv[3]) : 0;
    int core    = argc > 4 ? atoi(argv[4]) : 2;
    const size_t xs = (size_t)N + 2;                 /* 2*(N/2+1) doubles */
    setvbuf(stdout, NULL, _IONBF, 0);
    if (core >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    /* ---- FFTW: bind, plan (PATIENT), gate --------------------------------- */
    fftwx_api_t fx; char err[512];
    if (!fftwx_bind(&fx, err, sizeof err)) { fprintf(stderr, "FFTW bind failed: %s\n", err); return 2; }
    const char *wis = getenv("VFFT_FFTW_WIS");
    int wis_loaded = wis && fx.import_wisdom_from_filename(wis);
    double *fx_x = fx.fmalloc(sizeof(double) * xs), *fx_z = fx.fmalloc(sizeof(double) * xs);
    double t0 = now_ns();
    fftwx_plan fp = fx.plan_dft_r2c_1d(N, fx_x, (fftwx_complex *)fx_z, FFTWX_PATIENT);
    double fplan_ms = (now_ns() - t0) / 1e6;
    if (wis) fx.export_wisdom_to_filename(wis);
    fill(fx_x, N);                                    /* AFTER planning */
    fx.execute(fp);
    double fgate = gate(fx_x, fx_z, N);

    /* ---- ours: front door, PATIENT, banks on a miss ----------------------- */
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.rigor = VFFT_PATIENT; cfg.nthreads = 1; cfg.wisdom_write = 1;
    double *vx = _aligned_malloc(sizeof(double) * xs, 64), *vz = _aligned_malloc(sizeof(double) * xs, 64);
    t0 = now_ns();
    vfft_plan vp = vfft_create(&cfg);
    double vplan_ms = (now_ns() - t0) / 1e6;
    if (!vp) { fprintf(stderr, "vfft_create REFUSED\n"); return 2; }
    fill(vx, N);
    vfft_execute(vp, VFFT_FORWARD, vx, NULL, vz, NULL);
    double vgate = gate(vx, vz, N);

    printf("zen4_r2c_il_race  N=%d K=1 r2c INTERLEAVED OOP  reps=%d x best-of-5  core=%d cool=%dms flip=%d\n"
           "  vfft %s isa=%s wisdom_dir=%s  (create %.0f ms)\n  %s  dll=%s  (plan %.0f ms, wisdom %s)\n",
           N, reps_for(N), core, cool_ms, flip, vfft_version(), vfft_isa(),
           getenv("VFFT_WISDOM_DIR") ? getenv("VFFT_WISDOM_DIR") : "(UNSET)", vplan_ms,
           fx.version, fx.dll_path, fplan_ms, wis ? (wis_loaded ? "loaded" : "cold") : "none");

    /* ---- A/B, order-neutralised ------------------------------------------ */
    /* CLOCK RAMP. On a 35 W laptop part the first timed arm ran 15-20% slower
     * than the second in BOTH flip orders (measured 2026-09-03: 3356/3164 vs
     * 2845/3775): ten warm-ups are not enough for the core to reach its
     * boost clock, so the first arm was measured on a cold clock and the
     * second on a warm one. Spin ~400 ms of real work before the first arm
     * so both start warm; cachebust + cool between them stays as the bench
     * prescribes. Pure arithmetic, no engine executed. */
    { volatile double acc = 1.0; double t1 = now_ns();
      while (now_ns() - t1 < 400e6) { for (int i = 0; i < 4096; i++) acc = acc * 1.0000001 + 1e-9; } (void)acc; }
    double vns = 0, fns = 0;
    if (flip) {
        TIME(fns, fx.execute(fp)); cachebust(); pace(cool_ms);
        TIME(vns, vfft_execute(vp, VFFT_FORWARD, vx, NULL, vz, NULL));
    } else {
        TIME(vns, vfft_execute(vp, VFFT_FORWARD, vx, NULL, vz, NULL)); cachebust(); pace(cool_ms);
        TIME(fns, fx.execute(fp));
    }
    double gf = 2.5 * N * log2((double)N);            /* real-input op count, half of c2c */
    printf("%-6d r2c-il  v=%9.1f  f=%9.1f  ratio=%5.2f  vGFLOPS=%5.2f fGFLOPS=%5.2f  vgate=%.2e fgate=%.2e %s\n",
           N, vns, fns, vns > 0 ? fns / vns : 0, gf / vns, gf / fns, vgate, fgate,
           (vgate < 1e-10 && fgate < 1e-10) ? "OK" : "*** GATE FAIL ***");

    vfft_destroy(vp); fx.destroy_plan(fp);
    fx.ffree(fx_x); fx.ffree(fx_z); _aligned_free(vx); _aligned_free(vz);
    return (vgate < 1e-10 && fgate < 1e-10) ? 0 : 1;
}
