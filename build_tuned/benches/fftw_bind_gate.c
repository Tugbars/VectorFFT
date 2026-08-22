/* fftw_bind_gate.c — PERMANENT GATE (phase P0 of docs/roadmap/fftw_bench_design.md).
 *
 * Proves, on this host, before any FFTW number is ever published:
 *   G1  genuine FFTW binds at runtime from an absolute DLL path — NOT MKL's
 *       92-symbol fftw_* wrapper layer (version must read "fftw-…", never
 *       "FFTW 3.3.4 wrappers to Intel oneMKL");
 *   G2  fftw_alignment_of works through the shim (banner data);
 *   G3  interleaved c2c N=512, FFTW_MEASURE, forward gates ELEMENTWISE vs a
 *       naive DFT (never a roundtrip) — also pins the sign convention:
 *       FFTW_FORWARD=-1 == e^{-i2πkn/N} == MKL ComputeForward;
 *   G4  guru split c2c N=1000 IN-PLACE, MEASURE — THE historical catastrophe
 *       shape (ESTIMATE gave errors of 60+, sometimes 1e+299; MEASURE gates at
 *       ~2e-11). Planes come from ref_planes_alloc (deterministic delta).
 *       Ordering law demonstrated in code: alloc -> PLAN -> fill -> execute,
 *       because MEASURE overwrites the arrays while planning;
 *   G5  wisdom roundtrip through the shim: export -> forget -> import ->
 *       WISDOM_ONLY replan on a FRESH deterministic plane pair -> plan found
 *       (non-NULL) with the IDENTICAL plan_id. This is the in-process half of
 *       P0.75 (cross-process/reboot is the smoke test's job).
 *
 * Build+run (NO --fftw and NO --mkl — the whole point is nothing on the link line):
 *   python build_tuned/build.py --src benches/fftw_bind_gate.c
 * Override the DLL under test: VFFT_FFTW_DLL=<path>.
 * Untimed except plan-cost banners; this gate is correctness machinery, not a bench.
 */
#include "ref_fftw.h"

#include <math.h>

#ifdef _WIN32
#  include <windows.h>
static double now_ms(void)
{
    static double inv = 0.0; LARGE_INTEGER c;
    if (inv == 0.0) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); inv = 1000.0 / (double)f.QuadPart; }
    QueryPerformanceCounter(&c); return (double)c.QuadPart * inv;
}
#else
#  include <time.h>
static double now_ms(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#endif

static int g_fail = 0;
#define CHECK(name, cond, ...) do {                                   \
    if (cond) printf("BINDGATE PASS  %-34s ", name);                  \
    else      { printf("BINDGATE FAIL  %-34s ", name); g_fail++; }    \
    printf(__VA_ARGS__); putchar('\n'); } while (0)

/* deterministic fill — xorshift64, values in [-1,1) */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double frand(void)
{ g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
  return (double)(int64_t)(g_rng >> 11) / 4503599627370496.0; }

/* naive forward DFT of split data: X[k] = sum x[n]·e^{-i2πkn/N} */
static void naive_split_fwd(int n, const double *ri, const double *ii,
                            double *xr, double *xi)
{
    for (int k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < n; j++) {
            double th = -2.0 * 3.14159265358979323846 * (double)k * (double)j / (double)n;
            double c = cos(th), s = sin(th);
            sr += ri[j] * c - ii[j] * s;
            si += ri[j] * s + ii[j] * c;
        }
        xr[k] = sr; xi[k] = si;
    }
}

static double maxrel_split(int n, const double *ar, const double *ai,
                           const double *br, const double *bi)
{
    double mag = 1e-300, err = 0.0;
    for (int k = 0; k < n; k++) {
        double m = fabs(br[k]) + fabs(bi[k]); if (m > mag) mag = m;
        double e = fabs(ar[k] - br[k]) + fabs(ai[k] - bi[k]); if (e > err) err = e;
    }
    return err / mag;
}

int main(void)
{
    fftwx_api_t api; char err[640];

    printf("BINDGATE fftw_bind_gate — P0 of docs/roadmap/fftw_bench_design.md\n");

    /* ---------------------------------------------------------------- G1 */
    if (!fftwx_bind(&api, err, sizeof err)) {
        printf("BINDGATE FAIL  bind                               %s\n", err);
        printf("BINDGATE VERDICT FAIL (1 check, bind is the gatekeeper)\n");
        return 1;
    }
    CHECK("bind+genuine", 1, "dll=%s", api.dll_path);
    printf("BINDGATE INFO  fftw_version=\"%s\"\n", api.version);

    /* ---------------------------------------------------------------- G2 */
    {
        ref_planes_t p = ref_planes_alloc(64);
        if (!p.blk) { CHECK("alignment_of", 0, "planes alloc failed"); }
        else {
            int a0 = api.alignment_of(p.re), a1 = api.alignment_of(p.im);
            int a2 = api.alignment_of((double *)((char *)p.re + 8));
            CHECK("alignment_of", 1,
                  "re=%d im=%d re+8B=%d (equal classes for re/im = wisdom-key precondition)",
                  a0, a1, a2);
            if (a0 != a1) { g_fail++; printf("BINDGATE FAIL  plane-class-equal          re/im alignment classes differ\n"); }
            ref_planes_free(&p);
        }
    }

    /* ------------------------------------------- G3: interleaved c2c N=512 */
    {
        enum { N = 512 };
        fftwx_complex *in  = (fftwx_complex *)api.fmalloc(sizeof(fftwx_complex) * N);
        fftwx_complex *out = (fftwx_complex *)api.fmalloc(sizeof(fftwx_complex) * N);
        double t0 = now_ms();
        fftwx_plan pl = api.plan_dft_1d(N, in, out, FFTWX_FORWARD, FFTWX_MEASURE);
        double t1 = now_ms();
        if (!pl) CHECK("c2c512-il-measure", 0, "planner returned NULL");
        else {
            /* fill AFTER planning — MEASURE just scribbled on both arrays */
            static double ri[N], ii[N], xr[N], xi[N], fr[N], fi[N];
            for (int i = 0; i < N; i++) { ri[i] = frand(); ii[i] = frand();
                                          in[i][0] = ri[i]; in[i][1] = ii[i]; }
            api.execute(pl);
            for (int i = 0; i < N; i++) { fr[i] = out[i][0]; fi[i] = out[i][1]; }
            naive_split_fwd(N, ri, ii, xr, xi);
            double e = maxrel_split(N, fr, fi, xr, xi);
            CHECK("c2c512-il-measure", e < 1e-8,
                  "maxrel=%.3e vs naive fwd (sign: FFTW_FORWARD=-1 == e^{-i2pi kn/N}) plan_ms=%.1f",
                  e, t1 - t0);
            api.destroy_plan(pl);
        }
        api.ffree(in); api.ffree(out);
    }

    /* ------------------- G4: guru split N=1000 in-place — THE catastrophe shape */
    uint64_t id_cold = 0;
    {
        enum { N = 1000 };
        ref_planes_t p = ref_planes_alloc(N);
        fftwx_iodim d = { N, 1, 1 };
        double t0 = now_ms();
        fftwx_plan pl = api.plan_guru_split_dft(1, &d, 0, NULL,
                                                p.re, p.im, p.re, p.im, FFTWX_MEASURE);
        double t1 = now_ms();
        if (!pl) CHECK("split1000-ip-measure", 0, "planner returned NULL");
        else {
            static double ri[N], ii[N], xr[N], xi[N];
            for (int i = 0; i < N; i++) { ri[i] = frand(); ii[i] = frand();
                                          p.re[i] = ri[i]; p.im[i] = ii[i]; }
            api.execute(pl);
            naive_split_fwd(N, ri, ii, xr, xi);
            double e = maxrel_split(N, p.re, p.im, xr, xi);
            CHECK("split1000-ip-measure", e < 1e-8,
                  "maxrel=%.3e (ESTIMATE here gave 60..1e+299; MEASURE law holds) plan_ms=%.1f delta=%llu",
                  e, t1 - t0, (unsigned long long)p.stride);
            id_cold = fftwx_plan_id(&api, pl);
            printf("BINDGATE INFO  split1000 plan_id=%016llx (cold)\n",
                   (unsigned long long)id_cold);
            api.destroy_plan(pl);
        }

        /* ------------------------------------------------ G5: wisdom roundtrip */
        const char *wis = "build_tuned/benches/_fftw_bind_gate.wis";
        int ex = api.export_wisdom_to_filename(wis);
        api.forget_wisdom();
        int im = api.import_wisdom_from_filename(wis);
        ref_planes_t q = ref_planes_alloc(N);       /* FRESH pair, same deterministic delta */
        double t2 = now_ms();
        fftwx_plan pw = api.plan_guru_split_dft(1, &d, 0, NULL,
                                                q.re, q.im, q.re, q.im,
                                                FFTWX_MEASURE | FFTWX_WISDOM_ONLY);
        double t3 = now_ms();
        if (!pw)
            CHECK("wisdom-roundtrip", 0,
                  "WISDOM_ONLY replan on fresh deterministic planes MISSED (export=%d import=%d) "
                  "— the 9.4%% plan-drift disease would be live", ex, im);
        else {
            uint64_t id_warm = fftwx_plan_id(&api, pw);
            CHECK("wisdom-roundtrip", id_warm == id_cold,
                  "export=%d import=%d WISDOM_ONLY hit, plan_id=%016llx %s cold, replan_ms=%.1f",
                  ex, im, (unsigned long long)id_warm,
                  id_warm == id_cold ? "==" : "!=", t3 - t2);
            api.destroy_plan(pw);
        }
        ref_planes_free(&q);
        ref_planes_free(&p);
    }

    printf("BINDGATE VERDICT %s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
