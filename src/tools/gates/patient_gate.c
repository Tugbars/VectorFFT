/* patient_gate.c — EVERY CREATE PLANS PATIENT (include/vfft.h, the VFFT_FLAG_* word;
 * docs/roadmap/api_v2_design.md, 2026-10-17).
 *
 * One small cell, 1D C2C 16, through the public API. A create MEASURED when the diagnostics
 * header's scope count (vfft_measure_scopes) moved across it: a create served from wisdom reads no
 * clock and enters no scope.
 *   E  VFFT_FLAG_EXHAUSTIVE is refused: NULL, vfft_error() names the flag, nothing measured, the
 *      store's files unchanged; first in the process, on the cold store, where an accepted flag
 *      would race the cell and save it;
 *   U  the first bit no flag names is refused the same way;
 *   R  a create without VFFT_FLAG_PATIENT plans PATIENT: two creates of the cell flagged
 *      VFFT_FLAG_RECALIBRATE alone both measure in this process (planned MEASURE, the second would
 *      believe the planner's cached costs and measure nothing);
 *   P  VFFT_FLAG_PATIENT is accepted (vfft_error() empty) and serves the zeroed config's route;
 *   F  every served plan's forward against the naive DFT.
 * Only the R pair measures. Usage: patient_gate --wisdir <cold scratch dir> */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#define CELL_N 16   /* the cell every create asks for: 1D C2C, the zeroed config */

static int g_checks = 0, g_fail = 0;
static const char *g_W = NULL;
#define CHECK(c, ...) do { g_checks++; if (c) printf("  ok  " __VA_ARGS__); else { g_fail++; printf("  *** FAIL: " __VA_ARGS__); } printf("\n"); } while (0)

/* the cell's create with these flags on the gate's store; *measured: it entered a scope */
static vfft_plan create(unsigned flags, int *measured)
{
    vfft_config_t cfg;
    vfft_plan p;
    long s0;
    memset(&cfg, 0, sizeof cfg);
    cfg.n[0] = CELL_N;
    cfg.flags = flags;
    cfg.wisdom_dir = g_W;
    s0 = vfft_measure_scopes();
    p = vfft_create(&cfg);
    *measured = vfft_measure_scopes() != s0;
    return p;
}

/* FNV-1a over a file's name, size and last write time */
static unsigned long long file_key(const char *name, unsigned long long size, unsigned long long wtime)
{
    unsigned long long h = 1469598103934665603ull;
    for (; *name; name++)
        h = (h ^ (unsigned char)*name) * 1099511628211ull;
    h = (h ^ size) * 1099511628211ull;
    return (h ^ wtime) * 1099511628211ull;
}
/* the store directory's files: their count, and the sum of their keys (the listing's order does
 * not matter); a save adds a file or moves a file's size or write time */
static unsigned long long store_digest(int *files)
{
    unsigned long long d = 0;
    *files = 0;
#ifdef _WIN32
    {
        char pat[1024];
        WIN32_FIND_DATAA f;
        HANDLE h;
        snprintf(pat, sizeof pat, "%s/*", g_W);
        h = FindFirstFileA(pat, &f);
        if (h == INVALID_HANDLE_VALUE)
            return 0;
        do
            if (!(f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                ++*files;
                d += file_key(f.cFileName, ((unsigned long long)f.nFileSizeHigh << 32) | f.nFileSizeLow,
                              ((unsigned long long)f.ftLastWriteTime.dwHighDateTime << 32) | f.ftLastWriteTime.dwLowDateTime);
            }
        while (FindNextFileA(h, &f));
        FindClose(h);
    }
#else
    {
        DIR *dp = opendir(g_W);
        struct dirent *e;
        struct stat st;
        char path[1024];
        if (!dp)
            return 0;
        while ((e = readdir(dp)) != NULL)
        {
            snprintf(path, sizeof path, "%s/%s", g_W, e->d_name);
            if (stat(path, &st) == 0 && S_ISREG(st.st_mode))
            {
                ++*files;
                d += file_key(e->d_name, (unsigned long long)st.st_size, (unsigned long long)st.st_mtime);
            }
        }
        closedir(dp);
    }
#endif
    return d;
}

/* E and U: NULL with the reason named, nothing measured, the store's files as they were */
static void refused(const char *tag, unsigned flags, const char *want)
{
    int measured, f0, f1;
    const unsigned long long d0 = store_digest(&f0);
    vfft_plan p = create(flags, &measured);
    const unsigned long long d1 = store_digest(&f1);
    const int same = (f1 == f0 && d1 == d0);
    CHECK(!p && strstr(vfft_error(), want) && !measured && same,
          "%s flags 0x%02x: %s, vfft_error() \"%s\", %s, the store's %d file(s) %s", tag, flags,
          p ? "SERVED" : "refused", vfft_error(), measured ? "MEASURED" : "nothing measured", f0,
          same ? "unchanged" : "CHANGED");
    vfft_destroy(p);
}

/* F: the plan's forward of a fixed input against the naive DFT, the largest error against the
 * largest reference magnitude */
static void forward(const char *label, vfft_plan p)
{
    double *x = (double *)vfft_malloc(2 * CELL_N * sizeof(double));
    double *y = (double *)vfft_malloc(2 * CELL_N * sizeof(double));
    double m = 0, e = 0;
    unsigned s = 12345u;
    int k, n;
    for (k = 0; k < 2 * CELL_N; k++)
    {
        s = s * 1664525u + 1013904223u;
        x[k] = (double)(s >> 8) / 16777216.0 - 0.5;
    }
    vfft_execute(p, VFFT_FORWARD, x, y);
    for (k = 0; k < CELL_N; k++)
    {
        double sr = 0, si = 0;
        for (n = 0; n < CELL_N; n++)
        {
            const double a = -2.0 * 3.14159265358979323846 * (double)((k * n) % CELL_N) / CELL_N;
            sr += x[2 * n] * cos(a) - x[2 * n + 1] * sin(a);
            si += x[2 * n] * sin(a) + x[2 * n + 1] * cos(a);
        }
        m = fmax(m, fmax(fabs(sr), fabs(si)));
        e = fmax(e, fmax(fabs(y[2 * k] - sr), fabs(y[2 * k + 1] - si)));
    }
    CHECK(e <= 1e-12 * m, "F  %s: forward against the naive DFT, max error %.1e of %.1e", label, e, m);
    vfft_free(x);
    vfft_free(y);
}

int main(int argc, char **argv)
{
    const unsigned unnamed = (unsigned)VFFT_FLAG_ESTIMATE << 1;   /* the first bit no flag names */
    char want[64], r0[64], r1[64], err[256];
    vfft_plan p, p0, p1;
    int a, i, measured, m0, m1;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir"))
            g_W = argv[a + 1];
    if (!g_W)
    {
        printf("usage: patient_gate --wisdir <cold scratch dir>\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("PATIENT GATE: every create plans PATIENT, 1D C2C %d, store %s\n", CELL_N, g_W);

    /* E, U: first in the process, on the cold store */
    refused("E  VFFT_FLAG_EXHAUSTIVE,", VFFT_FLAG_EXHAUSTIVE, "VFFT_FLAG_EXHAUSTIVE");
    snprintf(want, sizeof want, "unknown flag bits 0x%x", unnamed);
    refused("U  an unnamed bit,", unnamed, want);

    /* R: VFFT_FLAG_RECALIBRATE alone, twice in this process */
    for (i = 1; i <= 2; i++)
    {
        p = create(VFFT_FLAG_RECALIBRATE, &measured);
        CHECK(p && measured, "R  VFFT_FLAG_RECALIBRATE alone, create %d of 2 in this process: %s%s", i,
              !p ? "refused: " : measured ? "measured" : "NOT measured", !p ? vfft_error() : "");
        if (p)
            forward(i == 1 ? "the first RECALIBRATE" : "the second RECALIBRATE", p);
        vfft_destroy(p);
    }

    /* P: the zeroed config, then VFFT_FLAG_PATIENT */
    p0 = create(0, &m0);
    p1 = create(VFFT_FLAG_PATIENT, &m1);
    snprintf(err, sizeof err, "%s", vfft_error());
    snprintf(r0, sizeof r0, "%s", p0 ? vfft_plan_route(p0) : "-");
    snprintf(r1, sizeof r1, "%s", p1 ? vfft_plan_route(p1) : "-");
    CHECK(p0 && p1 && !err[0] && !strcmp(r0, r1),
          "P  VFFT_FLAG_PATIENT: %s (%s), vfft_error() \"%s\", route %s; the zeroed config: %s (%s), route %s",
          p1 ? "served" : "REFUSED", m1 ? "measured" : "from wisdom", err, r1,
          p0 ? "served" : "REFUSED", m0 ? "measured" : "from wisdom", r0);
    if (p0)
        forward("the zeroed config", p0);
    if (p1)
        forward("VFFT_FLAG_PATIENT", p1);
    vfft_destroy(p0);
    vfft_destroy(p1);

    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** PATIENT GATE FAILED ***" : "PATIENT GATE ALL PASS");
    return g_fail ? 1 : 0;
}
