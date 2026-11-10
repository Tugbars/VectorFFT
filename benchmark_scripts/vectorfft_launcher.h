/* vectorfft_launcher.h -- the vectorfft_vs_* programs as the user starts them (benchmark_scripts/CMakeLists.txt).
 *
 * A program here is built for plain x86-64 (no AVX), so it runs on any x86-64 CPU and can say what is missing
 * when the CPU is too old, instead of dying on an illegal instruction. It asks the CPU and the operating system
 * what they run (CPUID and XGETBV): AVX2 with FMA at least (Intel from Haswell, 2013; AMD from Zen, 2017), and
 * AVX-512 (F and DQ) for the AVX-512 build. It picks AVX-512 when the CPU runs it and the build holds that
 * worker, else AVX2; --avx2 / --avx512 force one. Then it runs that instruction set's worker
 * (vectorfft_bench_worker: vectorfft_bench.h) with the program's name, comparator, plans, precision and
 * transforms, and returns its exit code. */
#ifndef VECTORFFT_LAUNCHER_H
#define VECTORFFT_LAUNCHER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

/* the programs' transforms: 1D N, 2D N1xN2 (N1 rows of N2 points), 3D N1xN2xN3. The selection: commonly used
 * transforms VectorFFT wins clearly (timed on the i9-14900KF 2026-11-10, gauntlet/results/sortie_*_est_20261110;
 * 1080x1920 = a 1920 x 1080 frame) */
#define VL_CELLS_SELECTED \
    "1920,2048,2560,3072,3840,1048576,4194304," \
    "512x512,1024x1024,2048x1024,2048x2048,1080x1920,720x1280," \
    "128x128x128,256x128x128,256x256x64,64x256x256,256x256x256"
#define VL_CELLS_POW2 \
    "64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304"

typedef struct {
    const char *program;   /* its name (the target's) */
    const char *title;     /* one line: what it times */
    const char *vs;        /* the comparator: "mkl" | "fftw" */
    const char *plans;     /* "estimate" | "planned" */
    const char *prec;      /* "f64" | "f32" */
    const char *cells;     /* its transforms (--cells replaces them) */
} vl_program_t;

/* what the CPU and the operating system run: *avx2 = AVX2 and FMA with the YMM state saved by the OS, *avx512 =
 * also AVX-512 F and DQ with the opmask and ZMM state saved */
static void vl_cpu(int *avx2, int *avx512)
{
    unsigned a = 0, b = 0, c = 0, d = 0, lo = 0;
    *avx2 = *avx512 = 0;
#if defined(_MSC_VER) && !defined(__clang__)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7)
        return;
    __cpuid(r, 1);
    c = (unsigned)r[2];
#else
    if (__get_cpuid_max(0, NULL) < 7 || !__get_cpuid(1, &a, &b, &c, &d))
        return;
#endif
    const int fma = (c >> 12) & 1, osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
    if (!fma || !osxsave || !avx)
        return;
#if defined(_MSC_VER) && !defined(__clang__)
    lo = (unsigned)_xgetbv(0);
    __cpuidex(r, 7, 0);
    b = (unsigned)r[1];
#else
    unsigned hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    (void)hi;
    __cpuid_count(7, 0, a, b, c, d);
#endif
    if ((lo & 0x6u) != 0x6u)   /* XMM and YMM state */
        return;
    *avx2 = (b >> 5) & 1;
    *avx512 = *avx2 && ((b >> 16) & 1) && ((b >> 17) & 1) && (lo & 0xE0u) == 0xE0u;   /* F, DQ; opmask, ZMM */
}

static int vl_exists(const char *p)
{
#ifdef _WIN32
    return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
#else
    return access(p, X_OK) == 0;
#endif
}

/* the worker of one instruction set: in a release package beside this program, <here>/<isa>/
 * (benchmark_scripts/scripts/package_release.py), else where the build put it (compiled, may be NULL) */
static const char *vl_worker(const char *isa, const char *compiled, char *buf, size_t n)
{
    char dir[1200];
#ifdef _WIN32
    const DWORD l = GetModuleFileNameA(NULL, dir, (DWORD)sizeof dir);
    const int ok = l > 0 && l < sizeof dir;
#else
    const ssize_t l = readlink("/proc/self/exe", dir, sizeof dir - 1);
    const int ok = l > 0;
    if (ok)
        dir[l] = 0;
#endif
    if (ok)
    {
        char *s = strrchr(dir, '/'), *b = strrchr(dir, '\\');
        if (b && (!s || b > s))
            s = b;
        if (s)
        {
            *s = 0;
#ifdef _WIN32
            snprintf(buf, n, "%s\\%s\\vectorfft_bench_worker.exe", dir, isa);
#else
            snprintf(buf, n, "%s/%s/vectorfft_bench_worker", dir, isa);
#endif
            if (vl_exists(buf))
                return buf;
        }
    }
    return compiled && vl_exists(compiled) ? compiled : NULL;
}

/* run argv[0] with argv on this program's console and streams; its exit code (-1: it could not be started) */
static int vl_run(char *const argv[])
{
#ifdef _WIN32
    static char cmd[32768];
    size_t l = 0;
    for (int i = 0; argv[i]; i++)
    {   /* each argument quoted; a quote inside escaped with the backslashes before it doubled, and the
         * backslashes before the closing quote doubled (the Windows command-line rules) */
        size_t bs = 0;
        if (i && l < sizeof cmd - 2)
            cmd[l++] = ' ';
        if (l < sizeof cmd - 2)
            cmd[l++] = '"';
        for (const char *s = argv[i]; *s && l < sizeof cmd - 8; s++)
        {
            if (*s == '\\')
                bs++;
            else if (*s == '"')
            {
                for (size_t q = 0; q <= bs && l < sizeof cmd - 8; q++)
                    cmd[l++] = '\\';
                bs = 0;
            }
            else
                bs = 0;
            cmd[l++] = *s;
        }
        for (size_t q = 0; q < bs && l < sizeof cmd - 4; q++)
            cmd[l++] = '\\';
        if (l < sizeof cmd - 2)
            cmd[l++] = '"';
    }
    cmd[l] = 0;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi))
        return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
#else
    execv(argv[0], argv);   /* returns only when it could not be started */
    return -1;
#endif
}

static int vl_usage(const vl_program_t *p, int rc)
{
    printf("%s -- %s\n\n"
           "  %s [--cells LIST] [--out DIR] [--avx2 | --avx512] [--check-only]\n\n"
           "  --cells LIST       your own transforms instead of this program's, separated by commas: N (1D),\n"
           "                     N1xN2 (2D, N1 rows of N2 points), N1xN2xN3 (3D)%s\n"
           "  --out DIR          where the run directory is made (default: vectorfft_results in the current directory)\n"
           "  --avx2, --avx512   the instruction set (default: AVX-512 when the CPU runs it, else AVX2)\n"
           "  --check-only       plan and check the transforms and show the bench's commands, time nothing\n\n"
           "Its transforms: %s\n",
           p->program, p->title, p->program,
           !strcmp(p->prec, "f32") ? "; single precision plans 1D powers of two up to 16777216" : "", p->cells);
    return rc;
}

static int vl_main(int argc, char **argv, const vl_program_t *p)
{
    const char *cells = p->cells, *out = NULL;
    int force = 0, check = 0;
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        if (!strcmp(a, "--cells") && i + 1 < argc)
            cells = argv[++i];
        else if (!strcmp(a, "--out") && i + 1 < argc)
            out = argv[++i];
        else if (!strcmp(a, "--avx2"))
            force = 2;
        else if (!strcmp(a, "--avx512"))
            force = 512;
        else if (!strcmp(a, "--check-only"))
            check = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h") || !strcmp(a, "/?"))
            return vl_usage(p, 0);
        else
        {
            printf("%s: unknown option '%s'\n\n", p->program, a);
            return vl_usage(p, 2);
        }
    }

    int c2 = 0, c512 = 0;
    vl_cpu(&c2, &c512);
    if (!c2)
    {
        printf("%s: this CPU (or its operating system) does not run AVX2 with FMA. VectorFFT needs an x86-64 CPU\n"
               "with AVX2 and FMA: Intel from Haswell (2013), AMD from Zen (2017), or later.\n", p->program);
        return 1;
    }
    if (force == 512 && !c512)
    {
        printf("%s: --avx512: this CPU (or its operating system) does not run AVX-512 (F and DQ)\n", p->program);
        return 1;
    }
    static char b2[1400], b512[1400];
#ifdef VL_WORKER_AVX2
    const char *w2 = vl_worker("avx2", VL_WORKER_AVX2, b2, sizeof b2);
#else
    const char *w2 = vl_worker("avx2", NULL, b2, sizeof b2);
#endif
#ifdef VL_WORKER_AVX512
    const char *w512 = vl_worker("avx512", VL_WORKER_AVX512, b512, sizeof b512);
#else
    const char *w512 = vl_worker("avx512", NULL, b512, sizeof b512);
#endif
    const int use512 = force == 512 || (!force && c512 && w512);
    const char *w = use512 ? w512 : w2;
    if (!w)
    {
        printf("%s: this build holds no %s version of the benchmark: configure it with -DVFFT_BENCH_BOTH_ISAS=ON\n"
               "to build both instruction sets, or with -DVFFT_ISA=%s, then build this program again\n",
               p->program, use512 ? "AVX-512" : "AVX2", use512 ? "avx512" : "avx2");
        return 1;
    }
    if (!force && c512 && !w512)
        printf("%s: this CPU runs AVX-512, but this build holds the AVX2 version only: running AVX2\n"
               "(configure with -DVFFT_BENCH_BOTH_ISAS=ON to build both)\n\n", p->program);

    char *wargv[24];
    int k = 0;
    wargv[k++] = (char *)w;
    wargv[k++] = (char *)"--program";
    wargv[k++] = (char *)p->program;
    wargv[k++] = (char *)"--title";
    wargv[k++] = (char *)p->title;
    wargv[k++] = (char *)"--vs";
    wargv[k++] = (char *)p->vs;
    wargv[k++] = (char *)"--plans";
    wargv[k++] = (char *)p->plans;
    wargv[k++] = (char *)"--prec";
    wargv[k++] = (char *)p->prec;
    wargv[k++] = (char *)"--cells";
    wargv[k++] = (char *)cells;
    if (out)
    {
        wargv[k++] = (char *)"--out";
        wargv[k++] = (char *)out;
    }
    if (check)
        wargv[k++] = (char *)"--check-only";
    wargv[k] = NULL;
    fflush(stdout);
    const int rc = vl_run(wargv);
    if (rc == -1)
        printf("%s: cannot start %s\n", p->program, w);
    return rc;
}

#endif /* VECTORFFT_LAUNCHER_H */
