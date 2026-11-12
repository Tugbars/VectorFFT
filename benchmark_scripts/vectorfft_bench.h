/* vectorfft_bench.h -- the benchmark worker: what the vectorfft_vs_* programs run (benchmark_scripts/CMakeLists.txt).
 *
 * A vectorfft_vs_* program is a launcher (vectorfft_launcher.h, built for plain x86-64): it checks the CPU and
 * runs the worker of the instruction set it picks, vectorfft_bench_worker (vectorfft_bench_worker.c, one build
 * per instruction set), with the program's name, its comparator, its plans, its precision and its list of
 * transforms, its cells: 1D N, 2D N1xN2 (N1 rows of N2 points) or 3D N1xN2xN3; complex to complex, interleaved,
 * natural order, out of place (--inplace: in place, one buffer on both engines: the bench's --k1nat 1D cell, its
 * 2D / 3D cells with --ilip), one transform per call, one thread.
 *
 * The worker times them against Intel MKL or FFTW with the gauntlet's own tools, as gauntlet/gauntlet.py runs
 * them: one bench_1d_vs_mkl process per cell and engine order (the bench pins its core, raises its priority and
 * paces its timing), a control cell before and after each dimension's cells, the bench's own csv. FFTW is the
 * bench's run-time comparator (--cmp fftw: the repository's gauntlet/fftw/libfftw3-3.dll on Windows, the system's
 * libfftw3 elsewhere, FFTW_MEASURE plans kept in <out>/fftw.wis). Nothing is timed in the worker itself.
 *
 * The bench creates its plans from a wisdom store. Two sources of plans:
 *
 *   ESTIMATE (--plans estimate; double precision): the library's VFFT_FLAG_ESTIMATE plans (planned from
 *   formulas, nothing measured). The worker writes them as rows into a new store in the run directory, with the
 *   calls the estimate's front door makes for one thread (src/core/il/estimate/c2c_estimate_il.h,
 *   c2c_estimate_nd_il.h). Before anything is timed every cell is checked: the store's plan must be served
 *   without a measurement, and its output must equal, bit for bit, the output of a plan created with
 *   VFFT_FLAG_ESTIMATE. So the bench times exactly the estimate's plan. The worker of each instruction set writes
 *   and checks that instruction set's plans (the library compiled into it is that instruction set's).
 *
 *   THE PLANNER (--plans planned; double or single precision): the library's default planning, which measures.
 *   The store is this machine's own wisdom folder, the one the library itself serves from (vfft_wisdom_folder():
 *   src/wisdom/<CPU>/). Each cell is planned first with the gauntlet's planning tool (recal_1d_probe, as
 *   gauntlet.py's calibrate stage runs it): a cell this machine planned before is read back at once, a new one is
 *   measured and saved there, so the next run is quick. Single precision (FP32: 1D powers of two) has no
 *   estimate: its programs plan.
 *
 * Output: <out>/<program>_<date>-<time>/ with summary.txt (the table printed at the end), summary.csv (one row
 * per cell), results_<d>d.csv and control_<d>d.csv (the bench's rows, both engine orders), bench.log (the
 * tools' own output), and store/ (the estimate's plans) or plan.log (the planner's output). */
#ifndef VECTORFFT_BENCH_H
#define VECTORFFT_BENCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static long vs_clock_reads;
static void vs_clock_probe(void);
#define VFFT_CLOCK_PROBE vs_clock_probe()   /* counts every read of the library's clock: a create that reads it measured */
#include "vfft.c"                            /* the whole library, as the bench has it: its estimate and its store writers */
static void vs_clock_probe(void) { vs_clock_reads++; }

#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <fcntl.h>
#include <sched.h>     /* sched_getaffinity: vs_core */
#include <sys/wait.h>
#include <unistd.h>
#endif

#if !defined(VS_BENCH_EXE) || !defined(VS_RECAL_EXE)
#error "VS_BENCH_EXE and VS_RECAL_EXE (the paths of bench_1d_vs_mkl and recal_1d_probe) are set by benchmark_scripts/CMakeLists.txt"
#endif

#define VS_PACE_MS "300"     /* gauntlet.py PACE_MS: the bench's pacing and cool-down between its timing blocks */
#define VS_MAXCELL 256
#define VS_F32_MAXN (1 << 24) /* the FP32 front door: 1D C2C powers of two up to 2^24 */

/* The logical CPU the bench pins its thread to (gauntlet.py's pin_cpu): 2, the second P-core of a
 * hyperthreaded part, when this process may run there; else the last CPU it may run on (a small
 * virtual machine has no CPU 2). */
static const char *vs_core(void)
{
    static char buf[16];
    int c = -1;
#ifdef _WIN32
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) && proc)
    {
        if (proc & ((DWORD_PTR)1 << 2))
            c = 2;
        else
            for (int i = (int)(8 * sizeof proc) - 1; i >= 0 && c < 0; i--)
                if (proc & ((DWORD_PTR)1 << i))
                    c = i;
    }
#elif defined(__linux__)
    cpu_set_t s;
    if (sched_getaffinity(0, sizeof s, &s) == 0)
    {
        if (CPU_ISSET(2, &s))
            c = 2;
        else
            for (int i = CPU_SETSIZE - 1; i >= 0 && c < 0; i--)
                if (CPU_ISSET(i, &s))
                    c = i;
    }
#endif
    if (c < 0)
        c = 2;
    snprintf(buf, sizeof buf, "%d", c);
    return buf;
}

typedef struct {
    int d;                   /* 1, 2 or 3 */
    int n[3];
    char name[40];           /* N, N1xN2, N1xN2xN3: as the gauntlet names it */
} vs_cell_t;

typedef struct {
    int nread;               /* the bench's rows for this cell: one per engine order */
    double ours_ns, cmp_ns;  /* of the worse order */
    double ratio;            /* the comparator's time over ours, the worse of the two orders (the front page's rule) */
    double err;              /* the largest error the bench printed */
} vs_result_t;

typedef struct {
    const char *program, *title;   /* the launcher's program name and its one-line description */
    int fftw;                      /* the comparator: 0 = Intel MKL, 1 = FFTW */
    int measured;                  /* 0 = the estimate's plans, 1 = the planner's */
    int f32;                       /* 1 = single precision (1D powers of two, the planner's plans) */
    int check_only;                /* 1 = plan and check the cells, print the bench's commands, time nothing */
    int ip;                        /* 1 = in place (--inplace): our plans and the comparator's, one buffer */
    const char *out;               /* the directory the run directories are made in */
    const char *cmp, *cmp_short;   /* "Intel MKL" / "MKL", "FFTW" / "FFTW" */
} vs_opts_t;

static double vs_points(const vs_cell_t *c)
{
    double p = 1;
    for (int i = 0; i < c->d; i++)
        p *= c->n[i];
    return p;
}

static int vs_parse_cell(const char *s, vs_cell_t *c)
{
    memset(c, 0, sizeof *c);
    char tail = 0;
    c->d = sscanf(s, "%dx%dx%d%c", &c->n[0], &c->n[1], &c->n[2], &tail);
    if (c->d < 1 || c->d > 3)
        return -1;
    for (int i = 0; i < c->d; i++)
        if (c->n[i] < 2)
            return -1;
    if (c->d == 1)
        snprintf(c->name, sizeof c->name, "%d", c->n[0]);
    else if (c->d == 2)
        snprintf(c->name, sizeof c->name, "%dx%d", c->n[0], c->n[1]);
    else
        snprintf(c->name, sizeof c->name, "%dx%dx%d", c->n[0], c->n[1], c->n[2]);
    return 0;
}

/* ── the estimate's plans as store rows ─────────────────────────────────── */

static void vs_machine(est_machine_t *m)
{   /* the machine the estimate's front door describes (c2c_estimate_il.h, _vfft_create_estimate) */
    memset(m, 0, sizeof *m);
    m->lanes = VFFT_IL_VW;
    m->l1d = vfft_cpu_l1d_bytes();
    m->l2 = vfft_cpu_l2_bytes();
    m->l3 = vfft_cpu_l3_bytes();
    m->cores = vfft_topology()->ncores;
    m->l1d_ways = (int)vfft_cpu_l1d_ways();
}

/* the rows a VFFT_FLAG_ESTIMATE create puts in its private store for this cell at one thread (ip: in place; the 2D
 * and 3D c2c rows serve both placements) */
static int vs_put_estimate(vw2_store_t *st, const est_machine_t *m, const vs_cell_t *c, int ip, char *why, size_t whylen)
{
    if (c->d == 1)
    {
        est_request_t rq;
        est_recipe_t rc;
        memset(&rq, 0, sizeof rq);
        rq.N = c->n[0];
        rq.transform = VFFT_C2C;
        rq.order = VFFT_ORDER_NATURAL;
        rq.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
        rq.nthreads = 1;
        rq.howmany = 1;
        if (est_plan_1d(m, &rq, &rc, why, whylen))
            return -1;
        return est_put_rows(st, &rc, &rq, 1, why, whylen) ? -1 : 0;
    }
    if (c->d == 2)
    {
        est_plan2d_t *p = (est_plan2d_t *)calloc(1, sizeof *p);
        int r = -1;
        if (!p)
            snprintf(why, whylen, "out of memory");
        else if (!est_plan_2d_r(m, c->n[0], c->n[1], EST_ND_2D_RULES, p, why, whylen))
        {
            est_mt_2d(m, 1, p);
            est_mt_2d_serial_1t(m, 1, p);
            r = est_put_rows_2d(st, p, why, whylen) ? -1 : 0;
        }
        free(p);
        return r;
    }
    est_plan3d_t *p = (est_plan3d_t *)calloc(1, sizeof *p);
    int r = -1;
    if (!p)
        snprintf(why, whylen, "out of memory");
    else if (!est_plan_3d(m, c->n[0], c->n[1], c->n[2], p, why, whylen))
    {
        est_mt_3d(m, 1, p);
        r = est_put_rows_3d(st, p, why, whylen) ? -1 : 0;
    }
    free(p);
    return r;
}

static void vs_config(const vs_cell_t *c, int ip, vfft_config_t *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    cfg->transform = VFFT_C2C;
    cfg->placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    cfg->layout = VFFT_LAYOUT_INTERLEAVED;
    cfg->rank = c->d;
    for (int i = 0; i < c->d; i++)
        cfg->n[i] = c->n[i];
    cfg->howmany = 1;
    cfg->nthreads = 1;
}

/* the store's plan for this cell: served with no measurement, and the same output as VFFT_FLAG_ESTIMATE's plan (in
 * place: each plan on its own copy of the input) */
static int vs_check_cell(const vs_cell_t *c, int ip, const char *store, char *why, size_t whylen)
{
    vfft_config_t a, e;
    vs_config(c, ip, &a);
    a.wisdom_dir = store;
    vs_config(c, ip, &e);
    e.flags = VFFT_FLAG_ESTIMATE;
    const long r0 = vs_clock_reads;
    vfft_plan pa = vfft_create(&a);
    const long reads = vs_clock_reads - r0;
    vfft_plan pe = pa ? vfft_create(&e) : NULL;
    int rc = -1;
    if (!pa || !pe)
        snprintf(why, whylen, "%s plan refused: %s", !pa ? "the store's" : "the estimate's", vfft_error());
    else if (reads)
        snprintf(why, whylen, "the store's plan measured (%ld clock reads): its rows are not complete", reads);
    else
    {
        const size_t P = (size_t)vs_points(c), bytes = P * 16;
        double *in = (double *)vfft_malloc(bytes), *oa = (double *)vfft_malloc(bytes), *oe = (double *)vfft_malloc(bytes);
        if (!in || !oa || !oe)
            snprintf(why, whylen, "out of memory (3 x %.0f MiB)", bytes / 1048576.0);
        else
        {
            unsigned s = 12345u;
            for (size_t i = 0; i < 2 * P; i++)
            {
                s = s * 1664525u + 1013904223u;
                in[i] = (double)(s >> 8) / 16777216.0 - 0.5;
            }
            if (ip)
            {
                memcpy(oa, in, bytes);
                memcpy(oe, in, bytes);
                vfft_execute(pa, VFFT_FORWARD, oa, oa);
                vfft_execute(pe, VFFT_FORWARD, oe, oe);
            }
            else
            {
                vfft_execute(pa, VFFT_FORWARD, in, oa);
                vfft_execute(pe, VFFT_FORWARD, in, oe);
            }
            if (memcmp(oa, oe, bytes))
                snprintf(why, whylen, "the store's plan and the estimate's plan give different outputs");
            else
                rc = 0;
        }
        vfft_free(in);
        vfft_free(oa);
        vfft_free(oe);
    }
    if (pa)
        vfft_destroy(pa);
    if (pe)
        vfft_destroy(pe);
    return rc;
}

/* ── the gauntlet's tools as child processes ────────────────────────────── */

static int vs_exists(const char *p)
{
#ifdef _WIN32
    return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
#else
    return access(p, F_OK) == 0;
#endif
}

static int vs_mkdir(const char *p)
{
#ifdef _WIN32
    return _mkdir(p) == 0 || vs_exists(p) ? 0 : -1;
#else
    return mkdir(p, 0755) == 0 || vs_exists(p) ? 0 : -1;
#endif
}

static void vs_setenv(const char *k, const char *v)
{
#ifdef _WIN32
    SetEnvironmentVariableA(k, v);   /* the block CreateProcess hands the children */
#else
    setenv(k, v, 1);
#endif
}

/* Windows: the bench loads MKL's runtime, the Intel OpenMP runtime MKL's default threading layer needs
 * (libiomp5md.dll: without it the bench dies at MKL's first call, 0xC06D007E) and the toolchain's runtime DLLs
 * through PATH (gauntlet.py env()); a terminal set up for oneAPI has them, a plain one may not */
static void vs_child_path(void)
{
#ifdef _WIN32
    const char *cand[12];
    int nc = 0;
    char mklroot[1024] = "", omp[1100] = "";
    char pre[8192] = "";
    size_t l = 0;
#ifdef VS_MKL_BIN
    cand[nc++] = VS_MKL_BIN;
#endif
#ifdef VS_OMP_BIN
    cand[nc++] = VS_OMP_BIN;
#endif
    if (GetEnvironmentVariableA("MKLROOT", mklroot, sizeof mklroot - 40) > 0)
    {
        snprintf(omp, sizeof omp, "%s\\..\\..\\compiler\\latest\\bin", mklroot);
        strcat(mklroot, "\\bin");
        cand[nc++] = mklroot;
        cand[nc++] = omp;
    }
    cand[nc++] = "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\bin";
    cand[nc++] = "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin";
    cand[nc++] = "C:\\Program Files\\Intel\\oneAPI\\mkl\\latest\\bin";
    cand[nc++] = "C:\\Program Files\\Intel\\oneAPI\\compiler\\latest\\bin";
#ifdef VS_TOOLCHAIN_BIN
    cand[nc++] = VS_TOOLCHAIN_BIN;
#endif
    for (int i = 0; i < nc; i++)
        if (vs_exists(cand[i]) && l + strlen(cand[i]) + 2 < sizeof pre)
            l += (size_t)snprintf(pre + l, sizeof pre - l, "%s;", cand[i]);
    const DWORD old = GetEnvironmentVariableA("PATH", NULL, 0);
    char *path = (char *)malloc(l + old + 1);
    if (!path)
        return;
    memcpy(path, pre, l);
    path[l] = 0;
    if (old)
        GetEnvironmentVariableA("PATH", path + l, old);
    SetEnvironmentVariableA("PATH", path);
    free(path);
#endif
}

/* ── where the tools are: a release package or the build tree ─────────────
 * In a release package (benchmark_scripts/scripts/package_release.py) the worker sits in <package>/<isa>/ beside
 * that instruction set's bench and planning tool, FFTW's DLLs are in <package>/fftw/ and the shipped wisdom in
 * <package>/wisdom/ (a root of per-CPU folders, as src/wisdom/ is). In a build tree none of those is beside the
 * worker, and the paths the build compiled in serve. */
#ifdef _WIN32
#define VS_EXE ".exe"
#else
#define VS_EXE ""
#endif
static char vs_bench_exe[1400] = VS_BENCH_EXE;
static char vs_recal_exe[1400] = VS_RECAL_EXE;
static char vs_pkg_wisdom[1400] = "";   /* the package's wisdom root, when the worker runs from a package */

/* the folder this program is in */
static int vs_self_dir(char *out, size_t n)
{
#ifdef _WIN32
    const DWORD l = GetModuleFileNameA(NULL, out, (DWORD)n);
    if (!l || l >= n)
        return -1;
#else
    const ssize_t l = readlink("/proc/self/exe", out, n - 1);
    if (l <= 0)
        return -1;
    out[l] = 0;
#endif
    char *s = strrchr(out, '/'), *b = strrchr(out, '\\');
    if (b && (!s || b > s))
        s = b;
    if (!s)
        return -1;
    *s = 0;
    return 0;
}

static void vs_locate(void)
{
    char dir[1200], p[1400];
    if (vs_self_dir(dir, sizeof dir))
        return;
    snprintf(p, sizeof p, "%s/bench_1d_vs_mkl" VS_EXE, dir);
    if (vs_exists(p))
        snprintf(vs_bench_exe, sizeof vs_bench_exe, "%s", p);
    snprintf(p, sizeof p, "%s/recal_1d_probe" VS_EXE, dir);
    if (vs_exists(p))
        snprintf(vs_recal_exe, sizeof vs_recal_exe, "%s", p);
    /* the package's FFTW, unless the user named one (ref_fftw.h: VFFT_FFTW_DLL / VFFT_FFTWF_DLL) */
    static const char *const dll[2][2] = { { "VFFT_FFTW_DLL", "libfftw3-3.dll" }, { "VFFT_FFTWF_DLL", "libfftw3f-3.dll" } };
    for (int i = 0; i < 2; i++)
    {
        const char *e = getenv(dll[i][0]);
        snprintf(p, sizeof p, "%s/../fftw/%s", dir, dll[i][1]);
        if (!(e && *e) && vs_exists(p))
            vs_setenv(dll[i][0], p);
    }
    snprintf(p, sizeof p, "%s/../wisdom", dir);
    if (vs_exists(p))
        snprintf(vs_pkg_wisdom, sizeof vs_pkg_wisdom, "%s", p);
}

/* run argv[0] with argv, its output appended to logpath; returns its exit code (-1: could not start it) */
static int vs_run(char *const argv[], const char *logpath)
{
#ifdef _WIN32
    char cmd[8192];
    size_t l = 0;
    for (int i = 0; argv[i] && l < sizeof cmd; i++)
        l += (size_t)snprintf(cmd + l, sizeof cmd - l, "%s\"%s\"", i ? " " : "", argv[i]);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE h = CreateFileA(logpath, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE)
        return -1;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = h;
    si.hStdError = h;
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi))
    {
        CloseHandle(h);
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(h);
    return (int)code;
#else
    const pid_t pid = fork();
    if (pid < 0)
        return -1;
    if (pid == 0)
    {
        const int fd = open(logpath, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0)
        {
            dup2(fd, 1);
            dup2(fd, 2);
            close(fd);
        }
        execv(argv[0], argv);
        _exit(127);
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0)
        return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
#endif
}

/* one cell, both engine orders, rows appended to csv: gauntlet.py bench_cell (with --check-only the commands are
 * printed instead). In place: the 1D cell --k1nat, the 2D / 3D cells with --ilip (both engines in place) */
static int vs_bench_cell(const vs_opts_t *o, const vs_cell_t *c, const char *store_file, const char *csv,
                         const char *logpath)
{
    char nstr[40], kstr[16];
    const char *flag;
    if (c->d == 3)
    {   /* the 3D interleaved cell: the shape in the N slot (bench --3dilnat) */
        flag = "--3dilnat";
        snprintf(nstr, sizeof nstr, "%s", c->name);
        snprintf(kstr, sizeof kstr, "1");
    }
    else if (c->d == 2)
    {   /* the 2D interleaved cell: N1 in the N slot, N2 in the K slot (bench --2dilnat) */
        flag = "--2dilnat";
        snprintf(nstr, sizeof nstr, "%d", c->n[0]);
        snprintf(kstr, sizeof kstr, "%d", c->n[1]);
    }
    else
    {   /* the 1D cell: K = 1, natural order, out of place, in place --k1nat (FP32: the bench's --f32 arm of the cell) */
        flag = o->ip ? "--k1nat" : "--k1noop";
        snprintf(nstr, sizeof nstr, "%d", c->n[0]);
        snprintf(kstr, sizeof kstr, "1");
    }
    int bad = 0;
    for (int flip = 0; flip < 2; flip++)
    {
        char fl[4];
        snprintf(fl, sizeof fl, "%d", flip);
        char *argv[16];
        int k = 0;
        argv[k++] = vs_bench_exe;
        if (o->fftw)
        {   /* gauntlet.py: ["--cmp", cmp] + flag */
            argv[k++] = (char *)"--cmp";
            argv[k++] = (char *)"fftw";
        }
        if (o->f32)
            argv[k++] = (char *)"--f32";
        argv[k++] = (char *)flag;
        if (o->ip && c->d > 1)
            argv[k++] = (char *)"--ilip";
        argv[k++] = (char *)store_file;
        argv[k++] = (char *)csv;
        argv[k++] = (char *)VS_PACE_MS;
        argv[k++] = nstr;
        argv[k++] = kstr;
        argv[k++] = (char *)VS_PACE_MS;
        argv[k++] = fl;
        argv[k++] = (char *)vs_core();
        argv[k] = NULL;
        if (o->check_only)
        {
            printf("    would run:");
            for (int i = 0; argv[i]; i++)
                printf(" %s", argv[i]);
            printf("\n");
            continue;
        }
        const int rc = vs_run(argv, logpath);
        if (rc != 0)
        {
            printf("  %s, engine order %d: the bench exited with %d (its output: %s)\n", c->name, flip, rc, logpath);
            if ((unsigned)rc == 0xC06D007Eu || (unsigned)rc == 0xC0000135u || (unsigned)rc == 0xC0000139u)
                printf("  a DLL the bench needs was not found: put Intel MKL's bin folder and the oneAPI compiler's bin "
                       "folder (libiomp5md.dll) on PATH, or run from a oneAPI command prompt\n");
            bad = 1;
        }
    }
    return bad ? -1 : 0;
}

/* the bench's rows of csv for cell c (both orders): the worse order's numbers. The bench names its comparator's
 * columns mkl_ns and ratio_vs_mkl whichever the comparator is. */
static void vs_read_rows(const char *csv, const vs_cell_t *c, vs_result_t *r)
{
    memset(r, 0, sizeof *r);
    FILE *f = fopen(csv, "r");
    if (!f)
        return;
    char line[1024];
    int col_n[3] = { -1, -1, -1 }, col_vf = -1, col_mk = -1, col_ra = -1, col_er = -1;
    if (fgets(line, sizeof line, f))
    {
        int i = 0;
        for (char *t = strtok(line, ",\r\n"); t; t = strtok(NULL, ",\r\n"), i++)
        {
            if (!strcmp(t, "N") || !strcmp(t, "N1")) col_n[0] = i;
            else if (!strcmp(t, "N2")) col_n[1] = i;
            else if (!strcmp(t, "N3")) col_n[2] = i;
            else if (!strcmp(t, "vfft_ns")) col_vf = i;
            else if (!strcmp(t, "mkl_ns")) col_mk = i;
            else if (!strcmp(t, "ratio_vs_mkl")) col_ra = i;
            else if (!strcmp(t, "rt_err")) col_er = i;
        }
    }
    while (fgets(line, sizeof line, f))
    {
        char *fld[32];
        int nf = 0;
        for (char *t = strtok(line, ",\r\n"); t && nf < 32; t = strtok(NULL, ",\r\n"))
            fld[nf++] = t;
        int match = 1;
        for (int k = 0; k < c->d; k++)
            if (col_n[k] < 0 || col_n[k] >= nf || atoi(fld[col_n[k]]) != c->n[k])
                match = 0;
        if (c->d < 3 && col_n[c->d] >= 0)   /* the 2D csv has no N3; a 1D csv no N2 */
            match = 0;
        if (!match || col_ra < 0 || col_ra >= nf || col_vf >= nf || col_mk >= nf || col_er >= nf)
            continue;
        const double ra = atof(fld[col_ra]);
        if (!r->nread || ra < r->ratio)
        {
            r->ratio = ra;
            r->ours_ns = col_vf >= 0 ? atof(fld[col_vf]) : 0;
            r->cmp_ns = col_mk >= 0 ? atof(fld[col_mk]) : 0;
        }
        if (col_er >= 0 && atof(fld[col_er]) > r->err)
            r->err = atof(fld[col_er]);
        r->nread++;
    }
    fclose(f);
}

/* the control cell's readings in a control csv: the run's noise (every reading, both orders) */
static int vs_control_range(const char *csv, double *lo, double *hi)
{
    FILE *f = fopen(csv, "r");
    if (!f)
        return 0;
    char line[1024];
    int col_ra = -1, n = 0;
    if (fgets(line, sizeof line, f))
    {
        int i = 0;
        for (char *t = strtok(line, ",\r\n"); t; t = strtok(NULL, ",\r\n"), i++)
            if (!strcmp(t, "ratio_vs_mkl"))
                col_ra = i;
    }
    while (col_ra >= 0 && fgets(line, sizeof line, f))
    {
        int i = 0;
        for (char *t = strtok(line, ",\r\n"); t; t = strtok(NULL, ",\r\n"), i++)
            if (i == col_ra)
            {
                const double v = atof(t);
                if (!n || v < *lo) *lo = v;
                if (!n || v > *hi) *hi = v;
                n++;
            }
    }
    fclose(f);
    return n;
}

static void vs_time_text(double ns, char *out, size_t len)
{
    if (ns < 1e4)
        snprintf(out, len, "%.2f us", ns / 1e3);
    else if (ns < 1e7)
        snprintf(out, len, "%.1f us", ns / 1e3);
    else
        snprintf(out, len, "%.2f ms", ns / 1e6);
}

/* the timing's length, both engine orders of a cell: about 1 s plus 1 s per 500,000 points (these programs'
 * runs on the i9-14900KF 2026-11-10 against MKL: the powers of two 33 s, the selection 111-138 s), plus two
 * control cells per dimension */
static double vs_expected_seconds(const vs_cell_t *cells, int n)
{
    double s = 0;
    int dims[4] = { 0 };
    for (int i = 0; i < n; i++)
    {
        dims[cells[i].d] = 1;
        s += 1.0 + vs_points(&cells[i]) / 500000.0;
    }
    return s + 2 * 1.5 * (dims[1] + dims[2] + dims[3]);
}

/* gauntlet.py estimate_seconds' planning model when every cell is new (a race), the control cells included */
static double vs_expected_plan_seconds(const vs_cell_t *cells, int n, const vs_cell_t *ctl, int nctl)
{
    double s = 0;
    for (int j = 0; j < n + nctl; j++)
    {
        const vs_cell_t *c = j < n ? &cells[j] : &ctl[j - n];
        const double p = vs_points(c);
        const long long N = (long long)p;
        if (c->d > 1)
            s += 2.0 + p / 50000.0;
        else if ((N & (N - 1)) == 0)
            s += N <= 2048 ? 3 : N <= 262144 ? 20 : 60;
        else
            s += N <= 2048 ? 4 : N <= 4096 ? 8 : N <= 65536 ? 20 : 60;
    }
    return s;
}

/* a stamp of the store's rows: the files' contents without their date= tokens (gauntlet.py cell_rows compares the
 * rows the same way), so a cell that was measured and saved changes it and a cell read back does not */
static unsigned long long vs_store_stamp(const char *store)
{
    static const char *const files[] = { "wisdom2_oop.txt", "wisdom2_prime.txt", "wisdom2_scr.txt", "wisdom2_2d.txt",
                                         "wisdom2_3d.txt", "wisdom2_real.txt", "wisdom2f_oop.txt" };
    unsigned long long h = 1469598103934665603ull;
    for (size_t k = 0; k < sizeof files / sizeof files[0]; k++)
    {
        char p[1400];
        snprintf(p, sizeof p, "%s/%s", store, files[k]);
        FILE *f = fopen(p, "rb");
        if (!f)
            continue;
        int ch, skip = 0;
        char last[6] = { 0 };   /* the last 5 bytes, to see " date=" arrive */
        while ((ch = fgetc(f)) != EOF)
        {
            if (skip)
            {
                if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '|')
                    skip = 0;
                else
                    continue;
            }
            memmove(last, last + 1, 4);
            last[4] = (char)ch;
            if (!memcmp(last, "date=", 5))
            {
                skip = 1;
                continue;
            }
            h = (h ^ (unsigned char)ch) * 1099511628211ull;
        }
        fclose(f);
    }
    return h;
}

/* plan one cell with the library's planner into store, as gauntlet.py's calibrate stage does (recal_1d_probe:
 * natural order, the run's placement, one thread, no recalibration; FP32: its --f32 cell); its output appended to
 * planlog. 1 = planned */
static int vs_plan_cell(const vs_opts_t *o, const vs_cell_t *c, const char *store, const char *run, const char *planlog)
{
    char a[3][16], tmp[1400];
    snprintf(tmp, sizeof tmp, "%s/probe.out", run);
    remove(tmp);
    char *argv[12];
    int k = 0;
    argv[k++] = vs_recal_exe;
    argv[k++] = (char *)store;
    if (o->f32)
        argv[k++] = (char *)"--f32";
    else if (c->d > 1)
        argv[k++] = (char *)(c->d == 2 ? "--2d" : "--3d");
    for (int i = 0; i < c->d; i++)
    {
        snprintf(a[i], sizeof a[i], "%d", c->n[i]);
        argv[k++] = a[i];
    }
    argv[k++] = (char *)"0";                    /* natural order */
    argv[k++] = (char *)(o->ip ? "1" : "0");   /* in place or out of place */
    argv[k++] = (char *)"1";   /* one thread */
    argv[k++] = (char *)"0";   /* no recalibration: a planned cell is read back */
    argv[k] = NULL;
    const int rc = vs_run(argv, tmp);
    char text[4096] = "";
    FILE *f = fopen(tmp, "rb");
    if (f)
    {
        const size_t n = fread(text, 1, sizeof text - 1, f);
        text[n] = 0;
        fclose(f);
    }
    FILE *lg = fopen(planlog, "ab");
    if (lg)
    {
        fputs(text, lg);
        fclose(lg);
    }
    remove(tmp);
    return rc == 0 && strstr(text, "banked") && !strstr(text, "REFUSED");
}

/* the estimate's plans of the cells and the control cells written into store, then each cell checked: the
 * store's plan served with no measurement and equal, bit for bit, to VFFT_FLAG_ESTIMATE's. 0 = ready */
static int vs_plan_estimate(const vs_opts_t *o, const vs_cell_t *cells, int ncell, const vs_cell_t *ctl, int nctl,
                            const char *store)
{
    est_machine_t m;
    vs_machine(&m);
    vw2_store_t st;
    if (vw2_open(&st, store, 1) != VW2_OK && st.nrec != 0)
    {
        printf("%s: cannot open the store %s\n", o->program, store);
        return 1;
    }
    char why[1024];
    int bad = 0;
    for (int j = 0; j < ncell + nctl; j++)
    {
        const vs_cell_t *c = j < ncell ? &cells[j] : &ctl[j - ncell];
        if (vs_put_estimate(&st, &m, c, o->ip, why, sizeof why))
        {
            printf("  %s%s: the estimate refused it: %s\n", j < ncell ? "" : "control ", c->name, why);
            bad = 1;
        }
    }
    const int saved = vw2_save(&st);
    vw2_close(&st);
    if (bad || saved != VW2_OK || vs_clock_reads)
    {
        printf("%s: the estimate's plans could not be written (save %s, clock reads %ld)\n", o->program,
               saved == VW2_OK ? "ok" : "FAILED", vs_clock_reads);
        return 1;
    }
    printf("checking the %d plans (the store's plan = the estimate's plan, bit for bit) ...\n", ncell);
    fflush(stdout);
    for (int i = 0; i < ncell; i++)
        if (vs_check_cell(&cells[i], o->ip, store, why, sizeof why))
        {
            printf("  %s: %s\n", cells[i].name, why);
            bad = 1;
        }
    if (bad)
        printf("%s: a plan failed its check; nothing was timed\n", o->program);
    else if (o->check_only)
        printf("  all %d plans checked\n", ncell);
    return bad;
}

/* the control cells and every cell planned by the library's planner into the wisdom folder store. 0 = ready */
static int vs_plan_measured(const vs_opts_t *o, const vs_cell_t *cells, int ncell, const vs_cell_t *ctl, int nctl,
                            const char *store, const char *run)
{
    if (!vs_exists(vs_recal_exe))
    {
        printf("%s: the planning tool %s is missing: build the target recal_1d_probe (building this target builds it)\n",
               o->program, vs_recal_exe);
        return 1;
    }
    char planlog[1300];
    snprintf(planlog, sizeof planlog, "%s/plan.log", run);
    int bad = 0, fresh = 0;
    const time_t p0 = time(NULL);
    for (int j = 0; j < nctl + ncell; j++)
    {
        const vs_cell_t *c = j < nctl ? &ctl[j] : &cells[j - nctl];
        const unsigned long long s0 = vs_store_stamp(store);
        const time_t c0 = time(NULL);
        const int ok = vs_plan_cell(o, c, store, run, planlog);
        const int new_row = vs_store_stamp(store) != s0;
        fresh += ok && new_row;
        printf("  plan %s%dD %-14s %s (%.0f s)\n", j < nctl ? "control " : "", c->d, c->name,
               !ok ? "REFUSED (see plan.log)" : new_row ? "measured now and saved" : "read from this machine's wisdom",
               difftime(time(NULL), c0));
        fflush(stdout);
        bad |= !ok;
    }
    printf("planning done in %.0f s (%d planned now)\n\n", difftime(time(NULL), p0), fresh);
    if (bad)
        printf("%s: a transform could not be planned; nothing was timed\n", o->program);
    return bad;
}

/* ── the program ────────────────────────────────────────────────────────── */

static int vs_usage(const char *why)
{
    if (why)
        printf("vectorfft_bench_worker: %s\n\n", why);
    printf("vectorfft_bench_worker -- what the vectorfft_vs_* programs run (start those instead)\n\n"
           "  vectorfft_bench_worker --program NAME --title TEXT --vs mkl|fftw --plans estimate|planned\n"
           "                         --prec f64|f32 --cells LIST [--out DIR] [--inplace] [--check-only]\n\n"
           "LIST: transforms separated by commas, N (1D), N1xN2 (2D) or N1xN2xN3 (3D). --inplace times every\n"
           "transform in place (one buffer, both engines). --check-only plans and checks them and prints the\n"
           "bench's commands, timing nothing.\n");
    return why ? 2 : 0;
}

static int vs_main(int argc, char **argv)
{
    vs_opts_t o;
    memset(&o, 0, sizeof o);
    o.out = "vectorfft_results";
    const char *list = NULL, *vs = NULL, *plans = NULL, *prec = "f64";
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--check-only"))
            o.check_only = 1;
        else if (!strcmp(a, "--inplace"))
            o.ip = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h"))
            return vs_usage(NULL);
        else if (!v)
            return vs_usage("an option without its value");
        else if (!strcmp(a, "--program")) o.program = v, i++;
        else if (!strcmp(a, "--title")) o.title = v, i++;
        else if (!strcmp(a, "--vs")) vs = v, i++;
        else if (!strcmp(a, "--plans")) plans = v, i++;
        else if (!strcmp(a, "--prec")) prec = v, i++;
        else if (!strcmp(a, "--cells")) list = v, i++;
        else if (!strcmp(a, "--out")) o.out = v, i++;
        else
            return vs_usage("an option it does not know");
    }
    if (!o.program || !o.title || !list || !vs || !plans)
        return vs_usage("--program, --title, --vs, --plans and --cells are needed");
    if (strcmp(vs, "mkl") && strcmp(vs, "fftw"))
        return vs_usage("--vs is mkl or fftw");
    if (strcmp(plans, "estimate") && strcmp(plans, "planned"))
        return vs_usage("--plans is estimate or planned");
    if (strcmp(prec, "f64") && strcmp(prec, "f32"))
        return vs_usage("--prec is f64 or f32");
    o.fftw = !strcmp(vs, "fftw");
    o.measured = !strcmp(plans, "planned");
    o.f32 = !strcmp(prec, "f32");
    o.cmp = o.fftw ? "FFTW" : "Intel MKL";
    o.cmp_short = o.fftw ? "FFTW" : "MKL";
#ifndef VS_BENCH_HAS_MKL
    if (!o.fftw)
    {
        printf("%s: this build's bench has no MKL comparator (Intel MKL was not found when it was configured)\n", o.program);
        return 1;
    }
#endif
#ifndef VS_HAS_FLOAT
    if (o.f32)
    {
        printf("%s: this build has no single-precision library: configure with -DVFFT_ENABLE_FLOAT=ON\n", o.program);
        return 1;
    }
#endif
    if (o.f32 && !o.measured)
    {
        printf("%s: the single-precision library has no estimate: its programs use the planner's plans\n", o.program);
        return 1;
    }

    /* the cells: the list, separated by commas (or spaces, semicolons) */
    static vs_cell_t cells[VS_MAXCELL];
    int ncell = 0;
    {
        char buf[8192];
        snprintf(buf, sizeof buf, "%s", list);
        for (char *t = strtok(buf, ", ;"); t; t = strtok(NULL, ", ;"))
        {
            if (ncell == VS_MAXCELL)
            {
                printf("%s: at most %d transforms in one run\n", o.program, VS_MAXCELL);
                return 2;
            }
            if (vs_parse_cell(t, &cells[ncell]))
            {
                printf("%s: '%s' is not a transform: N, N1xN2 or N1xN2xN3, every length 2 or more\n", o.program, t);
                return 2;
            }
            const vs_cell_t *c = &cells[ncell];
            if (o.f32 && (c->d != 1 || (c->n[0] & (c->n[0] - 1)) || c->n[0] > VS_F32_MAXN))
            {
                printf("%s: '%s': the single-precision library plans 1D powers of two up to %d\n", o.program, t,
                       VS_F32_MAXN);
                return 2;
            }
            ncell++;
        }
    }
    if (!ncell)
    {
        printf("%s: no transforms to time\n", o.program);
        return 2;
    }
    /* the control cells (gauntlet.py CONTROL_N / _2D / _3D), one per dimension the run has */
    static const char *const control_of[4] = { NULL, "4096", "64x64", "64x64x64" };
    vs_cell_t ctl[4];
    int nctl = 0, ctl_d[4] = { 0 }, has_d[4] = { 0 };
    for (int d = 1; d <= 3; d++)
        for (int i = 0; i < ncell; i++)
            if (cells[i].d == d)
            {
                vs_parse_cell(control_of[d], &ctl[nctl]);
                ctl_d[d] = nctl++;
                has_d[d] = 1;
                break;
            }

    vs_locate();   /* a release package's tools, FFTW and wisdom, else the build tree's */
    if (!vs_exists(vs_bench_exe))
    {
        printf("%s: the bench %s is missing: build the target bench_1d_vs_mkl (building this program builds it)\n",
               o.program, vs_bench_exe);
        return 1;
    }

    /* the run directory */
    char run[1200], store[1300], store_file[1400], benchlog[1300], path[1400], fftw_wis[1300];
    const time_t now = time(NULL);
    struct tm tmv = *localtime(&now);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tmv);
    if (vs_mkdir(o.out))
    {
        printf("%s: cannot create %s\n", o.program, o.out);
        return 1;
    }
    snprintf(run, sizeof run, "%s/%s%s_%s", o.out, o.program, o.ip ? "_inplace" : "", stamp);
    if (o.measured && vs_pkg_wisdom[0])
    {   /* a release package: this CPU's folder of the package's wisdom root, chosen as the library chooses its own
         * (wisdom2_folders.h: the folder stamped with this CPU's identity, else new/, else one made for it) */
        char id_name[160];
        const char *id = vfft_cpu_identity();
        vfft_cpu_identity_folder_name(id, id_name, sizeof id_name);
        vw2_folder_select(vs_pkg_wisdom, id, id_name, store, sizeof store);
    }
    else if (o.measured)
        snprintf(store, sizeof store, "%s", vfft_wisdom_folder());   /* this machine's folder of the library's store */
    else
        snprintf(store, sizeof store, "%s/store", run);
    /* the bench's wisdom slot: a file of the store it must find (it serves its plans from the store's directory) */
    snprintf(store_file, sizeof store_file, "%s/%s", store, o.f32 ? "wisdom2f_oop.txt" : "wisdom2_oop.txt");
    snprintf(benchlog, sizeof benchlog, "%s/bench.log", run);
    snprintf(fftw_wis, sizeof fftw_wis, "%s/%s", o.out, o.f32 ? "fftwf.wis" : "fftw.wis");
    if (vs_mkdir(run) || vs_mkdir(store))
    {
        printf("%s: cannot create %s\n", o.program, store);
        return 1;
    }
    if (!has_d[1] && (!o.measured || !vs_exists(store_file)))
        /* the bench opens <store>/wisdom2_oop.txt by name: a store with no 1D row yet gets the 1D control's (never
         * timed in this run) */
        vs_parse_cell(control_of[1], &ctl[nctl++]);

    printf("%s -- %s\n", o.program, o.title);
    printf("VectorFFT (codelets %s, %s precision) against %s, one thread, %s; the gauntlet's bench, both engine "
           "orders per cell\n", VFFT_IL_VW == 8 ? "AVX-512" : "AVX2", o.f32 ? "single" : "double", o.cmp,
           o.ip ? "in place (one buffer, both engines)" : "out of place");
    const double secs = vs_expected_seconds(cells, ncell);
    if (secs < 75)
        printf("%d transforms; timing them takes about a minute on a fast desktop; run directory %s\n", ncell, run);
    else
        printf("%d transforms; timing them takes about %.0f minutes on a fast desktop; run directory %s\n", ncell,
               floor(secs / 60.0 + 0.5), run);
    if (o.measured)
        printf("plans: the library's planner, which measures; this machine's wisdom folder %s\n"
               "a transform planned on this machine before is read back at once; a new one is measured first and saved\n"
               "there (with none planned yet, planning takes about %.0f minutes)\n",
               store, ceil(vs_expected_plan_seconds(cells, ncell, ctl, nctl) / 60.0));
    else
        printf("plans: the library's estimate (VFFT_FLAG_ESTIMATE), nothing measured\n");
    if (o.fftw)
        printf("FFTW plans each transform by measuring (FFTW_MEASURE) the first time and keeps the plans in %s:\n"
               "the first run takes longer, the next ones do not\n", fftw_wis);
    if (o.check_only)
        printf("--check-only: the plans are made and checked, nothing is timed\n");
    printf("\n");
    fflush(stdout);

    /* 1. the plans (the planner's children get the store and the runtimes first) */
    vs_setenv("VFFT_WISDOM_DIR", store);
    vs_child_path();
    if (o.fftw)   /* gauntlet.py env(): FFTW's wisdom, measured once, replayed after (single precision: its own file) */
        vs_setenv(o.f32 ? "VFFT_FFTWF_WIS" : "VFFT_FFTW_WIS", fftw_wis);
    if (o.measured ? vs_plan_measured(&o, cells, ncell, ctl, nctl, store, run)
                   : vs_plan_estimate(&o, cells, ncell, ctl, nctl, store))
        return 1;

    /* 2. the timing: the gauntlet's bench, one dimension at a time, a control cell before and after */
#ifdef _WIN32
    if (!o.check_only)
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);   /* gauntlet.py keep_awake */
#endif
    static vs_result_t res[VS_MAXCELL];
    double clo[4] = { 0 }, chi[4] = { 0 };
    int cn[4] = { 0 };
    const time_t t0 = time(NULL);
    for (int d = 1; d <= 3; d++)
    {
        int any = 0;
        for (int i = 0; i < ncell; i++)
            any |= cells[i].d == d;
        if (!any)
            continue;
        char csv[1400], ctl_csv[1400];
        snprintf(csv, sizeof csv, "%s/results_%dd.csv", run, d);
        snprintf(ctl_csv, sizeof ctl_csv, "%s/control_%dd.csv", run, d);
        const vs_cell_t *cc = &ctl[ctl_d[d]];
        if (vs_bench_cell(&o, cc, store_file, ctl_csv, benchlog))
        {
            printf("%s: the bench did not run its control cell %s; nothing more is timed\n", o.program, cc->name);
#ifdef _WIN32
            SetThreadExecutionState(ES_CONTINUOUS);
#endif
            return 1;
        }
        for (int i = 0; i < ncell; i++)
        {
            if (cells[i].d != d)
                continue;
            printf("  %dD %-14s%s", d, cells[i].name, o.check_only ? "\n" : "");
            fflush(stdout);
            vs_bench_cell(&o, &cells[i], store_file, csv, benchlog);
            if (o.check_only)
                continue;
            vs_read_rows(csv, &cells[i], &res[i]);
            if (res[i].nread >= 2)
                printf(" %5.2fx  (%.0f s)\n", res[i].ratio, difftime(time(NULL), t0));
            else
                printf(" no result (see %s)\n", benchlog);
            fflush(stdout);
        }
        vs_bench_cell(&o, cc, store_file, ctl_csv, benchlog);
        cn[d] = vs_control_range(ctl_csv, &clo[d], &chi[d]);
    }
#ifdef _WIN32
    if (!o.check_only)
        SetThreadExecutionState(ES_CONTINUOUS);
#endif
    if (o.check_only)
    {
        printf("\n%s: --check-only done, nothing timed (run directory %s)\n", o.program, run);
        return 0;
    }

    /* 3. the table */
    snprintf(path, sizeof path, "%s/summary.txt", run);
    FILE *sum = fopen(path, "w");
    snprintf(path, sizeof path, "%s/summary.csv", run);
    FILE *scsv = fopen(path, "w");
    if (scsv)
        fprintf(scsv, "cell,dims,points,vectorfft_ns,%s_ns,%s_over_vectorfft,max_error\n", o.fftw ? "fftw" : "mkl",
                o.fftw ? "fftw" : "mkl");
    static char text[65536];
    size_t tl = 0;
    tl += (size_t)snprintf(text + tl, sizeof text - tl,
                           "\n%s -- %s\n"
                           "complex to complex, %s, interleaved, natural order, %s, one thread\n"
                           "each speedup = %s's time / VectorFFT's time, the worse of the two engine orders\n\n"
                           "  cell                 points       VectorFFT  %12s    speedup\n",
                           o.program, o.title, o.f32 ? "single precision" : "double",
                           o.ip ? "in place (both engines)" : "out of place", o.cmp_short, o.cmp_short);
    double lg = 0;
    int nok = 0, nwin = 0;
    for (int i = 0; i < ncell; i++)
    {
        char a[32], b[32];
        if (res[i].nread < 2)
        {
            tl += (size_t)snprintf(text + tl, sizeof text - tl, "  %dD %-16s %10.0f    no result\n", cells[i].d,
                                   cells[i].name, vs_points(&cells[i]));
            continue;
        }
        vs_time_text(res[i].ours_ns, a, sizeof a);
        vs_time_text(res[i].cmp_ns, b, sizeof b);
        tl += (size_t)snprintf(text + tl, sizeof text - tl, "  %dD %-16s %10.0f  %12s  %12s    %5.2fx\n", cells[i].d,
                               cells[i].name, vs_points(&cells[i]), a, b, res[i].ratio);
        if (scsv)
            fprintf(scsv, "%s,%d,%.0f,%.1f,%.1f,%.4f,%.3e\n", cells[i].name, cells[i].d, vs_points(&cells[i]),
                    res[i].ours_ns, res[i].cmp_ns, res[i].ratio, res[i].err);
        lg += log(res[i].ratio);
        nok++;
        nwin += res[i].ratio > 1.0;
    }
    if (nok)
        tl += (size_t)snprintf(text + tl, sizeof text - tl,
                               "\n  geometric mean speedup %.2fx over %d transforms; VectorFFT faster on %d of %d\n",
                               exp(lg / nok), nok, nwin, nok);
    for (int d = 1; d <= 3; d++)
        if (cn[d])
            tl += (size_t)snprintf(text + tl, sizeof text - tl,
                                   "  control %s timed %d times through the run: %.2fx .. %.2fx (the machine's noise)\n",
                                   control_of[d], cn[d], clo[d], chi[d]);
    tl += (size_t)snprintf(text + tl, sizeof text - tl, "\n  run directory: %s (%.0f s)\n", run, difftime(time(NULL), t0));
    fputs(text, stdout);
    if (sum)
    {
        fputs(text, sum);
        fclose(sum);
    }
    if (scsv)
        fclose(scsv);
    return nok == ncell ? 0 : 1;
}

#endif /* VECTORFFT_BENCH_H */
