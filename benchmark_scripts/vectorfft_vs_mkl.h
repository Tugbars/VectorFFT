/* vectorfft_vs_mkl.h -- what the vectorfft_vs_mkl_* programs share (benchmark_scripts/CMakeLists.txt).
 *
 * Each program is a fixed list of transforms, its cells: 1D N, 2D N1xN2 (N1 rows of N2 points) or 3D N1xN2xN3;
 * complex to complex, double precision, interleaved, natural order, out of place, one transform per call, one
 * thread. It times them against Intel MKL with the gauntlet's own tools, as gauntlet/gauntlet.py runs them: one
 * bench_1d_vs_mkl process per cell and engine order (the bench pins its core, raises its priority and paces its
 * timing), a control cell before and after each dimension's cells, the bench's own csv. Nothing is timed in
 * this program itself.
 *
 * The plans are the library's ESTIMATE (VFFT_FLAG_ESTIMATE: planned from formulas, nothing measured). The bench
 * creates its plans from a wisdom store, so this program writes the estimate's plans as rows into a new store
 * in the run directory, with the calls the estimate's front door makes for one thread
 * (src/core/il/estimate/c2c_estimate_il.h, c2c_estimate_nd_il.h). Before anything is timed every cell is
 * checked: the store's plan must be served without a measurement, and its output must equal, bit for bit, the
 * output of a plan created with VFFT_FLAG_ESTIMATE. So the bench times exactly the estimate's plan.
 *
 * Output: <out>/<program>_<date>-<time>/ with summary.txt (the table printed at the end), summary.csv (one row
 * per cell), results_<d>d.csv and control_<d>d.csv (the bench's rows, both engine orders), bench.log (the
 * tools' own output) and store/ (the plans). */
#ifndef VECTORFFT_VS_MKL_H
#define VECTORFFT_VS_MKL_H

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

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef VS_BENCH_EXE
#error "VS_BENCH_EXE (the path of bench_1d_vs_mkl) is set by benchmark_scripts/CMakeLists.txt"
#endif

#define VS_PACE_MS "300"     /* gauntlet.py PACE_MS: the bench's pacing and cool-down between its timing blocks */
#define VS_CORE "2"          /* gauntlet.py: the logical CPU the bench pins its thread to */
#define VS_MAXCELL 64

typedef struct {
    int d;                   /* 1, 2 or 3 */
    int n[3];
    char name[40];           /* N, N1xN2, N1xN2xN3: as the gauntlet names it */
} vs_cell_t;

typedef struct {
    int nread;               /* the bench's rows for this cell: one per engine order */
    double ours_ns, mkl_ns;  /* of the worse order */
    double ratio;            /* MKL's time over ours, the worse of the two orders (the front page's rule) */
    double err;              /* the largest error the bench printed */
} vs_result_t;

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
    c->d = sscanf(s, "%dx%dx%d", &c->n[0], &c->n[1], &c->n[2]);
    if (c->d < 1 || c->d > 3)
        return -1;
    for (int i = 0; i < c->d; i++)
        if (c->n[i] < 2)
            return -1;
    snprintf(c->name, sizeof c->name, "%s", s);
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

/* the rows a VFFT_FLAG_ESTIMATE create puts in its private store for this cell at one thread */
static int vs_put_estimate(vw2_store_t *st, const est_machine_t *m, const vs_cell_t *c, char *why, size_t whylen)
{
    if (c->d == 1)
    {
        est_request_t rq;
        est_recipe_t rc;
        memset(&rq, 0, sizeof rq);
        rq.N = c->n[0];
        rq.transform = VFFT_C2C;
        rq.order = VFFT_ORDER_NATURAL;
        rq.placement = VFFT_PLACEMENT_OUT;
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

static void vs_config(const vs_cell_t *c, vfft_config_t *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    cfg->transform = VFFT_C2C;
    cfg->placement = VFFT_PLACEMENT_OUT;
    cfg->layout = VFFT_LAYOUT_INTERLEAVED;
    cfg->rank = c->d;
    for (int i = 0; i < c->d; i++)
        cfg->n[i] = c->n[i];
    cfg->howmany = 1;
    cfg->nthreads = 1;
}

/* the store's plan for this cell: served with no measurement, and the same output as VFFT_FLAG_ESTIMATE's plan */
static int vs_check_cell(const vs_cell_t *c, const char *store, char *why, size_t whylen)
{
    vfft_config_t a, e;
    vs_config(c, &a);
    a.wisdom_dir = store;
    vs_config(c, &e);
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
            vfft_execute(pa, VFFT_FORWARD, in, oa);
            vfft_execute(pe, VFFT_FORWARD, in, oe);
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

/* one cell, both engine orders, rows appended to csv: gauntlet.py bench_cell */
static int vs_bench_cell(const vs_cell_t *c, const char *store_file, const char *csv, const char *logpath)
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
    {   /* the 1D cell: K = 1, natural order, out of place */
        flag = "--k1noop";
        snprintf(nstr, sizeof nstr, "%d", c->n[0]);
        snprintf(kstr, sizeof kstr, "1");
    }
    int bad = 0;
    for (int flip = 0; flip < 2; flip++)
    {
        char fl[4];
        snprintf(fl, sizeof fl, "%d", flip);
        char *const argv[] = { (char *)VS_BENCH_EXE, (char *)flag, (char *)store_file, (char *)csv, (char *)VS_PACE_MS,
                               nstr, kstr, (char *)VS_PACE_MS, fl, (char *)VS_CORE, NULL };
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

/* the bench's rows of csv for cell c (both orders): the worse order's numbers */
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
            r->mkl_ns = col_mk >= 0 ? atof(fld[col_mk]) : 0;
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

/* gauntlet.py estimate_seconds' bench model (the 14900KF's pace), plus two control cells per dimension */
static double vs_expected_seconds(const vs_cell_t *cells, int n)
{
    double s = 0;
    int dims[4] = { 0 };
    for (int i = 0; i < n; i++)
    {
        const double p = vs_points(&cells[i]);
        dims[cells[i].d] = 1;
        if (cells[i].d == 1)
            s += p <= 4096 ? 1.5 : p <= 262144 ? 5 : 30;
        else
            s += 1.5 + p / 300000.0;
    }
    return s + 2 * 1.5 * (dims[1] + dims[2] + dims[3]) + 5;
}

/* ── the program ────────────────────────────────────────────────────────── */

static int vs_main(int argc, char **argv, const char *program, const char *title, const char *const *list)
{
    const char *out = "vectorfft_results";
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out = argv[++i];
        else
        {
            printf("%s: %s\n\n"
                   "  %s [--out DIR]\n\n"
                   "Times VectorFFT against Intel MKL on its fixed list of transforms with the gauntlet's bench\n"
                   "(complex to complex, double, interleaved, natural order, out of place, one thread). The run\n"
                   "directory is created under DIR (default: vectorfft_results in the current directory).\n",
                   program, title, program);
            return strcmp(argv[i], "--help") && strcmp(argv[i], "-h") ? 2 : 0;
        }
    }

    vs_cell_t cells[VS_MAXCELL];
    int ncell = 0;
    for (int i = 0; list[i] && ncell < VS_MAXCELL; i++)
        if (vs_parse_cell(list[i], &cells[ncell]) == 0)
            ncell++;
    static const char *const controls[4] = { NULL, "4096", "64x64", "64x64x64" };   /* gauntlet.py CONTROL_N / _2D / _3D */

    if (!vs_exists(VS_BENCH_EXE))
    {
        printf("%s: the bench %s is missing: build the target bench_1d_vs_mkl (building this target builds it)\n",
               program, VS_BENCH_EXE);
        return 1;
    }

    /* the run directory */
    char run[1200], store[1300], store_file[1400], benchlog[1300], path[1400];
    const time_t now = time(NULL);
    struct tm tmv = *localtime(&now);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tmv);
    if (vs_mkdir(out))
    {
        printf("%s: cannot create %s\n", program, out);
        return 1;
    }
    snprintf(run, sizeof run, "%s/%s_%s", out, program, stamp);
    snprintf(store, sizeof store, "%s/store", run);
    snprintf(store_file, sizeof store_file, "%s/wisdom2_oop.txt", store);
    snprintf(benchlog, sizeof benchlog, "%s/bench.log", run);
    if (vs_mkdir(run) || vs_mkdir(store))
    {
        printf("%s: cannot create %s\n", program, store);
        return 1;
    }

    printf("%s -- %s\n", program, title);
    printf("VectorFFT (codelets %s) against Intel MKL, one thread; the gauntlet's bench, both engine orders per cell\n",
           VFFT_IL_VW == 8 ? "AVX-512" : "AVX2");
    printf("%d transforms, expected time about %.0f minutes on a fast desktop; run directory %s\n\n", ncell,
           ceil(vs_expected_seconds(cells, ncell) / 60.0), run);
    fflush(stdout);

    /* 1. the estimate's plans into the store: the cells and the control cells */
    est_machine_t m;
    vs_machine(&m);
    vw2_store_t st;
    if (vw2_open(&st, store, 1) != VW2_OK && st.nrec != 0)
    {
        printf("%s: cannot open the store %s\n", program, store);
        return 1;
    }
    char why[1024];
    int bad = 0;
    for (int i = 0; i < ncell; i++)
        if (vs_put_estimate(&st, &m, &cells[i], why, sizeof why))
        {
            printf("  %s: the estimate refused it: %s\n", cells[i].name, why);
            bad = 1;
        }
    for (int d = 1; d <= 3; d++)
    {
        vs_cell_t ctl;
        vs_parse_cell(controls[d], &ctl);
        if (vs_put_estimate(&st, &m, &ctl, why, sizeof why))
        {
            printf("  control %s: the estimate refused it: %s\n", ctl.name, why);
            bad = 1;
        }
    }
    const int saved = vw2_save(&st);
    vw2_close(&st);
    if (bad || saved != VW2_OK || vs_clock_reads)
    {
        printf("%s: the estimate's plans could not be written (save %s, clock reads %ld)\n", program,
               saved == VW2_OK ? "ok" : "FAILED", vs_clock_reads);
        return 1;
    }

    /* 2. every plan checked before anything is timed */
    printf("checking the %d plans (the store's plan = the estimate's plan, bit for bit) ...\n", ncell);
    fflush(stdout);
    for (int i = 0; i < ncell; i++)
        if (vs_check_cell(&cells[i], store, why, sizeof why))
        {
            printf("  %s: %s\n", cells[i].name, why);
            bad = 1;
        }
    if (bad)
    {
        printf("%s: a plan failed its check; nothing was timed\n", program);
        return 1;
    }

    /* 3. the timing: the gauntlet's bench, one dimension at a time, a control cell before and after */
    vs_setenv("VFFT_WISDOM_DIR", store);
    vs_child_path();
#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);   /* gauntlet.py keep_awake */
#endif
    vs_result_t res[VS_MAXCELL];
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
        vs_cell_t ctl;
        vs_parse_cell(controls[d], &ctl);
        if (vs_bench_cell(&ctl, store_file, ctl_csv, benchlog))
        {
            printf("%s: the bench did not run its control cell %s; nothing more is timed\n", program, ctl.name);
#ifdef _WIN32
            SetThreadExecutionState(ES_CONTINUOUS);
#endif
            return 1;
        }
        for (int i = 0; i < ncell; i++)
        {
            if (cells[i].d != d)
                continue;
            printf("  %dD %-14s", d, cells[i].name);
            fflush(stdout);
            vs_bench_cell(&cells[i], store_file, csv, benchlog);
            vs_read_rows(csv, &cells[i], &res[i]);
            if (res[i].nread >= 2)
                printf(" %5.2fx  (%.0f s)\n", res[i].ratio, difftime(time(NULL), t0));
            else
                printf(" no result (see %s)\n", benchlog);
            fflush(stdout);
        }
        vs_bench_cell(&ctl, store_file, ctl_csv, benchlog);
        cn[d] = vs_control_range(ctl_csv, &clo[d], &chi[d]);
    }
#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS);
#endif

    /* 4. the table */
    snprintf(path, sizeof path, "%s/summary.txt", run);
    FILE *sum = fopen(path, "w");
    snprintf(path, sizeof path, "%s/summary.csv", run);
    FILE *scsv = fopen(path, "w");
    if (scsv)
        fprintf(scsv, "cell,dims,points,vectorfft_ns,mkl_ns,mkl_over_vectorfft,max_error\n");
    char text[16384];
    size_t tl = 0;
    tl += (size_t)snprintf(text + tl, sizeof text - tl,
                           "\n%s -- %s\n"
                           "complex to complex, double, interleaved, natural order, out of place, one thread\n"
                           "each speedup = MKL's time / VectorFFT's time, the worse of the two engine orders\n\n"
                           "  cell                 points       VectorFFT          MKL    speedup\n",
                           program, title);
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
        vs_time_text(res[i].mkl_ns, b, sizeof b);
        tl += (size_t)snprintf(text + tl, sizeof text - tl, "  %dD %-16s %10.0f  %12s  %12s    %5.2fx\n", cells[i].d,
                               cells[i].name, vs_points(&cells[i]), a, b, res[i].ratio);
        if (scsv)
            fprintf(scsv, "%s,%d,%.0f,%.1f,%.1f,%.4f,%.3e\n", cells[i].name, cells[i].d, vs_points(&cells[i]),
                    res[i].ours_ns, res[i].mkl_ns, res[i].ratio, res[i].err);
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
                                   controls[d], cn[d], clo[d], chi[d]);
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

#endif /* VECTORFFT_VS_MKL_H */
