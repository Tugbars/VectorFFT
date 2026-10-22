/* estimate_tool.c -- the 1D estimate's tool: print the recipes, build and check the plans, compare
 * the recipes with the stored winners. Tracked, not wired into the library. The c2c recipes and plans are the LIBRARY's
 * (VFFT_FLAG_ESTIMATE: src/core/il/planning/estimate_il.h, src/core/il/rank1/c2c_estimate_il.h); the
 * real ones are this folder's (estimate_1d.h, estimate_build.h).
 *
 *   estimate_tool show    N... [options]   the recipe of each cell, in the store's own text
 *   estimate_tool check   N... [options]   build every cell through est_build_any and check it:
 *                                          built / refused(reason) / measured(FAIL) /
 *                                          rewritten(FAIL) / wrong(FAIL)
 *   estimate_tool compare [N...] [options] the recipes against the stored winner rows of this CPU's
 *                                          store folder, read only: route and recipe match counts
 *   estimate_tool cell    N t ord place T  one check cell (the isolated runner, below)
 *
 * N: 1024 | 2^20 | A-B (every N) | A-B:pow2 | A-B:sK (step K) | A-B:rK (K random N, fixed seed)
 * options: -t c2c,r2c,c2r   -o nat,scr   -p oop,ip   -T 1,8   --tol 1e-11   --isolate   -q   -v
 *          --store DIR (compare)   --l1d B --l2 B --l3 B (show, compare: another machine's caches; the
 *          formulas follow this build's lanes, so --lanes must name VFFT_IL_VW)
 *
 * The correctness check (no timing anywhere): the forward against a long-double DFT -- every bin for
 * N <= 4096 (scrambled: the sorted multiset of the outputs), 32 bins by direct long-double sums
 * above (natural: at their positions; scrambled: each found among the outputs) -- a Parseval check,
 * and a forward-backward round trip: c2c on the plan itself, real through the r2c plan of the same
 * cell feeding the c2r plan. A round trip alone never gates a permuted forward.
 *
 * Nothing here measures. A create that reads the library's clock has started a race: the clock
 * probe below ends the process at that read (exit code 3) and reports the cell measured(FAIL). A
 * cell whose recipe holds a nested create (a four-step child, which no pre-flight can build) runs in
 * its own process, so a race there costs only that cell; --isolate runs every cell that way.
 *
 * Build (the TU includes vfft.c itself, so NO --vfft; REPLAY_API section 1.1):
 *   python gauntlet/build.py --compile --src src/estimate/estimate_tool.c
 * Run from the Bash tool (the mingw DLLs are on its PATH). See README.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* THE NO-MEASUREMENT PROBE (REPLAY_API section 1.2): evaluated at every library clock read
 * (vfft_now_ns, race_scope.h:636-642), before vfft.c so every engine sees it. A function, so two reads
 * in one expression stay sequenced. est_clock_reads is read as "changed or not" (worker threads may
 * lose increments). */
static long est_clock_reads;
static int est_clock_trap;           /* 1 around an estimate build: the first clock read ends the process */
static void est_clock_probe(void);
#define VFFT_CLOCK_PROBE est_clock_probe()
#include "vfft.c"                    /* FIRST: the whole library in its own order (form_slot_gate.c:23) */
#include "estimate_1d.h"             /* the formulas: est_plan_any (c2c: the library's est_plan_1d), est_recipe_print */
#include "estimate_build.h"          /* the build: est_build_any (c2c: the library's est_build_c2c) */

#define EST_FULL_N 4096   /* every bin against the long-double DFT up to here */
#define EST_NBINS 32      /* sampled bins above */

static char est_cell[192];   /* the cell under build: the trap's line names it */
static int est_child;        /* 1 = this process runs one isolated cell */

static void est_clock_trapped(void)
{
    est_clock_trap = 0;
    printf("%-30s measured(FAIL): the create read the library's clock (a race started there); the process ends at "
           "that read\n", est_cell);
    printf("@result measured 0\n");
    if (!est_child)
        printf("[estimate] a create that measures cannot be stopped from outside the library: the sweep ends at this "
               "cell -- rerun the cells after it, or pass --isolate\n");
    fflush(stdout);
    fflush(stderr);
    _exit(3);
}
static void est_clock_probe(void)
{
    est_clock_reads++;
    if (est_clock_trap)
        est_clock_trapped();
}

/* ================================================================ options */
typedef struct
{
    int mode;                  /* 's' show, 'k' check, 'c' compare, 'x' one cell */
    int *Ns, nN, capN;
    int tr[3];                 /* c2c, r2c, c2r */
    int ord[2];                /* nat, scr */
    int pl[2];                 /* oop, ip */
    int Ts[16], nT, Tgiven;
    double tol;
    int isolate, quiet, verbose;
    const char *store;
    long l1d, l2, l3;
    int lanes;
} est_opt_t;

static void est_usage(void)
{
    printf("usage: estimate_tool show|check|compare [N...] [options]\n"
           "       estimate_tool cell N c2c|r2c|c2r nat|scr oop|ip T [--tol X]\n"
           "  N        1024 | 2^20 | A-B | A-B:pow2 | A-B:sK | A-B:rK\n"
           "  -t list  transforms c2c,r2c,c2r (default all)\n"
           "  -o list  orders nat,scr (default both; real is natural only)\n"
           "  -p list  placements oop,ip (default both)\n"
           "  -T list  thread counts (default 1; compare: every stored T)\n"
           "  --tol X  the relative error a check accepts (default 1e-11)\n"
           "  --isolate  every check cell in its own process (nested creates always are)\n"
           "  -q / -v  only failures and refusals / more detail\n"
           "  --store DIR  compare against this store folder (default: this CPU's under src/wisdom)\n"
           "  --l1d B --l2 B --l3 B  plan for another machine's caches (show, compare)\n"
           "  --lanes 4|8  must name this build's lanes: the formulas follow its laws\n");
}

static long est_num(const char *s, const char **end)
{
    char *e;
    long v = strtol(s, &e, 10);
    if (*e == '^')
    {
        long p = strtol(e + 1, &e, 10), r = 1;
        while (p-- > 0) r *= v;
        v = r;
    }
    if (end) *end = e;
    return v;
}
static void est_add_N(est_opt_t *o, long v)
{
    if (v < 1 || v > (1L << 30))
        return;
    if (o->nN == o->capN)
    {
        o->capN = o->capN ? 2 * o->capN : 256;
        o->Ns = (int *)realloc(o->Ns, (size_t)o->capN * sizeof *o->Ns);
        if (!o->Ns) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    o->Ns[o->nN++] = (int)v;
}
static int est_cmp_int(const void *a, const void *b) { return (*(const int *)a > *(const int *)b) - (*(const int *)a < *(const int *)b); }
/* N, A-B, A-B:pow2, A-B:sK, A-B:rK */
static int est_add_range(est_opt_t *o, const char *s)
{
    const char *e;
    long a = est_num(s, &e), b, v;
    if (*e == 0) { est_add_N(o, a); return 0; }
    if (*e != '-') return -1;
    b = est_num(e + 1, &e);
    if (b < a) return -1;
    if (*e == 0) { for (v = a; v <= b; v++) est_add_N(o, v); return 0; }
    if (*e != ':') return -1;
    e++;
    if (!strcmp(e, "pow2"))
    {
        for (v = 1; v <= b; v *= 2) if (v >= a) est_add_N(o, v);
        return 0;
    }
    if (*e == 's')
    {
        long st = atol(e + 1);
        if (st < 1) return -1;
        for (v = a; v <= b; v += st) est_add_N(o, v);
        return 0;
    }
    if (*e == 'r')
    {   /* K values drawn with a fixed seed, so a run is repeatable */
        long k = atol(e + 1), i;
        unsigned long long x = 0x9E3779B97F4A7C15ull ^ (unsigned long long)(a * 31 + b);
        for (i = 0; i < k; i++)
        {
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            est_add_N(o, a + (long)((x * 2685821657736338717ull) % (unsigned long long)(b - a + 1)));
        }
        return 0;
    }
    return -1;
}
static int est_list_has(const char *list, const char *w)
{
    char b[64];
    const char *p = list;
    while (*p)
    {
        size_t n = strcspn(p, ",");
        if (n < sizeof b) { memcpy(b, p, n); b[n] = 0; if (!strcmp(b, w)) return 1; }
        p += n;
        if (*p == ',') p++;
    }
    return 0;
}

static int est_parse(int argc, char **argv, est_opt_t *o)
{
    int i;
    memset(o, 0, sizeof *o);
    o->tr[0] = o->tr[1] = o->tr[2] = 1;
    o->ord[0] = o->ord[1] = 1;
    o->pl[0] = o->pl[1] = 1;
    o->Ts[0] = 1;
    o->nT = 1;
    o->tol = 1e-11;
    if (argc < 2) return -1;
    if (!strcmp(argv[1], "show")) o->mode = 's';
    else if (!strcmp(argv[1], "check")) o->mode = 'k';
    else if (!strcmp(argv[1], "compare")) o->mode = 'c';
    else if (!strcmp(argv[1], "cell")) o->mode = 'x';
    else return -1;
    if (o->mode == 'x')
    {   /* cell N t ord place T */
        if (argc < 7) return -1;
        est_add_N(o, est_num(argv[2], NULL));
        o->tr[0] = !strcmp(argv[3], "c2c"); o->tr[1] = !strcmp(argv[3], "r2c"); o->tr[2] = !strcmp(argv[3], "c2r");
        o->ord[0] = !strcmp(argv[4], "nat"); o->ord[1] = !strcmp(argv[4], "scr");
        o->pl[0] = !strcmp(argv[5], "oop"); o->pl[1] = !strcmp(argv[5], "ip");
        o->Ts[0] = atoi(argv[6]);
        i = 7;
    }
    else
        i = 2;
    for (; i < argc; i++)
    {
        const char *a = argv[i];
        if (!strcmp(a, "-t") && i + 1 < argc)
        { a = argv[++i]; o->tr[0] = est_list_has(a, "c2c"); o->tr[1] = est_list_has(a, "r2c"); o->tr[2] = est_list_has(a, "c2r"); }
        else if (!strcmp(a, "-o") && i + 1 < argc)
        { a = argv[++i]; o->ord[0] = est_list_has(a, "nat"); o->ord[1] = est_list_has(a, "scr"); }
        else if (!strcmp(a, "-p") && i + 1 < argc)
        { a = argv[++i]; o->pl[0] = est_list_has(a, "oop"); o->pl[1] = est_list_has(a, "ip"); }
        else if (!strcmp(a, "-T") && i + 1 < argc)
        {
            const char *p = argv[++i];
            o->nT = 0;
            o->Tgiven = 1;
            while (*p && o->nT < 16) { o->Ts[o->nT++] = (int)strtol(p, (char **)&p, 10); if (*p == ',') p++; else break; }
        }
        else if (!strcmp(a, "--tol") && i + 1 < argc) o->tol = atof(argv[++i]);
        else if (!strcmp(a, "--isolate")) o->isolate = 1;
        else if (!strcmp(a, "-q")) o->quiet = 1;
        else if (!strcmp(a, "-v")) o->verbose = 1;
        else if (!strcmp(a, "--store") && i + 1 < argc) o->store = argv[++i];
        else if (!strcmp(a, "--l1d") && i + 1 < argc) o->l1d = atol(argv[++i]);
        else if (!strcmp(a, "--l2") && i + 1 < argc) o->l2 = atol(argv[++i]);
        else if (!strcmp(a, "--l3") && i + 1 < argc) o->l3 = atol(argv[++i]);
        else if (!strcmp(a, "--lanes") && i + 1 < argc) o->lanes = atoi(argv[++i]);
        else if (a[0] != '-' && o->mode != 'x')
        {
            if (est_add_range(o, a)) { fprintf(stderr, "bad N range '%s'\n", a); return -1; }
        }
        else { fprintf(stderr, "unknown argument '%s'\n", a); return -1; }
    }
    if (o->nN > 1)
    {   /* sorted, unique */
        int j, k = 1;
        qsort(o->Ns, (size_t)o->nN, sizeof *o->Ns, est_cmp_int);
        for (j = 1; j < o->nN; j++) if (o->Ns[j] != o->Ns[k - 1]) o->Ns[k++] = o->Ns[j];
        o->nN = k;
    }
    return 0;
}

/* ================================================================ the machine */
/* Read ONCE at start-up (estimate_recipe.h:29-32): the first vfft_cpu_cache() call moves the thread
 * to a P-core and back (cpu_cache.h:495-505), never inside a create. */
static est_machine_t est_machine_here(void)
{
    est_machine_t m;
    memset(&m, 0, sizeof m);
    m.lanes = VFFT_IL_VW;                 /* il/isa/il_isa.h:37-46 */
    m.l1d = vfft_cpu_l1d_bytes();         /* cpu_cache.h:560 */
    m.l2 = vfft_cpu_l2_bytes();           /* :564 */
    m.l3 = vfft_cpu_l3_bytes();           /* :569 */
    m.cores = vfft_topology()->ncores;    /* cpu_topology.h:47,189 */
    return m;
}
static void est_machine_print(const char *what, const est_machine_t *m)
{
    printf("%s: lanes %d (%s build), L1d %ld, L2 %ld, L3 %ld, cores %d\n", what, m->lanes, VFFT_ISA_NAME, m->l1d,
           m->l2, m->l3, m->cores);
}

/* ================================================================ cells */
static const char *est_tname(int t) { return t == VFFT_C2C ? "c2c" : t == VFFT_R2C ? "r2c" : "c2r"; }
static void est_cell_name(char *b, size_t n, const est_request_t *rq)
{
    snprintf(b, n, "%s N=%d %s %s T=%d", est_tname(rq->transform), rq->N,
             rq->order == VFFT_ORDER_SCRAMBLED ? "scr" : "nat", rq->placement == VFFT_PLACEMENT_IN ? "ip" : "oop",
             rq->nthreads > 1 ? rq->nthreads : 1);
}

/* one line of the recipe: the route and its shape */
static void est_brief_c2c(char *b, size_t n, const est_recipe_t *rc, int T)
{
    char s[96];
    const int r = rc->k1_il_route;
    const char *nm = (r >= 0 && r <= VW2_OOP_IL_ROUTE_MAX) ? vw2_oop_il_name[r] : "?";
    s[0] = 0;
    switch (r)
    {
    case VFFT_K1_IL_MONO: snprintf(b, n, "mono kv=%d", rc->il_kv); return;
    case VFFT_K1_IL_2P_PURE: snprintf(b, n, "2p %d.%d kv=0x%x bkv=0x%x", rc->il_R1, rc->il_R2, rc->il_kv, rc->il_bkv); return;
    case VFFT_K1_IL_CHAIN3: snprintf(b, n, "chain3 %d.%d.%d kv=0x%x bkv=0x%x", rc->il_c3[0], rc->il_c3[1], rc->il_c3[2], rc->il_kv, rc->il_bkv); return;
    case VFFT_K1_IL_FLAT:
        est__ints(s, sizeof s, rc->il_fl, rc->il_fl_n);
        if (T > 1) snprintf(b, n, "flat %s forms=%s tw=%d mt=%d/%d", s, rc->il_flf[0] ? rc->il_flf : "-", rc->il_tw, rc->il_mt, rc->il_mt_tw);
        else snprintf(b, n, "flat %s forms=%s tw=%d", s, rc->il_flf[0] ? rc->il_flf : "-", rc->il_tw);
        return;
    case VFFT_K1_IL_ZTT:
        est__ints(s, sizeof s, rc->il_zt, rc->il_zt_n);
        if (T > 1) snprintf(b, n, "ztt %s tw=%d mt=%d", s, rc->il_tw, rc->il_mt);
        else snprintf(b, n, "ztt %s tw=%d", s, rc->il_tw);
        return;
    case VFFT_K1_IL_FS:
        if (T > 1) snprintf(b, n, "fs %dx%d mt=%d", rc->il_R1, rc->il_R2, rc->il_mt);
        else snprintf(b, n, "fs %dx%d kv=%d", rc->il_R1, rc->il_R2, rc->il_kv);
        return;
    case VFFT_K1_IL_PRIME:
        snprintf(b, n, "prime %s %s %s tw=%d%s", rc->prime.method == 1 ? "rader" : "bluestein", rc->prime.in_kind,
                 rc->prime.in_shape, rc->prime.in_tw, T > 1 ? (rc->il_mt ? " mt" : " st") : "");
        return;
    default: snprintf(b, n, "%s", nm); return;
    }
}
static void est_brief(char *b, size_t n, const est_any_recipe_t *rc, const est_request_t *rq)
{
    char s[160];
    const int T = rq->nthreads > 1 ? rq->nthreads : 1;
    switch (est__eng(rc, rq))
    {
    case EST_ENG_K1: est_brief_c2c(b, n, &rc->c, T); return;
    case EST_ENG_ZRM: snprintf(b, n, "zrm"); return;
    case EST_ENG_ZRP: snprintf(b, n, "zrp %d.%d %s", rc->zrp.R1, rc->zrp.R2, rc->zrp.form ? "r2z" : "n1t"); return;
    case EST_ENG_ZTTR:
        est__ints(s, sizeof s, rc->zttr.chain, rc->zttr.nf);
        snprintf(b, n, "zttr %s tile=%zu stk=%d mt=%d", s, rc->zttr.tile, rc->zttr.stk, rc->zttr.mt);
        return;
    case EST_ENG_ZR2C:
        est_brief_c2c(s, sizeof s, &rc->c, T > 1 && rc->c.k1_il_route == VFFT_K1_IL_PRIME ? T : 1);
        snprintf(b, n, "zr2c %s [%s]%s", rc->zr2c.route ? "child_nat_ip" : "child_oop_il", s,
                 T > 1 ? (rc->zr2c.fold_mt ? " fold=mt" : " fold=st") : "");
        return;
    case EST_ENG_ZFSR: snprintf(b, n, "zfsr %dx%d", rc->zfsr.n1, rc->zfsr.n2); return;
    case EST_ENG_ZRF:
        est__ints(s, sizeof s, rc->zrf.chain, rc->zrf.K);
        snprintf(b, n, "zrf %s msz=%d tile=%d mt=%d", s, !rc->zrf.nomsz, rc->zrf.tile, rc->zrf.mt);
        return;
    case EST_ENG_ZRB: snprintf(b, n, "zrb m=%d %s %s tw=%d mt=%d", rc->zrb.M, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw, rc->zrb.mt); return;
    default: snprintf(b, n, "eng=%s", rc->eng); return;
    }
}

/* ================================================================ the reference */
/* deterministic inputs in [-1, 1) */
static unsigned long long est_rng;
static double est_rand(void)
{
    est_rng ^= est_rng >> 12; est_rng ^= est_rng << 25; est_rng ^= est_rng >> 27;
    return (double)((est_rng * 2685821657736338717ull) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}
static void *est_alloc(size_t bytes)
{
    void *p = vfft_aligned_alloc(bytes ? bytes : 64);   /* every plan was measured on aligned buffers */
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(2); }
    return p;
}
/* the tool's own small buffers (references, lists, text): out of memory ends the run cleanly, as
 * est_alloc does, rather than writing through NULL */
static void *est_xmalloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(2); }
    return p;
}
static void *est_xrealloc(void *p, size_t bytes)
{
    void *q = realloc(p, bytes ? bytes : 1);
    if (!q) { free(p); fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(2); }
    return q;
}

static const long double est_pi = 3.141592653589793238462643383279502884L;

/* out_f = sum_{m < cnt} v[m] exp(sgn 2 pi i (m f mod N) / N) for f in [0, nout): the full long-double
 * DFT, one twiddle table of N entries (N <= EST_FULL_N). v complex interleaved, or real (cplx 0). */
static void est_dft_full(const double *v, int cplx, long cnt, long N, int sgn, long nout, long double *outr,
                         long double *outi)
{
    long double *c = (long double *)malloc((size_t)N * sizeof *c), *s = (long double *)malloc((size_t)N * sizeof *s);
    long f, m;
    if (!c || !s) { fprintf(stderr, "out of memory\n"); exit(2); }
    for (m = 0; m < N; m++)
    {
        c[m] = cosl(2.0L * est_pi * (long double)m / (long double)N);
        s[m] = (long double)sgn * sinl(2.0L * est_pi * (long double)m / (long double)N);
    }
    for (f = 0; f < nout; f++)
    {
        long double sr = 0.0L, si = 0.0L;
        long idx = 0;
        for (m = 0; m < cnt; m++)
        {
            const long double xr = cplx ? v[2 * m] : v[m], xi = cplx ? v[2 * m + 1] : 0.0;
            sr += xr * c[idx] - xi * s[idx];
            si += xr * s[idx] + xi * c[idx];
            idx += f;
            if (idx >= N) idx -= N;
        }
        outr[f] = sr;
        outi[f] = si;
    }
    free(c);
    free(s);
}
/* one output f of the same sum, blocked: exp(theta(m f)) = exp(theta(aB f)) exp(theta(b f)) with
 * m = aB + b, so N up to 2^25 needs only B + N/B long-double twiddles per output */
static void est_dft_one(const double *v, int cplx, long cnt, long N, int sgn, long f, long double *orr, long double *oi)
{
    enum { B = 4096 };
    const long nb = (cnt + B - 1) / B;
    long double *ic = (long double *)malloc(2 * B * sizeof *ic), *oc = (long double *)malloc((size_t)(2 * nb) * sizeof *oc);
    long double tr = 0.0L, ti = 0.0L;
    long a, b;
    if (!ic || !oc) { fprintf(stderr, "out of memory\n"); exit(2); }
    for (b = 0; b < B; b++)
    {
        const long j = (long)(((unsigned long long)b * (unsigned long long)f) % (unsigned long long)N);
        ic[2 * b] = cosl(2.0L * est_pi * j / N);
        ic[2 * b + 1] = (long double)sgn * sinl(2.0L * est_pi * j / N);
    }
    for (a = 0; a < nb; a++)
    {
        const long j = (long)((((unsigned long long)a * B) % (unsigned long long)N) * (unsigned long long)f % (unsigned long long)N);
        oc[2 * a] = cosl(2.0L * est_pi * j / N);
        oc[2 * a + 1] = (long double)sgn * sinl(2.0L * est_pi * j / N);
    }
    for (a = 0; a < nb; a++)
    {
        long double sr = 0.0L, si = 0.0L;
        const long m0 = a * B, m1 = (m0 + B < cnt) ? m0 + B : cnt;
        long m;
        for (m = m0; m < m1; m++)
        {
            const long double xr = cplx ? v[2 * m] : v[m], xi = cplx ? v[2 * m + 1] : 0.0;
            const long double cr = ic[2 * (m - m0)], ci = ic[2 * (m - m0) + 1];
            sr += xr * cr - xi * ci;
            si += xr * ci + xi * cr;
        }
        tr += sr * oc[2 * a] - si * oc[2 * a + 1];
        ti += sr * oc[2 * a + 1] + si * oc[2 * a];
    }
    *orr = tr;
    *oi = ti;
    free(ic);
    free(oc);
}

/* the sampled positions: the edges, then fixed-seed draws in [0, range) */
static int est_samples(long range, long N, long *k)
{
    int n = 0, i;
    unsigned long long x = 0xD1B54A32D192ED03ull ^ (unsigned long long)N;
    const long fixed[5] = { 0, 1, 2, range - 1, range / 2 };
    for (i = 0; i < 5 && n < EST_NBINS; i++)
    {
        int j, dup = 0;
        for (j = 0; j < n; j++) if (k[j] == fixed[i]) dup = 1;
        if (!dup && fixed[i] >= 0 && fixed[i] < range) k[n++] = fixed[i];
    }
    while (n < EST_NBINS && n < range)
    {
        long v;
        int j, dup = 0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        v = (long)((x * 2685821657736338717ull) % (unsigned long long)range);
        for (j = 0; j < n; j++) if (k[j] == v) dup = 1;
        if (!dup) k[n++] = v;
    }
    return n;
}

/* one N's inputs and references, each made when first asked */
typedef struct
{
    long N;
    int have_c, have_r, have_h;
    /* c2c */
    double *xc;                          /* the input */
    long double sxc;                     /* sum |x|^2, long double: the Parseval reference */
    long double *Xr_c, *Xi_c;            /* N <= EST_FULL_N: every bin */
    long kc[EST_NBINS]; long double Kr_c[EST_NBINS], Ki_c[EST_NBINS]; int nkc;
    /* r2c */
    double *xr;
    long double sxr;
    long double *Xr_r, *Xi_r;            /* bins 0..N/2 */
    long kr[EST_NBINS]; long double Kr_r[EST_NBINS], Ki_r[EST_NBINS]; int nkr;
    /* c2r: a Hermitian half spectrum and the reals it makes */
    double *hc;
    long double eh;                      /* 2 (N/2 + 1) doubles; E = |H0|^2 + |H_N/2|^2 + 2 sum |H_k|^2 */
    long double *xh;                     /* N <= EST_FULL_N: every real */
    long nh[EST_NBINS]; long double Kh[EST_NBINS]; int nnh;
} est_ref_t;

static void est_ref_free(est_ref_t *r)
{
    vfft_aligned_free(r->xc); vfft_aligned_free(r->xr); vfft_aligned_free(r->hc);
    free(r->Xr_c); free(r->Xi_c); free(r->Xr_r); free(r->Xi_r); free(r->xh);
    memset(r, 0, sizeof *r);
}
static void est_ref_c(est_ref_t *r)
{
    const long N = r->N;
    long i;
    if (r->have_c) return;
    r->xc = (double *)est_alloc((size_t)2 * N * sizeof(double));
    est_rng = 0x1234567ull ^ ((unsigned long long)N << 20);
    r->sxc = 0.0;
    for (i = 0; i < 2 * N; i++) { r->xc[i] = est_rand(); r->sxc += (long double)r->xc[i] * r->xc[i]; }
    if (N <= EST_FULL_N)
    {
        r->Xr_c = (long double *)est_xmalloc((size_t)N * sizeof(long double));
        r->Xi_c = (long double *)est_xmalloc((size_t)N * sizeof(long double));
        est_dft_full(r->xc, 1, N, N, -1, N, r->Xr_c, r->Xi_c);
    }
    else
    {
        int j;
        r->nkc = est_samples(N, N, r->kc);
        for (j = 0; j < r->nkc; j++) est_dft_one(r->xc, 1, N, N, -1, r->kc[j], &r->Kr_c[j], &r->Ki_c[j]);
    }
    r->have_c = 1;
}
static void est_ref_r(est_ref_t *r)
{
    const long N = r->N, H = N / 2 + 1;
    long i;
    if (r->have_r) return;
    r->xr = (double *)est_alloc((size_t)N * sizeof(double));
    est_rng = 0x7654321ull ^ ((unsigned long long)N << 20);
    r->sxr = 0.0;
    for (i = 0; i < N; i++) { r->xr[i] = est_rand(); r->sxr += (long double)r->xr[i] * r->xr[i]; }
    if (N <= EST_FULL_N)
    {
        r->Xr_r = (long double *)est_xmalloc((size_t)H * sizeof(long double));
        r->Xi_r = (long double *)est_xmalloc((size_t)H * sizeof(long double));
        est_dft_full(r->xr, 0, N, N, -1, H, r->Xr_r, r->Xi_r);
    }
    else
    {
        int j;
        r->nkr = est_samples(H, N, r->kr);
        for (j = 0; j < r->nkr; j++) est_dft_one(r->xr, 0, N, N, -1, r->kr[j], &r->Kr_r[j], &r->Ki_r[j]);
    }
    r->have_r = 1;
}
/* x[n] = 2 Re sum_{k=0}^{K} H_k e^{+2 pi i nk/N} - H_0 + [N even] (-1)^n H_{N/2}, K = (N-1)/2 */
static void est_ref_h(est_ref_t *r)
{
    const long N = r->N, H = N / 2 + 1, K = (N - 1) / 2;
    long i;
    if (r->have_h) return;
    r->hc = (double *)est_alloc((size_t)2 * H * sizeof(double));
    est_rng = 0xABCDEF1ull ^ ((unsigned long long)N << 20);
    for (i = 0; i < 2 * H; i++) r->hc[i] = est_rand();
    r->hc[1] = 0.0;                        /* the imaginary part of bin 0 and, even N, of bin N/2: outside */
    if (!(N & 1)) r->hc[2 * (N / 2) + 1] = 0.0;   /* the c2r contract (include/vfft.h:40-41) */
    r->eh = (long double)r->hc[0] * r->hc[0];
    for (i = 1; i <= K; i++) r->eh += 2.0L * ((long double)r->hc[2 * i] * r->hc[2 * i] + (long double)r->hc[2 * i + 1] * r->hc[2 * i + 1]);
    if (!(N & 1)) r->eh += (long double)r->hc[2 * (N / 2)] * r->hc[2 * (N / 2)];
    if (N <= EST_FULL_N)
    {
        long double *sr = (long double *)est_xmalloc((size_t)N * sizeof(long double));
        long double *si = (long double *)est_xmalloc((size_t)N * sizeof(long double));
        r->xh = (long double *)est_xmalloc((size_t)N * sizeof(long double));
        est_dft_full(r->hc, 1, K + 1, N, +1, N, sr, si);
        for (i = 0; i < N; i++)
            r->xh[i] = 2.0L * sr[i] - r->hc[0] + ((N & 1) ? 0.0L : ((i & 1) ? -1.0L : 1.0L) * r->hc[2 * (N / 2)]);
        free(sr);
        free(si);
    }
    else
    {
        int j;
        r->nnh = est_samples(N, N, r->nh);
        for (j = 0; j < r->nnh; j++)
        {
            long double sr, si;
            est_dft_one(r->hc, 1, K + 1, N, +1, r->nh[j], &sr, &si);
            r->Kh[j] = 2.0L * sr - r->hc[0] + ((N & 1) ? 0.0L : ((r->nh[j] & 1) ? -1.0L : 1.0L) * r->hc[2 * (N / 2)]);
        }
    }
    r->have_h = 1;
}

/* ================================================================ the checks */
typedef struct { double bins, par, rt; int has_bins, has_rt; } est_err_t;

static int est_cmp_cplx(const void *a, const void *b)
{
    const long double *x = (const long double *)a, *y = (const long double *)b;
    if (x[0] != y[0]) return x[0] < y[0] ? -1 : 1;
    return (x[1] > y[1]) - (x[1] < y[1]);
}
static double est_max_err_sorted(const double *y, const long double *Xr, const long double *Xi, long N)
{   /* the scrambled forward: the outputs are the reference's bins in some order -- compare the sorted multisets */
    long double *a = (long double *)malloc((size_t)N * 2 * sizeof(long double));
    long double *b = (long double *)malloc((size_t)N * 2 * sizeof(long double));
    long double m = 0.0L;
    long i;
    if (!a || !b) { fprintf(stderr, "out of memory\n"); exit(2); }
    for (i = 0; i < N; i++) { a[2 * i] = y[2 * i]; a[2 * i + 1] = y[2 * i + 1]; b[2 * i] = Xr[i]; b[2 * i + 1] = Xi[i]; }
    qsort(a, (size_t)N, 2 * sizeof(long double), est_cmp_cplx);
    qsort(b, (size_t)N, 2 * sizeof(long double), est_cmp_cplx);
    for (i = 0; i < N; i++)
    {
        const long double d = hypotl(a[2 * i] - b[2 * i], a[2 * i + 1] - b[2 * i + 1]);
        if (!(d <= m)) m = d;   /* a NaN wins */
    }
    free(a);
    free(b);
    return (double)m;
}
/* The scrambled forward above EST_FULL_N. The SCRAMBLED contract is any self-consistent
 * permutation of the bins (vfft.c:1922-1928), so a round trip cannot gate it -- a wrong forward
 * and its matching wrong backward pass that -- and the forward is gated instead: each sampled
 * reference bin must be one of the N outputs. Returns the largest distance from a sampled bin to
 * its nearest output. The scan runs in double (the outputs are doubles); the nearest is then
 * measured in long double. A NaN output is never nearest; Parseval and the round trip catch it. */
static double est_max_err_scr_sampled(const double *y, long N, const long double *Kr, const long double *Ki, int nk)
{
    double kr[EST_NBINS], ki[EST_NBINS], best[EST_NBINS];
    long at[EST_NBINS], i;
    long double m = 0.0L;
    int j;
    for (j = 0; j < nk; j++)
    {
        kr[j] = (double)Kr[j];
        ki[j] = (double)Ki[j];
        best[j] = HUGE_VAL;
        at[j] = -1;
    }
    for (i = 0; i < N; i++)
    {
        const double yr = y[2 * i], yi = y[2 * i + 1];
        for (j = 0; j < nk; j++)
        {
            const double dr = yr - kr[j], di = yi - ki[j], d = dr * dr + di * di;
            if (d < best[j]) { best[j] = d; at[j] = i; }
        }
    }
    for (j = 0; j < nk; j++)
    {
        long double d;
        if (at[j] < 0)
            return (double)NAN;   /* no output is a number: the forward is wrong */
        d = hypotl(y[2 * at[j]] - Kr[j], y[2 * at[j] + 1] - Ki[j]);
        if (!(d <= m)) m = d;
    }
    return (double)m;
}

/* c2c: the forward, Parseval, the round trip -- on the plan itself */
static void est_check_c2c(vfft_plan h, const est_request_t *rq, est_ref_t *r, est_err_t *e)
{
    const long N = rq->N;
    const size_t nb = (size_t)2 * N * sizeof(double);
    const int ip = rq->placement == VFFT_PLACEMENT_IN, scr = rq->order == VFFT_ORDER_SCRAMBLED;
    double *in = (double *)est_alloc(nb), *out = ip ? in : (double *)est_alloc(nb), *z = ip ? in : (double *)est_alloc(nb);
    double rms;
    long double m = 0.0L, sy = 0.0L;
    long i;
    est_ref_c(r);
    rms = sqrtl(r->sxc);   /* rms |X| (Parseval) */
    memcpy(in, r->xc, nb);
    vfft_execute(h, VFFT_FORWARD, in, out);
    for (i = 0; i < 2 * N; i++) sy += (long double)out[i] * out[i];
    e->par = (double)(fabsl(sy - (long double)N * r->sxc) / ((long double)N * r->sxc));
    e->has_bins = 1;
    if (N <= EST_FULL_N && !scr)
        for (i = 0; i < N; i++)
        {
            const long double d = hypotl(out[2 * i] - r->Xr_c[i], out[2 * i + 1] - r->Xi_c[i]);
            if (!(d <= m)) m = d;
        }
    else if (N <= EST_FULL_N)
        m = est_max_err_sorted(out, r->Xr_c, r->Xi_c, N);
    else if (!scr)
        for (i = 0; i < r->nkc; i++)
        {
            const long k = r->kc[i];
            const long double d = hypotl(out[2 * k] - r->Kr_c[i], out[2 * k + 1] - r->Ki_c[i]);
            if (!(d <= m)) m = d;
        }
    else   /* a scrambled forward above EST_FULL_N: each sampled bin found among the outputs */
        m = est_max_err_scr_sampled(out, N, r->Kr_c, r->Ki_c, r->nkc);
    e->bins = (double)(m / rms);
    vfft_execute(h, VFFT_BACKWARD, out, z);
    m = 0.0L;
    for (i = 0; i < 2 * N; i++)
    {
        const long double d = fabsl((long double)z[i] / N - r->xc[i]);
        if (!(d <= m)) m = d;
    }
    e->rt = (double)(m / sqrtl(r->sxc / N));
    e->has_rt = 1;
    vfft_aligned_free(in);
    if (!ip) { vfft_aligned_free(out); vfft_aligned_free(z); }
}

/* r2c: the forward and Parseval (the round trip is the c2r cell's) */
static void est_check_r2c(vfft_plan h, const est_request_t *rq, est_ref_t *r, est_err_t *e)
{
    const long N = rq->N, H = N / 2 + 1, K = (N - 1) / 2;
    const int ip = rq->placement == VFFT_PLACEMENT_IN;
    double *in = (double *)est_alloc((size_t)2 * H * sizeof(double));
    double *out = ip ? in : (double *)est_alloc((size_t)2 * H * sizeof(double));
    long double m = 0.0L, E;
    long i;
    est_ref_r(r);
    memcpy(in, r->xr, (size_t)N * sizeof(double));
    vfft_execute(h, VFFT_FORWARD, in, out);
    E = (long double)out[0] * out[0] + (long double)out[1] * out[1];
    for (i = 1; i <= K; i++) E += 2.0L * ((long double)out[2 * i] * out[2 * i] + (long double)out[2 * i + 1] * out[2 * i + 1]);
    if (!(N & 1)) E += (long double)out[2 * (N / 2)] * out[2 * (N / 2)] + (long double)out[2 * (N / 2) + 1] * out[2 * (N / 2) + 1];
    e->par = (double)(fabsl(E - (long double)N * r->sxr) / ((long double)N * r->sxr));
    if (N <= EST_FULL_N)
        for (i = 0; i < H; i++)
        {
            const long double d = hypotl(out[2 * i] - r->Xr_r[i], out[2 * i + 1] - r->Xi_r[i]);
            if (!(d <= m)) m = d;
        }
    else
        for (i = 0; i < r->nkr; i++)
        {
            const long k = r->kr[i];
            const long double d = hypotl(out[2 * k] - r->Kr_r[i], out[2 * k + 1] - r->Ki_r[i]);
            if (!(d <= m)) m = d;
        }
    e->bins = (double)(m / sqrtl(r->sxr));
    e->has_bins = 1;
    vfft_aligned_free(in);
    if (!ip) vfft_aligned_free(out);
}

/* c2r: the backward against the Hermitian reference and Parseval; with the r2c plan of the same cell
 * (hs, may be NULL), the round trip x -> r2c -> c2r -> N x */
static void est_check_c2r(vfft_plan h, vfft_plan hs, const est_request_t *rq, est_ref_t *r, est_err_t *e)
{
    const long N = rq->N, H = N / 2 + 1;
    const int ip = rq->placement == VFFT_PLACEMENT_IN;
    const size_t hb = (size_t)2 * H * sizeof(double);
    double *in = (double *)est_alloc(hb), *out = ip ? in : (double *)est_alloc(hb);
    long double m = 0.0L, sx = 0.0L;
    long i;
    est_ref_h(r);
    memcpy(in, r->hc, hb);
    vfft_execute(h, VFFT_BACKWARD, in, out);
    for (i = 0; i < N; i++) sx += (long double)out[i] * out[i];
    e->par = (double)(fabsl(sx - (long double)N * r->eh) / ((long double)N * r->eh));
    if (N <= EST_FULL_N)
        for (i = 0; i < N; i++)
        {
            const long double d = fabsl(out[i] - r->xh[i]);
            if (!(d <= m)) m = d;
        }
    else
        for (i = 0; i < r->nnh; i++)
        {
            const long double d = fabsl(out[r->nh[i]] - r->Kh[i]);
            if (!(d <= m)) m = d;
        }
    e->bins = (double)(m / sqrtl(r->eh));
    e->has_bins = 1;
    if (hs)
    {   /* x -> r2c -> c2r: N x */
        double *a = (double *)est_alloc(hb), *b = ip ? a : (double *)est_alloc(hb), *c = ip ? a : (double *)est_alloc(hb);
        est_ref_r(r);
        memcpy(a, r->xr, (size_t)N * sizeof(double));
        vfft_execute(hs, VFFT_FORWARD, a, b);
        vfft_execute(h, VFFT_BACKWARD, b, c);
        m = 0.0L;
        for (i = 0; i < N; i++)
        {
            const long double d = fabsl((long double)c[i] / N - r->xr[i]);
            if (!(d <= m)) m = d;
        }
        e->rt = (double)(m / sqrtl(r->sxr / N));
        e->has_rt = 1;
        vfft_aligned_free(a);
        if (!ip) { vfft_aligned_free(b); vfft_aligned_free(c); }
    }
    vfft_aligned_free(in);
    if (!ip) vfft_aligned_free(out);
}

/* the recipe in the store's own text (est_recipe_print, snprintf's contract), one row per line */
static void est_print_recipe(const est_any_recipe_t *rc, const est_request_t *rq)
{
    char small[4096], *buf = small;
    const int n = est_recipe_print(rc, rq, small, sizeof small);
    if (n < 0)
        return;
    if ((size_t)n >= sizeof small && (buf = (char *)malloc((size_t)n + 1)) != NULL)
        est_recipe_print(rc, rq, buf, (size_t)n + 1);
    if (!buf)
        buf = small;
    printf("%s%s", buf, (buf[0] && buf[strlen(buf) - 1] == '\n') ? "" : "\n");
    if (buf != small)
        free(buf);
}

/* ================================================================ one check cell */
typedef struct
{
    int cells, ok, wrong, ref_formula, ref_glue, measured, crashed, rewritten;
    double worst;
    char worst_cell[192];
} est_tally_t;

enum { EST_RES_OK, EST_RES_WRONG, EST_RES_REF_FORMULA, EST_RES_REF_GLUE, EST_RES_MEASURED, EST_RES_REWRITTEN,
       EST_RES_CRASHED, EST_RES_N };
static const char *const est_rname[EST_RES_N] = { "ok", "wrong", "refused_formula", "refused_glue", "measured",
                                                  "rewritten", "crashed" };

static void est_tally(est_tally_t *t, int res, double err, const char *cell)
{
    t->cells++;
    switch (res)
    {
    case EST_RES_OK: t->ok++; break;
    case EST_RES_WRONG: t->wrong++; break;
    case EST_RES_REF_FORMULA: t->ref_formula++; break;
    case EST_RES_REF_GLUE: t->ref_glue++; break;
    case EST_RES_MEASURED: t->measured++; break;
    case EST_RES_REWRITTEN: t->rewritten++; break;
    default: t->crashed++; break;
    }
    if ((res == EST_RES_OK || res == EST_RES_WRONG) && !(err <= t->worst))
    {
        t->worst = err;
        snprintf(t->worst_cell, sizeof t->worst_cell, "%s", cell);
    }
}

static void est_fmt_err(char *b, size_t n, const est_err_t *e)
{
    char x[32], y[32];
    if (e->has_bins) snprintf(x, sizeof x, "%.1e", e->bins); else snprintf(x, sizeof x, "-");
    if (e->has_rt) snprintf(y, sizeof y, "%.1e", e->rt); else snprintf(y, sizeof y, "-");
    snprintf(b, n, "bins %s par %.1e rt %s", x, e->par, y);
}

/* the r2c plan of a c2r cell, for its round trip; NULL when its recipe refuses or holds a nested
 * create (that one is checked in its own cell) */
static vfft_plan est_sibling(const est_machine_t *m, const est_request_t *rq)
{
    est_request_t sq = *rq;
    est_any_recipe_t sr;
    char why[512];
    vfft_plan h;
    sq.transform = VFFT_R2C;
    if (est_plan_any(m, &sq, &sr, why, sizeof why) || est_recipe_nested(&sr, &sq))
        return NULL;
    est_cell_name(est_cell, sizeof est_cell, &sq);
    strncat(est_cell, " (sibling)", sizeof est_cell - strlen(est_cell) - 1);
    est_clock_trap = 1;
    h = (vfft_plan)est_build_any(&sr, &sq, why, sizeof why);
    est_clock_trap = 0;
    return h;
}

/* build and check one cell in this process; prints its line; returns EST_RES_* and the worst error */
static int est_check_one(const est_opt_t *o, const est_machine_t *m, const est_request_t *rq, est_ref_t *ref,
                         double *worst)
{
    est_any_recipe_t rc;
    est_build_report_t rep;
    char why[768], cell[192], brief[256], eb[128];
    vfft_plan h;
    est_err_t e;
    int res;
    *worst = 0.0;
    est_cell_name(cell, sizeof cell, rq);
    if (est_plan_any(m, rq, &rc, why, sizeof why))
    {
        if (!o->quiet || o->verbose)
            printf("%-30s refused(formula: %s)\n", cell, why);
        return EST_RES_REF_FORMULA;
    }
    est_brief(brief, sizeof brief, &rc, rq);
    snprintf(est_cell, sizeof est_cell, "%s", cell);
    est_clock_trap = 1;
    h = (vfft_plan)est_build_any_ex(&rc, rq, why, sizeof why, &rep);
    est_clock_trap = 0;
    if (!h)
    {
        if (rep.status == EST_MEASURED)
        {
            printf("%-30s measured(FAIL): %s  [%s]\n", cell, why, brief);
            return EST_RES_MEASURED;
        }
        if (rep.status == EST_REWRITTEN)
        {   /* nothing measured, but the door replaced the recipe: a failure of the recipe, not a refusal */
            printf("%-30s rewritten(FAIL): %s  [%s]\n", cell, why, brief);
            return EST_RES_REWRITTEN;
        }
        printf("%-30s refused(glue: %s)  [%s]\n", cell, why, brief);
        return EST_RES_REF_GLUE;
    }
    memset(&e, 0, sizeof e);
    if (rq->transform == VFFT_C2C)
        est_check_c2c(h, rq, ref, &e);
    else if (rq->transform == VFFT_R2C)
        est_check_r2c(h, rq, ref, &e);
    else
    {
        vfft_plan hs = est_sibling(m, rq);
        est_check_c2r(h, hs, rq, ref, &e);
        vfft_destroy(hs);
    }
    vfft_destroy(h);
    *worst = e.par;   /* the largest error, a NaN anywhere winning (a NaN Parseval stays NaN) */
    if (e.has_bins && !isnan(*worst) && !(e.bins <= *worst)) *worst = e.bins;
    if (e.has_rt && !isnan(*worst) && !(e.rt <= *worst)) *worst = e.rt;
    res = (*worst <= o->tol) ? EST_RES_OK : EST_RES_WRONG;
    est_fmt_err(eb, sizeof eb, &e);
    if (res == EST_RES_WRONG)
        printf("%-30s wrong(FAIL): %s > tol %.0e  [%s]\n", cell, eb, o->tol, brief);
    else if (!o->quiet)
        printf("%-30s built  %-44s %s\n", cell, brief, eb);
    if (rep.rewritten && (o->verbose || !o->quiet))
        printf("%-30s   note: the door re-banked %d row(s) with the same payload, measuring nothing (first: %s)\n", "",
               rep.rewritten, rep.rewritten_key);
    if (o->verbose)
        est_print_recipe(&rc, rq);
    return res;
}

/* the same cell in its own process: a race there ends only that process */
static int est_check_isolated(const est_opt_t *o, const est_request_t *rq, double *worst)
{
    char self[1024], cmd[1600], line[4096], cell[192];
    FILE *f;
    int res = -1, rc;
    *worst = 0.0;
    est_cell_name(cell, sizeof cell, rq);
#ifdef _WIN32
    if (!GetModuleFileNameA(NULL, self, sizeof self)) { printf("%-30s crashed(FAIL): cannot name this program\n", cell); return EST_RES_CRASHED; }
#else
    {
        const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
        if (n <= 0) { printf("%-30s crashed(FAIL): cannot name this program\n", cell); return EST_RES_CRASHED; }
        self[n] = 0;
    }
#endif
    snprintf(cmd, sizeof cmd, "%s%s%s cell %d %s %s %s %d --tol %g%s%s%s", strchr(self, ' ') ? "\"\"" : "", self,
             strchr(self, ' ') ? "\"" : "", rq->N, est_tname(rq->transform), rq->order == VFFT_ORDER_SCRAMBLED ? "scr" : "nat",
             rq->placement == VFFT_PLACEMENT_IN ? "ip" : "oop", rq->nthreads > 1 ? rq->nthreads : 1, o->tol,
             o->quiet ? " -q" : "", o->verbose ? " -v" : "", strchr(self, ' ') ? "\"" : "");
    fflush(stdout);
#ifdef _WIN32
    f = _popen(cmd, "r");
#else
    f = popen(cmd, "r");
#endif
    if (!f) { printf("%-30s crashed(FAIL): cannot start the isolated cell\n", cell); return EST_RES_CRASHED; }
    while (fgets(line, sizeof line, f))
    {
        if (!strncmp(line, "@result ", 8))
        {
            char st[32];
            int k;
            if (sscanf(line + 8, "%31s %lf", st, worst) >= 1)
                for (k = 0; k < EST_RES_N; k++)
                    if (!strcmp(st, est_rname[k])) res = k;
            continue;
        }
        fputs(line, stdout);
    }
#ifdef _WIN32
    rc = _pclose(f);
#else
    rc = pclose(f);
#endif
    if (res < 0)
    {
        printf("%-30s crashed(FAIL): the isolated cell ended without a result (exit %d)\n", cell, rc);
        return EST_RES_CRASHED;
    }
    return res;
}

/* ================================================================ check */
static void est_requests_of(const est_opt_t *o, int N, int T, int pl, est_request_t *rqs, int *n)
{
    int t, od;
    *n = 0;
    for (t = 0; t < 3; t++)
    {
        if (!o->tr[t]) continue;
        for (od = 0; od < 2; od++)
        {
            if (!o->ord[od]) continue;
            if (t > 0 && od == 1) continue;   /* real is natural only (vfft.c:1940-1957) */
            rqs[*n].N = N;
            rqs[*n].transform = t;            /* VFFT_C2C 0, VFFT_R2C 1, VFFT_C2R 2 (include/vfft.h:61-73) */
            rqs[*n].order = od;               /* VFFT_ORDER_NATURAL 0, SCRAMBLED 1 */
            rqs[*n].placement = pl;           /* VFFT_PLACEMENT_OUT 0, IN 1 */
            rqs[*n].nthreads = T;
            (*n)++;
        }
    }
}

static int est_check(const est_opt_t *o, const est_machine_t *m)
{
    est_tally_t tal;
    est_ref_t ref;
    int i, ti, pl, q, nq;
    est_request_t rqs[8];
    memset(&tal, 0, sizeof tal);
    memset(&ref, 0, sizeof ref);
    for (i = 0; i < o->nN; i++)
    {
        est_ref_free(&ref);
        ref.N = o->Ns[i];
        for (ti = 0; ti < o->nT; ti++)
            for (pl = 0; pl < 2; pl++)
            {
                if (!o->pl[pl]) continue;
                est_requests_of(o, o->Ns[i], o->Ts[ti], pl, rqs, &nq);
                for (q = 0; q < nq; q++)
                {
                    est_any_recipe_t rc;
                    char why[512], cell[192];
                    double w = 0.0;
                    int res;
                    est_cell_name(cell, sizeof cell, &rqs[q]);
                    if (o->mode != 'x' && (o->isolate ||
                        (!est_plan_any(m, &rqs[q], &rc, why, sizeof why) && est_recipe_nested(&rc, &rqs[q]))))
                        res = est_check_isolated(o, &rqs[q], &w);
                    else
                        res = est_check_one(o, m, &rqs[q], &ref, &w);
                    if (o->mode == 'x')
                        printf("@result %s %.3e\n", est_rname[res], w);
                    fflush(stdout);
                    est_tally(&tal, res, w, cell);
                }
            }
    }
    est_ref_free(&ref);
    if (o->mode == 'x')
        return 0;
    printf("\ncells %d: built %d (correct %d, wrong %d), refused %d (formulas %d, glue %d), measured %d, rewritten %d, "
           "crashed %d\n", tal.cells, tal.ok + tal.wrong, tal.ok, tal.wrong, tal.ref_formula + tal.ref_glue,
           tal.ref_formula, tal.ref_glue, tal.measured, tal.rewritten, tal.crashed);
    if (tal.ok + tal.wrong)
        printf("worst relative error %.2e (%s), tolerance %.0e\n", tal.worst, tal.worst_cell, o->tol);
    printf("=== %s ===\n", (tal.wrong || tal.measured || tal.rewritten || tal.crashed) ? "*** FAIL ***" : "PASS");
    return (tal.wrong || tal.measured || tal.rewritten || tal.crashed) ? 1 : 0;
}

/* ================================================================ show */
static int est_show(const est_opt_t *o, const est_machine_t *m)
{
    int i, ti, pl, q, nq;
    est_request_t rqs[8];
    for (i = 0; i < o->nN; i++)
        for (ti = 0; ti < o->nT; ti++)
            for (pl = 0; pl < 2; pl++)
            {
                if (!o->pl[pl]) continue;
                est_requests_of(o, o->Ns[i], o->Ts[ti], pl, rqs, &nq);
                for (q = 0; q < nq; q++)
                {
                    est_any_recipe_t rc;
                    char why[512], cell[192];
                    est_cell_name(cell, sizeof cell, &rqs[q]);
                    if (est_plan_any(m, &rqs[q], &rc, why, sizeof why))
                    {
                        printf("%s: refused: %s\n", cell, why);
                        continue;
                    }
                    printf("%s:\n", cell);
                    est_print_recipe(&rc, &rqs[q]);
                }
            }
    return 0;
}

/* ================================================================ compare */
/* The store folder, READ ONLY: each shard file is opened "rb" and parsed by the library's own
 * loader (vw2__load_shard, wisdom2.h:830-912). Not vw2_open: its open sweeps stale tmp files and
 * may take the store lock (wisdom2.h:1101-1143). Nothing is ever written. */
static int est_store_load(const char *dir, vw2_store_t *s)
{
    int i, any = 0;
    memset(s, 0, sizeof *s);
    if (strlen(dir) >= sizeof s->dir)
        return -1;   /* the store's own path buffer (vw2_store_t.dir): a cut path would name another folder */
    snprintf(s->dir, sizeof s->dir, "%s", dir);
    s->writable = 0;
    for (i = 0; i < VW2_NSHARDS; i++)
    {
        (void)vw2__load_shard(s, i);
        any |= s->present[i];
    }
    (void)vw2__migrate_nthreads(s);   /* pre-1.3 rows split in memory, as the library reads them (wisdom2.h:931) */
    return any ? 0 : -1;
}
static const vw2_rec_t *est_store_exact(const vw2_store_t *s, const vw2_key_t *k)
{
    int i;
    for (i = 0; i < s->nrec; i++)
        if (vw2_key_eq(&s->rec[i].key, k) && !vw2__is_seed(&s->rec[i]))
            return &s->rec[i];
    return NULL;
}
/* this CPU's folder under the library's store root (wisdom2_folders.h:105-150, read only) */
static int est_store_dir(const est_opt_t *o, char *dir, size_t n)
{
    char name[256];
    if (o->store) { snprintf(dir, n, "%s", o->store); return 0; }
#ifdef VFFT_WISDOM_DIR_DEFAULT
    if (vw2__folder_find(VFFT_WISDOM_DIR_DEFAULT, vfft_cpu_identity(), name, sizeof name))
    {
        snprintf(dir, n, "%s/%s", VFFT_WISDOM_DIR_DEFAULT, name);
        return 0;
    }
#endif
    return -1;
}
/* the machine a folder's rows were raced on: its @meta stamp (cpu_identity.h:41-56) */
static void est_machine_of_stamp(const char *stamp, est_machine_t *m)
{
    const char *p;
    if ((p = strstr(stamp, "l1d="))) m->l1d = atol(p + 4);
    if ((p = strstr(stamp, "l2="))) m->l2 = atol(p + 3);
    if ((p = strstr(stamp, "l3="))) m->l3 = atol(p + 3);
    if ((p = strstr(stamp, "isa="))) m->lanes = !strncmp(p + 4, "avx512", 6) ? 8 : 4;
    if ((p = strstr(stamp, "pcores="))) m->cores = atoi(p + 7);
    if ((p = strstr(stamp, "ecores="))) m->cores += atoi(p + 7);
}

/* a payload token's comparable name: the four-step child's row plan under fs_row_ (the shipped
 * spelling) or fs_rp_<i>_ (the 2D tier's, wisdom2_child.h:5-21) read as one name, its key tokens
 * dropped; a measurement inside the payload (fs_mtns) is no recipe */
static int est_canon(const vw2_rec_t *r, char ***out)
{
    char role[16];
    int i, n = 0;
    char **v = (char **)est_xmalloc(((size_t)r->ntok + 1) * sizeof *v);
    memset(role, 'X', sizeof role);
    for (i = 0; i < r->ntok; i++)
    {
        const char *nm = r->tok[i].name;
        int k;
        if (r->tok[i].sect == 1 && !strncmp(nm, "fs_rp_k", 7) && (k = atoi(nm + 7)) > 0 && k < 16)
            role[k] = strstr(r->tok[i].val, "dir=bwd") ? 'B' : 'R';
    }
    for (i = 0; i < r->ntok; i++)
    {
        const char *nm = r->tok[i].name, *rest = NULL;
        char c = 0, buf[512];
        int k;
        if (r->tok[i].sect != 1 || !strncmp(nm, "fs_rp_k", 7) || !strcmp(nm, "fs_mtns"))
            continue;
        if (!strncmp(nm, "fs_row_bwd_", 11)) { c = 'B'; rest = nm + 11; }
        else if (!strncmp(nm, "fs_row_", 7)) { c = 'R'; rest = nm + 7; }
        else if (!strncmp(nm, "fs_rp_", 6) && (k = atoi(nm + 6)) > 0 && k < 16 && strchr(nm + 6, '_'))
        { c = role[k]; rest = strchr(nm + 6, '_') + 1; }
        if (c) snprintf(buf, sizeof buf, "fs_row%c_%s=%s", c, rest, r->tok[i].val);
        else snprintf(buf, sizeof buf, "%s=%s", nm, r->tok[i].val);
        v[n] = (char *)est_xmalloc(strlen(buf) + 1);
        memcpy(v[n++], buf, strlen(buf) + 1);
    }
    for (i = 1; i < n; i++)
    {   /* sorted: the comparison is of the sets */
        int j;
        for (j = i; j > 0 && strcmp(v[j], v[j - 1]) < 0; j--) { char *t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
    }
    *out = v;
    return n;
}
static int est_payload_eq(const vw2_rec_t *a, const vw2_rec_t *b, char *diff, size_t dn)
{
    char **x, **y;
    const int nx = est_canon(a, &x), ny = est_canon(b, &y);
    int i = 0, j = 0, eq = 1;
    diff[0] = 0;
    while (i < nx || j < ny)
    {
        const int c = (i < nx && j < ny) ? strcmp(x[i], y[j]) : (i < nx ? -1 : 1);
        if (c == 0) { i++; j++; continue; }
        if (eq) snprintf(diff, dn, "%s%s%s", c < 0 ? "estimate " : "stored ", c < 0 ? x[i] : y[j], c < 0 ? " (not stored)" : " (not estimated)");
        eq = 0;
        if (c < 0) i++; else j++;
    }
    for (i = 0; i < nx; i++) free(x[i]);
    for (j = 0; j < ny; j++) free(y[j]);
    free(x);
    free(y);
    return eq;
}
static void est_payload_str(const vw2_rec_t *r, char *b, size_t n)
{
    size_t off = 0;
    int i;
    b[0] = 0;
    for (i = 0; i < r->ntok && off + 2 < n; i++)
        if (r->tok[i].sect == 1)
        {
            const int w = snprintf(b + off, n - off, "%s%s=%s", off ? " " : "", r->tok[i].name, r->tok[i].val);
            if (w < 0) break;
            off += (size_t)w < n - off ? (size_t)w : n - off - 1;
        }
}

typedef struct { char name[24]; int cells, route, recipe; } est_rt_t;
static void est_rt_add(est_rt_t *t, int *nt, const char *name, int route, int recipe)
{
    int i;
    for (i = 0; i < *nt; i++)
        if (!strcmp(t[i].name, name)) break;
    if (i == *nt)
    {
        if (*nt >= 64) return;
        memset(&t[i], 0, sizeof t[i]);
        snprintf(t[i].name, sizeof t[i].name, "%s", name);
        (*nt)++;
    }
    t[i].cells++;
    t[i].route += route;
    t[i].recipe += recipe;
}

static int est_cell_cmp(const void *a, const void *b)
{
    const est_request_t *x = (const est_request_t *)a, *y = (const est_request_t *)b;
    if (x->transform != y->transform) return x->transform - y->transform;
    if (x->N != y->N) return x->N < y->N ? -1 : 1;
    if (x->nthreads != y->nthreads) return x->nthreads - y->nthreads;
    if (x->placement != y->placement) return x->placement - y->placement;
    return x->order - y->order;
}

static int est_compare(const est_opt_t *o, est_machine_t m)
{
    char dir[1024], stamp[256];
    vw2_store_t st;
    est_request_t *cells = NULL;
    int nc = 0, cap = 0, i;
    int n_ref = 0, n_unw = 0, n_none = 0, n_stored = 0, n_route = 0, n_recipe = 0, nrt = 0;
    est_rt_t rt[64];
    if (est_store_dir(o, dir, sizeof dir))
    {
        printf("no store folder under %s carries this CPU's identity (%s): pass --store DIR\n",
#ifdef VFFT_WISDOM_DIR_DEFAULT
               VFFT_WISDOM_DIR_DEFAULT,
#else
               "(no compiled store root)",
#endif
               vfft_cpu_identity());
        return 2;
    }
    if (est_store_load(dir, &st))
    {
        printf("no shard file under %s\n", dir);
        return 2;
    }
    (void)vw2_folder_stamp(dir, stamp, sizeof stamp);
    if (stamp[0])
    {   /* the formulas plan for the machine the rows were raced on, unless told otherwise */
        est_machine_of_stamp(stamp, &m);
        if (o->l1d) m.l1d = o->l1d;
        if (o->l2) m.l2 = o->l2;
        if (o->l3) m.l3 = o->l3;
        if (o->lanes) m.lanes = o->lanes;
    }
    printf("store: %s (read only, %d rows)\nstamp: %s\n", dir, st.nrec, stamp[0] ? stamp : "(none)");
    est_machine_print("planned for", &m);
    /* the cells: the N given, else every stored winner row of the transforms asked */
    for (i = 0; i < (o->nN ? o->nN : 0); i++)
    {
        int ti, pl, q, nq;
        est_request_t rqs[8];
        for (ti = 0; ti < o->nT; ti++)
            for (pl = 0; pl < 2; pl++)
            {
                if (!o->pl[pl]) continue;
                est_requests_of(o, o->Ns[i], o->Ts[ti], pl, rqs, &nq);
                for (q = 0; q < nq; q++)
                {
                    if (nc == cap) { cap = cap ? 2 * cap : 1024; cells = (est_request_t *)est_xrealloc(cells, (size_t)cap * sizeof *cells); }
                    cells[nc++] = rqs[q];
                }
            }
    }
    if (!o->nN)
        for (i = 0; i < st.nrec; i++)
        {
            const vw2_rec_t *r = &st.rec[i];
            const char *eng = vw2_rec_get(r, "eng");
            est_request_t rq;
            int t, T, k, keep = 0;
            if (r->key.rank != 1 || r->key.q != 1 || r->key.dir != VW2_DIR_NONE || !eng || vw2__is_seed(r))
                continue;
            if (r->key.t == VW2_T_C2C)
                t = VFFT_C2C, keep = r->key.lay == VW2_LAY_IL && r->key.role == VW2_ROLE_COMP && !strcmp(eng, "k1");
            else if (r->key.t == VW2_T_R2C || r->key.t == VW2_T_C2R)
            {
                static const char *const reng[] = { "zrm", "zrp", "zttr", "zr2c", "zfsr", "zrf", "zrb", NULL };
                t = r->key.t == VW2_T_R2C ? VFFT_R2C : VFFT_C2R;
                for (k = 0; reng[k]; k++) if (!strcmp(eng, reng[k])) keep = r->key.ord == VW2_ORD_NAT;
            }
            else
                continue;
            if (!keep || !o->tr[t] || r->key.ord < 0 || r->key.pl < 0) continue;
            if (!o->ord[r->key.ord == VW2_ORD_SCR] || !o->pl[r->key.pl == VW2_PL_IP]) continue;
            T = VW2__NT(&r->key);
            if (o->Tgiven)
            {
                int ok = 0;
                for (k = 0; k < o->nT; k++) if ((o->Ts[k] > 1 ? o->Ts[k] : 1) == T) ok = 1;
                if (!ok) continue;
            }
            rq.N = r->key.n[0];
            rq.transform = t;
            rq.order = r->key.ord == VW2_ORD_SCR;
            rq.placement = r->key.pl == VW2_PL_IP;
            rq.nthreads = T;
            if (nc == cap) { cap = cap ? 2 * cap : 1024; cells = (est_request_t *)est_xrealloc(cells, (size_t)cap * sizeof *cells); }
            cells[nc++] = rq;
        }
    if (nc > 1)
    {
        int k = 1;
        qsort(cells, (size_t)nc, sizeof *cells, est_cell_cmp);
        for (i = 1; i < nc; i++) if (est_cell_cmp(&cells[i], &cells[k - 1])) cells[k++] = cells[i];
        nc = k;
    }
    for (i = 0; i < nc; i++)
    {
        const est_request_t *rq = &cells[i];
        est_any_recipe_t rc;
        struct vfft_wisdom_s *E;
        char why[512], cell[192], diff[600];
        vw2_key_t mk;
        const vw2_rec_t *er, *sr;
        const char *rn_e, *rn_s;
        int j, route_ok, recipe_ok = 1;
        est_cell_name(cell, sizeof cell, rq);
        if (est_plan_any(&m, rq, &rc, why, sizeof why))
        {
            n_ref++;
            if (o->verbose) printf("%-30s refused: %s\n", cell, why);
            continue;
        }
        E = vfft_child_store_new();   /* memory only: the rows the glue would bank, in the recipe's spelling */
        if (!E || est__put_rows_any(&E->vw2, &rc, rq, 0, why, sizeof why))
        {
            n_unw++;
            if (o->verbose || !o->quiet) printf("%-30s unwritable: %s\n", cell, why);
            vfft_child_store_free(E);
            continue;
        }
        if (rq->transform == VFFT_C2C)
        {
            er = vw2__oop_k1_scan_pl(&E->vw2, rq->N, VW2_LAY_IL, rq->order == VFFT_ORDER_SCRAMBLED,
                                     rq->placement == VFFT_PLACEMENT_IN ? VW2_PL_IP : VW2_PL_OOP, est__T(rq));
            if (!er) { n_unw++; vfft_child_store_free(E); continue; }
            mk = er->key;
        }
        else
        {
            vw2_real_il_key(&mk, rq->N, rq->transform == VFFT_C2R, rq->placement == VFFT_PLACEMENT_IN, est__T(rq));
            er = est_store_exact(&E->vw2, &mk);
        }
        sr = est_store_exact(&st, &mk);
        if (!sr || !er)
        {
            n_none++;
            if (o->verbose) printf("%-30s no stored row\n", cell);
            vfft_child_store_free(E);
            continue;
        }
        n_stored++;
        rn_e = vw2_rec_get(er, rq->transform == VFFT_C2C ? "il_route" : "eng");
        rn_s = vw2_rec_get(sr, rq->transform == VFFT_C2C ? "il_route" : "eng");
        route_ok = rn_e && rn_s && !strcmp(rn_e, rn_s);
        diff[0] = 0;
        for (j = 0; j < E->vw2.nrec; j++)
        {   /* every row the glue banks against the stored row at its key */
            const vw2_rec_t *s2 = est_store_exact(&st, &E->vw2.rec[j].key);
            char d2[600];
            if (!s2)
            {
                if (recipe_ok) vw2__key_format(&E->vw2.rec[j].key, diff, sizeof diff), strncat(diff, ": no stored row", sizeof diff - strlen(diff) - 1);
                recipe_ok = 0;
            }
            else if (!est_payload_eq(&E->vw2.rec[j], s2, d2, sizeof d2))
            {
                if (recipe_ok) snprintf(diff, sizeof diff, "%s", d2);
                recipe_ok = 0;
            }
        }
        n_route += route_ok;
        n_recipe += recipe_ok;
        {
            char nm[24];
            snprintf(nm, sizeof nm, "%s %s", est_tname(rq->transform), rn_s ? rn_s : "?");
            est_rt_add(rt, &nrt, nm, route_ok, recipe_ok);
        }
        if (o->verbose || (!o->quiet && !route_ok))
        {
            char pe[2048], ps[2048];
            est_payload_str(er, pe, sizeof pe);
            est_payload_str(sr, ps, sizeof ps);
            printf("%-30s route %s, recipe %s\n    estimate: %s\n    stored:   %s\n", cell, route_ok ? "match" : "DIFFERS",
                   recipe_ok ? "match" : "differs", pe, ps);
            if (!recipe_ok) printf("    first difference: %s\n", diff);
        }
        vfft_child_store_free(E);
    }
    printf("\ncells %d: refused by the formulas %d, unwritable %d, no stored row %d, compared %d\n", nc, n_ref, n_unw,
           n_none, n_stored);
    if (n_stored)
        printf("route match %d / %d (%.1f%%), recipe match %d / %d (%.1f%%)\n", n_route, n_stored,
               100.0 * n_route / n_stored, n_recipe, n_stored, 100.0 * n_recipe / n_stored);
    if (nrt)
    {
        printf("\n%-20s %7s %7s %7s\n", "stored route", "cells", "route", "recipe");
        for (i = 0; i < nrt; i++)
            printf("%-20s %7d %7d %7d\n", rt[i].name, rt[i].cells, rt[i].route, rt[i].recipe);
    }
    free(cells);
    vw2_close(&st);
    return 0;
}

/* ================================================================ main */
int main(int argc, char **argv)
{
    est_opt_t o;
    est_machine_t m;
    const char *env;
    int rc;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (est_parse(argc, argv, &o) || (o.mode != 'c' && !o.nN))
    {
        est_usage();
        return 2;
    }
    if ((env = est_env_steering()) != NULL)
    {   /* REPLAY_API section 1.3: a create-steering variable changes what a create builds */
        printf("%s is set: it steers what a create builds -- unset every VFFT_* variable but the logging ones\n", env);
        return 2;
    }
    m = est_machine_here();
    est_child = (o.mode == 'x');
    if (o.mode == 's' || o.mode == 'c')
    {
        if (o.l1d) m.l1d = o.l1d;
        if (o.l2) m.l2 = o.l2;
        if (o.l3) m.l3 = o.l3;
        if (o.lanes) m.lanes = o.lanes;
    }
    else if (o.l1d || o.l2 || o.l3 || (o.lanes && o.lanes != VFFT_IL_VW))
    {   /* a plan is built on this machine: its recipe must be this machine's (estimate_recipe.h:183-185) */
        printf("check builds plans on this machine: the machine overrides apply to show and compare only\n");
        return 2;
    }
    if (o.mode != 'x')
        est_machine_print("machine", &m);
    if (o.mode == 's')
        rc = est_show(&o, &m);
    else if (o.mode == 'c')
        rc = est_compare(&o, m);
    else
        rc = est_check(&o, &m);
    free(o.Ns);
    return rc;
}
