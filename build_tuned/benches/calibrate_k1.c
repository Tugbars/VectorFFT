/* calibrate_k1.c — the K=1 kind-3 FOUR-AXIS calibrator, v2 (2026-08-06).
 *
 * Per cell N: measure the SPLIT four-step race (route × pair, this file) and
 * the IL race (route × pair × blocked-form kv, delegated WHOLE to
 * dp_planner_il.h's vfft_il_dp_plan_and_bank — the shipped measured search),
 * and bank the kind-3 line through the SHIPPED writer via a load-merge-save
 * (replace-or-append by (N, K=1, kind), same dedup class as vfft.c's saver,
 * so the rest of the file survives byte-for-byte modulo canonical
 * formatting). plan_and_bank also refreshes the cell's kind-4 SCRAMBLED
 * verdict as a side effect — t2q-class picks are placement-luck sized and
 * want re-measuring per binary anyway.
 *
 * v1 of this file was deleted in 0e6e105d and is NOT restored verbatim, on
 * purpose: it hand-rolled the IL arms (predating dp_planner_il), hand-printed
 * the wisdom line (the two-places-know-the-format bug), timed candidates with
 * NO correctness gate, and never reseeded its in-place buffers (the
 * magnitude-amplification hazard the vfft.c ordering race documents). All
 * four are fixed here. What v1 got right is kept: the split route table, the
 * order-rotated trial discipline, and banking the SPLIT-IP axis winner as
 * sp_route (the oop/ip execute calls are route-identical, so one code serves
 * both — v1's contract, preserved).
 *
 * Methodology: ONE process per cell batch; per-candidate gate (naive DFT,
 * 1e-9) BEFORE any timing — a candidate that fails is reported and dropped,
 * never silently timed; 10 warmup + reps hot loop; trials with 32MB
 * cachebust between; candidate order ROTATED per trial; best-of-trials.
 * In-place candidates reseed from the pristine input EVERY burst. Pin
 * core 2, HIGH priority.
 *
 * Usage: calibrate_k1.exe <wisdir> <rigor 0|1> [N...]
 *   rigor 0 = 3 trials (smoke) · 1 = 5 trials (the real run)
 *   cells default to 128 256 512 1024 2048 4096 (the BAILEY2V band)
 *   🔴 <wisdir> is WRITTEN — run against a SCRATCH COPY, promote after gates.
 * Build: python build.py --src benches/calibrate_k1.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#include "il2p.h"
#include "dp_planner_il.h"   /* vfft_il_dp_plan_and_bank + ctx             */
#include "oop_wisdom.h"      /* shipped reader/writer for the merge        */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* QPC directly (v1's choice, same reason): vfft_proto_now_ns is a C99
 * inline whose external definition is not guaranteed in this TU. */
static double now_ns_(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9 * ts.tv_sec + ts.tv_nsec;
#endif
}

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
        exit(1);
    return p;
}
static void cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a;
    free(j);
}

/* split-side routes (v1's table minus the retired arms: the hand-gated JIT
 * and stride-spec twins are plan-time concerns of the FRONT DOOR now, and
 * the IL arms live in dp_planner_il). */
enum { R_3P = 0, R_3P_IP, R_2PA_IP, R_2PB_IP, R_TWL_IP,
       R_3PL3_IP, R_2PAL3_IP, R_MONO, R_MONO_ALT, R_CCOL, R_NROUTES };
static const char *RNAME[R_NROUTES] = {
    "3p", "3p-ip", "2pa-ip", "2pb-ip", "twl-ip",
    "3p-l3-ip", "2pa-l3-ip", "mono", "mono-alt", "cc" };
/* axis: 0 = split-oop, 1 = split-ip (the banked sp_route axis) */
static const int RAXIS[R_NROUTES] = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const int SPMAP[R_NROUTES] = {
    VFFT_K1_SP_3P, VFFT_K1_SP_3P, VFFT_K1_SP_2PA, VFFT_K1_SP_2PB,
    VFFT_K1_SP_TWL, VFFT_K1_SP_3P_L3, VFFT_K1_SP_2PA_L3,
    VFFT_K1_SP_MONO, VFFT_K1_SP_MONO, VFFT_K1_SP_CCOL };

typedef struct {
    int route, R1, R2, cc_code;
    vfft_oop_plan_t *p;
    double best;
    int gated; /* 1 = passed the correctness gate */
} cand_t;

static int N_g;
static double *xr, *xi, *dr, *di, *wr, *wi; /* pristine in xr/xi */

static void reseed(void) { memcpy(wr, xr, (size_t)N_g * 8); memcpy(wi, xi, (size_t)N_g * 8); }

static void run_cand(cand_t *c)
{
    switch (c->route) {
    case R_3P:       vfft_oop_execute_fwd(c->p, xr, xi, dr, di); break;
    case R_3P_IP:    vfft_oop_execute_fwd(c->p, wr, wi, wr, wi); break;
    case R_2PA_IP:   vfft_oop_execute_fwd_2pa(c->p, wr, wi, wr, wi); break;
    case R_2PB_IP:   vfft_oop_execute_fwd_2pb(c->p, wr, wi, wr, wi); break;
    case R_TWL_IP:   vfft_oop_execute_fwd_2pa_twl(c->p, wr, wi, wr, wi); break;
    case R_3PL3_IP:  vfft_oop_execute_fwd(c->p, wr, wi, wr, wi); break;
    case R_2PAL3_IP: vfft_oop_execute_fwd_2pa(c->p, wr, wi, wr, wi); break;
    case R_MONO:     vfft_k1_mono_fn(N_g)(xr, xi, dr, di, 0,0,0,0,0,0,0); break;
    case R_MONO_ALT: vfft_k1_mono_alt_fn(N_g)(xr, xi, dr, di, 0,0,0,0,0,0,0); break;
    case R_CCOL:     vfft_oop_execute_fwd_ccol(c->p, wr, wi, wr, wi); break;
    }
}

/* naive natural-order DFT reference, split planes (O(N^2): ~17M mults at
 * 4096 — milliseconds, calibrate-time only) */
static void ref_dft(const double *ar, const double *ai, double *Rr, double *Ri, int N)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)((long long)k * n % N) / N;
            double c = cos(a), s = sin(a);
            sr += ar[n] * c - ai[n] * s;
            si += ar[n] * s + ai[n] * c;
        }
        Rr[k] = sr; Ri[k] = si;
    }
}

static int gate_cand(cand_t *c, const double *Rr, const double *Ri)
{
    reseed();
    memset(dr, 0, (size_t)N_g * 8); memset(di, 0, (size_t)N_g * 8);
    run_cand(c);
    const double *or_ = (RAXIS[c->route] == 0 && c->route != R_MONO) ? dr
                        : (c->route == R_MONO || c->route == R_MONO_ALT) ? dr
                        : wr;
    const double *oi_ = (or_ == dr) ? di : wi;
    double e = 0, m = 0;
    for (int n = 0; n < N_g; n++) {
        double dr_ = fabs(or_[n] - Rr[n]), di_ = fabs(oi_[n] - Ri[n]);
        if (dr_ > e) e = dr_;
        if (di_ > e) e = di_;
        double mm = fabs(Rr[n]) > fabs(Ri[n]) ? fabs(Rr[n]) : fabs(Ri[n]);
        if (mm > m) m = mm;
    }
    c->gated = (m > 0 && e / m < 1e-9);
    if (!c->gated)
        printf("#   GATE-FAIL %s %dx%d relerr=%.2e — dropped, not timed\n",
               RNAME[c->route], c->R1, c->R2, m > 0 ? e / m : e);
    return c->gated;
}

/* merge lines emitted by plan_and_bank (a temp file in oop_wisdom format)
 * into <wisdir>/oop_wisdom.txt: load both through the SHIPPED reader,
 * replace-or-append by (N,K,kind), save through the shipped writer. */
static vfft_oop_wisdom_t WMAIN, WNEW; /* large; static, not stack */
static int merge_bank(const char *main_path, const char *tmp_path)
{
    memset(&WMAIN, 0, sizeof WMAIN);
    (void)vfft_oop_wisdom_load(&WMAIN, main_path);
    memset(&WNEW, 0, sizeof WNEW);
    if (vfft_oop_wisdom_load(&WNEW, tmp_path) != 0) return 0;
    int merged = 0;
    for (int i = 0; i < WNEW.count; i++) {
        /* plan_and_bank also emits the cell's kind-4 SCRAMBLED verdict.
         * Below 2048 that is a WRONG-SLOT row: the cascade is the ≥2048
         * tier and the sub-2048 scrambled champion is the identity ILP
         * engine (which the kind-3 line already encodes) — banking the
         * cascade there flips explicit-SCRAMBLED onto a comb and breaks
         * the scr==nat identity gate (caught live, 2026-08-06, with the
         * cascade measuring 2.2× SLOWER than what it displaced). */
        if (WNEW.e[i].kind == VFFT_OOP_KIND_ZSPLIT && WNEW.e[i].N < 2048) {
            printf("#   skip sub-2048 kind-4 row (N=%d) — wrong-slot verdict\n",
                   WNEW.e[i].N);
            continue;
        }
        int idx = -1;
        for (int j = 0; j < WMAIN.count; j++)
            if (WMAIN.e[j].N == WNEW.e[i].N && WMAIN.e[j].K == WNEW.e[i].K &&
                WMAIN.e[j].kind == WNEW.e[i].kind) { idx = j; break; }
        if (idx < 0) {
            if (WMAIN.count >= VFFT_OOP_WISDOM_MAX) continue;
            idx = WMAIN.count++;
        }
        WMAIN.e[idx] = WNEW.e[i];
        merged++;
    }
    FILE *f = fopen(main_path, "w");
    if (!f) return -1;
    for (int j = 0; j < WMAIN.count; j++)
        vfft_oop_wisdom_write_entry(f, &WMAIN.e[j]);
    fclose(f);
    return merged;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* core 2 */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 3) {
        printf("usage: calibrate_k1.exe <wisdir> <rigor 0|1> [N...]\n"
               "🔴 wisdir is WRITTEN — use a scratch copy, promote after gates\n");
        return 2;
    }
    const char *wisdir = argv[1];
    int rigor = atoi(argv[2]);
    int trials = rigor ? 5 : 3;
    static const int DEF[] = { 128, 256, 512, 1024, 2048, 4096 };
    int cells[64], ncell = 0;
    if (argc > 3)
        for (int i = 3; i < argc && ncell < 64; i++) cells[ncell++] = atoi(argv[i]);
    else
        for (int i = 0; i < 6; i++) cells[ncell++] = DEF[i];

    char wpath[600], tpath[600];
    snprintf(wpath, sizeof wpath, "%s/oop_wisdom.txt", wisdir);
    snprintf(tpath, sizeof tpath, "%s/k1_bank_tmp.txt", wisdir);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int maxN = 0;
    for (int i = 0; i < ncell; i++) if (cells[i] > maxN) maxN = cells[i];
    static vfft_il_dp_context_t ctx; /* large */
    vfft_il_dp_init(&ctx, maxN);

    printf("# calibrate_k1 v2: %d cell(s), rigor=%s, wisdom=%s\n",
           ncell, rigor ? "PATIENT(5 trials)" : "SMOKE(3 trials)", wpath);

    int total_banked = 0;
    for (int ci = 0; ci < ncell; ci++) {
        int N = cells[ci];
        N_g = N;
        xr = ad(N); xi = ad(N); dr = ad(N); di = ad(N); wr = ad(N); wi = ad(N);
        double *Rr = ad(N), *Ri = ad(N);
        srand(77 + N);
        for (int n = 0; n < N; n++) {
            xr[n] = (double)rand() / RAND_MAX - 0.5;
            xi[n] = (double)rand() / RAND_MAX - 0.5;
        }
        ref_dft(xr, xi, Rr, Ri, N);

        /* ── enumerate + gate split candidates ── */
        cand_t cand[128]; int nc = 0;
        vfft_oop_plan_t *plans[24]; int np = 0;
        for (int R2 = (N < 128 ? N : 128); R2 >= 4; R2--) {
            if (N % R2) continue;
            int R1 = N / R2;
            if (R1 < 4 || R1 > 128 || (R1 % 4) || (R2 % 4)) continue;
            vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
            if (!p || np >= 22) { if (p) vfft_oop_plan_destroy(p); continue; }
            plans[np++] = p;
            struct { int route; int avail; } rs[] = {
                { R_3P, 1 }, { R_3P_IP, 1 },
                { R_2PA_IP, p->t1_ul != 0 }, { R_2PB_IP, p->leaf_ul != 0 },
                { R_TWL_IP, p->t1_ul_twl != 0 },
            };
            for (int i = 0; i < 5 && nc < 120; i++)
                if (rs[i].avail) {
                    cand[nc] = (cand_t){ rs[i].route, R1, R2, 0, p, 1e18, 0 };
                    nc++;
                }
            /* LOG3 twins on a DEDICATED plan (pointer swap once, at setup) */
            if ((p->t1_l3 || p->t1_ul_l3) && np < 22 && nc < 118) {
                vfft_oop_plan_t *pl = vfft_oop_plan_create_k1(N, R1, R2);
                if (pl) {
                    plans[np++] = pl;
                    if (pl->t1_l3) {
                        pl->t1p = pl->t1_l3;
                        cand[nc] = (cand_t){ R_3PL3_IP, R1, R2, 0, pl, 1e18, 0 }; nc++;
                    }
                    if (pl->t1_ul_l3 && nc < 120) {
                        pl->t1_ul = pl->t1_ul_l3;
                        cand[nc] = (cand_t){ R_2PAL3_IP, R1, R2, 0, pl, 1e18, 0 }; nc++;
                    }
                }
            }
        }
        /* CCOL: the only split arm past the leaf/t1 reach */
        if ((N % 64) == 0 && N / 64 >= 16) {
            int ccf[VFFT_K1_CC_MAX_NF];
            int ccn = vfft_k1_cc_default_chain(N / 64, ccf);
            if (ccn && np < 22) {
                vfft_oop_plan_t *pc = vfft_oop_plan_create_k1_cc(N, 64, ccf, ccn, &reg);
                if (pc) {
                    plans[np++] = pc;
                    cand[nc] = (cand_t){ R_CCOL, 64, N / 64,
                                         vfft_k1_cc_chain_encode(ccf, ccn), pc, 1e18, 0 };
                    nc++;
                }
            }
        }
        if (vfft_k1_mono_fn(N) && nc < 120)
        { cand[nc] = (cand_t){ R_MONO, 0, 0, 0, NULL, 1e18, 0 }; nc++; }
        if (vfft_k1_mono_alt_fn(N) && nc < 120)
        { cand[nc] = (cand_t){ R_MONO_ALT, 0, 0, 0, NULL, 1e18, 0 }; nc++; }

        int ngated = 0;
        for (int k = 0; k < nc; k++) ngated += gate_cand(&cand[k], Rr, Ri);

        int reps = (int)(2e6 / (double)N);
        if (reps < 100) reps = 100;
        if (reps > 400000) reps = 400000;
        printf("# N=%d split candidates=%d gated=%d trials=%d reps=%d\n",
               N, nc, ngated, trials, reps);

        /* ── order-rotated timing, reseed per burst ── */
        for (int t = 0; t < trials; t++) {
            if (t) cachebust();
            for (int k = 0; k < nc; k++) {
                cand_t *c = &cand[(k + t) % nc];
                if (!c->gated) continue;
                reseed();
                for (int w = 0; w < 10; w++) run_cand(c);
                reseed();
                double t0 = now_ns_();
                for (int i = 0; i < reps; i++) run_cand(c);
                double ns = (now_ns_() - t0) / reps;
                if (ns < c->best) c->best = ns;
            }
        }

        int win_ip = -1, win_oop = -1;
        for (int k = 0; k < nc; k++) {
            if (!cand[k].gated) continue;
            int *w = RAXIS[cand[k].route] ? &win_ip : &win_oop;
            if (*w < 0 || cand[k].best < cand[*w].best) *w = k;
            printf("cand,%d,%s,%d,%d,%.1f\n", N, RNAME[cand[k].route],
                   cand[k].R1, cand[k].R2, cand[k].best);
        }

        int spr = -1, sR1 = 0, sR2 = 0;
        if (win_ip >= 0) {
            spr = SPMAP[cand[win_ip].route];
            sR1 = cand[win_ip].R1; sR2 = cand[win_ip].R2;
            if (cand[win_ip].route == R_MONO_ALT) { sR1 = 8; sR2 = N / 8; }
            printf("# N=%d SPLIT-IP winner: %s %dx%d %.1f ns"
                   "  (split-oop winner: %s %.1f ns)\n",
                   N, RNAME[cand[win_ip].route], sR1, sR2, cand[win_ip].best,
                   win_oop >= 0 ? RNAME[cand[win_oop].route] : "-",
                   win_oop >= 0 ? cand[win_oop].best : 0.0);
        } else
            printf("# N=%d NO gated split candidate — kind-3 line will be "
                   "SKIPPED (sp_route<0 refusal, not zero-filled)\n", N);

        /* ── IL axis + banking: the shipped measured search, whole ── */
        FILE *tf = fopen(tpath, "w");
        if (tf) {
            int lines = vfft_il_dp_plan_and_bank(&ctx, tf, N, spr, sR1, sR2, 1);
            fclose(tf);
            int m = merge_bank(wpath, tpath);
            printf("# N=%d banked %d line(s) (merged %d into %s)\n",
                   N, lines, m, wpath);
            total_banked += (m > 0 ? m : 0);
        }

        for (int i = 0; i < np; i++) vfft_oop_plan_destroy(plans[i]);
        vfft_proto_aligned_free(xr); vfft_proto_aligned_free(xi);
        vfft_proto_aligned_free(dr); vfft_proto_aligned_free(di);
        vfft_proto_aligned_free(wr); vfft_proto_aligned_free(wi);
        vfft_proto_aligned_free(Rr); vfft_proto_aligned_free(Ri);
    }
    remove(tpath);
    printf("calibrate_k1 DONE — %d wisdom line(s) merged\n", total_banked);
    return 0;
}
