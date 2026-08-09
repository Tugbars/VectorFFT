/* ═══════════════════════════════════════════════════════════════════════════
 * cil_ab.c — artifact-vs-artifact race harness for IL (cil-family) codelets.
 *
 * PURPOSE. Race a CANDIDATE emitted .c kernel against the form the shipped
 * plan actually resolves, inside a REAL front-door plan, before the candidate
 * is wired into any registry or wisdom. The harness compares emitted C
 * ARTIFACTS — it does not care which OCaml path produced them — so the same
 * binary serves, in order:
 *   1. hand-edited proxy kernels   (port-diversifier step 1, ILP remat probes)
 *   2. cx_passes.ml output         (diversifier pass variants)
 *   3. full-pipeline-emitted cil   vs the current hybrid emission
 *
 * MECHANISM (bench_vtune.c precedent, docs/research/mkl512_gap_campaign/):
 * #include vfft.c so struct vfft_plan_s is a complete type; create the plan
 * through the front door; ASSERT the resolved route/pair/forms; then swap the
 * candidate into p->k1il2p->leaf_f / ->mid_f per burst. Everything else —
 * geometry, twiddle tables, the partner stage, execution — is the production
 * path via vfft_execute. No executor re-implementation, no ABI drift risk.
 *
 * GATES (per the n1tb44 provenance-header doctrine):
 *   exact arms   (same math, reordered/rescheduled/duplicated-subtree):
 *                BITWISE-IDENTICAL output required.
 *   restructured arms (reassociated math): tolerance — candidate's error vs
 *                the naive O(N^2) forward DFT must be <= 4x the baseline's.
 *                Roundtrip is FORBIDDEN as a gate (permutation-blind).
 *
 * PROTOCOL (house rules): pin core 2 (mask 0x4), HIGH priority, >=15 rounds,
 * alternating arm order per round, one arena, untimed reseed per burst,
 * report FLOOR/MEDIAN/SPREAD and a CONTROL arm (byte-copy of the baseline
 * under another symbol). |delta| < control spread  =>  NOT a result.
 *
 * ADDING A CANDIDATE: drop the .c into cil_ab_arms/ with a UNIQUE symbol
 * (sed-rename; the emitter tags no variant in the name), add a DECL + ARMS[]
 * row below, rebuild with build_cil_ab.py.
 *
 * WISDOM: VFFT_WISDOM_DIR if set, else ./cil_ab_wis. The dir must be a
 * POPULATED SCRATCH COPY of generator/generated/ — this binary's creates
 * write back, and pointing any wisdom-writer at the shipped tree has
 * destroyed banked rows before. The harness refuses an empty dir.
 *
 * BUILD:  python build_cil_ab.py     (direct gcc; build.py --vfft would link
 *                                     a second vfft.c and collide)
 * RUN:    cil_ab.exe [--n 512] [--slot leaf|mid] [--rounds 25] [--arm NAME]
 * ═══════════════════════════════════════════════════════════════════════════ */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* The front door, compiled in, so struct vfft_plan_s is a complete type. */
#include "vfft.c"

/* ── Candidate + known-form symbols (frozen 11-arg z ABI, il2p.h) ───────── */
#define DECL(sym) extern void sym( \
    const double *, const double *, double *, double *,                       \
    const double *, const double *, size_t, size_t, size_t, size_t, size_t)
/* shipped forms we may need to NAME in route prints */
DECL(radix8_z_t2_fwd_avx2);
DECL(radix16_z_n1tb44_fwd_avx2);
DECL(radix32_z_n1tb48_fwd_avx2);
DECL(radix32_z_t2b48_fwd_avx2);
/* candidates (cil_ab_arms/) */
DECL(radix16_z_n1tb44_ctl_fwd_avx2);       /* byte-copy CONTROL arm          */
DECL(radix16_z_n1tb44_rmw_fwd_avx2);       /* remat: constants per-use       */
DECL(radix16_z_n1tb44_rms_fwd_avx2);       /* remat: constants + S per-use   */
DECL(radix16_z_n1tb44_sdb_fwd_avx2);       /* Leg 0: 2-way rotated S plane   */
DECL(radix16_z_n1tb44_sdb4_fwd_avx2);      /* Leg 0: 4-way rotated S plane   */
DECL(radix16_z_n1tb44_npl_fwd_avx2);       /* NEW pipeline-hosted generator  */
DECL(radix32_z_t2b48_npl_fwd_avx2);        /* NEW pipeline-hosted generator  */
DECL(radix16_z_n1tb44_brd_fwd_avx2);       /* BRAIDED scopes (window test)   */
DECL(radix32_z_t2b48_brd_fwd_avx2);        /* BRAIDED scopes (window test)   */
#undef DECL

typedef struct {
    const char  *name;      /* short label in tables                          */
    const char  *slot;      /* "leaf" or "mid"                                */
    int          radix;     /* must match the plan's slot radix to be admitted */
    vfft_il2p_fn fn;
    int          exact;     /* 1 => BITID gate; 0 => tolerance gate           */
    const char  *note;
} arm_t;

static const arm_t ARMS[] = {
    { "ctl",  "leaf", 16, radix16_z_n1tb44_ctl_fwd_avx2, 1,
      "byte-copy of shipped n1tb44 — twin-arm noise floor, expects BITID + ~0 delta" },
    /* rmw/rms (remat) and sdb/sdb4 (S rotation) RACED 2026-08-09, both NULL/
     * REFUTED; files and symbols kept as the negative record, out of the race. */
    { "npl",  "leaf", 16, radix16_z_n1tb44_npl_fwd_avx2, 1,
      "emitted by the NEW pipeline-hosted generator — expects BITID + ~0 delta" },
    { "npl",  "mid",  32, radix32_z_t2b48_npl_fwd_avx2, 1,
      "emitted by the NEW pipeline-hosted generator — expects BITID + ~0 delta" },
    { "brd",  "leaf", 16, radix16_z_n1tb44_brd_fwd_avx2, 1,
      "BRAIDED pass groups (4+2) — buys window-width with registers" },
    { "brd",  "mid",  32, radix32_z_t2b48_brd_fwd_avx2, 1,
      "BRAIDED pass groups (4+8) — buys window-width with registers" },
};
#define N_ARMS ((int)(sizeof ARMS / sizeof ARMS[0]))

typedef struct { vfft_il2p_fn fn; const char *name; } symrow_t;
static const symrow_t SYMS[] = {
    { radix8_z_t2_fwd_avx2,        "radix8_z_t2"      },
    { radix16_z_n1tb44_fwd_avx2,   "radix16_z_n1tb44" },
    { radix32_z_n1tb48_fwd_avx2,   "radix32_z_n1tb48" },
    { radix32_z_t2b48_fwd_avx2,    "radix32_z_t2b48"  },
    { radix16_z_n1tb44_ctl_fwd_avx2, "n1tb44_CTL"     },
    { radix16_z_n1tb44_rmw_fwd_avx2, "n1tb44_RMW"     },
    { radix16_z_n1tb44_rms_fwd_avx2, "n1tb44_RMS"     },
    { radix16_z_n1tb44_sdb_fwd_avx2, "n1tb44_SDB"     },
    { radix16_z_n1tb44_sdb4_fwd_avx2, "n1tb44_SDB4"   },
    { radix16_z_n1tb44_npl_fwd_avx2, "n1tb44_NPL"     },
    { radix32_z_t2b48_npl_fwd_avx2,  "t2b48_NPL"      },
    { radix16_z_n1tb44_brd_fwd_avx2, "n1tb44_BRD"     },
    { radix32_z_t2b48_brd_fwd_avx2,  "t2b48_BRD"      },
};
static const char *symname(vfft_il2p_fn f)
{
    if (!f) return "(null)";
    for (size_t i = 0; i < sizeof SYMS / sizeof SYMS[0]; i++)
        if (SYMS[i].fn == f) return SYMS[i].name;
    return "unnamed-form";      /* fine: baseline prints are informational */
}

/* ── Timing / rng / stats ──────────────────────────────────────────────── */
#define BLOCK   256      /* reps per timed sample; 1e-200 seed growth-safe   */
#define PACE_MS 20       /* spin-paced (busy), so short is fine              */

static LARGE_INTEGER qf;
static double now_ns(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * (1e9 / (double)qf.QuadPart);
}
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd01(void)
{
    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg >> 11)) / 4503599627370496.0;
}
static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void naive_dft(const double *in, double *out, int N)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * 3.14159265358979323846 * (double)k * (double)n / (double)N;
            double c = cos(a), s = sin(a);
            sr += in[2*n] * c - in[2*n+1] * s;
            si += in[2*n] * s + in[2*n+1] * c;
        }
        out[2*k] = sr; out[2*k+1] = si;
    }
}

/* max |got-ref| / max|ref| over the plane */
static double rel_err(const double *got, const double *ref, int N2)
{
    double scale = 0, err = 0;
    for (int i = 0; i < N2; i++) {
        double a = fabs(ref[i]);   if (a > scale) scale = a;
        double d = fabs(got[i] - ref[i]); if (d > err) err = d;
    }
    return scale > 0 ? err / scale : err;
}

/* ── Main ──────────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    int N = 512, rounds = 25;
    const char *slot = "leaf", *only_arm = NULL;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n")      && i+1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i+1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slot")   && i+1 < argc) slot = argv[++i];
        else if (!strcmp(argv[i], "--arm")    && i+1 < argc) only_arm = argv[++i];
        else { printf("unknown arg %s\n", argv[i]); return 2; }
    }
    int is_leaf = !strcmp(slot, "leaf");
    int is_both = !strcmp(slot, "both");
    if (!is_leaf && !is_both && strcmp(slot, "mid")) {
        printf("--slot must be leaf|mid|both\n");
        return 2;
    }
    if (rounds < 15) { printf("house protocol: >=15 rounds\n"); return 2; }

    QueryPerformanceFrequency(&qf);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x4);   /* core 2 */

    /* wisdom: scratch copy only, never the shipped tree */
    const char *wis = getenv("VFFT_WISDOM_DIR");
    if (!wis) { _putenv_s("VFFT_WISDOM_DIR", "cil_ab_wis"); wis = "cil_ab_wis"; }
    {
        char probe[512];
        snprintf(probe, sizeof probe, "%s\\spike_wisdom.txt", wis);
        FILE *f = fopen(probe, "rb");
        if (!f) {
            printf("ABORT: wisdom dir '%s' missing/empty (this binary writes back;\n"
                   "       never point it at the shipped generated/).  Populate with:\n"
                   "  mkdir %s & copy ..\\..\\src\\dag-fft-compiler\\generator\\generated\\*.txt %s\\\n",
                   wis, wis, wis);
            return 2;
        }
        fclose(f);
    }
    printf("cil_ab  N=%d slot=%s rounds=%d block=%d  core mask 0x4  HIGH  wisdom=%s\n",
           N, slot, rounds, BLOCK, wis);

    /* ── plan, through the front door ── */
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_NATURAL;
    cfg.rigor     = VFFT_MEASURE;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = 1;
    cfg.nthreads  = 1;
    double tc0 = now_ns();
    vfft_plan p = vfft_create(&cfg);
    double tc1 = now_ns();
    if (!p) { printf("ABORT: vfft_create(N=%d) NULL\n", N); return 3; }
    printf("[create %.0f ms]\n", (tc1 - tc0) / 1e6);

    struct vfft_plan_s *h = (struct vfft_plan_s *)p;
    if (!h->k1il2p) {
        printf("ABORT: N=%d did not resolve to the IL2P route — this harness\n"
               "       swaps il2p leaf/mid pointers only.\n", N);
        return 3;
    }
    vfft_il2p_plan_t *q = (vfft_il2p_plan_t *)h->k1il2p;
    printf("route=IL2P pair(R1=%d,R2=%d) leaf=%s mid=%s\n",
           q->R1, q->R2, symname(q->leaf_f), symname(q->mid_f));

    vfft_il2p_fn orig_leaf = q->leaf_f, orig_mid = q->mid_f;

    /* Each lane is a (leaf, mid) PAIR; single-slot modes hold the other slot
     * at the original. --slot both pairs arms BY NAME (a name must supply
     * both a leaf-radix and a mid-radix row) and swaps the WHOLE PLAN —
     * old kernels vs new kernels end-to-end. */
    typedef struct { vfft_il2p_fn lf, mf; const char *name; int exact; } lane_arm_t;
    lane_arm_t arm[N_ARMS]; int n_arm = 0;
    if (is_both) {
        for (int i = 0; i < N_ARMS; i++) {
            if (strcmp(ARMS[i].slot, "leaf") || ARMS[i].radix != q->R2) continue;
            if (only_arm && strcmp(ARMS[i].name, only_arm)) continue;
            for (int j = 0; j < N_ARMS; j++) {
                if (strcmp(ARMS[j].slot, "mid") || ARMS[j].radix != q->R1) continue;
                if (strcmp(ARMS[i].name, ARMS[j].name)) continue;
                arm[n_arm].lf = ARMS[i].fn;
                arm[n_arm].mf = ARMS[j].fn;
                arm[n_arm].name = ARMS[i].name;
                arm[n_arm].exact = ARMS[i].exact && ARMS[j].exact;
                n_arm++;
            }
        }
    } else {
        int slot_radix = is_leaf ? q->R2 : q->R1;
        for (int i = 0; i < N_ARMS; i++) {
            if (strcmp(ARMS[i].slot, slot)) continue;
            if (only_arm && strcmp(ARMS[i].name, only_arm)) continue;
            if (ARMS[i].radix != slot_radix) {
                printf("skip arm '%s': radix %d != plan %s radix %d\n",
                       ARMS[i].name, ARMS[i].radix, slot, slot_radix);
                continue;
            }
            arm[n_arm].lf = is_leaf ? ARMS[i].fn : orig_leaf;
            arm[n_arm].mf = is_leaf ? orig_mid : ARMS[i].fn;
            arm[n_arm].name = ARMS[i].name;
            arm[n_arm].exact = ARMS[i].exact;
            n_arm++;
        }
    }
    if (!n_arm) { printf("ABORT: no admissible arms for slot=%s\n", slot); return 3; }
#define SET_LANE(lf_, mf_) do { q->leaf_f = (lf_); q->mid_f = (mf_); } while (0)

    int N2 = 2 * N;
    size_t nb = (size_t)N2 * sizeof(double);
    double *z    = (double *)_aligned_malloc(nb, 64);
    double *gold = (double *)_aligned_malloc(nb, 64);
    double *x    = (double *)malloc(nb);
    double *ref  = (double *)malloc(nb);
    double *outA = (double *)malloc(nb);
    double *outB = (double *)malloc(nb);
    if (!z || !gold || !x || !ref || !outA || !outB) { printf("ABORT: alloc\n"); return 3; }
    for (int i = 0; i < N2; i++) gold[i] = 1e-200 * rnd01();

    /* ── correctness gates ── */
    printf("\n== correctness (forward vs naive O(N^2); exact arms need BITID) ==\n");
    for (int i = 0; i < N2; i++) x[i] = rnd01();
    naive_dft(x, ref, N);

    SET_LANE(orig_leaf, orig_mid);
    memcpy(z, x, nb);
    vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);
    memcpy(outA, z, nb);
    double errA = rel_err(outA, ref, N2);
    printf("  baseline           vs naive %.3e\n", errA);

    int gate_fail = 0;
    for (int a = 0; a < n_arm; a++) {
        SET_LANE(arm[a].lf, arm[a].mf);
        memcpy(z, x, nb);
        vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);
        memcpy(outB, z, nb);
        SET_LANE(orig_leaf, orig_mid);

        double errB  = rel_err(outB, ref, N2);
        int    bitid = !memcmp(outA, outB, nb);
        int    pass  = arm[a].exact ? bitid : (errB <= 4.0 * errA + 1e-300);
        printf("  %-18s vs naive %.3e  vs baseline %s  %s\n",
               arm[a].name, errB,
               bitid ? "BITID" : "differs",
               pass ? "PASS" : "FAIL");
        if (!pass) gate_fail = 1;
    }
    if (gate_fail) { printf("GATE: FAIL — no timing on incorrect arms\n"); return 4; }

    /* ── race: alternating order, reseed per burst, floor/median/spread ── */
    int lanes = n_arm + 2;                    /* [0]=baseline [1..n]=arms [n+1]=base again */
    double *samp = (double *)malloc((size_t)lanes * rounds * sizeof(double));
    printf("\n== race (%d rounds x %d lanes x %d reps; alternating) ==\n",
           rounds, lanes, BLOCK);
    for (int r = 0; r < rounds; r++) {
        for (int li = 0; li < lanes; li++) {
            int lane = (r & 1) ? (lanes - 1 - li) : li;   /* alternate order */
            if (lane == 0 || lane == lanes - 1)
                SET_LANE(orig_leaf, orig_mid);
            else
                SET_LANE(arm[lane - 1].lf, arm[lane - 1].mf);
            /* untimed warmup block: after any pause the core has downclocked;
             * timing the first block would charge the ramp to the arm */
            memcpy(z, gold, nb);
            for (int b = 0; b < BLOCK; b++)
                vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);
            memcpy(z, gold, nb);                            /* untimed reseed */
            double t0 = now_ns();
            for (int b = 0; b < BLOCK; b++)
                vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);
            samp[lane * rounds + r] = (now_ns() - t0) / (double)BLOCK;
            SET_LANE(orig_leaf, orig_mid);
            /* spin-pace (busy) so the core never sleeps into a downclock */
            { double until = now_ns() + PACE_MS * 1e6;
              while (now_ns() < until) _mm_pause(); }
        }
    }

    /* stats first (control noise must exist before any verdict is printed) */
    double flr[16], med[16], spread[16];
    for (int lane = 0; lane < lanes; lane++) {
        double *s = samp + (size_t)lane * rounds;
        qsort(s, rounds, sizeof(double), cmp_d);
        flr[lane]    = s[0];
        med[lane]    = s[rounds / 2];
        spread[lane] = (s[rounds - 1] - s[0]) / s[0];
    }
    double base_med   = med[0];
    double ctl_noise  = fabs(med[lanes - 1] - base_med) / base_med;

    printf("\n%-18s %9s %9s %8s %10s  verdict\n",
           "arm", "floor_ns", "med_ns", "spread", "d_med");
    for (int lane = 0; lane < lanes; lane++) {
        const char *nm = lane == 0 ? "baseline"
                       : lane == lanes - 1 ? "baseline(2nd)" : arm[lane - 1].name;
        double d = (med[lane] - base_med) / base_med;
        char verdict[64] = "";
        if (lane > 0 && lane < lanes - 1)
            snprintf(verdict, sizeof verdict, "%s",
                     fabs(d) > ctl_noise + 0.005 ? "RESULT" : "not a result (< control)");
        printf("%-18s %9.1f %9.1f %7.1f%% %+9.2f%%  %s\n",
               nm, flr[lane], med[lane], spread[lane] * 100.0, d * 100.0, verdict);
    }
    printf("\ntwin-arm control disagreement %.2f%% — deltas inside this are noise.\n",
           ctl_noise * 100.0);
    vfft_destroy(p);
    return 0;
}
