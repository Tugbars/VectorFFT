/* bench_pad_vs_tail.c — for a caller wanting K real transforms, is it faster to run
 * the arbitrary-K TAIL at exact K, or PAD to Kp=roundup(K,VW) and ignore the pad cols?
 * Padding cost is literally our full-width time at Kp (pad = "call with Kp"). We compare
 * OUR T(K) vs OUR T(Kp) by TIGHT INTERLEAVING in one process (ratio of summed times
 * cancels thermal drift; cross-process absolute compare is the noisy thing the lesson
 * warns about). SAME reps for K and Kp => ratio is per-CALL: ratio = T_pad/T_tail,
 * ratio < 1 => PADDING WINS. rem=3 only. Plans come from the CALIBRATED spike_wisdom
 * (nearest-K), built via plan_create_ex (factors + variants + use_dif).
 *
 * Build: build_tuned/build.py --src benches/bench_pad_vs_tail.c --compile   (no --mkl)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "dp_planner.h"

#define MAXF 12
typedef struct { int N; int K; int nf; int f[MAXF]; int v[MAXF]; int use_dif; } went_t;
static went_t WIS[4096]; static int NWIS = 0;

static void load_wisdom(const char *path)
{
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "no wisdom %s\n", path); exit(1); }
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        if (line[0]=='#' || line[0]=='@' || line[0]=='\n') continue;
        char *s, *t = strtok_r(line, " \t\n", &s); if (!t) continue;
        went_t e; memset(&e, 0, sizeof e);
        e.N = atoi(t);
        t = strtok_r(NULL," \t\n",&s); if(!t) continue; e.K = atoi(t);
        t = strtok_r(NULL," \t\n",&s); if(!t) continue; e.nf = atoi(t);
        if (e.nf < 1 || e.nf > MAXF) continue;
        int ok = 1;
        for (int i=0;i<e.nf;i++){ t=strtok_r(NULL," \t\n",&s); if(!t){ok=0;break;} e.f[i]=atoi(t); }
        if(!ok) continue;
        t = strtok_r(NULL," \t\n",&s); if(!t) continue; /* best_ns */
        int flags[4];
        for (int i=0;i<4;i++){ t=strtok_r(NULL," \t\n",&s); if(!t){ok=0;break;} flags[i]=atoi(t); }
        if(!ok) continue;
        e.use_dif = flags[3];
        for (int i=0;i<e.nf;i++){ t=strtok_r(NULL," \t\n",&s); e.v[i]= t?atoi(t):0; }
        if (NWIS < 4096) WIS[NWIS++] = e;
    }
    fclose(f);
}
static const went_t *wis_lookup(int N, int K)
{
    const went_t *best = NULL; int bestd = 1<<30;
    for (int i=0;i<NWIS;i++) if (WIS[i].N==N) {
        int d = abs(WIS[i].K - K); if (d < bestd) { bestd = d; best = &WIS[i]; }
    }
    return best;
}

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { fprintf(stderr,"alloc\n"); exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }
static void fill(double *re, double *im, int N, size_t K, size_t Kp)
{
    srand(7 + N + (int)K);
    for (size_t e=0;e<(size_t)N;e++) for (size_t l=0;l<Kp;l++) {
        re[e*Kp+l] = (l<K) ? (double)rand()/RAND_MAX-0.5 : 0.0;
        im[e*Kp+l] = (l<K) ? (double)rand()/RAND_MAX-0.5 : 0.0;
    }
}
static double burst(stride_plan_t *p, double *re, double *im, size_t K, int reps)
{
    double t0 = vfft_proto_now_ns();
    for (int i=0;i<reps;i++) vfft_proto_execute_fwd(p, re, im, K);
    return vfft_proto_now_ns() - t0;
}
static void facstr(const went_t *w, char *out, size_t n)
{
    int o = 0;
    for (int i=0;i<w->nf;i++) o += snprintf(out+o, n-o, "%s%d", i?".":"", w->f[i]);
    snprintf(out+o, n-o, "%s", w->use_dif ? "/DIF" : "");
}
static int dcmp(const void *a, const void *b){ double d=*(const double*)a-*(const double*)b; return d<0?-1:d>0?1:0; }
static double med(double *v, int n){ qsort(v,n,sizeof(double),dcmp); return n&1?v[n/2]:0.5*(v[n/2-1]+v[n/2]); }

int main(void)
{
    load_wisdom("../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    int Ns[]  = {256, 512, 1024, 2048, 4096};
    size_t Ks[] = {7, 11, 15, 19, 23, 27, 31};   /* rem=3 only */
    int nN = 5, nK = 7, rounds = 199;

    printf("# PAD vs TAIL, rem=3 only, CALIBRATED plans (spike_wisdom nearest-K). %d wisdom entries.\n", NWIS);
    printf("# interleaved, per-call ratio = pad/tail. ratio<1 => PAD wins. Kp = K+1 (roundup to VW=4).\n");
    for (int ni=0; ni<nN; ni++) {
        int N = Ns[ni];
        printf("\n=== N=%d ===\n", N);
        printf("  K   Kp   factor(K)        tail_ns    pad_ns    pad/tail  | tail/K   pad/K   | winner\n");
        for (int c=0;c<nK;c++) {
            size_t K = Ks[c], Kp = (K + 3) & ~(size_t)3;
            const went_t *wK = wis_lookup(N, (int)K), *wP = wis_lookup(N, (int)Kp);
            if (!wK || !wP) { printf("  K=%zu no wisdom for N=%d\n", K, N); continue; }
            stride_plan_t *pk = vfft_proto_plan_create_ex(N,(int)K,(int*)wK->f,(int*)wK->v,wK->nf,wK->use_dif,&reg);
            stride_plan_t *pp = vfft_proto_plan_create_ex(N,(int)Kp,(int*)wP->f,(int*)wP->v,wP->nf,wP->use_dif,&reg);
            if (!pk || !pp) { printf("  K=%zu plan FAILED\n", K); continue; }
            double *rk=ad((size_t)N*K), *ik=ad((size_t)N*K), *rp=ad((size_t)N*Kp), *ip=ad((size_t)N*Kp);
            fill(rk,ik,N,K,K); fill(rp,ip,N,K,Kp);
            int reps = (int)(8000000ull / ((size_t)N * Kp)); if (reps < 40) reps = 40;
            for (int w=0;w<8;w++){ burst(pk,rk,ik,K,reps); burst(pp,rp,ip,Kp,reps); }
            static double rat[256], tns[256], pns[256];
            int RR = rounds; if (RR > 256) RR = 256;
            for (int r=0;r<RR;r++){
                double t, p;   /* adjacent bursts (same thermal state), order-flipped per round */
                if (r&1){ t=burst(pk,rk,ik,K,reps); p=burst(pp,rp,ip,Kp,reps); }
                else    { p=burst(pp,rp,ip,Kp,reps); t=burst(pk,rk,ik,K,reps); }
                tns[r]=t/reps; pns[r]=p/reps; rat[r]=p/t;   /* per-round per-call ratio */
            }
            double tpc = med(tns,RR), ppc = med(pns,RR), rr = med(rat,RR);  /* MEDIAN rejects outlier rounds */
            char fs[64]; facstr(wK, fs, sizeof fs);
            printf("  %-3zu %-3zu  %-15s  %8.0f  %8.0f   %.3f    | %7.1f %7.1f | %s\n",
                   K, Kp, fs, tpc, ppc, rr, tpc/K, ppc/K, rr < 1.0 ? "PAD":"tail");
            afree(rk); afree(ik); afree(rp); afree(ip);
        }
    }
    return 0;
}
