/* remeasure.c — re-time the ALREADY-FOUND plan for given cells, a few rounds.
 *
 * NOT a re-calibration (no exhaustive search). Reads each cell's wisdom entry
 * (factors + variants + DIT/DIF), builds that exact plan, and times it
 * (warmup + best-of) for several rounds — to see whether a recorded best_ns
 * was hot-box-inflated (re-measure drops it) or genuinely that slow (it holds).
 * Generic executor (matches what the calibrator's deploy-rebench recorded).
 *
 * usage: remeasure <wisdom> <N,N,...> [rounds=3] [core=2]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/wisdom_reader.h"
#include "../generator/generated/registry.h"
#include <windows.h>

static double now_ns(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

int main(int argc, char **argv){
    stride_env_init();
    if (argc < 3) { fprintf(stderr,"usage: remeasure <wisdom> <N,N,...> [rounds=3] [core=2]\n"); return 2; }
    const char *wpath = argv[1];
    int rounds = (argc>3)?atoi(argv[3]):3;
    int core   = (argc>4)?atoi(argv[4]):2;
    if (stride_pin_thread(core)!=0) fprintf(stderr,"warn: pin cpu%d failed\n", core);
    size_t K = 4;

    int Ns[64], nN=0;
    { char b[512]; strncpy(b, argv[2], sizeof b-1); b[sizeof b-1]=0;
      char *t=strtok(b,","); while(t && nN<64){ Ns[nN++]=atoi(t); t=strtok(NULL,","); } }

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis; memset(&wis,0,sizeof wis);
    if (vfft_proto_wisdom_load(&wis, wpath)!=0){ fprintf(stderr,"cannot load %s\n",wpath); return 1; }

    /* shared buffers at the max N in the list */
    int maxN=0; for(int i=0;i<nN;i++) if(Ns[i]>maxN) maxN=Ns[i];
    size_t maxtot=(size_t)maxN*K;
    double *re,*im,*ore,*oim;
    vfft_proto_posix_memalign((void**)&re,64,maxtot*sizeof(double));
    vfft_proto_posix_memalign((void**)&im,64,maxtot*sizeof(double));
    vfft_proto_posix_memalign((void**)&ore,64,maxtot*sizeof(double));
    vfft_proto_posix_memalign((void**)&oim,64,maxtot*sizeof(double));

    printf("=== re-measure FOUND plans (cpu%d, %d rounds, 1s warmup + best-of-7) ===\n", core, rounds);
    for (int i=0;i<nN;i++){
        int N=Ns[i]; size_t total=(size_t)N*K;
        const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&wis, N, K);
        if (!e){ printf("  N=%d: no wisdom entry\n", N); continue; }
        stride_plan_t *plan = vfft_proto_plan_create_ex(N,K,e->factors,e->variants,e->nf,e->use_dif_forward,&reg);
        if (!plan){ printf("  N=%d: plan_create FAILED\n", N); continue; }
        for(size_t j=0;j<total;j++){ ore[j]=(double)rand()/RAND_MAX-0.5; oim[j]=(double)rand()/RAND_MAX-0.5; }
        /* warmup ~1s */
        double tw=now_ns();
        while(now_ns()-tw<1.0e9){ memcpy(re,ore,total*sizeof(double)); memcpy(im,oim,total*sizeof(double));
            vfft_proto_execute_fwd(plan,re,im,K); }
        int reps=(int)(1e6/(total+1)); if(reps<20)reps=20; if(reps>100000)reps=100000;

        char fs[64]={0}; for(int s=0;s<e->nf;s++){ char b[8]; snprintf(b,sizeof b,"%s%d",s?"x":"",e->factors[s]); strncat(fs,b,sizeof(fs)-strlen(fs)-1); }
        printf("  N=%-7d %-16s %s  recorded=%.1f ns\n", N, fs, e->use_dif_forward?"DIF":"DIT", e->best_ns);
        double omin=1e18;
        for(int r=0;r<rounds;r++){
            double best=1e18;
            for(int t=0;t<7;t++){
                memcpy(re,ore,total*sizeof(double)); memcpy(im,oim,total*sizeof(double));
                double t0=now_ns();
                for(int k=0;k<reps;k++) vfft_proto_execute_fwd(plan,re,im,K);
                double ns=(now_ns()-t0)/reps;
                if(ns<best)best=ns;
            }
            if(best<omin)omin=best;
            printf("      round %d: %10.1f ns\n", r+1, best);
        }
        printf("      -> min %.1f vs recorded %.1f  (%+.1f%%)\n", omin, e->best_ns, (omin/e->best_ns-1.0)*100.0);
        vfft_proto_plan_destroy(plan);
    }
    return 0;
}
