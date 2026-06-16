/* cmp_old_new.c — old-wisdom plan vs new-wisdom plan, BOTH measured now.
 *
 * Reads two wisdom files; for each K=4 cell present in both, builds the OLD
 * plan and the NEW plan and times them back-to-back under identical conditions
 * (same warmup + best-of, same box state) — so the ratio is pure plan quality,
 * free of the cross-time/thermal confound that makes recorded best_ns
 * incomparable across calibration runs. ratio = old_ns/new_ns (>1 => new faster).
 *
 * usage: cmp_old_new <old_wisdom> <new_wisdom> [N,N,...|all] [core=2]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/wisdom_reader.h"
#include "../generator/generated/registry.h"
#include <windows.h>

static double now_ns(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static double measure(stride_plan_t *plan, size_t total, size_t K,
                      double *re, double *im, const double *ore, const double *oim) {
    double tw=now_ns();
    while(now_ns()-tw < 0.5e9){ memcpy(re,ore,total*sizeof(double)); memcpy(im,oim,total*sizeof(double));
        vfft_proto_execute_fwd(plan,re,im,K); }
    int reps=(int)(1e6/(total+1)); if(reps<20)reps=20; if(reps>100000)reps=100000;
    double best=1e18;
    for(int t=0;t<9;t++){
        memcpy(re,ore,total*sizeof(double)); memcpy(im,oim,total*sizeof(double));
        double t0=now_ns();
        for(int k=0;k<reps;k++) vfft_proto_execute_fwd(plan,re,im,K);
        double ns=(now_ns()-t0)/reps; if(ns<best)best=ns;
    }
    return best;
}

static void fmt(char *b, size_t cap, const vfft_proto_wisdom_entry_t *e){
    size_t p=0; b[0]=0;
    for(int s=0;s<e->nf && p<cap-10;s++) p+=(size_t)snprintf(b+p,cap-p,"%s%d",s?"x":"",e->factors[s]);
    snprintf(b+p,cap-p,"/%s", e->use_dif_forward?"DIF":"DIT");
}

int main(int argc, char **argv){
    stride_env_init();
    if (argc < 3){ fprintf(stderr,"usage: cmp_old_new <old> <new> [N,..|all] [core=2]\n"); return 2; }
    int core = (argc>4)?atoi(argv[4]):2;
    if (stride_pin_thread(core)!=0) fprintf(stderr,"warn: pin cpu%d\n",core);
    size_t K=4;

    int grid[]={8,16,32,64,126,128,250,256,400,512,1024,2048,4096,8192,16384,32768,65536,131072};
    int Ns[64], nN=0;
    if (argc>3 && strcmp(argv[3],"all")!=0){
        char b[512]; strncpy(b,argv[3],sizeof b-1); b[sizeof b-1]=0;
        char *t=strtok(b,","); while(t&&nN<64){ Ns[nN++]=atoi(t); t=strtok(NULL,","); }
    } else { nN=(int)(sizeof(grid)/sizeof(grid[0])); for(int i=0;i<nN;i++) Ns[i]=grid[i]; }

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t oldw, neww; memset(&oldw,0,sizeof oldw); memset(&neww,0,sizeof neww);
    if (vfft_proto_wisdom_load(&oldw,argv[1])!=0){ fprintf(stderr,"load old %s failed\n",argv[1]); return 1; }
    if (vfft_proto_wisdom_load(&neww,argv[2])!=0){ fprintf(stderr,"load new %s failed\n",argv[2]); return 1; }

    int maxN=0; for(int i=0;i<nN;i++) if(Ns[i]>maxN) maxN=Ns[i];
    size_t maxtot=(size_t)maxN*K;
    double *re,*im,*ore,*oim;
    vfft_proto_posix_memalign((void**)&re,64,maxtot*sizeof(double));
    vfft_proto_posix_memalign((void**)&im,64,maxtot*sizeof(double));
    vfft_proto_posix_memalign((void**)&ore,64,maxtot*sizeof(double));
    vfft_proto_posix_memalign((void**)&oim,64,maxtot*sizeof(double));

    printf("=== OLD plan vs NEW plan, both measured now (cpu%d, 0.5s warmup + best-of-9) ===\n", core);
    printf("  %-8s %-16s %12s   %-16s %12s   %8s\n","N","old_plan","old_ns","new_plan","new_ns","new/old");
    double logsum=0; int ncmp=0;
    for(int i=0;i<nN;i++){
        int N=Ns[i]; size_t total=(size_t)N*K;
        const vfft_proto_wisdom_entry_t *eo=vfft_proto_wisdom_lookup(&oldw,N,K);
        const vfft_proto_wisdom_entry_t *en=vfft_proto_wisdom_lookup(&neww,N,K);
        if(!eo||!en){ continue; }
        stride_plan_t *po=vfft_proto_plan_create_ex(N,K,eo->factors,eo->variants,eo->nf,eo->use_dif_forward,&reg);
        stride_plan_t *pn=vfft_proto_plan_create_ex(N,K,en->factors,en->variants,en->nf,en->use_dif_forward,&reg);
        if(!po||!pn){ if(po)vfft_proto_plan_destroy(po); if(pn)vfft_proto_plan_destroy(pn); continue; }
        for(size_t j=0;j<total;j++){ ore[j]=(double)rand()/RAND_MAX-0.5; oim[j]=(double)rand()/RAND_MAX-0.5; }
        double ons=measure(po,total,K,re,im,ore,oim);
        double nns=measure(pn,total,K,re,im,ore,oim);
        char ob[32],nb[32]; fmt(ob,sizeof ob,eo); fmt(nb,sizeof nb,en);
        double r = (nns>0)?ons/nns:0;
        const char *v = (r>1.02)?"new faster":(r<0.98?"OLD faster":"tie");
        printf("  %-8d %-16s %12.1f   %-16s %12.1f   %6.3fx  %s\n", N, ob, ons, nb, nns, r, v);
        if(r>0){ logsum+=log(r); ncmp++; }
        vfft_proto_plan_destroy(po); vfft_proto_plan_destroy(pn);
    }
    if(ncmp) printf("\n  geomean new/old over %d cells: %.3fx\n", ncmp, exp(logsum/ncmp));
    return 0;
}
