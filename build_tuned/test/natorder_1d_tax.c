/* natorder_1d_tax.c — 1D c2c NATURAL-order reorder tax: natural fwd vs the DEFAULT (scrambled) fwd,
 * SAME calibrated FFT plan (DEFAULT create calibrates+banks the base plan; NATURAL create is a lookup
 * + the reorder verdict), so only the reorder pass differs. Order-neutralized (interleaved rounds with
 * cooldown, min-of-N each), QPC, core-pinned, HIGH priority. No MKL — the tax is MKL-independent.
 *
 * Persists BOTH the wisdom dir (so the chosen per-cell mode survives) AND a CSV with the tax numbers
 * plus the selected nat_mode (5=opportunistic PSWAP / 4=PURE / else) read back from spike_wisdom.txt.
 * A partial/timed-out run still leaves the CSV rows written so far.
 *
 * Purpose: quantify what the opportunistic-PSWAP fix changed. 256/4 (palindromic 16·16) now
 * deterministically picks PSWAP (cheap pair-swap reorder) instead of the unpaced race sometimes
 * landing on PURE (expensive 16·16 cycle).
 *
 * Build: python build.py --src test/natorder_1d_tax.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

#define WISDIR "natorder_1dtax_wis"
#define CSVOUT "natorder_1d_tax.csv"

static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static vfft_plan mk(int N, size_t K, int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1; c.order=order;
    return vfft_create(&c);
}

static double burst(vfft_plan p, double *re, double *im, int reps){
    double t0=now_ns();
    for(int i=0;i<reps;i++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    return (now_ns()-t0)/reps;
}

/* Parse spike_wisdom.txt for (N,K): positionally skip to the v7 nat block (mirrors wisdom_reader.h).
 * Line: N K nf factors[nf] best_ns use_blocked split_stage block_groups use_dif_forward variants[nf]
 *       exec_me nat_mode nat_ns [nat_nf factors... nat_prof (only if nat_mode==5 and nat_nf>0)].
 * Returns 1 + fills *mode,*nf on hit; 0 if not found. */
static int read_nat(int N, size_t K, int *mode, int *nf){
    char path[700]; snprintf(path,sizeof path, "%s/spike_wisdom.txt", WISDIR);
    FILE *f=fopen(path,"r"); if(!f) return 0;
    char line[4096]; int found=0;
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#'||line[0]=='@') continue;
        char *t=strtok(line," \t\r\n");
        if(!t) continue; int n=atoi(t);
        t=strtok(NULL," \t\r\n"); if(!t) continue; long k=atol(t);
        if(n!=N||k!=(long)K) continue;
        t=strtok(NULL," \t\r\n"); if(!t) break; int wnf=atoi(t);
        if(wnf<=0||wnf>16) break;
        for(int i=0;i<wnf;i++){ t=strtok(NULL," \t\r\n"); if(!t) goto done; }   /* factors */
        for(int i=0;i<5;i++){ t=strtok(NULL," \t\r\n"); if(!t) goto done; }      /* 5 mid fields */
        for(int i=0;i<wnf;i++){ t=strtok(NULL," \t\r\n"); if(!t) goto done; }    /* variants */
        t=strtok(NULL," \t\r\n"); if(!t) goto done;                              /* exec_me */
        t=strtok(NULL," \t\r\n"); if(!t) goto done; *mode=atoi(t);               /* nat_mode */
        t=strtok(NULL," \t\r\n");                                                /* nat_ns  */
        *nf=0;
        if(*mode==5){ t=strtok(NULL," \t\r\n"); if(t) *nf=atoi(t); }             /* nat_nf (opp=absent=0) */
        found=1; break;
    }
done:
    fclose(f); return found;
}

static const char *mode_name(int mode,int nf){
    if(mode==1) return "FREE";
    if(mode==4) return "PURE";
    if(mode==5) return (nf>0)?"PSWAP-inj":"PSWAP-opp";
    if(mode==3) return "SCR";
    if(mode==2) return "LEAF_IP";
    return "UNSET";
}

static void cell(FILE *csv, int N, size_t K){
    size_t tot=(size_t)N*K;
    double *re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ re[i]=(double)((i*2654435761u)&1023)/1024.0-0.5;
                               im[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    vfft_plan pd=mk(N,K,VFFT_ORDER_DEFAULT);   /* scrambled: calibrates+banks the base plan */
    vfft_plan pn=mk(N,K,VFFT_ORDER_NATURAL);   /* natural: lookup + reorder verdict          */
    if(!pd||!pn){ printf("N=%-5d K=%-4zu  (NULL plan)\n",N,K); free(re);free(im); return; }
    int mode=0,nf=0; read_nat(N,K,&mode,&nf);
    int reps=(int)(4e6/(tot+1)); if(reps<20)reps=20; if(reps>4000)reps=4000;
    for(int w=0;w<5;w++){ burst(pd,re,im,reps); burst(pn,re,im,reps); }   /* warm-up both */
    double bd=1e18,bn=1e18;
    for(int r=0;r<5;r++){                       /* interleaved rounds, min-of-5, cooldown */
        double d=burst(pd,re,im,reps); if(d<bd)bd=d;
        Sleep(10);
        double n=burst(pn,re,im,reps); if(n<bn)bn=n;
        Sleep(10);
    }
    printf("N=%-5d K=%-4zu  scrambled=%8.0f ns  natural=%8.0f ns  tax=%.2fx  mode=%s\n",
           N,K,bd,bn,bn/bd,mode_name(mode,nf));
    fprintf(csv,"%d,%zu,%.0f,%.0f,%.3f,%d,%d,%s\n",N,(size_t)K,bd,bn,bn/bd,mode,nf,mode_name(mode,nf));
    fflush(csv);
    vfft_destroy(pd); vfft_destroy(pn); free(re);free(im);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1<<2);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    CreateDirectoryA(WISDIR,NULL);            /* persist wisdom (chosen mode survives) */
    putenv("VFFT_WISDOM_DIR=" WISDIR);
    FILE *csv=fopen(CSVOUT,"w");
    if(csv) fprintf(csv,"N,K,scrambled_ns,natural_ns,tax,nat_mode,nat_nf,mode\n");
    printf("# 1D natural-order reorder tax (natural fwd / scrambled fwd, same base plan) -> %s\n",CSVOUT);
    printf("# FIX beneficiaries (palindromic chain -> opportunistic PSWAP):\n");
    cell(csv,256,4);      /* 16·16 palindrome — THE fixed flip cell */
    cell(csv,512,4);      /* check marker */
    cell(csv,1024,4);
    printf("# UNAFFECTED by the fix (non-palindromic -> PURE, for contrast):\n");
    cell(csv,256,32);
    cell(csv,1024,32);
    cell(csv,64,64);
    if(csv) fclose(csv);
    printf("# CSV written: %s\n",CSVOUT);
    return 0;
}
