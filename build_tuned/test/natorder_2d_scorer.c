/* natorder_2d_scorer.c — natural-aware planning dev-tool for 2D c2c (column axis).
 *
 * QUESTION: for a 2D cell N1xN2, does the column-axis factorization that minimizes the NATURAL total
 * (col-FFT + dim1 whole-row reorder) differ from the one the SCRAMBLED calibrator picks (min col-FFT)?
 * If a different chain wins for natural, natural-aware planning beats the current "scrambled-optimal +
 * bolt-on reorder" — and by how much.
 *
 * METHOD: force one column chain via a pre-written fft2d_c2c_wisdom.txt (row axis = single-stage [N2]
 * so dim2 is FREE and only dim1/col reorder differs). Measure BOTH orders (DEFAULT=scrambled,
 * NATURAL) for that chain, order-neutralized best-of-N, core-pinned. One chain per process (wisdom is
 * cached after first create). Correctness: natural fwd vs naive separable 2D DFT at a bin subset.
 * Appends a CSV row; the driver loops chains and I pick argmin(scrambled) vs argmin(natural).
 *
 * argv: N1 N2 B  col_nf  f0 f1 ...            (row chain is auto = single-stage [N2])
 * Build: python build.py --src test/natorder_2d_scorer.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#define WISDIR "natorder_2dscore_wis"
#define CSVOUT "natorder_2d_scorer.csv"

static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

/* write fft2d_c2c_wisdom.txt: row + col chains explicit (stage0 FLAT, rest T1S, DIT). B = tile height. */
static void write_wisdom(int N1,int N2,int B,int row_nf,const int *rf,int col_nf,const int *cf){
    char path[700]; snprintf(path,sizeof path,"%s/fft2d_c2c_wisdom.txt",WISDIR);
    FILE *f=fopen(path,"w"); if(!f){ printf("cannot write %s\n",path); exit(2); }
    fprintf(f,"@fft2d_c2c_version 1\n");
    fprintf(f,"%d %d %d",N1,N2,B);
    fprintf(f," %d",row_nf);                          /* row: nf */
    for(int i=0;i<row_nf;i++) fprintf(f," %d",rf[i]);
    for(int i=0;i<row_nf;i++) fprintf(f," %d",i==0?0:2); /* row variants: stage0 FLAT, rest T1S */
    fprintf(f," 0");                                  /* row dif=0 */
    fprintf(f," %d",col_nf);                          /* col: nf */
    for(int i=0;i<col_nf;i++) fprintf(f," %d",cf[i]);
    for(int i=0;i<col_nf;i++) fprintf(f," %d",i==0?0:2); /* col variants: stage0 FLAT, rest T1S */
    fprintf(f," 0");                                  /* col dif=0 (DIT) */
    fprintf(f," 1000.0\n");                            /* best_ns (dummy; lookup ignores) */
    fclose(f);
}

static vfft_plan mk(int N1,int N2,int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=2; c.n[0]=N1; c.n[1]=N2; c.howmany=1; c.nthreads=1; c.order=order;
    return vfft_create(&c);
}
static double burst(vfft_plan p,double *re,double *im,int reps){
    double t0=now_ns(); for(int i=0;i<reps;i++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    return (now_ns()-t0)/reps;
}
/* naive separable 2D DFT (row-major [n1*N2+n2]) at one (k1,k2) bin */
static void dft_bin(const double*x,const double*xi,int N1,int N2,int k1,int k2,double*Xr,double*Xi){
    double ar=0,ai=0;
    for(int n1=0;n1<N1;n1++)for(int n2=0;n2<N2;n2++){
        double a=-2.0*M_PI*((double)k1*n1/N1+(double)k2*n2/N2),c=cos(a),s=sin(a);
        double xr=x[n1*N2+n2],xii=xi[n1*N2+n2]; ar+=xr*c-xii*s; ai+=xr*s+xii*c; }
    *Xr=ar; *Xi=ai;
}
static int is_palin(const int*f,int nf){ for(int i=0;i<nf;i++) if(f[i]!=f[nf-1-i]) return 0; return 1; }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1<<2);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    if(argc<8){ printf("usage: N1 N2 B  row_nf r0.. col_nf c0..\n"); return 2; }
    int N1=atoi(argv[1]),N2=atoi(argv[2]),B=atoi(argv[3]);
    int ai=4;
    int row_nf=atoi(argv[ai++]); int rf[8]={0}; for(int i=0;i<row_nf&&i<8;i++) rf[i]=atoi(argv[ai++]);
    int col_nf=atoi(argv[ai++]); int cf[8]={0}; for(int i=0;i<col_nf&&i<8;i++) cf[i]=atoi(argv[ai++]);
    char chainstr[64]={0}; for(int i=0;i<col_nf;i++){ char t[12]; snprintf(t,sizeof t,i?"·%d":"%d",cf[i]); strcat(chainstr,t); }
    char rowstr[64]={0}; for(int i=0;i<row_nf;i++){ char t[12]; snprintf(t,sizeof t,i?"·%d":"%d",rf[i]); strcat(rowstr,t); }

    CreateDirectoryA(WISDIR,NULL);
    write_wisdom(N1,N2,B,row_nf,rf,col_nf,cf);
    putenv("VFFT_WISDOM_DIR=" WISDIR);

    size_t tot=(size_t)N1*N2;
    double *x=malloc(tot*8),*xi=malloc(tot*8),*re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ x[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }

    vfft_plan pn=mk(N1,N2,VFFT_ORDER_NATURAL);      /* NATURAL first (reads forced col chain) */
    vfft_plan pd=mk(N1,N2,VFFT_ORDER_DEFAULT);
    if(!pn||!pd){ printf("N1=%d N2=%d col=%s  plan NULL\n",N1,N2,chainstr); return 1; }

    /* correctness: natural fwd vs naive 2D DFT at 24 pseudo-random bins */
    memcpy(re,x,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(pn,VFFT_FORWARD,re,im,re,im);
    double emax=0,sc=0;
    for(int t=0;t<24;t++){ int k1=(t*37+5)%N1,k2=(t*19+3)%N2; double Xr,Xi; dft_bin(x,xi,N1,N2,k1,k2,&Xr,&Xi);
        double d1=fabs(re[k1*N2+k2]-Xr),d2=fabs(im[k1*N2+k2]-Xi); if(d1>emax)emax=d1; if(d2>emax)emax=d2;
        if(fabs(Xr)>sc)sc=fabs(Xr); }
    emax/=(sc>0?sc:1);

    int reps=(int)(6e6/(tot+1)); if(reps<20)reps=20; if(reps>3000)reps=3000;
    for(int w=0;w<6;w++){ burst(pd,re,im,reps); burst(pn,re,im,reps); }
    double bd=1e18,bn=1e18;
    for(int r=0;r<6;r++){ double d=burst(pd,re,im,reps); if(d<bd)bd=d; Sleep(10);
                          double n=burst(pn,re,im,reps); if(n<bn)bn=n; Sleep(10); }
    int pal=is_palin(cf,col_nf);
    printf("N1=%-4d N2=%-4d row=%-8s col=%-10s %s  scrambled=%.0f  natural=%.0f  tax=%.2fx  err=%.1e %s\n",
           N1,N2,rowstr,chainstr,pal?"[palin]":"       ",bd,bn,bn/bd,emax,emax<1e-8?"ok":"<FAIL>");
    FILE *csv=fopen(CSVOUT,"a");
    if(csv){ fprintf(csv,"%d,%d,%s,%s,%d,%.0f,%.0f,%.3f,%.1e\n",N1,N2,rowstr,chainstr,pal,bd,bn,bn/bd,emax); fclose(csv); }
    vfft_destroy(pn); vfft_destroy(pd);
    return 0;
}
