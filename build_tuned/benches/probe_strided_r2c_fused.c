#include "src/core/vfft.c"
void radix16_r2c_fwd_avx2_strided(const double*,double*,double*,size_t,size_t,size_t);
void radix16_r2c_fwd_avx2_strided_EMIT(const double*,double*,double*,const double*,const double*,size_t,size_t,size_t);
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
int main(int argc,char**argv){
    int R=argc>1?atoi(argv[1]):4096;
    enum { N=16, H=9 };
    double *x=aligned_alloc(64,(size_t)R*N*8);
    double *rowre=aligned_alloc(64,(size_t)R*H*8), *rowim=aligned_alloc(64,(size_t)R*H*8);
    double *cre=aligned_alloc(64,(size_t)R*H*8), *cim=aligned_alloc(64,(size_t)R*H*8);
    double are[9*8], aim[9*8], lm[N*8];
    srand(99); for(int i=0;i<R*N;i++) x[i]=2.0*rand()/RAND_MAX-1;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=8; cf.wisdom=w;
    vfft_plan p1=vfft_create(&cf);
    for(int t0=0;t0<R;t0+=8){
        stride_transpose(x+(size_t)t0*N,N,lm,8,8,N);
        vfft_execute(p1,VFFT_FORWARD,lm,NULL,are,aim);
        for(int f=0;f<H;f++) for(int r2=0;r2<8;r2++){
            rowre[(size_t)(t0+r2)*H+f]=are[f*8+r2]; rowim[(size_t)(t0+r2)*H+f]=aim[f*8+r2]; }
    }
    radix16_r2c_fwd_avx2_strided(x, cre, cim, N, H, (size_t)R/2);
    double mx=0; for(size_t i=0;i<(size_t)R*H;i++){
        double d1=fabs(cre[i]-rowre[i]),d2=fabs(cim[i]-rowim[i]);
        if(d1>mx)mx=d1; if(d2>mx)mx=d2; }
    printf("R=%d fused gate: max|fused - native| = %.3e %s\n",R,mx,mx<1e-13?"PASS":"**FAIL**");
    {   /* emitted vs hand: expect BIT (same ops, same order) */
        double *ere=aligned_alloc(64,(size_t)R*H*8), *eim=aligned_alloc(64,(size_t)R*H*8);
        radix16_r2c_fwd_avx2_strided_EMIT(x, ere, eim, (const double*)0, (const double*)0, N, H, (size_t)R/2);
        size_t bad=0; for(size_t i=0;i<(size_t)R*H;i++)
            if(ere[i]!=cre[i]||eim[i]!=cim[i]) bad++;
        printf("R=%d emitted-vs-hand: %s (%zu diffs)\n",R,bad?"**DIFF**":"BIT-IDENTICAL",bad);
        free(ere); free(eim);
    }
    if(mx>=1e-13){
        printf("row0 native: "); for(int f=0;f<5;f++) printf("(%.4f,%.4f) ",rowre[f],rowim[f]);
        printf("\nrow0 fused:  "); for(int f=0;f<5;f++) printf("(%.4f,%.4f) ",cre[f],cim[f]);
        printf("\nrow1 native: "); for(int f=0;f<5;f++) printf("(%.4f,%.4f) ",rowre[H+f],rowim[H+f]);
        printf("\nrow1 fused:  "); for(int f=0;f<5;f++) printf("(%.4f,%.4f) ",cre[H+f],cim[H+f]);
        printf("\n"); return 1; }
    int L=(int)(4e6/((double)R*N)); if(L<20)L=20;
    double ta[9],tc[9],te2[9];
    for(int t2=0;t2<9;t2++){
        double t0=bnow();
        for(int i=0;i<L;i++)
            for(int b0=0;b0<R;b0+=8){
                stride_transpose(x+(size_t)b0*N,N,lm,8,8,N);
                vfft_execute(p1,VFFT_FORWARD,lm,NULL,are,aim);
                for(int f=0;f<H;f++) for(int r2=0;r2<8;r2++){
                    rowre[(size_t)(b0+r2)*H+f]=are[f*8+r2]; rowim[(size_t)(b0+r2)*H+f]=aim[f*8+r2]; }
            }
        ta[t2]=(bnow()-t0)/L;
        t0=bnow();
        for(int i=0;i<L;i++)
            radix16_r2c_fwd_avx2_strided(x, cre, cim, N, H, (size_t)R/2);
        tc[t2]=(bnow()-t0)/L;
        t0=bnow();
        for(int i=0;i<L;i++)
            radix16_r2c_fwd_avx2_strided_EMIT(x, cre, cim, (const double*)0, (const double*)0, N, H, (size_t)R/2);
        te2[t2]=(bnow()-t0)/L;
    }
    double A=med9(ta),C=med9(tc);
    printf("A (tiled v1 composition)      = %8.2f us\n",A);
    printf("C (hand reference)            = %8.2f us  (%+.1f%%)\n",C,100*(C-A)/A);
    { double E2=med9(te2);
      printf("E (EMITTED --strided-r2c)     = %8.2f us  (%+.1f%% vs A, %+.1f%% vs hand)\n",
          E2,100*(E2-A)/A,100*(E2-C)/C); }
    return 0; }
