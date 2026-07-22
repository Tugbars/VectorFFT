/* Probe v2: correct amortization regime. R rows x N=16. Arm A = the faithful
 * v1 tiled composition (8-row tiles: transp-in + (16,8) inner + transp-out to
 * row-major). Arm B = ONE strided two-for-one sweep + epilogue. */
#include "src/core/vfft.c"
void radix16_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
int main(int argc,char**argv){
    int R=argc>1?atoi(argv[1]):256;
    enum { N=16, H=N/2+1 };
    double *x=aligned_alloc(64,(size_t)R*N*8), *work=aligned_alloc(64,(size_t)R*N*8);
    double *are=aligned_alloc(64,(size_t)H*8*8), *aim=aligned_alloc(64,(size_t)H*8*8);
    double *rowre=aligned_alloc(64,(size_t)R*H*8), *rowim=aligned_alloc(64,(size_t)R*H*8);
    double *bre=aligned_alloc(64,(size_t)R*H*8), *bim=aligned_alloc(64,(size_t)R*H*8);
    double lm[N*8];
    srand(93); for(int i=0;i<R*N;i++) x[i]=2.0*rand()/RAND_MAX-1;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=8; cf.wisdom=w;
    vfft_plan p1=vfft_create(&cf);
    /* correctness once */
    for(int t0=0;t0<R;t0+=8){
        stride_transpose(x+(size_t)t0*N,N,lm,8,8,N);
        vfft_execute(p1,VFFT_FORWARD,lm,NULL,are,aim);
        for(int f=0;f<H;f++) for(int r2=0;r2<8;r2++){
            rowre[(size_t)(t0+r2)*H+f]=are[f*8+r2]; rowim[(size_t)(t0+r2)*H+f]=aim[f*8+r2]; }
    }
    memcpy(work,x,(size_t)R*N*8);
    radix16_n1_fwd_avx2_strided(work, work+N, NULL, NULL, 2*N, (size_t)R/2);
    for(int wpr=0;wpr<R/2;wpr++){
        const double *zr=work+(size_t)(2*wpr)*N, *zi=work+(size_t)(2*wpr+1)*N;
        double *x1r=bre+(size_t)(2*wpr)*H, *x1i=bim+(size_t)(2*wpr)*H;
        double *x2r=bre+(size_t)(2*wpr+1)*H, *x2i=bim+(size_t)(2*wpr+1)*H;
        for(int f=0;f<=N/2;f++){ int g=(N-f)&(N-1);
            x1r[f]=0.5*(zr[f]+zr[g]);  x1i[f]=0.5*(zi[f]-zi[g]);
            x2r[f]=0.5*(zi[f]+zi[g]);  x2i[f]=0.5*(zr[g]-zr[f]); } }
    double mx=0; for(size_t i=0;i<(size_t)R*H;i++){ double d1=fabs(bre[i]-rowre[i]),d2=fabs(bim[i]-rowim[i]);
        if(d1>mx)mx=d1; if(d2>mx)mx=d2; }
    printf("R=%d gate: max|strided-native| = %.3e %s\n",R,mx,mx<1e-13?"PASS":"**FAIL**");
    if(mx>=1e-13) return 1;
    int L=(int)(4e6/((double)R*N)); if(L<20)L=20;
    double ta[9],tb[9];
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
        for(int i=0;i<L;i++){
            memcpy(work,x,(size_t)R*N*8);
            radix16_n1_fwd_avx2_strided(work, work+N, NULL, NULL, 2*N, (size_t)R/2);
            for(int wpr=0;wpr<R/2;wpr++){
                const double *zr=work+(size_t)(2*wpr)*N, *zi=work+(size_t)(2*wpr+1)*N;
                double *x1r=bre+(size_t)(2*wpr)*H, *x1i=bim+(size_t)(2*wpr)*H;
                double *x2r=bre+(size_t)(2*wpr+1)*H, *x2i=bim+(size_t)(2*wpr+1)*H;
                for(int f=0;f<=N/2;f++){ int g=(N-f)&(N-1);
                    x1r[f]=0.5*(zr[f]+zr[g]);  x1i[f]=0.5*(zi[f]-zi[g]);
                    x2r[f]=0.5*(zi[f]+zi[g]);  x2i[f]=0.5*(zr[g]-zr[f]); } }
        }
        tb[t2]=(bnow()-t0)/L;
    }
    double A=med9(ta),B=med9(tb);
    printf("A (tiled v1: transp+inner+transp-out) = %8.2f us\n",A);
    printf("B (strided 2for1 + epilogue)          = %8.2f us  (%+.1f%%)\n",B,100*(B-A)/A);
    return 0; }
