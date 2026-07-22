#include "src/core/vfft.c"
void radix16_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
/* vectorized conj-split: f=1..4 and 5..8-ish handled as one 4-wide chunk + specials.
 * N=16: bins f=0..8. Chunk f={1,2,3,4}: g={15,14,13,12}; f={5,6,7}: tail scalar
 * (or a masked 4-chunk with g={11,10,9,8} covering f={5,6,7,8} incl. Nyquist). */
static void vep(const double *zr, const double *zi, double *x1r, double *x1i,
                double *x2r, double *x2i){
    /* specials f=0 (g=0) and handled-in-chunks Nyquist f=8 (g=8) */
    x1r[0]=zr[0]; x1i[0]=0.0; x2r[0]=zi[0]; x2i[0]=0.0;
    const __m256d h=_mm256_set1_pd(0.5);
    { __m256d a_r=_mm256_loadu_pd(zr+1), a_i=_mm256_loadu_pd(zi+1);
      __m256d b_r=_mm256_permute4x64_pd(_mm256_loadu_pd(zr+12),0x1B);
      __m256d b_i=_mm256_permute4x64_pd(_mm256_loadu_pd(zi+12),0x1B);
      _mm256_storeu_pd(x1r+1,_mm256_mul_pd(h,_mm256_add_pd(a_r,b_r)));
      _mm256_storeu_pd(x1i+1,_mm256_mul_pd(h,_mm256_sub_pd(a_i,b_i)));
      _mm256_storeu_pd(x2r+1,_mm256_mul_pd(h,_mm256_add_pd(a_i,b_i)));
      _mm256_storeu_pd(x2i+1,_mm256_mul_pd(h,_mm256_sub_pd(b_r,a_r))); }
    { __m256d a_r=_mm256_loadu_pd(zr+5), a_i=_mm256_loadu_pd(zi+5);
      __m256d b_r=_mm256_permute4x64_pd(_mm256_loadu_pd(zr+8),0x1B);
      __m256d b_i=_mm256_permute4x64_pd(_mm256_loadu_pd(zi+8),0x1B);
      _mm256_storeu_pd(x1r+5,_mm256_mul_pd(h,_mm256_add_pd(a_r,b_r)));
      _mm256_storeu_pd(x1i+5,_mm256_mul_pd(h,_mm256_sub_pd(a_i,b_i)));
      _mm256_storeu_pd(x2r+5,_mm256_mul_pd(h,_mm256_add_pd(a_i,b_i)));
      _mm256_storeu_pd(x2i+5,_mm256_mul_pd(h,_mm256_sub_pd(b_r,a_r))); }
}
int main(void){
    enum { N=16, H=9 }; int R=4096;
    double *x=aligned_alloc(64,(size_t)R*N*8), *work=aligned_alloc(64,(size_t)R*N*8);
    double *bre=aligned_alloc(64,(size_t)R*(H+3)*8), *bim=aligned_alloc(64,(size_t)R*(H+3)*8);
    double *sre=aligned_alloc(64,(size_t)R*(H+3)*8), *sim=aligned_alloc(64,(size_t)R*(H+3)*8);
    srand(97); for(int i=0;i<R*N;i++) x[i]=2.0*rand()/RAND_MAX-1;
    memcpy(work,x,(size_t)R*N*8);
    radix16_n1_fwd_avx2_strided(work, work+N, NULL, NULL, 2*N, (size_t)R/2);
    size_t HP=H+3; /* padded row pitch so the 4-wide stores at +5 don't clobber */
    /* correctness: vec vs scalar epilogue on identical mono output */
    for(int wpr=0;wpr<R/2;wpr++){
        const double *zr=work+(size_t)(2*wpr)*N, *zi=work+(size_t)(2*wpr+1)*N;
        double *x1r=sre+(size_t)(2*wpr)*HP, *x1i=sim+(size_t)(2*wpr)*HP;
        double *x2r=sre+(size_t)(2*wpr+1)*HP, *x2i=sim+(size_t)(2*wpr+1)*HP;
        for(int f=0;f<=N/2;f++){ int g=(N-f)&(N-1);
            x1r[f]=0.5*(zr[f]+zr[g]);  x1i[f]=0.5*(zi[f]-zi[g]);
            x2r[f]=0.5*(zi[f]+zi[g]);  x2i[f]=0.5*(zr[g]-zr[f]); }
        vep(zr,zi,bre+(size_t)(2*wpr)*HP,bim+(size_t)(2*wpr)*HP,
            bre+(size_t)(2*wpr+1)*HP,bim+(size_t)(2*wpr+1)*HP);
    }
    double mx=0; for(int wpr=0;wpr<R;wpr++) for(int f=0;f<H;f++){
        double d1=fabs(bre[(size_t)wpr*HP+f]-sre[(size_t)wpr*HP+f]);
        double d2=fabs(bim[(size_t)wpr*HP+f]-sim[(size_t)wpr*HP+f]);
        if(d1>mx)mx=d1; if(d2>mx)mx=d2; }
    printf("vec epilogue gate: max diff = %.3e %s\n",mx,mx==0.0?"BIT":"(eps)");
    int L=200; double te[9];
    for(int t2=0;t2<9;t2++){ double t0=bnow();
        for(int i=0;i<L;i++)
            for(int wpr=0;wpr<R/2;wpr++){
                const double *zr=work+(size_t)(2*wpr)*N, *zi=work+(size_t)(2*wpr+1)*N;
                vep(zr,zi,bre+(size_t)(2*wpr)*HP,bim+(size_t)(2*wpr)*HP,
                    bre+(size_t)(2*wpr+1)*HP,bim+(size_t)(2*wpr+1)*HP); }
        te[t2]=(bnow()-t0)/L; }
    printf("vec epilogue = %.1f us (scalar was 48.9)  -> B_projected(OOP mono + vec ep) = %.1f vs A 79.7 (%+.0f%%)\n",
        med9(te), 31.9+med9(te), 100*(31.9+med9(te)-79.7)/79.7);
    return 0; }
