/* v2 feasibility spike: standalone engineered transpose vs 8x4 in-register
 * block transpose vs memcpy bound, at the tile shape (N2 x B). The design's
 * load-bearing claim: register-block IO has enough headroom over the
 * standalone pass to fund fusing it into codelet loads/stores. */
#include "src/core/vfft.c"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
/* 8 rows x 4 cols block: load 8 rows of 4, transpose to 4 rows of 8, store. */
static void bt_8x4(const double *src, size_t ld_src, double *dst, size_t ld_dst,
                   size_t R, size_t C){
    for (size_t r = 0; r + 8 <= R; r += 8)
        for (size_t c = 0; c + 4 <= C; c += 4) {
            const double *s = src + r * ld_src + c;
            __m256d a0=_mm256_loadu_pd(s), a1=_mm256_loadu_pd(s+ld_src),
                    a2=_mm256_loadu_pd(s+2*ld_src), a3=_mm256_loadu_pd(s+3*ld_src),
                    a4=_mm256_loadu_pd(s+4*ld_src), a5=_mm256_loadu_pd(s+5*ld_src),
                    a6=_mm256_loadu_pd(s+6*ld_src), a7=_mm256_loadu_pd(s+7*ld_src);
            __m256d t0=_mm256_unpacklo_pd(a0,a1), t1=_mm256_unpackhi_pd(a0,a1);
            __m256d t2=_mm256_unpacklo_pd(a2,a3), t3=_mm256_unpackhi_pd(a2,a3);
            __m256d t4=_mm256_unpacklo_pd(a4,a5), t5=_mm256_unpackhi_pd(a4,a5);
            __m256d t6=_mm256_unpacklo_pd(a6,a7), t7=_mm256_unpackhi_pd(a6,a7);
            __m256d r0=_mm256_permute2f128_pd(t0,t2,0x20), r1=_mm256_permute2f128_pd(t1,t3,0x20);
            __m256d r2=_mm256_permute2f128_pd(t0,t2,0x31), r3=_mm256_permute2f128_pd(t1,t3,0x31);
            __m256d r4=_mm256_permute2f128_pd(t4,t6,0x20), r5=_mm256_permute2f128_pd(t5,t7,0x20);
            __m256d r6=_mm256_permute2f128_pd(t4,t6,0x31), r7=_mm256_permute2f128_pd(t5,t7,0x31);
            double *d0 = dst + c * ld_dst + r;
            _mm256_storeu_pd(d0, r0);           _mm256_storeu_pd(d0+4, r4);
            _mm256_storeu_pd(d0+ld_dst, r1);    _mm256_storeu_pd(d0+ld_dst+4, r5);
            _mm256_storeu_pd(d0+2*ld_dst, r2);  _mm256_storeu_pd(d0+2*ld_dst+4, r6);
            _mm256_storeu_pd(d0+3*ld_dst, r3);  _mm256_storeu_pd(d0+3*ld_dst+4, r7);
        }
}
int main(int argc,char**argv){
    size_t N2=argc>1?(size_t)atoi(argv[1]):256, B=argc>2?(size_t)atoi(argv[2]):8;
    int tiles=argc>3?atoi(argv[3]):32;
    double *src=aligned_alloc(64,N2*B*8), *dst=aligned_alloc(64,N2*B*8);
    srand(71); for(size_t i=0;i<N2*B;i++)src[i]=2.0*rand()/RAND_MAX-1;
    int L=200;
    double ta[9],tb[9],tc[9];
    stride_transpose(src,N2,dst,B,B,N2);  /* warm: B x N2 -> N2 x B */
    for(int t=0;t<9;t++){
        double t0=bnow();
        for(int i=0;i<L*tiles;i++) stride_transpose(src,N2,dst,B,B,N2);
        ta[t]=(bnow()-t0)/L;
        t0=bnow();
        for(int i=0;i<L*tiles;i++) bt_8x4(dst,B,src,N2,N2,B);  /* N2xB -> BxN2 (8-row blocks over N2) */
        tb[t]=(bnow()-t0)/L;
        t0=bnow();
        for(int i=0;i<L*tiles;i++) memcpy(dst,src,N2*B*8);
        tc[t]=(bnow()-t0)/L;
    }
    double A=med9(ta),Bt=med9(tb),C=med9(tc);
    printf("tile %zux%zu x%d tiles/plane:\n",N2,B,tiles);
    printf("  engineered stride_transpose = %7.2f us/plane\n",A);
    printf("  8x4 register-block          = %7.2f us/plane (%+.0f%% vs engineered)\n",Bt,100*(Bt-A)/A);
    printf("  memcpy bound                = %7.2f us/plane  (transpose overhead vs copy: %.2fx / %.2fx)\n",
        C,A/C,Bt/C);
    return 0; }
