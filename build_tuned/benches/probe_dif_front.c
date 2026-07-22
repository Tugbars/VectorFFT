#include "src/core/vfft.c"
void radix64_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
static void dif2_front_oop(const double *sre, const double *sim, double *dre, double *dim,
                           size_t rs, size_t R, size_t N,
                           const double *twr, const double *twi){
    const size_t h=N/2;
    for(size_t r=0;r<R;r++){
        const double *xr=sre+r*rs,*xi=sim+r*rs; double *yr=dre+r*rs,*yi=dim+r*rs;
        for(size_t c=0;c<h;c+=4){
            __m256d ar=_mm256_loadu_pd(xr+c),ai=_mm256_loadu_pd(xi+c);
            __m256d br=_mm256_loadu_pd(xr+c+h),bi=_mm256_loadu_pd(xi+c+h);
            __m256d sr=_mm256_add_pd(ar,br),si=_mm256_add_pd(ai,bi);
            __m256d dr=_mm256_sub_pd(ar,br),di=_mm256_sub_pd(ai,bi);
            __m256d wr=_mm256_loadu_pd(twr+c),wi2=_mm256_loadu_pd(twi+c);
            _mm256_storeu_pd(yr+c,sr); _mm256_storeu_pd(yi+c,si);
            _mm256_storeu_pd(yr+c+h,_mm256_fmsub_pd(dr,wr,_mm256_mul_pd(di,wi2)));
            _mm256_storeu_pd(yi+c+h,_mm256_fmadd_pd(dr,wi2,_mm256_mul_pd(di,wr)));
        } } }
int main(int argc,char**argv){
    size_t R=argc>1?(size_t)atoi(argv[1]):4096; enum { N=128, H=64 };
    double *xr=aligned_alloc(64,R*N*8),*xi=aligned_alloc(64,R*N*8);
    double *wr=aligned_alloc(64,R*N*8),*wi=aligned_alloc(64,R*N*8);
    double *twr=aligned_alloc(64,H*8),*twi=aligned_alloc(64,H*8);
    for(int c=0;c<H;c++){ double a=-2.0*M_PI*c/N; twr[c]=cos(a); twi[c]=sin(a); }
    srand(113); for(size_t i=0;i<R*N;i++){ xr[i]=2.0*rand()/RAND_MAX-1; xi[i]=2.0*rand()/RAND_MAX-1; }
    /* arm A: tiled (transpose + native inner + transpose), the incumbent */
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=8; cf.wisdom=vfft_wisdom_load("/tmp/wbr2c3");
    vfft_plan p8=vfft_create(&cf);
    double lmr[N*8], lmi[N*8];
    int L=(int)(3e7/((double)R*N)); if(L<4)L=4;
    double ta[9],tb[9],tr[9];
    for(int t=0;t<9;t++){
        double t0=bnow();
        for(int i=0;i<L;i++)
            for(size_t b0=0;b0<R;b0+=8){
                stride_transpose(xr+b0*N,N,lmr,8,8,N);
                stride_transpose(xi+b0*N,N,lmi,8,8,N);
                vfft_execute(p8,VFFT_FORWARD,lmr,lmi,NULL,NULL);
                stride_transpose(lmr,8,wr+b0*N,N,N,8);
                stride_transpose(lmi,8,wi+b0*N,N,N,8);
            }
        ta[t]=(bnow()-t0)/L;
        t0=bnow();
        for(int i=0;i<L;i++){   /* plain 2-sweep composition */
            dif2_front_oop(xr,xi,wr,wi,N,R,N,twr,twi);
            radix64_n1_fwd_avx2_strided(wr,wi,NULL,NULL,N,R);
            radix64_n1_fwd_avx2_strided(wr+H,wi+H,NULL,NULL,N,R);
        }
        tb[t]=(bnow()-t0)/L;
        t0=bnow();
        for(int i=0;i<L;i++)    /* row-blocked fused: one DRAM pass */
            for(size_t b0=0;b0<R;b0+=8){
                dif2_front_oop(xr+b0*N,xi+b0*N,wr+b0*N,wi+b0*N,N,8,N,twr,twi);
                radix64_n1_fwd_avx2_strided(wr+b0*N,wi+b0*N,NULL,NULL,N,8);
                radix64_n1_fwd_avx2_strided(wr+b0*N+H,wi+b0*N+H,NULL,NULL,N,8);
            }
        tr[t]=(bnow()-t0)/L;
    }
    printf("R=%zu: A(tiled)=%.0f  B(2-sweep)=%.0f (%+.0f%%)  C(row-blocked fused)=%.0f (%+.0f%%)\n",
        R,med9(ta),med9(tb),100*(med9(tb)-med9(ta))/med9(ta),
        med9(tr),100*(med9(tr)-med9(ta))/med9(ta));
    return 0; }
