#include "src/core/vfft.c"
#include <immintrin.h>
static double bn(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int dc(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void dein_hand(const double*z,double*wr,double*wi,size_t n){
    size_t i=0;
    for(; i+4<=n; i+=4){
        __m256d v0=_mm256_loadu_pd(z+2*i), v1=_mm256_loadu_pd(z+2*i+4);
        __m256d t0=_mm256_permute2f128_pd(v0,v1,0x20), t1=_mm256_permute2f128_pd(v0,v1,0x31);
        _mm256_storeu_pd(wr+i,_mm256_unpacklo_pd(t0,t1));
        _mm256_storeu_pd(wi+i,_mm256_unpackhi_pd(t0,t1));
    }
    for(; i<n; i++){ wr[i]=z[2*i]; wi[i]=z[2*i+1]; }
}
static void inter_hand(const double*wr,const double*wi,double*z,size_t n){
    size_t i=0;
    for(; i+4<=n; i+=4){
        __m256d r=_mm256_loadu_pd(wr+i), m=_mm256_loadu_pd(wi+i);
        __m256d lo=_mm256_unpacklo_pd(r,m), hi=_mm256_unpackhi_pd(r,m);
        _mm256_storeu_pd(z+2*i,_mm256_permute2f128_pd(lo,hi,0x20));
        _mm256_storeu_pd(z+2*i+4,_mm256_permute2f128_pd(lo,hi,0x31));
    }
    for(; i<n; i++){ z[2*i]=wr[i]; z[2*i+1]=wi[i]; }
}
int main(void){
    enum{N=64}; size_t K=512, NK=(size_t)N*K;
    double *z=aligned_alloc(64,2*NK*8), *zo=aligned_alloc(64,2*NK*8);
    double *wr=aligned_alloc(64,NK*8), *wi=aligned_alloc(64,NK*8);
    double *wr2=aligned_alloc(64,NK*8), *wi2=aligned_alloc(64,NK*8);
    srand(7); for(size_t i=0;i<2*NK;i++) z[i]=2.0*rand()/RAND_MAX-1;
    /* BIT: hand vs scalar must be exact (pure movement) */
    for(size_t j=0;j<NK;j++){ wr[j]=z[2*j]; wi[j]=z[2*j+1]; }
    dein_hand(z,wr2,wi2,NK);
    size_t d=0; for(size_t j=0;j<NK;j++) if(wr[j]!=wr2[j]||wi[j]!=wi2[j]) d++;
    inter_hand(wr2,wi2,zo,NK);
    for(size_t j=0;j<2*NK;j++) if(zo[j]!=z[j]) d++;
    printf("hand-vs-scalar BIT diffs=%zu %s\n",d,d?"**FAIL**":"PASS");
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.layout=VFFT_LAYOUT_INTERLEAVED;
    vfft_plan p=vfft_create(&cf);
    struct vfft_plan_s *h=(struct vfft_plan_s*)p;
    memcpy(zo,z,2*NK*8); vfft_execute(p,VFFT_FORWARD,zo,NULL,zo,NULL);
    printf("plan ns=%d (single-stage => fallback path)\n", h->cplan->num_stages);
    int L=400; double tf[9],tc[9],th[9],t0;
    for(int t=0;t<9;t++){
        t0=bn(); for(int i=0;i<L;i++) vfft_execute(p,VFFT_FORWARD,zo,NULL,zo,NULL);
        tf[t]=(bn()-t0)/L;
        t0=bn(); for(int i=0;i<L;i++){
            for(size_t j=0;j<NK;j++){ wr[j]=z[2*j]; wi[j]=z[2*j+1]; }
            for(size_t j=0;j<NK;j++){ zo[2*j]=wr[j]; zo[2*j+1]=wi[j]; }
        }
        tc[t]=(bn()-t0)/L;
        t0=bn(); for(int i=0;i<L;i++){ dein_hand(z,wr,wi,NK); inter_hand(wr,wi,zo,NK); }
        th[t]=(bn()-t0)/L;
    }
    qsort(tf,9,8,dc); qsort(tc,9,8,dc); qsort(th,9,8,dc);
    printf("(64,512) fallback exec=%.2fus | converts: compiler-loop=%.2fus (%.0f%% of exec)  hand-avx2=%.2fus (hand %+.1f%% vs loop)\n",
        tf[4],tc[4],100*tc[4]/tf[4],th[4],100*(th[4]-tc[4])/tc[4]);
    return 0; }
