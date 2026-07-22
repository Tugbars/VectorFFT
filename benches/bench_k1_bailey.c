#include "src/core/vfft.c"
#include <math.h>
static double bn(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int dc(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(void){
    int Ns[3]={256,1024,4096};
    double scal[3]={1.86,10.19,54.74};
    for(int t=0;t<3;t++){
        int N=Ns[t];
        vfft_config_t cf; memset(&cf,0,sizeof cf);
        cf.transform=VFFT_C2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
        cf.dims=1; cf.n[0]=N; cf.howmany=1; cf.order=VFFT_ORDER_NATURAL;
        vfft_plan p=vfft_create(&cf);
        if(!p){ printf("N=%d nat OOP create NULL\n",N); continue; }
        struct vfft_plan_s *h=(struct vfft_plan_s*)p;
        const char *kn = h->oplan ? (h->oplan->kind==VFFT_OOP_KIND_BAILEY2?"BAILEY2":
                          h->oplan->kind==VFFT_OOP_KIND_LEAF?"LEAF":
                          h->oplan->kind==VFFT_OOP_KIND_MODEB?"MODEB":"other") : "?";
        double *sr=aligned_alloc(64,(size_t)N*8),*si=aligned_alloc(64,(size_t)N*8);
        double *dr=aligned_alloc(64,(size_t)N*8),*di=aligned_alloc(64,(size_t)N*8);
        srand(5); for(int i=0;i<N;i++){sr[i]=2.0*rand()/RAND_MAX-1;si[i]=2.0*rand()/RAND_MAX-1;}
        vfft_execute(p,VFFT_FORWARD,sr,si,dr,di);
        /* naive check 8 bins (natural order!) */
        double worst=0;
        for(int c=0;c<8;c++){ int k=rand()%N; double xr=0,xi=0;
            for(int n=0;n<N;n++){double a=-2.0*M_PI*(double)k*n/N;
                xr+=sr[n]*cos(a)-si[n]*sin(a); xi+=sr[n]*sin(a)+si[n]*cos(a);}
            double d1=fabs(dr[k]-xr),d2=fabs(di[k]-xi);
            if(d1>worst)worst=d1; if(d2>worst)worst=d2;}
        int L=(int)(200000.0/(N*0.02))+16; double tm[9],t0;
        for(int r=0;r<9;r++){ t0=bn();
            for(int i=0;i<L;i++) vfft_execute(p,VFFT_FORWARD,sr,si,dr,di);
            tm[r]=(bn()-t0)/L; }
        qsort(tm,9,8,dc);
        printf("N=%-5d kind=%-7s natK1=%.2fus  vs scalar-tier %.2fus (%+.0f%%)  naive=%.1e %s\n",
            N,kn,tm[4],scal[t],100*(tm[4]-scal[t])/scal[t],worst,worst<1e-9?"OK":"**BAD**");
        vfft_destroy(p); free(sr);free(si);free(dr);free(di);
    }
    return 0; }
