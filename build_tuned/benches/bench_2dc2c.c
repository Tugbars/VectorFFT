/* c2c-2D control: vfft in-place scrambled (v1.0 contract) vs MKL complex split /
 * complex IL / real CCE — one table reconciling v1.0 vs §6a29. */
#include "src/core/vfft.c"
#include "mkl_dfti.h"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med(double *v,int n){ qsort(v,n,8,dcmp); return v[n/2]; }
int main(int argc,char**argv){
    int N1=argc>1?atoi(argv[1]):256, N2=argc>2?atoi(argv[2]):256;
    size_t P=(size_t)N1*N2, H2=(size_t)N2/2+1;
    int L=(int)(3e7/((double)N1*N2)); if(L<5)L=5; if(L>300)L=300;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_PATIENT;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    vfft_plan pc=vfft_create(&cf);
    if(!pc){puts("c2c 2D create fail");return 1;}
    double *re=aligned_alloc(64,P*8),*im=aligned_alloc(64,P*8);
    double *re0=aligned_alloc(64,P*8),*im0=aligned_alloc(64,P*8);
    srand(41); for(size_t i=0;i<P;i++){re0[i]=2.0*rand()/RAND_MAX-1;im0[i]=2.0*rand()/RAND_MAX-1;}
    MKL_LONG dims[2]={N1,N2};
    DFTI_DESCRIPTOR_HANDLE ms,mi,mr;
    DftiCreateDescriptor(&ms,DFTI_DOUBLE,DFTI_COMPLEX,2,dims);
    DftiSetValue(ms,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
    DftiSetValue(ms,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiCommitDescriptor(ms);
    DftiCreateDescriptor(&mi,DFTI_DOUBLE,DFTI_COMPLEX,2,dims);
    DftiSetValue(mi,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiCommitDescriptor(mi);
    MKL_LONG is[3]={0,(MKL_LONG)N2,1}, os[3]={0,(MKL_LONG)H2,1};
    DftiCreateDescriptor(&mr,DFTI_DOUBLE,DFTI_REAL,2,dims);
    DftiSetValue(mr,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mr,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(mr,DFTI_INPUT_STRIDES,is); DftiSetValue(mr,DFTI_OUTPUT_STRIDES,os);
    DftiCommitDescriptor(mr);
    double *mre=aligned_alloc(64,P*8),*mim=aligned_alloc(64,P*8);
    MKL_Complex16 *zi=aligned_alloc(64,P*sizeof(MKL_Complex16));
    MKL_Complex16 *zo=aligned_alloc(64,P*sizeof(MKL_Complex16));
    MKL_Complex16 *zr=aligned_alloc(64,(size_t)N1*H2*sizeof(MKL_Complex16));
    for(size_t i=0;i<P;i++){zi[i].real=re0[i];zi[i].imag=im0[i];}
    memcpy(re,re0,P*8); memcpy(im,im0,P*8);
    vfft_execute(pc,VFFT_FORWARD,re,im,re,im);
    DftiComputeForward(ms,re0,im0,mre,mim);
    DftiComputeForward(mi,zi,zo);
    DftiComputeForward(mr,re0,zr);
    double tv[9],ts[9],ti[9],tr[9];
    for(int t=0;t<9;t++){
        double t0=bnow();
        for(int i=0;i<L;i++){ memcpy(re,re0,P*8); memcpy(im,im0,P*8);
            vfft_execute(pc,VFFT_FORWARD,re,im,re,im); }
        tv[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeForward(ms,re0,im0,mre,mim);
        ts[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeForward(mi,zi,zo);
        ti[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeForward(mr,re0,zr);
        tr[t]=(bnow()-t0)/L;
    }
    /* vfft arm pays 2 memcpys of P each per iter (in-place needs fresh input);
       report both raw and copy-corrected (copy timed separately). */
    double tc[9];
    for(int t=0;t<9;t++){ double t0=bnow();
        for(int i=0;i<L;i++){ memcpy(re,re0,P*8); memcpy(im,im0,P*8); }
        tc[t]=(bnow()-t0)/L; }
    double V=med(tv,9),C=med(tc,9),S=med(ts,9),I=med(ti,9),R=med(tr,9);
    printf("(%dx%d) c2c-2D control:\n",N1,N2);
    printf("  vfft c2c inplace scrambled = %8.1f us (raw incl. refill) | %8.1f (copy-corrected)\n",V,V-C);
    printf("  MKL complex SPLIT  = %8.1f   -> dag/MKL = %.3fx   (v1.0 config; v1.0 recorded 1.26x at 256^2)\n",S,S/(V-C));
    printf("  MKL complex IL     = %8.1f   -> dag/MKL = %.3fx\n",I,I/(V-C));
    printf("  MKL real CCE       = %8.1f   -> MKL real/complex-split factor = %.2fx\n",R,S/R);
    return 0; }
