#include "src/core/vfft.c"
#include <math.h>
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double bn(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int cellz(int N, size_t K, int expect_pad){
    size_t sz=(size_t)N*K;
    double *z=aligned_alloc(64,((2*sz*8)+63)&~63UL), *za=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    double *zb=aligned_alloc(64,((2*sz*8)+63)&~63UL), *y=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    srand(21+N); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.layout=VFFT_LAYOUT_INTERLEAVED;
    /* the arm decision is FIRST-EXECUTE-lazy: lock each arm with one
       execute while its env is set. */
    setenv("VFFT_IL_PAD","0",1); vfft_plan p0=vfft_create(&cf);
    memcpy(za,z,2*sz*8); vfft_execute(p0,VFFT_FORWARD,za,NULL,za,NULL);
    setenv("VFFT_IL_PAD","1",1); vfft_plan p1=vfft_create(&cf);
    memcpy(zb,z,2*sz*8); vfft_execute(p1,VFFT_FORWARD,zb,NULL,zb,NULL);
    unsetenv("VFFT_IL_PAD");
    struct vfft_plan_s *h0=(struct vfft_plan_s*)p0, *h1=(struct vfft_plan_s*)p1;
    int pad_on = h1->il_me!=(int)K;
    int ok = (pad_on == expect_pad);
    if(!ok) printf("  (%d,%zu) arm=%s expected=%s **FAIL**\n",N,K,pad_on?"Kp":"K",expect_pad?"Kp":"K");
    /* per-arm roundtrips */
    memcpy(y,za,2*sz*8); vfft_execute(p0,VFFT_BACKWARD,y,NULL,y,NULL);
    double r0=0; for(size_t i=0;i<2*sz;i++){double d=fabs(y[i]/N-z[i]); if(d>r0)r0=d;}
    memcpy(y,zb,2*sz*8); vfft_execute(p1,VFFT_BACKWARD,y,NULL,y,NULL);
    double r1=0; for(size_t i=0;i<2*sz;i++){double d=fabs(y[i]/N-z[i]); if(d>r1)r1=d;}
    /* chain match? -> BIT; else sorted-magnitude multiset per lane */
    int chains_eq = 1;
    stride_plan_t *c0=h0->cplan, *c1=h1->cplan_il?h1->cplan_il:h1->cplan;
    if(c0->num_stages!=c1->num_stages) chains_eq=0;
    else for(int s=0;s<c0->num_stages;s++) if(c0->stages[s].radix!=c1->stages[s].radix) chains_eq=0;
    double xchk=0;
    if(chains_eq){
        size_t d=0; for(size_t i=0;i<2*sz;i++) if(za[i]!=zb[i]) d++;
        xchk=(double)d;
        printf("  (%d,%-3zu) arm=%-2s rt=%.1e/%.1e chain-eq BIT diffs=%.0f %s\n",
            N,K,pad_on?"Kp":"K",r0,r1,xchk,(d==0&&r0<1e-12&&r1<1e-12)?"PASS":"**FAIL**");
        ok &= (d==0);
    } else {
        double *m0=malloc(sz*8),*m1=malloc(sz*8);
        for(size_t l=0;l<K;l++){
            for(int k=0;k<N;k++){
                size_t i=(size_t)k*K+l;
                m0[k]=hypot(za[2*i],za[2*i+1]); m1[k]=hypot(zb[2*i],zb[2*i+1]);
            }
            qsort(m0,N,8,dcmp); qsort(m1,N,8,dcmp);
            for(int k=0;k<N;k++){double d=fabs(m0[k]-m1[k]);
                double mm=m0[k]>1?m0[k]:1; if(d/mm>xchk)xchk=d/mm;}
        }
        free(m0);free(m1);
        printf("  (%d,%-3zu) arm=%-2s rt=%.1e/%.1e chain-div sortmag rel=%.1e %s\n",
            N,K,pad_on?"Kp":"K",r0,r1,xchk,(xchk<1e-11&&r0<1e-12&&r1<1e-12)?"PASS":"**FAIL**");
        ok &= (xchk<1e-11);
    }
    ok &= (r0<1e-12 && r1<1e-12);
    vfft_destroy(p0); vfft_destroy(p1);
    free(z);free(za);free(zb);free(y);
    return ok;
}
static int bit_sweep(void){
    /* §6a57: converts vs scalar reference, every epilogue residue. */
    double zr[64], re[32], im[32], z2[64], rr[32], mm[32];
    srand(99); for(int i=0;i<64;i++) zr[i]=2.0*rand()/RAND_MAX-1;
    size_t d=0;
    size_t ns[13]={1,2,3,4,5,6,7,8,9,12,15,16,17};
    for(int t=0;t<13;t++){ size_t n=ns[t];
        for(size_t i=0;i<n;i++){ rr[i]=zr[2*i]; mm[i]=zr[2*i+1]; }
        _vfft_z_dein(zr,re,im,n);
        for(size_t i=0;i<n;i++) if(re[i]!=rr[i]||im[i]!=mm[i]) d++;
        _vfft_z_inter(re,im,z2,n);
        for(size_t i=0;i<2*n;i++) if(z2[i]!=zr[i]) d++;
    }
    printf("  [converts] BIT residue sweep (13 n): %zu diffs %s\n",d,d?"**FAIL**":"PASS");
    return d==0;
}
static int cell_nat(int N, size_t K){
    /* natural-order z: the REAL fallback path (nat_mode!=0 gate). */
    size_t sz=(size_t)N*K;
    double *z=aligned_alloc(64,((2*sz*8)+63)&~63UL), *y=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    srand(5); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.layout=VFFT_LAYOUT_INTERLEAVED; cf.order=VFFT_ORDER_NATURAL;
    vfft_plan p=vfft_create(&cf);
    if(!p){ printf("  [nat z] (%d,%zu) create FAIL\n",N,K); return 0; }
    memcpy(y,z,2*sz*8);
    vfft_execute(p,VFFT_FORWARD,y,NULL,y,NULL);
    vfft_execute(p,VFFT_BACKWARD,y,NULL,y,NULL);
    double rt=0; for(size_t i=0;i<2*sz;i++){double d=fabs(y[i]/N-z[i]); if(d>rt)rt=d;}
    printf("  [nat z] (%d,%-3zu) fallback roundtrip=%.1e %s\n",N,K,rt,rt<1e-12?"PASS":"**FAIL**");
    vfft_destroy(p); free(z); free(y);
    return rt<1e-12;
}
static int cell_verdict(int N, size_t K){
    /* §6a59: unforced cell -> the A/B runs once, stamps, second plan reuses. */
    size_t sz=(size_t)N*K, Kp=((K+7)/8)*8;
    double *z=aligned_alloc(64,((2*sz*8)+63)&~63UL), *y=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    srand(61); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.layout=VFFT_LAYOUT_INTERLEAVED;
    unsetenv("VFFT_IL_PAD");
    int r0=_il_ab_runs;
    vfft_plan p1=vfft_create(&cf);
    memcpy(y,z,2*sz*8); vfft_execute(p1,VFFT_FORWARD,y,NULL,y,NULL);
    struct vfft_plan_s *h1=(struct vfft_plan_s*)p1;
    int v1=h1->il_me, ran1=_il_ab_runs-r0;
    vfft_execute(p1,VFFT_BACKWARD,y,NULL,y,NULL);
    double rt=0; for(size_t i=0;i<2*sz;i++){double d=fabs(y[i]/N-z[i]); if(d>rt)rt=d;}
    vfft_plan p2=vfft_create(&cf);
    memcpy(y,z,2*sz*8); vfft_execute(p2,VFFT_FORWARD,y,NULL,y,NULL);
    int v2=((struct vfft_plan_s*)p2)->il_me, ran2=_il_ab_runs-r0-ran1;
    vfft_proto_wisdom_entry_t *te=vfft_proto_wisdom_lookup(&_default_wisdom()->c2c,N,K);
    int reuse_ok = te ? (ran2==0 && v2==v1) : 1;   /* stamp needs an entry */
    int ok = (ran1==1) && (v1==(int)K||v1==(int)Kp) && rt<1e-12 && reuse_ok;
    printf("  [verdict ] (%d,%-3zu) v=%s ran=%d/%d te=%s reuse=%s rt=%.1e %s\n",
        N,K,v1==(int)Kp?"Kp":"K",ran1,ran2,te?"HIT":"miss",
        reuse_ok?"ok":"**BAD**",rt,ok?"PASS":"**FAIL**");
    vfft_destroy(p1); vfft_destroy(p2); free(z); free(y);
    return ok;
}
int main(void){
    int ok=1;
    ok &= bit_sweep();
    ok &= cell_verdict(200, 20);
    ok &= cell_verdict(1000, 12);
    ok &= cell_nat(100, 7);
    ok &= cell_nat(256, 5);
    ok &= cellz(100, 5, 1);
    ok &= cellz(100, 7, 1);
    ok &= cellz(200, 12, 1);
    ok &= cellz(512, 7, 1);
    ok &= cellz(100, 8, 0);   /* aligned control: tight arm */
    /* same-process perf: env-flip arms, med9 */
    printf("perf (fused-K vs padded-Kp, med9):\n");
    int cells[3][2]={{100,7},{512,7},{1000,12}};
    for(int c=0;c<3;c++){
        int N=cells[c][0]; size_t K=(size_t)cells[c][1], sz=(size_t)N*K;
        double *z=aligned_alloc(64,((2*sz*8)+63)&~63UL);
        srand(3); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
        vfft_config_t cf; memset(&cf,0,sizeof cf);
        cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
        cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.layout=VFFT_LAYOUT_INTERLEAVED;
        setenv("VFFT_IL_PAD","0",1); vfft_plan pK=vfft_create(&cf);
        vfft_execute(pK,VFFT_FORWARD,z,NULL,z,NULL);   /* lock K arm */
        setenv("VFFT_IL_PAD","1",1); vfft_plan pP=vfft_create(&cf);
        vfft_execute(pP,VFFT_FORWARD,z,NULL,z,NULL);   /* lock Kp arm */
        unsetenv("VFFT_IL_PAD");
        int L=300; double tk[9],tp[9],t0;
        for(int t=0;t<9;t++){
            t0=bn(); for(int i=0;i<L;i++) vfft_execute(pK,VFFT_FORWARD,z,NULL,z,NULL); tk[t]=(bn()-t0)/L;
            t0=bn(); for(int i=0;i<L;i++) vfft_execute(pP,VFFT_FORWARD,z,NULL,z,NULL); tp[t]=(bn()-t0)/L;
        }
        qsort(tk,9,8,dcmp); qsort(tp,9,8,dcmp);
        printf("  (%d,%zu): fused-K=%.2fus padded-Kp=%.2fus (pad %+.1f%%)\n",
            N,K,tk[4],tp[4],100*(tp[4]-tk[4])/tk[4]);
        vfft_destroy(pK); vfft_destroy(pP); free(z);
    }
    printf(ok?"IL PAD GATE: ALL PASS\n":"IL PAD GATE: **FAILURES**\n");
    return ok?0:1;
}
