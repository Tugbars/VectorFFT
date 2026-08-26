/* il2d_rowmt_ab.c — same-run ALTERNATED A/B for the S1-increment-1 experiment.
 * Both plans built in ONE process (env read at create), then timed alternately
 * with pacing.  Reports per-arm min and spread + a control (A vs A).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void)
{ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
  return (double)t.QuadPart*1e9/(double)f.QuadPart; }

static int cmpd(const void *a, const void *b)
{ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

static void arm(int N1,int N2)
{
    const size_t RN=(size_t)N1*N2, hp1=N2/2+1, CN=(size_t)N1*hp1;
    double *x=malloc(RN*8), *za=malloc((2*CN+8)*8), *zb=malloc((2*CN+8)*8);
    double ta[41], tb[41], tc[41];
    vfft_config_t c; vfft_plan pa, pb; size_t i; int r;
    for(i=0;i<RN;i++) x[i]=1.0+1e-6*(double)(i&1023);
    memset(za,0,(2*CN+8)*8); memset(zb,0,(2*CN+8)*8);
    memset(&c,0,sizeof c);
    c.transform=VFFT_R2C; c.placement=VFFT_OUTOFPLACE; c.dims=2;
    c.n[0]=N1; c.n[1]=N2; c.howmany=1;
    c.layout=VFFT_LAYOUT_INTERLEAVED; c.nthreads=8; c.wisdom_write=0;

    /* BOTH arms build the row plan at nthreads=8 so h->nthreads matches and
     * the execute-time vfft_set_num_threads() never churns the OS pool.
     * A differs only by VFFT_NO_TCMT (the clone kill switch) => workers=0. */
    _putenv("VFFT_IL2D_ROWMT=8");
    _putenv("VFFT_NO_TCMT=1");     pa=vfft_create(&c);   /* A = serial rows */
    _putenv("VFFT_NO_TCMT=");      pb=vfft_create(&c);   /* B = MT rows     */
    _putenv("VFFT_IL2D_ROWMT=");
    if(!pa||!pb){printf("%dx%d create failed\n",N1,N2);return;}
    /* warm */
    for(r=0;r<3;r++){ vfft_execute(pa,VFFT_FORWARD,x,NULL,za,NULL);
                      vfft_execute(pb,VFFT_FORWARD,x,NULL,zb,NULL); }
    for(r=0;r<41;r++){
        double t0;
        Sleep(2);
        t0=now_ns(); vfft_execute(pa,VFFT_FORWARD,x,NULL,za,NULL); ta[r]=now_ns()-t0;
        Sleep(2);
        t0=now_ns(); vfft_execute(pb,VFFT_FORWARD,x,NULL,zb,NULL); tb[r]=now_ns()-t0;
        Sleep(2);
        t0=now_ns(); vfft_execute(pa,VFFT_FORWARD,x,NULL,za,NULL); tc[r]=now_ns()-t0;
    }
    qsort(ta,41,8,cmpd); qsort(tb,41,8,cmpd); qsort(tc,41,8,cmpd);
    printf("%5dx%-5d  A(serial rows) med %8.1f us  [min %8.1f]\n",N1,N2,ta[20]*1e-3,ta[0]*1e-3);
    printf("            A'(control)    med %8.1f us  -> control delta %+.1f%%\n",
           tc[20]*1e-3, 100.0*(tc[20]-ta[20])/ta[20]);
    printf("            B(MT rows)     med %8.1f us  -> B/A = %.2fx\n",
           tb[20]*1e-3, ta[20]/tb[20]);
    printf("            bitwise A vs B : %s\n",
           memcmp(za,zb,2*CN*8)==0 ? "IDENTICAL" : "*** DIFFERS ***");
    vfft_destroy(pa); vfft_destroy(pb); free(x); free(za); free(zb);
}

int main(void){ arm(256,256); arm(512,512); arm(1024,1024); arm(8192,64); return 0; }
