/* isolate_2d_r2c_128.c — pinpoint the 2D r2c crash at 128x128 (stdout checkpoints). */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include "core/fft2d_r2c.h"
#include "core/env.h"
#include "generator/generated/registry.h"
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#endif

static void cell(int N1,int N2){
    size_t B=8; if(B>(size_t)N1)B=(size_t)N1;
    size_t hp1=(size_t)(N2/2+1), K_pad=((hp1+3)/4)*4;
    printf("=== %dx%d  B=%zu hp1=%zu K_pad=%zu ===\n",N1,N2,B,hp1,K_pad);fflush(stdout);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_plan_t *inner=vfft_proto_auto_plan(N2/2,B,&reg,NULL);
    printf("  inner=%p stages=%d\n",(void*)inner,inner?inner->num_stages:-1);fflush(stdout);
    stride_plan_t *plan_r2c=inner?stride_r2c_plan(N2,B,B,inner):NULL;
    printf("  plan_r2c=%p\n",(void*)plan_r2c);fflush(stdout);
    stride_plan_t *plan_col=vfft_proto_auto_plan(N1,K_pad,&reg,NULL);
    printf("  plan_col=%p stages=%d\n",(void*)plan_col,plan_col?plan_col->num_stages:-1);fflush(stdout);
    if(!plan_r2c||!plan_col){printf("  sub NULL\n");return;}
    stride_plan_t *p=stride_plan_2d_r2c_from(N1,N2,B,K_pad,plan_r2c,plan_col);
    printf("  p=%p (2d_from done)\n",(void*)p);fflush(stdout);
    if(!p)return;
    size_t RN=(size_t)N1*N2,CN=(size_t)N1*hp1;
    double *x=AALLOC(RN*8),*orr=AALLOC(CN*8),*oii=AALLOC(CN*8);
    for(size_t i=0;i<RN;i++)x[i]=0.1*(double)(i%97)-1.0;
    printf("  buffers ok, fwd...\n");fflush(stdout);
    stride_execute_2d_r2c(p,x,orr,oii);
    printf("  fwd OK\n");fflush(stdout);
    double *xr=AALLOC(RN*8);
    stride_execute_2d_c2r(p,orr,oii,xr);
    printf("  c2r OK\n");fflush(stdout);
    stride_plan_destroy(p);
}
int main(void){
    stride_env_init();
    cell(64,64);
    cell(128,128);
    cell(256,256);
    printf("ALL DONE\n");
    return 0;
}
