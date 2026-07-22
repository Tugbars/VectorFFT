#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "il_execute.h"
#include "jit_runtime.h"
#include "generator/generated/registry.h"
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    if(vfft_proto_wisdom_load(w,"/mnt/user-data/uploads/spike_wisdom.txt")){puts("wload");return 1;}
    int cells[2][2]={{100,4},{1024,4}};
    for(int c=0;c<2;c++){
        vfft_proto_handle_t h;
        if(vfft_proto_plan(&h,cells[c][0],(size_t)cells[c][1],VFFT_PROTO_WISDOM_ONLY,&reg,w,NULL)){printf("plan fail %d\n",c);continue;}
        vfft_proto_exec_range_fn rf=vfft_proto_plan_jit_fwd_range(h.plan);
        vfft_proto_exec_range_fn rb=vfft_proto_plan_jit_bwd_range(h.plan);
        printf("(%d,%d) exec_fwd=%p exec_bwd=%p range_fwd=%p range_bwd=%p\n",
               cells[c][0],cells[c][1],(void*)h.exec_fwd,(void*)h.exec_bwd,(void*)rf,(void*)rb);
    }
    return 0;
}
