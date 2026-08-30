#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc,char**argv){
  static char buf[65536]; long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
  int t=atoi(argv[1]), lay=atoi(argv[2]), pl=atoi(argv[3]), ord=atoi(argv[4]);
  int n0=atoi(argv[5]), n1=atoi(argv[6]);
  memset(&cfg,0,sizeof cfg);
  cfg.transform=(vfft_transform_t)t; cfg.layout=(vfft_layout_t)lay;
  cfg.placement=(vfft_placement_t)pl; cfg.order=ord;
  cfg.dims=n1?2:1; cfg.n[0]=n0; cfg.n[1]=n1; cfg.howmany=1; cfg.rigor=VFFT_MEASURE;
  cfg.wisdom_write=0;
  p=vfft_create(&cfg); vfft__fp_counters(c);
  if(!p){printf("@@refuse races=%ld\n",c[5]);return 0;}
  printf("@@accept races=%ld\n",c[5]);
  vfft__fingerprint(p,buf,sizeof buf); fputs(buf,stdout); vfft_destroy(p); return 0;
}
