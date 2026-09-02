#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
typedef struct { const char *tag; int t,lay,place,ord,n0,n1; } cell_t;
static const cell_t CELLS[] = {
 {"45x64.c2c.oop.NAT",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,45,64},
 {"45x64.c2c.ip.NAT",   VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_INPLACE,   VFFT_ORDER_NATURAL,45,64},
 {"45x64.c2c.oop.DEF",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_DEFAULT,45,64},
 {"45x64.r2c.oop.NAT",  VFFT_R2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,45,64},
 {"45x64.r2c.ip.NAT",   VFFT_R2C,VFFT_LAYOUT_INTERLEAVED,VFFT_INPLACE,   VFFT_ORDER_NATURAL,45,64},
 {"45x64.c2r.oop.NAT",  VFFT_C2R,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,45,64},
 {"127x100.c2c.oop.NAT",VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,127,100},
 {"64x64.c2c.oop.NAT",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,64,64},
 {"256x256.c2c.ip.NAT", VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_INPLACE,   VFFT_ORDER_NATURAL,256,256},
 {"256x256.r2c.oop.NAT",VFFT_R2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,256,256},
 {"45x64.c2c.oop.SCR",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,45,64},
 {"45x64.r2c.oop.SCR",  VFFT_R2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,45,64},
 {"45x64.c2c.oop.NAT.sp",VFFT_C2C,VFFT_LAYOUT_SPLIT,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,45,64},
};
#define NC ((int)(sizeof CELLS/sizeof CELLS[0]))
int main(int argc,char**argv){
  static char buf[65536]; long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
  if(argc<3||strcmp(argv[1],"--cell")){printf("n=%d\n",NC);return 2;}
  i=atoi(argv[2]); if(i<0||i>=NC){printf("range\n");return 2;}
  memset(&cfg,0,sizeof cfg);
  cfg.transform=(vfft_transform_t)CELLS[i].t; cfg.placement=(vfft_placement_t)CELLS[i].place;
  cfg.layout=(vfft_layout_t)CELLS[i].lay; cfg.order=CELLS[i].ord; cfg.dims=2;
  cfg.n[0]=CELLS[i].n0; cfg.n[1]=CELLS[i].n1; cfg.howmany=1;
  cfg.rigor=VFFT_MEASURE; cfg.wisdom_write=0;
  p=vfft_create(&cfg); vfft__fp_counters(c);
  printf("@@cell %s\n",CELLS[i].tag);
  if(!p){printf("@@status refuse races=%ld\n",c[5]);return 0;}
  printf("@@status accept races=%ld\n",c[5]);
  vfft__fingerprint(p,buf,sizeof buf); fputs(buf,stdout);
  vfft_destroy(p); return 0;
}
