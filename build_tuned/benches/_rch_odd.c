#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
typedef struct { const char *tag; int t,lay,place,n; size_t K; } cell_t;
static const cell_t C[] = {
 {"sp.oop.r2c.255.K1",  VFFT_R2C, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE, 255, 1},
 {"il.oop.r2c.255.K1",  VFFT_R2C, VFFT_LAYOUT_INTERLEAVED,  VFFT_OUTOFPLACE, 255, 1},
 {"sp.oop.c2r.255.K1",  VFFT_C2R, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE, 255, 1},
 {"il.oop.c2r.255.K1",  VFFT_C2R, VFFT_LAYOUT_INTERLEAVED,  VFFT_OUTOFPLACE, 255, 1},
 {"sp.oop.r2c.255.K4",  VFFT_R2C, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE, 255, 4},
 {"sp.oop.c2r.255.K4",  VFFT_C2R, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE, 255, 4},
 {"sp.oop.r2c.1024.K4", VFFT_R2C, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE,1024, 4},
 {"sp.oop.c2r.1024.K4", VFFT_C2R, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE,1024, 4},
 {"il.oop.r2c.1024.K4", VFFT_R2C, VFFT_LAYOUT_INTERLEAVED,  VFFT_OUTOFPLACE,1024, 4},
 {"il.oop.c2r.1024.K4", VFFT_C2R, VFFT_LAYOUT_INTERLEAVED,  VFFT_OUTOFPLACE,1024, 4},
 {"sp.oop.r2c.1024.K64",VFFT_R2C, VFFT_LAYOUT_SPLIT,        VFFT_OUTOFPLACE,1024,64},
};
#define NC ((int)(sizeof C/sizeof C[0]))
int main(int argc,char**argv){
  long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
  if(argc<3||strcmp(argv[1],"--cell")){printf("n=%d\n",NC);return 2;}
  i=atoi(argv[2]); if(i<0||i>=NC){printf("oor\n");return 2;}
  memset(&cfg,0,sizeof cfg);
  cfg.transform=(vfft_transform_t)C[i].t; cfg.placement=(vfft_placement_t)C[i].place;
  cfg.layout=(vfft_layout_t)C[i].lay; cfg.order=VFFT_ORDER_DEFAULT;
  cfg.dims=1; cfg.n[0]=C[i].n; cfg.howmany=C[i].K;
  cfg.rigor=VFFT_MEASURE; cfg.wisdom_write=1;
  p=vfft_create(&cfg); vfft__fp_counters(c);
  printf("@@cell %s %s races=%ld\n",C[i].tag,p?"accept":"refuse",c[5]);
  if(p) vfft_destroy(p);
  return 0;
}
