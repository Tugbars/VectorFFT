/* _reach_c2r_route.c — does 1D c2r OOP K>1 (split AND interleaved) reach the
 * §W2 route race?  races = _vfft_create_race_count via the fingerprint counters. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t,lay,place,rig; int n0; size_t K; } cell_t;
static const cell_t C[] = {
 {"c2r.sp.oop.1024.K8.PATIENT",  VFFT_C2R,VFFT_LAYOUT_SPLIT,       VFFT_OUTOFPLACE,VFFT_PATIENT,1024,8},
 {"c2r.il.oop.1024.K8.PATIENT",  VFFT_C2R,VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE,VFFT_PATIENT,1024,8},
 {"c2r.sp.oop.1024.K8.MEASURE",  VFFT_C2R,VFFT_LAYOUT_SPLIT,       VFFT_OUTOFPLACE,VFFT_MEASURE,1024,8},
 {"c2r.il.oop.1024.K8.MEASURE",  VFFT_C2R,VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE,VFFT_MEASURE,1024,8},
 {"c2r.sp.oop.1024.K1.PATIENT",  VFFT_C2R,VFFT_LAYOUT_SPLIT,       VFFT_OUTOFPLACE,VFFT_PATIENT,1024,1},
 {"c2r.il.oop.1024.K1.PATIENT",  VFFT_C2R,VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE,VFFT_PATIENT,1024,1},
 {"c2r.sp.ip.1024.K8.PATIENT",   VFFT_C2R,VFFT_LAYOUT_SPLIT,       VFFT_INPLACE,   VFFT_PATIENT,1024,8},
 {"c2r.il.ip.1024.K8.PATIENT",   VFFT_C2R,VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,   VFFT_PATIENT,1024,8},
 {"c2r.sp.oop.1024.K256.PATIENT",VFFT_C2R,VFFT_LAYOUT_SPLIT,       VFFT_OUTOFPLACE,VFFT_PATIENT,1024,256},
 {"r2c.sp.oop.1024.K8.PATIENT",  VFFT_R2C,VFFT_LAYOUT_SPLIT,       VFFT_OUTOFPLACE,VFFT_PATIENT,1024,8},
 {"r2c.il.oop.1024.K8.PATIENT",  VFFT_R2C,VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE,VFFT_PATIENT,1024,8},
};
#define NC ((int)(sizeof C/sizeof C[0]))
int main(int argc,char**argv){
  static char buf[65536]; long ct[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
  if(argc>1&&!strcmp(argv[1],"--list")){for(i=0;i<NC;i++)printf("%2d %s\n",i,C[i].tag);return 0;}
  if(argc<3||strcmp(argv[1],"--cell")){printf("usage: --cell <0..%d>|--list\n",NC-1);return 2;}
  i=atoi(argv[2]); if(i<0||i>=NC){printf("range\n");return 2;}
  memset(&cfg,0,sizeof cfg);
  cfg.transform=(vfft_transform_t)C[i].t; cfg.placement=(vfft_placement_t)C[i].place;
  cfg.layout=(vfft_layout_t)C[i].lay; cfg.order=VFFT_ORDER_DEFAULT; cfg.dims=1;
  cfg.n[0]=C[i].n0; cfg.howmany=C[i].K; cfg.rigor=(vfft_rigor_t)C[i].rig; cfg.wisdom_write=getenv("WW")?1:0;
  p=vfft_create(&cfg); vfft__fp_counters(ct);
  printf("@@cell %s\n",C[i].tag);
  if(!p){printf("@@status refuse races=%ld\n",ct[5]);return 0;}
  printf("@@status accept races=%ld\n",ct[5]);
  vfft__fingerprint(p,buf,sizeof buf); fputs(buf,stdout); vfft_destroy(p); return 0;
}
