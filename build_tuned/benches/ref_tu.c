#include "ref.h"
int main(void){
  ref_shape_t s={REF_C2C,REF_LAYOUT_SPLIT_LM,1,512,0,4,4,1,4,1,0};
  if(!ref_shape_check(&s)) return 1;
  ref_sched_t sc={3,0,0}; unsigned p[8];
  for(unsigned r=0;r<6;r++){ sc.round=r; ref_sched_perm(&sc,p);
    printf("rot%%3! r=%u -> %u%u%u\n",r,p[0],p[1],p[2]); }
  sc.n_arms=6; sc.seed=12345;
  for(unsigned r=0;r<3;r++){ sc.round=r; ref_sched_perm(&sc,p);
    printf("rand n=6 r=%u -> %u%u%u%u%u%u\n",r,p[0],p[1],p[2],p[3],p[4],p[5]); }
  char b[256]; printf("csv=%s\n", csv_for("vfft_1d.csv","fftw",b,sizeof b));
  printf("stride(512*4*8=16384)=%zu\n", ref_plane_stride(16384));
  printf("role=%s\n", ref_role_name(REF_ROLE_HOME));
  return 0;
}
