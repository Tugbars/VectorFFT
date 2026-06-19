#define _GNU_SOURCE 1
#include <stdio.h>
#include <string.h>
#include "oop_registry_avx512.h"
int main(void){
  oop_codelets_t reg; memset(&reg,0,sizeof reg);
  oop_register_all_avx512(&reg);
  /* spot-check: flat and log3 are DISTINCT and both reachable */
  printf("n1[16]       = %p\n", (void*)reg.n1[16]);
  printf("t1p[16]      = %p\n", (void*)reg.t1p[16]);
  printf("t1p_log3[16] = %p\n", (void*)reg.t1p_log3[16]);
  printf("flat != log3 : %s\n", (void*)reg.t1p[16] != (void*)reg.t1p_log3[16] ? "YES (both reachable)" : "NO (collision!)");
  printf("spec ABI slot: n1_spec[32] = %p\n", (void*)reg.n1_spec[32]);
  /* the spec slot is a DIFFERENT type (7-arg) — verify it compiles as such */
  vfft_oop7_fn s = reg.n1_spec[32];
  vfft_oop11_fn r = reg.t1p[16];
  printf("type-check: spec=7arg reg=11arg both assigned: %s\n", (s&&r)?"OK":"OK(null ptrs fine)");
  return 0;
}
