#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "r2c.h"
#include "../generator/generated/registry.h"

static void test(int halfN, const int* f, int nf, vfft_proto_registry_t* reg, const char* label){
  stride_plan_t* inner = vfft_proto_plan_create_ex(halfN, 8, f, NULL, nf, 0, reg);
  if(!inner){ printf("  %-10s inner NULL (codelet for this radix may be absent)\n", label); return; }
  printf("  %-10s factors=[", label);
  for(int i=0;i<nf;i++) printf("%d ", f[i]); printf("] -> ");
  stride_plan_t* p = stride_r2c_plan(halfN*2, 8, 8, inner);
  printf("%s\n", p ? "ACCEPTED" : "REFUSED");
  if(p) stride_plan_destroy(p);
}
int main(void){
  vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
  int a[]={8,16};   test(128, a, 2, &reg, "(8,16)");
  int b[]={16,8};   test(128, b, 2, &reg, "(16,8)");
  int c[]={16,16};  test(256, c, 2, &reg, "(16,16)");
  int d[]={8,32};   test(256, d, 2, &reg, "(8,32)");
  int e[]={4,64};   test(256, e, 2, &reg, "(4,64)");
  return 0;
}
