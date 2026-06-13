#include <stdio.h>
#include <stdlib.h>
#include "fftw3.h"
int main(int argc,char**argv){
  int K=argc>1?atoi(argv[1]):512, N=1024;
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*(size_t)N*K),*go=fftw_malloc(sizeof(fftw_complex)*(size_t)N*K);
  int nn[1]={N};
  fftw_plan pe=fftw_plan_many_dft(1,nn,K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  printf("=== FFTW plan  N=%d  K=%d  (PATIENT, out-of-place) ===\n",N,K);
  fftw_print_plan(pe); printf("\n\n");
  return 0;
}
