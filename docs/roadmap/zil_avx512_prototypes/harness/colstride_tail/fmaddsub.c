#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double pow2(int e){return ldexp(1.0,e);}
int main(void){ long bad=0, badsub=0; srand(1);
 for(int it=0; it<2000000; it++){ double a[8],y[8],r1[8],r2[8],r3[8];
  for(int i=0;i<8;i++){ a[i]=(rand()/(double)RAND_MAX-0.5)*pow2(rand()%40-20); y[i]=(rand()/(double)RAND_MAX-0.5)*pow2(rand()%40-20);} 
  __m512d A=_mm512_loadu_pd(a), Y=_mm512_permute_pd(_mm512_loadu_pd(y),0x55);
  __m512d M=_mm512_set_pd(0,-0.0,0,-0.0,0,-0.0,0,-0.0); /* lanes hi..lo: im lanes (odd) -0 */
  _mm512_storeu_pd(r1,_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),A,Y));
  for(int i=0;i<8;i+=2){ __m128d a2=_mm_loadu_pd(a+i), y2=_mm_permute_pd(_mm_loadu_pd(y+i),1); _mm_storeu_pd(r2+i,_mm_addsub_pd(a2,y2)); }
  /* current 512 rendering: a - xor(cflip y, _M_IM) with _M_IM = {0,-0,...} */
  __m512d MIM=_mm512_castsi512_pd(_mm512_set_epi64(0x8000000000000000ll,0,0x8000000000000000ll,0,0x8000000000000000ll,0,0x8000000000000000ll,0));
  _mm512_storeu_pd(r3,_mm512_sub_pd(A,_mm512_xor_pd(Y,MIM)));
  if(memcmp(r1,r2,64)) bad++; if(memcmp(r3,r2,64)) badsub++; (void)M; }
 printf("fmaddsub(1,a,cflip y) vs addsub: %ld mismatching vectors; current sub/xor form vs addsub: %ld\n",bad,badsub); return 0;}

