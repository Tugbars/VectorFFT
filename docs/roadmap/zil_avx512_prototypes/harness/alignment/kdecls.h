struct row { const char *name; int R; kfn f; };
void A_avx2_radix16_z_n1_fwd_avx2(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_ladder_radix16_z_n1_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_avx2_radix16_z_t2_fwd_avx2(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_ladder_radix16_z_t2_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_avx2_radix32_z_n1_fwd_avx2(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_ladder_radix32_z_n1_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_avx2_radix8_z_t2_fwd_avx2(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void A_ladder_radix8_z_t2_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
static struct row rows[] = {
  { "n1_16 avx2(ymm)", 16, A_avx2_radix16_z_n1_fwd_avx2 },
  { "n1_16 avx512(zmm)", 16, A_ladder_radix16_z_n1_fwd_avx512 },
  { "t2_16 avx2(ymm)", 16, A_avx2_radix16_z_t2_fwd_avx2 },
  { "t2_16 avx512(zmm)", 16, A_ladder_radix16_z_t2_fwd_avx512 },
  { "n1_32 avx2(ymm)", 32, A_avx2_radix32_z_n1_fwd_avx2 },
  { "n1_32 avx512(zmm)", 32, A_ladder_radix32_z_n1_fwd_avx512 },
  { "t2_8 avx2(ymm)", 8, A_avx2_radix8_z_t2_fwd_avx2 },
  { "t2_8 avx512(zmm)", 8, A_ladder_radix8_z_t2_fwd_avx512 },
};
