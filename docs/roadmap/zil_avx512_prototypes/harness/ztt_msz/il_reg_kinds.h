/* harness: the zsplit kinds' radix lists per ISA (what the generated
 * il_registry_<isa>.h would carry) + extern decls in the frozen z ABI */
#ifndef IL_REG_KINDS_H
#define IL_REG_KINDS_H
#include <stddef.h>
#define _KDECL(name) extern void name(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#if VFFT_IL_VW == 8
#define VFFT_IL_T0TP_FWD_RADICES(X) X(8)
#define VFFT_IL_T0TP_BWD_RADICES(X) X(8)
#define VFFT_IL_TLD_FWD_RADICES(X) X(8)
#define VFFT_IL_TLD_BWD_RADICES(X) X(8)
#else
#define VFFT_IL_T0TP_FWD_RADICES(X) X(4) X(8)
#define VFFT_IL_T0TP_BWD_RADICES(X) X(4) X(8)
#define VFFT_IL_TLD_FWD_RADICES(X) X(4) X(8)
#define VFFT_IL_TLD_BWD_RADICES(X) X(4) X(8)
#endif
#define VFFT_IL_TLF_FWD_RADICES(X) X(4) X(8)
#define VFFT_IL_TLF_BWD_RADICES(X) X(4) X(8)
#define VFFT_IL_TLFI_FWD_RADICES(X) X(4) X(8)
#define VFFT_IL_TLFI_BWD_RADICES(X) X(4) X(8)
#define VFFT_IL_TMG_FWD_RADICES(X) X(3) X(4) X(5) X(7) X(8) X(9) X(15)
#define VFFT_IL_TMG_BWD_RADICES(X) X(3) X(4) X(5) X(7) X(8) X(9) X(15)
#define VFFT_IL_TMGD_FWD_RADICES(X) X(3) X(4) X(5) X(7) X(8) X(9) X(15)
#define VFFT_IL_T0D_FWD_RADICES(X) X(4) X(8)
#define VFFT_IL_MSZ_FWD_RADICES(X) X(3) X(5) X(7) X(9) X(15)
#define _D2(K) _KDECL(_ZTT_CAT3(radix, K, VFFT_ISA_SFX))
#define _ZTT_CAT3_(a, b, c) a##b##c
#define _ZTT_CAT3(a, b, c) _ZTT_CAT3_(a, b, c)
#define _DECLF(R, KD) _KDECL(_ZTT_CAT3(radix##R##_z_, KD, VFFT_ISA_SFX))
#define _DF_t0tp(R) _DECLF(R, t0tp_fwd_) 
#define _DB_t0tp(R) _DECLF(R, t0tp_bwd_)
#define _DF_tmg(R) _DECLF(R, tmg_fwd_)
#define _DB_tmg(R) _DECLF(R, tmg_bwd_)
#define _DF_tlf(R) _DECLF(R, tlf_fwd_)
#define _DB_tlf(R) _DECLF(R, tlf_bwd_)
#define _DF_tlfi(R) _DECLF(R, tlfi_fwd_)
#define _DB_tlfi(R) _DECLF(R, tlfi_bwd_)
#define _DF_tld(R) _DECLF(R, tld_fwd_)
#define _DB_tld(R) _DECLF(R, tld_bwd_)
#define _DF_t0d(R) _DECLF(R, t0d_fwd_)
#define _DF_tmgd(R) _DECLF(R, tmgd_fwd_)
#define _DF_msz(R) _DECLF(R, msz_fwd_)
#define _DB_msz(R) _DECLF(R, msz_bwd_)
#define _DT_mszt(R) _DECLF(R, mszt_bwd_)
VFFT_IL_T0TP_FWD_RADICES(_DF_t0tp) VFFT_IL_T0TP_BWD_RADICES(_DB_t0tp)
VFFT_IL_TMG_FWD_RADICES(_DF_tmg) VFFT_IL_TMG_BWD_RADICES(_DB_tmg)
VFFT_IL_TLF_FWD_RADICES(_DF_tlf) VFFT_IL_TLF_BWD_RADICES(_DB_tlf)
VFFT_IL_TLFI_FWD_RADICES(_DF_tlfi) VFFT_IL_TLFI_BWD_RADICES(_DB_tlfi)
VFFT_IL_TLD_FWD_RADICES(_DF_tld) VFFT_IL_TLD_BWD_RADICES(_DB_tld)
VFFT_IL_T0D_FWD_RADICES(_DF_t0d) VFFT_IL_TMGD_FWD_RADICES(_DF_tmgd)
VFFT_IL_MSZ_FWD_RADICES(_DF_msz) VFFT_IL_MSZ_FWD_RADICES(_DB_msz) VFFT_IL_MSZ_FWD_RADICES(_DT_mszt)
#endif
