/* build_isa.h — THE build's ISA, decided once.
 *
 * The user picks the ISA when building (CMake -DVFFT_ISA=avx2|avx512, or
 * build.py's option); the build turns it into compiler flags (-mavx512f
 * -mavx512dq for avx512; -mavx2 plus the -mno-avx512f clamp for avx2), and
 * this header reads those flags back. It is the ONE test in the core: the
 * public vfft_isa() (STRIDE_ISA_NAME, env.h) and the IL family's kernel and
 * registry selection (oop/il_isa.h) both take it from here, so what the
 * library reports and what it runs cannot disagree. Build-time only; there is
 * no runtime detection and no fallback from one ISA to another. */
#ifndef VFFT_BUILD_ISA_H
#define VFFT_BUILD_ISA_H

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define VFFT_BUILD_ISA_AVX512 1
#define STRIDE_ISA_NAME "avx512"
#elif defined(__AVX2__)
#define VFFT_BUILD_ISA_AVX2 1
#define STRIDE_ISA_NAME "avx2"
#else
#define STRIDE_ISA_NAME "scalar"
#endif

#endif /* VFFT_BUILD_ISA_H */
