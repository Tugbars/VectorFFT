/* diag_plan.c — print vfft_plan_print for selected N values */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2   = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

static inline vfft_isa_level_t vfft_detect_isa(void)
{
#if defined(__AVX512F__)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"

#define vfft_detect_isa _diag_detect_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa

#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"

#define vfft_detect_isa _diag_detect_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"

#include "fft_radix2_dif_dispatch.h"
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix16_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"
#include "fft_radix10_dif_dispatch.h"
#include "fft_radix25_dif_dispatch.h"

#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"

#include "fft_radix64_n1.h"
#include "fft_radix128_n1.h"

#include "vfft_planner.h"

#define vfft_detect_isa _diag_detect_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

int main(void)
{
    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    static const size_t Ns[] = { 4000, 8000 };
    for (int i = 0; i < 2; i++) {
        size_t N = Ns[i];
        printf("=== N=%zu ===\n", N);
        vfft_plan *p = vfft_plan_create(N, &reg);
        if (!p) { printf("  PLAN FAILED\n\n"); continue; }
        vfft_plan_print(p);
        vfft_plan_destroy(p);
        printf("\n");
    }
    return 0;
}
