#include <stdio.h>
#include <immintrin.h>
#include <intrin.h>

int main() {
    int info[4];
    __cpuidex(info, 1, 0);
    int osxsave = (info[2] >> 27) & 1;
    printf("CPUID(1) OSXSAVE (ECX bit 27): %d\n", osxsave);

    unsigned long long xcr0 = _xgetbv(0);
    printf("XCR0 = 0x%llx\n", xcr0);
    printf("  opmask (bit 5): %d, ZMM hi (bit 6): %d, ZMM 16-31 (bit 7): %d\n",
        (int)((xcr0 >> 5) & 1), (int)((xcr0 >> 6) & 1), (int)((xcr0 >> 7) & 1));
    int xcr0_ok = ((xcr0 & 0xE6) == 0xE6);
    printf("  XCR0 check (0xE6 mask): %s\n", xcr0_ok ? "PASS" : "FAIL");

    __cpuidex(info, 7, 0);
    int f  = (info[1] >> 16) & 1;
    int dq = (info[1] >> 17) & 1;
    printf("CPUID(7) EBX = 0x%x\n", info[1]);
    printf("  AVX-512F (bit 16): %d\n", f);
    printf("  AVX-512DQ (bit 17): %d\n", dq);

    int detected = osxsave && xcr0_ok && f && dq;
    printf("\nhave_avx512() would return: %d\n", detected);
    return 0;
}