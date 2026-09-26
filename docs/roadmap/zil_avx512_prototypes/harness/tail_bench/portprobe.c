/* portprobe.c — per-op THROUGHPUT at xmm / ymm / zmm on this host (indicative, VM).
 * Inline asm, 12 independent register chains per op, so latency is hidden and
 * the compiler cannot reshape it. Reports vector-ops per ns. */
#include <stdio.h>
#include <time.h>

static double now_ns(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

#define ITER 4000000L
#define R12(OP, P) OP " %%" P "12, %%" P "0, %%" P "0\n\t" OP " %%" P "12, %%" P "1, %%" P "1\n\t" \
  OP " %%" P "12, %%" P "2, %%" P "2\n\t" OP " %%" P "12, %%" P "3, %%" P "3\n\t" \
  OP " %%" P "12, %%" P "4, %%" P "4\n\t" OP " %%" P "12, %%" P "5, %%" P "5\n\t" \
  OP " %%" P "12, %%" P "6, %%" P "6\n\t" OP " %%" P "12, %%" P "7, %%" P "7\n\t" \
  OP " %%" P "12, %%" P "8, %%" P "8\n\t" OP " %%" P "12, %%" P "9, %%" P "9\n\t" \
  OP " %%" P "12, %%" P "10, %%" P "10\n\t" OP " %%" P "12, %%" P "11, %%" P "11\n\t"
#define P12(OP, P) OP " $5, %%" P "0, %%" P "0\n\t" OP " $5, %%" P "1, %%" P "1\n\t" \
  OP " $5, %%" P "2, %%" P "2\n\t" OP " $5, %%" P "3, %%" P "3\n\t" \
  OP " $5, %%" P "4, %%" P "4\n\t" OP " $5, %%" P "5, %%" P "5\n\t" \
  OP " $5, %%" P "6, %%" P "6\n\t" OP " $5, %%" P "7, %%" P "7\n\t" \
  OP " $5, %%" P "8, %%" P "8\n\t" OP " $5, %%" P "9, %%" P "9\n\t" \
  OP " $5, %%" P "10, %%" P "10\n\t" OP " $5, %%" P "11, %%" P "11\n\t"

#define PROBE(NAME, BODY) \
static double NAME(void) { double best = 1e30; \
  for (int t = 0; t < 7; t++) { double t0 = now_ns(); \
    for (long i = 0; i < ITER; i++) __asm__ volatile(BODY ::: "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12"); \
    double dt = now_ns() - t0; if (dt < best) best = dt; } \
  return 12.0 * ITER / best; }

PROBE(add_x, R12("vaddpd", "xmm"))
PROBE(add_y, R12("vaddpd", "ymm"))
PROBE(add_z, R12("vaddpd", "zmm"))
PROBE(fma_x, R12("vfmadd231pd", "xmm"))
PROBE(fma_y, R12("vfmadd231pd", "ymm"))
PROBE(fma_z, R12("vfmadd231pd", "zmm"))
PROBE(prm_x, P12("vpermilpd", "xmm"))
PROBE(prm_y, P12("vpermilpd", "ymm"))
PROBE(prm_z, P12("vpermilpd", "zmm"))
PROBE(xor_x, R12("vxorpd", "xmm"))
PROBE(xor_y, R12("vxorpd", "ymm"))
PROBE(xor_z, R12("vxorpd", "zmm"))
/* mix: 3 add + 3 fma + 3 perm + 3 xor over 12 chains */
#define MIX(P) "vaddpd %%" P "12, %%" P "0, %%" P "0\n\t vaddpd %%" P "12, %%" P "1, %%" P "1\n\t vaddpd %%" P "12, %%" P "2, %%" P "2\n\t" \
  "vfmadd231pd %%" P "12, %%" P "3, %%" P "3\n\t vfmadd231pd %%" P "12, %%" P "4, %%" P "4\n\t vfmadd231pd %%" P "12, %%" P "5, %%" P "5\n\t" \
  "vpermilpd $5, %%" P "6, %%" P "6\n\t vpermilpd $5, %%" P "7, %%" P "7\n\t vpermilpd $5, %%" P "8, %%" P "8\n\t" \
  "vxorpd %%" P "12, %%" P "9, %%" P "9\n\t vxorpd %%" P "12, %%" P "10, %%" P "10\n\t vxorpd %%" P "12, %%" P "11, %%" P "11\n\t"
PROBE(mix_x, MIX("xmm"))
PROBE(mix_y, MIX("ymm"))
PROBE(mix_z, MIX("zmm"))

int main(void)
{
    struct { const char *n; double (*f)(void); } p[] = {
        {"vaddpd", 0}, {"xmm", add_x}, {"ymm", add_y}, {"zmm", add_z},
        {"vfmadd231pd", 0}, {"xmm", fma_x}, {"ymm", fma_y}, {"zmm", fma_z},
        {"vpermilpd imm", 0}, {"xmm", prm_x}, {"ymm", prm_y}, {"zmm", prm_z},
        {"vxorpd", 0}, {"xmm", xor_x}, {"ymm", xor_y}, {"zmm", xor_z},
        {"mix 3add+3fma+3perm+3xor", 0}, {"xmm", mix_x}, {"ymm", mix_y}, {"zmm", mix_z},
    };
    add_z();
    for (unsigned i = 0; i < sizeof p / sizeof p[0]; i++) {
        if (!p[i].f) { printf("%s\n", p[i].n); continue; }
        printf("   %s %6.2f ops/ns\n", p[i].n, p[i].f());
    }
    return 0;
}
