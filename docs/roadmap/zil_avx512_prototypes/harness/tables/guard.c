/* masked tail at a buffer end: input's last complex sits right before a PROT_NONE page */
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
extern void radix8_z_t2_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
int main(void) {
    long pg = sysconf(_SC_PAGESIZE);
    for (int count = 1; count <= 7; count++) {
        size_t nd = 2 * 8 * (size_t)count;                 /* R=8 legs x count complex */
        char *m = mmap(0, 4 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mprotect(m + 2 * pg, pg, PROT_NONE);
        double *x = (double *)(m + 2 * pg) - nd;           /* ends at the guard page */
        for (size_t i = 0; i < nd; i++) x[i] = 1.0;
        static double y[2 * 8 * 16], tw[7 * 16 * 4];
        for (int i = 0; i < 7 * 16 * 4; i++) tw[i] = (i % 16) < 8 ? 1.0 : 0.0;
        radix8_z_t2_fwd_avx512(x, 0, y, 0, tw, 0, count, 0, count, 0, count);
        printf("count=%d ok (y0=%g)\n", count, y[0]);
        munmap(m, 4 * pg);
    }
    return 0;
}
