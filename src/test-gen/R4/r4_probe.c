/* R=4 large-ios probe harness -- cross-platform (Linux + Windows) */
#include "radix4_avx2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
  static double now_ns(void) {
      LARGE_INTEGER c, f;
      QueryPerformanceFrequency(&f);
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
  }
  /* MSVC/Windows: _aligned_malloc / _aligned_free */
  static void *aalloc(size_t align, size_t size) {
      return _aligned_malloc(size, align);
  }
  static void afree(void *p) { _aligned_free(p); }
#else
  #include <time.h>
  static double now_ns(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec * 1e9 + ts.tv_nsec;
  }
  static void *aalloc(size_t align, size_t size) {
      /* size must be a multiple of align for aligned_alloc (C11). Round up. */
      size_t rounded = (size + align - 1) & ~(align - 1);
      return aligned_alloc(align, rounded);
  }
  static void afree(void *p) { free(p); }
#endif

#define R 4

typedef struct {
    size_t me;
    size_t ios;
    const char *label;
} probe_case_t;

int main(void) {
    size_t max_total = 4ULL * 65536ULL * 2;
    double *rio_re = aalloc(64, max_total * sizeof(double));
    double *rio_im = aalloc(64, max_total * sizeof(double));
    double *orig_re = aalloc(64, max_total * sizeof(double));
    double *orig_im = aalloc(64, max_total * sizeof(double));

    size_t max_me = 4096;
    double *W_re = aalloc(64, 3 * max_me * sizeof(double));
    double *W_im = aalloc(64, 3 * max_me * sizeof(double));

    if (!rio_re || !rio_im || !orig_re || !orig_im || !W_re || !W_im) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < max_total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (size_t i = 0; i < 3 * max_me; i++) {
        W_re[i] = (double)rand() / RAND_MAX - 0.5;
        W_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    probe_case_t cases[] = {
        {  64,     64, "baseline_small"          },
        { 256,    264, "late_inner_samepage"     },
        { 256,   2048, "mid_inner_fewpages"      },
        { 512,    512, "ios_eq_me_pow2"          },
        {2048,   2056, "late_large_me_samepage"  },
        {1024,   1032, "late_large_me_1024"      },
        {2048,   8192, "heavy_dtlb_4pages"       },
        {2048,  32768, "catastrophic_dtlb"       },
        {1024,  16384, "N1M_stage1"              },
        {4096,   4096, "ios_eq_me_4k"            },
        { 256,  65536, "extreme_ios_small_me"    },
    };
    size_t ncases = sizeof(cases) / sizeof(cases[0]);

    printf("%-28s %6s %8s %12s %12s %12s %10s %10s\n",
           "case", "me", "ios", "t1_dit_ns", "log3_ns", "t1_dif_ns",
           "log3/flat", "dif/flat");
    printf("%-28s %6s %8s %12s %12s %12s %10s %10s\n",
           "----", "--", "---", "---------", "-------", "---------",
           "---------", "--------");

    for (size_t ci = 0; ci < ncases; ci++) {
        size_t me = cases[ci].me;
        size_t ios = cases[ci].ios;
        size_t total = R * ios;

        size_t reps;
        if (me <= 256) reps = 50000;
        else if (me <= 1024) reps = 10000;
        else if (me <= 2048) reps = 5000;
        else reps = 2000;

        /* t1_dit */
        double best_dit = 1e18;
        memcpy(rio_re, orig_re, total * sizeof(double));
        memcpy(rio_im, orig_im, total * sizeof(double));
        for (int w = 0; w < 5; w++) {
            radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        }
        for (int t = 0; t < 3; t++) {
            memcpy(rio_re, orig_re, total * sizeof(double));
            memcpy(rio_im, orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (size_t r = 0; r < reps; r++) {
                radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            }
            double ns = (now_ns() - t0) / reps;
            if (ns < best_dit) best_dit = ns;
        }

        /* log3 */
        double best_log3 = 1e18;
        memcpy(rio_re, orig_re, total * sizeof(double));
        memcpy(rio_im, orig_im, total * sizeof(double));
        for (int w = 0; w < 5; w++) {
            radix4_t1_dit_log3_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        }
        for (int t = 0; t < 3; t++) {
            memcpy(rio_re, orig_re, total * sizeof(double));
            memcpy(rio_im, orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (size_t r = 0; r < reps; r++) {
                radix4_t1_dit_log3_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            }
            double ns = (now_ns() - t0) / reps;
            if (ns < best_log3) best_log3 = ns;
        }

        /* dif */
        double best_dif = 1e18;
        memcpy(rio_re, orig_re, total * sizeof(double));
        memcpy(rio_im, orig_im, total * sizeof(double));
        for (int w = 0; w < 5; w++) {
            radix4_t1_dif_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        }
        for (int t = 0; t < 3; t++) {
            memcpy(rio_re, orig_re, total * sizeof(double));
            memcpy(rio_im, orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (size_t r = 0; r < reps; r++) {
                radix4_t1_dif_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            }
            double ns = (now_ns() - t0) / reps;
            if (ns < best_dif) best_dif = ns;
        }

        printf("%-28s %6zu %8zu %12.1f %12.1f %12.1f %10.3f %10.3f\n",
               cases[ci].label, me, ios,
               best_dit, best_log3, best_dif,
               best_log3 / best_dit, best_dif / best_dit);
    }

    afree(rio_re); afree(rio_im); afree(orig_re); afree(orig_im);
    afree(W_re); afree(W_im);
    return 0;
}
