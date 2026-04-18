
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#endif

/* CPUID via the compiler's preferred intrinsic header */
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
  #include <intrin.h>
  static void _cpuid_count(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
      int r[4]; __cpuidex(r, leaf, subleaf);
      *eax = r[0]; *ebx = r[1]; *ecx = r[2]; *edx = r[3];
  }
#else
  #include <cpuid.h>
  static void _cpuid_count(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
      unsigned int a, b, c, d;
      __cpuid_count(leaf, subleaf, a, b, c, d);
      *eax = (int)a; *ebx = (int)b; *ecx = (int)c; *edx = (int)d;
  }
#endif

/* ─── timing: portable high-resolution monotonic clock in ns ─── */
#ifdef _WIN32
static double now_ns(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
}
#else
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ─── aligned alloc / free ─── */
#ifdef _WIN32
static void *aalloc(size_t b) {
    void *p = _aligned_malloc(b, 64);
    if (p) memset(p, 0, b);
    return p;
}
static void afree(void *p) { _aligned_free(p); }
#else
static void *aalloc(size_t b) {
    void *p = NULL;
    if (posix_memalign(&p, 64, b) != 0) return NULL;
    memset(p, 0, b); return p;
}
static void afree(void *p) { free(p); }
#endif

static int _dcmp(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Comparator for qsort/bsearch on arrays of char*: both args are char**,
 * dereference once to compare the actual strings. */
static int cmp_strp(const void *a, const void *b) {
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

static int have_avx512(void) {
    int a, b, c, d;
    /* OSXSAVE required for AVX/AVX-512 state save */
    _cpuid_count(1, 0, &a, &b, &c, &d);
    if (!(c & (1 << 27))) return 0;
    /* XGETBV to check XCR0 bits for ZMM state saving */
    unsigned long long xcr0;
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    xcr0 = _xgetbv(0);
#else
    unsigned int lo, hi;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = ((unsigned long long)hi << 32) | lo;
#endif
    /* Bits 1(SSE) 2(AVX) 5(opmask) 6(zmm hi) 7(zmm16-31) */
    if ((xcr0 & 0xE6) != 0xE6) return 0;
    _cpuid_count(7, 0, &a, &b, &c, &d);
    return ((b & (1<<16)) != 0) && ((b & (1<<17)) != 0);
}

typedef void (*t1_fn)(double*, double*, const double*, const double*, size_t, size_t);
#define R 4
#define PI 3.14159265358979323846

static void fill_twiddles(double *Wr, double *Wi, size_t me) {
    for (size_t n = 0; n < R-1; n++) for (size_t m = 0; m < me; m++) {
        double g = -2.0 * PI * (double)(n+1) * (double)m / (double)(R*me);
        Wr[n*me+m] = cos(g); Wi[n*me+m] = sin(g);
    }
}

static double measure(t1_fn fn, size_t ios, size_t me) {
    size_t an = R * (ios > me ? ios : me);
    double *rio_re = aalloc(an*8), *rio_im = aalloc(an*8);
    double *src_re = aalloc(an*8), *src_im = aalloc(an*8);
    double *Wr = aalloc((R-1)*me*8), *Wi = aalloc((R-1)*me*8);
    unsigned s = 12345;
    for (size_t i = 0; i < an; i++) {
        s = s*1103515245u + 12345u;
        src_re[i] = ((double)(s>>16) / 32768.0) - 1.0;
        s = s*1103515245u + 12345u;
        src_im[i] = ((double)(s>>16) / 32768.0) - 1.0;
    }
    fill_twiddles(Wr, Wi, me);
    size_t work = R * me;
    int reps = (int)(1000000.0 / ((double)work + 1));
    if (reps < 20) reps = 20;
    if (reps > 10000) reps = 10000;
    for (int i = 0; i < 100; i++) {
        memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
        fn(rio_re, rio_im, Wr, Wi, ios, me);
    }
    double s1[21], s2[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
            fn(rio_re, rio_im, Wr, Wi, ios, me);
        }
        s1[i] = (now_ns() - t0) / reps;
    }
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
        }
        s2[i] = (now_ns() - t0) / reps;
    }
    qsort(s1, 21, sizeof(double), _dcmp);
    qsort(s2, 21, sizeof(double), _dcmp);
    double net = s1[10] - s2[10];
    if (net < 0) net = s1[10];
    afree(rio_re); afree(rio_im); afree(src_re); afree(src_im); afree(Wr); afree(Wi);
    return net;
}
extern void radix4_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_log3_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_log3_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_log1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_log1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_u2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_u2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dif_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dif_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);

typedef struct { const char *id; const char *isa; t1_fn fwd, bwd; int req512; } cand_t;
static cand_t CANDS[] = {
    {"ct_t1_dit__avx2", "avx2", radix4_t1_dit_fwd_avx2, radix4_t1_dit_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2", "avx2", radix4_t1_dit_log3_fwd_avx2, radix4_t1_dit_log3_bwd_avx2, 0},
    {"ct_t1_dit_log1__avx2", "avx2", radix4_t1_dit_log1_fwd_avx2, radix4_t1_dit_log1_bwd_avx2, 0},
    {"ct_t1_dit_u2__avx2", "avx2", radix4_t1_dit_u2_fwd_avx2, radix4_t1_dit_u2_bwd_avx2, 0},
    {"ct_t1_dif__avx2", "avx2", radix4_t1_dif_fwd_avx2, radix4_t1_dif_bwd_avx2, 0},
    {NULL,NULL,NULL,NULL,0}
};
typedef struct { size_t ios, me; } sp_t;
static sp_t SW[] = {
    {64, 64},
    {72, 64},
    {128, 64},
    {128, 128},
    {136, 128},
    {192, 128},
    {256, 256},
    {264, 256},
    {320, 256},
    {512, 512},
    {520, 512},
    {576, 512},
    {1024, 1024},
    {1032, 1024},
    {1088, 1024},
    {2048, 2048},
    {2056, 2048},
    {2112, 2048},
    {2048, 256},
    {8192, 256},
    {8192, 512},
    {16384, 1024},
    {8192, 2048},
    {32768, 2048},
    {0,0}
};

int main(int argc, char **argv) {
    int skip512 = !have_avx512();
    int n_cand = sizeof(CANDS)/sizeof(CANDS[0]) - 1;
    int n_sp = sizeof(SW)/sizeof(SW[0]) - 1;
    const char *outpath = argc > 1 ? argv[1] : "measurements.jsonl";

    /* Resume support: build a set of (id, ios, me, dir) tuples already in
     * the output file. On each planned measurement below, skip if the tuple
     * is in the set. Robust to candidate list changes (add, remove, reorder)
     * which the old line-count-based skip was not. Keys are stored as
     * plain strings "id|ios|me|dir", sorted for binary search lookup.
     */
    size_t done_cap = 256, done_n = 0;
    char **done_keys = (char**)malloc(done_cap * sizeof(char*));
    if (!done_keys) { fprintf(stderr, "OOM allocating done_keys\n"); return 1; }

    FILE *rd = fopen(outpath, "r");
    if (rd) {
        char line[2048];
        while (fgets(line, sizeof(line), rd)) {
            /* Minimal parse: find "id":"...", "ios":N, "me":N, "dir":"..." */
            char id[256] = {0}, dir[16] = {0};
            unsigned long ios = 0, me = 0;
            const char *p = strstr(line, "\"id\":\"");
            if (!p) continue;
            p += 6;  /* past '"id":"' */
            const char *q = strchr(p, '"');
            if (!q || (size_t)(q-p) >= sizeof(id)) continue;
            memcpy(id, p, q-p); id[q-p] = 0;
            p = strstr(line, "\"ios\":");
            if (!p || sscanf(p+6, "%lu", &ios) != 1) continue;
            p = strstr(line, "\"me\":");
            if (!p || sscanf(p+5, "%lu", &me) != 1) continue;
            p = strstr(line, "\"dir\":\"");
            if (!p) continue;
            p += 7;
            q = strchr(p, '"');
            if (!q || (size_t)(q-p) >= sizeof(dir)) continue;
            memcpy(dir, p, q-p); dir[q-p] = 0;

            /* Grow if needed */
            if (done_n >= done_cap) {
                done_cap *= 2;
                char **nk = (char**)realloc(done_keys, done_cap * sizeof(char*));
                if (!nk) { fprintf(stderr, "OOM growing done_keys\n"); return 1; }
                done_keys = nk;
            }
            /* Build key "id|ios|me|dir" */
            size_t need = strlen(id) + 32;
            char *k = (char*)malloc(need);
            if (!k) { fprintf(stderr, "OOM\n"); return 1; }
            snprintf(k, need, "%s|%lu|%lu|%s", id, ios, me, dir);
            done_keys[done_n++] = k;
        }
        fclose(rd);
    }
    /* Sort for binary search. qsort/bsearch comparators receive pointers
     * TO array elements; elements are char*, so callbacks get char**.
     * Dereference once before strcmp. */
    qsort(done_keys, done_n, sizeof(char*), cmp_strp);
    /* Dedup (in case file has duplicates from crashes mid-write) */
    if (done_n > 1) {
        size_t w = 1;
        for (size_t r = 1; r < done_n; r++) {
            if (strcmp(done_keys[w-1], done_keys[r]) != 0) {
                done_keys[w++] = done_keys[r];
            } else {
                free(done_keys[r]);
            }
        }
        done_n = w;
    }
    fprintf(stderr, "harness: %d candidates, %d sweeps, %d dirs; %zu done (will skip)\n",
            n_cand, n_sp, 2, done_n);

    FILE *out = fopen(outpath, "a");
    if (!out) { perror("fopen"); return 1; }

    int skipped = 0, measured = 0;
    for (int ci = 0; ci < n_cand; ci++) {
        cand_t *c = &CANDS[ci];
        if (c->req512 && skip512) continue;
        fprintf(stderr, "[%d/%d] %s\n", ci+1, n_cand, c->id);
        for (int sp = 0; sp < n_sp; sp++) {
            size_t ios = SW[sp].ios, me = SW[sp].me;
            for (int d = 0; d < 2; d++) {
                /* Build lookup key */
                char key[512];
                snprintf(key, sizeof(key), "%s|%zu|%zu|%s",
                         c->id, ios, me, d == 0 ? "fwd" : "bwd");
                /* bsearch needs pointer-to-key (matching array element type) */
                char *k = key;
                char **found = (char**)bsearch(&k, done_keys, done_n,
                                               sizeof(char*), cmp_strp);
                if (found) { skipped++; continue; }

                t1_fn fn = (d == 0) ? c->fwd : c->bwd;
                double ns = measure(fn, ios, me);
                fprintf(out, "{\"id\":\"%s\",\"ios\":%zu,\"me\":%zu,\"dir\":\"%s\",\"ns\":%.2f}\n",
                        c->id, ios, me, d == 0 ? "fwd" : "bwd", ns);
                fflush(out);
                measured++;
            }
        }
    }
    fclose(out);
    fprintf(stderr, "done: measured=%d, skipped=%d\n", measured, skipped);
    for (size_t i = 0; i < done_n; i++) free(done_keys[i]);
    free(done_keys);
    return 0;
}
