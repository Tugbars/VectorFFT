/* il2d_real_probe.c — attribution probe for the 4096x16 native loss
 * (6.47 ms/plane vs veneer 0.128 ms). Decomposes the native execute:
 *   full plane  = row pass + column pass
 *   row pass    = the TC K=4096 door timed alone
 *   per-row     = a bare K=1 zr2c N=16 execute timed alone x4096
 * plus VFFT_CONV_LOG sampling on ONE row to see what the child serves.
 * Single-cell probe, production wisdom (read-only), ~seconds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

static double now_ns(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(int argc, char **argv)
{
    const char *wisdir = argc > 1 ? argv[1] : ".";
    const int N1 = 4096, N2 = 16;
    const size_t RN = (size_t)N1 * N2, hp1 = N2 / 2 + 1;
    const size_t CN = (size_t)N1 * hp1;
    vfft_wisdom *W;
    double *x = malloc(RN * 8), *z = malloc(2 * CN * 8);
    double *xr = malloc(N2 * 8), *zr = malloc(2 * hp1 * 8);
    size_t i;
    int r;
#ifdef _WIN32
    _putenv("VFFT_IL2D_REAL=1");
#else
    putenv("VFFT_IL2D_REAL=1");
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    W = vfft_wisdom_load(wisdir);
    printf("=== 4096x16 attribution probe (wisdom=%s %s) ===\n", wisdir,
           W ? "loaded" : "MISSING");
    srand(7);
    for (i = 0; i < RN; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    for (i = 0; i < (size_t)N2; i++) xr[i] = x[i];

    /* A: the full native 2D execute */
    {
        vfft_config_t c;
        vfft_plan h;
        double t0, best = 1e300;
        memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE;
        c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        h = vfft_create(&c);
        if (!h) { printf("2D create FAIL\n"); return 1; }
        vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL); /* warm */
        for (r = 0; r < 20; r++) {
            t0 = now_ns();
            vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL);
            t0 = now_ns() - t0;
            if (t0 < best) best = t0;
        }
        printf("A full native 2D plane : %10.0f ns\n", best);
        vfft_destroy(h);
    }
    /* B: the TC K=4096 row door alone (exactly the driver's row pass) */
    {
        vfft_config_t c;
        vfft_plan h;
        double t0, best = 1e300;
        memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE;
        c.dims = 1; c.n[0] = N2;
        c.howmany = (size_t)N1;
        c.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        h = vfft_create(&c);
        if (!h) { printf("TC row-door create FAIL\n"); return 1; }
        vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL); /* warm */
        for (r = 0; r < 20; r++) {
            t0 = now_ns();
            vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL);
            t0 = now_ns() - t0;
            if (t0 < best) best = t0;
        }
        printf("B TC row door (4096 rows): %10.0f ns  (%.0f ns/row)\n",
               best, best / N1);
        vfft_destroy(h);
    }
    /* C: one bare K=1 zr2c N=16 OOP execute, x4096 serially */
    {
        vfft_config_t c;
        vfft_plan h;
        double t0, best = 1e300;
        memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE;
        c.dims = 1; c.n[0] = N2;
        c.howmany = 1; c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        h = vfft_create(&c);
        if (!h) { printf("K=1 zr2c create FAIL\n"); return 1; }
        vfft_execute(h, VFFT_FORWARD, xr, NULL, zr, NULL); /* warm */
        for (r = 0; r < 20; r++) {
            int k;
            t0 = now_ns();
            for (k = 0; k < N1; k++)
                vfft_execute(h, VFFT_FORWARD, xr, NULL, zr, NULL);
            t0 = now_ns() - t0;
            if (t0 < best) best = t0;
        }
        printf("C bare K=1 zr2c x4096  : %10.0f ns  (%.0f ns/row)\n",
               best, best / N1);
        vfft_destroy(h);
    }
    /* D: what serves the child? one row with VFFT_CONV_LOG on */
    {
        vfft_config_t c;
        vfft_plan h;
        memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE;
        c.dims = 1; c.n[0] = N2;
        c.howmany = 1; c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        h = vfft_create(&c);
        if (h) {
#ifdef _WIN32
            _putenv("VFFT_CONV_LOG=1");
#else
            putenv("VFFT_CONV_LOG=1");
#endif
            fprintf(stderr, "-- one K=1 zr2c execute with VFFT_CONV_LOG --\n");
            vfft_execute(h, VFFT_FORWARD, xr, NULL, zr, NULL);
#ifdef _WIN32
            _putenv("VFFT_CONV_LOG=");
#else
            unsetenv("VFFT_CONV_LOG");
#endif
            vfft_destroy(h);
        }
    }
    /* E/F: where does the 1.4us live? E = the child shape alone
     * (c2c(8) K=1 OOP INTERLEAVED = the convert-served cell); F = the
     * same engine through the SPLIT door (no z convert wrap). */
    {
        static const int LAY[2] = { 0, 1 }; /* 0 = interleaved, 1 = split */
        int li;
        for (li = 0; li < 2; li++) {
            vfft_config_t c;
            vfft_plan h;
            double t0, best = 1e300;
            double a[16], b[16], c1[16], d1[16];
            memset(a, 0, sizeof a); memset(b, 0, sizeof b);
            memset(&c, 0, sizeof c);
            c.transform = VFFT_C2C;
            c.placement = VFFT_OUTOFPLACE;
            c.rigor = VFFT_MEASURE;
            c.dims = 1; c.n[0] = 8;
            c.howmany = 1; c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
            c.layout = LAY[li] ? VFFT_LAYOUT_SPLIT : VFFT_LAYOUT_INTERLEAVED;
            h = vfft_create(&c);
            if (!h) { printf("%s c2c8 create FAIL\n", LAY[li] ? "F" : "E"); continue; }
            for (i = 0; i < 16; i++) a[i] = 0.5 - 0.01 * (double)i;
            if (LAY[li]) {
                vfft_execute(h, VFFT_FORWARD, a, b, c1, d1);
                for (r = 0; r < 20; r++) {
                    int k;
                    t0 = now_ns();
                    for (k = 0; k < N1; k++)
                        vfft_execute(h, VFFT_FORWARD, a, b, c1, d1);
                    t0 = now_ns() - t0;
                    if (t0 < best) best = t0;
                }
                printf("F c2c(8) SPLIT  x4096  : %10.0f ns  (%.0f ns/exec)\n",
                       best, best / N1);
            } else {
                vfft_execute(h, VFFT_FORWARD, a, NULL, b, NULL);
                for (r = 0; r < 20; r++) {
                    int k;
                    t0 = now_ns();
                    for (k = 0; k < N1; k++)
                        vfft_execute(h, VFFT_FORWARD, a, NULL, b, NULL);
                    t0 = now_ns() - t0;
                    if (t0 < best) best = t0;
                }
                printf("E c2c(8) IL conv x4096 : %10.0f ns  (%.0f ns/exec)\n",
                       best, best / N1);
            }
            vfft_destroy(h);
        }
    }
    /* G: the honest 4096x16 re-verdict — native vs veneer vs MKL CCE,
     * same-run alternated, min-of-20 (the race table's 0.02 row was the
     * getenv tax; this is the corrected number). */
    {
        vfft_config_t c;
        vfft_plan hn = NULL, hv = NULL;
        double tn = 1e300, tv = 1e300, tm = 1e300, t0;
        memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE;
        c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        vfft_plan hr = NULL;
        double tr = 1e300;
#ifdef _WIN32
        _putenv("VFFT_IL2D_ROWSPLIT=0"); /* pin the per-row door */
#else
        putenv("VFFT_IL2D_ROWSPLIT=0");
#endif
        fprintf(stderr, "[G] native per-row create...\n");
        hn = vfft_create(&c);
#ifdef _WIN32
        _putenv("VFFT_IL2D_ROWSPLIT="); /* banked route (rw=32, fused) */
#else
        unsetenv("VFFT_IL2D_ROWSPLIT");
#endif
        fprintf(stderr, "[G] banked-route create...\n");
        hr = vfft_create(&c);
#ifdef _WIN32
        _putenv("VFFT_IL2D_REAL=");
#else
        unsetenv("VFFT_IL2D_REAL");
#endif
        fprintf(stderr, "[G] veneer create...\n");
        hv = vfft_create(&c); /* veneer */
        fprintf(stderr, "[G] creates done hn=%p hr=%p hv=%p\n",
                (void *)hn, (void *)hr, (void *)hv);
#ifdef VFFT_HAS_MKL
        {
            DFTI_DESCRIPTOR_HANDLE hm = 0;
            MKL_LONG dims[2] = { N1, N2 };
            /* MKL's default 2D-real CCE output strides pitch rows at N2
             * complex (not hp1) — size like the bench (RN*2), a tight
             * 2*CN plane overflows (segfaulted here). */
            double *cce = malloc(RN * 2 * 8);
            int mok = 0;
            if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_REAL, 2,
                                     dims) == DFTI_NO_ERROR)
            {
                DftiSetValue(hm, DFTI_CONJUGATE_EVEN_STORAGE,
                             DFTI_COMPLEX_COMPLEX);
                DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                mok = (DftiCommitDescriptor(hm) == DFTI_NO_ERROR);
            }
            fprintf(stderr, "[G] mkl mok=%d cce=%p\n", mok, (void *)cce);
            if (hn && hv && mok && cce)
            {
                fprintf(stderr, "[G] warm native...\n");
                vfft_execute(hn, VFFT_FORWARD, x, NULL, z, NULL);
                fprintf(stderr, "[G] warm veneer...\n");
                vfft_execute(hv, VFFT_FORWARD, x, NULL, z, NULL);
                fprintf(stderr, "[G] warm mkl...\n");
                DftiComputeForward(hm, (void *)x, cce);
                fprintf(stderr, "[G] warm done, timing...\n");
                if (hr)
                    vfft_execute(hr, VFFT_FORWARD, x, NULL, z, NULL);
                for (r = 0; r < 20; r++)
                {
                    t0 = now_ns();
                    vfft_execute(hn, VFFT_FORWARD, x, NULL, z, NULL);
                    t0 = now_ns() - t0;
                    if (t0 < tn) tn = t0;
                    if (hr)
                    {
                        t0 = now_ns();
                        vfft_execute(hr, VFFT_FORWARD, x, NULL, z, NULL);
                        t0 = now_ns() - t0;
                        if (t0 < tr) tr = t0;
                    }
                    t0 = now_ns();
                    vfft_execute(hv, VFFT_FORWARD, x, NULL, z, NULL);
                    t0 = now_ns() - t0;
                    if (t0 < tv) tv = t0;
                    t0 = now_ns();
                    DftiComputeForward(hm, (void *)x, cce);
                    t0 = now_ns() - t0;
                    if (t0 < tm) tm = t0;
                }
                printf("G 4096x16 re-verdict   : nat %8.0f | rowsplit "
                       "%8.0f | veneer %8.0f | MKL %8.0f | rsUp-vs-veneer "
                       "%.2f | rs xMKL %.2f\n",
                       tn, tr, tv, tm, tv / tr, tm / tr);
            }
            if (hm) DftiFreeDescriptor(&hm);
            free(cce);
        }
#endif
        if (hn) vfft_destroy(hn);
        if (hr) vfft_destroy(hr);
        if (hv) vfft_destroy(hv);
    }
    /* H: the batched-row-interior ceiling — split r2c at (N=16, K=4096),
     * the lane-batch engine the "IL boundary, split inside" row pass
     * would run. Lane-major buffers (the engine's own geometry; the
     * gather/store boundaries are NOT priced here — this is the
     * interior's floor). */
    {
        vfft_config_t c;
        vfft_plan h;
        double *lx = malloc(RN * 8), *lre = malloc(CN * 8);
        double *lim = malloc(CN * 8);
        double t0, best = 1e300;
        memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE;
        c.dims = 1; c.n[0] = N2;
        c.howmany = (size_t)N1;
        c.nthreads = 1; c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_SPLIT; /* lane-major K-batch */
        h = vfft_create(&c);
        if (!h || !lx || !lre || !lim)
            printf("H split (16,K=4096) create FAIL\n");
        else
        {
            for (i = 0; i < RN; i++) lx[i] = x[i];
            vfft_execute(h, VFFT_FORWARD, lx, NULL, lre, lim); /* warm */
            for (r = 0; r < 20; r++)
            {
                t0 = now_ns();
                vfft_execute(h, VFFT_FORWARD, lx, NULL, lre, lim);
                t0 = now_ns() - t0;
                if (t0 < best) best = t0;
            }
            printf("H split rfft (16,K=4096): %10.0f ns  (%.1f ns/row) — "
                   "the batched-interior ceiling\n", best, best / N1);
        }
        if (h) vfft_destroy(h);
        free(lx); free(lre); free(lim);
    }
    /* I: the rowz-fusion A/B — fused doors vs the staged 3-pass route,
     * SAME process, alternated, min-of-20, both directions. Route pinned
     * via env ROWSPLIT=32 for both arms; the arms differ only by the
     * create-time VFFT_IL2D_NO_ROWZ knob. */
    {
        static const int IC[][2] = { { 4096, 16 }, { 512, 16 } };
        int ic;
#ifdef _WIN32
        _putenv("VFFT_IL2D_REAL=1");
        _putenv("VFFT_IL2D_ROWSPLIT=32");
#else
        putenv("VFFT_IL2D_REAL=1");
        putenv("VFFT_IL2D_ROWSPLIT=32");
#endif
        for (ic = 0; ic < 2; ic++) {
            const int N1i = IC[ic][0], N2i = IC[ic][1];
            const size_t hp1i = (size_t)N2i / 2 + 1;
            const size_t RNi = (size_t)N1i * N2i, CNi = (size_t)N1i * hp1i;
            double *xi = malloc(RNi * 8), *zi2 = malloc(2 * CNi * 8);
            double *xo = malloc(RNi * 8);
            vfft_config_t c2;
            vfft_plan fA, fB, bA, bB;
            double tfA = 1e300, tfB = 1e300, tbA = 1e300, tbB = 1e300;
            double t0;
            if (!xi || !zi2 || !xo) return 2;
            for (i = 0; i < RNi; i++)
                xi[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&c2, 0, sizeof c2);
            c2.transform = VFFT_R2C;
            c2.placement = VFFT_OUTOFPLACE;
            c2.rigor = VFFT_MEASURE;
            c2.dims = 2; c2.n[0] = N1i; c2.n[1] = N2i;
            c2.howmany = 1; c2.nthreads = 1; c2.wisdom = W;
            c2.wisdom_write = 0;
            c2.layout = VFFT_LAYOUT_INTERLEAVED;
            fA = vfft_create(&c2);                 /* fused doors */
            c2.transform = VFFT_C2R;
            bA = vfft_create(&c2);
#ifdef _WIN32
            _putenv("VFFT_IL2D_NO_ROWZ=1");
#else
            putenv("VFFT_IL2D_NO_ROWZ=1");
#endif
            c2.transform = VFFT_R2C;
            fB = vfft_create(&c2);                 /* staged route */
            c2.transform = VFFT_C2R;
            bB = vfft_create(&c2);
#ifdef _WIN32
            _putenv("VFFT_IL2D_NO_ROWZ=");
#else
            unsetenv("VFFT_IL2D_NO_ROWZ");
#endif
            if (!fA || !fB || !bA || !bB) {
                printf("I %dx%d create FAIL\n", N1i, N2i);
                return 2;
            }
            /* correctness cross-check: fused == staged BITWISE (pure
             * data movement) */
            vfft_execute(fA, VFFT_FORWARD, xi, NULL, zi2, NULL);
            {
                double *z2 = malloc(2 * CNi * 8);
                vfft_execute(fB, VFFT_FORWARD, xi, NULL, z2, NULL);
                if (memcmp(zi2, z2, 2 * CNi * 8) != 0)
                    printf("I %dx%d fwd fused != staged *** FAIL ***\n",
                           N1i, N2i);
                free(z2);
            }
            vfft_execute(bA, VFFT_BACKWARD, zi2, NULL, xo, NULL);
            for (r = 0; r < 20; r++) {
                t0 = now_ns();
                vfft_execute(fA, VFFT_FORWARD, xi, NULL, zi2, NULL);
                t0 = now_ns() - t0;
                if (t0 < tfA) tfA = t0;
                t0 = now_ns();
                vfft_execute(fB, VFFT_FORWARD, xi, NULL, zi2, NULL);
                t0 = now_ns() - t0;
                if (t0 < tfB) tfB = t0;
                t0 = now_ns();
                vfft_execute(bA, VFFT_BACKWARD, zi2, NULL, xo, NULL);
                t0 = now_ns() - t0;
                if (t0 < tbA) tbA = t0;
                t0 = now_ns();
                vfft_execute(bB, VFFT_BACKWARD, zi2, NULL, xo, NULL);
                t0 = now_ns() - t0;
                if (t0 < tbB) tbB = t0;
            }
            printf("I %4dx%-3d rowz A/B: fwd fused %8.0f vs staged %8.0f"
                   " (x%.3f) | bwd fused %8.0f vs staged %8.0f (x%.3f)\n",
                   N1i, N2i, tfA, tfB, tfB / tfA, tbA, tbB, tbB / tbA);
            vfft_destroy(fA); vfft_destroy(fB);
            vfft_destroy(bA); vfft_destroy(bB);
            free(xi); free(zi2); free(xo);
        }
    }
    if (W) vfft_wisdom_free(W);
    free(x); free(z); free(xr); free(zr);
    return 0;
}
