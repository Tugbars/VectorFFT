/* mkl_caps_probe.c — what storage/placement does MKL r2c actually accept?
 * Tests whether the in-place + split (REAL_REAL) config the user wants is legal,
 * so we know whether a "split in-place r2c" MKL reference even exists.
 * Build: build_tuned/build.py --src benches/mkl_caps_probe.c --mkl --compile */
#include <stdio.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

static void try(const char *label, int real, int inplace, int storage_split, int cce_split) {
    DFTI_DESCRIPTOR_HANDLE h = 0;
    MKL_LONG s;
    s = DftiCreateDescriptor(&h, DFTI_DOUBLE, real ? DFTI_REAL : DFTI_COMPLEX, 1, 256);
    if (s) { printf("%-46s create: %s\n", label, DftiErrorMessage(s)); return; }
    DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, 256);
    DftiSetValue(h, DFTI_PLACEMENT, inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    if (storage_split) {
        s = DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        if (s) { printf("%-46s set COMPLEX_STORAGE=REAL_REAL: %s\n", label, DftiErrorMessage(s)); }
    }
    if (real) {
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        if (cce_split) {
            s = DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
            if (s) printf("%-46s set CE+REAL_REAL: %s\n", label, DftiErrorMessage(s));
        }
    }
    s = DftiCommitDescriptor(h);
    printf("%-46s commit: %s\n", label, s ? DftiErrorMessage(s) : "OK (accepted)");
    DftiFreeDescriptor(&h);
}

int main(void) {
    mkl_set_num_threads(1);
    printf("=== MKL DFTI capability probe (N=256, K=256) ===\n");
    try("complex  not-inplace  interleaved",   0, 0, 0, 0);
    try("complex  INPLACE      interleaved",   0, 1, 0, 0);
    try("complex  INPLACE      SPLIT(RR)",     0, 1, 1, 0);
    try("real(r2c) not-inplace CCE-interleaved",1, 0, 0, 0);
    try("real(r2c) INPLACE     CCE-interleaved",1, 1, 0, 0);
    try("real(r2c) INPLACE     CCE+SPLIT(RR)",  1, 1, 0, 1);
    return 0;
}
