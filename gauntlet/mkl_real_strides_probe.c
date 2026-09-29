/* mkl_real_strides_probe.c -- what MKL's DFTI defaults are for a real 2D CCE
 * descriptor (NOT_INPLACE) and a batched 1D real descriptor: the strides and
 * distances it applies to each domain. Build: python gauntlet/build.py
 * --compile --mkl --src gauntlet/mkl_real_strides_probe.c */
#include <stdio.h>
#include <mkl_dfti.h>
int main(void)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[2] = { 16, 32 }, is[3] = { 9, 9, 9 }, os[3] = { 9, 9, 9 }, v = 0;
    DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 2, dims);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    printf("2D 16x32 commit rc=%ld\n", DftiCommitDescriptor(d));
    DftiGetValue(d, DFTI_INPUT_STRIDES, is);
    DftiGetValue(d, DFTI_OUTPUT_STRIDES, os);
    printf("2D defaults: input strides {%ld,%ld,%ld} output strides {%ld,%ld,%ld}\n", is[0], is[1], is[2], os[0], os[1], os[2]);
    DftiFreeDescriptor(&d);
    d = NULL;
    DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)1024);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)8);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)1024);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)513);
    printf("1D K=8 commit rc=%ld\n", DftiCommitDescriptor(d));
    DftiGetValue(d, DFTI_INPUT_DISTANCE, &v); printf("1D K=8: input distance %ld", v);
    DftiGetValue(d, DFTI_OUTPUT_DISTANCE, &v); printf(" output distance %ld\n", v);
    DftiFreeDescriptor(&d);
    return 0;
}
