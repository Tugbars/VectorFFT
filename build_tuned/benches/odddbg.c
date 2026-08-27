#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    const int N1 = 4, N2 = 63; const size_t hp1 = N2/2 + 1;
    const size_t RN = (size_t)N1*N2, CN = (size_t)N1*hp1;
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
    c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0;
    vfft_plan pf = vfft_create(&c);
    double *X = malloc(RN*8), *Z = malloc(2*CN*8);
    double R[4][2*64], Cc[4][2*64];
    srand(7);
    for (size_t i = 0; i < RN; i++) X[i] = (double)rand()/RAND_MAX - 0.5;
    vfft_execute(pf, VFFT_FORWARD, X, NULL, Z, NULL);
    /* reference: per-row naive FFT bins 0..hp1-1 */
    for (int r = 0; r < N1; r++)
        for (size_t k = 0; k < hp1; k++) {
            double er=0, ei=0;
            for (int n = 0; n < N2; n++) {
                double a = -2.0*3.14159265358979323846*(double)k*n/N2;
                er += X[(size_t)r*N2+n]*cos(a); ei += X[(size_t)r*N2+n]*sin(a);
            }
            R[r][2*k]=er; R[r][2*k+1]=ei;
        }
    /* then 4-point column DFT along r */
    for (int k1 = 0; k1 < N1; k1++)
        for (size_t k = 0; k < hp1; k++) {
            double er=0, ei=0;
            for (int r = 0; r < N1; r++) {
                double a = -2.0*3.14159265358979323846*(double)k1*r/N1;
                double cr=cos(a), sr=sin(a);
                er += R[r][2*k]*cr - R[r][2*k+1]*sr;
                ei += R[r][2*k]*sr + R[r][2*k+1]*cr;
            }
            Cc[k1][2*k]=er; Cc[k1][2*k+1]=ei;
        }
    for (int r = 0; r < 2; r++) {
        printf("row %d  vfft: ", r);
        for (int k = 0; k < 3; k++) printf("(%7.3f %7.3f) ", Z[(size_t)r*2*hp1+2*k], Z[(size_t)r*2*hp1+2*k+1]);
        printf("\n  ref-col:     ");
        for (int k = 0; k < 3; k++) printf("(%7.3f %7.3f) ", Cc[r][2*k], Cc[r][2*k+1]);
        printf("\n  ref-rowonly: ");
        for (int k = 0; k < 3; k++) printf("(%7.3f %7.3f) ", R[r][2*k], R[r][2*k+1]);
        printf("\n");
    }
    return 0;
}
