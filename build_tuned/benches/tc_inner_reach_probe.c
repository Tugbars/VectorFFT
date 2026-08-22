/* tc_inner_reach_probe.c — ADVERSARIAL CHECK of the claim that flipping the
 * interleaved REAL batch_geom DEFAULT to transform-contiguous can turn a
 * today-succeeding K>1 create into NULL, because the wrapper hard-fails when
 * its K=1 inner cannot be built (src/core/vfft.c:3304-3312).
 *
 * The hazard REQUIRES the K=1 inner to fail. So: build the wrapper's inner
 * EXACTLY as vfft.c:3300-3304 constructs it (same cfg, howmany=1,
 * batch_geom=LANE_MAJOR) for a wide N sweep. Only where that fails do we
 * bother asking whether today's K>1 DEFAULT plan builds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static int mk(vfft_transform_t xf, int N, size_t K, int geom, vfft_placement_t pl)
{
    vfft_config_t c;
    vfft_plan p;
    memset(&c, 0, sizeof c);
    c.transform = xf;
    c.placement = pl;
    c.rigor = VFFT_MEASURE;
    c.dims = 1;
    c.n[0] = N;
    c.howmany = K;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.batch_geom = geom;
    c.wisdom_write = 0;
    p = vfft_create(&c);
    if (!p) return 0;
    vfft_destroy(p);
    return 1;
}

int main(void)
{
    static const int Ns[] = {
        4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28,30,32,33,34,35,36,
        38,40,44,45,46,48,49,50,52,54,55,56,60,62,63,64,65,66,70,72,75,77,80,
        81,84,88,90,96,98,99,100,110,112,120,121,125,126,128,130,132,143,144,
        150,160,162,169,176,180,192,196,200,210,216,220,240,242,243,250,252,
        256,264,286,288,300,320,338,360,384,400,405,440,480,500,512,576,600,
        625,640,720,768,800,960,1000,1024,1200,1280,1440,1536,1600,2000,2048,
        2400,2560,3072,4096,8192,
        11,13,17,19,23,29,31,37,41,101,127,257,509,
    };
    const size_t Ks[] = {2,3,4,5,8,16};
    int nN = (int)(sizeof Ns / sizeof Ns[0]);
    int hazards = 0, innerfail = 0, i, ki, t;
    for (t = 0; t < 2; t++)
    {
        vfft_transform_t xf = t ? VFFT_C2R : VFFT_R2C;
        const char *nm = t ? "c2r" : "r2c";
        for (i = 0; i < nN; i++)
        {
            int N = Ns[i];
            int inner = mk(xf, N, 1, VFFT_BATCH_LANE_MAJOR, VFFT_OUTOFPLACE);
            printf("%s N=%-6d K1inner=%s\n", nm, N, inner ? "ok" : "FAIL");
            fflush(stdout);
            if (inner) continue;
            innerfail++;
            for (ki = 0; ki < (int)(sizeof Ks / sizeof Ks[0]); ki++)
            {
                size_t K = Ks[ki];
                if (mk(xf, N, K, VFFT_BATCH_DEFAULT, VFFT_OUTOFPLACE))
                {
                    printf("   *** HAZARD *** %s N=%d K=%zu builds today, K=1 inner does not\n",
                           nm, N, K);
                    hazards++;
                }
            }
            fflush(stdout);
        }
    }
    printf("\nK=1 inner failures: %d   HAZARD cells: %d\n", innerfail, hazards);
    return 0;
}
