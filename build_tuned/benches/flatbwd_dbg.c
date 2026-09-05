#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vfft.h"
#include "../../src/core/oop/il_flatdit.h"
int main(void) {
    static const int NS[] = { 405, 972, 3125 };
    for (int i = 0; i < 3; i++) {
        const int N = NS[i];
        vfft_ilfd_plan_t *p = vfft_ilfd_create(N);
        double *x = malloc(2*(size_t)N*8), *z = malloc(2*(size_t)N*8), *y = malloc(2*(size_t)N*8);
        for (int j = 0; j < 2*N; j++) x[j] = (double)rand()/RAND_MAX-0.5;
        fprintf(stderr, "N=%d K=%d bwd_ok=%d\n", N, p->K, p->bwd_ok);
        vfft_ilfd_execute_fwd(p, x, z);
        for (int s = 0; s < p->K; s++) {
            fprintf(stderr, "  bwd stage %d R%d D%zu tail=%d msz=%d gl=%d fb=%p fcsb=%p fglb=%p fzb=%p tfb=%p t2gb=%p tzb=%p\n",
                    s, p->R[s], p->D[s], p->tail[s], p->msz[s], p->gl[s], (void*)p->fb[s], (void*)p->fcsb[s], (void*)p->fglb[s], (void*)p->fzb[s], (void*)p->tfb[s], (void*)p->t2gb[s], (void*)p->tzb[s]);
            fflush(stderr);
            vfft_ilfd_stage_bwd(p, s, z, y);
        }
        double rt = 0; for (int j = 0; j < 2*N; j++) { double d = fabs(y[j]/N - x[j]); if (d > rt) rt = d; }
        fprintf(stderr, "  roundtrip err %.2e\n", rt);
        vfft_ilfd_destroy(p); free(x); free(z); free(y);
    }
    return 0;
}
