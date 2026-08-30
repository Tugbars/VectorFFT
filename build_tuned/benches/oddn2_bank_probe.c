/* oddn2_bank_probe.c - did the column-MT verdict RACE, or replay a bank?
 *
 * The create-race counter cannot answer this: none of the four MT racers
 * (_zt_mt_race, _pq_mt_race, _il2d_real_colmt_race, _il2d_c2c_mt_race)
 * increments _vfft_create_race_count, so races=0 on an MT plan is a FALSE
 * ZERO - it means "no instrumented race ran", not "nothing was measured".
 *
 * So ask the store instead: run with wisdom_write=1 against a SCRATCH copy and
 * see whether a new record appears. A race that banks leaves evidence; a replay
 * leaves the store byte-identical.
 *
 * Scratch only - never point this at the shipped tree.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/oddn2_bank_probe.c --vfft --compile
 * Run  : oddn2_bank_probe.exe <nthreads>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    int nthr = (argc > 1) ? atoi(argv[1]) : 8;
    char *q;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = 128; cfg.n[1] = 127;
    cfg.howmany  = 1;
    cfg.nthreads = nthr;
    cfg.rigor    = VFFT_MEASURE;
    cfg.wisdom_write = 1;             /* let it bank, into the SCRATCH dir */

    p = vfft_create(&cfg);
    if (!p) { printf("REFUSED\n"); return 2; }
    vfft__fp_counters(c);
    vfft__fingerprint(p, buf, sizeof buf);
    q = strstr(buf, "cmt=");
    printf("nthr=%d  cmt=%s  counted_races=%ld\n", nthr,
           q ? (strncmp(q, "cmt=0", 5) ? "1" : "0") : "?", c[5]);
    vfft_destroy(p);
    return 0;
}
