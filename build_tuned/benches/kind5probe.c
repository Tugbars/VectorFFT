#include <stdio.h>
#include <string.h>
#include "wisdom2.h"
#include "wisdom2_oop_reader.h"
int main(int argc, char **argv)
{
    vw2_store_t st; int kv = 0, rc, i;
    if (argc < 2) return 2;
    if (vw2_open(&st, argv[1], 1) != VW2_OK) { printf("open failed\n"); return 1; }
    printf("opened, nrec=%d\n", st.nrec);
    for (i = 0; i < 4; i++) {
        rc = vw2_oop_bank_zr2c_slot(&st, 8192, (i >> 1) & 1, i & 1, i & 1, 100.0 + i);
        printf("  bank slot %d -> rc=%d  nrec=%d\n", i, rc, st.nrec);
    }
    vw2_close(&st);
    if (vw2_open(&st, argv[1], 1) != VW2_OK) { printf("reopen failed\n"); return 1; }
    printf("reopened, nrec=%d\n", st.nrec);
    rc = vw2_oop_lookup_zr2c(&st, 8192, &kv);
    printf("lookup -> %d  kv=0x%x\n", rc, kv);
    for (i = 0; i < 4; i++) printf("  slot %d = %d\n", i, vfft_zr2c_kv_get(kv, i));
    vw2_close(&st);
    return 0;
}
