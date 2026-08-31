/* split_state_probe.c - is the r2c dispatch state SHARED with the library?
 *
 * This reproduces the exact shape of the bug step 21b fixes. A bench that
 * includes r2c_dispatch.h and links vfft.c as a SEPARATE translation unit gets
 * its own copy of the header's file-scope statics. The header's setter is
 * `static inline`, so it writes THIS TU's copy; vfft_create keeps reading the
 * library's. The write appears to succeed and changes nothing.
 *
 * Two arms:
 *   OLD  the header's static-inline setter  -> expected NOT to reach vfft.c
 *   NEW  the library-side hook (step 21b)   -> expected to reach vfft.c
 *
 * "Reaching vfft.c" is read back through vfft_r2c_get_decouple_min_k(), which
 * is compiled into vfft.c and therefore reports the library's copy.
 *
 * Build: python build.py --src benches/split_state_probe.c --vfft --compile
 */
#include <stdio.h>
#include "vfft.h"
#include "r2c_dispatch.h"   /* the same include the vs-MKL bench uses */

int main(void)
{
    int fail = 0;
    size_t lib0 = vfft_r2c_get_decouple_min_k();
    printf("library default            : %zu\n", lib0);

    /* ARM 1 - the old spelling. Writes THIS TU's copy. */
    vfft_r2c_dispatch_set_decouple_min_k(777);
    size_t mine = vfft_r2c_dispatch_get_decouple_min_k();
    size_t lib1 = vfft_r2c_get_decouple_min_k();
    printf("after header setter(777)   : this TU=%zu  library=%zu  -> %s\n",
           mine, lib1,
           lib1 == 777 ? "shared" : "SPLIT (this is the bug)");
    if (mine != 777) { printf("  !! header setter did not even write its own copy\n"); fail = 1; }

    /* ARM 2 - the library-side hook. Must reach vfft.c. */
    vfft_r2c_set_decouple_min_k(4242);
    size_t lib2 = vfft_r2c_get_decouple_min_k();
    printf("after library hook(4242)   : library=%zu  -> %s\n",
           lib2, lib2 == 4242 ? "REACHED (fixed)" : "still split");
    if (lib2 != 4242) { printf("  !! the fix does not work\n"); fail = 1; }

    printf("\nVERDICT: %s\n", fail ? "FAIL" : "the library-side hook reaches vfft.c");
    return fail;
}
