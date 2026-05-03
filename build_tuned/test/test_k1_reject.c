/* test_k1_reject.c -- verify R2C / DCT / DST / DHT all reject K=1 cleanly.
 *
 * Prior bug: K=1 collapsed inner-FFT block size to 1, codelets did aligned
 * loads of 4 doubles and corrupted the heap. Fix returns NULL from the
 * R2C planner; r2r features built atop R2C inherit the rejection.
 */
#include <stdio.h>
#include <stdlib.h>

#include "planner.h"
#include "r2c.h"
#include "dct.h"
#include "dst.h"
#include "dct4.h"
#include "dht.h"
#include "env.h"

static int check_null(const char *name, stride_plan_t *p) {
    if (p == NULL) {
        printf("  %-30s K=1 -> NULL  PASS\n", name);
        return 0;
    }
    printf("  %-30s K=1 returned non-NULL  FAIL\n", name);
    stride_plan_destroy(p);
    return 1;
}

static int check_ok(const char *name, stride_plan_t *p) {
    if (p == NULL) {
        printf("  %-30s K=2 returned NULL    FAIL\n", name);
        return 1;
    }
    printf("  %-30s K=2 -> plan          PASS\n", name);
    stride_plan_destroy(p);
    return 0;
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== test_k1_reject -- K=1 rejected, K=2 accepted across r2r ===\n\n");

    int N = 16;
    int fail = 0;

    printf("[K=1 must be rejected]\n");
    fail += check_null("stride_r2c_auto_plan",      stride_r2c_auto_plan(N, 1, &reg));
    fail += check_null("stride_dct2_auto_plan",     stride_dct2_auto_plan(N, 1, &reg));
    fail += check_null("stride_dst2_auto_plan",     stride_dst2_auto_plan(N, 1, &reg));
    fail += check_null("stride_dct4_auto_plan",     stride_dct4_auto_plan(N, 1, &reg));
    fail += check_null("stride_dht_auto_plan",      stride_dht_auto_plan(N, 1, &reg));

    printf("\n[K=2 must succeed]\n");
    fail += check_ok("stride_r2c_auto_plan",      stride_r2c_auto_plan(N, 2, &reg));
    fail += check_ok("stride_dct2_auto_plan",     stride_dct2_auto_plan(N, 2, &reg));
    fail += check_ok("stride_dst2_auto_plan",     stride_dst2_auto_plan(N, 2, &reg));
    fail += check_ok("stride_dct4_auto_plan",     stride_dct4_auto_plan(N, 2, &reg));
    fail += check_ok("stride_dht_auto_plan",      stride_dht_auto_plan(N, 2, &reg));

    printf("\n=== %s ===\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}
