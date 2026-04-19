# VectorFFT R=4 tuning report

Total measurements: **384**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 88 | 91 | 72 | 86 | t1s |
| avx2 | 64 | 72 | 84 | 90 | 74 | 87 | t1s |
| avx2 | 64 | 512 | 84 | 88 | 74 | 94 | t1s |
| avx2 | 128 | 128 | 163 | 170 | 152 | 172 | t1s |
| avx2 | 128 | 136 | 167 | 167 | 145 | 167 | t1s |
| avx2 | 128 | 1024 | 170 | 169 | 143 | 168 | t1s |
| avx2 | 256 | 256 | 326 | 334 | — | 319 | flat |
| avx2 | 256 | 264 | 325 | 338 | — | 325 | flat |
| avx2 | 256 | 2048 | 322 | 362 | — | 329 | flat |
| avx2 | 512 | 512 | 646 | 673 | — | 647 | flat |
| avx2 | 512 | 520 | 665 | 682 | — | 657 | flat |
| avx2 | 512 | 4096 | 648 | 666 | — | 663 | flat |
| avx2 | 1024 | 1024 | 1672 | 1471 | — | 1630 | log3 |
| avx2 | 1024 | 1032 | 1617 | 1557 | — | 1553 | log3 |
| avx2 | 1024 | 8192 | 1635 | 1485 | — | 1636 | log3 |
| avx2 | 2048 | 2048 | 3257 | 3148 | — | 3230 | log3 |
| avx2 | 2048 | 2056 | 3364 | 3121 | — | 3180 | log3 |
| avx2 | 2048 | 16384 | 3359 | 3134 | — | 3390 | log3 |
| avx512 | 64 | 64 | 60 | 63 | 53 | 62 | t1s |
| avx512 | 64 | 72 | 61 | 66 | 54 | 63 | t1s |
| avx512 | 64 | 512 | 59 | 66 | 55 | 62 | t1s |
| avx512 | 128 | 128 | 120 | 130 | 108 | 120 | t1s |
| avx512 | 128 | 136 | 118 | 131 | 105 | 121 | t1s |
| avx512 | 128 | 1024 | 117 | 131 | 109 | 123 | t1s |
| avx512 | 256 | 256 | 234 | 262 | — | 248 | flat |
| avx512 | 256 | 264 | 229 | 256 | — | 242 | flat |
| avx512 | 256 | 2048 | 236 | 260 | — | 250 | flat |
| avx512 | 512 | 512 | 494 | 527 | — | 487 | flat |
| avx512 | 512 | 520 | 490 | 592 | — | 489 | flat |
| avx512 | 512 | 4096 | 484 | 523 | — | 484 | flat |
| avx512 | 1024 | 1024 | 1506 | 1403 | — | 1606 | log3 |
| avx512 | 1024 | 1032 | 1449 | 1514 | — | 1447 | flat |
| avx512 | 1024 | 8192 | 1430 | 1406 | — | 1411 | log3 |
| avx512 | 2048 | 2048 | 2945 | 3140 | — | 2909 | flat |
| avx512 | 2048 | 2056 | 2913 | 3082 | — | 2916 | flat |
| avx512 | 2048 | 16384 | 2990 | 3171 | — | 2956 | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit_log1` | 86 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit_log1` | 84 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 84 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit_log1` | 163 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_log1` | 167 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 168 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 319 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 325 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 322 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 646 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 657 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 648 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1630 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1553 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1635 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3230 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3180 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3359 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 91 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 90 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 88 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 170 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 167 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 169 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 334 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 338 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 362 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 673 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 682 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 666 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1471 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1557 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1485 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3148 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3121 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3134 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 72 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 74 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 74 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 152 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 145 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 143 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 60 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 61 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit_u2` | 59 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit_log1` | 120 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit_u2` | 118 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_u2` | 117 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 234 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 229 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 236 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 487 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 489 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 484 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 1506 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1447 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1411 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 2909 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 2913 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 2956 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 63 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 66 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 66 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 130 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 131 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 131 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 262 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 256 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 260 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 527 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 592 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 523 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1403 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1514 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1406 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3140 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3082 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3171 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 53 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 54 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 55 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 108 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 105 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 109 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 89 | 86 | -3.6% |
| avx2 | 64 | 72 | 84 | 87 | +3.9% |
| avx2 | 64 | 512 | 84 | 94 | +11.2% |
| avx2 | 128 | 128 | 163 | 172 | +5.6% |
| avx2 | 128 | 136 | 167 | 167 | +0.1% |
| avx2 | 128 | 1024 | 170 | 168 | -1.3% |
| avx2 | 256 | 256 | 326 | 319 | -2.2% |
| avx2 | 256 | 264 | 325 | 325 | -0.0% |
| avx2 | 256 | 2048 | 322 | 329 | +2.4% |
| avx2 | 512 | 512 | 646 | 647 | +0.0% |
| avx2 | 512 | 520 | 665 | 657 | -1.3% |
| avx2 | 512 | 4096 | 648 | 663 | +2.3% |
| avx2 | 1024 | 1024 | 1672 | 1630 | -2.5% |
| avx2 | 1024 | 1032 | 1617 | 1553 | -4.0% |
| avx2 | 1024 | 8192 | 1635 | 1636 | +0.1% |
| avx2 | 2048 | 2048 | 3257 | 3230 | -0.8% |
| avx2 | 2048 | 2056 | 3472 | 3180 | -8.4% |
| avx2 | 2048 | 16384 | 3359 | 3390 | +0.9% |
| avx512 | 64 | 64 | 61 | 62 | +0.3% |
| avx512 | 64 | 72 | 61 | 63 | +1.9% |
| avx512 | 64 | 512 | 63 | 62 | -2.5% |
| avx512 | 128 | 128 | 123 | 120 | -2.1% |
| avx512 | 128 | 136 | 121 | 121 | -0.1% |
| avx512 | 128 | 1024 | 124 | 123 | -0.6% |
| avx512 | 256 | 256 | 248 | 248 | +0.3% |
| avx512 | 256 | 264 | 242 | 242 | -0.1% |
| avx512 | 256 | 2048 | 242 | 250 | +3.4% |
| avx512 | 512 | 512 | 494 | 487 | -1.5% |
| avx512 | 512 | 520 | 490 | 489 | -0.4% |
| avx512 | 512 | 4096 | 484 | 484 | +0.1% |
| avx512 | 1024 | 1024 | 1509 | 1606 | +6.5% |
| avx512 | 1024 | 1032 | 1449 | 1447 | -0.1% |
| avx512 | 1024 | 8192 | 1430 | 1411 | -1.3% |
| avx512 | 2048 | 2048 | 2945 | 2909 | -1.2% |
| avx512 | 2048 | 2056 | 2913 | 2916 | +0.1% |
| avx512 | 2048 | 16384 | 2990 | 2956 | -1.2% |