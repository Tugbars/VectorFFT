# VectorFFT R=4 tuning report

Total measurements: **384**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 84 | 85 | 72 | 82 | t1s |
| avx2 | 64 | 72 | 83 | 84 | 69 | 83 | t1s |
| avx2 | 64 | 512 | 83 | 84 | 70 | 82 | t1s |
| avx2 | 128 | 128 | 164 | 175 | 139 | 163 | t1s |
| avx2 | 128 | 136 | 166 | 166 | 137 | 158 | t1s |
| avx2 | 128 | 1024 | 163 | 167 | 142 | 164 | t1s |
| avx2 | 256 | 256 | 324 | 332 | — | 328 | flat |
| avx2 | 256 | 264 | 324 | 332 | — | 332 | flat |
| avx2 | 256 | 2048 | 330 | 332 | — | 331 | flat |
| avx2 | 512 | 512 | 654 | 668 | — | 664 | flat |
| avx2 | 512 | 520 | 691 | 663 | — | 673 | log3 |
| avx2 | 512 | 4096 | 658 | 690 | — | 662 | flat |
| avx2 | 1024 | 1024 | 1779 | 1616 | — | 1804 | log3 |
| avx2 | 1024 | 1032 | 1709 | 1617 | — | 1661 | log3 |
| avx2 | 1024 | 8192 | 1750 | 1624 | — | 1713 | log3 |
| avx2 | 2048 | 2048 | 3352 | 3266 | — | 3380 | log3 |
| avx2 | 2048 | 2056 | 3307 | 3235 | — | 3265 | log3 |
| avx2 | 2048 | 16384 | 3532 | 3359 | — | 3537 | log3 |
| avx512 | 64 | 64 | 46 | 64 | 43 | 53 | t1s |
| avx512 | 64 | 72 | 45 | 63 | 44 | 54 | t1s |
| avx512 | 64 | 512 | 49 | 61 | 45 | 53 | t1s |
| avx512 | 128 | 128 | 92 | 124 | 88 | 104 | t1s |
| avx512 | 128 | 136 | 91 | 123 | 88 | 103 | t1s |
| avx512 | 128 | 1024 | 94 | 201 | 88 | 102 | t1s |
| avx512 | 256 | 256 | 180 | 258 | — | 202 | flat |
| avx512 | 256 | 264 | 178 | 254 | — | 204 | flat |
| avx512 | 256 | 2048 | 182 | 252 | — | 199 | flat |
| avx512 | 512 | 512 | 405 | 487 | — | 402 | flat |
| avx512 | 512 | 520 | 408 | 516 | — | 426 | flat |
| avx512 | 512 | 4096 | 407 | 511 | — | 426 | flat |
| avx512 | 1024 | 1024 | 1657 | 1468 | — | 1650 | log3 |
| avx512 | 1024 | 1032 | 1656 | 1622 | — | 1645 | log3 |
| avx512 | 1024 | 8192 | 1622 | 1411 | — | 1601 | log3 |
| avx512 | 2048 | 2048 | 3388 | 3222 | — | 3289 | log3 |
| avx512 | 2048 | 2056 | 3390 | 3219 | — | 3416 | log3 |
| avx512 | 2048 | 16384 | 3336 | 3331 | — | 3338 | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit_log1` | 82 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit_log1` | 83 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 82 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 164 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_log1` | 158 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 163 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 324 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 324 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 330 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 654 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 673 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 658 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 1779 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1661 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1713 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3352 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3265 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3532 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 85 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 175 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 166 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 167 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 332 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 332 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 332 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 668 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 663 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 690 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1616 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1617 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1624 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3266 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3235 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3359 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 72 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 69 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 70 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 139 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 137 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 142 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 46 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 45 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 49 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 92 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 91 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 94 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit_u2` | 180 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_u2` | 178 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit_u2` | 182 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 402 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 408 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 407 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1650 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1645 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1601 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3289 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 3390 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3336 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 64 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 63 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 61 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 124 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 123 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 201 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 258 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 254 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 252 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 487 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 516 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 511 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1468 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1622 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1411 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3222 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3219 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3331 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 43 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 44 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 45 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 88 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 88 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 88 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 89 | 82 | -7.3% |
| avx2 | 64 | 72 | 85 | 83 | -2.4% |
| avx2 | 64 | 512 | 83 | 82 | -2.0% |
| avx2 | 128 | 128 | 166 | 163 | -1.9% |
| avx2 | 128 | 136 | 170 | 158 | -7.0% |
| avx2 | 128 | 1024 | 166 | 164 | -1.1% |
| avx2 | 256 | 256 | 329 | 328 | -0.5% |
| avx2 | 256 | 264 | 330 | 332 | +0.6% |
| avx2 | 256 | 2048 | 334 | 331 | -0.8% |
| avx2 | 512 | 512 | 654 | 664 | +1.6% |
| avx2 | 512 | 520 | 691 | 673 | -2.6% |
| avx2 | 512 | 4096 | 658 | 662 | +0.6% |
| avx2 | 1024 | 1024 | 1896 | 1804 | -4.8% |
| avx2 | 1024 | 1032 | 1709 | 1661 | -2.8% |
| avx2 | 1024 | 8192 | 1750 | 1713 | -2.1% |
| avx2 | 2048 | 2048 | 3352 | 3380 | +0.8% |
| avx2 | 2048 | 2056 | 3307 | 3265 | -1.3% |
| avx2 | 2048 | 16384 | 3532 | 3537 | +0.1% |
| avx512 | 64 | 64 | 51 | 53 | +2.5% |
| avx512 | 64 | 72 | 53 | 54 | +0.5% |
| avx512 | 64 | 512 | 51 | 53 | +3.4% |
| avx512 | 128 | 128 | 102 | 104 | +1.6% |
| avx512 | 128 | 136 | 102 | 103 | +0.7% |
| avx512 | 128 | 1024 | 100 | 102 | +1.7% |
| avx512 | 256 | 256 | 200 | 202 | +1.2% |
| avx512 | 256 | 264 | 199 | 204 | +2.3% |
| avx512 | 256 | 2048 | 205 | 199 | -2.6% |
| avx512 | 512 | 512 | 405 | 402 | -0.7% |
| avx512 | 512 | 520 | 408 | 426 | +4.5% |
| avx512 | 512 | 4096 | 407 | 426 | +4.7% |
| avx512 | 1024 | 1024 | 1657 | 1650 | -0.4% |
| avx512 | 1024 | 1032 | 1656 | 1645 | -0.7% |
| avx512 | 1024 | 8192 | 1622 | 1601 | -1.3% |
| avx512 | 2048 | 2048 | 3525 | 3289 | -6.7% |
| avx512 | 2048 | 2056 | 3324 | 3416 | +2.8% |
| avx512 | 2048 | 16384 | 3336 | 3338 | +0.1% |