# VectorFFT R=4 tuning report

Total measurements: **384**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 82 | 84 | 74 | 87 | t1s |
| avx2 | 64 | 72 | 82 | 84 | 73 | 87 | t1s |
| avx2 | 64 | 512 | 83 | 84 | 74 | 83 | t1s |
| avx2 | 128 | 128 | 164 | 168 | 143 | 169 | t1s |
| avx2 | 128 | 136 | 165 | 175 | 147 | 166 | t1s |
| avx2 | 128 | 1024 | 162 | 211 | 143 | 173 | t1s |
| avx2 | 256 | 256 | 329 | 343 | — | 326 | flat |
| avx2 | 256 | 264 | 329 | 349 | — | 326 | flat |
| avx2 | 256 | 2048 | 327 | 348 | — | 326 | flat |
| avx2 | 512 | 512 | 656 | 706 | — | 661 | flat |
| avx2 | 512 | 520 | 670 | 681 | — | 666 | flat |
| avx2 | 512 | 4096 | 736 | 676 | — | 656 | log3 |
| avx2 | 1024 | 1024 | 1732 | 1663 | — | 1668 | log3 |
| avx2 | 1024 | 1032 | 1667 | 1627 | — | 1654 | log3 |
| avx2 | 1024 | 8192 | 1755 | 1635 | — | 1734 | log3 |
| avx2 | 2048 | 2048 | 3388 | 3327 | — | 3311 | log3 |
| avx2 | 2048 | 2056 | 3318 | 3216 | — | 3296 | log3 |
| avx2 | 2048 | 16384 | 3498 | 3505 | — | 3503 | flat |
| avx512 | 64 | 64 | 46 | 61 | 46 | 53 | t1s |
| avx512 | 64 | 72 | 48 | 63 | 45 | 54 | t1s |
| avx512 | 64 | 512 | 48 | 63 | 47 | 51 | t1s |
| avx512 | 128 | 128 | 90 | 123 | 92 | 105 | flat |
| avx512 | 128 | 136 | 94 | 130 | 92 | 101 | t1s |
| avx512 | 128 | 1024 | 98 | 132 | 89 | 103 | t1s |
| avx512 | 256 | 256 | 179 | 257 | — | 211 | flat |
| avx512 | 256 | 264 | 181 | 259 | — | 201 | flat |
| avx512 | 256 | 2048 | 180 | 256 | — | 212 | flat |
| avx512 | 512 | 512 | 412 | 869 | — | 422 | flat |
| avx512 | 512 | 520 | 432 | 516 | — | 448 | flat |
| avx512 | 512 | 4096 | 409 | 509 | — | 432 | flat |
| avx512 | 1024 | 1024 | 1659 | 1515 | — | 1941 | log3 |
| avx512 | 1024 | 1032 | 1672 | 1626 | — | 1665 | log3 |
| avx512 | 1024 | 8192 | 1611 | 1436 | — | 1620 | log3 |
| avx512 | 2048 | 2048 | 3405 | 3278 | — | 3368 | log3 |
| avx512 | 2048 | 2056 | 3293 | 3260 | — | 3270 | log3 |
| avx512 | 2048 | 16384 | 3347 | 3377 | — | 3352 | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit_u2` | 82 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit_log1` | 82 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 83 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 164 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_log1` | 165 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 162 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 329 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 326 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 326 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 656 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 666 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 656 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1668 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1654 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1734 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3311 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3296 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3498 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 168 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 175 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 211 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 343 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 349 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 348 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 706 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 681 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 676 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1663 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1627 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1635 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3327 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3216 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3505 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 74 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 74 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 143 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 147 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 143 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 46 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 48 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 48 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 90 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 94 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_u2` | 98 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 179 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_u2` | 181 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 180 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 412 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 432 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 409 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1659 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1665 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1611 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 3405 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3270 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3347 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 61 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 63 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 63 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 123 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 130 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 132 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 257 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 259 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 256 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 869 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 516 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 509 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1515 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1626 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1436 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3278 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3260 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3377 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 46 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 45 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 47 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 92 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 92 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 89 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 85 | 87 | +2.4% |
| avx2 | 64 | 72 | 82 | 87 | +5.7% |
| avx2 | 64 | 512 | 82 | 83 | +1.5% |
| avx2 | 128 | 128 | 163 | 169 | +3.5% |
| avx2 | 128 | 136 | 165 | 166 | +0.4% |
| avx2 | 128 | 1024 | 162 | 173 | +6.6% |
| avx2 | 256 | 256 | 329 | 326 | -0.7% |
| avx2 | 256 | 264 | 329 | 326 | -0.8% |
| avx2 | 256 | 2048 | 327 | 326 | -0.4% |
| avx2 | 512 | 512 | 656 | 661 | +0.8% |
| avx2 | 512 | 520 | 670 | 666 | -0.6% |
| avx2 | 512 | 4096 | 736 | 656 | -10.9% |
| avx2 | 1024 | 1024 | 1732 | 1668 | -3.7% |
| avx2 | 1024 | 1032 | 1667 | 1654 | -0.8% |
| avx2 | 1024 | 8192 | 1755 | 1734 | -1.2% |
| avx2 | 2048 | 2048 | 3388 | 3311 | -2.3% |
| avx2 | 2048 | 2056 | 3318 | 3296 | -0.7% |
| avx2 | 2048 | 16384 | 3498 | 3503 | +0.1% |
| avx512 | 64 | 64 | 52 | 53 | +2.9% |
| avx512 | 64 | 72 | 51 | 54 | +5.5% |
| avx512 | 64 | 512 | 51 | 51 | +1.1% |
| avx512 | 128 | 128 | 105 | 105 | -0.2% |
| avx512 | 128 | 136 | 104 | 101 | -2.4% |
| avx512 | 128 | 1024 | 105 | 103 | -1.8% |
| avx512 | 256 | 256 | 203 | 211 | +3.7% |
| avx512 | 256 | 264 | 200 | 201 | +0.7% |
| avx512 | 256 | 2048 | 200 | 212 | +6.1% |
| avx512 | 512 | 512 | 412 | 422 | +2.4% |
| avx512 | 512 | 520 | 432 | 448 | +3.7% |
| avx512 | 512 | 4096 | 409 | 432 | +5.4% |
| avx512 | 1024 | 1024 | 1659 | 1941 | +17.0% |
| avx512 | 1024 | 1032 | 1672 | 1665 | -0.4% |
| avx512 | 1024 | 8192 | 1611 | 1620 | +0.6% |
| avx512 | 2048 | 2048 | 3346 | 3368 | +0.7% |
| avx512 | 2048 | 2056 | 3293 | 3270 | -0.7% |
| avx512 | 2048 | 16384 | 3347 | 3352 | +0.1% |