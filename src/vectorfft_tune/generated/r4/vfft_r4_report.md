# VectorFFT R=4 tuning report

Total measurements: **384**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 88 | 87 | 73 | 84 | t1s |
| avx2 | 64 | 72 | 85 | 87 | 72 | 84 | t1s |
| avx2 | 64 | 512 | 86 | 88 | 73 | 84 | t1s |
| avx2 | 128 | 128 | 168 | 171 | 144 | 174 | t1s |
| avx2 | 128 | 136 | 170 | 176 | 143 | 166 | t1s |
| avx2 | 128 | 1024 | 172 | 168 | 174 | 168 | log3 |
| avx2 | 256 | 256 | 334 | 333 | — | 337 | log3 |
| avx2 | 256 | 264 | 341 | 363 | — | 333 | flat |
| avx2 | 256 | 2048 | 334 | 337 | — | 327 | flat |
| avx2 | 512 | 512 | 669 | 676 | — | 684 | flat |
| avx2 | 512 | 520 | 701 | 667 | — | 681 | log3 |
| avx2 | 512 | 4096 | 679 | 687 | — | 673 | flat |
| avx2 | 1024 | 1024 | 1626 | 1588 | — | 1634 | log3 |
| avx2 | 1024 | 1032 | 1580 | 1621 | — | 1597 | flat |
| avx2 | 1024 | 8192 | 1700 | 1478 | — | 1669 | log3 |
| avx2 | 2048 | 2048 | 3212 | 3224 | — | 3260 | flat |
| avx2 | 2048 | 2056 | 3208 | 3183 | — | 3198 | log3 |
| avx2 | 2048 | 16384 | 3381 | 3125 | — | 3371 | log3 |
| avx512 | 64 | 64 | 60 | 65 | 53 | 64 | t1s |
| avx512 | 64 | 72 | 59 | 67 | 53 | 62 | t1s |
| avx512 | 64 | 512 | 59 | 66 | 54 | 62 | t1s |
| avx512 | 128 | 128 | 120 | 128 | 106 | 122 | t1s |
| avx512 | 128 | 136 | 118 | 127 | 104 | 122 | t1s |
| avx512 | 128 | 1024 | 118 | 131 | 106 | 122 | t1s |
| avx512 | 256 | 256 | 235 | 270 | — | 249 | flat |
| avx512 | 256 | 264 | 228 | 261 | — | 247 | flat |
| avx512 | 256 | 2048 | 234 | 267 | — | 244 | flat |
| avx512 | 512 | 512 | 504 | 521 | — | 497 | flat |
| avx512 | 512 | 520 | 518 | 530 | — | 739 | flat |
| avx512 | 512 | 4096 | 486 | 525 | — | 513 | flat |
| avx512 | 1024 | 1024 | 1516 | 1406 | — | 1594 | log3 |
| avx512 | 1024 | 1032 | 1465 | 1484 | — | 1454 | flat |
| avx512 | 1024 | 8192 | 1553 | 1389 | — | 1535 | log3 |
| avx512 | 2048 | 2048 | 2948 | 3139 | — | 2928 | flat |
| avx512 | 2048 | 2056 | 3084 | 2890 | — | 3067 | log3 |
| avx512 | 2048 | 16384 | 2981 | 3150 | — | 3037 | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit_log1` | 84 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit_log1` | 84 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 84 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit_log1` | 168 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_log1` | 166 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 168 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 334 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 333 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 327 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 669 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 681 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 673 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1626 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1580 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1669 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3212 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3198 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3371 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 87 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 87 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 88 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 171 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 176 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 168 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 333 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 363 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 337 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 676 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 667 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 687 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1588 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1621 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1478 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3224 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3183 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3125 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 72 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 144 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 143 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 174 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit_u2` | 60 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit_u2` | 59 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit_u2` | 59 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 120 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 118 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_u2` | 118 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 235 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 228 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 234 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 497 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 518 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 486 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 1516 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1454 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 1553 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 2928 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 3084 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 2981 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 65 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 67 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 66 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 128 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 127 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 131 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 270 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 261 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 267 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 521 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 530 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 525 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1406 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1484 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1389 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3139 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 2890 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3150 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 53 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 53 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 54 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 106 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 104 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 106 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 86 | 84 | -3.2% |
| avx2 | 64 | 72 | 85 | 84 | -1.9% |
| avx2 | 64 | 512 | 86 | 84 | -2.9% |
| avx2 | 128 | 128 | 168 | 174 | +3.9% |
| avx2 | 128 | 136 | 170 | 166 | -2.2% |
| avx2 | 128 | 1024 | 194 | 168 | -13.0% |
| avx2 | 256 | 256 | 334 | 337 | +1.0% |
| avx2 | 256 | 264 | 341 | 333 | -2.5% |
| avx2 | 256 | 2048 | 334 | 327 | -2.2% |
| avx2 | 512 | 512 | 669 | 684 | +2.2% |
| avx2 | 512 | 520 | 701 | 681 | -2.9% |
| avx2 | 512 | 4096 | 679 | 673 | -0.9% |
| avx2 | 1024 | 1024 | 1626 | 1634 | +0.5% |
| avx2 | 1024 | 1032 | 1580 | 1597 | +1.1% |
| avx2 | 1024 | 8192 | 1700 | 1669 | -1.9% |
| avx2 | 2048 | 2048 | 3212 | 3260 | +1.5% |
| avx2 | 2048 | 2056 | 3208 | 3198 | -0.3% |
| avx2 | 2048 | 16384 | 3381 | 3371 | -0.3% |
| avx512 | 64 | 64 | 64 | 64 | -0.2% |
| avx512 | 64 | 72 | 62 | 62 | -1.2% |
| avx512 | 64 | 512 | 64 | 62 | -2.7% |
| avx512 | 128 | 128 | 124 | 122 | -1.2% |
| avx512 | 128 | 136 | 122 | 122 | -0.6% |
| avx512 | 128 | 1024 | 123 | 122 | -0.2% |
| avx512 | 256 | 256 | 246 | 249 | +1.3% |
| avx512 | 256 | 264 | 246 | 247 | +0.1% |
| avx512 | 256 | 2048 | 245 | 244 | -0.5% |
| avx512 | 512 | 512 | 504 | 497 | -1.3% |
| avx512 | 512 | 520 | 518 | 739 | +42.6% |
| avx512 | 512 | 4096 | 486 | 513 | +5.6% |
| avx512 | 1024 | 1024 | 1682 | 1594 | -5.2% |
| avx512 | 1024 | 1032 | 1465 | 1454 | -0.7% |
| avx512 | 1024 | 8192 | 1542 | 1535 | -0.4% |
| avx512 | 2048 | 2048 | 2948 | 2928 | -0.7% |
| avx512 | 2048 | 2056 | 3127 | 3067 | -1.9% |
| avx512 | 2048 | 16384 | 2981 | 3037 | +1.8% |