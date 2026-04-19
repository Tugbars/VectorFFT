# VectorFFT R=4 tuning report

Total measurements: **384**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 80 | 85 | 71 | 83 | t1s |
| avx2 | 64 | 72 | 80 | 86 | 69 | 82 | t1s |
| avx2 | 64 | 512 | 81 | 85 | 68 | 81 | t1s |
| avx2 | 128 | 128 | 160 | 167 | 135 | 165 | t1s |
| avx2 | 128 | 136 | 159 | 171 | 134 | 160 | t1s |
| avx2 | 128 | 1024 | 158 | 169 | 134 | 162 | t1s |
| avx2 | 256 | 256 | 329 | 341 | — | 315 | flat |
| avx2 | 256 | 264 | 321 | 339 | — | 322 | flat |
| avx2 | 256 | 2048 | 324 | 342 | — | 324 | flat |
| avx2 | 512 | 512 | 649 | 651 | — | 644 | flat |
| avx2 | 512 | 520 | 655 | 664 | — | 658 | flat |
| avx2 | 512 | 4096 | 660 | 648 | — | 655 | log3 |
| avx2 | 1024 | 1024 | 1775 | 1586 | — | 1680 | log3 |
| avx2 | 1024 | 1032 | 1711 | 1597 | — | 1648 | log3 |
| avx2 | 1024 | 8192 | 1698 | 1608 | — | 1736 | log3 |
| avx2 | 2048 | 2048 | 3325 | 3564 | — | 3357 | flat |
| avx2 | 2048 | 2056 | 3189 | 3166 | — | 3346 | log3 |
| avx2 | 2048 | 16384 | 3379 | 3261 | — | 3510 | log3 |
| avx512 | 64 | 64 | 46 | 61 | 43 | 53 | t1s |
| avx512 | 64 | 72 | 46 | 60 | 42 | 53 | t1s |
| avx512 | 64 | 512 | 50 | 64 | 43 | 53 | t1s |
| avx512 | 128 | 128 | 90 | 128 | 86 | 101 | t1s |
| avx512 | 128 | 136 | 89 | 123 | 85 | 103 | t1s |
| avx512 | 128 | 1024 | 91 | 201 | 87 | 101 | t1s |
| avx512 | 256 | 256 | 180 | 244 | — | 199 | flat |
| avx512 | 256 | 264 | 179 | 247 | — | 207 | flat |
| avx512 | 256 | 2048 | 184 | 247 | — | 203 | flat |
| avx512 | 512 | 512 | 407 | 497 | — | 415 | flat |
| avx512 | 512 | 520 | 417 | 491 | — | 428 | flat |
| avx512 | 512 | 4096 | 401 | 496 | — | 461 | flat |
| avx512 | 1024 | 1024 | 1622 | 1402 | — | 1622 | log3 |
| avx512 | 1024 | 1032 | 1687 | 1593 | — | 1670 | log3 |
| avx512 | 1024 | 8192 | 1603 | 1311 | — | 1610 | log3 |
| avx512 | 2048 | 2048 | 3276 | 3215 | — | 3337 | log3 |
| avx512 | 2048 | 2056 | 3289 | 3193 | — | 3191 | log3 |
| avx512 | 2048 | 16384 | 3395 | 3180 | — | 3318 | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit_log1` | 80 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit_log1` | 80 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 81 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 160 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 159 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 158 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 315 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 321 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 324 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 644 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 655 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 655 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1680 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1648 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1698 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3325 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3189 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3379 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 85 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 86 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 85 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 167 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 171 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 169 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 341 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 339 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 342 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 651 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 664 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 648 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1586 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1597 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1608 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3564 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3166 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3261 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 71 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 69 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 68 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 135 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 134 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 134 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 46 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 46 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit_u2` | 50 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 90 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit_u2` | 89 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 91 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 180 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_u2` | 179 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 184 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 407 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 417 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 401 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 1622 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 1687 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 1603 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 3276 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 3191 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 3318 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 61 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 60 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 64 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 128 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 123 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 201 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 244 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 247 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 247 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 497 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 491 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 496 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 1402 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1593 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 1311 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 3215 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3193 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 3180 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 43 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 42 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 43 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 86 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 85 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 87 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 80 | 83 | +3.5% |
| avx2 | 64 | 72 | 80 | 82 | +2.8% |
| avx2 | 64 | 512 | 81 | 81 | -0.0% |
| avx2 | 128 | 128 | 163 | 165 | +0.7% |
| avx2 | 128 | 136 | 165 | 160 | -3.0% |
| avx2 | 128 | 1024 | 162 | 162 | +0.1% |
| avx2 | 256 | 256 | 329 | 315 | -4.4% |
| avx2 | 256 | 264 | 321 | 322 | +0.3% |
| avx2 | 256 | 2048 | 322 | 324 | +0.4% |
| avx2 | 512 | 512 | 649 | 644 | -0.6% |
| avx2 | 512 | 520 | 655 | 658 | +0.3% |
| avx2 | 512 | 4096 | 660 | 655 | -0.9% |
| avx2 | 1024 | 1024 | 1986 | 1680 | -15.4% |
| avx2 | 1024 | 1032 | 1790 | 1648 | -7.9% |
| avx2 | 1024 | 8192 | 1698 | 1736 | +2.2% |
| avx2 | 2048 | 2048 | 3325 | 3357 | +1.0% |
| avx2 | 2048 | 2056 | 3189 | 3346 | +4.9% |
| avx2 | 2048 | 16384 | 3379 | 3510 | +3.9% |
| avx512 | 64 | 64 | 50 | 53 | +5.5% |
| avx512 | 64 | 72 | 50 | 53 | +6.2% |
| avx512 | 64 | 512 | 51 | 53 | +4.2% |
| avx512 | 128 | 128 | 101 | 101 | +0.1% |
| avx512 | 128 | 136 | 103 | 103 | +0.4% |
| avx512 | 128 | 1024 | 100 | 101 | +0.8% |
| avx512 | 256 | 256 | 205 | 199 | -3.0% |
| avx512 | 256 | 264 | 201 | 207 | +3.4% |
| avx512 | 256 | 2048 | 201 | 203 | +0.9% |
| avx512 | 512 | 512 | 407 | 415 | +1.9% |
| avx512 | 512 | 520 | 417 | 428 | +2.8% |
| avx512 | 512 | 4096 | 401 | 461 | +15.0% |
| avx512 | 1024 | 1024 | 1622 | 1622 | -0.0% |
| avx512 | 1024 | 1032 | 1656 | 1670 | +0.9% |
| avx512 | 1024 | 8192 | 1603 | 1610 | +0.4% |
| avx512 | 2048 | 2048 | 3276 | 3337 | +1.9% |
| avx512 | 2048 | 2056 | 3289 | 3191 | -3.0% |
| avx512 | 2048 | 16384 | 3395 | 3318 | -2.2% |