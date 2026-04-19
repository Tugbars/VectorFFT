# VectorFFT R=4 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point.

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 29 | 38 | 28 | 33 | t1s |
| avx2 | 64 | 72 | 29 | 41 | 28 | 32 | t1s |
| avx2 | 64 | 512 | 29 | 40 | 29 | 33 | flat |
| avx2 | 128 | 128 | 58 | 77 | 57 | 107 | t1s |
| avx2 | 128 | 136 | 58 | 78 | 57 | 66 | t1s |
| avx2 | 128 | 1024 | 56 | 80 | 57 | 66 | flat |
| avx2 | 256 | 256 | 112 | 158 | — | 133 | flat |
| avx2 | 256 | 264 | 111 | 160 | — | 134 | flat |
| avx2 | 256 | 2048 | 113 | 154 | — | 137 | flat |
| avx2 | 512 | 512 | 263 | 313 | — | 263 | flat |
| avx2 | 512 | 520 | 296 | 302 | — | 262 | flat |
| avx2 | 512 | 4096 | 262 | 302 | — | 267 | flat |
| avx2 | 1024 | 1024 | 723 | 674 | — | 721 | log3 |
| avx2 | 1024 | 1032 | 762 | 712 | — | 726 | log3 |
| avx2 | 1024 | 8192 | 735 | 702 | — | 747 | log3 |
| avx2 | 2048 | 2048 | 1520 | 1418 | — | 1469 | log3 |
| avx2 | 2048 | 2056 | 1459 | 1391 | — | 1481 | log3 |
| avx2 | 2048 | 16384 | 1499 | 1482 | — | 1575 | log3 |

## Flat-protocol variant winners (fwd)

Within the flat protocol, which variant (dit/u2/log1) wins each point?

| isa | me | ios | winner | ns |
|---|---|---|---|---|
| avx2 | 64 | 64 | `ct_t1_dit` | 29 |
| avx2 | 64 | 72 | `ct_t1_dit` | 29 |
| avx2 | 64 | 512 | `ct_t1_dit` | 29 |
| avx2 | 128 | 128 | `ct_t1_dit` | 58 |
| avx2 | 128 | 136 | `ct_t1_dit` | 58 |
| avx2 | 128 | 1024 | `ct_t1_dit` | 56 |
| avx2 | 256 | 256 | `ct_t1_dit` | 112 |
| avx2 | 256 | 264 | `ct_t1_dit` | 111 |
| avx2 | 256 | 2048 | `ct_t1_dit_u2` | 113 |
| avx2 | 512 | 512 | `ct_t1_dit` | 263 |
| avx2 | 512 | 520 | `ct_t1_dit` | 296 |
| avx2 | 512 | 4096 | `ct_t1_dit_log1` | 262 |
| avx2 | 1024 | 1024 | `ct_t1_dit_log1` | 723 |
| avx2 | 1024 | 1032 | `ct_t1_dit` | 762 |
| avx2 | 1024 | 8192 | `ct_t1_dit_log1` | 735 |
| avx2 | 2048 | 2048 | `ct_t1_dit` | 1520 |
| avx2 | 2048 | 2056 | `ct_t1_dit_log1` | 1459 |
| avx2 | 2048 | 16384 | `ct_t1_dit_log1` | 1499 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped by a
full (R-1)*me twiddle table vs a tight 2*me table. Same codelet body;
different harness allocation. Expected: within 2% at all points on a
chip where memory footprint isnt the bottleneck.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 33 | 33 | -0.1% |
| avx2 | 64 | 72 | 33 | 32 | -1.1% |
| avx2 | 64 | 512 | 32 | 33 | +1.3% |
| avx2 | 128 | 128 | 65 | 107 | +63.3% |
| avx2 | 128 | 136 | 67 | 66 | -1.2% |
| avx2 | 128 | 1024 | 66 | 66 | -0.6% |
| avx2 | 256 | 256 | 131 | 133 | +1.9% |
| avx2 | 256 | 264 | 131 | 134 | +1.9% |
| avx2 | 256 | 2048 | 130 | 137 | +5.5% |
| avx2 | 512 | 512 | 260 | 263 | +1.1% |
| avx2 | 512 | 520 | 345 | 262 | -24.1% |
| avx2 | 512 | 4096 | 262 | 267 | +2.0% |
| avx2 | 1024 | 1024 | 723 | 721 | -0.2% |
| avx2 | 1024 | 1032 | 750 | 726 | -3.2% |
| avx2 | 1024 | 8192 | 735 | 747 | +1.6% |
| avx2 | 2048 | 2048 | 1496 | 1469 | -1.8% |
| avx2 | 2048 | 2056 | 1459 | 1481 | +1.5% |
| avx2 | 2048 | 16384 | 1499 | 1575 | +5.0% |