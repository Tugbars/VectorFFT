# VectorFFT R=4 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 31 | 40 | 30 | 34 | t1s |
| avx2 | 64 | 72 | 31 | 40 | 29 | 34 | t1s |
| avx2 | 64 | 512 | 30 | 40 | 29 | 34 | t1s |
| avx2 | 128 | 128 | 67 | 79 | 112 | 68 | flat |
| avx2 | 128 | 136 | 61 | 79 | 58 | 69 | t1s |
| avx2 | 128 | 1024 | 61 | 79 | 112 | 67 | flat |
| avx2 | 256 | 256 | 123 | 162 | — | 136 | flat |
| avx2 | 256 | 264 | 119 | 164 | — | 136 | flat |
| avx2 | 256 | 2048 | 120 | 164 | — | 136 | flat |
| avx2 | 512 | 512 | 273 | 318 | — | 273 | flat |
| avx2 | 512 | 520 | 271 | 317 | — | 274 | flat |
| avx2 | 512 | 4096 | 282 | 317 | — | 275 | flat |
| avx2 | 1024 | 1024 | 760 | 711 | — | 768 | log3 |
| avx2 | 1024 | 1032 | 759 | 733 | — | 772 | log3 |
| avx2 | 1024 | 8192 | 771 | 725 | — | 771 | log3 |
| avx2 | 2048 | 2048 | 1531 | 1490 | — | 1536 | log3 |
| avx2 | 2048 | 2056 | 1525 | 1456 | — | 1524 | log3 |
| avx2 | 2048 | 16384 | 1573 | 1477 | — | 1564 | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit_log1` | 67 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_u2` | 61 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_u2` | 61 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 123 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_u2` | 119 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_u2` | 120 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 273 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 271 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 275 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 760 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 759 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 771 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 1531 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 1524 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 1564 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 79 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 79 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 79 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 162 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 318 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 317 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 317 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 711 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 733 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 725 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 1490 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 1456 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 1477 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 112 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 58 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 112 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 35 | 34 | -2.4% |
| avx2 | 64 | 72 | 34 | 34 | +0.7% |
| avx2 | 64 | 512 | 34 | 34 | +0.1% |
| avx2 | 128 | 128 | 67 | 68 | +1.3% |
| avx2 | 128 | 136 | 67 | 69 | +1.7% |
| avx2 | 128 | 1024 | 68 | 67 | -0.7% |
| avx2 | 256 | 256 | 136 | 136 | -0.6% |
| avx2 | 256 | 264 | 136 | 136 | -0.2% |
| avx2 | 256 | 2048 | 136 | 136 | -0.1% |
| avx2 | 512 | 512 | 273 | 273 | -0.1% |
| avx2 | 512 | 520 | 271 | 274 | +0.9% |
| avx2 | 512 | 4096 | 282 | 275 | -2.7% |
| avx2 | 1024 | 1024 | 760 | 768 | +1.0% |
| avx2 | 1024 | 1032 | 759 | 772 | +1.6% |
| avx2 | 1024 | 8192 | 771 | 771 | +0.0% |
| avx2 | 2048 | 2048 | 1531 | 1536 | +0.3% |
| avx2 | 2048 | 2056 | 1525 | 1524 | -0.1% |
| avx2 | 2048 | 16384 | 1573 | 1564 | -0.6% |