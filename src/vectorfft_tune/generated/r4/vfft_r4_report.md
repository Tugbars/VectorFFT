# VectorFFT R=4 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 31 | 42 | 31 | 35 | flat |
| avx2 | 64 | 72 | 31 | 42 | 31 | 34 | flat |
| avx2 | 64 | 512 | 30 | 43 | 31 | 35 | flat |
| avx2 | 128 | 128 | 63 | 84 | 108 | 73 | flat |
| avx2 | 128 | 136 | 62 | 84 | 62 | 69 | flat |
| avx2 | 128 | 1024 | 65 | 90 | 62 | 68 | t1s |
| avx2 | 256 | 256 | 121 | 172 | — | 155 | flat |
| avx2 | 256 | 264 | 121 | 170 | — | 141 | flat |
| avx2 | 256 | 2048 | 126 | 172 | — | 145 | flat |
| avx2 | 512 | 512 | 289 | 368 | — | 276 | flat |
| avx2 | 512 | 520 | 299 | 336 | — | 279 | flat |
| avx2 | 512 | 4096 | 271 | 348 | — | 341 | flat |
| avx2 | 1024 | 1024 | 824 | 753 | — | 757 | log3 |
| avx2 | 1024 | 1032 | 786 | 797 | — | 815 | flat |
| avx2 | 1024 | 8192 | 842 | 783 | — | 768 | log3 |
| avx2 | 2048 | 2048 | 1655 | 1610 | — | 1531 | log3 |
| avx2 | 2048 | 2056 | 1667 | 1578 | — | 1632 | log3 |
| avx2 | 2048 | 16384 | 1732 | 1640 | — | 1633 | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 63 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_u2` | 62 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 65 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 121 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 121 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 126 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 276 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 279 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 271 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 757 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 786 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 768 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 1531 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 1632 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 1633 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 42 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 42 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 90 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 172 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 170 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 172 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 368 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 336 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 348 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 753 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 797 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 783 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 1610 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 1578 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 1640 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 108 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 62 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 62 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 38 | 35 | -6.8% |
| avx2 | 64 | 72 | 35 | 34 | -3.5% |
| avx2 | 64 | 512 | 37 | 35 | -4.9% |
| avx2 | 128 | 128 | 75 | 73 | -3.0% |
| avx2 | 128 | 136 | 73 | 69 | -4.2% |
| avx2 | 128 | 1024 | 74 | 68 | -8.2% |
| avx2 | 256 | 256 | 140 | 155 | +10.1% |
| avx2 | 256 | 264 | 146 | 141 | -3.6% |
| avx2 | 256 | 2048 | 148 | 145 | -1.8% |
| avx2 | 512 | 512 | 290 | 276 | -5.0% |
| avx2 | 512 | 520 | 299 | 279 | -6.6% |
| avx2 | 512 | 4096 | 271 | 341 | +25.8% |
| avx2 | 1024 | 1024 | 832 | 757 | -9.0% |
| avx2 | 1024 | 1032 | 876 | 815 | -7.1% |
| avx2 | 1024 | 8192 | 842 | 768 | -8.8% |
| avx2 | 2048 | 2048 | 1655 | 1531 | -7.5% |
| avx2 | 2048 | 2056 | 1696 | 1632 | -3.8% |
| avx2 | 2048 | 16384 | 1732 | 1633 | -5.7% |