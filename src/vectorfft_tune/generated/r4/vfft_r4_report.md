# VectorFFT R=4 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 31 | 43 | 31 | 36 | t1s |
| avx2 | 64 | 72 | 34 | 42 | 33 | 37 | t1s |
| avx2 | 64 | 512 | 34 | 41 | 32 | 37 | t1s |
| avx2 | 128 | 128 | 64 | 90 | 58 | 76 | t1s |
| avx2 | 128 | 136 | 64 | 88 | 64 | 69 | t1s |
| avx2 | 128 | 1024 | 70 | 86 | 61 | 75 | t1s |
| avx2 | 256 | 256 | 128 | 170 | — | 150 | flat |
| avx2 | 256 | 264 | 124 | 172 | — | 157 | flat |
| avx2 | 256 | 2048 | 137 | 165 | — | 137 | flat |
| avx2 | 512 | 512 | 282 | 355 | — | 294 | flat |
| avx2 | 512 | 520 | 302 | 337 | — | 306 | flat |
| avx2 | 512 | 4096 | 297 | 327 | — | 293 | flat |
| avx2 | 1024 | 1024 | 809 | 757 | — | 785 | log3 |
| avx2 | 1024 | 1032 | 786 | 817 | — | 815 | flat |
| avx2 | 1024 | 8192 | 807 | 785 | — | 820 | log3 |
| avx2 | 2048 | 2048 | 1699 | 1571 | — | 1626 | log3 |
| avx2 | 2048 | 2056 | 1709 | 1719 | — | 1681 | flat |
| avx2 | 2048 | 16384 | 1789 | 1606 | — | 1736 | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 34 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 34 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 64 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit_u2` | 64 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 70 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 128 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 124 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 137 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 282 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 302 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 293 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 785 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 786 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 807 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 1626 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 1681 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 1736 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 42 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 41 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 90 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 88 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 86 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 170 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 172 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 165 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 355 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 337 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 327 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 757 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 817 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 785 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 1571 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 1719 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 1606 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 33 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 32 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 58 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 64 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 61 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 37 | 36 | -2.1% |
| avx2 | 64 | 72 | 37 | 37 | -1.1% |
| avx2 | 64 | 512 | 38 | 37 | -3.0% |
| avx2 | 128 | 128 | 75 | 76 | +1.8% |
| avx2 | 128 | 136 | 77 | 69 | -10.5% |
| avx2 | 128 | 1024 | 73 | 75 | +3.8% |
| avx2 | 256 | 256 | 143 | 150 | +5.0% |
| avx2 | 256 | 264 | 148 | 157 | +5.9% |
| avx2 | 256 | 2048 | 142 | 137 | -3.2% |
| avx2 | 512 | 512 | 282 | 294 | +4.3% |
| avx2 | 512 | 520 | 302 | 306 | +1.3% |
| avx2 | 512 | 4096 | 297 | 293 | -1.6% |
| avx2 | 1024 | 1024 | 809 | 785 | -2.9% |
| avx2 | 1024 | 1032 | 786 | 815 | +3.7% |
| avx2 | 1024 | 8192 | 807 | 820 | +1.6% |
| avx2 | 2048 | 2048 | 1869 | 1626 | -13.0% |
| avx2 | 2048 | 2056 | 1709 | 1681 | -1.6% |
| avx2 | 2048 | 16384 | 1789 | 1736 | -3.0% |