# VectorFFT R=4 tuning report

Total measurements: **348**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 29 | 43 | 30 | 36 | flat |
| avx2 | 64 | 72 | 32 | 40 | 30 | 36 | t1s |
| avx2 | 64 | 512 | 30 | 40 | 30 | 36 | t1s |
| avx2 | 96 | 96 | 46 | 62 | 46 | 52 | t1s |
| avx2 | 96 | 104 | 47 | 62 | 44 | 52 | t1s |
| avx2 | 96 | 768 | 46 | 61 | 44 | 54 | t1s |
| avx2 | 128 | 128 | 61 | 80 | 58 | 68 | t1s |
| avx2 | 128 | 136 | 61 | 81 | 58 | 70 | t1s |
| avx2 | 128 | 1024 | 62 | 80 | 58 | 71 | t1s |
| avx2 | 192 | 192 | 89 | 125 | — | 105 | flat |
| avx2 | 192 | 200 | 91 | 122 | — | 101 | flat |
| avx2 | 192 | 1536 | 90 | 123 | — | 106 | flat |
| avx2 | 256 | 256 | 119 | 165 | — | 134 | flat |
| avx2 | 256 | 264 | 121 | 165 | — | 136 | flat |
| avx2 | 256 | 2048 | 120 | 158 | — | 137 | flat |
| avx2 | 384 | 384 | 181 | 252 | — | 205 | flat |
| avx2 | 384 | 392 | 179 | 241 | — | 205 | flat |
| avx2 | 384 | 3072 | 182 | 240 | — | 208 | flat |
| avx2 | 512 | 512 | 287 | 319 | — | 279 | flat |
| avx2 | 512 | 520 | 311 | 325 | — | 279 | flat |
| avx2 | 512 | 4096 | 290 | 325 | — | 281 | flat |
| avx2 | 768 | 768 | 583 | 531 | — | 582 | log3 |
| avx2 | 768 | 776 | 591 | 573 | — | 576 | log3 |
| avx2 | 768 | 6144 | 536 | 522 | — | 522 | log3 |
| avx2 | 1024 | 1024 | 808 | 772 | — | 802 | log3 |
| avx2 | 1024 | 1032 | 822 | 745 | — | 792 | log3 |
| avx2 | 1024 | 8192 | 800 | 758 | — | 787 | log3 |
| avx2 | 1536 | 1536 | 1153 | 1162 | — | 1158 | flat |
| avx2 | 1536 | 1544 | 1145 | 1165 | — | 1145 | flat |
| avx2 | 1536 | 12288 | 1157 | 1127 | — | 1187 | log3 |
| avx2 | 2048 | 2048 | 1639 | 1559 | — | 1550 | log3 |
| avx2 | 2048 | 2056 | 1570 | 1521 | — | 1611 | log3 |
| avx2 | 2048 | 16384 | 1575 | 1523 | — | 1751 | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 29 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 32 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit` | 47 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit_u2` | 61 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 61 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_u2` | 62 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit` | 89 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit` | 91 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit` | 90 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_u2` | 119 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 121 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_u2` | 120 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit` | 181 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit` | 179 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit` | 182 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 279 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 279 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 281 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit_log1` | 582 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit_log1` | 576 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit_log1` | 522 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 808 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 792 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 787 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_log1` | 1153 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_log1` | 1145 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_log1` | 1157 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 1550 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 1570 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 1575 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3` | 61 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 80 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 81 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 80 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3` | 125 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3` | 122 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3` | 123 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 165 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 165 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 158 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3` | 252 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3` | 241 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3` | 240 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 319 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 325 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 325 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3` | 531 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3` | 573 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3` | 522 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 772 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 745 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 758 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3` | 1162 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3` | 1165 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3` | 1127 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 1559 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 1521 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 1523 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit` | 46 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit` | 44 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit` | 44 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 58 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 58 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 58 |

## log1 vs log1_tight (handicap experiment)

Tests whether the log1 variant in flat protocol is handicapped
by a full (R-1)*me twiddle table vs a tight 2*me table. Same
codelet body; different harness allocation.

| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |
|---|---|---|---|---|---|
| avx2 | 64 | 64 | 36 | 36 | +1.0% |
| avx2 | 64 | 72 | 34 | 36 | +5.2% |
| avx2 | 64 | 512 | 37 | 36 | -2.7% |
| avx2 | 96 | 96 | 51 | 52 | +2.2% |
| avx2 | 96 | 104 | 53 | 52 | -3.2% |
| avx2 | 96 | 768 | 51 | 54 | +5.8% |
| avx2 | 128 | 128 | 69 | 68 | -1.9% |
| avx2 | 128 | 136 | 70 | 70 | -0.1% |
| avx2 | 128 | 1024 | 68 | 71 | +4.6% |
| avx2 | 192 | 192 | 103 | 105 | +2.6% |
| avx2 | 192 | 200 | 104 | 101 | -2.8% |
| avx2 | 192 | 1536 | 102 | 106 | +4.4% |
| avx2 | 256 | 256 | 135 | 134 | -0.8% |
| avx2 | 256 | 264 | 136 | 136 | +0.3% |
| avx2 | 256 | 2048 | 140 | 137 | -2.3% |
| avx2 | 384 | 384 | 203 | 205 | +1.0% |
| avx2 | 384 | 392 | 209 | 205 | -2.1% |
| avx2 | 384 | 3072 | 210 | 208 | -1.0% |
| avx2 | 512 | 512 | 287 | 279 | -2.5% |
| avx2 | 512 | 520 | 436 | 279 | -36.0% |
| avx2 | 512 | 4096 | 290 | 281 | -3.1% |
| avx2 | 768 | 768 | 583 | 582 | -0.2% |
| avx2 | 768 | 776 | 587 | 576 | -1.9% |
| avx2 | 768 | 6144 | 536 | 522 | -2.6% |
| avx2 | 1024 | 1024 | 795 | 802 | +0.9% |
| avx2 | 1024 | 1032 | 808 | 792 | -2.0% |
| avx2 | 1024 | 8192 | 800 | 787 | -1.6% |
| avx2 | 1536 | 1536 | 1153 | 1158 | +0.4% |
| avx2 | 1536 | 1544 | 1145 | 1145 | -0.0% |
| avx2 | 1536 | 12288 | 1157 | 1187 | +2.6% |
| avx2 | 2048 | 2048 | 1610 | 1550 | -3.7% |
| avx2 | 2048 | 2056 | 1633 | 1611 | -1.4% |
| avx2 | 2048 | 16384 | 1575 | 1751 | +11.2% |