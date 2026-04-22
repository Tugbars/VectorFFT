# VectorFFT R=13 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 52 | 57 | 53 | — | flat |
| avx2 | 8 | 16 | 50 | 58 | 54 | — | flat |
| avx2 | 8 | 104 | 52 | 58 | 72 | — | flat |
| avx2 | 8 | 256 | 52 | 59 | 77 | — | flat |
| avx2 | 16 | 16 | 90 | 104 | 116 | — | flat |
| avx2 | 16 | 24 | 95 | 106 | 129 | — | flat |
| avx2 | 16 | 208 | 104 | 109 | 126 | — | flat |
| avx2 | 16 | 512 | 243 | 220 | 225 | — | log3 |
| avx2 | 32 | 32 | 178 | 207 | 180 | — | flat |
| avx2 | 32 | 40 | 184 | 203 | 177 | — | t1s |
| avx2 | 32 | 416 | 199 | 207 | 185 | — | t1s |
| avx2 | 32 | 1024 | 451 | 418 | 445 | — | log3 |
| avx2 | 64 | 64 | 379 | 392 | 417 | — | flat |
| avx2 | 64 | 72 | 356 | 400 | 337 | — | t1s |
| avx2 | 64 | 832 | 407 | 408 | 376 | — | t1s |
| avx2 | 64 | 2048 | 1019 | 850 | 964 | — | log3 |
| avx2 | 128 | 128 | 914 | 778 | 688 | — | t1s |
| avx2 | 128 | 136 | 903 | 800 | 773 | — | t1s |
| avx2 | 128 | 1664 | 900 | 818 | 1127 | — | log3 |
| avx2 | 128 | 4096 | 2176 | 1690 | 2148 | — | log3 |
| avx2 | 256 | 256 | 2926 | 1581 | 1740 | — | log3 |
| avx2 | 256 | 264 | 2023 | 1673 | 1391 | — | t1s |
| avx2 | 256 | 3328 | 2761 | 2036 | 2109 | — | log3 |
| avx2 | 256 | 8192 | 4127 | 3797 | 3981 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 52 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 50 |
| avx2 | `t1_dit` | 8 | 104 | `ct_t1_dit` | 52 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 52 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 90 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 95 |
| avx2 | `t1_dit` | 16 | 208 | `ct_t1_dit` | 104 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 243 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 178 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 184 |
| avx2 | `t1_dit` | 32 | 416 | `ct_t1_dit` | 199 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 451 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 379 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 356 |
| avx2 | `t1_dit` | 64 | 832 | `ct_t1_dit` | 407 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1019 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 914 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 903 |
| avx2 | `t1_dit` | 128 | 1664 | `ct_t1_dit` | 900 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2176 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 2926 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2023 |
| avx2 | `t1_dit` | 256 | 3328 | `ct_t1_dit` | 2761 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 4127 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 57 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 58 |
| avx2 | `t1_dit_log3` | 8 | 104 | `ct_t1_dit_log3` | 58 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 59 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 104 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 16 | 208 | `ct_t1_dit_log3` | 109 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 220 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 207 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 203 |
| avx2 | `t1_dit_log3` | 32 | 416 | `ct_t1_dit_log3` | 207 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 418 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 392 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 400 |
| avx2 | `t1_dit_log3` | 64 | 832 | `ct_t1_dit_log3` | 408 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 850 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 778 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 800 |
| avx2 | `t1_dit_log3` | 128 | 1664 | `ct_t1_dit_log3` | 818 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1690 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1581 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1673 |
| avx2 | `t1_dit_log3` | 256 | 3328 | `ct_t1_dit_log3` | 2036 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 3797 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 53 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 54 |
| avx2 | `t1s_dit` | 8 | 104 | `ct_t1s_dit` | 72 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 77 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 116 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 129 |
| avx2 | `t1s_dit` | 16 | 208 | `ct_t1s_dit` | 126 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 225 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 180 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 177 |
| avx2 | `t1s_dit` | 32 | 416 | `ct_t1s_dit` | 185 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 445 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 417 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 337 |
| avx2 | `t1s_dit` | 64 | 832 | `ct_t1s_dit` | 376 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 964 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 688 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 773 |
| avx2 | `t1s_dit` | 128 | 1664 | `ct_t1s_dit` | 1127 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2148 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1740 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1391 |
| avx2 | `t1s_dit` | 256 | 3328 | `ct_t1s_dit` | 2109 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 3981 |