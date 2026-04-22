# VectorFFT R=7 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 56 | 56 | 109 | 127 | 102 | — | t1s |
| avx2 | 56 | 64 | 111 | 128 | 110 | — | t1s |
| avx2 | 56 | 392 | 112 | 144 | 101 | — | t1s |
| avx2 | 56 | 448 | 108 | 130 | 103 | — | t1s |
| avx2 | 112 | 112 | 223 | 248 | 199 | — | t1s |
| avx2 | 112 | 120 | 215 | 247 | 203 | — | t1s |
| avx2 | 112 | 784 | 215 | 254 | 207 | — | t1s |
| avx2 | 112 | 896 | 214 | 256 | 195 | — | t1s |
| avx2 | 224 | 224 | 437 | 521 | 412 | — | t1s |
| avx2 | 224 | 232 | 435 | 485 | 419 | — | t1s |
| avx2 | 224 | 1568 | 466 | 509 | 453 | — | t1s |
| avx2 | 224 | 1792 | 448 | 498 | 418 | — | t1s |
| avx2 | 448 | 448 | 1017 | 1002 | 849 | — | t1s |
| avx2 | 448 | 456 | 1029 | 1031 | 864 | — | t1s |
| avx2 | 448 | 3136 | 1049 | 1054 | 878 | — | t1s |
| avx2 | 448 | 3584 | 1581 | 1542 | 1716 | — | log3 |
| avx2 | 896 | 896 | 2052 | 2005 | 1691 | — | t1s |
| avx2 | 896 | 904 | 1892 | 2169 | 1752 | — | t1s |
| avx2 | 896 | 6272 | 2031 | 2111 | 1701 | — | t1s |
| avx2 | 896 | 7168 | 3223 | 2994 | 3207 | — | log3 |
| avx2 | 1792 | 1792 | 4047 | 4349 | 3375 | — | t1s |
| avx2 | 1792 | 1800 | 4157 | 4224 | 3246 | — | t1s |
| avx2 | 1792 | 12544 | 3941 | 4385 | 3460 | — | t1s |
| avx2 | 1792 | 14336 | 4654 | 4547 | 3211 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 56 | 56 | `ct_t1_dit` | 109 |
| avx2 | `t1_dit` | 56 | 64 | `ct_t1_dit` | 111 |
| avx2 | `t1_dit` | 56 | 392 | `ct_t1_dit` | 112 |
| avx2 | `t1_dit` | 56 | 448 | `ct_t1_dit` | 108 |
| avx2 | `t1_dit` | 112 | 112 | `ct_t1_dit` | 223 |
| avx2 | `t1_dit` | 112 | 120 | `ct_t1_dit` | 215 |
| avx2 | `t1_dit` | 112 | 784 | `ct_t1_dit` | 215 |
| avx2 | `t1_dit` | 112 | 896 | `ct_t1_dit` | 214 |
| avx2 | `t1_dit` | 224 | 224 | `ct_t1_dit` | 437 |
| avx2 | `t1_dit` | 224 | 232 | `ct_t1_dit` | 435 |
| avx2 | `t1_dit` | 224 | 1568 | `ct_t1_dit` | 466 |
| avx2 | `t1_dit` | 224 | 1792 | `ct_t1_dit` | 448 |
| avx2 | `t1_dit` | 448 | 448 | `ct_t1_dit` | 1017 |
| avx2 | `t1_dit` | 448 | 456 | `ct_t1_dit` | 1029 |
| avx2 | `t1_dit` | 448 | 3136 | `ct_t1_dit` | 1049 |
| avx2 | `t1_dit` | 448 | 3584 | `ct_t1_dit` | 1581 |
| avx2 | `t1_dit` | 896 | 896 | `ct_t1_dit` | 2052 |
| avx2 | `t1_dit` | 896 | 904 | `ct_t1_dit` | 1892 |
| avx2 | `t1_dit` | 896 | 6272 | `ct_t1_dit` | 2031 |
| avx2 | `t1_dit` | 896 | 7168 | `ct_t1_dit` | 3223 |
| avx2 | `t1_dit` | 1792 | 1792 | `ct_t1_dit` | 4047 |
| avx2 | `t1_dit` | 1792 | 1800 | `ct_t1_dit` | 4157 |
| avx2 | `t1_dit` | 1792 | 12544 | `ct_t1_dit` | 3941 |
| avx2 | `t1_dit` | 1792 | 14336 | `ct_t1_dit` | 4654 |
| avx2 | `t1_dit_log3` | 56 | 56 | `ct_t1_dit_log3` | 127 |
| avx2 | `t1_dit_log3` | 56 | 64 | `ct_t1_dit_log3` | 128 |
| avx2 | `t1_dit_log3` | 56 | 392 | `ct_t1_dit_log3` | 144 |
| avx2 | `t1_dit_log3` | 56 | 448 | `ct_t1_dit_log3` | 130 |
| avx2 | `t1_dit_log3` | 112 | 112 | `ct_t1_dit_log3` | 248 |
| avx2 | `t1_dit_log3` | 112 | 120 | `ct_t1_dit_log3` | 247 |
| avx2 | `t1_dit_log3` | 112 | 784 | `ct_t1_dit_log3` | 254 |
| avx2 | `t1_dit_log3` | 112 | 896 | `ct_t1_dit_log3` | 256 |
| avx2 | `t1_dit_log3` | 224 | 224 | `ct_t1_dit_log3` | 521 |
| avx2 | `t1_dit_log3` | 224 | 232 | `ct_t1_dit_log3` | 485 |
| avx2 | `t1_dit_log3` | 224 | 1568 | `ct_t1_dit_log3` | 509 |
| avx2 | `t1_dit_log3` | 224 | 1792 | `ct_t1_dit_log3` | 498 |
| avx2 | `t1_dit_log3` | 448 | 448 | `ct_t1_dit_log3` | 1002 |
| avx2 | `t1_dit_log3` | 448 | 456 | `ct_t1_dit_log3` | 1031 |
| avx2 | `t1_dit_log3` | 448 | 3136 | `ct_t1_dit_log3` | 1054 |
| avx2 | `t1_dit_log3` | 448 | 3584 | `ct_t1_dit_log3` | 1542 |
| avx2 | `t1_dit_log3` | 896 | 896 | `ct_t1_dit_log3` | 2005 |
| avx2 | `t1_dit_log3` | 896 | 904 | `ct_t1_dit_log3` | 2169 |
| avx2 | `t1_dit_log3` | 896 | 6272 | `ct_t1_dit_log3` | 2111 |
| avx2 | `t1_dit_log3` | 896 | 7168 | `ct_t1_dit_log3` | 2994 |
| avx2 | `t1_dit_log3` | 1792 | 1792 | `ct_t1_dit_log3` | 4349 |
| avx2 | `t1_dit_log3` | 1792 | 1800 | `ct_t1_dit_log3` | 4224 |
| avx2 | `t1_dit_log3` | 1792 | 12544 | `ct_t1_dit_log3` | 4385 |
| avx2 | `t1_dit_log3` | 1792 | 14336 | `ct_t1_dit_log3` | 4547 |
| avx2 | `t1s_dit` | 56 | 56 | `ct_t1s_dit` | 102 |
| avx2 | `t1s_dit` | 56 | 64 | `ct_t1s_dit` | 110 |
| avx2 | `t1s_dit` | 56 | 392 | `ct_t1s_dit` | 101 |
| avx2 | `t1s_dit` | 56 | 448 | `ct_t1s_dit` | 103 |
| avx2 | `t1s_dit` | 112 | 112 | `ct_t1s_dit` | 199 |
| avx2 | `t1s_dit` | 112 | 120 | `ct_t1s_dit` | 203 |
| avx2 | `t1s_dit` | 112 | 784 | `ct_t1s_dit` | 207 |
| avx2 | `t1s_dit` | 112 | 896 | `ct_t1s_dit` | 195 |
| avx2 | `t1s_dit` | 224 | 224 | `ct_t1s_dit` | 412 |
| avx2 | `t1s_dit` | 224 | 232 | `ct_t1s_dit` | 419 |
| avx2 | `t1s_dit` | 224 | 1568 | `ct_t1s_dit` | 453 |
| avx2 | `t1s_dit` | 224 | 1792 | `ct_t1s_dit` | 418 |
| avx2 | `t1s_dit` | 448 | 448 | `ct_t1s_dit` | 849 |
| avx2 | `t1s_dit` | 448 | 456 | `ct_t1s_dit` | 864 |
| avx2 | `t1s_dit` | 448 | 3136 | `ct_t1s_dit` | 878 |
| avx2 | `t1s_dit` | 448 | 3584 | `ct_t1s_dit` | 1716 |
| avx2 | `t1s_dit` | 896 | 896 | `ct_t1s_dit` | 1691 |
| avx2 | `t1s_dit` | 896 | 904 | `ct_t1s_dit` | 1752 |
| avx2 | `t1s_dit` | 896 | 6272 | `ct_t1s_dit` | 1701 |
| avx2 | `t1s_dit` | 896 | 7168 | `ct_t1s_dit` | 3207 |
| avx2 | `t1s_dit` | 1792 | 1792 | `ct_t1s_dit` | 3375 |
| avx2 | `t1s_dit` | 1792 | 1800 | `ct_t1s_dit` | 3246 |
| avx2 | `t1s_dit` | 1792 | 12544 | `ct_t1s_dit` | 3460 |
| avx2 | `t1s_dit` | 1792 | 14336 | `ct_t1s_dit` | 3211 |