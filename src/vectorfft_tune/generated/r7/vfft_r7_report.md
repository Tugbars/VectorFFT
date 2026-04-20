# VectorFFT R=7 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 56 | 56 | 104 | 124 | 98 | — | t1s |
| avx2 | 56 | 64 | 104 | 124 | 97 | — | t1s |
| avx2 | 56 | 392 | 104 | 125 | 98 | — | t1s |
| avx2 | 56 | 448 | 104 | 124 | 98 | — | t1s |
| avx2 | 112 | 112 | 207 | 246 | 197 | — | t1s |
| avx2 | 112 | 120 | 207 | 248 | 197 | — | t1s |
| avx2 | 112 | 784 | 208 | 246 | 193 | — | t1s |
| avx2 | 112 | 896 | 211 | 244 | 194 | — | t1s |
| avx2 | 224 | 224 | 407 | 493 | 387 | — | t1s |
| avx2 | 224 | 232 | 408 | 491 | 397 | — | t1s |
| avx2 | 224 | 1568 | 425 | 490 | 394 | — | t1s |
| avx2 | 224 | 1792 | 435 | 493 | 392 | — | t1s |
| avx2 | 448 | 448 | 983 | 1013 | 767 | — | t1s |
| avx2 | 448 | 456 | 999 | 1011 | 780 | — | t1s |
| avx2 | 448 | 3136 | 1000 | 1007 | 769 | — | t1s |
| avx2 | 448 | 3584 | 1806 | 1531 | 1708 | — | log3 |
| avx2 | 896 | 896 | 2060 | 2054 | 1633 | — | t1s |
| avx2 | 896 | 904 | 1924 | 2003 | 1619 | — | t1s |
| avx2 | 896 | 6272 | 2058 | 2056 | 1625 | — | t1s |
| avx2 | 896 | 7168 | 3349 | 3151 | 2742 | — | t1s |
| avx2 | 1792 | 1792 | 3957 | 4057 | 3166 | — | t1s |
| avx2 | 1792 | 1800 | 4134 | 4090 | 3231 | — | t1s |
| avx2 | 1792 | 12544 | 4115 | 4100 | 3413 | — | t1s |
| avx2 | 1792 | 14336 | 3923 | 4089 | 3110 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 56 | 56 | `ct_t1_dit` | 104 |
| avx2 | `t1_dit` | 56 | 64 | `ct_t1_dit` | 104 |
| avx2 | `t1_dit` | 56 | 392 | `ct_t1_dit` | 104 |
| avx2 | `t1_dit` | 56 | 448 | `ct_t1_dit` | 104 |
| avx2 | `t1_dit` | 112 | 112 | `ct_t1_dit` | 207 |
| avx2 | `t1_dit` | 112 | 120 | `ct_t1_dit` | 207 |
| avx2 | `t1_dit` | 112 | 784 | `ct_t1_dit` | 208 |
| avx2 | `t1_dit` | 112 | 896 | `ct_t1_dit` | 211 |
| avx2 | `t1_dit` | 224 | 224 | `ct_t1_dit` | 407 |
| avx2 | `t1_dit` | 224 | 232 | `ct_t1_dit` | 408 |
| avx2 | `t1_dit` | 224 | 1568 | `ct_t1_dit` | 425 |
| avx2 | `t1_dit` | 224 | 1792 | `ct_t1_dit` | 435 |
| avx2 | `t1_dit` | 448 | 448 | `ct_t1_dit` | 983 |
| avx2 | `t1_dit` | 448 | 456 | `ct_t1_dit` | 999 |
| avx2 | `t1_dit` | 448 | 3136 | `ct_t1_dit` | 1000 |
| avx2 | `t1_dit` | 448 | 3584 | `ct_t1_dit` | 1806 |
| avx2 | `t1_dit` | 896 | 896 | `ct_t1_dit` | 2060 |
| avx2 | `t1_dit` | 896 | 904 | `ct_t1_dit` | 1924 |
| avx2 | `t1_dit` | 896 | 6272 | `ct_t1_dit` | 2058 |
| avx2 | `t1_dit` | 896 | 7168 | `ct_t1_dit` | 3349 |
| avx2 | `t1_dit` | 1792 | 1792 | `ct_t1_dit` | 3957 |
| avx2 | `t1_dit` | 1792 | 1800 | `ct_t1_dit` | 4134 |
| avx2 | `t1_dit` | 1792 | 12544 | `ct_t1_dit` | 4115 |
| avx2 | `t1_dit` | 1792 | 14336 | `ct_t1_dit` | 3923 |
| avx2 | `t1_dit_log3` | 56 | 56 | `ct_t1_dit_log3` | 124 |
| avx2 | `t1_dit_log3` | 56 | 64 | `ct_t1_dit_log3` | 124 |
| avx2 | `t1_dit_log3` | 56 | 392 | `ct_t1_dit_log3` | 125 |
| avx2 | `t1_dit_log3` | 56 | 448 | `ct_t1_dit_log3` | 124 |
| avx2 | `t1_dit_log3` | 112 | 112 | `ct_t1_dit_log3` | 246 |
| avx2 | `t1_dit_log3` | 112 | 120 | `ct_t1_dit_log3` | 248 |
| avx2 | `t1_dit_log3` | 112 | 784 | `ct_t1_dit_log3` | 246 |
| avx2 | `t1_dit_log3` | 112 | 896 | `ct_t1_dit_log3` | 244 |
| avx2 | `t1_dit_log3` | 224 | 224 | `ct_t1_dit_log3` | 493 |
| avx2 | `t1_dit_log3` | 224 | 232 | `ct_t1_dit_log3` | 491 |
| avx2 | `t1_dit_log3` | 224 | 1568 | `ct_t1_dit_log3` | 490 |
| avx2 | `t1_dit_log3` | 224 | 1792 | `ct_t1_dit_log3` | 493 |
| avx2 | `t1_dit_log3` | 448 | 448 | `ct_t1_dit_log3` | 1013 |
| avx2 | `t1_dit_log3` | 448 | 456 | `ct_t1_dit_log3` | 1011 |
| avx2 | `t1_dit_log3` | 448 | 3136 | `ct_t1_dit_log3` | 1007 |
| avx2 | `t1_dit_log3` | 448 | 3584 | `ct_t1_dit_log3` | 1531 |
| avx2 | `t1_dit_log3` | 896 | 896 | `ct_t1_dit_log3` | 2054 |
| avx2 | `t1_dit_log3` | 896 | 904 | `ct_t1_dit_log3` | 2003 |
| avx2 | `t1_dit_log3` | 896 | 6272 | `ct_t1_dit_log3` | 2056 |
| avx2 | `t1_dit_log3` | 896 | 7168 | `ct_t1_dit_log3` | 3151 |
| avx2 | `t1_dit_log3` | 1792 | 1792 | `ct_t1_dit_log3` | 4057 |
| avx2 | `t1_dit_log3` | 1792 | 1800 | `ct_t1_dit_log3` | 4090 |
| avx2 | `t1_dit_log3` | 1792 | 12544 | `ct_t1_dit_log3` | 4100 |
| avx2 | `t1_dit_log3` | 1792 | 14336 | `ct_t1_dit_log3` | 4089 |
| avx2 | `t1s_dit` | 56 | 56 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 56 | 64 | `ct_t1s_dit` | 97 |
| avx2 | `t1s_dit` | 56 | 392 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 56 | 448 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 112 | 112 | `ct_t1s_dit` | 197 |
| avx2 | `t1s_dit` | 112 | 120 | `ct_t1s_dit` | 197 |
| avx2 | `t1s_dit` | 112 | 784 | `ct_t1s_dit` | 193 |
| avx2 | `t1s_dit` | 112 | 896 | `ct_t1s_dit` | 194 |
| avx2 | `t1s_dit` | 224 | 224 | `ct_t1s_dit` | 387 |
| avx2 | `t1s_dit` | 224 | 232 | `ct_t1s_dit` | 397 |
| avx2 | `t1s_dit` | 224 | 1568 | `ct_t1s_dit` | 394 |
| avx2 | `t1s_dit` | 224 | 1792 | `ct_t1s_dit` | 392 |
| avx2 | `t1s_dit` | 448 | 448 | `ct_t1s_dit` | 767 |
| avx2 | `t1s_dit` | 448 | 456 | `ct_t1s_dit` | 780 |
| avx2 | `t1s_dit` | 448 | 3136 | `ct_t1s_dit` | 769 |
| avx2 | `t1s_dit` | 448 | 3584 | `ct_t1s_dit` | 1708 |
| avx2 | `t1s_dit` | 896 | 896 | `ct_t1s_dit` | 1633 |
| avx2 | `t1s_dit` | 896 | 904 | `ct_t1s_dit` | 1619 |
| avx2 | `t1s_dit` | 896 | 6272 | `ct_t1s_dit` | 1625 |
| avx2 | `t1s_dit` | 896 | 7168 | `ct_t1s_dit` | 2742 |
| avx2 | `t1s_dit` | 1792 | 1792 | `ct_t1s_dit` | 3166 |
| avx2 | `t1s_dit` | 1792 | 1800 | `ct_t1s_dit` | 3231 |
| avx2 | `t1s_dit` | 1792 | 12544 | `ct_t1s_dit` | 3413 |
| avx2 | `t1s_dit` | 1792 | 14336 | `ct_t1s_dit` | 3110 |