# VectorFFT R=7 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 56 | 56 | 110 | 135 | 108 | — | t1s |
| avx2 | 56 | 64 | 119 | 139 | 100 | — | t1s |
| avx2 | 56 | 392 | 110 | 133 | 112 | — | flat |
| avx2 | 56 | 448 | 111 | 142 | 109 | — | t1s |
| avx2 | 112 | 112 | 216 | 276 | 198 | — | t1s |
| avx2 | 112 | 120 | 225 | 268 | 207 | — | t1s |
| avx2 | 112 | 784 | 219 | 268 | 218 | — | t1s |
| avx2 | 112 | 896 | 225 | 263 | 219 | — | t1s |
| avx2 | 224 | 224 | 428 | 532 | 405 | — | t1s |
| avx2 | 224 | 232 | 460 | 575 | 409 | — | t1s |
| avx2 | 224 | 1568 | 438 | 541 | 436 | — | t1s |
| avx2 | 224 | 1792 | 470 | 524 | 404 | — | t1s |
| avx2 | 448 | 448 | 1064 | 1060 | 850 | — | t1s |
| avx2 | 448 | 456 | 983 | 1021 | 809 | — | t1s |
| avx2 | 448 | 3136 | 1143 | 1092 | 900 | — | t1s |
| avx2 | 448 | 3584 | 1630 | 1750 | 1580 | — | t1s |
| avx2 | 896 | 896 | 2096 | 2289 | 1779 | — | t1s |
| avx2 | 896 | 904 | 2163 | 2171 | 1620 | — | t1s |
| avx2 | 896 | 6272 | 2169 | 2220 | 1732 | — | t1s |
| avx2 | 896 | 7168 | 3703 | 3220 | 3085 | — | t1s |
| avx2 | 1792 | 1792 | 4529 | 4476 | 3471 | — | t1s |
| avx2 | 1792 | 1800 | 4001 | 4738 | 3645 | — | t1s |
| avx2 | 1792 | 12544 | 4227 | 4313 | 3215 | — | t1s |
| avx2 | 1792 | 14336 | 4378 | 4524 | 3856 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 56 | 56 | `ct_t1_dit` | 110 |
| avx2 | `t1_dit` | 56 | 64 | `ct_t1_dit` | 119 |
| avx2 | `t1_dit` | 56 | 392 | `ct_t1_dit` | 110 |
| avx2 | `t1_dit` | 56 | 448 | `ct_t1_dit` | 111 |
| avx2 | `t1_dit` | 112 | 112 | `ct_t1_dit` | 216 |
| avx2 | `t1_dit` | 112 | 120 | `ct_t1_dit` | 225 |
| avx2 | `t1_dit` | 112 | 784 | `ct_t1_dit` | 219 |
| avx2 | `t1_dit` | 112 | 896 | `ct_t1_dit` | 225 |
| avx2 | `t1_dit` | 224 | 224 | `ct_t1_dit` | 428 |
| avx2 | `t1_dit` | 224 | 232 | `ct_t1_dit` | 460 |
| avx2 | `t1_dit` | 224 | 1568 | `ct_t1_dit` | 438 |
| avx2 | `t1_dit` | 224 | 1792 | `ct_t1_dit` | 470 |
| avx2 | `t1_dit` | 448 | 448 | `ct_t1_dit` | 1064 |
| avx2 | `t1_dit` | 448 | 456 | `ct_t1_dit` | 983 |
| avx2 | `t1_dit` | 448 | 3136 | `ct_t1_dit` | 1143 |
| avx2 | `t1_dit` | 448 | 3584 | `ct_t1_dit` | 1630 |
| avx2 | `t1_dit` | 896 | 896 | `ct_t1_dit` | 2096 |
| avx2 | `t1_dit` | 896 | 904 | `ct_t1_dit` | 2163 |
| avx2 | `t1_dit` | 896 | 6272 | `ct_t1_dit` | 2169 |
| avx2 | `t1_dit` | 896 | 7168 | `ct_t1_dit` | 3703 |
| avx2 | `t1_dit` | 1792 | 1792 | `ct_t1_dit` | 4529 |
| avx2 | `t1_dit` | 1792 | 1800 | `ct_t1_dit` | 4001 |
| avx2 | `t1_dit` | 1792 | 12544 | `ct_t1_dit` | 4227 |
| avx2 | `t1_dit` | 1792 | 14336 | `ct_t1_dit` | 4378 |
| avx2 | `t1_dit_log3` | 56 | 56 | `ct_t1_dit_log3` | 135 |
| avx2 | `t1_dit_log3` | 56 | 64 | `ct_t1_dit_log3` | 139 |
| avx2 | `t1_dit_log3` | 56 | 392 | `ct_t1_dit_log3` | 133 |
| avx2 | `t1_dit_log3` | 56 | 448 | `ct_t1_dit_log3` | 142 |
| avx2 | `t1_dit_log3` | 112 | 112 | `ct_t1_dit_log3` | 276 |
| avx2 | `t1_dit_log3` | 112 | 120 | `ct_t1_dit_log3` | 268 |
| avx2 | `t1_dit_log3` | 112 | 784 | `ct_t1_dit_log3` | 268 |
| avx2 | `t1_dit_log3` | 112 | 896 | `ct_t1_dit_log3` | 263 |
| avx2 | `t1_dit_log3` | 224 | 224 | `ct_t1_dit_log3` | 532 |
| avx2 | `t1_dit_log3` | 224 | 232 | `ct_t1_dit_log3` | 575 |
| avx2 | `t1_dit_log3` | 224 | 1568 | `ct_t1_dit_log3` | 541 |
| avx2 | `t1_dit_log3` | 224 | 1792 | `ct_t1_dit_log3` | 524 |
| avx2 | `t1_dit_log3` | 448 | 448 | `ct_t1_dit_log3` | 1060 |
| avx2 | `t1_dit_log3` | 448 | 456 | `ct_t1_dit_log3` | 1021 |
| avx2 | `t1_dit_log3` | 448 | 3136 | `ct_t1_dit_log3` | 1092 |
| avx2 | `t1_dit_log3` | 448 | 3584 | `ct_t1_dit_log3` | 1750 |
| avx2 | `t1_dit_log3` | 896 | 896 | `ct_t1_dit_log3` | 2289 |
| avx2 | `t1_dit_log3` | 896 | 904 | `ct_t1_dit_log3` | 2171 |
| avx2 | `t1_dit_log3` | 896 | 6272 | `ct_t1_dit_log3` | 2220 |
| avx2 | `t1_dit_log3` | 896 | 7168 | `ct_t1_dit_log3` | 3220 |
| avx2 | `t1_dit_log3` | 1792 | 1792 | `ct_t1_dit_log3` | 4476 |
| avx2 | `t1_dit_log3` | 1792 | 1800 | `ct_t1_dit_log3` | 4738 |
| avx2 | `t1_dit_log3` | 1792 | 12544 | `ct_t1_dit_log3` | 4313 |
| avx2 | `t1_dit_log3` | 1792 | 14336 | `ct_t1_dit_log3` | 4524 |
| avx2 | `t1s_dit` | 56 | 56 | `ct_t1s_dit` | 108 |
| avx2 | `t1s_dit` | 56 | 64 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 56 | 392 | `ct_t1s_dit` | 112 |
| avx2 | `t1s_dit` | 56 | 448 | `ct_t1s_dit` | 109 |
| avx2 | `t1s_dit` | 112 | 112 | `ct_t1s_dit` | 198 |
| avx2 | `t1s_dit` | 112 | 120 | `ct_t1s_dit` | 207 |
| avx2 | `t1s_dit` | 112 | 784 | `ct_t1s_dit` | 218 |
| avx2 | `t1s_dit` | 112 | 896 | `ct_t1s_dit` | 219 |
| avx2 | `t1s_dit` | 224 | 224 | `ct_t1s_dit` | 405 |
| avx2 | `t1s_dit` | 224 | 232 | `ct_t1s_dit` | 409 |
| avx2 | `t1s_dit` | 224 | 1568 | `ct_t1s_dit` | 436 |
| avx2 | `t1s_dit` | 224 | 1792 | `ct_t1s_dit` | 404 |
| avx2 | `t1s_dit` | 448 | 448 | `ct_t1s_dit` | 850 |
| avx2 | `t1s_dit` | 448 | 456 | `ct_t1s_dit` | 809 |
| avx2 | `t1s_dit` | 448 | 3136 | `ct_t1s_dit` | 900 |
| avx2 | `t1s_dit` | 448 | 3584 | `ct_t1s_dit` | 1580 |
| avx2 | `t1s_dit` | 896 | 896 | `ct_t1s_dit` | 1779 |
| avx2 | `t1s_dit` | 896 | 904 | `ct_t1s_dit` | 1620 |
| avx2 | `t1s_dit` | 896 | 6272 | `ct_t1s_dit` | 1732 |
| avx2 | `t1s_dit` | 896 | 7168 | `ct_t1s_dit` | 3085 |
| avx2 | `t1s_dit` | 1792 | 1792 | `ct_t1s_dit` | 3471 |
| avx2 | `t1s_dit` | 1792 | 1800 | `ct_t1s_dit` | 3645 |
| avx2 | `t1s_dit` | 1792 | 12544 | `ct_t1s_dit` | 3215 |
| avx2 | `t1s_dit` | 1792 | 14336 | `ct_t1s_dit` | 3856 |