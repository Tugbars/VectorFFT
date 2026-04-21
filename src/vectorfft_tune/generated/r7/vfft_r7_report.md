# VectorFFT R=7 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 56 | 56 | 110 | 137 | 106 | — | t1s |
| avx2 | 56 | 64 | 114 | 133 | 105 | — | t1s |
| avx2 | 56 | 392 | 115 | 138 | 111 | — | t1s |
| avx2 | 56 | 448 | 106 | 134 | 111 | — | flat |
| avx2 | 112 | 112 | 233 | 266 | 199 | — | t1s |
| avx2 | 112 | 120 | 227 | 269 | 196 | — | t1s |
| avx2 | 112 | 784 | 214 | 282 | 224 | — | flat |
| avx2 | 112 | 896 | 215 | 256 | 222 | — | flat |
| avx2 | 224 | 224 | 443 | 557 | 400 | — | t1s |
| avx2 | 224 | 232 | 444 | 549 | 408 | — | t1s |
| avx2 | 224 | 1568 | 421 | 523 | 429 | — | flat |
| avx2 | 224 | 1792 | 447 | 545 | 425 | — | t1s |
| avx2 | 448 | 448 | 1060 | 1128 | 819 | — | t1s |
| avx2 | 448 | 456 | 1082 | 1058 | 868 | — | t1s |
| avx2 | 448 | 3136 | 1026 | 1092 | 915 | — | t1s |
| avx2 | 448 | 3584 | 1673 | 1767 | 1574 | — | t1s |
| avx2 | 896 | 896 | 2130 | 2219 | 1782 | — | t1s |
| avx2 | 896 | 904 | 1898 | 2134 | 1722 | — | t1s |
| avx2 | 896 | 6272 | 2086 | 2228 | 1755 | — | t1s |
| avx2 | 896 | 7168 | 3342 | 3250 | 3029 | — | t1s |
| avx2 | 1792 | 1792 | 4198 | 4460 | 3584 | — | t1s |
| avx2 | 1792 | 1800 | 4211 | 4631 | 3358 | — | t1s |
| avx2 | 1792 | 12544 | 4164 | 4620 | 3380 | — | t1s |
| avx2 | 1792 | 14336 | 4341 | 4267 | 3286 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 56 | 56 | `ct_t1_dit` | 110 |
| avx2 | `t1_dit` | 56 | 64 | `ct_t1_dit` | 114 |
| avx2 | `t1_dit` | 56 | 392 | `ct_t1_dit` | 115 |
| avx2 | `t1_dit` | 56 | 448 | `ct_t1_dit` | 106 |
| avx2 | `t1_dit` | 112 | 112 | `ct_t1_dit` | 233 |
| avx2 | `t1_dit` | 112 | 120 | `ct_t1_dit` | 227 |
| avx2 | `t1_dit` | 112 | 784 | `ct_t1_dit` | 214 |
| avx2 | `t1_dit` | 112 | 896 | `ct_t1_dit` | 215 |
| avx2 | `t1_dit` | 224 | 224 | `ct_t1_dit` | 443 |
| avx2 | `t1_dit` | 224 | 232 | `ct_t1_dit` | 444 |
| avx2 | `t1_dit` | 224 | 1568 | `ct_t1_dit` | 421 |
| avx2 | `t1_dit` | 224 | 1792 | `ct_t1_dit` | 447 |
| avx2 | `t1_dit` | 448 | 448 | `ct_t1_dit` | 1060 |
| avx2 | `t1_dit` | 448 | 456 | `ct_t1_dit` | 1082 |
| avx2 | `t1_dit` | 448 | 3136 | `ct_t1_dit` | 1026 |
| avx2 | `t1_dit` | 448 | 3584 | `ct_t1_dit` | 1673 |
| avx2 | `t1_dit` | 896 | 896 | `ct_t1_dit` | 2130 |
| avx2 | `t1_dit` | 896 | 904 | `ct_t1_dit` | 1898 |
| avx2 | `t1_dit` | 896 | 6272 | `ct_t1_dit` | 2086 |
| avx2 | `t1_dit` | 896 | 7168 | `ct_t1_dit` | 3342 |
| avx2 | `t1_dit` | 1792 | 1792 | `ct_t1_dit` | 4198 |
| avx2 | `t1_dit` | 1792 | 1800 | `ct_t1_dit` | 4211 |
| avx2 | `t1_dit` | 1792 | 12544 | `ct_t1_dit` | 4164 |
| avx2 | `t1_dit` | 1792 | 14336 | `ct_t1_dit` | 4341 |
| avx2 | `t1_dit_log3` | 56 | 56 | `ct_t1_dit_log3` | 137 |
| avx2 | `t1_dit_log3` | 56 | 64 | `ct_t1_dit_log3` | 133 |
| avx2 | `t1_dit_log3` | 56 | 392 | `ct_t1_dit_log3` | 138 |
| avx2 | `t1_dit_log3` | 56 | 448 | `ct_t1_dit_log3` | 134 |
| avx2 | `t1_dit_log3` | 112 | 112 | `ct_t1_dit_log3` | 266 |
| avx2 | `t1_dit_log3` | 112 | 120 | `ct_t1_dit_log3` | 269 |
| avx2 | `t1_dit_log3` | 112 | 784 | `ct_t1_dit_log3` | 282 |
| avx2 | `t1_dit_log3` | 112 | 896 | `ct_t1_dit_log3` | 256 |
| avx2 | `t1_dit_log3` | 224 | 224 | `ct_t1_dit_log3` | 557 |
| avx2 | `t1_dit_log3` | 224 | 232 | `ct_t1_dit_log3` | 549 |
| avx2 | `t1_dit_log3` | 224 | 1568 | `ct_t1_dit_log3` | 523 |
| avx2 | `t1_dit_log3` | 224 | 1792 | `ct_t1_dit_log3` | 545 |
| avx2 | `t1_dit_log3` | 448 | 448 | `ct_t1_dit_log3` | 1128 |
| avx2 | `t1_dit_log3` | 448 | 456 | `ct_t1_dit_log3` | 1058 |
| avx2 | `t1_dit_log3` | 448 | 3136 | `ct_t1_dit_log3` | 1092 |
| avx2 | `t1_dit_log3` | 448 | 3584 | `ct_t1_dit_log3` | 1767 |
| avx2 | `t1_dit_log3` | 896 | 896 | `ct_t1_dit_log3` | 2219 |
| avx2 | `t1_dit_log3` | 896 | 904 | `ct_t1_dit_log3` | 2134 |
| avx2 | `t1_dit_log3` | 896 | 6272 | `ct_t1_dit_log3` | 2228 |
| avx2 | `t1_dit_log3` | 896 | 7168 | `ct_t1_dit_log3` | 3250 |
| avx2 | `t1_dit_log3` | 1792 | 1792 | `ct_t1_dit_log3` | 4460 |
| avx2 | `t1_dit_log3` | 1792 | 1800 | `ct_t1_dit_log3` | 4631 |
| avx2 | `t1_dit_log3` | 1792 | 12544 | `ct_t1_dit_log3` | 4620 |
| avx2 | `t1_dit_log3` | 1792 | 14336 | `ct_t1_dit_log3` | 4267 |
| avx2 | `t1s_dit` | 56 | 56 | `ct_t1s_dit` | 106 |
| avx2 | `t1s_dit` | 56 | 64 | `ct_t1s_dit` | 105 |
| avx2 | `t1s_dit` | 56 | 392 | `ct_t1s_dit` | 111 |
| avx2 | `t1s_dit` | 56 | 448 | `ct_t1s_dit` | 111 |
| avx2 | `t1s_dit` | 112 | 112 | `ct_t1s_dit` | 199 |
| avx2 | `t1s_dit` | 112 | 120 | `ct_t1s_dit` | 196 |
| avx2 | `t1s_dit` | 112 | 784 | `ct_t1s_dit` | 224 |
| avx2 | `t1s_dit` | 112 | 896 | `ct_t1s_dit` | 222 |
| avx2 | `t1s_dit` | 224 | 224 | `ct_t1s_dit` | 400 |
| avx2 | `t1s_dit` | 224 | 232 | `ct_t1s_dit` | 408 |
| avx2 | `t1s_dit` | 224 | 1568 | `ct_t1s_dit` | 429 |
| avx2 | `t1s_dit` | 224 | 1792 | `ct_t1s_dit` | 425 |
| avx2 | `t1s_dit` | 448 | 448 | `ct_t1s_dit` | 819 |
| avx2 | `t1s_dit` | 448 | 456 | `ct_t1s_dit` | 868 |
| avx2 | `t1s_dit` | 448 | 3136 | `ct_t1s_dit` | 915 |
| avx2 | `t1s_dit` | 448 | 3584 | `ct_t1s_dit` | 1574 |
| avx2 | `t1s_dit` | 896 | 896 | `ct_t1s_dit` | 1782 |
| avx2 | `t1s_dit` | 896 | 904 | `ct_t1s_dit` | 1722 |
| avx2 | `t1s_dit` | 896 | 6272 | `ct_t1s_dit` | 1755 |
| avx2 | `t1s_dit` | 896 | 7168 | `ct_t1s_dit` | 3029 |
| avx2 | `t1s_dit` | 1792 | 1792 | `ct_t1s_dit` | 3584 |
| avx2 | `t1s_dit` | 1792 | 1800 | `ct_t1s_dit` | 3358 |
| avx2 | `t1s_dit` | 1792 | 12544 | `ct_t1s_dit` | 3380 |
| avx2 | `t1s_dit` | 1792 | 14336 | `ct_t1s_dit` | 3286 |