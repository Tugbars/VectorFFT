# VectorFFT R=5 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 40 | 40 | 38 | 49 | 37 | — | t1s |
| avx2 | 40 | 48 | 38 | 49 | 37 | — | t1s |
| avx2 | 40 | 200 | 37 | 49 | 37 | — | t1s |
| avx2 | 40 | 320 | 37 | 49 | 37 | — | t1s |
| avx2 | 80 | 80 | 73 | 98 | 73 | — | t1s |
| avx2 | 80 | 88 | 74 | 98 | 75 | — | flat |
| avx2 | 80 | 400 | 75 | 98 | 75 | — | t1s |
| avx2 | 80 | 640 | 74 | 98 | 76 | — | flat |
| avx2 | 160 | 160 | 146 | 194 | 145 | — | t1s |
| avx2 | 160 | 168 | 191 | 194 | 148 | — | t1s |
| avx2 | 160 | 800 | 147 | 195 | 148 | — | flat |
| avx2 | 160 | 1280 | 146 | 195 | 145 | — | t1s |
| avx2 | 320 | 320 | 287 | 387 | 296 | — | flat |
| avx2 | 320 | 328 | 288 | 384 | 296 | — | flat |
| avx2 | 320 | 1600 | 292 | 387 | 289 | — | t1s |
| avx2 | 320 | 2560 | 320 | 388 | 297 | — | t1s |
| avx2 | 640 | 640 | 676 | 791 | 602 | — | t1s |
| avx2 | 640 | 648 | 667 | 791 | 583 | — | t1s |
| avx2 | 640 | 3200 | 668 | 798 | 585 | — | t1s |
| avx2 | 640 | 5120 | 708 | 792 | 857 | — | flat |
| avx2 | 1280 | 1280 | 1347 | 1573 | 1180 | — | t1s |
| avx2 | 1280 | 1288 | 1381 | 1577 | 1167 | — | t1s |
| avx2 | 1280 | 6400 | 1357 | 1583 | 1214 | — | t1s |
| avx2 | 1280 | 10240 | 1481 | 1587 | 1904 | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 40 | 40 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 48 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 200 | `ct_t1_dit` | 37 |
| avx2 | `t1_dit` | 40 | 320 | `ct_t1_dit` | 37 |
| avx2 | `t1_dit` | 80 | 80 | `ct_t1_dit` | 73 |
| avx2 | `t1_dit` | 80 | 88 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 80 | 400 | `ct_t1_dit` | 75 |
| avx2 | `t1_dit` | 80 | 640 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 160 | 160 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 160 | 168 | `ct_t1_dit` | 191 |
| avx2 | `t1_dit` | 160 | 800 | `ct_t1_dit` | 147 |
| avx2 | `t1_dit` | 160 | 1280 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 320 | 320 | `ct_t1_dit` | 287 |
| avx2 | `t1_dit` | 320 | 328 | `ct_t1_dit` | 288 |
| avx2 | `t1_dit` | 320 | 1600 | `ct_t1_dit` | 292 |
| avx2 | `t1_dit` | 320 | 2560 | `ct_t1_dit` | 320 |
| avx2 | `t1_dit` | 640 | 640 | `ct_t1_dit` | 676 |
| avx2 | `t1_dit` | 640 | 648 | `ct_t1_dit` | 667 |
| avx2 | `t1_dit` | 640 | 3200 | `ct_t1_dit` | 668 |
| avx2 | `t1_dit` | 640 | 5120 | `ct_t1_dit` | 708 |
| avx2 | `t1_dit` | 1280 | 1280 | `ct_t1_dit` | 1347 |
| avx2 | `t1_dit` | 1280 | 1288 | `ct_t1_dit` | 1381 |
| avx2 | `t1_dit` | 1280 | 6400 | `ct_t1_dit` | 1357 |
| avx2 | `t1_dit` | 1280 | 10240 | `ct_t1_dit` | 1481 |
| avx2 | `t1_dit_log3` | 40 | 40 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 48 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 200 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 320 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 80 | 80 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 88 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 400 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 640 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 160 | 160 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 160 | 168 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 160 | 800 | `ct_t1_dit_log3` | 195 |
| avx2 | `t1_dit_log3` | 160 | 1280 | `ct_t1_dit_log3` | 195 |
| avx2 | `t1_dit_log3` | 320 | 320 | `ct_t1_dit_log3` | 387 |
| avx2 | `t1_dit_log3` | 320 | 328 | `ct_t1_dit_log3` | 384 |
| avx2 | `t1_dit_log3` | 320 | 1600 | `ct_t1_dit_log3` | 387 |
| avx2 | `t1_dit_log3` | 320 | 2560 | `ct_t1_dit_log3` | 388 |
| avx2 | `t1_dit_log3` | 640 | 640 | `ct_t1_dit_log3` | 791 |
| avx2 | `t1_dit_log3` | 640 | 648 | `ct_t1_dit_log3` | 791 |
| avx2 | `t1_dit_log3` | 640 | 3200 | `ct_t1_dit_log3` | 798 |
| avx2 | `t1_dit_log3` | 640 | 5120 | `ct_t1_dit_log3` | 792 |
| avx2 | `t1_dit_log3` | 1280 | 1280 | `ct_t1_dit_log3` | 1573 |
| avx2 | `t1_dit_log3` | 1280 | 1288 | `ct_t1_dit_log3` | 1577 |
| avx2 | `t1_dit_log3` | 1280 | 6400 | `ct_t1_dit_log3` | 1583 |
| avx2 | `t1_dit_log3` | 1280 | 10240 | `ct_t1_dit_log3` | 1587 |
| avx2 | `t1s_dit` | 40 | 40 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 40 | 48 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 40 | 200 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 40 | 320 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 80 | 80 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 80 | 88 | `ct_t1s_dit` | 75 |
| avx2 | `t1s_dit` | 80 | 400 | `ct_t1s_dit` | 75 |
| avx2 | `t1s_dit` | 80 | 640 | `ct_t1s_dit` | 76 |
| avx2 | `t1s_dit` | 160 | 160 | `ct_t1s_dit` | 145 |
| avx2 | `t1s_dit` | 160 | 168 | `ct_t1s_dit` | 148 |
| avx2 | `t1s_dit` | 160 | 800 | `ct_t1s_dit` | 148 |
| avx2 | `t1s_dit` | 160 | 1280 | `ct_t1s_dit` | 145 |
| avx2 | `t1s_dit` | 320 | 320 | `ct_t1s_dit` | 296 |
| avx2 | `t1s_dit` | 320 | 328 | `ct_t1s_dit` | 296 |
| avx2 | `t1s_dit` | 320 | 1600 | `ct_t1s_dit` | 289 |
| avx2 | `t1s_dit` | 320 | 2560 | `ct_t1s_dit` | 297 |
| avx2 | `t1s_dit` | 640 | 640 | `ct_t1s_dit` | 602 |
| avx2 | `t1s_dit` | 640 | 648 | `ct_t1s_dit` | 583 |
| avx2 | `t1s_dit` | 640 | 3200 | `ct_t1s_dit` | 585 |
| avx2 | `t1s_dit` | 640 | 5120 | `ct_t1s_dit` | 857 |
| avx2 | `t1s_dit` | 1280 | 1280 | `ct_t1s_dit` | 1180 |
| avx2 | `t1s_dit` | 1280 | 1288 | `ct_t1s_dit` | 1167 |
| avx2 | `t1s_dit` | 1280 | 6400 | `ct_t1s_dit` | 1214 |
| avx2 | `t1s_dit` | 1280 | 10240 | `ct_t1s_dit` | 1904 |