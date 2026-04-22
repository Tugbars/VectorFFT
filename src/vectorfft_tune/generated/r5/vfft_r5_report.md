# VectorFFT R=5 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 40 | 40 | 38 | 49 | 37 | — | t1s |
| avx2 | 40 | 48 | 38 | 49 | 37 | — | t1s |
| avx2 | 40 | 200 | 39 | 49 | 37 | — | t1s |
| avx2 | 40 | 320 | 38 | 49 | 37 | — | t1s |
| avx2 | 80 | 80 | 77 | 99 | 73 | — | t1s |
| avx2 | 80 | 88 | 74 | 98 | 75 | — | flat |
| avx2 | 80 | 400 | 79 | 98 | 73 | — | t1s |
| avx2 | 80 | 640 | 74 | 99 | 73 | — | t1s |
| avx2 | 160 | 160 | 145 | 196 | 150 | — | flat |
| avx2 | 160 | 168 | 147 | 195 | 146 | — | t1s |
| avx2 | 160 | 800 | 147 | 194 | 150 | — | flat |
| avx2 | 160 | 1280 | 147 | 195 | 146 | — | t1s |
| avx2 | 320 | 320 | 290 | 389 | 296 | — | flat |
| avx2 | 320 | 328 | 293 | 388 | 296 | — | flat |
| avx2 | 320 | 1600 | 296 | 388 | 288 | — | t1s |
| avx2 | 320 | 2560 | 318 | 389 | 296 | — | t1s |
| avx2 | 640 | 640 | 670 | 801 | 580 | — | t1s |
| avx2 | 640 | 648 | 684 | 806 | 585 | — | t1s |
| avx2 | 640 | 3200 | 703 | 804 | 633 | — | t1s |
| avx2 | 640 | 5120 | 729 | 790 | 604 | — | t1s |
| avx2 | 1280 | 1280 | 1366 | 1618 | 1230 | — | t1s |
| avx2 | 1280 | 1288 | 1380 | 1575 | 1186 | — | t1s |
| avx2 | 1280 | 6400 | 1370 | 1612 | 1211 | — | t1s |
| avx2 | 1280 | 10240 | 1428 | 1619 | 1241 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 40 | 40 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 48 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 200 | `ct_t1_dit` | 39 |
| avx2 | `t1_dit` | 40 | 320 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 80 | 80 | `ct_t1_dit` | 77 |
| avx2 | `t1_dit` | 80 | 88 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 80 | 400 | `ct_t1_dit` | 79 |
| avx2 | `t1_dit` | 80 | 640 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 160 | 160 | `ct_t1_dit` | 145 |
| avx2 | `t1_dit` | 160 | 168 | `ct_t1_dit` | 147 |
| avx2 | `t1_dit` | 160 | 800 | `ct_t1_dit` | 147 |
| avx2 | `t1_dit` | 160 | 1280 | `ct_t1_dit` | 147 |
| avx2 | `t1_dit` | 320 | 320 | `ct_t1_dit` | 290 |
| avx2 | `t1_dit` | 320 | 328 | `ct_t1_dit` | 293 |
| avx2 | `t1_dit` | 320 | 1600 | `ct_t1_dit` | 296 |
| avx2 | `t1_dit` | 320 | 2560 | `ct_t1_dit` | 318 |
| avx2 | `t1_dit` | 640 | 640 | `ct_t1_dit` | 670 |
| avx2 | `t1_dit` | 640 | 648 | `ct_t1_dit` | 684 |
| avx2 | `t1_dit` | 640 | 3200 | `ct_t1_dit` | 703 |
| avx2 | `t1_dit` | 640 | 5120 | `ct_t1_dit` | 729 |
| avx2 | `t1_dit` | 1280 | 1280 | `ct_t1_dit` | 1366 |
| avx2 | `t1_dit` | 1280 | 1288 | `ct_t1_dit` | 1380 |
| avx2 | `t1_dit` | 1280 | 6400 | `ct_t1_dit` | 1370 |
| avx2 | `t1_dit` | 1280 | 10240 | `ct_t1_dit` | 1428 |
| avx2 | `t1_dit_log3` | 40 | 40 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 48 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 200 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 320 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 80 | 80 | `ct_t1_dit_log3` | 99 |
| avx2 | `t1_dit_log3` | 80 | 88 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 400 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 640 | `ct_t1_dit_log3` | 99 |
| avx2 | `t1_dit_log3` | 160 | 160 | `ct_t1_dit_log3` | 196 |
| avx2 | `t1_dit_log3` | 160 | 168 | `ct_t1_dit_log3` | 195 |
| avx2 | `t1_dit_log3` | 160 | 800 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 160 | 1280 | `ct_t1_dit_log3` | 195 |
| avx2 | `t1_dit_log3` | 320 | 320 | `ct_t1_dit_log3` | 389 |
| avx2 | `t1_dit_log3` | 320 | 328 | `ct_t1_dit_log3` | 388 |
| avx2 | `t1_dit_log3` | 320 | 1600 | `ct_t1_dit_log3` | 388 |
| avx2 | `t1_dit_log3` | 320 | 2560 | `ct_t1_dit_log3` | 389 |
| avx2 | `t1_dit_log3` | 640 | 640 | `ct_t1_dit_log3` | 801 |
| avx2 | `t1_dit_log3` | 640 | 648 | `ct_t1_dit_log3` | 806 |
| avx2 | `t1_dit_log3` | 640 | 3200 | `ct_t1_dit_log3` | 804 |
| avx2 | `t1_dit_log3` | 640 | 5120 | `ct_t1_dit_log3` | 790 |
| avx2 | `t1_dit_log3` | 1280 | 1280 | `ct_t1_dit_log3` | 1618 |
| avx2 | `t1_dit_log3` | 1280 | 1288 | `ct_t1_dit_log3` | 1575 |
| avx2 | `t1_dit_log3` | 1280 | 6400 | `ct_t1_dit_log3` | 1612 |
| avx2 | `t1_dit_log3` | 1280 | 10240 | `ct_t1_dit_log3` | 1619 |
| avx2 | `t1s_dit` | 40 | 40 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 40 | 48 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 40 | 200 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 40 | 320 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 80 | 80 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 80 | 88 | `ct_t1s_dit` | 75 |
| avx2 | `t1s_dit` | 80 | 400 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 80 | 640 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 160 | 160 | `ct_t1s_dit` | 150 |
| avx2 | `t1s_dit` | 160 | 168 | `ct_t1s_dit` | 146 |
| avx2 | `t1s_dit` | 160 | 800 | `ct_t1s_dit` | 150 |
| avx2 | `t1s_dit` | 160 | 1280 | `ct_t1s_dit` | 146 |
| avx2 | `t1s_dit` | 320 | 320 | `ct_t1s_dit` | 296 |
| avx2 | `t1s_dit` | 320 | 328 | `ct_t1s_dit` | 296 |
| avx2 | `t1s_dit` | 320 | 1600 | `ct_t1s_dit` | 288 |
| avx2 | `t1s_dit` | 320 | 2560 | `ct_t1s_dit` | 296 |
| avx2 | `t1s_dit` | 640 | 640 | `ct_t1s_dit` | 580 |
| avx2 | `t1s_dit` | 640 | 648 | `ct_t1s_dit` | 585 |
| avx2 | `t1s_dit` | 640 | 3200 | `ct_t1s_dit` | 633 |
| avx2 | `t1s_dit` | 640 | 5120 | `ct_t1s_dit` | 604 |
| avx2 | `t1s_dit` | 1280 | 1280 | `ct_t1s_dit` | 1230 |
| avx2 | `t1s_dit` | 1280 | 1288 | `ct_t1s_dit` | 1186 |
| avx2 | `t1s_dit` | 1280 | 6400 | `ct_t1s_dit` | 1211 |
| avx2 | `t1s_dit` | 1280 | 10240 | `ct_t1s_dit` | 1241 |