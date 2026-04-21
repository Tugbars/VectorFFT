# VectorFFT R=5 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 40 | 40 | 38 | 50 | 38 | — | t1s |
| avx2 | 40 | 48 | 38 | 52 | 40 | — | flat |
| avx2 | 40 | 200 | 38 | 49 | 41 | — | flat |
| avx2 | 40 | 320 | 38 | 49 | 39 | — | flat |
| avx2 | 80 | 80 | 74 | 98 | 78 | — | flat |
| avx2 | 80 | 88 | 76 | 106 | 79 | — | flat |
| avx2 | 80 | 400 | 74 | 98 | 79 | — | flat |
| avx2 | 80 | 640 | 78 | 102 | 78 | — | flat |
| avx2 | 160 | 160 | 146 | 207 | 149 | — | flat |
| avx2 | 160 | 168 | 151 | 195 | 149 | — | t1s |
| avx2 | 160 | 800 | 156 | 200 | 163 | — | flat |
| avx2 | 160 | 1280 | 146 | 194 | 149 | — | flat |
| avx2 | 320 | 320 | 291 | 424 | 299 | — | flat |
| avx2 | 320 | 328 | 291 | 420 | 294 | — | flat |
| avx2 | 320 | 1600 | 307 | 402 | 306 | — | t1s |
| avx2 | 320 | 2560 | 321 | 397 | 309 | — | t1s |
| avx2 | 640 | 640 | 683 | 848 | 583 | — | t1s |
| avx2 | 640 | 648 | 700 | 841 | 580 | — | t1s |
| avx2 | 640 | 3200 | 698 | 811 | 653 | — | t1s |
| avx2 | 640 | 5120 | 734 | 790 | 652 | — | t1s |
| avx2 | 1280 | 1280 | 1348 | 1600 | 1171 | — | t1s |
| avx2 | 1280 | 1288 | 1355 | 1698 | 1288 | — | t1s |
| avx2 | 1280 | 6400 | 1461 | 1578 | 1196 | — | t1s |
| avx2 | 1280 | 10240 | 1490 | 1633 | 1243 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 40 | 40 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 48 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 200 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 40 | 320 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 80 | 80 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 80 | 88 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 80 | 400 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 80 | 640 | `ct_t1_dit` | 78 |
| avx2 | `t1_dit` | 160 | 160 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 160 | 168 | `ct_t1_dit` | 151 |
| avx2 | `t1_dit` | 160 | 800 | `ct_t1_dit` | 156 |
| avx2 | `t1_dit` | 160 | 1280 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 320 | 320 | `ct_t1_dit` | 291 |
| avx2 | `t1_dit` | 320 | 328 | `ct_t1_dit` | 291 |
| avx2 | `t1_dit` | 320 | 1600 | `ct_t1_dit` | 307 |
| avx2 | `t1_dit` | 320 | 2560 | `ct_t1_dit` | 321 |
| avx2 | `t1_dit` | 640 | 640 | `ct_t1_dit` | 683 |
| avx2 | `t1_dit` | 640 | 648 | `ct_t1_dit` | 700 |
| avx2 | `t1_dit` | 640 | 3200 | `ct_t1_dit` | 698 |
| avx2 | `t1_dit` | 640 | 5120 | `ct_t1_dit` | 734 |
| avx2 | `t1_dit` | 1280 | 1280 | `ct_t1_dit` | 1348 |
| avx2 | `t1_dit` | 1280 | 1288 | `ct_t1_dit` | 1355 |
| avx2 | `t1_dit` | 1280 | 6400 | `ct_t1_dit` | 1461 |
| avx2 | `t1_dit` | 1280 | 10240 | `ct_t1_dit` | 1490 |
| avx2 | `t1_dit_log3` | 40 | 40 | `ct_t1_dit_log3` | 50 |
| avx2 | `t1_dit_log3` | 40 | 48 | `ct_t1_dit_log3` | 52 |
| avx2 | `t1_dit_log3` | 40 | 200 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 40 | 320 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 80 | 80 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 88 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 80 | 400 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 80 | 640 | `ct_t1_dit_log3` | 102 |
| avx2 | `t1_dit_log3` | 160 | 160 | `ct_t1_dit_log3` | 207 |
| avx2 | `t1_dit_log3` | 160 | 168 | `ct_t1_dit_log3` | 195 |
| avx2 | `t1_dit_log3` | 160 | 800 | `ct_t1_dit_log3` | 200 |
| avx2 | `t1_dit_log3` | 160 | 1280 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 320 | 320 | `ct_t1_dit_log3` | 424 |
| avx2 | `t1_dit_log3` | 320 | 328 | `ct_t1_dit_log3` | 420 |
| avx2 | `t1_dit_log3` | 320 | 1600 | `ct_t1_dit_log3` | 402 |
| avx2 | `t1_dit_log3` | 320 | 2560 | `ct_t1_dit_log3` | 397 |
| avx2 | `t1_dit_log3` | 640 | 640 | `ct_t1_dit_log3` | 848 |
| avx2 | `t1_dit_log3` | 640 | 648 | `ct_t1_dit_log3` | 841 |
| avx2 | `t1_dit_log3` | 640 | 3200 | `ct_t1_dit_log3` | 811 |
| avx2 | `t1_dit_log3` | 640 | 5120 | `ct_t1_dit_log3` | 790 |
| avx2 | `t1_dit_log3` | 1280 | 1280 | `ct_t1_dit_log3` | 1600 |
| avx2 | `t1_dit_log3` | 1280 | 1288 | `ct_t1_dit_log3` | 1698 |
| avx2 | `t1_dit_log3` | 1280 | 6400 | `ct_t1_dit_log3` | 1578 |
| avx2 | `t1_dit_log3` | 1280 | 10240 | `ct_t1_dit_log3` | 1633 |
| avx2 | `t1s_dit` | 40 | 40 | `ct_t1s_dit` | 38 |
| avx2 | `t1s_dit` | 40 | 48 | `ct_t1s_dit` | 40 |
| avx2 | `t1s_dit` | 40 | 200 | `ct_t1s_dit` | 41 |
| avx2 | `t1s_dit` | 40 | 320 | `ct_t1s_dit` | 39 |
| avx2 | `t1s_dit` | 80 | 80 | `ct_t1s_dit` | 78 |
| avx2 | `t1s_dit` | 80 | 88 | `ct_t1s_dit` | 79 |
| avx2 | `t1s_dit` | 80 | 400 | `ct_t1s_dit` | 79 |
| avx2 | `t1s_dit` | 80 | 640 | `ct_t1s_dit` | 78 |
| avx2 | `t1s_dit` | 160 | 160 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 160 | 168 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 160 | 800 | `ct_t1s_dit` | 163 |
| avx2 | `t1s_dit` | 160 | 1280 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 320 | 320 | `ct_t1s_dit` | 299 |
| avx2 | `t1s_dit` | 320 | 328 | `ct_t1s_dit` | 294 |
| avx2 | `t1s_dit` | 320 | 1600 | `ct_t1s_dit` | 306 |
| avx2 | `t1s_dit` | 320 | 2560 | `ct_t1s_dit` | 309 |
| avx2 | `t1s_dit` | 640 | 640 | `ct_t1s_dit` | 583 |
| avx2 | `t1s_dit` | 640 | 648 | `ct_t1s_dit` | 580 |
| avx2 | `t1s_dit` | 640 | 3200 | `ct_t1s_dit` | 653 |
| avx2 | `t1s_dit` | 640 | 5120 | `ct_t1s_dit` | 652 |
| avx2 | `t1s_dit` | 1280 | 1280 | `ct_t1s_dit` | 1171 |
| avx2 | `t1s_dit` | 1280 | 1288 | `ct_t1s_dit` | 1288 |
| avx2 | `t1s_dit` | 1280 | 6400 | `ct_t1s_dit` | 1196 |
| avx2 | `t1s_dit` | 1280 | 10240 | `ct_t1s_dit` | 1243 |