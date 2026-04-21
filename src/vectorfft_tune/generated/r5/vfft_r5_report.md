# VectorFFT R=5 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 40 | 40 | 39 | 52 | 41 | — | flat |
| avx2 | 40 | 48 | 44 | 56 | 43 | — | t1s |
| avx2 | 40 | 200 | 41 | 55 | 41 | — | flat |
| avx2 | 40 | 320 | 40 | 53 | 39 | — | t1s |
| avx2 | 80 | 80 | 78 | 106 | 78 | — | t1s |
| avx2 | 80 | 88 | 79 | 115 | 86 | — | flat |
| avx2 | 80 | 400 | 78 | 112 | 85 | — | flat |
| avx2 | 80 | 640 | 84 | 103 | 90 | — | flat |
| avx2 | 160 | 160 | 155 | 222 | 151 | — | t1s |
| avx2 | 160 | 168 | 169 | 216 | 161 | — | t1s |
| avx2 | 160 | 800 | 158 | 225 | 167 | — | flat |
| avx2 | 160 | 1280 | 147 | 204 | 148 | — | flat |
| avx2 | 320 | 320 | 305 | 440 | 312 | — | flat |
| avx2 | 320 | 328 | 350 | 439 | 312 | — | t1s |
| avx2 | 320 | 1600 | 325 | 412 | 322 | — | t1s |
| avx2 | 320 | 2560 | 354 | 426 | 330 | — | t1s |
| avx2 | 640 | 640 | 716 | 889 | 657 | — | t1s |
| avx2 | 640 | 648 | 711 | 923 | 626 | — | t1s |
| avx2 | 640 | 3200 | 778 | 843 | 670 | — | t1s |
| avx2 | 640 | 5120 | 830 | 865 | 619 | — | t1s |
| avx2 | 1280 | 1280 | 1387 | 1742 | 1384 | — | t1s |
| avx2 | 1280 | 1288 | 1411 | 1751 | 1290 | — | t1s |
| avx2 | 1280 | 6400 | 1474 | 1715 | 1335 | — | t1s |
| avx2 | 1280 | 10240 | 1702 | 1724 | 1360 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 40 | 40 | `ct_t1_dit` | 39 |
| avx2 | `t1_dit` | 40 | 48 | `ct_t1_dit` | 44 |
| avx2 | `t1_dit` | 40 | 200 | `ct_t1_dit` | 41 |
| avx2 | `t1_dit` | 40 | 320 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 80 | 80 | `ct_t1_dit` | 78 |
| avx2 | `t1_dit` | 80 | 88 | `ct_t1_dit` | 79 |
| avx2 | `t1_dit` | 80 | 400 | `ct_t1_dit` | 78 |
| avx2 | `t1_dit` | 80 | 640 | `ct_t1_dit` | 84 |
| avx2 | `t1_dit` | 160 | 160 | `ct_t1_dit` | 155 |
| avx2 | `t1_dit` | 160 | 168 | `ct_t1_dit` | 169 |
| avx2 | `t1_dit` | 160 | 800 | `ct_t1_dit` | 158 |
| avx2 | `t1_dit` | 160 | 1280 | `ct_t1_dit` | 147 |
| avx2 | `t1_dit` | 320 | 320 | `ct_t1_dit` | 305 |
| avx2 | `t1_dit` | 320 | 328 | `ct_t1_dit` | 350 |
| avx2 | `t1_dit` | 320 | 1600 | `ct_t1_dit` | 325 |
| avx2 | `t1_dit` | 320 | 2560 | `ct_t1_dit` | 354 |
| avx2 | `t1_dit` | 640 | 640 | `ct_t1_dit` | 716 |
| avx2 | `t1_dit` | 640 | 648 | `ct_t1_dit` | 711 |
| avx2 | `t1_dit` | 640 | 3200 | `ct_t1_dit` | 778 |
| avx2 | `t1_dit` | 640 | 5120 | `ct_t1_dit` | 830 |
| avx2 | `t1_dit` | 1280 | 1280 | `ct_t1_dit` | 1387 |
| avx2 | `t1_dit` | 1280 | 1288 | `ct_t1_dit` | 1411 |
| avx2 | `t1_dit` | 1280 | 6400 | `ct_t1_dit` | 1474 |
| avx2 | `t1_dit` | 1280 | 10240 | `ct_t1_dit` | 1702 |
| avx2 | `t1_dit_log3` | 40 | 40 | `ct_t1_dit_log3` | 52 |
| avx2 | `t1_dit_log3` | 40 | 48 | `ct_t1_dit_log3` | 56 |
| avx2 | `t1_dit_log3` | 40 | 200 | `ct_t1_dit_log3` | 55 |
| avx2 | `t1_dit_log3` | 40 | 320 | `ct_t1_dit_log3` | 53 |
| avx2 | `t1_dit_log3` | 80 | 80 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 80 | 88 | `ct_t1_dit_log3` | 115 |
| avx2 | `t1_dit_log3` | 80 | 400 | `ct_t1_dit_log3` | 112 |
| avx2 | `t1_dit_log3` | 80 | 640 | `ct_t1_dit_log3` | 103 |
| avx2 | `t1_dit_log3` | 160 | 160 | `ct_t1_dit_log3` | 222 |
| avx2 | `t1_dit_log3` | 160 | 168 | `ct_t1_dit_log3` | 216 |
| avx2 | `t1_dit_log3` | 160 | 800 | `ct_t1_dit_log3` | 225 |
| avx2 | `t1_dit_log3` | 160 | 1280 | `ct_t1_dit_log3` | 204 |
| avx2 | `t1_dit_log3` | 320 | 320 | `ct_t1_dit_log3` | 440 |
| avx2 | `t1_dit_log3` | 320 | 328 | `ct_t1_dit_log3` | 439 |
| avx2 | `t1_dit_log3` | 320 | 1600 | `ct_t1_dit_log3` | 412 |
| avx2 | `t1_dit_log3` | 320 | 2560 | `ct_t1_dit_log3` | 426 |
| avx2 | `t1_dit_log3` | 640 | 640 | `ct_t1_dit_log3` | 889 |
| avx2 | `t1_dit_log3` | 640 | 648 | `ct_t1_dit_log3` | 923 |
| avx2 | `t1_dit_log3` | 640 | 3200 | `ct_t1_dit_log3` | 843 |
| avx2 | `t1_dit_log3` | 640 | 5120 | `ct_t1_dit_log3` | 865 |
| avx2 | `t1_dit_log3` | 1280 | 1280 | `ct_t1_dit_log3` | 1742 |
| avx2 | `t1_dit_log3` | 1280 | 1288 | `ct_t1_dit_log3` | 1751 |
| avx2 | `t1_dit_log3` | 1280 | 6400 | `ct_t1_dit_log3` | 1715 |
| avx2 | `t1_dit_log3` | 1280 | 10240 | `ct_t1_dit_log3` | 1724 |
| avx2 | `t1s_dit` | 40 | 40 | `ct_t1s_dit` | 41 |
| avx2 | `t1s_dit` | 40 | 48 | `ct_t1s_dit` | 43 |
| avx2 | `t1s_dit` | 40 | 200 | `ct_t1s_dit` | 41 |
| avx2 | `t1s_dit` | 40 | 320 | `ct_t1s_dit` | 39 |
| avx2 | `t1s_dit` | 80 | 80 | `ct_t1s_dit` | 78 |
| avx2 | `t1s_dit` | 80 | 88 | `ct_t1s_dit` | 86 |
| avx2 | `t1s_dit` | 80 | 400 | `ct_t1s_dit` | 85 |
| avx2 | `t1s_dit` | 80 | 640 | `ct_t1s_dit` | 90 |
| avx2 | `t1s_dit` | 160 | 160 | `ct_t1s_dit` | 151 |
| avx2 | `t1s_dit` | 160 | 168 | `ct_t1s_dit` | 161 |
| avx2 | `t1s_dit` | 160 | 800 | `ct_t1s_dit` | 167 |
| avx2 | `t1s_dit` | 160 | 1280 | `ct_t1s_dit` | 148 |
| avx2 | `t1s_dit` | 320 | 320 | `ct_t1s_dit` | 312 |
| avx2 | `t1s_dit` | 320 | 328 | `ct_t1s_dit` | 312 |
| avx2 | `t1s_dit` | 320 | 1600 | `ct_t1s_dit` | 322 |
| avx2 | `t1s_dit` | 320 | 2560 | `ct_t1s_dit` | 330 |
| avx2 | `t1s_dit` | 640 | 640 | `ct_t1s_dit` | 657 |
| avx2 | `t1s_dit` | 640 | 648 | `ct_t1s_dit` | 626 |
| avx2 | `t1s_dit` | 640 | 3200 | `ct_t1s_dit` | 670 |
| avx2 | `t1s_dit` | 640 | 5120 | `ct_t1s_dit` | 619 |
| avx2 | `t1s_dit` | 1280 | 1280 | `ct_t1s_dit` | 1384 |
| avx2 | `t1s_dit` | 1280 | 1288 | `ct_t1s_dit` | 1290 |
| avx2 | `t1s_dit` | 1280 | 6400 | `ct_t1s_dit` | 1335 |
| avx2 | `t1s_dit` | 1280 | 10240 | `ct_t1s_dit` | 1360 |