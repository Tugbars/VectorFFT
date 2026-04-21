# VectorFFT R=5 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 40 | 40 | 91 | 88 | 84 | — | t1s |
| avx2 | 40 | 48 | 87 | 89 | 84 | — | t1s |
| avx2 | 40 | 200 | 86 | 88 | 83 | — | t1s |
| avx2 | 40 | 320 | 87 | 89 | 83 | — | t1s |
| avx2 | 80 | 80 | 173 | 174 | 161 | — | t1s |
| avx2 | 80 | 88 | 169 | 183 | 161 | — | t1s |
| avx2 | 80 | 400 | 170 | 174 | 159 | — | t1s |
| avx2 | 80 | 640 | 171 | 174 | 163 | — | t1s |
| avx2 | 160 | 160 | 338 | 348 | 307 | — | t1s |
| avx2 | 160 | 168 | 337 | 346 | 310 | — | t1s |
| avx2 | 160 | 800 | 340 | 346 | 303 | — | t1s |
| avx2 | 160 | 1280 | 336 | 356 | 311 | — | t1s |
| avx2 | 320 | 320 | 663 | 692 | 616 | — | t1s |
| avx2 | 320 | 328 | 683 | 696 | 600 | — | t1s |
| avx2 | 320 | 1600 | 690 | 700 | 610 | — | t1s |
| avx2 | 320 | 2560 | 712 | 762 | 610 | — | t1s |
| avx2 | 640 | 640 | 1450 | 1409 | 1253 | — | t1s |
| avx2 | 640 | 648 | 1453 | 1409 | 1258 | — | t1s |
| avx2 | 640 | 3200 | 1446 | 1426 | 1216 | — | t1s |
| avx2 | 640 | 5120 | 1611 | 1558 | 1250 | — | t1s |
| avx2 | 1280 | 1280 | 2932 | 3010 | 2444 | — | t1s |
| avx2 | 1280 | 1288 | 2894 | 2917 | 2500 | — | t1s |
| avx2 | 1280 | 6400 | 3031 | 2904 | 2631 | — | t1s |
| avx2 | 1280 | 10240 | 3440 | 3049 | 2773 | — | t1s |
| avx512 | 40 | 40 | 62 | 66 | 53 | — | t1s |
| avx512 | 40 | 48 | 59 | 69 | 52 | — | t1s |
| avx512 | 40 | 200 | 60 | 68 | 55 | — | t1s |
| avx512 | 40 | 320 | 63 | 68 | 55 | — | t1s |
| avx512 | 80 | 80 | 120 | 135 | 110 | — | t1s |
| avx512 | 80 | 88 | 120 | 135 | 107 | — | t1s |
| avx512 | 80 | 400 | 120 | 134 | 104 | — | t1s |
| avx512 | 80 | 640 | 122 | 139 | 104 | — | t1s |
| avx512 | 160 | 160 | 236 | 270 | 216 | — | t1s |
| avx512 | 160 | 168 | 239 | 268 | 217 | — | t1s |
| avx512 | 160 | 800 | 244 | 262 | 211 | — | t1s |
| avx512 | 160 | 1280 | 236 | 259 | 215 | — | t1s |
| avx512 | 320 | 320 | 467 | 540 | 424 | — | t1s |
| avx512 | 320 | 328 | 473 | 522 | 417 | — | t1s |
| avx512 | 320 | 1600 | 482 | 526 | 413 | — | t1s |
| avx512 | 320 | 2560 | 543 | 540 | 418 | — | t1s |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 40 | 40 | `ct_t1_dit` | 91 |
| avx2 | `t1_dit` | 40 | 48 | `ct_t1_dit` | 87 |
| avx2 | `t1_dit` | 40 | 200 | `ct_t1_dit` | 86 |
| avx2 | `t1_dit` | 40 | 320 | `ct_t1_dit` | 87 |
| avx2 | `t1_dit` | 80 | 80 | `ct_t1_dit` | 173 |
| avx2 | `t1_dit` | 80 | 88 | `ct_t1_dit` | 169 |
| avx2 | `t1_dit` | 80 | 400 | `ct_t1_dit` | 170 |
| avx2 | `t1_dit` | 80 | 640 | `ct_t1_dit` | 171 |
| avx2 | `t1_dit` | 160 | 160 | `ct_t1_dit` | 338 |
| avx2 | `t1_dit` | 160 | 168 | `ct_t1_dit` | 337 |
| avx2 | `t1_dit` | 160 | 800 | `ct_t1_dit` | 340 |
| avx2 | `t1_dit` | 160 | 1280 | `ct_t1_dit` | 336 |
| avx2 | `t1_dit` | 320 | 320 | `ct_t1_dit` | 663 |
| avx2 | `t1_dit` | 320 | 328 | `ct_t1_dit` | 683 |
| avx2 | `t1_dit` | 320 | 1600 | `ct_t1_dit` | 690 |
| avx2 | `t1_dit` | 320 | 2560 | `ct_t1_dit` | 712 |
| avx2 | `t1_dit` | 640 | 640 | `ct_t1_dit` | 1450 |
| avx2 | `t1_dit` | 640 | 648 | `ct_t1_dit` | 1453 |
| avx2 | `t1_dit` | 640 | 3200 | `ct_t1_dit` | 1446 |
| avx2 | `t1_dit` | 640 | 5120 | `ct_t1_dit` | 1611 |
| avx2 | `t1_dit` | 1280 | 1280 | `ct_t1_dit` | 2932 |
| avx2 | `t1_dit` | 1280 | 1288 | `ct_t1_dit` | 2894 |
| avx2 | `t1_dit` | 1280 | 6400 | `ct_t1_dit` | 3031 |
| avx2 | `t1_dit` | 1280 | 10240 | `ct_t1_dit` | 3440 |
| avx2 | `t1_dit_log3` | 40 | 40 | `ct_t1_dit_log3` | 88 |
| avx2 | `t1_dit_log3` | 40 | 48 | `ct_t1_dit_log3` | 89 |
| avx2 | `t1_dit_log3` | 40 | 200 | `ct_t1_dit_log3` | 88 |
| avx2 | `t1_dit_log3` | 40 | 320 | `ct_t1_dit_log3` | 89 |
| avx2 | `t1_dit_log3` | 80 | 80 | `ct_t1_dit_log3` | 174 |
| avx2 | `t1_dit_log3` | 80 | 88 | `ct_t1_dit_log3` | 183 |
| avx2 | `t1_dit_log3` | 80 | 400 | `ct_t1_dit_log3` | 174 |
| avx2 | `t1_dit_log3` | 80 | 640 | `ct_t1_dit_log3` | 174 |
| avx2 | `t1_dit_log3` | 160 | 160 | `ct_t1_dit_log3` | 348 |
| avx2 | `t1_dit_log3` | 160 | 168 | `ct_t1_dit_log3` | 346 |
| avx2 | `t1_dit_log3` | 160 | 800 | `ct_t1_dit_log3` | 346 |
| avx2 | `t1_dit_log3` | 160 | 1280 | `ct_t1_dit_log3` | 356 |
| avx2 | `t1_dit_log3` | 320 | 320 | `ct_t1_dit_log3` | 692 |
| avx2 | `t1_dit_log3` | 320 | 328 | `ct_t1_dit_log3` | 696 |
| avx2 | `t1_dit_log3` | 320 | 1600 | `ct_t1_dit_log3` | 700 |
| avx2 | `t1_dit_log3` | 320 | 2560 | `ct_t1_dit_log3` | 762 |
| avx2 | `t1_dit_log3` | 640 | 640 | `ct_t1_dit_log3` | 1409 |
| avx2 | `t1_dit_log3` | 640 | 648 | `ct_t1_dit_log3` | 1409 |
| avx2 | `t1_dit_log3` | 640 | 3200 | `ct_t1_dit_log3` | 1426 |
| avx2 | `t1_dit_log3` | 640 | 5120 | `ct_t1_dit_log3` | 1558 |
| avx2 | `t1_dit_log3` | 1280 | 1280 | `ct_t1_dit_log3` | 3010 |
| avx2 | `t1_dit_log3` | 1280 | 1288 | `ct_t1_dit_log3` | 2917 |
| avx2 | `t1_dit_log3` | 1280 | 6400 | `ct_t1_dit_log3` | 2904 |
| avx2 | `t1_dit_log3` | 1280 | 10240 | `ct_t1_dit_log3` | 3049 |
| avx2 | `t1s_dit` | 40 | 40 | `ct_t1s_dit` | 84 |
| avx2 | `t1s_dit` | 40 | 48 | `ct_t1s_dit` | 84 |
| avx2 | `t1s_dit` | 40 | 200 | `ct_t1s_dit` | 83 |
| avx2 | `t1s_dit` | 40 | 320 | `ct_t1s_dit` | 83 |
| avx2 | `t1s_dit` | 80 | 80 | `ct_t1s_dit` | 161 |
| avx2 | `t1s_dit` | 80 | 88 | `ct_t1s_dit` | 161 |
| avx2 | `t1s_dit` | 80 | 400 | `ct_t1s_dit` | 159 |
| avx2 | `t1s_dit` | 80 | 640 | `ct_t1s_dit` | 163 |
| avx2 | `t1s_dit` | 160 | 160 | `ct_t1s_dit` | 307 |
| avx2 | `t1s_dit` | 160 | 168 | `ct_t1s_dit` | 310 |
| avx2 | `t1s_dit` | 160 | 800 | `ct_t1s_dit` | 303 |
| avx2 | `t1s_dit` | 160 | 1280 | `ct_t1s_dit` | 311 |
| avx2 | `t1s_dit` | 320 | 320 | `ct_t1s_dit` | 616 |
| avx2 | `t1s_dit` | 320 | 328 | `ct_t1s_dit` | 600 |
| avx2 | `t1s_dit` | 320 | 1600 | `ct_t1s_dit` | 610 |
| avx2 | `t1s_dit` | 320 | 2560 | `ct_t1s_dit` | 610 |
| avx2 | `t1s_dit` | 640 | 640 | `ct_t1s_dit` | 1253 |
| avx2 | `t1s_dit` | 640 | 648 | `ct_t1s_dit` | 1258 |
| avx2 | `t1s_dit` | 640 | 3200 | `ct_t1s_dit` | 1216 |
| avx2 | `t1s_dit` | 640 | 5120 | `ct_t1s_dit` | 1250 |
| avx2 | `t1s_dit` | 1280 | 1280 | `ct_t1s_dit` | 2444 |
| avx2 | `t1s_dit` | 1280 | 1288 | `ct_t1s_dit` | 2500 |
| avx2 | `t1s_dit` | 1280 | 6400 | `ct_t1s_dit` | 2631 |
| avx2 | `t1s_dit` | 1280 | 10240 | `ct_t1s_dit` | 2773 |
| avx512 | `t1_dit` | 40 | 40 | `ct_t1_dit` | 62 |
| avx512 | `t1_dit` | 40 | 48 | `ct_t1_dit` | 59 |
| avx512 | `t1_dit` | 40 | 200 | `ct_t1_dit` | 60 |
| avx512 | `t1_dit` | 40 | 320 | `ct_t1_dit` | 63 |
| avx512 | `t1_dit` | 80 | 80 | `ct_t1_dit` | 120 |
| avx512 | `t1_dit` | 80 | 88 | `ct_t1_dit` | 120 |
| avx512 | `t1_dit` | 80 | 400 | `ct_t1_dit` | 120 |
| avx512 | `t1_dit` | 80 | 640 | `ct_t1_dit` | 122 |
| avx512 | `t1_dit` | 160 | 160 | `ct_t1_dit` | 236 |
| avx512 | `t1_dit` | 160 | 168 | `ct_t1_dit` | 239 |
| avx512 | `t1_dit` | 160 | 800 | `ct_t1_dit` | 244 |
| avx512 | `t1_dit` | 160 | 1280 | `ct_t1_dit` | 236 |
| avx512 | `t1_dit` | 320 | 320 | `ct_t1_dit` | 467 |
| avx512 | `t1_dit` | 320 | 328 | `ct_t1_dit` | 473 |
| avx512 | `t1_dit` | 320 | 1600 | `ct_t1_dit` | 482 |
| avx512 | `t1_dit` | 320 | 2560 | `ct_t1_dit` | 543 |
| avx512 | `t1_dit` | 640 | 640 | `ct_t1_dit` | 1332 |
| avx512 | `t1_dit` | 640 | 648 | `ct_t1_dit` | 1323 |
| avx512 | `t1_dit` | 640 | 3200 | `ct_t1_dit` | 1338 |
| avx512 | `t1_dit` | 640 | 5120 | `ct_t1_dit` | 1344 |
| avx512 | `t1_dit` | 1280 | 1280 | `ct_t1_dit` | 2686 |
| avx512 | `t1_dit` | 1280 | 1288 | `ct_t1_dit` | 2623 |
| avx512 | `t1_dit` | 1280 | 6400 | `ct_t1_dit` | 2688 |
| avx512 | `t1_dit` | 1280 | 10240 | `ct_t1_dit` | 2797 |
| avx512 | `t1_dit_log3` | 40 | 40 | `ct_t1_dit_log3` | 66 |
| avx512 | `t1_dit_log3` | 40 | 48 | `ct_t1_dit_log3` | 69 |
| avx512 | `t1_dit_log3` | 40 | 200 | `ct_t1_dit_log3` | 68 |
| avx512 | `t1_dit_log3` | 40 | 320 | `ct_t1_dit_log3` | 68 |
| avx512 | `t1_dit_log3` | 80 | 80 | `ct_t1_dit_log3` | 135 |
| avx512 | `t1_dit_log3` | 80 | 88 | `ct_t1_dit_log3` | 135 |
| avx512 | `t1_dit_log3` | 80 | 400 | `ct_t1_dit_log3` | 134 |
| avx512 | `t1_dit_log3` | 80 | 640 | `ct_t1_dit_log3` | 139 |
| avx512 | `t1_dit_log3` | 160 | 160 | `ct_t1_dit_log3` | 270 |
| avx512 | `t1_dit_log3` | 160 | 168 | `ct_t1_dit_log3` | 268 |
| avx512 | `t1_dit_log3` | 160 | 800 | `ct_t1_dit_log3` | 262 |
| avx512 | `t1_dit_log3` | 160 | 1280 | `ct_t1_dit_log3` | 259 |
| avx512 | `t1_dit_log3` | 320 | 320 | `ct_t1_dit_log3` | 540 |
| avx512 | `t1_dit_log3` | 320 | 328 | `ct_t1_dit_log3` | 522 |
| avx512 | `t1_dit_log3` | 320 | 1600 | `ct_t1_dit_log3` | 526 |
| avx512 | `t1_dit_log3` | 320 | 2560 | `ct_t1_dit_log3` | 540 |
| avx512 | `t1_dit_log3` | 640 | 640 | `ct_t1_dit_log3` | 1143 |
| avx512 | `t1_dit_log3` | 640 | 648 | `ct_t1_dit_log3` | 1125 |
| avx512 | `t1_dit_log3` | 640 | 3200 | `ct_t1_dit_log3` | 1160 |
| avx512 | `t1_dit_log3` | 640 | 5120 | `ct_t1_dit_log3` | 1153 |
| avx512 | `t1_dit_log3` | 1280 | 1280 | `ct_t1_dit_log3` | 2460 |
| avx512 | `t1_dit_log3` | 1280 | 1288 | `ct_t1_dit_log3` | 2475 |
| avx512 | `t1_dit_log3` | 1280 | 6400 | `ct_t1_dit_log3` | 2558 |
| avx512 | `t1_dit_log3` | 1280 | 10240 | `ct_t1_dit_log3` | 2648 |
| avx512 | `t1s_dit` | 40 | 40 | `ct_t1s_dit` | 53 |
| avx512 | `t1s_dit` | 40 | 48 | `ct_t1s_dit` | 52 |
| avx512 | `t1s_dit` | 40 | 200 | `ct_t1s_dit` | 55 |
| avx512 | `t1s_dit` | 40 | 320 | `ct_t1s_dit` | 55 |
| avx512 | `t1s_dit` | 80 | 80 | `ct_t1s_dit` | 110 |
| avx512 | `t1s_dit` | 80 | 88 | `ct_t1s_dit` | 107 |
| avx512 | `t1s_dit` | 80 | 400 | `ct_t1s_dit` | 104 |
| avx512 | `t1s_dit` | 80 | 640 | `ct_t1s_dit` | 104 |
| avx512 | `t1s_dit` | 160 | 160 | `ct_t1s_dit` | 216 |
| avx512 | `t1s_dit` | 160 | 168 | `ct_t1s_dit` | 217 |
| avx512 | `t1s_dit` | 160 | 800 | `ct_t1s_dit` | 211 |
| avx512 | `t1s_dit` | 160 | 1280 | `ct_t1s_dit` | 215 |
| avx512 | `t1s_dit` | 320 | 320 | `ct_t1s_dit` | 424 |
| avx512 | `t1s_dit` | 320 | 328 | `ct_t1s_dit` | 417 |
| avx512 | `t1s_dit` | 320 | 1600 | `ct_t1s_dit` | 413 |
| avx512 | `t1s_dit` | 320 | 2560 | `ct_t1s_dit` | 418 |
| avx512 | `t1s_dit` | 640 | 640 | `ct_t1s_dit` | 857 |
| avx512 | `t1s_dit` | 640 | 648 | `ct_t1s_dit` | 918 |
| avx512 | `t1s_dit` | 640 | 3200 | `ct_t1s_dit` | 852 |
| avx512 | `t1s_dit` | 640 | 5120 | `ct_t1s_dit` | 921 |
| avx512 | `t1s_dit` | 1280 | 1280 | `ct_t1s_dit` | 2322 |
| avx512 | `t1s_dit` | 1280 | 1288 | `ct_t1s_dit` | 2315 |
| avx512 | `t1s_dit` | 1280 | 6400 | `ct_t1s_dit` | 2356 |
| avx512 | `t1s_dit` | 1280 | 10240 | `ct_t1s_dit` | 2364 |