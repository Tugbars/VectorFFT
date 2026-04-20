# VectorFFT R=25 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 316 | 283 | 279 | — | t1s |
| avx2 | 8 | 16 | 318 | 287 | 280 | — | t1s |
| avx2 | 8 | 200 | 325 | 283 | 275 | — | t1s |
| avx2 | 8 | 256 | 497 | 453 | 482 | — | log3 |
| avx2 | 16 | 16 | 620 | 559 | 531 | — | t1s |
| avx2 | 16 | 24 | 608 | 556 | 558 | — | log3 |
| avx2 | 16 | 400 | 615 | 556 | 529 | — | t1s |
| avx2 | 16 | 512 | 1108 | 1057 | 1053 | — | t1s |
| avx2 | 32 | 32 | 1209 | 1074 | 1027 | — | t1s |
| avx2 | 32 | 40 | 1203 | 1072 | 1032 | — | t1s |
| avx2 | 32 | 800 | 1236 | 1108 | 1049 | — | t1s |
| avx2 | 32 | 1024 | 2373 | 2777 | 2362 | — | t1s |
| avx2 | 64 | 64 | 2566 | 2179 | 2041 | — | t1s |
| avx2 | 64 | 72 | 2564 | 2203 | 2160 | — | t1s |
| avx2 | 64 | 1600 | 2863 | 2312 | 2084 | — | t1s |
| avx2 | 64 | 2048 | 4912 | 5303 | 4758 | — | t1s |
| avx2 | 128 | 128 | 5325 | 4619 | 4469 | — | t1s |
| avx2 | 128 | 136 | 5218 | 4526 | 4330 | — | t1s |
| avx2 | 128 | 3200 | 7483 | 7075 | 6345 | — | t1s |
| avx2 | 128 | 4096 | 9962 | 11334 | 9523 | — | t1s |
| avx2 | 256 | 256 | 16208 | 14001 | 13731 | — | t1s |
| avx2 | 256 | 264 | 11207 | 8912 | 8461 | — | t1s |
| avx2 | 256 | 6400 | 19227 | 17799 | 16700 | — | t1s |
| avx2 | 256 | 8192 | 20521 | 21146 | 18871 | — | t1s |
| avx512 | 8 | 8 | 206 | 166 | 176 | — | log3 |
| avx512 | 8 | 16 | 208 | 172 | 179 | — | log3 |
| avx512 | 8 | 200 | 207 | 176 | 178 | — | log3 |
| avx512 | 8 | 256 | 299 | 269 | 261 | — | t1s |
| avx512 | 16 | 16 | 377 | 341 | 335 | — | t1s |
| avx512 | 16 | 24 | 382 | 329 | 338 | — | log3 |
| avx512 | 16 | 400 | 381 | 349 | 336 | — | t1s |
| avx512 | 16 | 512 | 630 | 542 | 537 | — | t1s |
| avx512 | 32 | 32 | 733 | 649 | 630 | — | t1s |
| avx512 | 32 | 40 | 738 | 646 | 640 | — | t1s |
| avx512 | 32 | 800 | 724 | 655 | 644 | — | t1s |
| avx512 | 32 | 1024 | 1336 | 1105 | 1102 | — | t1s |
| avx512 | 64 | 64 | 1557 | 1280 | 1236 | — | t1s |
| avx512 | 64 | 72 | 1575 | 1285 | 1295 | — | log3 |
| avx512 | 64 | 1600 | 1683 | 1325 | 1264 | — | t1s |
| avx512 | 64 | 2048 | 2707 | 2122 | 2108 | — | t1s |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 316 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 318 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 325 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 497 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 620 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 608 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 615 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 1108 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 1209 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 1203 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 1236 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 2373 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 2566 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2564 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 2863 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 4912 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 5325 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 5218 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 7483 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 9962 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 16208 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 11207 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_dit` | 19227 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 20521 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 283 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 287 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 283 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 453 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 559 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 556 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 556 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 1057 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 1074 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 1072 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 1108 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 2777 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2179 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2203 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 2312 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 5303 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 4619 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4526 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 7075 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 11334 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 14001 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 8912 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 17799 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 21146 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 279 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 280 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 275 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 482 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 531 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 558 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 529 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 1053 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 1027 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 1032 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 1049 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 2362 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 2041 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 2160 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 2084 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 4758 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 4469 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 4330 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 6345 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 9523 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 13731 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 8461 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 16700 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 18871 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 206 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 208 |
| avx512 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 207 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 299 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 377 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 382 |
| avx512 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 381 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 630 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 733 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 738 |
| avx512 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 724 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1336 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1557 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1575 |
| avx512 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 1683 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 2707 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3324 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 3169 |
| avx512 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 3915 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 5435 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 8613 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 6478 |
| avx512 | `t1_dit` | 256 | 6400 | `ct_t1_dit` | 9879 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 11157 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 166 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 172 |
| avx512 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 176 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 269 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 341 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 329 |
| avx512 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 349 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 542 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 649 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 646 |
| avx512 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 655 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 1105 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1280 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1285 |
| avx512 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 1325 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 2122 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 2702 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2675 |
| avx512 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 3203 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 4163 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 7179 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 5144 |
| avx512 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 7518 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 8405 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 176 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 179 |
| avx512 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 178 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 261 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 335 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 338 |
| avx512 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 336 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 537 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 630 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 640 |
| avx512 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 644 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 1102 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1236 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1295 |
| avx512 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 1264 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 2108 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2793 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2596 |
| avx512 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 3438 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 4189 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 7071 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 5100 |
| avx512 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 7545 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 8400 |