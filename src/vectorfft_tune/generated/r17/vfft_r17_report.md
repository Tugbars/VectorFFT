# VectorFFT R=17 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 204 | 193 | 174 | — | t1s |
| avx2 | 8 | 16 | 195 | 192 | 180 | — | t1s |
| avx2 | 8 | 136 | 197 | 191 | 174 | — | t1s |
| avx2 | 8 | 256 | 204 | 194 | 178 | — | t1s |
| avx2 | 16 | 16 | 380 | 389 | 390 | — | flat |
| avx2 | 16 | 24 | 383 | 415 | 338 | — | t1s |
| avx2 | 16 | 272 | 378 | 462 | 337 | — | t1s |
| avx2 | 16 | 512 | 557 | 556 | 581 | — | log3 |
| avx2 | 32 | 32 | 754 | 745 | 671 | — | t1s |
| avx2 | 32 | 40 | 765 | 747 | 688 | — | t1s |
| avx2 | 32 | 544 | 773 | 750 | 679 | — | t1s |
| avx2 | 32 | 1024 | 1339 | 1293 | 1344 | — | log3 |
| avx2 | 64 | 64 | 1533 | 1454 | 1319 | — | t1s |
| avx2 | 64 | 72 | 1541 | 1503 | 1348 | — | t1s |
| avx2 | 64 | 1088 | 1487 | 1465 | 1376 | — | t1s |
| avx2 | 64 | 2048 | 2742 | 3351 | 2663 | — | t1s |
| avx2 | 128 | 128 | 3199 | 2905 | 2788 | — | t1s |
| avx2 | 128 | 136 | 3306 | 2941 | 2765 | — | t1s |
| avx2 | 128 | 2176 | 3589 | 2967 | 2821 | — | t1s |
| avx2 | 128 | 4096 | 5617 | 5253 | 5652 | — | log3 |
| avx2 | 256 | 256 | 7813 | 6642 | 7312 | — | log3 |
| avx2 | 256 | 264 | 6487 | 6033 | 5449 | — | t1s |
| avx2 | 256 | 4352 | 9783 | 8898 | 9512 | — | log3 |
| avx2 | 256 | 8192 | 11354 | 10406 | 11118 | — | log3 |
| avx512 | 8 | 8 | 114 | 94 | 100 | — | log3 |
| avx512 | 8 | 16 | 111 | 95 | 98 | — | log3 |
| avx512 | 8 | 136 | 108 | 94 | 98 | — | log3 |
| avx512 | 8 | 256 | 115 | 97 | 98 | — | log3 |
| avx512 | 16 | 16 | 219 | 187 | 176 | — | t1s |
| avx512 | 16 | 24 | 217 | 185 | 173 | — | t1s |
| avx512 | 16 | 272 | 216 | 186 | 174 | — | t1s |
| avx512 | 16 | 512 | 344 | 274 | 265 | — | t1s |
| avx512 | 32 | 32 | 410 | 363 | 335 | — | t1s |
| avx512 | 32 | 40 | 402 | 358 | 338 | — | t1s |
| avx512 | 32 | 544 | 417 | 364 | 359 | — | t1s |
| avx512 | 32 | 1024 | 671 | 650 | 588 | — | t1s |
| avx512 | 64 | 64 | 792 | 731 | 644 | — | t1s |
| avx512 | 64 | 72 | 820 | 695 | 645 | — | t1s |
| avx512 | 64 | 1088 | 806 | 706 | 655 | — | t1s |
| avx512 | 64 | 2048 | 1362 | 1328 | 1169 | — | t1s |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 204 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 195 |
| avx2 | `t1_dit` | 8 | 136 | `ct_t1_dit` | 197 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 204 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 380 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 383 |
| avx2 | `t1_dit` | 16 | 272 | `ct_t1_dit` | 378 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 557 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 754 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 765 |
| avx2 | `t1_dit` | 32 | 544 | `ct_t1_dit` | 773 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1339 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1533 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1541 |
| avx2 | `t1_dit` | 64 | 1088 | `ct_t1_dit` | 1487 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 2742 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3199 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 3306 |
| avx2 | `t1_dit` | 128 | 2176 | `ct_t1_dit` | 3589 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 5617 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 7813 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 6487 |
| avx2 | `t1_dit` | 256 | 4352 | `ct_t1_dit` | 9783 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 11354 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 193 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 192 |
| avx2 | `t1_dit_log3` | 8 | 136 | `ct_t1_dit_log3` | 191 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 389 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 415 |
| avx2 | `t1_dit_log3` | 16 | 272 | `ct_t1_dit_log3` | 462 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 556 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 745 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 747 |
| avx2 | `t1_dit_log3` | 32 | 544 | `ct_t1_dit_log3` | 750 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 1293 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1454 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1503 |
| avx2 | `t1_dit_log3` | 64 | 1088 | `ct_t1_dit_log3` | 1465 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 3351 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 2905 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2941 |
| avx2 | `t1_dit_log3` | 128 | 2176 | `ct_t1_dit_log3` | 2967 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 5253 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6642 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 6033 |
| avx2 | `t1_dit_log3` | 256 | 4352 | `ct_t1_dit_log3` | 8898 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 10406 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 174 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 180 |
| avx2 | `t1s_dit` | 8 | 136 | `ct_t1s_dit` | 174 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 178 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 390 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 338 |
| avx2 | `t1s_dit` | 16 | 272 | `ct_t1s_dit` | 337 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 581 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 671 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 688 |
| avx2 | `t1s_dit` | 32 | 544 | `ct_t1s_dit` | 679 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 1344 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1319 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1348 |
| avx2 | `t1s_dit` | 64 | 1088 | `ct_t1s_dit` | 1376 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 2663 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2788 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2765 |
| avx2 | `t1s_dit` | 128 | 2176 | `ct_t1s_dit` | 2821 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 5652 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 7312 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 5449 |
| avx2 | `t1s_dit` | 256 | 4352 | `ct_t1s_dit` | 9512 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 11118 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 114 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 111 |
| avx512 | `t1_dit` | 8 | 136 | `ct_t1_dit` | 108 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 115 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 219 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 217 |
| avx512 | `t1_dit` | 16 | 272 | `ct_t1_dit` | 216 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 344 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 410 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 402 |
| avx512 | `t1_dit` | 32 | 544 | `ct_t1_dit` | 417 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 671 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 792 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 820 |
| avx512 | `t1_dit` | 64 | 1088 | `ct_t1_dit` | 806 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1362 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1938 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1851 |
| avx512 | `t1_dit` | 128 | 2176 | `ct_t1_dit` | 1942 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2776 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5033 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3746 |
| avx512 | `t1_dit` | 256 | 4352 | `ct_t1_dit` | 4378 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 5500 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 94 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 95 |
| avx512 | `t1_dit_log3` | 8 | 136 | `ct_t1_dit_log3` | 94 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 97 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 187 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 185 |
| avx512 | `t1_dit_log3` | 16 | 272 | `ct_t1_dit_log3` | 186 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 274 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 363 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 358 |
| avx512 | `t1_dit_log3` | 32 | 544 | `ct_t1_dit_log3` | 364 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 650 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 731 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 695 |
| avx512 | `t1_dit_log3` | 64 | 1088 | `ct_t1_dit_log3` | 706 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1328 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1469 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1403 |
| avx512 | `t1_dit_log3` | 128 | 2176 | `ct_t1_dit_log3` | 1487 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2534 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3363 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2838 |
| avx512 | `t1_dit_log3` | 256 | 4352 | `ct_t1_dit_log3` | 3968 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 4942 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 100 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 98 |
| avx512 | `t1s_dit` | 8 | 136 | `ct_t1s_dit` | 98 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 98 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 176 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 173 |
| avx512 | `t1s_dit` | 16 | 272 | `ct_t1s_dit` | 174 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 265 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 335 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 338 |
| avx512 | `t1s_dit` | 32 | 544 | `ct_t1s_dit` | 359 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 588 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 644 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 645 |
| avx512 | `t1s_dit` | 64 | 1088 | `ct_t1s_dit` | 655 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1169 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1337 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1348 |
| avx512 | `t1s_dit` | 128 | 2176 | `ct_t1s_dit` | 1376 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2407 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 3201 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2737 |
| avx512 | `t1s_dit` | 256 | 4352 | `ct_t1s_dit` | 4000 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 4712 |