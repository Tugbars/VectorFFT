# VectorFFT R=11 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 104 | 99 | 92 | — | t1s |
| avx2 | 8 | 16 | 100 | 98 | 92 | — | t1s |
| avx2 | 8 | 88 | 101 | 99 | 92 | — | t1s |
| avx2 | 8 | 256 | 101 | 97 | 93 | — | t1s |
| avx2 | 16 | 16 | 198 | 195 | 181 | — | t1s |
| avx2 | 16 | 24 | 195 | 192 | 182 | — | t1s |
| avx2 | 16 | 176 | 200 | 192 | 181 | — | t1s |
| avx2 | 16 | 512 | 303 | 288 | 301 | — | log3 |
| avx2 | 32 | 32 | 397 | 377 | 358 | — | t1s |
| avx2 | 32 | 40 | 391 | 378 | 361 | — | t1s |
| avx2 | 32 | 352 | 395 | 376 | 352 | — | t1s |
| avx2 | 32 | 1024 | 516 | 530 | 526 | — | flat |
| avx2 | 64 | 64 | 781 | 757 | 697 | — | t1s |
| avx2 | 64 | 72 | 768 | 753 | 747 | — | t1s |
| avx2 | 64 | 704 | 781 | 749 | 711 | — | t1s |
| avx2 | 64 | 2048 | 1201 | 1111 | 1247 | — | log3 |
| avx2 | 128 | 128 | 1559 | 1525 | 1387 | — | t1s |
| avx2 | 128 | 136 | 1546 | 1521 | 1397 | — | t1s |
| avx2 | 128 | 1408 | 1561 | 1558 | 1400 | — | t1s |
| avx2 | 128 | 4096 | 2438 | 2232 | 2608 | — | log3 |
| avx2 | 256 | 256 | 3526 | 3418 | 2791 | — | t1s |
| avx2 | 256 | 264 | 3302 | 3431 | 2798 | — | t1s |
| avx2 | 256 | 2816 | 3998 | 3552 | 3364 | — | t1s |
| avx2 | 256 | 8192 | 5034 | 4957 | 5193 | — | log3 |
| avx512 | 8 | 8 | 56 | 52 | 56 | — | log3 |
| avx512 | 8 | 16 | 56 | 54 | 55 | — | log3 |
| avx512 | 8 | 88 | 57 | 52 | 56 | — | log3 |
| avx512 | 8 | 256 | 60 | 58 | 55 | — | t1s |
| avx512 | 16 | 16 | 103 | 99 | 98 | — | t1s |
| avx512 | 16 | 24 | 111 | 101 | 98 | — | t1s |
| avx512 | 16 | 176 | 104 | 102 | 99 | — | t1s |
| avx512 | 16 | 512 | 150 | 140 | 148 | — | log3 |
| avx512 | 32 | 32 | 203 | 198 | 186 | — | t1s |
| avx512 | 32 | 40 | 206 | 201 | 188 | — | t1s |
| avx512 | 32 | 352 | 202 | 199 | 190 | — | t1s |
| avx512 | 32 | 1024 | 288 | 265 | 278 | — | log3 |
| avx512 | 64 | 64 | 396 | 394 | 376 | — | t1s |
| avx512 | 64 | 72 | 404 | 394 | 370 | — | t1s |
| avx512 | 64 | 704 | 406 | 392 | 374 | — | t1s |
| avx512 | 64 | 2048 | 645 | 538 | 591 | — | log3 |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 104 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 100 |
| avx2 | `t1_dit` | 8 | 88 | `ct_t1_dit` | 101 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 101 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 198 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 195 |
| avx2 | `t1_dit` | 16 | 176 | `ct_t1_dit` | 200 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 303 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 397 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 391 |
| avx2 | `t1_dit` | 32 | 352 | `ct_t1_dit` | 395 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 516 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 781 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 768 |
| avx2 | `t1_dit` | 64 | 704 | `ct_t1_dit` | 781 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1201 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1559 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1546 |
| avx2 | `t1_dit` | 128 | 1408 | `ct_t1_dit` | 1561 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2438 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 3526 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3302 |
| avx2 | `t1_dit` | 256 | 2816 | `ct_t1_dit` | 3998 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 5034 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 99 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 8 | 88 | `ct_t1_dit_log3` | 99 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 97 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 195 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 192 |
| avx2 | `t1_dit_log3` | 16 | 176 | `ct_t1_dit_log3` | 192 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 288 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 377 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 378 |
| avx2 | `t1_dit_log3` | 32 | 352 | `ct_t1_dit_log3` | 376 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 530 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 757 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 753 |
| avx2 | `t1_dit_log3` | 64 | 704 | `ct_t1_dit_log3` | 749 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1111 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1525 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1521 |
| avx2 | `t1_dit_log3` | 128 | 1408 | `ct_t1_dit_log3` | 1558 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2232 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3418 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3431 |
| avx2 | `t1_dit_log3` | 256 | 2816 | `ct_t1_dit_log3` | 3552 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 4957 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 8 | 88 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 93 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 181 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 182 |
| avx2 | `t1s_dit` | 16 | 176 | `ct_t1s_dit` | 181 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 301 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 358 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 361 |
| avx2 | `t1s_dit` | 32 | 352 | `ct_t1s_dit` | 352 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 526 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 697 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 747 |
| avx2 | `t1s_dit` | 64 | 704 | `ct_t1s_dit` | 711 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1247 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1387 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1397 |
| avx2 | `t1s_dit` | 128 | 1408 | `ct_t1s_dit` | 1400 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2608 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 2791 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2798 |
| avx2 | `t1s_dit` | 256 | 2816 | `ct_t1s_dit` | 3364 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5193 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 56 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 56 |
| avx512 | `t1_dit` | 8 | 88 | `ct_t1_dit` | 57 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 60 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 103 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 111 |
| avx512 | `t1_dit` | 16 | 176 | `ct_t1_dit` | 104 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 150 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 203 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 206 |
| avx512 | `t1_dit` | 32 | 352 | `ct_t1_dit` | 202 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 288 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 396 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 404 |
| avx512 | `t1_dit` | 64 | 704 | `ct_t1_dit` | 406 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 645 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 807 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 814 |
| avx512 | `t1_dit` | 128 | 1408 | `ct_t1_dit` | 825 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1311 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1751 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1699 |
| avx512 | `t1_dit` | 256 | 2816 | `ct_t1_dit` | 1980 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2813 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 52 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 54 |
| avx512 | `t1_dit_log3` | 8 | 88 | `ct_t1_dit_log3` | 52 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 58 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 99 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 101 |
| avx512 | `t1_dit_log3` | 16 | 176 | `ct_t1_dit_log3` | 102 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 140 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 198 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 201 |
| avx512 | `t1_dit_log3` | 32 | 352 | `ct_t1_dit_log3` | 199 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 265 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 394 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 394 |
| avx512 | `t1_dit_log3` | 64 | 704 | `ct_t1_dit_log3` | 392 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 538 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 784 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 786 |
| avx512 | `t1_dit_log3` | 128 | 1408 | `ct_t1_dit_log3` | 801 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1136 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1803 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1625 |
| avx512 | `t1_dit_log3` | 256 | 2816 | `ct_t1_dit_log3` | 1667 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 2452 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 56 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 55 |
| avx512 | `t1s_dit` | 8 | 88 | `ct_t1s_dit` | 56 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 55 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 98 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 98 |
| avx512 | `t1s_dit` | 16 | 176 | `ct_t1s_dit` | 99 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 148 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 186 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 188 |
| avx512 | `t1s_dit` | 32 | 352 | `ct_t1s_dit` | 190 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 278 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 376 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 370 |
| avx512 | `t1s_dit` | 64 | 704 | `ct_t1s_dit` | 374 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 591 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 722 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 733 |
| avx512 | `t1s_dit` | 128 | 1408 | `ct_t1s_dit` | 734 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1187 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1468 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1454 |
| avx512 | `t1s_dit` | 256 | 2816 | `ct_t1s_dit` | 1583 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2531 |