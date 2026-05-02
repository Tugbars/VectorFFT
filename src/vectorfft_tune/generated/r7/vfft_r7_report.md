# VectorFFT R=7 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 56 | 56 | 285 | 276 | 255 | — | t1s |
| avx2 | 56 | 64 | 281 | 274 | 253 | — | t1s |
| avx2 | 56 | 392 | 283 | 274 | 254 | — | t1s |
| avx2 | 56 | 448 | 284 | 275 | 256 | — | t1s |
| avx2 | 112 | 112 | 574 | 544 | 506 | — | t1s |
| avx2 | 112 | 120 | 574 | 579 | 503 | — | t1s |
| avx2 | 112 | 784 | 564 | 556 | 506 | — | t1s |
| avx2 | 112 | 896 | 549 | 545 | 507 | — | t1s |
| avx2 | 224 | 224 | 1129 | 1100 | 1010 | — | t1s |
| avx2 | 224 | 232 | 1135 | 1114 | 995 | — | t1s |
| avx2 | 224 | 1568 | 1141 | 1084 | 1011 | — | t1s |
| avx2 | 224 | 1792 | 1129 | 1108 | 1016 | — | t1s |
| avx2 | 448 | 448 | 2383 | 2333 | 2035 | — | t1s |
| avx2 | 448 | 456 | 2431 | 2273 | 1998 | — | t1s |
| avx2 | 448 | 3136 | 2420 | 2269 | 1995 | — | t1s |
| avx2 | 448 | 3584 | 6006 | 6095 | 6257 | — | flat |
| avx2 | 896 | 896 | 4898 | 4588 | 4091 | — | t1s |
| avx2 | 896 | 904 | 4826 | 4583 | 4111 | — | t1s |
| avx2 | 896 | 6272 | 4930 | 4570 | 4207 | — | t1s |
| avx2 | 896 | 7168 | 12061 | 12006 | 11855 | — | t1s |
| avx2 | 1792 | 1792 | 9660 | 9050 | 8388 | — | t1s |
| avx2 | 1792 | 1800 | 9928 | 9294 | 8386 | — | t1s |
| avx2 | 1792 | 12544 | 9821 | 9426 | 8341 | — | t1s |
| avx2 | 1792 | 14336 | 24089 | 23839 | 23576 | — | t1s |
| avx512 | 56 | 56 | 127 | 164 | 122 | — | t1s |
| avx512 | 56 | 64 | 124 | 158 | 122 | — | t1s |
| avx512 | 56 | 392 | 126 | 162 | 122 | — | t1s |
| avx512 | 56 | 448 | 124 | 156 | 124 | — | flat |
| avx512 | 112 | 112 | 254 | 319 | 248 | — | t1s |
| avx512 | 112 | 120 | 252 | 320 | 251 | — | t1s |
| avx512 | 112 | 784 | 254 | 317 | 251 | — | t1s |
| avx512 | 112 | 896 | 249 | 318 | 247 | — | t1s |
| avx512 | 224 | 224 | 522 | 655 | 501 | — | t1s |
| avx512 | 224 | 232 | 514 | 622 | 500 | — | t1s |
| avx512 | 224 | 1568 | 513 | 629 | 489 | — | t1s |
| avx512 | 224 | 1792 | 534 | 637 | 499 | — | t1s |
| avx512 | 448 | 448 | 1351 | 1291 | 986 | — | t1s |
| avx512 | 448 | 456 | 1406 | 1309 | 1011 | — | t1s |
| avx512 | 448 | 3136 | 1375 | 1294 | 987 | — | t1s |
| avx512 | 448 | 3584 | 2617 | 2049 | 2078 | — | log3 |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 56 | 56 | `ct_t1_dit` | 285 |
| avx2 | `t1_dit` | 56 | 64 | `ct_t1_dit` | 281 |
| avx2 | `t1_dit` | 56 | 392 | `ct_t1_dit` | 283 |
| avx2 | `t1_dit` | 56 | 448 | `ct_t1_dit` | 284 |
| avx2 | `t1_dit` | 112 | 112 | `ct_t1_dit` | 574 |
| avx2 | `t1_dit` | 112 | 120 | `ct_t1_dit` | 574 |
| avx2 | `t1_dit` | 112 | 784 | `ct_t1_dit` | 564 |
| avx2 | `t1_dit` | 112 | 896 | `ct_t1_dit` | 549 |
| avx2 | `t1_dit` | 224 | 224 | `ct_t1_dit` | 1129 |
| avx2 | `t1_dit` | 224 | 232 | `ct_t1_dit` | 1135 |
| avx2 | `t1_dit` | 224 | 1568 | `ct_t1_dit` | 1141 |
| avx2 | `t1_dit` | 224 | 1792 | `ct_t1_dit` | 1129 |
| avx2 | `t1_dit` | 448 | 448 | `ct_t1_dit` | 2383 |
| avx2 | `t1_dit` | 448 | 456 | `ct_t1_dit` | 2431 |
| avx2 | `t1_dit` | 448 | 3136 | `ct_t1_dit` | 2420 |
| avx2 | `t1_dit` | 448 | 3584 | `ct_t1_dit` | 6006 |
| avx2 | `t1_dit` | 896 | 896 | `ct_t1_dit` | 4898 |
| avx2 | `t1_dit` | 896 | 904 | `ct_t1_dit` | 4826 |
| avx2 | `t1_dit` | 896 | 6272 | `ct_t1_dit` | 4930 |
| avx2 | `t1_dit` | 896 | 7168 | `ct_t1_dit` | 12061 |
| avx2 | `t1_dit` | 1792 | 1792 | `ct_t1_dit` | 9660 |
| avx2 | `t1_dit` | 1792 | 1800 | `ct_t1_dit` | 9928 |
| avx2 | `t1_dit` | 1792 | 12544 | `ct_t1_dit` | 9821 |
| avx2 | `t1_dit` | 1792 | 14336 | `ct_t1_dit` | 24089 |
| avx2 | `t1_dit_log3` | 56 | 56 | `ct_t1_dit_log3` | 276 |
| avx2 | `t1_dit_log3` | 56 | 64 | `ct_t1_dit_log3` | 274 |
| avx2 | `t1_dit_log3` | 56 | 392 | `ct_t1_dit_log3` | 274 |
| avx2 | `t1_dit_log3` | 56 | 448 | `ct_t1_dit_log3` | 275 |
| avx2 | `t1_dit_log3` | 112 | 112 | `ct_t1_dit_log3` | 544 |
| avx2 | `t1_dit_log3` | 112 | 120 | `ct_t1_dit_log3` | 579 |
| avx2 | `t1_dit_log3` | 112 | 784 | `ct_t1_dit_log3` | 556 |
| avx2 | `t1_dit_log3` | 112 | 896 | `ct_t1_dit_log3` | 545 |
| avx2 | `t1_dit_log3` | 224 | 224 | `ct_t1_dit_log3` | 1100 |
| avx2 | `t1_dit_log3` | 224 | 232 | `ct_t1_dit_log3` | 1114 |
| avx2 | `t1_dit_log3` | 224 | 1568 | `ct_t1_dit_log3` | 1084 |
| avx2 | `t1_dit_log3` | 224 | 1792 | `ct_t1_dit_log3` | 1108 |
| avx2 | `t1_dit_log3` | 448 | 448 | `ct_t1_dit_log3` | 2333 |
| avx2 | `t1_dit_log3` | 448 | 456 | `ct_t1_dit_log3` | 2273 |
| avx2 | `t1_dit_log3` | 448 | 3136 | `ct_t1_dit_log3` | 2269 |
| avx2 | `t1_dit_log3` | 448 | 3584 | `ct_t1_dit_log3` | 6095 |
| avx2 | `t1_dit_log3` | 896 | 896 | `ct_t1_dit_log3` | 4588 |
| avx2 | `t1_dit_log3` | 896 | 904 | `ct_t1_dit_log3` | 4583 |
| avx2 | `t1_dit_log3` | 896 | 6272 | `ct_t1_dit_log3` | 4570 |
| avx2 | `t1_dit_log3` | 896 | 7168 | `ct_t1_dit_log3` | 12006 |
| avx2 | `t1_dit_log3` | 1792 | 1792 | `ct_t1_dit_log3` | 9050 |
| avx2 | `t1_dit_log3` | 1792 | 1800 | `ct_t1_dit_log3` | 9294 |
| avx2 | `t1_dit_log3` | 1792 | 12544 | `ct_t1_dit_log3` | 9426 |
| avx2 | `t1_dit_log3` | 1792 | 14336 | `ct_t1_dit_log3` | 23839 |
| avx2 | `t1s_dit` | 56 | 56 | `ct_t1s_dit` | 255 |
| avx2 | `t1s_dit` | 56 | 64 | `ct_t1s_dit` | 253 |
| avx2 | `t1s_dit` | 56 | 392 | `ct_t1s_dit` | 254 |
| avx2 | `t1s_dit` | 56 | 448 | `ct_t1s_dit` | 256 |
| avx2 | `t1s_dit` | 112 | 112 | `ct_t1s_dit` | 506 |
| avx2 | `t1s_dit` | 112 | 120 | `ct_t1s_dit` | 503 |
| avx2 | `t1s_dit` | 112 | 784 | `ct_t1s_dit` | 506 |
| avx2 | `t1s_dit` | 112 | 896 | `ct_t1s_dit` | 507 |
| avx2 | `t1s_dit` | 224 | 224 | `ct_t1s_dit` | 1010 |
| avx2 | `t1s_dit` | 224 | 232 | `ct_t1s_dit` | 995 |
| avx2 | `t1s_dit` | 224 | 1568 | `ct_t1s_dit` | 1011 |
| avx2 | `t1s_dit` | 224 | 1792 | `ct_t1s_dit` | 1016 |
| avx2 | `t1s_dit` | 448 | 448 | `ct_t1s_dit` | 2035 |
| avx2 | `t1s_dit` | 448 | 456 | `ct_t1s_dit` | 1998 |
| avx2 | `t1s_dit` | 448 | 3136 | `ct_t1s_dit` | 1995 |
| avx2 | `t1s_dit` | 448 | 3584 | `ct_t1s_dit` | 6257 |
| avx2 | `t1s_dit` | 896 | 896 | `ct_t1s_dit` | 4091 |
| avx2 | `t1s_dit` | 896 | 904 | `ct_t1s_dit` | 4111 |
| avx2 | `t1s_dit` | 896 | 6272 | `ct_t1s_dit` | 4207 |
| avx2 | `t1s_dit` | 896 | 7168 | `ct_t1s_dit` | 11855 |
| avx2 | `t1s_dit` | 1792 | 1792 | `ct_t1s_dit` | 8388 |
| avx2 | `t1s_dit` | 1792 | 1800 | `ct_t1s_dit` | 8386 |
| avx2 | `t1s_dit` | 1792 | 12544 | `ct_t1s_dit` | 8341 |
| avx2 | `t1s_dit` | 1792 | 14336 | `ct_t1s_dit` | 23576 |
| avx512 | `t1_dit` | 56 | 56 | `ct_t1_dit` | 127 |
| avx512 | `t1_dit` | 56 | 64 | `ct_t1_dit` | 124 |
| avx512 | `t1_dit` | 56 | 392 | `ct_t1_dit` | 126 |
| avx512 | `t1_dit` | 56 | 448 | `ct_t1_dit` | 124 |
| avx512 | `t1_dit` | 112 | 112 | `ct_t1_dit` | 254 |
| avx512 | `t1_dit` | 112 | 120 | `ct_t1_dit` | 252 |
| avx512 | `t1_dit` | 112 | 784 | `ct_t1_dit` | 254 |
| avx512 | `t1_dit` | 112 | 896 | `ct_t1_dit` | 249 |
| avx512 | `t1_dit` | 224 | 224 | `ct_t1_dit` | 522 |
| avx512 | `t1_dit` | 224 | 232 | `ct_t1_dit` | 514 |
| avx512 | `t1_dit` | 224 | 1568 | `ct_t1_dit` | 513 |
| avx512 | `t1_dit` | 224 | 1792 | `ct_t1_dit` | 534 |
| avx512 | `t1_dit` | 448 | 448 | `ct_t1_dit` | 1351 |
| avx512 | `t1_dit` | 448 | 456 | `ct_t1_dit` | 1406 |
| avx512 | `t1_dit` | 448 | 3136 | `ct_t1_dit` | 1375 |
| avx512 | `t1_dit` | 448 | 3584 | `ct_t1_dit` | 2617 |
| avx512 | `t1_dit` | 896 | 896 | `ct_t1_dit` | 2753 |
| avx512 | `t1_dit` | 896 | 904 | `ct_t1_dit` | 2832 |
| avx512 | `t1_dit` | 896 | 6272 | `ct_t1_dit` | 2759 |
| avx512 | `t1_dit` | 896 | 7168 | `ct_t1_dit` | 4976 |
| avx512 | `t1_dit` | 1792 | 1792 | `ct_t1_dit` | 5638 |
| avx512 | `t1_dit` | 1792 | 1800 | `ct_t1_dit` | 5443 |
| avx512 | `t1_dit` | 1792 | 12544 | `ct_t1_dit` | 5866 |
| avx512 | `t1_dit` | 1792 | 14336 | `ct_t1_dit` | 9723 |
| avx512 | `t1_dit_log3` | 56 | 56 | `ct_t1_dit_log3` | 164 |
| avx512 | `t1_dit_log3` | 56 | 64 | `ct_t1_dit_log3` | 158 |
| avx512 | `t1_dit_log3` | 56 | 392 | `ct_t1_dit_log3` | 162 |
| avx512 | `t1_dit_log3` | 56 | 448 | `ct_t1_dit_log3` | 156 |
| avx512 | `t1_dit_log3` | 112 | 112 | `ct_t1_dit_log3` | 319 |
| avx512 | `t1_dit_log3` | 112 | 120 | `ct_t1_dit_log3` | 320 |
| avx512 | `t1_dit_log3` | 112 | 784 | `ct_t1_dit_log3` | 317 |
| avx512 | `t1_dit_log3` | 112 | 896 | `ct_t1_dit_log3` | 318 |
| avx512 | `t1_dit_log3` | 224 | 224 | `ct_t1_dit_log3` | 655 |
| avx512 | `t1_dit_log3` | 224 | 232 | `ct_t1_dit_log3` | 622 |
| avx512 | `t1_dit_log3` | 224 | 1568 | `ct_t1_dit_log3` | 629 |
| avx512 | `t1_dit_log3` | 224 | 1792 | `ct_t1_dit_log3` | 637 |
| avx512 | `t1_dit_log3` | 448 | 448 | `ct_t1_dit_log3` | 1291 |
| avx512 | `t1_dit_log3` | 448 | 456 | `ct_t1_dit_log3` | 1309 |
| avx512 | `t1_dit_log3` | 448 | 3136 | `ct_t1_dit_log3` | 1294 |
| avx512 | `t1_dit_log3` | 448 | 3584 | `ct_t1_dit_log3` | 2049 |
| avx512 | `t1_dit_log3` | 896 | 896 | `ct_t1_dit_log3` | 2632 |
| avx512 | `t1_dit_log3` | 896 | 904 | `ct_t1_dit_log3` | 2553 |
| avx512 | `t1_dit_log3` | 896 | 6272 | `ct_t1_dit_log3` | 2600 |
| avx512 | `t1_dit_log3` | 896 | 7168 | `ct_t1_dit_log3` | 4351 |
| avx512 | `t1_dit_log3` | 1792 | 1792 | `ct_t1_dit_log3` | 5113 |
| avx512 | `t1_dit_log3` | 1792 | 1800 | `ct_t1_dit_log3` | 5159 |
| avx512 | `t1_dit_log3` | 1792 | 12544 | `ct_t1_dit_log3` | 5145 |
| avx512 | `t1_dit_log3` | 1792 | 14336 | `ct_t1_dit_log3` | 8156 |
| avx512 | `t1s_dit` | 56 | 56 | `ct_t1s_dit` | 122 |
| avx512 | `t1s_dit` | 56 | 64 | `ct_t1s_dit` | 122 |
| avx512 | `t1s_dit` | 56 | 392 | `ct_t1s_dit` | 122 |
| avx512 | `t1s_dit` | 56 | 448 | `ct_t1s_dit` | 124 |
| avx512 | `t1s_dit` | 112 | 112 | `ct_t1s_dit` | 248 |
| avx512 | `t1s_dit` | 112 | 120 | `ct_t1s_dit` | 251 |
| avx512 | `t1s_dit` | 112 | 784 | `ct_t1s_dit` | 251 |
| avx512 | `t1s_dit` | 112 | 896 | `ct_t1s_dit` | 247 |
| avx512 | `t1s_dit` | 224 | 224 | `ct_t1s_dit` | 501 |
| avx512 | `t1s_dit` | 224 | 232 | `ct_t1s_dit` | 500 |
| avx512 | `t1s_dit` | 224 | 1568 | `ct_t1s_dit` | 489 |
| avx512 | `t1s_dit` | 224 | 1792 | `ct_t1s_dit` | 499 |
| avx512 | `t1s_dit` | 448 | 448 | `ct_t1s_dit` | 986 |
| avx512 | `t1s_dit` | 448 | 456 | `ct_t1s_dit` | 1011 |
| avx512 | `t1s_dit` | 448 | 3136 | `ct_t1s_dit` | 987 |
| avx512 | `t1s_dit` | 448 | 3584 | `ct_t1s_dit` | 2078 |
| avx512 | `t1s_dit` | 896 | 896 | `ct_t1s_dit` | 2369 |
| avx512 | `t1s_dit` | 896 | 904 | `ct_t1s_dit` | 2348 |
| avx512 | `t1s_dit` | 896 | 6272 | `ct_t1s_dit` | 2448 |
| avx512 | `t1s_dit` | 896 | 7168 | `ct_t1s_dit` | 4354 |
| avx512 | `t1s_dit` | 1792 | 1792 | `ct_t1s_dit` | 4712 |
| avx512 | `t1s_dit` | 1792 | 1800 | `ct_t1s_dit` | 4635 |
| avx512 | `t1s_dit` | 1792 | 12544 | `ct_t1s_dit` | 4838 |
| avx512 | `t1s_dit` | 1792 | 14336 | `ct_t1s_dit` | 8199 |