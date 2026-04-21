# VectorFFT R=13 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 123 | 117 | 114 | — | t1s |
| avx2 | 8 | 16 | 118 | 121 | 118 | — | t1s |
| avx2 | 8 | 104 | 119 | 119 | 115 | — | t1s |
| avx2 | 8 | 256 | 126 | 121 | 110 | — | t1s |
| avx2 | 16 | 16 | 242 | 232 | 212 | — | t1s |
| avx2 | 16 | 24 | 239 | 238 | 218 | — | t1s |
| avx2 | 16 | 208 | 233 | 242 | 218 | — | t1s |
| avx2 | 16 | 512 | 409 | 415 | 380 | — | t1s |
| avx2 | 32 | 32 | 480 | 474 | 422 | — | t1s |
| avx2 | 32 | 40 | 486 | 445 | 429 | — | t1s |
| avx2 | 32 | 416 | 474 | 473 | 422 | — | t1s |
| avx2 | 32 | 1024 | 832 | 793 | 796 | — | log3 |
| avx2 | 64 | 64 | 952 | 892 | 857 | — | t1s |
| avx2 | 64 | 72 | 954 | 908 | 873 | — | t1s |
| avx2 | 64 | 832 | 958 | 907 | 858 | — | t1s |
| avx2 | 64 | 2048 | 1936 | 1910 | 1770 | — | t1s |
| avx2 | 128 | 128 | 1915 | 1829 | 1720 | — | t1s |
| avx2 | 128 | 136 | 1959 | 1862 | 1667 | — | t1s |
| avx2 | 128 | 1664 | 1954 | 1868 | 1699 | — | t1s |
| avx2 | 128 | 4096 | 3812 | 3894 | 3564 | — | t1s |
| avx2 | 256 | 256 | 4362 | 3956 | 4010 | — | log3 |
| avx2 | 256 | 264 | 4078 | 3710 | 3336 | — | t1s |
| avx2 | 256 | 3328 | 7080 | 6426 | 5919 | — | t1s |
| avx2 | 256 | 8192 | 8425 | 8165 | 7345 | — | t1s |
| avx512 | 8 | 8 | 61 | 59 | 56 | — | t1s |
| avx512 | 8 | 16 | 61 | 60 | 56 | — | t1s |
| avx512 | 8 | 104 | 60 | 58 | 55 | — | t1s |
| avx512 | 8 | 256 | 63 | 61 | 57 | — | t1s |
| avx512 | 16 | 16 | 111 | 115 | 104 | — | t1s |
| avx512 | 16 | 24 | 113 | 115 | 102 | — | t1s |
| avx512 | 16 | 208 | 109 | 114 | 100 | — | t1s |
| avx512 | 16 | 512 | 219 | 188 | 159 | — | t1s |
| avx512 | 32 | 32 | 216 | 239 | 197 | — | t1s |
| avx512 | 32 | 40 | 210 | 236 | 191 | — | t1s |
| avx512 | 32 | 416 | 219 | 230 | 207 | — | t1s |
| avx512 | 32 | 1024 | 369 | 339 | 384 | — | log3 |
| avx512 | 64 | 64 | 426 | 461 | 374 | — | t1s |
| avx512 | 64 | 72 | 445 | 462 | 414 | — | t1s |
| avx512 | 64 | 832 | 523 | 453 | 398 | — | t1s |
| avx512 | 64 | 2048 | 938 | 742 | 851 | — | log3 |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 123 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 118 |
| avx2 | `t1_dit` | 8 | 104 | `ct_t1_dit` | 119 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 126 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 242 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 239 |
| avx2 | `t1_dit` | 16 | 208 | `ct_t1_dit` | 233 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 409 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 480 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 486 |
| avx2 | `t1_dit` | 32 | 416 | `ct_t1_dit` | 474 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 832 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 952 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 954 |
| avx2 | `t1_dit` | 64 | 832 | `ct_t1_dit` | 958 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1936 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1915 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1959 |
| avx2 | `t1_dit` | 128 | 1664 | `ct_t1_dit` | 1954 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3812 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4362 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4078 |
| avx2 | `t1_dit` | 256 | 3328 | `ct_t1_dit` | 7080 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 8425 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 117 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 121 |
| avx2 | `t1_dit_log3` | 8 | 104 | `ct_t1_dit_log3` | 119 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 121 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 232 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 238 |
| avx2 | `t1_dit_log3` | 16 | 208 | `ct_t1_dit_log3` | 242 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 415 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 474 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 445 |
| avx2 | `t1_dit_log3` | 32 | 416 | `ct_t1_dit_log3` | 473 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 793 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 892 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 908 |
| avx2 | `t1_dit_log3` | 64 | 832 | `ct_t1_dit_log3` | 907 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1910 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1829 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1862 |
| avx2 | `t1_dit_log3` | 128 | 1664 | `ct_t1_dit_log3` | 1868 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3894 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3956 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3710 |
| avx2 | `t1_dit_log3` | 256 | 3328 | `ct_t1_dit_log3` | 6426 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 8165 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 114 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 118 |
| avx2 | `t1s_dit` | 8 | 104 | `ct_t1s_dit` | 115 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 110 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 212 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 218 |
| avx2 | `t1s_dit` | 16 | 208 | `ct_t1s_dit` | 218 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 380 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 422 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 429 |
| avx2 | `t1s_dit` | 32 | 416 | `ct_t1s_dit` | 422 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 796 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 857 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 873 |
| avx2 | `t1s_dit` | 64 | 832 | `ct_t1s_dit` | 858 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1770 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1720 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1667 |
| avx2 | `t1s_dit` | 128 | 1664 | `ct_t1s_dit` | 1699 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 3564 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4010 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 3336 |
| avx2 | `t1s_dit` | 256 | 3328 | `ct_t1s_dit` | 5919 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 7345 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 61 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 61 |
| avx512 | `t1_dit` | 8 | 104 | `ct_t1_dit` | 60 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 63 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 111 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 113 |
| avx512 | `t1_dit` | 16 | 208 | `ct_t1_dit` | 109 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 219 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 216 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 210 |
| avx512 | `t1_dit` | 32 | 416 | `ct_t1_dit` | 219 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 369 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 426 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 445 |
| avx512 | `t1_dit` | 64 | 832 | `ct_t1_dit` | 523 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 938 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 941 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 920 |
| avx512 | `t1_dit` | 128 | 1664 | `ct_t1_dit` | 938 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1582 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 2491 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2214 |
| avx512 | `t1_dit` | 256 | 3328 | `ct_t1_dit` | 2855 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 3381 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 59 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 60 |
| avx512 | `t1_dit_log3` | 8 | 104 | `ct_t1_dit_log3` | 58 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 61 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 115 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 115 |
| avx512 | `t1_dit_log3` | 16 | 208 | `ct_t1_dit_log3` | 114 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 188 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 239 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 236 |
| avx512 | `t1_dit_log3` | 32 | 416 | `ct_t1_dit_log3` | 230 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 339 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 461 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 462 |
| avx512 | `t1_dit_log3` | 64 | 832 | `ct_t1_dit_log3` | 453 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 742 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 949 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 943 |
| avx512 | `t1_dit_log3` | 128 | 1664 | `ct_t1_dit_log3` | 966 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1534 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 2406 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1859 |
| avx512 | `t1_dit_log3` | 256 | 3328 | `ct_t1_dit_log3` | 2095 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 3021 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 56 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 56 |
| avx512 | `t1s_dit` | 8 | 104 | `ct_t1s_dit` | 55 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 57 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 104 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 102 |
| avx512 | `t1s_dit` | 16 | 208 | `ct_t1s_dit` | 100 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 159 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 197 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 191 |
| avx512 | `t1s_dit` | 32 | 416 | `ct_t1s_dit` | 207 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 384 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 374 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 414 |
| avx512 | `t1s_dit` | 64 | 832 | `ct_t1s_dit` | 398 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 851 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 775 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 784 |
| avx512 | `t1s_dit` | 128 | 1664 | `ct_t1s_dit` | 787 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1608 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 2465 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1636 |
| avx512 | `t1s_dit` | 256 | 3328 | `ct_t1s_dit` | 2401 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 3104 |