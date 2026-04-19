# VectorFFT R=32 tuning report

Total measurements: **240**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 2211 | 1933 | 1688 | — | t1s |
| avx2 | 64 | 72 | 2190 | 1939 | 1701 | — | t1s |
| avx2 | 64 | 512 | 5298 | 5182 | 4915 | — | t1s |
| avx2 | 128 | 128 | 4659 | 4079 | 3801 | — | t1s |
| avx2 | 128 | 136 | 4561 | 4020 | 3469 | — | t1s |
| avx2 | 128 | 1024 | 10638 | 9729 | 9788 | — | log3 |
| avx2 | 256 | 256 | 16601 | 14733 | — | — | log3 |
| avx2 | 256 | 264 | 9856 | 8142 | — | — | log3 |
| avx2 | 256 | 2048 | 22282 | 20179 | — | — | log3 |
| avx2 | 512 | 512 | 45829 | 40566 | — | — | log3 |
| avx2 | 512 | 520 | 22239 | 16066 | — | — | log3 |
| avx2 | 512 | 4096 | 44221 | 40045 | — | — | log3 |
| avx2 | 1024 | 1024 | 89967 | 78033 | — | — | log3 |
| avx2 | 1024 | 1032 | 46078 | 33290 | — | — | log3 |
| avx2 | 1024 | 8192 | 90527 | 80672 | — | — | log3 |
| avx2 | 2048 | 2048 | 197106 | 157915 | — | — | log3 |
| avx2 | 2048 | 2056 | 101502 | 70771 | — | — | log3 |
| avx2 | 2048 | 16384 | 209278 | 164560 | — | — | log3 |
| avx512 | 64 | 64 | 1452 | 1038 | 935 | — | t1s |
| avx512 | 64 | 72 | 1454 | 1051 | 963 | — | t1s |
| avx512 | 64 | 512 | 2529 | 2286 | 2526 | — | log3 |
| avx512 | 128 | 128 | 3583 | 3083 | 3256 | — | log3 |
| avx512 | 128 | 136 | 2970 | 2157 | 2006 | — | t1s |
| avx512 | 128 | 1024 | 5062 | 4367 | 4677 | — | log3 |
| avx512 | 256 | 256 | 8438 | 7119 | — | — | log3 |
| avx512 | 256 | 264 | 6198 | 4577 | — | — | log3 |
| avx512 | 256 | 2048 | 11294 | 9550 | — | — | log3 |
| avx512 | 512 | 512 | 20979 | 17659 | — | — | log3 |
| avx512 | 512 | 520 | 14391 | 9235 | — | — | log3 |
| avx512 | 512 | 4096 | 20896 | 19014 | — | — | log3 |
| avx512 | 1024 | 1024 | 41630 | 37014 | — | — | log3 |
| avx512 | 1024 | 1032 | 28636 | 19574 | — | — | log3 |
| avx512 | 1024 | 8192 | 39794 | 37316 | — | — | log3 |
| avx512 | 2048 | 2048 | 95051 | 76122 | — | — | log3 |
| avx512 | 2048 | 2056 | 65738 | 47301 | — | — | log3 |
| avx512 | 2048 | 16384 | 93342 | 77506 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 2211 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 2197 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 5962 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4896 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 4741 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12514 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 19031 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 10318 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 25049 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 48592 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 22378 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 48450 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 100260 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 45776 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 102451 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 213161 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 100450 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 225095 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 2262 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2190 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5298 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 4659 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 4561 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 10638 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 16601 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 9856 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 22282 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 45829 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 22239 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 44221 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 89967 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 46078 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 90527 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 197106 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 101502 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 209278 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1933 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1939 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 5182 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 4079 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4020 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 9729 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 14733 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 8142 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 20179 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 40566 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 16066 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 40045 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 78033 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 33290 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 80672 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 157915 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 70771 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 164560 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1688 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1701 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 4915 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3801 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3469 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 9788 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1464 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1476 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 3051 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4804 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 2995 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 5892 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 10327 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 6192 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 11988 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 24772 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 14427 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 23867 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 51595 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 29298 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 51284 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 117889 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 71335 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 121978 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1452 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1454 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2529 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3583 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2970 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 5062 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 8438 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 6198 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 11294 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 20979 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 14391 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 20896 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 41630 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 28636 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 39794 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 95051 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 65738 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 93342 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1038 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1051 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 2286 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3083 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2157 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 4367 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 7119 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 4577 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 9550 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 17659 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 9235 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 19014 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 37014 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 19574 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 37316 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 76122 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 47301 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 77506 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 935 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 963 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 2526 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3256 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2006 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 4677 |