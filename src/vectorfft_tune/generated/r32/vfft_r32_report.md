# VectorFFT R=32 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 837 | 865 | 697 | — | t1s |
| avx2 | 64 | 72 | 1053 | 882 | 689 | — | t1s |
| avx2 | 64 | 512 | 2071 | 1924 | 1776 | — | t1s |
| avx2 | 128 | 128 | 2486 | 1998 | 2239 | — | log3 |
| avx2 | 128 | 136 | 2382 | 1847 | 1439 | — | t1s |
| avx2 | 128 | 1024 | 4399 | 3860 | 3662 | — | t1s |
| avx2 | 256 | 256 | 8215 | 7965 | — | — | log3 |
| avx2 | 256 | 264 | 4284 | 5093 | — | — | flat |
| avx2 | 256 | 2048 | 8669 | 7854 | — | — | log3 |
| avx2 | 512 | 512 | 15158 | 18204 | — | — | flat |
| avx2 | 512 | 520 | 9948 | 7957 | — | — | log3 |
| avx2 | 512 | 4096 | 17841 | 21692 | — | — | flat |
| avx2 | 1024 | 1024 | 30031 | 30406 | — | — | flat |
| avx2 | 1024 | 1032 | 19047 | 15508 | — | — | log3 |
| avx2 | 1024 | 8192 | 39476 | 43726 | — | — | flat |
| avx2 | 2048 | 2048 | 68263 | 62426 | — | — | log3 |
| avx2 | 2048 | 2056 | 44852 | 31212 | — | — | log3 |
| avx2 | 2048 | 16384 | 109665 | 125396 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 2153 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 1917 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2348 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 4749 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 3940 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 4614 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 8106 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 8061 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 8669 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 17764 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 15680 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile256_temporal` | 17841 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 35679 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 31509 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 43332 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 68263 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 62838 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 109665 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1686 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1103 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2484 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 3692 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 3670 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 5711 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 10942 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 7480 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 11706 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 19460 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 10995 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 27716 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 38383 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 22198 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 59996 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 80068 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 50203 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 142416 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 837 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1053 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2071 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2486 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2382 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4399 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 8215 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4284 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 9493 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 15158 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 9948 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 22119 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 30031 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 19047 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 39476 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 76296 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 44852 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 138630 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 865 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 882 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1924 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1998 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1847 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3860 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 7965 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 5093 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 7854 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 18204 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 7957 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 21692 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 30406 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 15508 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 43726 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 62426 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 31212 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 125396 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 697 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 689 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1776 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2239 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1439 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3662 |