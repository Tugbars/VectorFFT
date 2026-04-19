# VectorFFT R=8 tuning report

Total measurements: **228**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 90 | 106 | 92 | — | flat |
| avx2 | 64 | 72 | 91 | 107 | 93 | — | flat |
| avx2 | 64 | 512 | 233 | 236 | 244 | — | flat |
| avx2 | 128 | 128 | 178 | 208 | 181 | — | flat |
| avx2 | 128 | 136 | 175 | 211 | 167 | — | t1s |
| avx2 | 128 | 1024 | 419 | 431 | 463 | — | flat |
| avx2 | 256 | 256 | 411 | 418 | — | — | flat |
| avx2 | 256 | 264 | 403 | 420 | — | — | flat |
| avx2 | 256 | 2048 | 879 | 876 | — | — | log3 |
| avx2 | 512 | 512 | 1715 | 1703 | — | — | log3 |
| avx2 | 512 | 520 | 1011 | 939 | — | — | log3 |
| avx2 | 512 | 4096 | 1740 | 1742 | — | — | flat |
| avx2 | 1024 | 1024 | 3383 | 3499 | — | — | flat |
| avx2 | 1024 | 1032 | 2039 | 1928 | — | — | log3 |
| avx2 | 1024 | 8192 | 3458 | 5373 | — | — | flat |
| avx2 | 2048 | 2048 | 6850 | 6672 | — | — | log3 |
| avx2 | 2048 | 2056 | 4243 | 3909 | — | — | log3 |
| avx2 | 2048 | 16384 | 15097 | 15453 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 168 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif_prefetch` | 127 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 326 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 231 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif_prefetch` | 250 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 632 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif_prefetch` | 615 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 577 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1374 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 2779 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1238 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 3028 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 6031 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3316 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 3458 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 11355 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 4962 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 17745 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 90 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 91 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 233 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 178 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 175 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_prefetch` | 419 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 411 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 403 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 879 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_prefetch` | 1715 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 1011 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 1740 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 3383 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_prefetch` | 2039 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 6658 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_prefetch` | 6850 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 4243 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 15097 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 107 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 236 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 208 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 211 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 431 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 418 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 420 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 876 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 1703 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 939 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 1742 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 3499 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1928 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 5373 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 6672 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3909 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 15453 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 93 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 244 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 181 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 167 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 463 |