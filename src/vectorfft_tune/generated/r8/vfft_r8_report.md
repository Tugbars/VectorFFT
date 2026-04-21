# VectorFFT R=8 tuning report

Total measurements: **228**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 94 | 113 | 103 | — | flat |
| avx2 | 64 | 72 | 99 | 114 | 97 | — | t1s |
| avx2 | 64 | 512 | 233 | 245 | 265 | — | flat |
| avx2 | 128 | 128 | 189 | 245 | 204 | — | flat |
| avx2 | 128 | 136 | 192 | 211 | 200 | — | flat |
| avx2 | 128 | 1024 | 440 | 518 | 475 | — | flat |
| avx2 | 256 | 256 | 450 | 464 | — | — | flat |
| avx2 | 256 | 264 | 432 | 429 | — | — | log3 |
| avx2 | 256 | 2048 | 910 | 922 | — | — | flat |
| avx2 | 512 | 512 | 1846 | 2001 | — | — | flat |
| avx2 | 512 | 520 | 1015 | 1051 | — | — | flat |
| avx2 | 512 | 4096 | 1794 | 1748 | — | — | log3 |
| avx2 | 1024 | 1024 | 3715 | 3756 | — | — | flat |
| avx2 | 1024 | 1032 | 2053 | 2051 | — | — | log3 |
| avx2 | 1024 | 8192 | 3185 | 5347 | — | — | flat |
| avx2 | 2048 | 2048 | 7627 | 7524 | — | — | log3 |
| avx2 | 2048 | 2056 | 4326 | 3838 | — | — | log3 |
| avx2 | 2048 | 16384 | 15677 | 16079 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 123 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif_prefetch` | 130 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif_prefetch` | 371 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 351 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 253 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 684 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 646 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif_prefetch` | 658 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1503 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 2840 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1687 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 3530 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 6172 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif_prefetch` | 2798 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 3185 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 13400 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif_prefetch` | 5524 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 17755 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 94 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 99 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 233 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 189 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 192 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 440 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 450 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 432 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 910 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 1846 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 1015 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 1794 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 3715 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 2053 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 7049 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 7627 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 4326 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_prefetch` | 15677 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 113 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 114 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 245 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 245 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 211 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 518 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 464 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 429 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 922 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 2001 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1051 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 1748 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 3756 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 2051 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 5347 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 7524 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3838 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 16079 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 103 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 97 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 265 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 204 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 200 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 475 |