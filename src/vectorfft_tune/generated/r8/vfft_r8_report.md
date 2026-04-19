# VectorFFT R=8 tuning report

Total measurements: **228**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 91 | 106 | 98 | — | flat |
| avx2 | 64 | 72 | 90 | 106 | 87 | — | t1s |
| avx2 | 64 | 512 | 236 | 231 | 261 | — | log3 |
| avx2 | 128 | 128 | 172 | 210 | 167 | — | t1s |
| avx2 | 128 | 136 | 182 | 210 | 182 | — | flat |
| avx2 | 128 | 1024 | 417 | 449 | 674 | — | flat |
| avx2 | 256 | 256 | 413 | 415 | — | — | flat |
| avx2 | 256 | 264 | 416 | 415 | — | — | log3 |
| avx2 | 256 | 2048 | 862 | 877 | — | — | flat |
| avx2 | 512 | 512 | 1753 | 1747 | — | — | log3 |
| avx2 | 512 | 520 | 988 | 930 | — | — | log3 |
| avx2 | 512 | 4096 | 1701 | 3391 | — | — | flat |
| avx2 | 1024 | 1024 | 3601 | 3358 | — | — | log3 |
| avx2 | 1024 | 1032 | 2044 | 1859 | — | — | log3 |
| avx2 | 1024 | 8192 | 2801 | 2458 | — | — | log3 |
| avx2 | 2048 | 2048 | 7045 | 6881 | — | — | log3 |
| avx2 | 2048 | 2056 | 4200 | 3851 | — | — | log3 |
| avx2 | 2048 | 16384 | 15103 | 15731 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 168 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 168 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 407 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 233 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif_prefetch` | 248 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 664 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif_prefetch` | 611 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif_prefetch` | 601 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1379 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 2742 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1247 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 2835 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 6067 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 2499 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 2801 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 11508 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 5958 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 15994 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 91 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 90 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 236 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 172 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 182 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 417 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 413 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 416 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 862 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 1753 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 988 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 1701 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 3601 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_prefetch` | 2044 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 6968 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 7045 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 4200 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 15103 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 231 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 210 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 210 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 449 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 415 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 415 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 877 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 1747 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 930 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 3391 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 3358 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1859 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 2458 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 6881 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3851 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 15731 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 87 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 261 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 167 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 182 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 674 |