# VectorFFT R=8 tuning report

Total measurements: **228**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 102 | 106 | 98 | — | t1s |
| avx2 | 64 | 72 | 93 | 107 | 97 | — | flat |
| avx2 | 64 | 512 | 236 | 221 | 246 | — | log3 |
| avx2 | 128 | 128 | 190 | 212 | 167 | — | t1s |
| avx2 | 128 | 136 | 201 | 212 | 177 | — | t1s |
| avx2 | 128 | 1024 | 467 | 442 | 478 | — | log3 |
| avx2 | 256 | 256 | 445 | 416 | — | — | log3 |
| avx2 | 256 | 264 | 448 | 417 | — | — | log3 |
| avx2 | 256 | 2048 | 895 | 1208 | — | — | flat |
| avx2 | 512 | 512 | 1759 | 1731 | — | — | log3 |
| avx2 | 512 | 520 | 1030 | 961 | — | — | log3 |
| avx2 | 512 | 4096 | 1721 | 1800 | — | — | flat |
| avx2 | 1024 | 1024 | 3771 | 3752 | — | — | log3 |
| avx2 | 1024 | 1032 | 1906 | 1952 | — | — | flat |
| avx2 | 1024 | 8192 | 3420 | 2632 | — | — | log3 |
| avx2 | 2048 | 2048 | 7262 | 7444 | — | — | flat |
| avx2 | 2048 | 2056 | 4117 | 3877 | — | — | log3 |
| avx2 | 2048 | 16384 | 15711 | 16562 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 134 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 119 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif_prefetch` | 370 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif_prefetch` | 332 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 261 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif_prefetch` | 790 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 771 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif_prefetch` | 597 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1503 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 2845 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif_prefetch` | 1289 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif_prefetch` | 3442 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 6345 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 2934 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 3420 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 11866 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif_prefetch` | 6737 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 17695 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 102 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 93 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 236 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 190 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 201 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 467 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 445 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 448 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 895 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 1759 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 1030 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 1721 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_prefetch` | 3771 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit_log1` | 1906 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 7408 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 7262 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 4117 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 15711 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 107 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 221 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 212 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 212 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 442 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 416 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 417 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 1208 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 1731 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 961 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 1800 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 3752 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 1952 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 2632 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 7444 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3877 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 16562 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 97 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 246 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 167 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 177 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 478 |