# VectorFFT R=8 tuning report

Total measurements: **228**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 92 | 121 | 99 | — | flat |
| avx2 | 64 | 72 | 99 | 122 | 100 | — | flat |
| avx2 | 64 | 512 | 229 | 233 | 294 | — | flat |
| avx2 | 128 | 128 | 190 | 227 | 174 | — | t1s |
| avx2 | 128 | 136 | 188 | 230 | 181 | — | t1s |
| avx2 | 128 | 1024 | 441 | 440 | 542 | — | log3 |
| avx2 | 256 | 256 | 429 | 472 | — | — | flat |
| avx2 | 256 | 264 | 435 | 467 | — | — | flat |
| avx2 | 256 | 2048 | 925 | 900 | — | — | log3 |
| avx2 | 512 | 512 | 1772 | 1816 | — | — | flat |
| avx2 | 512 | 520 | 1047 | 1070 | — | — | flat |
| avx2 | 512 | 4096 | 1820 | 2005 | — | — | flat |
| avx2 | 1024 | 1024 | 3850 | 3680 | — | — | log3 |
| avx2 | 1024 | 1032 | 2159 | 2089 | — | — | log3 |
| avx2 | 1024 | 8192 | 3682 | 5025 | — | — | flat |
| avx2 | 2048 | 2048 | 7232 | 6781 | — | — | log3 |
| avx2 | 2048 | 2056 | 4104 | 4533 | — | — | flat |
| avx2 | 2048 | 16384 | 16413 | 17715 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 134 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif_prefetch` | 136 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 440 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 247 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif_prefetch` | 332 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 889 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 690 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif_prefetch` | 640 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1545 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 3399 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1460 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif_prefetch` | 3227 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 6700 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3667 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 3682 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 12834 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif_prefetch` | 7655 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 19233 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 92 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 99 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 229 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit_log1` | 190 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 188 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 441 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 429 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 435 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 925 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_prefetch` | 1772 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 1047 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 1820 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 3850 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 2159 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 7772 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_prefetch` | 7232 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 4104 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 16413 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 121 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 122 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 233 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 227 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 230 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 440 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 472 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 467 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 900 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 1816 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1070 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 2005 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 3680 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 2089 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 5025 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 6781 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 4533 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 17715 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 99 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 294 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 174 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 181 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 542 |