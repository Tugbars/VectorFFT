# VectorFFT R=25 tuning report

Total measurements: **256**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 80 | 96 | 89 | — | flat |
| avx2 | 8 | 16 | 83 | 94 | 96 | — | flat |
| avx2 | 8 | 200 | 82 | 95 | 90 | — | flat |
| avx2 | 8 | 256 | 140 | 162 | 169 | — | flat |
| avx2 | 16 | 16 | 154 | 181 | 162 | — | flat |
| avx2 | 16 | 24 | 155 | 181 | 162 | — | flat |
| avx2 | 16 | 400 | 160 | 183 | 180 | — | flat |
| avx2 | 16 | 512 | 440 | 344 | 354 | — | log3 |
| avx2 | 32 | 32 | 301 | 360 | 314 | — | flat |
| avx2 | 32 | 40 | 380 | 403 | 304 | — | t1s |
| avx2 | 32 | 800 | 308 | 381 | 353 | — | flat |
| avx2 | 32 | 1024 | 778 | 914 | 697 | — | t1s |
| avx2 | 64 | 64 | 811 | 726 | 617 | — | t1s |
| avx2 | 64 | 72 | 850 | 711 | 628 | — | t1s |
| avx2 | 64 | 1600 | 908 | 889 | 641 | — | t1s |
| avx2 | 64 | 2048 | 1661 | 1543 | 1359 | — | t1s |
| avx2 | 128 | 128 | 1797 | 1603 | 1240 | — | t1s |
| avx2 | 128 | 136 | 1970 | 1689 | 1288 | — | t1s |
| avx2 | 128 | 3200 | 1860 | 1881 | 1355 | — | t1s |
| avx2 | 128 | 4096 | 3131 | 3022 | 2730 | — | t1s |
| avx2 | 256 | 256 | 5492 | 6230 | 4916 | — | t1s |
| avx2 | 256 | 264 | 3463 | 3527 | 2690 | — | t1s |
| avx2 | 256 | 6400 | 5971 | 6026 | 6071 | — | flat |
| avx2 | 256 | 8192 | 7004 | 7166 | 7353 | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 80 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 83 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 82 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 140 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 154 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 155 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 160 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 440 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 301 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 380 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 308 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_buf_dit_tile32_temporal` | 778 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 811 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 850 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 908 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_buf_dit_tile32_temporal` | 1661 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1797 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1970 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 1860 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_buf_dit_tile32_temporal` | 3131 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5492 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3463 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_buf_dit_tile64_temporal` | 5971 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_buf_dit_tile32_temporal` | 7004 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 96 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 94 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 95 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 162 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 181 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 181 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 183 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 344 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 360 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 403 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 381 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 914 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 726 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 711 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 889 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1543 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1603 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1689 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 1881 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3022 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6230 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3527 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 6026 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 7166 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 89 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 96 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 90 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 169 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 162 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 162 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 180 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 354 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 314 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 304 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 353 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 697 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 617 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 628 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 641 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1359 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1240 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1288 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 1355 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2730 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4916 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2690 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 6071 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 7353 |