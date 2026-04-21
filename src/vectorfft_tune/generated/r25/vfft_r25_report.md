# VectorFFT R=25 tuning report

Total measurements: **256**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 79 | 108 | 98 | — | flat |
| avx2 | 8 | 16 | 80 | 96 | 90 | — | flat |
| avx2 | 8 | 200 | 115 | 98 | 92 | — | t1s |
| avx2 | 8 | 256 | 179 | 143 | 131 | — | t1s |
| avx2 | 16 | 16 | 153 | 189 | 163 | — | flat |
| avx2 | 16 | 24 | 156 | 183 | 157 | — | flat |
| avx2 | 16 | 400 | 156 | 194 | 178 | — | flat |
| avx2 | 16 | 512 | 453 | 365 | 337 | — | t1s |
| avx2 | 32 | 32 | 306 | 375 | 324 | — | flat |
| avx2 | 32 | 40 | 306 | 377 | 308 | — | flat |
| avx2 | 32 | 800 | 317 | 360 | 315 | — | t1s |
| avx2 | 32 | 1024 | 786 | 715 | 854 | — | log3 |
| avx2 | 64 | 64 | 629 | 694 | 636 | — | flat |
| avx2 | 64 | 72 | 806 | 716 | 609 | — | t1s |
| avx2 | 64 | 1600 | 673 | 812 | 623 | — | t1s |
| avx2 | 64 | 2048 | 1421 | 1497 | 1292 | — | t1s |
| avx2 | 128 | 128 | 1593 | 1503 | 1319 | — | t1s |
| avx2 | 128 | 136 | 1659 | 1529 | 1297 | — | t1s |
| avx2 | 128 | 3200 | 1879 | 1720 | 1368 | — | t1s |
| avx2 | 128 | 4096 | 3041 | 3095 | 2758 | — | t1s |
| avx2 | 256 | 256 | 5539 | 6908 | 5222 | — | t1s |
| avx2 | 256 | 264 | 3569 | 3126 | 2691 | — | t1s |
| avx2 | 256 | 6400 | 6894 | 6239 | 6108 | — | t1s |
| avx2 | 256 | 8192 | 7634 | 12619 | 6807 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 79 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 80 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 115 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 179 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 153 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 156 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 156 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 453 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 306 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 306 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 317 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_buf_dit_tile32_temporal` | 786 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 629 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 806 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 673 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1421 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1593 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1659 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 1879 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3041 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5539 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3569 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_buf_dit_tile32_temporal` | 6894 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 7634 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 108 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 96 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 143 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 189 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 183 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 365 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 375 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 377 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 360 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 715 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 694 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 716 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 812 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1497 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1503 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1529 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 1720 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3095 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6908 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3126 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 6239 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 12619 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 90 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 131 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 163 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 157 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 178 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 337 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 324 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 308 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 315 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 854 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 636 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 609 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 623 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1292 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1319 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1297 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 1368 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2758 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 5222 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2691 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 6108 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 6807 |