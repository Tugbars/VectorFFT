# VectorFFT R=25 tuning report

Total measurements: **256**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 101 | 91 | 87 | — | t1s |
| avx2 | 8 | 16 | 100 | 92 | 87 | — | t1s |
| avx2 | 8 | 200 | 80 | 92 | 87 | — | flat |
| avx2 | 8 | 256 | 122 | 107 | 158 | — | log3 |
| avx2 | 16 | 16 | 193 | 185 | 158 | — | t1s |
| avx2 | 16 | 24 | 153 | 177 | 157 | — | flat |
| avx2 | 16 | 400 | 152 | 180 | 169 | — | flat |
| avx2 | 16 | 512 | 426 | 343 | 339 | — | t1s |
| avx2 | 32 | 32 | 292 | 346 | 298 | — | flat |
| avx2 | 32 | 40 | 291 | 349 | 298 | — | flat |
| avx2 | 32 | 800 | 296 | 387 | 301 | — | flat |
| avx2 | 32 | 1024 | 798 | 873 | 705 | — | t1s |
| avx2 | 64 | 64 | 596 | 699 | 599 | — | flat |
| avx2 | 64 | 72 | 616 | 747 | 621 | — | flat |
| avx2 | 64 | 1600 | 667 | 754 | 622 | — | t1s |
| avx2 | 64 | 2048 | 1644 | 1389 | 1309 | — | t1s |
| avx2 | 128 | 128 | 1551 | 1464 | 1574 | — | log3 |
| avx2 | 128 | 136 | 1613 | 1517 | 1215 | — | t1s |
| avx2 | 128 | 3200 | 1651 | 1768 | 1368 | — | t1s |
| avx2 | 128 | 4096 | 3176 | 2791 | 2606 | — | t1s |
| avx2 | 256 | 256 | 5299 | 6079 | 4834 | — | t1s |
| avx2 | 256 | 264 | 4106 | 3105 | 2618 | — | t1s |
| avx2 | 256 | 6400 | 5650 | 7156 | 5548 | — | t1s |
| avx2 | 256 | 8192 | 7194 | 8201 | 7087 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 101 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 100 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 80 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 122 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 193 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 153 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 152 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 426 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 292 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 291 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 296 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_buf_dit_tile32_temporal` | 798 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 596 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 616 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 667 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_buf_dit_tile32_temporal` | 1644 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1551 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1613 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 1651 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_buf_dit_tile32_temporal` | 3176 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 5299 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4106 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_buf_dit_tile64_temporal` | 5650 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_buf_dit_tile32_temporal` | 7194 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 91 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 92 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 92 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 107 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 185 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 177 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 180 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 343 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 346 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 349 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 387 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 873 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 699 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 747 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 754 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1389 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1464 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1517 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 1768 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2791 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6079 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3105 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 7156 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 8201 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 87 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 87 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 87 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 158 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 158 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 157 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 169 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 339 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 298 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 298 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 301 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 705 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 599 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 621 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 622 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1309 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1574 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1215 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 1368 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2606 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4834 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2618 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 5548 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 7087 |