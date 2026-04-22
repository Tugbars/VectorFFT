# VectorFFT R=12 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 30 | 40 | 31 | — | flat |
| avx2 | 8 | 16 | 33 | 41 | 30 | — | t1s |
| avx2 | 8 | 96 | 30 | 40 | 30 | — | t1s |
| avx2 | 8 | 256 | 30 | 40 | 31 | — | flat |
| avx2 | 16 | 16 | 56 | 78 | 54 | — | t1s |
| avx2 | 16 | 24 | 60 | 77 | 55 | — | t1s |
| avx2 | 16 | 192 | 56 | 78 | 55 | — | t1s |
| avx2 | 16 | 512 | 200 | 214 | 173 | — | t1s |
| avx2 | 32 | 32 | 108 | 155 | 106 | — | t1s |
| avx2 | 32 | 40 | 108 | 156 | 106 | — | t1s |
| avx2 | 32 | 384 | 112 | 153 | 109 | — | t1s |
| avx2 | 32 | 1024 | 345 | 315 | 393 | — | log3 |
| avx2 | 64 | 64 | 213 | 306 | 211 | — | t1s |
| avx2 | 64 | 72 | 223 | 293 | 208 | — | t1s |
| avx2 | 64 | 768 | 459 | 313 | 213 | — | t1s |
| avx2 | 64 | 2048 | 736 | 768 | 719 | — | t1s |
| avx2 | 128 | 128 | 455 | 616 | 405 | — | t1s |
| avx2 | 128 | 136 | 445 | 603 | 405 | — | t1s |
| avx2 | 128 | 1536 | 1423 | 1273 | 1404 | — | log3 |
| avx2 | 128 | 4096 | 1581 | 1410 | 1481 | — | log3 |
| avx2 | 256 | 256 | 1154 | 1299 | 806 | — | t1s |
| avx2 | 256 | 264 | 1061 | 1226 | 805 | — | t1s |
| avx2 | 256 | 3072 | 2802 | 2700 | 2718 | — | log3 |
| avx2 | 256 | 8192 | 2508 | 2102 | 1917 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 33 |
| avx2 | `t1_dit` | 8 | 96 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 56 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 60 |
| avx2 | `t1_dit` | 16 | 192 | `ct_t1_dit` | 56 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 200 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 108 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 108 |
| avx2 | `t1_dit` | 32 | 384 | `ct_t1_dit` | 112 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 345 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 213 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 223 |
| avx2 | `t1_dit` | 64 | 768 | `ct_t1_dit` | 459 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 736 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 455 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 445 |
| avx2 | `t1_dit` | 128 | 1536 | `ct_t1_dit` | 1423 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1581 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1154 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1061 |
| avx2 | `t1_dit` | 256 | 3072 | `ct_t1_dit` | 2802 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2508 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 41 |
| avx2 | `t1_dit_log3` | 8 | 96 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 78 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 77 |
| avx2 | `t1_dit_log3` | 16 | 192 | `ct_t1_dit_log3` | 78 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 214 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 155 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 156 |
| avx2 | `t1_dit_log3` | 32 | 384 | `ct_t1_dit_log3` | 153 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 315 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 306 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 293 |
| avx2 | `t1_dit_log3` | 64 | 768 | `ct_t1_dit_log3` | 313 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 768 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 616 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 603 |
| avx2 | `t1_dit_log3` | 128 | 1536 | `ct_t1_dit_log3` | 1273 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1410 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1299 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1226 |
| avx2 | `t1_dit_log3` | 256 | 3072 | `ct_t1_dit_log3` | 2700 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 2102 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 8 | 96 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 54 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 55 |
| avx2 | `t1s_dit` | 16 | 192 | `ct_t1s_dit` | 55 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 173 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 106 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 106 |
| avx2 | `t1s_dit` | 32 | 384 | `ct_t1s_dit` | 109 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 393 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 211 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 208 |
| avx2 | `t1s_dit` | 64 | 768 | `ct_t1s_dit` | 213 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 719 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 405 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 405 |
| avx2 | `t1s_dit` | 128 | 1536 | `ct_t1s_dit` | 1404 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1481 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 806 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 805 |
| avx2 | `t1s_dit` | 256 | 3072 | `ct_t1s_dit` | 2718 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 1917 |