# VectorFFT R=17 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 75 | 84 | 79 | — | flat |
| avx2 | 8 | 16 | 77 | 83 | 79 | — | flat |
| avx2 | 8 | 136 | 75 | 119 | 79 | — | flat |
| avx2 | 8 | 256 | 76 | 84 | 80 | — | flat |
| avx2 | 16 | 16 | 146 | 159 | 148 | — | flat |
| avx2 | 16 | 24 | 148 | 164 | 149 | — | flat |
| avx2 | 16 | 272 | 149 | 162 | 180 | — | flat |
| avx2 | 16 | 512 | 311 | 301 | 361 | — | log3 |
| avx2 | 32 | 32 | 289 | 365 | 285 | — | t1s |
| avx2 | 32 | 40 | 289 | 311 | 284 | — | t1s |
| avx2 | 32 | 544 | 299 | 343 | 347 | — | flat |
| avx2 | 32 | 1024 | 590 | 607 | 823 | — | flat |
| avx2 | 64 | 64 | 577 | 621 | 559 | — | t1s |
| avx2 | 64 | 72 | 567 | 616 | 563 | — | t1s |
| avx2 | 64 | 1088 | 597 | 648 | 891 | — | flat |
| avx2 | 64 | 2048 | 1179 | 1190 | 1096 | — | t1s |
| avx2 | 128 | 128 | 1329 | 1228 | 1978 | — | log3 |
| avx2 | 128 | 136 | 1637 | 1229 | 1940 | — | log3 |
| avx2 | 128 | 2176 | 1393 | 1459 | 1170 | — | t1s |
| avx2 | 128 | 4096 | 2531 | 2553 | 2245 | — | t1s |
| avx2 | 256 | 256 | 4676 | 4415 | 5530 | — | log3 |
| avx2 | 256 | 264 | 2746 | 2809 | 2638 | — | t1s |
| avx2 | 256 | 4352 | 2822 | 3826 | 2440 | — | t1s |
| avx2 | 256 | 8192 | 6112 | 5960 | 5305 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 75 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 77 |
| avx2 | `t1_dit` | 8 | 136 | `ct_t1_dit` | 75 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 148 |
| avx2 | `t1_dit` | 16 | 272 | `ct_t1_dit` | 149 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 311 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 289 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 289 |
| avx2 | `t1_dit` | 32 | 544 | `ct_t1_dit` | 299 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 590 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 577 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 567 |
| avx2 | `t1_dit` | 64 | 1088 | `ct_t1_dit` | 597 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1179 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1329 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1637 |
| avx2 | `t1_dit` | 128 | 2176 | `ct_t1_dit` | 1393 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2531 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4676 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2746 |
| avx2 | `t1_dit` | 256 | 4352 | `ct_t1_dit` | 2822 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 6112 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 83 |
| avx2 | `t1_dit_log3` | 8 | 136 | `ct_t1_dit_log3` | 119 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 159 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 16 | 272 | `ct_t1_dit_log3` | 162 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 301 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 365 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 311 |
| avx2 | `t1_dit_log3` | 32 | 544 | `ct_t1_dit_log3` | 343 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 607 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 621 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 616 |
| avx2 | `t1_dit_log3` | 64 | 1088 | `ct_t1_dit_log3` | 648 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1190 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1228 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1229 |
| avx2 | `t1_dit_log3` | 128 | 2176 | `ct_t1_dit_log3` | 1459 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2553 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4415 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2809 |
| avx2 | `t1_dit_log3` | 256 | 4352 | `ct_t1_dit_log3` | 3826 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 5960 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 79 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 79 |
| avx2 | `t1s_dit` | 8 | 136 | `ct_t1s_dit` | 79 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 80 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 148 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 16 | 272 | `ct_t1s_dit` | 180 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 361 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 285 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 284 |
| avx2 | `t1s_dit` | 32 | 544 | `ct_t1s_dit` | 347 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 823 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 559 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 563 |
| avx2 | `t1s_dit` | 64 | 1088 | `ct_t1s_dit` | 891 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1096 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1978 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1940 |
| avx2 | `t1s_dit` | 128 | 2176 | `ct_t1s_dit` | 1170 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2245 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 5530 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2638 |
| avx2 | `t1s_dit` | 256 | 4352 | `ct_t1s_dit` | 2440 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5305 |