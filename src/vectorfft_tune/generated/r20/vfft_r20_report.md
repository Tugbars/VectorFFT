# VectorFFT R=20 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 60 | 68 | 59 | — | t1s |
| avx2 | 8 | 16 | 61 | 67 | 70 | — | flat |
| avx2 | 8 | 160 | 58 | 68 | 59 | — | flat |
| avx2 | 8 | 256 | 59 | 69 | 59 | — | flat |
| avx2 | 16 | 16 | 115 | 130 | 113 | — | t1s |
| avx2 | 16 | 24 | 110 | 131 | 128 | — | flat |
| avx2 | 16 | 320 | 112 | 135 | 106 | — | t1s |
| avx2 | 16 | 512 | 279 | 357 | 296 | — | flat |
| avx2 | 32 | 32 | 211 | 256 | 217 | — | flat |
| avx2 | 32 | 40 | 212 | 257 | 215 | — | flat |
| avx2 | 32 | 640 | 224 | 261 | 218 | — | t1s |
| avx2 | 32 | 1024 | 517 | 620 | 565 | — | flat |
| avx2 | 64 | 64 | 449 | 505 | 418 | — | t1s |
| avx2 | 64 | 72 | 442 | 517 | 383 | — | t1s |
| avx2 | 64 | 1280 | 931 | 960 | 973 | — | flat |
| avx2 | 64 | 2048 | 1060 | 1041 | 1093 | — | log3 |
| avx2 | 128 | 128 | 1108 | 1023 | 815 | — | t1s |
| avx2 | 128 | 136 | 1076 | 1038 | 820 | — | t1s |
| avx2 | 128 | 2560 | 2118 | 2146 | 2178 | — | flat |
| avx2 | 128 | 4096 | 2300 | 2402 | 1941 | — | t1s |
| avx2 | 256 | 256 | 4119 | 4407 | 4409 | — | flat |
| avx2 | 256 | 264 | 2318 | 2133 | 1667 | — | t1s |
| avx2 | 256 | 5120 | 5155 | 4842 | 4178 | — | t1s |
| avx2 | 256 | 8192 | 5864 | 5611 | 5234 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 60 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 61 |
| avx2 | `t1_dit` | 8 | 160 | `ct_t1_dit` | 58 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 59 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 115 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 110 |
| avx2 | `t1_dit` | 16 | 320 | `ct_t1_dit` | 112 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 279 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 211 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 212 |
| avx2 | `t1_dit` | 32 | 640 | `ct_t1_dit` | 224 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 517 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 449 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 442 |
| avx2 | `t1_dit` | 64 | 1280 | `ct_t1_dit` | 931 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1060 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1108 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1076 |
| avx2 | `t1_dit` | 128 | 2560 | `ct_t1_dit` | 2118 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2300 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4119 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2318 |
| avx2 | `t1_dit` | 256 | 5120 | `ct_t1_dit` | 5155 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 5864 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 68 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 67 |
| avx2 | `t1_dit_log3` | 8 | 160 | `ct_t1_dit_log3` | 68 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 69 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 130 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 131 |
| avx2 | `t1_dit_log3` | 16 | 320 | `ct_t1_dit_log3` | 135 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 357 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 256 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 257 |
| avx2 | `t1_dit_log3` | 32 | 640 | `ct_t1_dit_log3` | 261 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 620 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 505 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 517 |
| avx2 | `t1_dit_log3` | 64 | 1280 | `ct_t1_dit_log3` | 960 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1041 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1023 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1038 |
| avx2 | `t1_dit_log3` | 128 | 2560 | `ct_t1_dit_log3` | 2146 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2402 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4407 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2133 |
| avx2 | `t1_dit_log3` | 256 | 5120 | `ct_t1_dit_log3` | 4842 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 5611 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 59 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 70 |
| avx2 | `t1s_dit` | 8 | 160 | `ct_t1s_dit` | 59 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 59 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 113 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 128 |
| avx2 | `t1s_dit` | 16 | 320 | `ct_t1s_dit` | 106 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 296 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 217 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 215 |
| avx2 | `t1s_dit` | 32 | 640 | `ct_t1s_dit` | 218 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 565 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 418 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 383 |
| avx2 | `t1s_dit` | 64 | 1280 | `ct_t1s_dit` | 973 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1093 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 815 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 820 |
| avx2 | `t1s_dit` | 128 | 2560 | `ct_t1s_dit` | 2178 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1941 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4409 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1667 |
| avx2 | `t1s_dit` | 256 | 5120 | `ct_t1s_dit` | 4178 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5234 |