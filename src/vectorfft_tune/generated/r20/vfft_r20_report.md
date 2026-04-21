# VectorFFT R=20 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 92 | 91 | 62 | — | t1s |
| avx2 | 8 | 16 | 58 | 82 | 71 | — | flat |
| avx2 | 8 | 160 | 61 | 77 | 62 | — | flat |
| avx2 | 8 | 256 | 63 | 77 | 70 | — | flat |
| avx2 | 16 | 16 | 120 | 149 | 121 | — | flat |
| avx2 | 16 | 24 | 123 | 130 | 113 | — | t1s |
| avx2 | 16 | 320 | 129 | 137 | 112 | — | t1s |
| avx2 | 16 | 512 | 301 | 287 | 309 | — | log3 |
| avx2 | 32 | 32 | 230 | 261 | 243 | — | flat |
| avx2 | 32 | 40 | 224 | 256 | 222 | — | t1s |
| avx2 | 32 | 640 | 338 | 267 | 229 | — | t1s |
| avx2 | 32 | 1024 | 569 | 606 | 562 | — | t1s |
| avx2 | 64 | 64 | 489 | 518 | 388 | — | t1s |
| avx2 | 64 | 72 | 469 | 505 | 392 | — | t1s |
| avx2 | 64 | 1280 | 978 | 1030 | 1092 | — | flat |
| avx2 | 64 | 2048 | 1155 | 1142 | 1191 | — | log3 |
| avx2 | 128 | 128 | 1200 | 1006 | 777 | — | t1s |
| avx2 | 128 | 136 | 1115 | 1060 | 808 | — | t1s |
| avx2 | 128 | 2560 | 2294 | 2049 | 2335 | — | log3 |
| avx2 | 128 | 4096 | 2720 | 2500 | 1967 | — | t1s |
| avx2 | 256 | 256 | 4507 | 3997 | 4253 | — | log3 |
| avx2 | 256 | 264 | 3400 | 2333 | 1729 | — | t1s |
| avx2 | 256 | 5120 | 4782 | 5046 | 4161 | — | t1s |
| avx2 | 256 | 8192 | 5953 | 5725 | 5131 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 92 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 58 |
| avx2 | `t1_dit` | 8 | 160 | `ct_t1_dit` | 61 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 63 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 120 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 123 |
| avx2 | `t1_dit` | 16 | 320 | `ct_t1_dit` | 129 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 301 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 230 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 224 |
| avx2 | `t1_dit` | 32 | 640 | `ct_t1_dit` | 338 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 569 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 489 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 469 |
| avx2 | `t1_dit` | 64 | 1280 | `ct_t1_dit` | 978 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1155 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1200 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1115 |
| avx2 | `t1_dit` | 128 | 2560 | `ct_t1_dit` | 2294 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2720 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4507 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3400 |
| avx2 | `t1_dit` | 256 | 5120 | `ct_t1_dit` | 4782 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 5953 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 91 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 82 |
| avx2 | `t1_dit_log3` | 8 | 160 | `ct_t1_dit_log3` | 77 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 77 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 149 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 130 |
| avx2 | `t1_dit_log3` | 16 | 320 | `ct_t1_dit_log3` | 137 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 287 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 261 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 256 |
| avx2 | `t1_dit_log3` | 32 | 640 | `ct_t1_dit_log3` | 267 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 606 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 518 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 505 |
| avx2 | `t1_dit_log3` | 64 | 1280 | `ct_t1_dit_log3` | 1030 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1142 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1006 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1060 |
| avx2 | `t1_dit_log3` | 128 | 2560 | `ct_t1_dit_log3` | 2049 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2500 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3997 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2333 |
| avx2 | `t1_dit_log3` | 256 | 5120 | `ct_t1_dit_log3` | 5046 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 5725 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 62 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 71 |
| avx2 | `t1s_dit` | 8 | 160 | `ct_t1s_dit` | 62 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 70 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 121 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 113 |
| avx2 | `t1s_dit` | 16 | 320 | `ct_t1s_dit` | 112 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 309 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 243 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 222 |
| avx2 | `t1s_dit` | 32 | 640 | `ct_t1s_dit` | 229 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 562 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 388 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 392 |
| avx2 | `t1s_dit` | 64 | 1280 | `ct_t1s_dit` | 1092 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1191 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 777 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 808 |
| avx2 | `t1s_dit` | 128 | 2560 | `ct_t1s_dit` | 2335 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1967 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4253 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1729 |
| avx2 | `t1s_dit` | 256 | 5120 | `ct_t1s_dit` | 4161 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5131 |