# VectorFFT R=20 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 65 | 70 | 60 | — | t1s |
| avx2 | 8 | 16 | 61 | 69 | 73 | — | flat |
| avx2 | 8 | 160 | 70 | 69 | 59 | — | t1s |
| avx2 | 8 | 256 | 68 | 71 | 61 | — | t1s |
| avx2 | 16 | 16 | 115 | 133 | 107 | — | t1s |
| avx2 | 16 | 24 | 115 | 131 | 118 | — | flat |
| avx2 | 16 | 320 | 130 | 134 | 106 | — | t1s |
| avx2 | 16 | 512 | 297 | 279 | 270 | — | t1s |
| avx2 | 32 | 32 | 314 | 258 | 224 | — | t1s |
| avx2 | 32 | 40 | 233 | 258 | 218 | — | t1s |
| avx2 | 32 | 640 | 237 | 259 | 220 | — | t1s |
| avx2 | 32 | 1024 | 771 | 516 | 524 | — | log3 |
| avx2 | 64 | 64 | 465 | 700 | 410 | — | t1s |
| avx2 | 64 | 72 | 434 | 506 | 390 | — | t1s |
| avx2 | 64 | 1280 | 1117 | 911 | 1058 | — | log3 |
| avx2 | 64 | 2048 | 1241 | 1071 | 1082 | — | log3 |
| avx2 | 128 | 128 | 1535 | 1008 | 782 | — | t1s |
| avx2 | 128 | 136 | 1167 | 1200 | 852 | — | t1s |
| avx2 | 128 | 2560 | 2279 | 2192 | 2253 | — | log3 |
| avx2 | 128 | 4096 | 2367 | 2332 | 2007 | — | t1s |
| avx2 | 256 | 256 | 4381 | 4338 | 4488 | — | log3 |
| avx2 | 256 | 264 | 2604 | 2170 | 1885 | — | t1s |
| avx2 | 256 | 5120 | 5290 | 4783 | 4169 | — | t1s |
| avx2 | 256 | 8192 | 5960 | 6712 | 5096 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 65 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 61 |
| avx2 | `t1_dit` | 8 | 160 | `ct_t1_dit` | 70 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 68 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 115 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 115 |
| avx2 | `t1_dit` | 16 | 320 | `ct_t1_dit` | 130 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 297 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 314 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 233 |
| avx2 | `t1_dit` | 32 | 640 | `ct_t1_dit` | 237 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 771 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 465 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 434 |
| avx2 | `t1_dit` | 64 | 1280 | `ct_t1_dit` | 1117 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1241 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1535 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1167 |
| avx2 | `t1_dit` | 128 | 2560 | `ct_t1_dit` | 2279 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2367 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4381 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2604 |
| avx2 | `t1_dit` | 256 | 5120 | `ct_t1_dit` | 5290 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 5960 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 70 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 69 |
| avx2 | `t1_dit_log3` | 8 | 160 | `ct_t1_dit_log3` | 69 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 71 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 133 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 131 |
| avx2 | `t1_dit_log3` | 16 | 320 | `ct_t1_dit_log3` | 134 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 279 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 258 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 258 |
| avx2 | `t1_dit_log3` | 32 | 640 | `ct_t1_dit_log3` | 259 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 516 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 700 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 506 |
| avx2 | `t1_dit_log3` | 64 | 1280 | `ct_t1_dit_log3` | 911 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1071 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1008 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1200 |
| avx2 | `t1_dit_log3` | 128 | 2560 | `ct_t1_dit_log3` | 2192 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2332 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4338 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2170 |
| avx2 | `t1_dit_log3` | 256 | 5120 | `ct_t1_dit_log3` | 4783 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 6712 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 60 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 8 | 160 | `ct_t1s_dit` | 59 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 61 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 107 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 118 |
| avx2 | `t1s_dit` | 16 | 320 | `ct_t1s_dit` | 106 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 270 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 224 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 218 |
| avx2 | `t1s_dit` | 32 | 640 | `ct_t1s_dit` | 220 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 524 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 410 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 390 |
| avx2 | `t1s_dit` | 64 | 1280 | `ct_t1s_dit` | 1058 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1082 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 782 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 852 |
| avx2 | `t1s_dit` | 128 | 2560 | `ct_t1s_dit` | 2253 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2007 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4488 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1885 |
| avx2 | `t1s_dit` | 256 | 5120 | `ct_t1s_dit` | 4169 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5096 |