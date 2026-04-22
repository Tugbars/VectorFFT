# VectorFFT R=20 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 63 | 88 | 61 | — | t1s |
| avx2 | 8 | 16 | 62 | 77 | 62 | — | t1s |
| avx2 | 8 | 160 | 60 | 79 | 62 | — | flat |
| avx2 | 8 | 256 | 63 | 73 | 62 | — | t1s |
| avx2 | 16 | 16 | 139 | 143 | 111 | — | t1s |
| avx2 | 16 | 24 | 112 | 135 | 108 | — | t1s |
| avx2 | 16 | 320 | 117 | 140 | 109 | — | t1s |
| avx2 | 16 | 512 | 271 | 292 | 306 | — | flat |
| avx2 | 32 | 32 | 231 | 271 | 225 | — | t1s |
| avx2 | 32 | 40 | 228 | 302 | 202 | — | t1s |
| avx2 | 32 | 640 | 230 | 277 | 221 | — | t1s |
| avx2 | 32 | 1024 | 579 | 547 | 575 | — | log3 |
| avx2 | 64 | 64 | 431 | 616 | 431 | — | t1s |
| avx2 | 64 | 72 | 693 | 665 | 423 | — | t1s |
| avx2 | 64 | 1280 | 1086 | 1032 | 1014 | — | t1s |
| avx2 | 64 | 2048 | 1143 | 1123 | 1109 | — | t1s |
| avx2 | 128 | 128 | 1119 | 1061 | 809 | — | t1s |
| avx2 | 128 | 136 | 1094 | 1062 | 826 | — | t1s |
| avx2 | 128 | 2560 | 2197 | 2111 | 2270 | — | log3 |
| avx2 | 128 | 4096 | 2612 | 2244 | 2058 | — | t1s |
| avx2 | 256 | 256 | 4291 | 4893 | 4263 | — | t1s |
| avx2 | 256 | 264 | 2641 | 2270 | 2146 | — | t1s |
| avx2 | 256 | 5120 | 5287 | 4769 | 4449 | — | t1s |
| avx2 | 256 | 8192 | 6448 | 6468 | 5166 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 63 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 62 |
| avx2 | `t1_dit` | 8 | 160 | `ct_t1_dit` | 60 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 63 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 139 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 112 |
| avx2 | `t1_dit` | 16 | 320 | `ct_t1_dit` | 117 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 271 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 231 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 228 |
| avx2 | `t1_dit` | 32 | 640 | `ct_t1_dit` | 230 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 579 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 431 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 693 |
| avx2 | `t1_dit` | 64 | 1280 | `ct_t1_dit` | 1086 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1143 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1119 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1094 |
| avx2 | `t1_dit` | 128 | 2560 | `ct_t1_dit` | 2197 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2612 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4291 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2641 |
| avx2 | `t1_dit` | 256 | 5120 | `ct_t1_dit` | 5287 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 6448 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 88 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 77 |
| avx2 | `t1_dit_log3` | 8 | 160 | `ct_t1_dit_log3` | 79 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 73 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 143 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 135 |
| avx2 | `t1_dit_log3` | 16 | 320 | `ct_t1_dit_log3` | 140 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 292 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 271 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 302 |
| avx2 | `t1_dit_log3` | 32 | 640 | `ct_t1_dit_log3` | 277 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 547 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 616 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 665 |
| avx2 | `t1_dit_log3` | 64 | 1280 | `ct_t1_dit_log3` | 1032 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1123 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1061 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1062 |
| avx2 | `t1_dit_log3` | 128 | 2560 | `ct_t1_dit_log3` | 2111 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2244 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4893 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2270 |
| avx2 | `t1_dit_log3` | 256 | 5120 | `ct_t1_dit_log3` | 4769 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 6468 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 61 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 62 |
| avx2 | `t1s_dit` | 8 | 160 | `ct_t1s_dit` | 62 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 62 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 111 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 108 |
| avx2 | `t1s_dit` | 16 | 320 | `ct_t1s_dit` | 109 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 306 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 225 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 202 |
| avx2 | `t1s_dit` | 32 | 640 | `ct_t1s_dit` | 221 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 575 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 431 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 423 |
| avx2 | `t1s_dit` | 64 | 1280 | `ct_t1s_dit` | 1014 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1109 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 809 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 826 |
| avx2 | `t1s_dit` | 128 | 2560 | `ct_t1s_dit` | 2270 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2058 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4263 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2146 |
| avx2 | `t1s_dit` | 256 | 5120 | `ct_t1s_dit` | 4449 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5166 |