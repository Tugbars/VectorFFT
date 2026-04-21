# VectorFFT R=6 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 11 | 15 | 11 | — | t1s |
| avx2 | 8 | 16 | 13 | 13 | 12 | — | t1s |
| avx2 | 8 | 80 | 12 | 13 | 12 | — | t1s |
| avx2 | 8 | 256 | 13 | 15 | 12 | — | t1s |
| avx2 | 16 | 16 | 21 | 27 | 20 | — | t1s |
| avx2 | 16 | 24 | 21 | 25 | 21 | — | t1s |
| avx2 | 16 | 160 | 23 | 27 | 21 | — | t1s |
| avx2 | 16 | 512 | 24 | 28 | 21 | — | t1s |
| avx2 | 32 | 32 | 37 | 53 | 38 | — | flat |
| avx2 | 32 | 40 | 40 | 57 | 38 | — | t1s |
| avx2 | 32 | 320 | 40 | 54 | 39 | — | t1s |
| avx2 | 32 | 1024 | 41 | 53 | 40 | — | t1s |
| avx2 | 64 | 64 | 76 | 102 | 82 | — | flat |
| avx2 | 64 | 72 | 75 | 112 | 77 | — | flat |
| avx2 | 64 | 640 | 71 | 111 | 78 | — | flat |
| avx2 | 64 | 2048 | 141 | 101 | 82 | — | t1s |
| avx2 | 128 | 128 | 153 | 209 | 160 | — | flat |
| avx2 | 128 | 136 | 147 | 221 | 158 | — | flat |
| avx2 | 128 | 1280 | 154 | 217 | 157 | — | flat |
| avx2 | 128 | 4096 | 285 | 198 | 162 | — | t1s |
| avx2 | 256 | 256 | 307 | 423 | 309 | — | flat |
| avx2 | 256 | 264 | 299 | 427 | 298 | — | t1s |
| avx2 | 256 | 2560 | 439 | 407 | 337 | — | t1s |
| avx2 | 256 | 8192 | 450 | 425 | 359 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 11 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 13 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 12 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 13 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 21 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 21 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 23 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 24 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 37 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 41 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 75 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 71 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 141 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 153 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 147 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 154 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 285 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 307 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 299 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 439 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 450 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 15 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 13 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 13 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 15 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 27 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 25 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 27 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 28 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 53 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 57 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 54 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 53 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 102 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 112 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 111 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 101 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 209 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 221 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 217 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 198 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 423 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 427 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 407 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 425 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 11 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 12 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 12 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 12 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 20 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 21 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 21 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 21 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 38 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 38 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 39 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 40 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 82 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 77 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 78 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 82 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 160 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 158 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 157 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 162 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 309 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 298 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 337 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 359 |