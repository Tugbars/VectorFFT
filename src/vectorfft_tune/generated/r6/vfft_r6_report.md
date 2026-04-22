# VectorFFT R=6 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 11 | 13 | 11 | — | t1s |
| avx2 | 8 | 16 | 11 | 13 | 11 | — | flat |
| avx2 | 8 | 80 | 12 | 13 | 11 | — | t1s |
| avx2 | 8 | 256 | 12 | 13 | 11 | — | t1s |
| avx2 | 16 | 16 | 21 | 25 | 20 | — | t1s |
| avx2 | 16 | 24 | 21 | 25 | 20 | — | t1s |
| avx2 | 16 | 160 | 21 | 25 | 20 | — | t1s |
| avx2 | 16 | 512 | 21 | 25 | 20 | — | t1s |
| avx2 | 32 | 32 | 40 | 49 | 37 | — | t1s |
| avx2 | 32 | 40 | 40 | 49 | 38 | — | t1s |
| avx2 | 32 | 320 | 40 | 50 | 37 | — | t1s |
| avx2 | 32 | 1024 | 40 | 49 | 37 | — | t1s |
| avx2 | 64 | 64 | 76 | 98 | 72 | — | t1s |
| avx2 | 64 | 72 | 75 | 99 | 73 | — | t1s |
| avx2 | 64 | 640 | 72 | 97 | 72 | — | t1s |
| avx2 | 64 | 2048 | 109 | 100 | 73 | — | t1s |
| avx2 | 128 | 128 | 141 | 194 | 143 | — | flat |
| avx2 | 128 | 136 | 140 | 200 | 141 | — | flat |
| avx2 | 128 | 1280 | 139 | 213 | 144 | — | flat |
| avx2 | 128 | 4096 | 200 | 198 | 150 | — | t1s |
| avx2 | 256 | 256 | 277 | 392 | 285 | — | flat |
| avx2 | 256 | 264 | 278 | 393 | 289 | — | flat |
| avx2 | 256 | 2560 | 398 | 390 | 380 | — | t1s |
| avx2 | 256 | 8192 | 386 | 398 | 361 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 11 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 11 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 12 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 12 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 21 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 21 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 21 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 21 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 75 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 72 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 109 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 141 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 140 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 139 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 200 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 277 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 278 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 398 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 386 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 13 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 13 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 13 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 13 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 25 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 25 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 25 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 25 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 50 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 98 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 99 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 97 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 100 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 200 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 213 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 198 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 392 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 393 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 390 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 398 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 11 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 11 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 11 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 11 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 20 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 20 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 20 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 20 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 38 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 37 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 72 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 72 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 143 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 141 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 144 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 150 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 285 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 289 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 380 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 361 |