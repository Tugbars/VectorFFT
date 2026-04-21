# VectorFFT R=10 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 28 | 34 | 31 | — | flat |
| avx2 | 8 | 16 | 27 | 36 | 29 | — | flat |
| avx2 | 8 | 80 | 27 | 36 | 29 | — | flat |
| avx2 | 8 | 256 | 28 | 35 | 31 | — | flat |
| avx2 | 16 | 16 | 50 | 70 | 57 | — | flat |
| avx2 | 16 | 24 | 48 | 64 | 51 | — | flat |
| avx2 | 16 | 160 | 47 | 64 | 54 | — | flat |
| avx2 | 16 | 512 | 177 | 164 | 163 | — | t1s |
| avx2 | 32 | 32 | 97 | 133 | 101 | — | flat |
| avx2 | 32 | 40 | 89 | 127 | 100 | — | flat |
| avx2 | 32 | 320 | 93 | 121 | 104 | — | flat |
| avx2 | 32 | 1024 | 301 | 295 | 218 | — | t1s |
| avx2 | 64 | 64 | 202 | 261 | 205 | — | flat |
| avx2 | 64 | 72 | 177 | 243 | 196 | — | flat |
| avx2 | 64 | 640 | 189 | 268 | 205 | — | flat |
| avx2 | 64 | 2048 | 569 | 541 | 387 | — | t1s |
| avx2 | 128 | 128 | 361 | 476 | 395 | — | flat |
| avx2 | 128 | 136 | 369 | 528 | 347 | — | t1s |
| avx2 | 128 | 1280 | 391 | 522 | 390 | — | t1s |
| avx2 | 128 | 4096 | 786 | 1080 | 983 | — | flat |
| avx2 | 256 | 256 | 935 | 1027 | 806 | — | t1s |
| avx2 | 256 | 264 | 819 | 1062 | 715 | — | t1s |
| avx2 | 256 | 2560 | 2307 | 2123 | 1709 | — | t1s |
| avx2 | 256 | 8192 | 1117 | 1131 | 859 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 28 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 27 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 27 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 28 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 50 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 48 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 47 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 177 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 97 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 89 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 93 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 301 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 202 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 177 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 189 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 569 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 361 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 369 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 391 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 786 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 935 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 819 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 2307 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 1117 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 34 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 36 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 36 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 35 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 70 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 64 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 64 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 133 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 127 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 121 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 295 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 261 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 243 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 268 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 541 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 476 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 528 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 522 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1080 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1027 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1062 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 2123 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1131 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 57 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 51 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 54 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 163 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 101 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 104 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 218 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 205 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 196 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 205 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 387 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 395 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 347 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 390 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 983 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 806 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 715 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 1709 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 859 |