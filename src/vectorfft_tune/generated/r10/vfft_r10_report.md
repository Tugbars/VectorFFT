# VectorFFT R=10 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 25 | 33 | 30 | — | flat |
| avx2 | 8 | 16 | 25 | 34 | 29 | — | flat |
| avx2 | 8 | 80 | 25 | 34 | 30 | — | flat |
| avx2 | 8 | 256 | 26 | 33 | 29 | — | flat |
| avx2 | 16 | 16 | 45 | 63 | 51 | — | flat |
| avx2 | 16 | 24 | 46 | 62 | 50 | — | flat |
| avx2 | 16 | 160 | 46 | 62 | 54 | — | flat |
| avx2 | 16 | 512 | 146 | 164 | 151 | — | flat |
| avx2 | 32 | 32 | 88 | 122 | 91 | — | flat |
| avx2 | 32 | 40 | 87 | 121 | 100 | — | flat |
| avx2 | 32 | 320 | 88 | 122 | 98 | — | flat |
| avx2 | 32 | 1024 | 275 | 275 | 221 | — | t1s |
| avx2 | 64 | 64 | 182 | 248 | 178 | — | t1s |
| avx2 | 64 | 72 | 170 | 240 | 209 | — | flat |
| avx2 | 64 | 640 | 185 | 245 | 195 | — | flat |
| avx2 | 64 | 2048 | 558 | 512 | 367 | — | t1s |
| avx2 | 128 | 128 | 337 | 482 | 372 | — | flat |
| avx2 | 128 | 136 | 341 | 474 | 386 | — | flat |
| avx2 | 128 | 1280 | 409 | 500 | 382 | — | t1s |
| avx2 | 128 | 4096 | 1012 | 1061 | 693 | — | t1s |
| avx2 | 256 | 256 | 869 | 945 | 750 | — | t1s |
| avx2 | 256 | 264 | 881 | 967 | 721 | — | t1s |
| avx2 | 256 | 2560 | 2207 | 2164 | 1458 | — | t1s |
| avx2 | 256 | 8192 | 1077 | 1056 | 915 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 26 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 45 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 88 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 87 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 88 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 275 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 182 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 170 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 185 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 558 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 337 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 341 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 409 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1012 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 869 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 881 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 2207 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 1077 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 33 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 34 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 34 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 33 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 63 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 122 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 121 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 122 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 275 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 248 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 240 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 245 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 512 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 482 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 474 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 500 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1061 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 945 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 967 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 2164 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1056 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 51 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 50 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 54 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 151 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 91 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 221 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 178 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 209 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 195 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 367 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 372 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 386 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 382 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 693 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 750 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 721 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 1458 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 915 |