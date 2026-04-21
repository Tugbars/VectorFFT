# VectorFFT R=10 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 28 | 35 | 29 | — | flat |
| avx2 | 8 | 16 | 26 | 35 | 31 | — | flat |
| avx2 | 8 | 80 | 27 | 35 | 28 | — | flat |
| avx2 | 8 | 256 | 27 | 34 | 48 | — | flat |
| avx2 | 16 | 16 | 50 | 64 | 52 | — | flat |
| avx2 | 16 | 24 | 48 | 66 | 55 | — | flat |
| avx2 | 16 | 160 | 50 | 62 | 60 | — | flat |
| avx2 | 16 | 512 | 165 | 174 | 161 | — | t1s |
| avx2 | 32 | 32 | 97 | 126 | 100 | — | flat |
| avx2 | 32 | 40 | 88 | 129 | 93 | — | flat |
| avx2 | 32 | 320 | 100 | 118 | 100 | — | t1s |
| avx2 | 32 | 1024 | 301 | 283 | 215 | — | t1s |
| avx2 | 64 | 64 | 181 | 249 | 187 | — | flat |
| avx2 | 64 | 72 | 178 | 238 | 186 | — | flat |
| avx2 | 64 | 640 | 185 | 270 | 203 | — | flat |
| avx2 | 64 | 2048 | 638 | 592 | 384 | — | t1s |
| avx2 | 128 | 128 | 385 | 481 | 392 | — | flat |
| avx2 | 128 | 136 | 344 | 492 | 356 | — | flat |
| avx2 | 128 | 1280 | 450 | 522 | 378 | — | t1s |
| avx2 | 128 | 4096 | 1339 | 1070 | 1256 | — | log3 |
| avx2 | 256 | 256 | 954 | 1037 | 770 | — | t1s |
| avx2 | 256 | 264 | 840 | 1042 | 723 | — | t1s |
| avx2 | 256 | 2560 | 2356 | 2052 | 1347 | — | t1s |
| avx2 | 256 | 8192 | 2037 | 1133 | 984 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 28 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 26 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 27 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 27 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 50 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 48 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 50 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 165 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 97 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 88 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 100 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 301 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 181 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 178 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 185 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 638 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 385 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 344 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 450 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1339 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 954 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 840 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 2356 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2037 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 35 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 35 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 35 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 34 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 64 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 66 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 174 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 126 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 129 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 118 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 283 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 249 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 238 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 270 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 592 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 481 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 492 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 522 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1070 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1037 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1042 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 2052 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1133 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 29 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 28 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 48 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 52 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 55 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 60 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 161 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 93 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 215 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 187 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 186 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 203 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 384 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 392 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 356 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 378 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1256 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 770 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 723 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 1347 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 984 |