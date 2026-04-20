# VectorFFT R=11 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 37 | 43 | 71 | — | flat |
| avx2 | 8 | 16 | 39 | 43 | 73 | — | flat |
| avx2 | 8 | 88 | 37 | 43 | 57 | — | flat |
| avx2 | 8 | 256 | 37 | 43 | 46 | — | flat |
| avx2 | 16 | 16 | 70 | 80 | 88 | — | flat |
| avx2 | 16 | 24 | 71 | 81 | 69 | — | t1s |
| avx2 | 16 | 176 | 67 | 81 | 81 | — | flat |
| avx2 | 16 | 512 | 187 | 177 | 185 | — | log3 |
| avx2 | 32 | 32 | 135 | 155 | 151 | — | flat |
| avx2 | 32 | 40 | 128 | 155 | 152 | — | flat |
| avx2 | 32 | 352 | 135 | 156 | 129 | — | t1s |
| avx2 | 32 | 1024 | 298 | 332 | 395 | — | flat |
| avx2 | 64 | 64 | 248 | 305 | 249 | — | flat |
| avx2 | 64 | 72 | 263 | 305 | 249 | — | t1s |
| avx2 | 64 | 704 | 270 | 312 | 253 | — | t1s |
| avx2 | 64 | 2048 | 701 | 729 | 672 | — | t1s |
| avx2 | 128 | 128 | 487 | 615 | 1087 | — | flat |
| avx2 | 128 | 136 | 523 | 608 | 488 | — | t1s |
| avx2 | 128 | 1408 | 550 | 609 | 493 | — | t1s |
| avx2 | 128 | 4096 | 1441 | 1328 | 1483 | — | log3 |
| avx2 | 256 | 256 | 1288 | 1240 | 1663 | — | log3 |
| avx2 | 256 | 264 | 1251 | 1241 | 970 | — | t1s |
| avx2 | 256 | 2816 | 1269 | 1255 | 1022 | — | t1s |
| avx2 | 256 | 8192 | 1885 | 1802 | 2361 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 37 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 39 |
| avx2 | `t1_dit` | 8 | 88 | `ct_t1_dit` | 37 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 37 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 70 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 71 |
| avx2 | `t1_dit` | 16 | 176 | `ct_t1_dit` | 67 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 187 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 135 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 128 |
| avx2 | `t1_dit` | 32 | 352 | `ct_t1_dit` | 135 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 298 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 248 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 263 |
| avx2 | `t1_dit` | 64 | 704 | `ct_t1_dit` | 270 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 701 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 487 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 523 |
| avx2 | `t1_dit` | 128 | 1408 | `ct_t1_dit` | 550 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1441 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1288 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1251 |
| avx2 | `t1_dit` | 256 | 2816 | `ct_t1_dit` | 1269 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 1885 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 8 | 88 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 80 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 81 |
| avx2 | `t1_dit_log3` | 16 | 176 | `ct_t1_dit_log3` | 81 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 177 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 155 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 155 |
| avx2 | `t1_dit_log3` | 32 | 352 | `ct_t1_dit_log3` | 156 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 332 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 305 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 305 |
| avx2 | `t1_dit_log3` | 64 | 704 | `ct_t1_dit_log3` | 312 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 729 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 615 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 608 |
| avx2 | `t1_dit_log3` | 128 | 1408 | `ct_t1_dit_log3` | 609 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1328 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1240 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1241 |
| avx2 | `t1_dit_log3` | 256 | 2816 | `ct_t1_dit_log3` | 1255 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1802 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 71 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 8 | 88 | `ct_t1s_dit` | 57 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 46 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 88 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 69 |
| avx2 | `t1s_dit` | 16 | 176 | `ct_t1s_dit` | 81 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 185 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 151 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 152 |
| avx2 | `t1s_dit` | 32 | 352 | `ct_t1s_dit` | 129 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 395 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 249 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 249 |
| avx2 | `t1s_dit` | 64 | 704 | `ct_t1s_dit` | 253 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 672 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1087 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 488 |
| avx2 | `t1s_dit` | 128 | 1408 | `ct_t1s_dit` | 493 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1483 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1663 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 970 |
| avx2 | `t1s_dit` | 256 | 2816 | `ct_t1s_dit` | 1022 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2361 |