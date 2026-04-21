# VectorFFT R=12 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 30 | 43 | 33 | — | flat |
| avx2 | 8 | 16 | 33 | 41 | 33 | — | t1s |
| avx2 | 8 | 96 | 31 | 42 | 31 | — | flat |
| avx2 | 8 | 256 | 29 | 41 | 32 | — | flat |
| avx2 | 16 | 16 | 60 | 82 | 56 | — | t1s |
| avx2 | 16 | 24 | 60 | 79 | 55 | — | t1s |
| avx2 | 16 | 192 | 57 | 81 | 57 | — | flat |
| avx2 | 16 | 512 | 203 | 224 | 184 | — | t1s |
| avx2 | 32 | 32 | 107 | 159 | 116 | — | flat |
| avx2 | 32 | 40 | 108 | 159 | 115 | — | flat |
| avx2 | 32 | 384 | 112 | 158 | 108 | — | t1s |
| avx2 | 32 | 1024 | 350 | 364 | 308 | — | t1s |
| avx2 | 64 | 64 | 221 | 302 | 222 | — | flat |
| avx2 | 64 | 72 | 217 | 304 | 205 | — | t1s |
| avx2 | 64 | 768 | 442 | 348 | 223 | — | t1s |
| avx2 | 64 | 2048 | 733 | 694 | 700 | — | log3 |
| avx2 | 128 | 128 | 445 | 616 | 417 | — | t1s |
| avx2 | 128 | 136 | 439 | 610 | 409 | — | t1s |
| avx2 | 128 | 1536 | 1305 | 1447 | 1430 | — | flat |
| avx2 | 128 | 4096 | 1523 | 1495 | 1394 | — | t1s |
| avx2 | 256 | 256 | 1136 | 1275 | 822 | — | t1s |
| avx2 | 256 | 264 | 1053 | 1242 | 815 | — | t1s |
| avx2 | 256 | 3072 | 3005 | 3075 | 2725 | — | t1s |
| avx2 | 256 | 8192 | 3458 | 2124 | 2159 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 30 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 33 |
| avx2 | `t1_dit` | 8 | 96 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 29 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 60 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 60 |
| avx2 | `t1_dit` | 16 | 192 | `ct_t1_dit` | 57 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 203 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 107 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 108 |
| avx2 | `t1_dit` | 32 | 384 | `ct_t1_dit` | 112 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 350 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 221 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 217 |
| avx2 | `t1_dit` | 64 | 768 | `ct_t1_dit` | 442 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 733 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 445 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 439 |
| avx2 | `t1_dit` | 128 | 1536 | `ct_t1_dit` | 1305 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1523 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1136 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1053 |
| avx2 | `t1_dit` | 256 | 3072 | `ct_t1_dit` | 3005 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 3458 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 43 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 41 |
| avx2 | `t1_dit_log3` | 8 | 96 | `ct_t1_dit_log3` | 42 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 41 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 82 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 79 |
| avx2 | `t1_dit_log3` | 16 | 192 | `ct_t1_dit_log3` | 81 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 224 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 159 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 159 |
| avx2 | `t1_dit_log3` | 32 | 384 | `ct_t1_dit_log3` | 158 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 364 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 302 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 304 |
| avx2 | `t1_dit_log3` | 64 | 768 | `ct_t1_dit_log3` | 348 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 694 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 616 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 610 |
| avx2 | `t1_dit_log3` | 128 | 1536 | `ct_t1_dit_log3` | 1447 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1495 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1275 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1242 |
| avx2 | `t1_dit_log3` | 256 | 3072 | `ct_t1_dit_log3` | 3075 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 2124 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 33 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 33 |
| avx2 | `t1s_dit` | 8 | 96 | `ct_t1s_dit` | 31 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 32 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 56 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 55 |
| avx2 | `t1s_dit` | 16 | 192 | `ct_t1s_dit` | 57 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 184 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 116 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 115 |
| avx2 | `t1s_dit` | 32 | 384 | `ct_t1s_dit` | 108 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 308 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 222 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 205 |
| avx2 | `t1s_dit` | 64 | 768 | `ct_t1s_dit` | 223 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 700 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 417 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 409 |
| avx2 | `t1s_dit` | 128 | 1536 | `ct_t1s_dit` | 1430 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1394 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 822 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 815 |
| avx2 | `t1s_dit` | 256 | 3072 | `ct_t1s_dit` | 2725 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2159 |