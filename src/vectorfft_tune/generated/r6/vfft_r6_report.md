# VectorFFT R=6 tuning report

Total measurements: **408**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 25 | 27 | 25 | — | t1s |
| avx2 | 8 | 16 | 25 | 27 | 25 | — | t1s |
| avx2 | 8 | 80 | 25 | 27 | 26 | — | flat |
| avx2 | 8 | 256 | 27 | 27 | 27 | — | flat |
| avx2 | 16 | 16 | 48 | 52 | 42 | — | t1s |
| avx2 | 16 | 24 | 46 | 51 | 42 | — | t1s |
| avx2 | 16 | 160 | 48 | 52 | 44 | — | t1s |
| avx2 | 16 | 512 | 47 | 52 | 45 | — | t1s |
| avx2 | 32 | 32 | 93 | 102 | 82 | — | t1s |
| avx2 | 32 | 40 | 93 | 104 | 84 | — | t1s |
| avx2 | 32 | 320 | 95 | 106 | 84 | — | t1s |
| avx2 | 32 | 1024 | 99 | 111 | 83 | — | t1s |
| avx2 | 64 | 64 | 182 | 208 | 156 | — | t1s |
| avx2 | 64 | 72 | 185 | 203 | 164 | — | t1s |
| avx2 | 64 | 640 | 183 | 200 | 156 | — | t1s |
| avx2 | 64 | 2048 | 270 | 214 | 160 | — | t1s |
| avx2 | 128 | 128 | 351 | 415 | 326 | — | t1s |
| avx2 | 128 | 136 | 369 | 414 | 313 | — | t1s |
| avx2 | 128 | 1280 | 364 | 406 | 314 | — | t1s |
| avx2 | 128 | 4096 | 852 | 631 | 397 | — | t1s |
| avx2 | 256 | 256 | 705 | 797 | 632 | — | t1s |
| avx2 | 256 | 264 | 695 | 804 | 611 | — | t1s |
| avx2 | 256 | 2560 | 882 | 838 | 780 | — | t1s |
| avx2 | 256 | 8192 | 1809 | 1218 | 946 | — | t1s |
| avx512 | 8 | 8 | 19 | 19 | 19 | — | t1s |
| avx512 | 8 | 16 | 19 | 19 | 19 | — | flat |
| avx512 | 8 | 80 | 20 | 19 | 20 | — | log3 |
| avx512 | 8 | 256 | 20 | 20 | 20 | — | t1s |
| avx512 | 16 | 16 | 30 | 33 | 26 | — | t1s |
| avx512 | 16 | 24 | 30 | 35 | 27 | — | t1s |
| avx512 | 16 | 160 | 30 | 34 | 27 | — | t1s |
| avx512 | 16 | 512 | 30 | 33 | 29 | — | t1s |
| avx512 | 32 | 32 | 54 | 62 | 51 | — | t1s |
| avx512 | 32 | 40 | 55 | 64 | 51 | — | t1s |
| avx512 | 32 | 320 | 56 | 65 | 52 | — | t1s |
| avx512 | 32 | 1024 | 56 | 65 | 50 | — | t1s |
| avx512 | 64 | 64 | 109 | 129 | 98 | — | t1s |
| avx512 | 64 | 72 | 112 | 125 | 102 | — | t1s |
| avx512 | 64 | 640 | 109 | 129 | 100 | — | t1s |
| avx512 | 64 | 2048 | 162 | 131 | 100 | — | t1s |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 27 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 48 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 48 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 47 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 93 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 93 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 95 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 99 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 182 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 185 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 183 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 270 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 351 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 369 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 364 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 852 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 705 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 695 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 882 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 1809 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 27 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 27 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 27 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 27 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 52 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 51 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 52 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 52 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 102 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 104 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 106 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 111 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 208 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 203 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 200 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 214 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 415 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 414 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 406 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 631 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 797 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 804 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 838 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1218 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 25 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 25 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 26 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 27 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 42 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 42 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 44 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 45 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 82 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 84 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 84 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 83 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 156 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 164 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 156 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 160 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 326 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 313 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 314 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 397 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 632 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 611 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 780 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 946 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 19 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 19 |
| avx512 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 20 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 20 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 30 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit_u2` | 30 |
| avx512 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 30 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit_u2` | 30 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit_u2` | 54 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit_u2` | 55 |
| avx512 | `t1_dit` | 32 | 320 | `ct_t1_dit_u2` | 56 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit_u2` | 56 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit_u2` | 109 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 112 |
| avx512 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 109 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit_u2` | 162 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit_u2` | 210 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 221 |
| avx512 | `t1_dit` | 128 | 1280 | `ct_t1_dit_u2` | 215 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit_u2` | 397 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 435 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 440 |
| avx512 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 649 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 875 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 19 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 19 |
| avx512 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 19 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 20 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 33 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 35 |
| avx512 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 34 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 33 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 62 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 64 |
| avx512 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 65 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 65 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 129 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 125 |
| avx512 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 129 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 131 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 246 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 244 |
| avx512 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 248 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3_u2` | 292 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 502 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 492 |
| avx512 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3_u2` | 548 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 567 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 19 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 19 |
| avx512 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 20 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 20 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit_u2` | 26 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 27 |
| avx512 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 27 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 29 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 51 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit_u2` | 51 |
| avx512 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 52 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit_u2` | 50 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit_u2` | 98 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 102 |
| avx512 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 100 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 100 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 200 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit_u2` | 198 |
| avx512 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 202 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit_u2` | 204 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit_u2` | 393 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 379 |
| avx512 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 403 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 409 |