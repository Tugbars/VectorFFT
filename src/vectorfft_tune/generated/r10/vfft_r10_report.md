# VectorFFT R=10 tuning report

Total measurements: **448**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 62 | 62 | 52 | — | t1s |
| avx2 | 8 | 16 | 62 | 62 | 52 | — | t1s |
| avx2 | 8 | 80 | 66 | 61 | 53 | — | t1s |
| avx2 | 8 | 256 | 64 | 64 | 51 | — | t1s |
| avx2 | 16 | 16 | 122 | 121 | 100 | — | t1s |
| avx2 | 16 | 24 | 122 | 123 | 98 | — | t1s |
| avx2 | 16 | 160 | 122 | 122 | 101 | — | t1s |
| avx2 | 16 | 512 | 262 | 212 | 184 | — | t1s |
| avx2 | 32 | 32 | 232 | 239 | 194 | — | t1s |
| avx2 | 32 | 40 | 234 | 239 | 198 | — | t1s |
| avx2 | 32 | 320 | 246 | 240 | 194 | — | t1s |
| avx2 | 32 | 1024 | 454 | 344 | 361 | — | log3 |
| avx2 | 64 | 64 | 472 | 477 | 392 | — | t1s |
| avx2 | 64 | 72 | 478 | 476 | 400 | — | t1s |
| avx2 | 64 | 640 | 459 | 474 | 390 | — | t1s |
| avx2 | 64 | 2048 | 1160 | 1151 | 1124 | — | t1s |
| avx2 | 128 | 128 | 1033 | 955 | 787 | — | t1s |
| avx2 | 128 | 136 | 947 | 971 | 808 | — | t1s |
| avx2 | 128 | 1280 | 969 | 962 | 780 | — | t1s |
| avx2 | 128 | 4096 | 2314 | 2281 | 2250 | — | t1s |
| avx2 | 256 | 256 | 2001 | 1900 | 1615 | — | t1s |
| avx2 | 256 | 264 | 2068 | 1889 | 1522 | — | t1s |
| avx2 | 256 | 2560 | 4848 | 4593 | 4485 | — | t1s |
| avx2 | 256 | 8192 | 4764 | 4557 | 4544 | — | t1s |
| avx512 | 8 | 8 | 39 | 37 | 35 | — | t1s |
| avx512 | 8 | 16 | 39 | 37 | 35 | — | t1s |
| avx512 | 8 | 80 | 39 | 37 | 35 | — | t1s |
| avx512 | 8 | 256 | 40 | 37 | 37 | — | t1s |
| avx512 | 16 | 16 | 71 | 71 | 61 | — | t1s |
| avx512 | 16 | 24 | 73 | 72 | 62 | — | t1s |
| avx512 | 16 | 160 | 75 | 73 | 62 | — | t1s |
| avx512 | 16 | 512 | 116 | 95 | 99 | — | log3 |
| avx512 | 32 | 32 | 129 | 143 | 120 | — | t1s |
| avx512 | 32 | 40 | 135 | 147 | 117 | — | t1s |
| avx512 | 32 | 320 | 131 | 145 | 118 | — | t1s |
| avx512 | 32 | 1024 | 248 | 215 | 204 | — | t1s |
| avx512 | 64 | 64 | 256 | 287 | 233 | — | t1s |
| avx512 | 64 | 72 | 254 | 286 | 232 | — | t1s |
| avx512 | 64 | 640 | 255 | 284 | 238 | — | t1s |
| avx512 | 64 | 2048 | 562 | 509 | 505 | — | t1s |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 62 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 62 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 66 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 64 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 122 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 122 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 122 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 262 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 232 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 234 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 246 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 454 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 472 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 478 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 459 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1160 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1033 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 947 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 969 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2314 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 2001 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2068 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 4848 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 4764 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 62 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 61 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 64 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 121 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 123 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 122 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 212 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 239 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 239 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 240 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 344 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 477 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 476 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 474 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1151 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 955 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 971 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 962 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2281 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1900 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1889 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 4593 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 4557 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 52 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 52 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 53 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 51 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 100 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 98 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 101 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 184 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 194 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 198 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 194 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 361 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 392 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 400 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 390 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1124 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 787 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 808 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 780 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2250 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1615 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1522 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 4485 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 4544 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 39 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 39 |
| avx512 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 39 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 40 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 71 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 73 |
| avx512 | `t1_dit` | 16 | 160 | `ct_t1_dit_u2b` | 75 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 116 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 129 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 135 |
| avx512 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 131 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 248 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 256 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 254 |
| avx512 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 255 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 562 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 532 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 514 |
| avx512 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 558 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1081 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1286 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1242 |
| avx512 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 2183 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2291 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 37 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 37 |
| avx512 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 37 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 37 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 71 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 72 |
| avx512 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3_u2b` | 73 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 95 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 143 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 147 |
| avx512 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 145 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3_u2b` | 215 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 287 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 286 |
| avx512 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 284 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 509 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 576 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 566 |
| avx512 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 572 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 981 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1137 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1126 |
| avx512 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 1873 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1974 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 35 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 35 |
| avx512 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 35 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 37 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 61 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 62 |
| avx512 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 62 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 99 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 120 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 117 |
| avx512 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 118 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 204 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 233 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 232 |
| avx512 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 238 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 505 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 470 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 464 |
| avx512 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 466 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 961 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 937 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 927 |
| avx512 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 1905 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2056 |