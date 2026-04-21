# VectorFFT R=19 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 266 | 254 | 245 | — | t1s |
| avx2 | 8 | 16 | 266 | 251 | 240 | — | t1s |
| avx2 | 8 | 152 | 261 | 253 | 237 | — | t1s |
| avx2 | 8 | 256 | 260 | 254 | 251 | — | t1s |
| avx2 | 16 | 16 | 505 | 503 | 465 | — | t1s |
| avx2 | 16 | 24 | 505 | 492 | 472 | — | t1s |
| avx2 | 16 | 304 | 513 | 499 | 476 | — | t1s |
| avx2 | 16 | 512 | 779 | 747 | 728 | — | t1s |
| avx2 | 32 | 32 | 987 | 970 | 865 | — | t1s |
| avx2 | 32 | 40 | 994 | 957 | 919 | — | t1s |
| avx2 | 32 | 608 | 988 | 1010 | 943 | — | t1s |
| avx2 | 32 | 1024 | 1892 | 1768 | 1746 | — | t1s |
| avx2 | 64 | 64 | 1995 | 2059 | 1837 | — | t1s |
| avx2 | 64 | 72 | 1939 | 1979 | 1871 | — | t1s |
| avx2 | 64 | 1216 | 2026 | 1927 | 1789 | — | t1s |
| avx2 | 64 | 2048 | 3672 | 3355 | 3413 | — | log3 |
| avx2 | 128 | 128 | 4295 | 3803 | 3676 | — | t1s |
| avx2 | 128 | 136 | 4188 | 3868 | 3700 | — | t1s |
| avx2 | 128 | 2432 | 4948 | 4051 | 3725 | — | t1s |
| avx2 | 128 | 4096 | 7399 | 6882 | 6685 | — | t1s |
| avx2 | 256 | 256 | 10294 | 10143 | 9759 | — | t1s |
| avx2 | 256 | 264 | 8451 | 7846 | 7266 | — | t1s |
| avx2 | 256 | 4864 | 12883 | 12352 | 11379 | — | t1s |
| avx2 | 256 | 8192 | 14435 | 13818 | 13374 | — | t1s |
| avx512 | 8 | 8 | 156 | 138 | 142 | — | log3 |
| avx512 | 8 | 16 | 154 | 138 | 143 | — | log3 |
| avx512 | 8 | 152 | 152 | 144 | 142 | — | t1s |
| avx512 | 8 | 256 | 152 | 136 | 141 | — | log3 |
| avx512 | 16 | 16 | 286 | 264 | 257 | — | t1s |
| avx512 | 16 | 24 | 297 | 266 | 261 | — | t1s |
| avx512 | 16 | 304 | 286 | 267 | 264 | — | t1s |
| avx512 | 16 | 512 | 424 | 359 | 399 | — | log3 |
| avx512 | 32 | 32 | 536 | 524 | 502 | — | t1s |
| avx512 | 32 | 40 | 551 | 528 | 491 | — | t1s |
| avx512 | 32 | 608 | 563 | 522 | 494 | — | t1s |
| avx512 | 32 | 1024 | 845 | 791 | 823 | — | log3 |
| avx512 | 64 | 64 | 1050 | 1049 | 994 | — | t1s |
| avx512 | 64 | 72 | 1066 | 1030 | 980 | — | t1s |
| avx512 | 64 | 1216 | 1059 | 1041 | 1013 | — | t1s |
| avx512 | 64 | 2048 | 1685 | 1534 | 1608 | — | log3 |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 266 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 266 |
| avx2 | `t1_dit` | 8 | 152 | `ct_t1_dit` | 261 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 260 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 505 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 505 |
| avx2 | `t1_dit` | 16 | 304 | `ct_t1_dit` | 513 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 779 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 987 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 994 |
| avx2 | `t1_dit` | 32 | 608 | `ct_t1_dit` | 988 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1892 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1995 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1939 |
| avx2 | `t1_dit` | 64 | 1216 | `ct_t1_dit` | 2026 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 3672 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 4295 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 4188 |
| avx2 | `t1_dit` | 128 | 2432 | `ct_t1_dit` | 4948 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 7399 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 10294 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 8451 |
| avx2 | `t1_dit` | 256 | 4864 | `ct_t1_dit` | 12883 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 14435 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 254 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 251 |
| avx2 | `t1_dit_log3` | 8 | 152 | `ct_t1_dit_log3` | 253 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 254 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 503 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 492 |
| avx2 | `t1_dit_log3` | 16 | 304 | `ct_t1_dit_log3` | 499 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 747 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 970 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 957 |
| avx2 | `t1_dit_log3` | 32 | 608 | `ct_t1_dit_log3` | 1010 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 1768 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2059 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1979 |
| avx2 | `t1_dit_log3` | 64 | 1216 | `ct_t1_dit_log3` | 1927 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 3355 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3803 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 3868 |
| avx2 | `t1_dit_log3` | 128 | 2432 | `ct_t1_dit_log3` | 4051 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 6882 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 10143 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 7846 |
| avx2 | `t1_dit_log3` | 256 | 4864 | `ct_t1_dit_log3` | 12352 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 13818 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 245 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 240 |
| avx2 | `t1s_dit` | 8 | 152 | `ct_t1s_dit` | 237 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 251 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 465 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 472 |
| avx2 | `t1s_dit` | 16 | 304 | `ct_t1s_dit` | 476 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 728 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 865 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 919 |
| avx2 | `t1s_dit` | 32 | 608 | `ct_t1s_dit` | 943 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 1746 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1837 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1871 |
| avx2 | `t1s_dit` | 64 | 1216 | `ct_t1s_dit` | 1789 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 3413 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3676 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3700 |
| avx2 | `t1s_dit` | 128 | 2432 | `ct_t1s_dit` | 3725 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 6685 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 9759 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 7266 |
| avx2 | `t1s_dit` | 256 | 4864 | `ct_t1s_dit` | 11379 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 13374 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 156 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 154 |
| avx512 | `t1_dit` | 8 | 152 | `ct_t1_dit` | 152 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 152 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 286 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 297 |
| avx512 | `t1_dit` | 16 | 304 | `ct_t1_dit` | 286 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 424 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 536 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 551 |
| avx512 | `t1_dit` | 32 | 608 | `ct_t1_dit` | 563 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 845 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1050 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1066 |
| avx512 | `t1_dit` | 64 | 1216 | `ct_t1_dit` | 1059 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1685 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2488 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2460 |
| avx512 | `t1_dit` | 128 | 2432 | `ct_t1_dit` | 2813 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3184 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5166 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5083 |
| avx512 | `t1_dit` | 256 | 4864 | `ct_t1_dit` | 5324 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 6497 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 138 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 138 |
| avx512 | `t1_dit_log3` | 8 | 152 | `ct_t1_dit_log3` | 144 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 136 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 264 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 266 |
| avx512 | `t1_dit_log3` | 16 | 304 | `ct_t1_dit_log3` | 267 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 359 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 524 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 528 |
| avx512 | `t1_dit_log3` | 32 | 608 | `ct_t1_dit_log3` | 522 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 791 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1049 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1030 |
| avx512 | `t1_dit_log3` | 64 | 1216 | `ct_t1_dit_log3` | 1041 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1534 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 2042 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2083 |
| avx512 | `t1_dit_log3` | 128 | 2432 | `ct_t1_dit_log3` | 2033 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3822 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4436 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 4276 |
| avx512 | `t1_dit_log3` | 256 | 4864 | `ct_t1_dit_log3` | 4970 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 5941 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 142 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 143 |
| avx512 | `t1s_dit` | 8 | 152 | `ct_t1s_dit` | 142 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 141 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 257 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 261 |
| avx512 | `t1s_dit` | 16 | 304 | `ct_t1s_dit` | 264 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 399 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 502 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 491 |
| avx512 | `t1s_dit` | 32 | 608 | `ct_t1s_dit` | 494 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 823 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 994 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 980 |
| avx512 | `t1s_dit` | 64 | 1216 | `ct_t1s_dit` | 1013 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1608 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1956 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1878 |
| avx512 | `t1s_dit` | 128 | 2432 | `ct_t1s_dit` | 1974 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 3269 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4322 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 4021 |
| avx512 | `t1s_dit` | 256 | 4864 | `ct_t1s_dit` | 5040 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 6230 |