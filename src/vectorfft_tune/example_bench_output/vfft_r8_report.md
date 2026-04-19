# VectorFFT R=8 tuning report

Total measurements: **492**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 254 | 236 | 250 | — | log3 |
| avx2 | 64 | 72 | 247 | 232 | 228 | — | t1s |
| avx2 | 64 | 512 | 425 | 393 | 412 | — | log3 |
| avx2 | 128 | 128 | 493 | 467 | 457 | — | t1s |
| avx2 | 128 | 136 | 486 | 460 | 457 | — | t1s |
| avx2 | 128 | 1024 | 850 | 793 | 851 | — | log3 |
| avx2 | 256 | 256 | 1043 | 967 | — | — | log3 |
| avx2 | 256 | 264 | 989 | 942 | — | — | log3 |
| avx2 | 256 | 2048 | 3615 | 3372 | — | — | log3 |
| avx2 | 512 | 512 | 3483 | 3184 | — | — | log3 |
| avx2 | 512 | 520 | 2063 | 1988 | — | — | log3 |
| avx2 | 512 | 4096 | 7494 | 7048 | — | — | log3 |
| avx2 | 1024 | 1024 | 7324 | 6531 | — | — | log3 |
| avx2 | 1024 | 1032 | 4109 | 4110 | — | — | flat |
| avx2 | 1024 | 8192 | 14533 | 13328 | — | — | log3 |
| avx2 | 2048 | 2048 | 29346 | 27331 | — | — | log3 |
| avx2 | 2048 | 2056 | 11780 | 11216 | — | — | log3 |
| avx2 | 2048 | 16384 | 29331 | 27497 | — | — | log3 |
| avx512 | 64 | 64 | 159 | 156 | 130 | — | t1s |
| avx512 | 64 | 72 | 154 | 158 | 130 | — | t1s |
| avx512 | 64 | 512 | 365 | 342 | 416 | — | log3 |
| avx512 | 128 | 128 | 309 | 321 | 266 | — | t1s |
| avx512 | 128 | 136 | 301 | 320 | 257 | — | t1s |
| avx512 | 128 | 1024 | 717 | 659 | 679 | — | log3 |
| avx512 | 256 | 256 | 646 | 616 | — | — | log3 |
| avx512 | 256 | 264 | 726 | 624 | — | — | log3 |
| avx512 | 256 | 2048 | 1591 | 1432 | — | — | log3 |
| avx512 | 512 | 512 | 3052 | 2692 | — | — | log3 |
| avx512 | 512 | 520 | 1650 | 1592 | — | — | log3 |
| avx512 | 512 | 4096 | 3235 | 2843 | — | — | log3 |
| avx512 | 1024 | 1024 | 6103 | 5220 | — | — | log3 |
| avx512 | 1024 | 1032 | 3111 | 3209 | — | — | flat |
| avx512 | 1024 | 8192 | 6821 | 6041 | — | — | log3 |
| avx512 | 2048 | 2048 | 13222 | 11645 | — | — | log3 |
| avx512 | 2048 | 2056 | 10772 | 8245 | — | — | log3 |
| avx512 | 2048 | 16384 | 13196 | 11188 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 250 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 247 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 555 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 493 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 486 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 1040 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 1043 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 989 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif_prefetch` | 4377 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif_prefetch` | 4703 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 2063 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif_prefetch` | 8764 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 8365 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 4109 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif_prefetch` | 17615 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif_prefetch` | 35452 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 14370 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 36017 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 254 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 259 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 425 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 507 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 510 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 850 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1073 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1082 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 3615 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 3483 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 2198 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 7494 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 7324 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 4319 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 14533 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 29346 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 11780 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 29331 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 236 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 232 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 393 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 467 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 460 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 793 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 967 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 942 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3372 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 3184 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1988 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7048 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 6531 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 4110 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 13328 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 27331 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 11216 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 27497 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 250 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 228 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 412 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 457 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 457 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 851 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 157 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 154 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 474 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 309 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 301 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 957 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 646 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 726 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 2016 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 3912 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1650 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 4030 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 7782 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3111 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 8044 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 16442 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 12142 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 16276 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 161 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 160 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit_u2` | 365 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 317 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 314 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 717 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 692 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 784 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 1591 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 3052 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 1732 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 3235 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 6103 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 3464 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 6821 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 13222 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 10772 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 13196 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 156 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 158 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 342 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 321 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 320 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 659 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 616 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 624 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 1432 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 2692 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1592 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 2843 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 5220 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 3209 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 6041 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 11645 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 8245 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 11188 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 130 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 130 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 416 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 266 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 257 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 679 |