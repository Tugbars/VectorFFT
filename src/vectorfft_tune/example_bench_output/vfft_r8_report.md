# VectorFFT R=8 tuning report

Total measurements: **492**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 253 | 241 | 227 | — | t1s |
| avx2 | 64 | 72 | 251 | 241 | 226 | — | t1s |
| avx2 | 64 | 512 | 459 | 407 | 402 | — | t1s |
| avx2 | 128 | 128 | 481 | 475 | 468 | — | t1s |
| avx2 | 128 | 136 | 500 | 476 | 448 | — | t1s |
| avx2 | 128 | 1024 | 862 | 785 | 834 | — | log3 |
| avx2 | 256 | 256 | 1067 | 1006 | — | — | log3 |
| avx2 | 256 | 264 | 1076 | 929 | — | — | log3 |
| avx2 | 256 | 2048 | 3544 | 3265 | — | — | log3 |
| avx2 | 512 | 512 | 3642 | 4390 | — | — | flat |
| avx2 | 512 | 520 | 2106 | 1920 | — | — | log3 |
| avx2 | 512 | 4096 | 7249 | 6716 | — | — | log3 |
| avx2 | 1024 | 1024 | 7896 | 6355 | — | — | log3 |
| avx2 | 1024 | 1032 | 4161 | 3819 | — | — | log3 |
| avx2 | 1024 | 8192 | 14053 | 13623 | — | — | log3 |
| avx2 | 2048 | 2048 | 29289 | 27154 | — | — | log3 |
| avx2 | 2048 | 2056 | 11374 | 10444 | — | — | log3 |
| avx2 | 2048 | 16384 | 28952 | 26969 | — | — | log3 |
| avx512 | 64 | 64 | 155 | 156 | 135 | — | t1s |
| avx512 | 64 | 72 | 155 | 158 | 131 | — | t1s |
| avx512 | 64 | 512 | 351 | 325 | 339 | — | log3 |
| avx512 | 128 | 128 | 305 | 313 | 260 | — | t1s |
| avx512 | 128 | 136 | 312 | 313 | 259 | — | t1s |
| avx512 | 128 | 1024 | 728 | 663 | 679 | — | log3 |
| avx512 | 256 | 256 | 669 | 608 | — | — | log3 |
| avx512 | 256 | 264 | 707 | 624 | — | — | log3 |
| avx512 | 256 | 2048 | 1586 | 1538 | — | — | log3 |
| avx512 | 512 | 512 | 3032 | 2691 | — | — | log3 |
| avx512 | 512 | 520 | 1708 | 1594 | — | — | log3 |
| avx512 | 512 | 4096 | 3071 | 2945 | — | — | log3 |
| avx512 | 1024 | 1024 | 6037 | 5306 | — | — | log3 |
| avx512 | 1024 | 1032 | 3070 | 3184 | — | — | flat |
| avx512 | 1024 | 8192 | 6374 | 6064 | — | — | log3 |
| avx512 | 2048 | 2048 | 13085 | 11287 | — | — | log3 |
| avx512 | 2048 | 2056 | 10439 | 7533 | — | — | log3 |
| avx512 | 2048 | 16384 | 13094 | 11916 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 255 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 247 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 534 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 481 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 496 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 1033 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 1067 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 1076 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif_prefetch` | 4264 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 4318 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 2106 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif_prefetch` | 8770 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 8589 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 4161 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif_prefetch` | 17712 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif_prefetch` | 35228 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 11374 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 35687 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 253 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 251 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 459 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 497 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 500 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 862 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1089 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1146 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 3544 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 3642 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 2218 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 7249 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 7896 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 4333 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 14053 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 29289 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 12944 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 28952 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 241 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 241 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 407 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 475 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 476 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 785 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1006 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 929 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3265 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 4390 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1920 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 6716 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 6355 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 3819 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 13623 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 27154 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 10444 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 26969 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 227 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 226 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 402 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 468 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 448 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 834 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 158 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 155 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 471 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 305 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 323 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 943 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 669 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 707 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1949 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 3828 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1691 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 4129 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 7598 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3364 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 8217 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 16214 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 13185 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 15845 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 155 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 161 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit_u2` | 351 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 313 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 312 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_u2` | 728 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 701 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 792 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 1586 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 3032 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 1708 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 3071 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 6037 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 3070 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 6374 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 13085 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 10439 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 13094 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 156 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 158 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 325 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 313 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 313 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 663 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 608 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 624 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 1538 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 2691 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1594 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 2945 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 5306 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 3184 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 6064 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 11287 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 7533 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 11916 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 135 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 131 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 339 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 260 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 259 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 679 |