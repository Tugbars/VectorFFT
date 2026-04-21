# VectorFFT R=16 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 292 | 316 | 298 | — | flat |
| avx2 | 64 | 72 | 273 | 314 | 292 | — | flat |
| avx2 | 64 | 512 | 819 | 956 | 844 | — | flat |
| avx2 | 128 | 128 | 717 | 623 | 562 | — | t1s |
| avx2 | 128 | 136 | 715 | 626 | 574 | — | t1s |
| avx2 | 128 | 1024 | 1705 | 1908 | 1699 | — | t1s |
| avx2 | 256 | 256 | 3061 | 3940 | — | — | flat |
| avx2 | 256 | 264 | 1609 | 1406 | — | — | log3 |
| avx2 | 256 | 2048 | 3600 | 3734 | — | — | flat |
| avx2 | 512 | 512 | 6909 | 7313 | — | — | flat |
| avx2 | 512 | 520 | 3596 | 3057 | — | — | log3 |
| avx2 | 512 | 4096 | 6316 | 7245 | — | — | flat |
| avx2 | 1024 | 1024 | 13191 | 13453 | — | — | flat |
| avx2 | 1024 | 1032 | 8454 | 6226 | — | — | log3 |
| avx2 | 1024 | 8192 | 13709 | 15714 | — | — | flat |
| avx2 | 2048 | 2048 | 28354 | 27521 | — | — | log3 |
| avx2 | 2048 | 2056 | 15307 | 11931 | — | — | log3 |
| avx2 | 2048 | 16384 | 30469 | 31461 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 421 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 438 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 819 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 1417 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 1353 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 1705 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 3061 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 2493 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 3688 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 6909 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 5815 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile128_temporal` | 6316 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 13191 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 11596 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 13709 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 28354 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 22220 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 30469 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 354 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 720 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1400 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1546 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 1586 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2190 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 4398 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 2457 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 4302 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 8482 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 4675 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 8556 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 19473 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 14269 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 23040 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 32943 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 17027 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 48268 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 292 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 273 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 957 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 717 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 715 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 1992 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4026 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1609 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 3600 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 7149 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 3596 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 6849 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 14772 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 8454 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 18177 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 37868 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 15307 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 38939 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 316 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 314 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 956 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 623 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 626 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1908 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3940 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1406 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3734 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 7313 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 3057 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7245 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 13453 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 6226 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 15714 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 27521 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 11931 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 31461 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 298 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 292 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 844 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 562 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 574 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1699 |