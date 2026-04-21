# VectorFFT R=32 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 1041 | 866 | 828 | — | t1s |
| avx2 | 64 | 72 | 923 | 788 | 622 | — | t1s |
| avx2 | 64 | 512 | 2044 | 2308 | 1940 | — | t1s |
| avx2 | 128 | 128 | 2986 | 2111 | 2335 | — | log3 |
| avx2 | 128 | 136 | 2309 | 1755 | 1507 | — | t1s |
| avx2 | 128 | 1024 | 4620 | 4316 | 3594 | — | t1s |
| avx2 | 256 | 256 | 6929 | 6368 | — | — | log3 |
| avx2 | 256 | 264 | 5150 | 3758 | — | — | log3 |
| avx2 | 256 | 2048 | 8838 | 8233 | — | — | log3 |
| avx2 | 512 | 512 | 16448 | 17054 | — | — | flat |
| avx2 | 512 | 520 | 11439 | 8708 | — | — | log3 |
| avx2 | 512 | 4096 | 18249 | 19664 | — | — | flat |
| avx2 | 1024 | 1024 | 33112 | 30294 | — | — | log3 |
| avx2 | 1024 | 1032 | 24167 | 18484 | — | — | log3 |
| avx2 | 1024 | 8192 | 40792 | 37520 | — | — | log3 |
| avx2 | 2048 | 2048 | 68182 | 62300 | — | — | log3 |
| avx2 | 2048 | 2056 | 43505 | 33256 | — | — | log3 |
| avx2 | 2048 | 16384 | 96131 | 130875 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 1957 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 2077 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2640 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 3939 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 3849 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 4820 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 8291 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 8005 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 8838 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 18081 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 16732 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 18249 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 36306 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile256_temporal` | 34280 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile256_temporal` | 40792 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 68182 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 68336 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 96131 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1691 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1123 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2489 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 3848 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 3607 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4804 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 11409 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 6033 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 12720 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 19673 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 16500 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 25598 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 51585 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 24084 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 73566 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 101517 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 74814 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 167481 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1041 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 923 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2044 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2986 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2309 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4620 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6929 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5150 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 10004 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 16448 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 11439 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 20088 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 33112 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 24167 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 48234 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 77749 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 43505 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 151600 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 866 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 788 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 2308 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 2111 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1755 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 4316 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6368 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3758 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 8233 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 17054 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 8708 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 19664 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 30294 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 18484 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 37520 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 62300 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 33256 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 130875 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 828 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 622 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1940 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2335 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1507 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3594 |