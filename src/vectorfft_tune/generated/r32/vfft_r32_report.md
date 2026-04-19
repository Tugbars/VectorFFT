# VectorFFT R=32 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 842 | 1139 | 662 | — | t1s |
| avx2 | 64 | 72 | 1058 | 791 | 728 | — | t1s |
| avx2 | 64 | 512 | 1899 | 1908 | 2009 | — | flat |
| avx2 | 128 | 128 | 3209 | 3040 | 2408 | — | t1s |
| avx2 | 128 | 136 | 2204 | 1738 | 2113 | — | log3 |
| avx2 | 128 | 1024 | 3780 | 3798 | 3698 | — | t1s |
| avx2 | 256 | 256 | 6482 | 7790 | — | — | flat |
| avx2 | 256 | 264 | 4279 | 3920 | — | — | log3 |
| avx2 | 256 | 2048 | 7964 | 9118 | — | — | flat |
| avx2 | 512 | 512 | 18110 | 17763 | — | — | log3 |
| avx2 | 512 | 520 | 11388 | 7840 | — | — | log3 |
| avx2 | 512 | 4096 | 17954 | 22309 | — | — | flat |
| avx2 | 1024 | 1024 | 35336 | 40845 | — | — | flat |
| avx2 | 1024 | 1032 | 21084 | 17347 | — | — | log3 |
| avx2 | 1024 | 8192 | 43576 | 44687 | — | — | flat |
| avx2 | 2048 | 2048 | 64715 | 73367 | — | — | flat |
| avx2 | 2048 | 2056 | 42555 | 40225 | — | — | log3 |
| avx2 | 2048 | 16384 | 117681 | 128747 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 1872 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 1921 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2508 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 3892 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 4018 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 5015 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 7981 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 8040 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 8797 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 18305 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 16269 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 17954 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 36301 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 31785 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 43576 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 67632 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 69492 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 131718 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1657 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1219 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2389 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4699 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 3927 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4867 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 9349 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 5836 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 10579 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 21175 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 16526 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 25391 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 42655 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 34829 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 87667 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 109051 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 68778 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 156270 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 842 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1058 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 1899 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3209 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2204 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 3780 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6482 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4279 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 7964 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 18110 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 11388 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 19561 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 35336 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 21084 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 50357 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 64715 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 42555 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 117681 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1139 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 791 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1908 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3040 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1738 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3798 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 7790 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3920 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 9118 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 17763 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 7840 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 22309 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 40845 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 17347 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 44687 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 73367 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 40225 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 128747 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 662 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 728 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 2009 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2408 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2113 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3698 |