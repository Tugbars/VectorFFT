# VectorFFT R=32 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 870 | 791 | 705 | — | t1s |
| avx2 | 64 | 72 | 865 | 801 | 656 | — | t1s |
| avx2 | 64 | 512 | 2122 | 1947 | 1862 | — | t1s |
| avx2 | 128 | 128 | 3608 | 3284 | 2261 | — | t1s |
| avx2 | 128 | 136 | 2569 | 2048 | 1432 | — | t1s |
| avx2 | 128 | 1024 | 3802 | 3892 | 3852 | — | flat |
| avx2 | 256 | 256 | 6620 | 6360 | — | — | log3 |
| avx2 | 256 | 264 | 4367 | 4266 | — | — | log3 |
| avx2 | 256 | 2048 | 8657 | 7964 | — | — | log3 |
| avx2 | 512 | 512 | 18074 | 18354 | — | — | flat |
| avx2 | 512 | 520 | 10370 | 8366 | — | — | log3 |
| avx2 | 512 | 4096 | 18226 | 19417 | — | — | flat |
| avx2 | 1024 | 1024 | 31036 | 36587 | — | — | flat |
| avx2 | 1024 | 1032 | 21030 | 16638 | — | — | log3 |
| avx2 | 1024 | 8192 | 40419 | 42700 | — | — | flat |
| avx2 | 2048 | 2048 | 67233 | 71480 | — | — | flat |
| avx2 | 2048 | 2056 | 43634 | 36125 | — | — | log3 |
| avx2 | 2048 | 16384 | 131077 | 117948 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 2035 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 2035 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2440 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 5235 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 4068 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 4521 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 8326 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 8317 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 8657 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 18074 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 15709 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 18226 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 35750 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 32628 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 41820 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 67233 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 67082 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 131077 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1098 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1645 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2704 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 3921 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 2569 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4806 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 10902 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 5117 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 9629 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 19401 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 10713 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 28197 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 38223 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 23956 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 52652 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 79838 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 48845 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 169019 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 870 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 865 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2122 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3608 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2661 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 3802 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6620 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4367 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 10365 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 19905 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 10370 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 25035 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 31036 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 21030 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 40419 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 78965 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 43634 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 137053 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 791 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 801 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1947 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3284 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2048 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3892 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6360 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 4266 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 7964 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 18354 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 8366 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 19417 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 36587 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 16638 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 42700 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 71480 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 36125 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 117948 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 705 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 656 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1862 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2261 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1432 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3852 |