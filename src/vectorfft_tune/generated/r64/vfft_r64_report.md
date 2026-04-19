# VectorFFT R=64 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 3499 | 2626 | 1918 | — | t1s |
| avx2 | 64 | 72 | 2854 | 2102 | 1824 | — | t1s |
| avx2 | 64 | 512 | 5388 | 5271 | 5536 | — | log3 |
| avx2 | 128 | 128 | 10412 | 9035 | 9323 | — | log3 |
| avx2 | 128 | 136 | 5682 | 4716 | 3596 | — | t1s |
| avx2 | 128 | 1024 | 11645 | 11073 | 9843 | — | t1s |
| avx2 | 256 | 256 | 17905 | 17897 | — | — | log3 |
| avx2 | 256 | 264 | 10814 | 9837 | — | — | log3 |
| avx2 | 256 | 2048 | 23956 | 24747 | — | — | flat |
| avx2 | 512 | 512 | 39712 | 42266 | — | — | flat |
| avx2 | 512 | 520 | 21368 | 20363 | — | — | log3 |
| avx2 | 512 | 4096 | 56653 | 67855 | — | — | flat |
| avx2 | 1024 | 1024 | 86777 | 79149 | — | — | log3 |
| avx2 | 1024 | 1032 | 46499 | 43445 | — | — | log3 |
| avx2 | 1024 | 8192 | 135962 | 169786 | — | — | flat |
| avx2 | 2048 | 2048 | 251772 | 202512 | — | — | log3 |
| avx2 | 2048 | 2056 | 161742 | 78639 | — | — | log3 |
| avx2 | 2048 | 16384 | 285938 | 353212 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 5735 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 5979 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 7251 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 12941 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 11991 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 12700 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile256_temporal` | 24146 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile256_temporal` | 23411 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 23956 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 48913 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile256_temporal` | 46617 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile256_temporal` | 60364 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 101584 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile256_temporal` | 96438 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile256_temporal` | 135962 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 251206 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile256_temporal` | 238144 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 285938 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 3938 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 5213 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 5843 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 10412 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 9931 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12091 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 19426 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 14183 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 30912 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 54365 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 34905 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 66491 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 111015 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 71988 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 164888 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 295319 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 195561 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 331366 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 3499 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2854 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5388 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 11034 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 5682 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 11645 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 17905 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 10814 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 26336 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 39712 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 21368 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 56653 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 86777 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 46499 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 174139 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 251772 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 161742 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 366581 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2626 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2102 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 5271 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9035 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4716 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 11073 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 17897 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 9837 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 24747 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 42266 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 20363 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 67855 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 79149 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 43445 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 169786 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 202512 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 78639 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 353212 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1918 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1824 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 5536 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 9323 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3596 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 9843 |