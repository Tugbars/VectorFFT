# VectorFFT R=64 tuning report

Total measurements: **402**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 4346 | 2583 | 2542 | — | t1s |
| avx2 | 64 | 72 | 2451 | 2450 | 1867 | — | t1s |
| avx2 | 64 | 512 | 4873 | 5427 | 4731 | — | t1s |
| avx2 | 96 | 96 | 3512 | 3767 | 3110 | — | t1s |
| avx2 | 96 | 104 | 3431 | 3674 | 2707 | — | t1s |
| avx2 | 96 | 768 | 6967 | 6985 | 7431 | — | flat |
| avx2 | 128 | 128 | 9933 | 9599 | 9048 | — | t1s |
| avx2 | 128 | 136 | 6773 | 4954 | 3877 | — | t1s |
| avx2 | 128 | 1024 | 11273 | 10937 | 10020 | — | t1s |
| avx2 | 192 | 192 | 9407 | 8295 | — | — | log3 |
| avx2 | 192 | 200 | 9574 | 7620 | — | — | log3 |
| avx2 | 192 | 1536 | 16055 | 16193 | — | — | flat |
| avx2 | 256 | 256 | 19505 | 20182 | — | — | flat |
| avx2 | 256 | 264 | 12797 | 9915 | — | — | log3 |
| avx2 | 256 | 2048 | 26068 | 24025 | — | — | log3 |
| avx2 | 384 | 384 | 32684 | 29857 | — | — | log3 |
| avx2 | 384 | 392 | 19542 | 15170 | — | — | log3 |
| avx2 | 384 | 3072 | 40304 | 35138 | — | — | log3 |
| avx2 | 512 | 512 | 45263 | 41818 | — | — | log3 |
| avx2 | 512 | 520 | 23851 | 18932 | — | — | log3 |
| avx2 | 512 | 4096 | 60362 | 49457 | — | — | log3 |
| avx2 | 768 | 768 | 61079 | 56160 | — | — | log3 |
| avx2 | 768 | 776 | 34123 | 30422 | — | — | log3 |
| avx2 | 768 | 6144 | 76195 | 71559 | — | — | log3 |
| avx2 | 1024 | 1024 | 91600 | 84084 | — | — | log3 |
| avx2 | 1024 | 1032 | 54893 | 40151 | — | — | log3 |
| avx2 | 1024 | 8192 | 161673 | 184305 | — | — | flat |
| avx2 | 1536 | 1536 | 149532 | 120912 | — | — | log3 |
| avx2 | 1536 | 1544 | 99339 | 61530 | — | — | log3 |
| avx2 | 1536 | 12288 | 198395 | 155503 | — | — | log3 |
| avx2 | 2048 | 2048 | 239202 | 260167 | — | — | flat |
| avx2 | 2048 | 2056 | 125922 | 88250 | — | — | log3 |
| avx2 | 2048 | 16384 | 326747 | 399938 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 6205 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 6381 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 7177 |
| avx2 | `t1_buf_dit` | 96 | 96 | `ct_t1_buf_dit_tile64_temporal` | 7493 |
| avx2 | `t1_buf_dit` | 96 | 104 | `ct_t1_buf_dit_tile64_temporal` | 7488 |
| avx2 | `t1_buf_dit` | 96 | 768 | `ct_t1_buf_dit_tile64_temporal` | 8383 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 13289 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 12752 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 12975 |
| avx2 | `t1_buf_dit` | 192 | 192 | `ct_t1_buf_dit_tile128_temporal` | 16877 |
| avx2 | `t1_buf_dit` | 192 | 200 | `ct_t1_buf_dit_tile128_temporal` | 15534 |
| avx2 | `t1_buf_dit` | 192 | 1536 | `ct_t1_buf_dit_tile128_temporal` | 17328 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 26924 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile128_temporal` | 26013 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 29023 |
| avx2 | `t1_buf_dit` | 384 | 384 | `ct_t1_buf_dit_tile64_temporal` | 38304 |
| avx2 | `t1_buf_dit` | 384 | 392 | `ct_t1_buf_dit_tile64_temporal` | 43219 |
| avx2 | `t1_buf_dit` | 384 | 3072 | `ct_t1_buf_dit_tile128_temporal` | 40509 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 52410 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile128_temporal` | 52238 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile128_temporal` | 60362 |
| avx2 | `t1_buf_dit` | 768 | 768 | `ct_t1_buf_dit_tile64_temporal` | 77623 |
| avx2 | `t1_buf_dit` | 768 | 776 | `ct_t1_buf_dit_tile64_temporal` | 73191 |
| avx2 | `t1_buf_dit` | 768 | 6144 | `ct_t1_buf_dit_tile128_temporal` | 80422 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 114118 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 102317 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 161673 |
| avx2 | `t1_buf_dit` | 1536 | 1536 | `ct_t1_buf_dit_tile64_temporal` | 189155 |
| avx2 | `t1_buf_dit` | 1536 | 1544 | `ct_t1_buf_dit_tile64_temporal` | 168070 |
| avx2 | `t1_buf_dit` | 1536 | 12288 | `ct_t1_buf_dit_tile64_temporal` | 201952 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 276205 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 237084 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 326747 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 7123 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 3902 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 7294 |
| avx2 | `t1_dif` | 96 | 96 | `ct_t1_dif` | 6117 |
| avx2 | `t1_dif` | 96 | 104 | `ct_t1_dif` | 7198 |
| avx2 | `t1_dif` | 96 | 768 | `ct_t1_dif` | 8641 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 10465 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 8089 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 14398 |
| avx2 | `t1_dif` | 192 | 192 | `ct_t1_dif` | 17541 |
| avx2 | `t1_dif` | 192 | 200 | `ct_t1_dif` | 15715 |
| avx2 | `t1_dif` | 192 | 1536 | `ct_t1_dif` | 21356 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 24718 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 19183 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 28117 |
| avx2 | `t1_dif` | 384 | 384 | `ct_t1_dif` | 39188 |
| avx2 | `t1_dif` | 384 | 392 | `ct_t1_dif` | 24951 |
| avx2 | `t1_dif` | 384 | 3072 | `ct_t1_dif` | 47564 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 46545 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 32911 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 67598 |
| avx2 | `t1_dif` | 768 | 768 | `ct_t1_dif` | 75188 |
| avx2 | `t1_dif` | 768 | 776 | `ct_t1_dif` | 54469 |
| avx2 | `t1_dif` | 768 | 6144 | `ct_t1_dif` | 93248 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 117200 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 60370 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 166534 |
| avx2 | `t1_dif` | 1536 | 1536 | `ct_t1_dif` | 183705 |
| avx2 | `t1_dif` | 1536 | 1544 | `ct_t1_dif` | 105297 |
| avx2 | `t1_dif` | 1536 | 12288 | `ct_t1_dif` | 279691 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 320981 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 196422 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 380672 |
| avx2 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 2675 |
| avx2 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 3205 |
| avx2 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 6010 |
| avx2 | `t1_dif_log3` | 96 | 96 | `ct_t1_dif_log3` | 4960 |
| avx2 | `t1_dif_log3` | 96 | 104 | `ct_t1_dif_log3` | 4446 |
| avx2 | `t1_dif_log3` | 96 | 768 | `ct_t1_dif_log3` | 6833 |
| avx2 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 8708 |
| avx2 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 6118 |
| avx2 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 9626 |
| avx2 | `t1_dif_log3` | 192 | 192 | `ct_t1_dif_log3` | 8398 |
| avx2 | `t1_dif_log3` | 192 | 200 | `ct_t1_dif_log3` | 7944 |
| avx2 | `t1_dif_log3` | 192 | 1536 | `ct_t1_dif_log3` | 15149 |
| avx2 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 20886 |
| avx2 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 10447 |
| avx2 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 26131 |
| avx2 | `t1_dif_log3` | 384 | 384 | `ct_t1_dif_log3` | 32600 |
| avx2 | `t1_dif_log3` | 384 | 392 | `ct_t1_dif_log3` | 19051 |
| avx2 | `t1_dif_log3` | 384 | 3072 | `ct_t1_dif_log3` | 38606 |
| avx2 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 43713 |
| avx2 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 27744 |
| avx2 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 53948 |
| avx2 | `t1_dif_log3` | 768 | 768 | `ct_t1_dif_log3` | 56151 |
| avx2 | `t1_dif_log3` | 768 | 776 | `ct_t1_dif_log3` | 38493 |
| avx2 | `t1_dif_log3` | 768 | 6144 | `ct_t1_dif_log3` | 68564 |
| avx2 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 96420 |
| avx2 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 45184 |
| avx2 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 153395 |
| avx2 | `t1_dif_log3` | 1536 | 1536 | `ct_t1_dif_log3` | 113273 |
| avx2 | `t1_dif_log3` | 1536 | 1544 | `ct_t1_dif_log3` | 80789 |
| avx2 | `t1_dif_log3` | 1536 | 12288 | `ct_t1_dif_log3` | 181144 |
| avx2 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 175872 |
| avx2 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 91626 |
| avx2 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 296703 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 4346 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2451 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 4873 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit` | 3512 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit` | 3431 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit` | 6967 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 9933 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 6773 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 11273 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit` | 9407 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit` | 9574 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit` | 16055 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 19505 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 12797 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 26068 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit` | 32684 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit` | 19542 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit` | 40304 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 45263 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 23851 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 93645 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit` | 61079 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit` | 34123 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit` | 76195 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 91600 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 54893 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 198192 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit` | 149532 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit` | 99339 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit` | 198395 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 239202 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 125922 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 402644 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2583 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2450 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 5427 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3` | 3767 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3` | 3674 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3` | 6985 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9599 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4954 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 10937 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3` | 8295 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3` | 7620 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3` | 16193 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 20182 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 9915 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 24025 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3` | 29857 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3` | 15170 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3` | 35138 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 41818 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 18932 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 49457 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3` | 56160 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3` | 30422 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3` | 71559 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 84084 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 40151 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 184305 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3` | 120912 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3` | 61530 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3` | 155503 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 260167 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 88250 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 399938 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 2542 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1867 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 4731 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit` | 3110 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit` | 2707 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit` | 7431 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 9048 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3877 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 10020 |