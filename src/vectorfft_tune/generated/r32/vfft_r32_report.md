# VectorFFT R=32 tuning report

Total measurements: **600**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 2249 | 1956 | 1720 | — | t1s |
| avx2 | 64 | 72 | 2198 | 1911 | 1719 | — | t1s |
| avx2 | 64 | 512 | 5194 | 4880 | 4869 | — | t1s |
| avx2 | 128 | 128 | 4682 | 4046 | 3745 | — | t1s |
| avx2 | 128 | 136 | 4636 | 3993 | 3472 | — | t1s |
| avx2 | 128 | 1024 | 10553 | 9962 | 9777 | — | t1s |
| avx2 | 256 | 256 | 16491 | 15012 | — | — | log3 |
| avx2 | 256 | 264 | 9806 | 7973 | — | — | log3 |
| avx2 | 256 | 2048 | 21261 | 20298 | — | — | log3 |
| avx2 | 512 | 512 | 43569 | 39854 | — | — | log3 |
| avx2 | 512 | 520 | 22015 | 16477 | — | — | log3 |
| avx2 | 512 | 4096 | 42500 | 39751 | — | — | log3 |
| avx2 | 1024 | 1024 | 90367 | 77351 | — | — | log3 |
| avx2 | 1024 | 1032 | 44438 | 34442 | — | — | log3 |
| avx2 | 1024 | 8192 | 84814 | 80616 | — | — | log3 |
| avx2 | 2048 | 2048 | 186442 | 160091 | — | — | log3 |
| avx2 | 2048 | 2056 | 100366 | 72098 | — | — | log3 |
| avx2 | 2048 | 16384 | 187939 | 163660 | — | — | log3 |
| avx512 | 64 | 64 | 1424 | 1048 | 935 | — | t1s |
| avx512 | 64 | 72 | 1445 | 1057 | 948 | — | t1s |
| avx512 | 64 | 512 | 2643 | 2282 | 2320 | — | log3 |
| avx512 | 128 | 128 | 3548 | 3057 | 3251 | — | log3 |
| avx512 | 128 | 136 | 2952 | 2242 | 2074 | — | t1s |
| avx512 | 128 | 1024 | 5087 | 4411 | 4584 | — | log3 |
| avx512 | 256 | 256 | 8428 | 6735 | — | — | log3 |
| avx512 | 256 | 264 | 6050 | 4566 | — | — | log3 |
| avx512 | 256 | 2048 | 10237 | 9466 | — | — | log3 |
| avx512 | 512 | 512 | 20944 | 18151 | — | — | log3 |
| avx512 | 512 | 520 | 13955 | 9399 | — | — | log3 |
| avx512 | 512 | 4096 | 20998 | 18764 | — | — | log3 |
| avx512 | 1024 | 1024 | 41254 | 35373 | — | — | log3 |
| avx512 | 1024 | 1032 | 28314 | 19704 | — | — | log3 |
| avx512 | 1024 | 8192 | 40765 | 38094 | — | — | log3 |
| avx512 | 2048 | 2048 | 92569 | 73933 | — | — | log3 |
| avx512 | 2048 | 2056 | 63155 | 38007 | — | — | log3 |
| avx512 | 2048 | 16384 | 96140 | 76779 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 4865 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 4864 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 6314 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 9543 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 9870 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 11460 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 18123 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 19418 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 21261 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 43569 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 40065 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile256_temporal` | 42500 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 89407 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 78858 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile256_temporal` | 84814 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 186442 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 164974 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 187939 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 2220 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 2253 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 5973 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4772 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 4555 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12293 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 19031 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 10444 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 24510 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 48097 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 22311 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 48562 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 99559 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 45419 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 101276 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 222735 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 106714 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 221363 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 2249 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2198 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5194 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 4682 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 4636 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 10553 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 16491 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 9806 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 22435 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 45070 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 22015 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 44827 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 90367 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 44438 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 91630 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 195707 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 100366 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 196111 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1956 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1911 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 4880 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 4046 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 3993 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 9962 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 15012 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 7973 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 20298 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 39854 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 16477 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 39751 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 77351 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 34442 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 80616 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 160091 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 72098 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 163660 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1720 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1719 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 4869 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3745 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3472 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 9777 |
| avx512 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 3039 |
| avx512 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 2920 |
| avx512 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 3118 |
| avx512 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 6013 |
| avx512 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile128_temporal` | 5825 |
| avx512 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 5931 |
| avx512 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 10844 |
| avx512 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile128_temporal` | 11938 |
| avx512 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 11433 |
| avx512 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 22842 |
| avx512 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 24680 |
| avx512 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 22897 |
| avx512 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 46748 |
| avx512 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 49093 |
| avx512 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 46257 |
| avx512 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 103644 |
| avx512 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 110156 |
| avx512 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 106950 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1464 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1471 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2874 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4869 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 2997 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 5936 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 10850 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 6461 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 11836 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 24209 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 14459 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 24316 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 53162 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 29212 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 54128 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 120244 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 71725 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 126760 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1424 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1445 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2643 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3548 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2952 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 5087 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 8428 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 6050 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 10237 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 20944 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 13955 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 20998 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 41254 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 28314 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 40765 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 92569 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 63155 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 96140 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1048 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1057 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 2282 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3057 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2242 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 4411 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6735 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 4566 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 9466 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 18151 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 9399 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 18764 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 35373 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 19704 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 38094 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 73933 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 38007 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 76779 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 935 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 948 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 2320 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3251 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2074 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 4584 |