# VectorFFT R=64 tuning report

Total measurements: **222**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 4208 | 2536 | 1986 | — | t1s |
| avx2 | 64 | 72 | 2910 | 2409 | 1822 | — | t1s |
| avx2 | 64 | 512 | 5046 | 4719 | 5030 | — | log3 |
| avx2 | 128 | 128 | 9875 | 8811 | 9140 | — | log3 |
| avx2 | 128 | 136 | 6585 | 4903 | 4790 | — | t1s |
| avx2 | 128 | 1024 | 10286 | 9758 | 10899 | — | log3 |
| avx2 | 256 | 256 | 18224 | 15560 | — | — | log3 |
| avx2 | 256 | 264 | 13020 | 10856 | — | — | log3 |
| avx2 | 256 | 2048 | 27266 | 25200 | — | — | log3 |
| avx2 | 512 | 512 | 41511 | 42389 | — | — | flat |
| avx2 | 512 | 520 | 25101 | 20293 | — | — | log3 |
| avx2 | 512 | 4096 | 68160 | 64750 | — | — | log3 |
| avx2 | 1024 | 1024 | 88548 | 85343 | — | — | log3 |
| avx2 | 1024 | 1032 | 54008 | 39751 | — | — | log3 |
| avx2 | 1024 | 8192 | 172222 | 146220 | — | — | log3 |
| avx2 | 2048 | 2048 | 254595 | 202425 | — | — | log3 |
| avx2 | 2048 | 2056 | 172469 | 91276 | — | — | log3 |
| avx2 | 2048 | 16384 | 322400 | 356353 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 7159 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 6463 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 7577 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 14183 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile128_temporal` | 13281 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 15448 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile128_temporal` | 27526 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 25453 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 31137 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 56820 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile128_temporal` | 53505 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 70750 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 125655 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile128_temporal` | 110480 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 172222 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 303197 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 293186 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 322400 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 6217 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 4810 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 6634 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 13126 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 10157 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12669 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 24161 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 20210 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 28813 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 55633 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 28380 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 68160 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 116276 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 77015 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 176978 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 305834 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 211695 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 376612 |
| avx2 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 3931 |
| avx2 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 3101 |
| avx2 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 4719 |
| avx2 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 8811 |
| avx2 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 5054 |
| avx2 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 9758 |
| avx2 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 15560 |
| avx2 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 13808 |
| avx2 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 27733 |
| avx2 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 46047 |
| avx2 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 26926 |
| avx2 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 79258 |
| avx2 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 97513 |
| avx2 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 61775 |
| avx2 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 146220 |
| avx2 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 231989 |
| avx2 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 108255 |
| avx2 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 356353 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 4208 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2910 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5046 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 9875 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 6585 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 10286 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 18224 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 13020 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 27266 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 41511 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 25101 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 74368 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 88548 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 54008 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 189339 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 254595 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 172469 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 408500 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2536 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2409 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 5498 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9930 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4903 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 10760 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 20155 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 10856 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 25200 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 42389 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 20293 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 64750 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 85343 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 39751 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 190430 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 202425 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 91276 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 414225 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1986 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1822 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 5030 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 9140 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 4790 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 10899 |