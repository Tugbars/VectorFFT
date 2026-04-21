# VectorFFT R=16 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 294 | 321 | 249 | — | t1s |
| avx2 | 64 | 72 | 279 | 311 | 288 | — | flat |
| avx2 | 64 | 512 | 837 | 860 | 839 | — | flat |
| avx2 | 128 | 128 | 656 | 656 | 575 | — | t1s |
| avx2 | 128 | 136 | 655 | 613 | 496 | — | t1s |
| avx2 | 128 | 1024 | 1839 | 1739 | 1640 | — | t1s |
| avx2 | 256 | 256 | 3146 | 3773 | — | — | flat |
| avx2 | 256 | 264 | 1719 | 1576 | — | — | log3 |
| avx2 | 256 | 2048 | 3683 | 3696 | — | — | flat |
| avx2 | 512 | 512 | 7229 | 7647 | — | — | flat |
| avx2 | 512 | 520 | 3911 | 3338 | — | — | log3 |
| avx2 | 512 | 4096 | 6815 | 7765 | — | — | flat |
| avx2 | 1024 | 1024 | 14701 | 16377 | — | — | flat |
| avx2 | 1024 | 1032 | 6905 | 7108 | — | — | flat |
| avx2 | 1024 | 8192 | 15602 | 16748 | — | — | flat |
| avx2 | 2048 | 2048 | 27585 | 32046 | — | — | flat |
| avx2 | 2048 | 2056 | 15165 | 12598 | — | — | log3 |
| avx2 | 2048 | 16384 | 36237 | 34643 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 478 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 496 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 995 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 1574 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 1357 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 1852 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 3146 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 2575 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 3777 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 7244 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 6053 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile128_temporal` | 6815 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 14701 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 12821 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 15602 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 27585 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 25187 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 36237 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 325 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 327 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1143 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 734 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 807 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2271 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 3481 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 1807 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 5140 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 10587 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 5763 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 8474 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 16677 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 11535 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 19367 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 32347 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 23174 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 46281 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 294 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 279 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 837 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 656 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 655 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 1839 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4296 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1719 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 3683 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 7229 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 3911 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 7452 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 15605 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 6905 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 17568 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 31862 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 15165 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 37309 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 321 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 311 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 860 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 656 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 613 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1739 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3773 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1576 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3696 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 7647 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 3338 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7765 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 16377 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 7108 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 16748 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 32046 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 12598 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 34643 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 249 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 288 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 839 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 575 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 496 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1640 |