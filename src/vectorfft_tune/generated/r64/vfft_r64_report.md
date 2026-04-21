# VectorFFT R=64 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 4647 | 2538 | 2003 | — | t1s |
| avx2 | 64 | 72 | 3358 | 2178 | 1845 | — | t1s |
| avx2 | 64 | 512 | 5496 | 5329 | 5307 | — | t1s |
| avx2 | 128 | 128 | 10682 | 9744 | 8832 | — | t1s |
| avx2 | 128 | 136 | 5841 | 4796 | 3457 | — | t1s |
| avx2 | 128 | 1024 | 11651 | 10697 | 9564 | — | t1s |
| avx2 | 256 | 256 | 18863 | 19231 | — | — | flat |
| avx2 | 256 | 264 | 13386 | 9525 | — | — | log3 |
| avx2 | 256 | 2048 | 26093 | 24567 | — | — | log3 |
| avx2 | 512 | 512 | 39616 | 44145 | — | — | flat |
| avx2 | 512 | 520 | 25447 | 18546 | — | — | log3 |
| avx2 | 512 | 4096 | 57090 | 49897 | — | — | log3 |
| avx2 | 1024 | 1024 | 94656 | 84770 | — | — | log3 |
| avx2 | 1024 | 1032 | 53045 | 40104 | — | — | log3 |
| avx2 | 1024 | 8192 | 146153 | 166383 | — | — | flat |
| avx2 | 2048 | 2048 | 223827 | 195120 | — | — | log3 |
| avx2 | 2048 | 2056 | 120705 | 87799 | — | — | log3 |
| avx2 | 2048 | 16384 | 311653 | 394194 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 6380 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 6011 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 7464 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 12736 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile128_temporal` | 12545 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 12376 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile256_temporal` | 26282 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 24626 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 25657 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 51375 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile256_temporal` | 48973 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile256_temporal` | 59213 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 103067 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile256_temporal` | 102866 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 146153 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 241877 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile256_temporal` | 237381 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 311653 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 6141 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 3832 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 5496 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 13637 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 9660 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 14066 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 24441 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 13959 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 26716 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 46189 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 26430 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 60151 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 114648 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 75998 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 171133 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 289225 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 156364 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 359931 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 4647 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 3358 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5703 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 10682 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 5841 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 11651 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 18863 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 13386 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 26093 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 39616 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 25447 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 57090 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 94656 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 53045 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 185580 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 223827 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 120705 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 395450 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2538 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2178 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 5329 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9744 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4796 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 10697 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 19231 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 9525 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 24567 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 44145 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 18546 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 49897 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 84770 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 40104 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 166383 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 195120 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 87799 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 394194 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 2003 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1845 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 5307 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 8832 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3457 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 9564 |