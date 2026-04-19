# VectorFFT R=16 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 284 | 309 | 240 | — | t1s |
| avx2 | 64 | 72 | 265 | 309 | 242 | — | t1s |
| avx2 | 64 | 512 | 915 | 856 | 841 | — | t1s |
| avx2 | 128 | 128 | 701 | 703 | 480 | — | t1s |
| avx2 | 128 | 136 | 741 | 671 | 482 | — | t1s |
| avx2 | 128 | 1024 | 1829 | 1656 | 1578 | — | t1s |
| avx2 | 256 | 256 | 3153 | 3529 | — | — | flat |
| avx2 | 256 | 264 | 1629 | 1381 | — | — | log3 |
| avx2 | 256 | 2048 | 3503 | 3725 | — | — | flat |
| avx2 | 512 | 512 | 6886 | 7374 | — | — | flat |
| avx2 | 512 | 520 | 3440 | 2894 | — | — | log3 |
| avx2 | 512 | 4096 | 6914 | 7128 | — | — | flat |
| avx2 | 1024 | 1024 | 12936 | 14713 | — | — | flat |
| avx2 | 1024 | 1032 | 7064 | 5918 | — | — | log3 |
| avx2 | 1024 | 8192 | 13531 | 16428 | — | — | flat |
| avx2 | 2048 | 2048 | 25838 | 27250 | — | — | flat |
| avx2 | 2048 | 2056 | 15011 | 11891 | — | — | log3 |
| avx2 | 2048 | 16384 | 32015 | 30853 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 428 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 452 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 902 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 1425 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 1332 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 1795 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 3153 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 2464 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 3503 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 7332 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 5355 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 6914 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 12936 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 11002 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 13531 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 25838 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 23329 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 32015 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 623 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 509 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1111 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1174 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 817 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2104 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 3437 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 3128 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 3988 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 10336 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 3518 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 8254 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 19231 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 7773 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 22492 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 32262 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 21546 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 39556 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 284 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 265 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 915 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 701 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 741 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 1829 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 3609 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1629 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 3716 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 6886 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 3440 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 7467 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 15234 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 7064 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 16073 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 30390 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 15011 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 35070 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 309 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 309 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 856 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 703 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 671 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1656 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3529 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1381 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3725 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 7374 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 2894 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7128 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 14713 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 5918 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 16428 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 27250 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 11891 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 30853 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 240 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 242 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 841 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 480 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 482 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1578 |