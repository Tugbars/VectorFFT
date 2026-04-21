# VectorFFT R=16 tuning report

Total measurements: **600**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 707 | 650 | 706 | — | log3 |
| avx2 | 64 | 72 | 726 | 657 | 689 | — | log3 |
| avx2 | 64 | 512 | 1790 | 1770 | 1598 | — | t1s |
| avx2 | 128 | 128 | 1504 | 1302 | 1383 | — | log3 |
| avx2 | 128 | 136 | 1507 | 1299 | 1387 | — | log3 |
| avx2 | 128 | 1024 | 3832 | 3916 | 3771 | — | t1s |
| avx2 | 256 | 256 | 5552 | 4992 | — | — | log3 |
| avx2 | 256 | 264 | 3036 | 2875 | — | — | log3 |
| avx2 | 256 | 2048 | 7172 | 7917 | — | — | flat |
| avx2 | 512 | 512 | 13718 | 14736 | — | — | flat |
| avx2 | 512 | 520 | 6700 | 5869 | — | — | log3 |
| avx2 | 512 | 4096 | 15107 | 15728 | — | — | flat |
| avx2 | 1024 | 1024 | 27842 | 31697 | — | — | flat |
| avx2 | 1024 | 1032 | 14766 | 11536 | — | — | log3 |
| avx2 | 1024 | 8192 | 30951 | 31781 | — | — | flat |
| avx2 | 2048 | 2048 | 62411 | 63424 | — | — | flat |
| avx2 | 2048 | 2056 | 31948 | 25195 | — | — | log3 |
| avx2 | 2048 | 16384 | 63299 | 63844 | — | — | flat |
| avx512 | 64 | 64 | 366 | 425 | 384 | — | flat |
| avx512 | 64 | 72 | 371 | 414 | 379 | — | flat |
| avx512 | 64 | 512 | 886 | 691 | 882 | — | log3 |
| avx512 | 128 | 128 | 958 | 825 | 746 | — | t1s |
| avx512 | 128 | 136 | 977 | 836 | 849 | — | log3 |
| avx512 | 128 | 1024 | 2020 | 1795 | 1859 | — | log3 |
| avx512 | 256 | 256 | 3639 | 2868 | — | — | log3 |
| avx512 | 256 | 264 | 2146 | 1766 | — | — | log3 |
| avx512 | 256 | 2048 | 4531 | 3691 | — | — | log3 |
| avx512 | 512 | 512 | 7701 | 5586 | — | — | log3 |
| avx512 | 512 | 520 | 4422 | 3792 | — | — | log3 |
| avx512 | 512 | 4096 | 8999 | 7372 | — | — | log3 |
| avx512 | 1024 | 1024 | 18298 | 14343 | — | — | log3 |
| avx512 | 1024 | 1032 | 11341 | 7344 | — | — | log3 |
| avx512 | 1024 | 8192 | 17928 | 14933 | — | — | log3 |
| avx512 | 2048 | 2048 | 36965 | 29377 | — | — | log3 |
| avx512 | 2048 | 2056 | 21624 | 15250 | — | — | log3 |
| avx512 | 2048 | 16384 | 36109 | 31328 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 1115 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 1159 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2057 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 3020 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 2921 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 3832 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 7163 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 5899 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 7172 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 13718 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 12263 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile128_temporal` | 15107 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 27842 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile128_temporal` | 28223 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 30951 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 62411 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile128_temporal` | 57959 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 63299 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 707 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 726 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1919 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1504 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 1507 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4636 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 5868 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 3036 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 9427 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 16787 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 6700 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 18762 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 36709 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 14766 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 36729 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 75642 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 31948 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 76175 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 854 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 839 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 1790 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1928 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1851 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4650 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5552 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4093 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 9794 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 15332 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 8214 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 19593 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 39723 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 18487 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 39275 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 80008 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 39208 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 80160 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 650 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 657 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1770 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1302 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1299 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3916 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4992 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2875 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 7917 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 14736 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 5869 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 15728 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 31697 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 11536 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 31781 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 63424 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 25195 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 63844 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 706 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 689 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1598 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1383 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1387 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3771 |
| avx512 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 702 |
| avx512 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 795 |
| avx512 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 988 |
| avx512 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 2389 |
| avx512 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 2191 |
| avx512 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 2020 |
| avx512 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile128_temporal` | 5317 |
| avx512 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 4444 |
| avx512 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 4683 |
| avx512 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 8796 |
| avx512 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 8825 |
| avx512 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 9314 |
| avx512 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 18070 |
| avx512 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 18643 |
| avx512 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 18575 |
| avx512 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 37899 |
| avx512 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 38928 |
| avx512 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 38478 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 366 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 371 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1052 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 958 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 977 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2276 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 3679 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 2146 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 4785 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 8512 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 4422 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 9724 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 19455 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 11767 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 19423 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 39281 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 21624 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 37102 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 471 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 472 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 886 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1134 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1302 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 2156 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 3639 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2415 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 4531 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 7701 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 5141 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 8999 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 18298 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 11341 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 17928 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 36965 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 23305 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 36109 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 425 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 414 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 691 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 825 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 836 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1795 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 2868 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1766 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3691 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 5586 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 3792 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7372 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 14343 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 7344 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 14933 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 29377 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 15250 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 31328 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 384 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 379 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 882 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 746 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 849 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1859 |