# VectorFFT R=16 tuning report

Total measurements: **240**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 721 | 642 | 712 | — | log3 |
| avx2 | 64 | 72 | 717 | 647 | 805 | — | log3 |
| avx2 | 64 | 512 | 1639 | 1595 | 1486 | — | t1s |
| avx2 | 128 | 128 | 1528 | 1332 | 1398 | — | log3 |
| avx2 | 128 | 136 | 1522 | 1316 | 1460 | — | log3 |
| avx2 | 128 | 1024 | 4070 | 3464 | 3412 | — | t1s |
| avx2 | 256 | 256 | 4925 | 4978 | — | — | flat |
| avx2 | 256 | 264 | 3194 | 2828 | — | — | log3 |
| avx2 | 256 | 2048 | 8221 | 7030 | — | — | log3 |
| avx2 | 512 | 512 | 13985 | 12854 | — | — | log3 |
| avx2 | 512 | 520 | 6498 | 5648 | — | — | log3 |
| avx2 | 512 | 4096 | 16634 | 14179 | — | — | log3 |
| avx2 | 1024 | 1024 | 33222 | 27982 | — | — | log3 |
| avx2 | 1024 | 1032 | 14995 | 11691 | — | — | log3 |
| avx2 | 1024 | 8192 | 33284 | 27525 | — | — | log3 |
| avx2 | 2048 | 2048 | 66816 | 56830 | — | — | log3 |
| avx2 | 2048 | 2056 | 29508 | 23226 | — | — | log3 |
| avx2 | 2048 | 16384 | 67398 | 55325 | — | — | log3 |
| avx512 | 64 | 64 | 472 | 436 | 437 | — | log3 |
| avx512 | 64 | 72 | 480 | 423 | 461 | — | log3 |
| avx512 | 64 | 512 | 949 | 702 | 899 | — | log3 |
| avx512 | 128 | 128 | 1059 | 886 | 882 | — | t1s |
| avx512 | 128 | 136 | 1014 | 876 | 898 | — | log3 |
| avx512 | 128 | 1024 | 2167 | 1812 | 1811 | — | t1s |
| avx512 | 256 | 256 | 3538 | 2707 | — | — | log3 |
| avx512 | 256 | 264 | 2189 | 1856 | — | — | log3 |
| avx512 | 256 | 2048 | 4579 | 3660 | — | — | log3 |
| avx512 | 512 | 512 | 8137 | 5622 | — | — | log3 |
| avx512 | 512 | 520 | 4633 | 3930 | — | — | log3 |
| avx512 | 512 | 4096 | 9119 | 7316 | — | — | log3 |
| avx512 | 1024 | 1024 | 18637 | 14628 | — | — | log3 |
| avx512 | 1024 | 1032 | 11251 | 7711 | — | — | log3 |
| avx512 | 1024 | 8192 | 18528 | 14694 | — | — | log3 |
| avx512 | 2048 | 2048 | 38301 | 28808 | — | — | log3 |
| avx512 | 2048 | 2056 | 21261 | 15614 | — | — | log3 |
| avx512 | 2048 | 16384 | 37050 | 29343 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 721 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 717 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1722 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1528 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 1522 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4070 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 4921 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 3194 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 8221 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 14867 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 6498 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 16634 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 33222 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 14995 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 33284 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 66816 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 29508 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 67398 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 891 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 906 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 1639 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2320 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1879 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4167 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4925 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3783 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 8509 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 13985 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 7904 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 17384 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 34822 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 15944 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 34825 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 70993 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 34930 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 71116 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 642 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 647 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1595 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1332 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1316 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3464 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4978 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2828 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 7030 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 12854 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 5648 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 14179 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 27982 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 11691 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 27525 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 56830 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 23226 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 55325 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 712 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 805 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1486 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1398 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1460 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3412 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 472 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 480 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1046 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1059 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 1014 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2260 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 3548 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 2189 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 4697 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 8251 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 4633 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 9316 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 18895 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 11311 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 18778 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 37995 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 22161 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 37556 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 540 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 559 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 949 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1237 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1205 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 2167 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 3538 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2445 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 4579 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 8137 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 5034 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 9119 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 18637 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 11251 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 18528 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 38301 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 21261 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 37050 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 436 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 423 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 702 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 886 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 876 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1812 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 2707 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1856 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3660 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 5622 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 3930 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7316 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 14628 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 7711 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 14694 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 28808 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 15614 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 29343 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 437 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 461 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 899 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 882 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 898 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1811 |