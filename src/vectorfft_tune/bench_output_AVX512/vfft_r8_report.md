# VectorFFT R=8 tuning report

Total measurements: **492**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 240 | 232 | 224 | — | t1s |
| avx2 | 64 | 72 | 237 | 228 | 220 | — | t1s |
| avx2 | 64 | 512 | 474 | 437 | 529 | — | log3 |
| avx2 | 128 | 128 | 461 | 444 | 434 | — | t1s |
| avx2 | 128 | 136 | 469 | 457 | 644 | — | log3 |
| avx2 | 128 | 1024 | 940 | 863 | 912 | — | log3 |
| avx2 | 256 | 256 | 1008 | 907 | — | — | log3 |
| avx2 | 256 | 264 | 1048 | 904 | — | — | log3 |
| avx2 | 256 | 2048 | 4054 | 3972 | — | — | log3 |
| avx2 | 512 | 512 | 4266 | 3608 | — | — | log3 |
| avx2 | 512 | 520 | 2121 | 2080 | — | — | log3 |
| avx2 | 512 | 4096 | 8184 | 7847 | — | — | log3 |
| avx2 | 1024 | 1024 | 7906 | 7030 | — | — | log3 |
| avx2 | 1024 | 1032 | 4257 | 4043 | — | — | log3 |
| avx2 | 1024 | 8192 | 16817 | 15961 | — | — | log3 |
| avx2 | 2048 | 2048 | 32971 | 31634 | — | — | log3 |
| avx2 | 2048 | 2056 | 14301 | 10824 | — | — | log3 |
| avx2 | 2048 | 16384 | 33480 | 31561 | — | — | log3 |
| avx512 | 64 | 64 | 120 | 143 | 125 | — | flat |
| avx512 | 64 | 72 | 122 | 148 | 121 | — | t1s |
| avx512 | 64 | 512 | 377 | 352 | 370 | — | log3 |
| avx512 | 128 | 128 | 247 | 339 | 238 | — | t1s |
| avx512 | 128 | 136 | 242 | 300 | 242 | — | flat |
| avx512 | 128 | 1024 | 745 | 688 | 726 | — | log3 |
| avx512 | 256 | 256 | 624 | 607 | — | — | log3 |
| avx512 | 256 | 264 | 738 | 597 | — | — | log3 |
| avx512 | 256 | 2048 | 1695 | 1553 | — | — | log3 |
| avx512 | 512 | 512 | 3167 | 2792 | — | — | log3 |
| avx512 | 512 | 520 | 1706 | 1658 | — | — | log3 |
| avx512 | 512 | 4096 | 3574 | 3107 | — | — | log3 |
| avx512 | 1024 | 1024 | 6313 | 5528 | — | — | log3 |
| avx512 | 1024 | 1032 | 3424 | 3229 | — | — | log3 |
| avx512 | 1024 | 8192 | 7053 | 6033 | — | — | log3 |
| avx512 | 2048 | 2048 | 14409 | 12299 | — | — | log3 |
| avx512 | 2048 | 2056 | 10052 | 8624 | — | — | log3 |
| avx512 | 2048 | 16384 | 14161 | 11884 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 240 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 237 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif_prefetch` | 609 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 461 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 469 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 1186 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 1008 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 1048 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif_prefetch` | 4919 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif_prefetch` | 5307 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 2121 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif_prefetch` | 9715 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 9875 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 4257 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif_prefetch` | 19997 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 39710 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif_prefetch` | 15638 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 39677 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 245 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 250 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_log1` | 474 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 478 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 502 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 940 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1117 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1096 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 4054 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 4266 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 2305 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_log1` | 8184 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_log1` | 7906 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 4631 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 16817 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_log1` | 32971 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 14301 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_log1` | 33480 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 232 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 228 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 437 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 444 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 457 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 863 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 907 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 904 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3972 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 3608 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 2080 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7847 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 7030 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 4043 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 15961 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 31634 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 10824 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 31561 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 224 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 220 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 529 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 434 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 644 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 912 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 120 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 122 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 499 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 247 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 242 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 977 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 624 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 738 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 2168 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 4111 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1685 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 4245 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 7774 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3361 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 8941 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 17054 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 12929 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 17101 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 129 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 129 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit_u2` | 377 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 258 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 258 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 745 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 661 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_log1` | 796 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 1695 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit_log1` | 3167 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit_log1` | 1706 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 3574 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_u2` | 6313 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 3488 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_log1` | 7053 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 14409 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 10052 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 14161 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 143 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 148 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 352 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 339 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 300 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 688 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 607 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 597 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 1553 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 2792 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1658 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 3107 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 5528 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 3229 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 6033 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 12299 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 8624 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 11884 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 125 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 121 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 370 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 238 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 242 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 726 |