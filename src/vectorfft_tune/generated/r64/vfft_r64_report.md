# VectorFFT R=64 tuning report

Total measurements: **300**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 5847 | 2254 | 2046 | — | t1s |
| avx2 | 64 | 72 | 2819 | 2492 | 1850 | — | t1s |
| avx2 | 64 | 512 | 5298 | 4778 | 4739 | — | t1s |
| avx2 | 128 | 128 | 10501 | 9236 | 8893 | — | t1s |
| avx2 | 128 | 136 | 5415 | 4631 | 3538 | — | t1s |
| avx2 | 128 | 1024 | 10533 | 10511 | 9257 | — | t1s |
| avx2 | 256 | 256 | 19450 | 20469 | — | — | flat |
| avx2 | 256 | 264 | 12768 | 9886 | — | — | log3 |
| avx2 | 256 | 2048 | 24255 | 23767 | — | — | log3 |
| avx2 | 512 | 512 | 44520 | 42855 | — | — | log3 |
| avx2 | 512 | 520 | 24370 | 19442 | — | — | log3 |
| avx2 | 512 | 4096 | 55356 | 50446 | — | — | log3 |
| avx2 | 1024 | 1024 | 84098 | 79147 | — | — | log3 |
| avx2 | 1024 | 1032 | 53068 | 37282 | — | — | log3 |
| avx2 | 1024 | 8192 | 148913 | 190378 | — | — | flat |
| avx2 | 2048 | 2048 | 220805 | 199430 | — | — | log3 |
| avx2 | 2048 | 2056 | 125406 | 98779 | — | — | log3 |
| avx2 | 2048 | 16384 | 295997 | 375219 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 6623 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 6261 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 7601 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 12807 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 12450 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 13860 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile256_temporal` | 23802 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile256_temporal` | 23274 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 24255 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 50583 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile256_temporal` | 46234 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile256_temporal` | 56399 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 101305 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile256_temporal` | 99290 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile256_temporal` | 148913 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 228234 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile256_temporal` | 239578 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 295997 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 7280 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 4875 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 5679 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 10501 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 9930 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 14177 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 19450 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 17694 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 32021 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 54636 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 26908 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 80855 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 119418 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 76760 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 160064 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 266505 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 139131 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 370734 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 5847 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2819 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5298 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 10770 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 5415 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 10533 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 19949 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 12768 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 26332 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 44520 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 24370 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 55356 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 84098 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 53068 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 172648 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 220805 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 125406 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 383856 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2254 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2492 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 4778 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9236 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4631 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 10511 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 20469 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 9886 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 23767 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 42855 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 19442 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 50446 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 79147 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 37282 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 190378 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 199430 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 98779 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 375219 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 2046 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1850 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 4739 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 8893 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3538 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 9257 |