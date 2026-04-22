# VectorFFT R=17 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 74 | 85 | 102 | — | flat |
| avx2 | 8 | 16 | 76 | 86 | 116 | — | flat |
| avx2 | 8 | 136 | 76 | 86 | 80 | — | flat |
| avx2 | 8 | 256 | 77 | 84 | 131 | — | flat |
| avx2 | 16 | 16 | 148 | 167 | 149 | — | flat |
| avx2 | 16 | 24 | 148 | 239 | 149 | — | flat |
| avx2 | 16 | 272 | 153 | 170 | 185 | — | flat |
| avx2 | 16 | 512 | 300 | 400 | 442 | — | flat |
| avx2 | 32 | 32 | 294 | 476 | 497 | — | flat |
| avx2 | 32 | 40 | 289 | 329 | 347 | — | flat |
| avx2 | 32 | 544 | 305 | 335 | 311 | — | flat |
| avx2 | 32 | 1024 | 610 | 611 | 805 | — | flat |
| avx2 | 64 | 64 | 570 | 641 | 576 | — | flat |
| avx2 | 64 | 72 | 1022 | 643 | 930 | — | log3 |
| avx2 | 64 | 1088 | 607 | 981 | 684 | — | flat |
| avx2 | 64 | 2048 | 1203 | 1251 | 1111 | — | t1s |
| avx2 | 128 | 128 | 1335 | 1292 | 1151 | — | t1s |
| avx2 | 128 | 136 | 1319 | 1798 | 1389 | — | flat |
| avx2 | 128 | 2176 | 1357 | 1389 | 1375 | — | flat |
| avx2 | 128 | 4096 | 2559 | 2642 | 2321 | — | t1s |
| avx2 | 256 | 256 | 5626 | 4577 | 5420 | — | log3 |
| avx2 | 256 | 264 | 2796 | 4099 | 3683 | — | flat |
| avx2 | 256 | 4352 | 2909 | 2992 | 3772 | — | flat |
| avx2 | 256 | 8192 | 6148 | 6281 | 5695 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 8 | 136 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 77 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 148 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 148 |
| avx2 | `t1_dit` | 16 | 272 | `ct_t1_dit` | 153 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 300 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 294 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 289 |
| avx2 | `t1_dit` | 32 | 544 | `ct_t1_dit` | 305 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 610 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 570 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1022 |
| avx2 | `t1_dit` | 64 | 1088 | `ct_t1_dit` | 607 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1203 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1335 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1319 |
| avx2 | `t1_dit` | 128 | 2176 | `ct_t1_dit` | 1357 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2559 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5626 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2796 |
| avx2 | `t1_dit` | 256 | 4352 | `ct_t1_dit` | 2909 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 6148 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 85 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 86 |
| avx2 | `t1_dit_log3` | 8 | 136 | `ct_t1_dit_log3` | 86 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 167 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 239 |
| avx2 | `t1_dit_log3` | 16 | 272 | `ct_t1_dit_log3` | 170 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 400 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 476 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 329 |
| avx2 | `t1_dit_log3` | 32 | 544 | `ct_t1_dit_log3` | 335 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 611 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 641 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 643 |
| avx2 | `t1_dit_log3` | 64 | 1088 | `ct_t1_dit_log3` | 981 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1251 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1292 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1798 |
| avx2 | `t1_dit_log3` | 128 | 2176 | `ct_t1_dit_log3` | 1389 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2642 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4577 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 4099 |
| avx2 | `t1_dit_log3` | 256 | 4352 | `ct_t1_dit_log3` | 2992 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 6281 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 102 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 116 |
| avx2 | `t1s_dit` | 8 | 136 | `ct_t1s_dit` | 80 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 131 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 16 | 272 | `ct_t1s_dit` | 185 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 442 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 497 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 347 |
| avx2 | `t1s_dit` | 32 | 544 | `ct_t1s_dit` | 311 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 805 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 576 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 930 |
| avx2 | `t1s_dit` | 64 | 1088 | `ct_t1s_dit` | 684 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1111 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1151 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1389 |
| avx2 | `t1s_dit` | 128 | 2176 | `ct_t1s_dit` | 1375 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2321 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 5420 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 3683 |
| avx2 | `t1s_dit` | 256 | 4352 | `ct_t1s_dit` | 3772 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 5695 |