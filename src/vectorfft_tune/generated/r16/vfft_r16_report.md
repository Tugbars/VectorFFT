# VectorFFT R=16 tuning report

Total measurements: **222**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 266 | 314 | 247 | — | t1s |
| avx2 | 64 | 72 | 290 | 317 | 245 | — | t1s |
| avx2 | 64 | 512 | 856 | 863 | 816 | — | t1s |
| avx2 | 128 | 128 | 660 | 627 | 488 | — | t1s |
| avx2 | 128 | 136 | 672 | 627 | 500 | — | t1s |
| avx2 | 128 | 1024 | 1741 | 1716 | 1621 | — | t1s |
| avx2 | 256 | 256 | 3749 | 3607 | — | — | log3 |
| avx2 | 256 | 264 | 1682 | 1458 | — | — | log3 |
| avx2 | 256 | 2048 | 4076 | 3413 | — | — | log3 |
| avx2 | 512 | 512 | 6783 | 7391 | — | — | flat |
| avx2 | 512 | 520 | 3886 | 3031 | — | — | log3 |
| avx2 | 512 | 4096 | 6430 | 7562 | — | — | flat |
| avx2 | 1024 | 1024 | 15181 | 15516 | — | — | flat |
| avx2 | 1024 | 1032 | 7390 | 6189 | — | — | log3 |
| avx2 | 1024 | 8192 | 13330 | 15562 | — | — | flat |
| avx2 | 2048 | 2048 | 27342 | 32055 | — | — | flat |
| avx2 | 2048 | 2056 | 15779 | 15746 | — | — | log3 |
| avx2 | 2048 | 16384 | 35514 | 35542 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 424 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 455 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 970 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 1369 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 1333 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 1741 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 3803 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 2548 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 4076 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 6783 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 5728 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile128_temporal` | 6430 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 15181 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 12859 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 13330 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 27342 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 22912 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 34978 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 508 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 332 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1334 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1438 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 1507 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2114 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 4326 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 3065 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 5020 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 10506 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 4525 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 9952 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 16780 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 8914 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 19985 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 33432 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 22746 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 47275 |
| avx2 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 407 |
| avx2 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 847 |
| avx2 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 1115 |
| avx2 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 734 |
| avx2 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 745 |
| avx2 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 2903 |
| avx2 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 4698 |
| avx2 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 2559 |
| avx2 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 4066 |
| avx2 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 8088 |
| avx2 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 5154 |
| avx2 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 8451 |
| avx2 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 21237 |
| avx2 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 11193 |
| avx2 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 19413 |
| avx2 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 43494 |
| avx2 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 27864 |
| avx2 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 52823 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 266 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 290 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 856 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 660 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 672 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 1931 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 3749 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1682 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 4169 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 16971 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 3886 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 7639 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 15491 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 7390 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 17977 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 28316 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 15779 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 35514 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 314 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 317 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 863 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 627 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 627 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1716 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3607 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1458 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3413 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 7391 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 3031 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 7562 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 15516 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 6189 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 15562 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 32055 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 15746 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 35542 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 247 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 245 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 816 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 488 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 500 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1621 |