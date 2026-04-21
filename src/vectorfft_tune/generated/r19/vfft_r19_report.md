# VectorFFT R=19 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 100 | 108 | 131 | — | flat |
| avx2 | 8 | 16 | 110 | 154 | 107 | — | t1s |
| avx2 | 8 | 152 | 96 | 108 | 134 | — | flat |
| avx2 | 8 | 256 | 111 | 109 | 167 | — | log3 |
| avx2 | 16 | 16 | 189 | 210 | 190 | — | flat |
| avx2 | 16 | 24 | 337 | 211 | 190 | — | t1s |
| avx2 | 16 | 304 | 217 | 217 | 203 | — | t1s |
| avx2 | 16 | 512 | 382 | 345 | 428 | — | log3 |
| avx2 | 32 | 32 | 553 | 414 | 551 | — | log3 |
| avx2 | 32 | 40 | 416 | 414 | 379 | — | t1s |
| avx2 | 32 | 608 | 423 | 441 | 422 | — | t1s |
| avx2 | 32 | 1024 | 775 | 730 | 766 | — | log3 |
| avx2 | 64 | 64 | 734 | 815 | 1053 | — | flat |
| avx2 | 64 | 72 | 839 | 823 | 739 | — | t1s |
| avx2 | 64 | 1216 | 1169 | 1356 | 777 | — | t1s |
| avx2 | 64 | 2048 | 1508 | 1439 | 1993 | — | log3 |
| avx2 | 128 | 128 | 1893 | 1587 | 1469 | — | t1s |
| avx2 | 128 | 136 | 1669 | 1604 | 2045 | — | log3 |
| avx2 | 128 | 2432 | 2904 | 1721 | 1623 | — | t1s |
| avx2 | 128 | 4096 | 3070 | 3192 | 3655 | — | flat |
| avx2 | 256 | 256 | 6112 | 4988 | 4984 | — | t1s |
| avx2 | 256 | 264 | 3861 | 3375 | 4725 | — | log3 |
| avx2 | 256 | 4864 | 5962 | 5219 | 4609 | — | t1s |
| avx2 | 256 | 8192 | 6826 | 7102 | 8673 | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 100 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 110 |
| avx2 | `t1_dit` | 8 | 152 | `ct_t1_dit` | 96 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 111 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 189 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 337 |
| avx2 | `t1_dit` | 16 | 304 | `ct_t1_dit` | 217 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 382 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 553 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 416 |
| avx2 | `t1_dit` | 32 | 608 | `ct_t1_dit` | 423 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 775 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 734 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 839 |
| avx2 | `t1_dit` | 64 | 1216 | `ct_t1_dit` | 1169 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1508 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1893 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1669 |
| avx2 | `t1_dit` | 128 | 2432 | `ct_t1_dit` | 2904 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3070 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6112 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3861 |
| avx2 | `t1_dit` | 256 | 4864 | `ct_t1_dit` | 5962 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 6826 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 108 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 154 |
| avx2 | `t1_dit_log3` | 8 | 152 | `ct_t1_dit_log3` | 108 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 109 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 210 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 211 |
| avx2 | `t1_dit_log3` | 16 | 304 | `ct_t1_dit_log3` | 217 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 345 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 414 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 414 |
| avx2 | `t1_dit_log3` | 32 | 608 | `ct_t1_dit_log3` | 441 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 730 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 815 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 823 |
| avx2 | `t1_dit_log3` | 64 | 1216 | `ct_t1_dit_log3` | 1356 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1439 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1587 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1604 |
| avx2 | `t1_dit_log3` | 128 | 2432 | `ct_t1_dit_log3` | 1721 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3192 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 4988 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3375 |
| avx2 | `t1_dit_log3` | 256 | 4864 | `ct_t1_dit_log3` | 5219 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 7102 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 131 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 107 |
| avx2 | `t1s_dit` | 8 | 152 | `ct_t1s_dit` | 134 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 167 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 190 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 190 |
| avx2 | `t1s_dit` | 16 | 304 | `ct_t1s_dit` | 203 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 428 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 551 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 379 |
| avx2 | `t1s_dit` | 32 | 608 | `ct_t1s_dit` | 422 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 766 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1053 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 739 |
| avx2 | `t1s_dit` | 64 | 1216 | `ct_t1s_dit` | 777 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1993 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1469 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2045 |
| avx2 | `t1s_dit` | 128 | 2432 | `ct_t1s_dit` | 1623 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 3655 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4984 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 4725 |
| avx2 | `t1s_dit` | 256 | 4864 | `ct_t1s_dit` | 4609 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 8673 |