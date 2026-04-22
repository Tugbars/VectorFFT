# VectorFFT R=19 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 98 | 146 | 106 | — | flat |
| avx2 | 8 | 16 | 163 | 112 | 131 | — | log3 |
| avx2 | 8 | 152 | 113 | 148 | 162 | — | flat |
| avx2 | 8 | 256 | 111 | 113 | 107 | — | t1s |
| avx2 | 16 | 16 | 218 | 215 | 198 | — | t1s |
| avx2 | 16 | 24 | 192 | 214 | 236 | — | flat |
| avx2 | 16 | 304 | 215 | 227 | 281 | — | flat |
| avx2 | 16 | 512 | 397 | 368 | 522 | — | log3 |
| avx2 | 32 | 32 | 380 | 418 | 381 | — | flat |
| avx2 | 32 | 40 | 374 | 422 | 381 | — | flat |
| avx2 | 32 | 608 | 644 | 442 | 399 | — | t1s |
| avx2 | 32 | 1024 | 694 | 725 | 962 | — | flat |
| avx2 | 64 | 64 | 754 | 1196 | 1157 | — | flat |
| avx2 | 64 | 72 | 753 | 841 | 1059 | — | flat |
| avx2 | 64 | 1216 | 813 | 944 | 866 | — | flat |
| avx2 | 64 | 2048 | 1534 | 1422 | 1527 | — | log3 |
| avx2 | 128 | 128 | 1741 | 1683 | 1510 | — | t1s |
| avx2 | 128 | 136 | 1724 | 1682 | 2318 | — | log3 |
| avx2 | 128 | 2432 | 2594 | 2309 | 1650 | — | t1s |
| avx2 | 128 | 4096 | 3326 | 3338 | 2715 | — | t1s |
| avx2 | 256 | 256 | 6177 | 5070 | 4912 | — | t1s |
| avx2 | 256 | 264 | 5061 | 3609 | 4771 | — | log3 |
| avx2 | 256 | 4864 | 3756 | 3935 | 3346 | — | t1s |
| avx2 | 256 | 8192 | 8032 | 7867 | 7902 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 98 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 163 |
| avx2 | `t1_dit` | 8 | 152 | `ct_t1_dit` | 113 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 111 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 218 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 192 |
| avx2 | `t1_dit` | 16 | 304 | `ct_t1_dit` | 215 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 397 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 380 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 374 |
| avx2 | `t1_dit` | 32 | 608 | `ct_t1_dit` | 644 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 694 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 754 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 753 |
| avx2 | `t1_dit` | 64 | 1216 | `ct_t1_dit` | 813 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1534 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1741 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1724 |
| avx2 | `t1_dit` | 128 | 2432 | `ct_t1_dit` | 2594 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3326 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6177 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5061 |
| avx2 | `t1_dit` | 256 | 4864 | `ct_t1_dit` | 3756 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 8032 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 146 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 112 |
| avx2 | `t1_dit_log3` | 8 | 152 | `ct_t1_dit_log3` | 148 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 113 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 215 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 214 |
| avx2 | `t1_dit_log3` | 16 | 304 | `ct_t1_dit_log3` | 227 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 368 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 418 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 422 |
| avx2 | `t1_dit_log3` | 32 | 608 | `ct_t1_dit_log3` | 442 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 725 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1196 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 841 |
| avx2 | `t1_dit_log3` | 64 | 1216 | `ct_t1_dit_log3` | 944 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1422 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1683 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1682 |
| avx2 | `t1_dit_log3` | 128 | 2432 | `ct_t1_dit_log3` | 2309 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3338 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 5070 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3609 |
| avx2 | `t1_dit_log3` | 256 | 4864 | `ct_t1_dit_log3` | 3935 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 7867 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 106 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 131 |
| avx2 | `t1s_dit` | 8 | 152 | `ct_t1s_dit` | 162 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 107 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 198 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 236 |
| avx2 | `t1s_dit` | 16 | 304 | `ct_t1s_dit` | 281 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 522 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 381 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 381 |
| avx2 | `t1s_dit` | 32 | 608 | `ct_t1s_dit` | 399 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 962 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1157 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1059 |
| avx2 | `t1s_dit` | 64 | 1216 | `ct_t1s_dit` | 866 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1527 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1510 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2318 |
| avx2 | `t1s_dit` | 128 | 2432 | `ct_t1s_dit` | 1650 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2715 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4912 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 4771 |
| avx2 | `t1s_dit` | 256 | 4864 | `ct_t1s_dit` | 3346 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 7902 |