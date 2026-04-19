# VectorFFT R=16 tuning report

Total measurements: **120**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 270 | 308 | 239 | — | t1s |
| avx2 | 64 | 72 | 265 | 310 | 287 | — | flat |
| avx2 | 64 | 512 | 831 | 922 | 817 | — | t1s |
| avx2 | 128 | 128 | 717 | 679 | 480 | — | t1s |
| avx2 | 128 | 136 | 686 | 648 | 540 | — | t1s |
| avx2 | 128 | 1024 | 1844 | 1644 | 1669 | — | log3 |
| avx2 | 256 | 256 | 4216 | 3818 | — | — | log3 |
| avx2 | 256 | 264 | 1623 | 1401 | — | — | log3 |
| avx2 | 256 | 2048 | 3742 | 3344 | — | — | log3 |
| avx2 | 512 | 512 | 6936 | 7224 | — | — | flat |
| avx2 | 512 | 520 | 3402 | 2883 | — | — | log3 |
| avx2 | 512 | 4096 | 7467 | 6365 | — | — | log3 |
| avx2 | 1024 | 1024 | 13730 | 13337 | — | — | log3 |
| avx2 | 1024 | 1032 | 6990 | 5894 | — | — | log3 |
| avx2 | 1024 | 8192 | 15913 | 16524 | — | — | flat |
| avx2 | 2048 | 2048 | 29971 | 29936 | — | — | log3 |
| avx2 | 2048 | 2056 | 14980 | 12232 | — | — | log3 |
| avx2 | 2048 | 16384 | 44077 | 32879 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 500 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 326 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 949 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1151 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 757 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 2640 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 4741 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 2390 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 4145 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 8210 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 6865 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 8457 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 21094 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 7252 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 19421 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 31496 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 15375 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 50807 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 270 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 265 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 831 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 717 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 686 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 1844 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4216 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1623 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 3742 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 6936 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 3402 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 7467 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 13730 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 6990 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 15913 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 29971 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 14980 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 44077 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 308 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 310 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 922 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 679 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 648 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 1644 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3818 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1401 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 3344 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 7224 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 2883 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 6365 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 13337 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 5894 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 16524 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 29936 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 12232 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 32879 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 239 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 287 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 817 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 480 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 540 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 1669 |