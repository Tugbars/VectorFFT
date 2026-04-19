# VectorFFT R=32 tuning report

Total measurements: **120**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 987 | 808 | 662 | — | t1s |
| avx2 | 64 | 72 | 856 | 795 | 736 | — | t1s |
| avx2 | 64 | 512 | 2164 | 1939 | 1717 | — | t1s |
| avx2 | 128 | 128 | 2320 | 3207 | 2320 | — | t1s |
| avx2 | 128 | 136 | 2028 | 2038 | 1413 | — | t1s |
| avx2 | 128 | 1024 | 4370 | 3824 | 3569 | — | t1s |
| avx2 | 256 | 256 | 7242 | 8504 | — | — | flat |
| avx2 | 256 | 264 | 5252 | 3794 | — | — | log3 |
| avx2 | 256 | 2048 | 9631 | 7855 | — | — | log3 |
| avx2 | 512 | 512 | 18341 | 15351 | — | — | log3 |
| avx2 | 512 | 520 | 11182 | 7800 | — | — | log3 |
| avx2 | 512 | 4096 | 21921 | 18746 | — | — | log3 |
| avx2 | 1024 | 1024 | 30182 | 30142 | — | — | log3 |
| avx2 | 1024 | 1032 | 18356 | 15785 | — | — | log3 |
| avx2 | 1024 | 8192 | 40419 | 43550 | — | — | flat |
| avx2 | 2048 | 2048 | 76140 | 71467 | — | — | log3 |
| avx2 | 2048 | 2056 | 48980 | 33404 | — | — | log3 |
| avx2 | 2048 | 16384 | 129058 | 122169 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1091 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1552 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2450 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4977 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 2535 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4791 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 8722 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 5252 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 9690 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 24253 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 15450 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 23011 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 38730 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 23363 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 56343 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 99272 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 49542 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 141723 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 987 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 856 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2164 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2320 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2028 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4370 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 7242 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5444 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 9631 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 18341 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 11182 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 21921 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 30182 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 18356 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 40419 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 76140 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 48980 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 129058 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 808 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 795 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1939 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3207 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2038 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3824 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 8504 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3794 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 7855 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 15351 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 7800 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 18746 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 30142 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 15785 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 43550 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 71467 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 33404 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 122169 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 662 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 736 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1717 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2320 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1413 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3569 |