# VectorFFT R=13 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 109 | 59 | 52 | — | t1s |
| avx2 | 8 | 16 | 53 | 56 | 75 | — | flat |
| avx2 | 8 | 104 | 53 | 55 | 80 | — | flat |
| avx2 | 8 | 256 | 55 | 58 | 70 | — | flat |
| avx2 | 16 | 16 | 102 | 107 | 92 | — | t1s |
| avx2 | 16 | 24 | 97 | 104 | 169 | — | flat |
| avx2 | 16 | 208 | 101 | 102 | 189 | — | flat |
| avx2 | 16 | 512 | 253 | 259 | 235 | — | t1s |
| avx2 | 32 | 32 | 195 | 194 | 173 | — | t1s |
| avx2 | 32 | 40 | 183 | 199 | 179 | — | t1s |
| avx2 | 32 | 416 | 210 | 223 | 313 | — | flat |
| avx2 | 32 | 1024 | 472 | 508 | 491 | — | flat |
| avx2 | 64 | 64 | 343 | 404 | 429 | — | flat |
| avx2 | 64 | 72 | 358 | 427 | 365 | — | flat |
| avx2 | 64 | 832 | 421 | 442 | 820 | — | flat |
| avx2 | 64 | 2048 | 957 | 877 | 920 | — | log3 |
| avx2 | 128 | 128 | 876 | 839 | 687 | — | t1s |
| avx2 | 128 | 136 | 715 | 807 | 854 | — | flat |
| avx2 | 128 | 1664 | 889 | 829 | 857 | — | log3 |
| avx2 | 128 | 4096 | 2103 | 1996 | 1834 | — | t1s |
| avx2 | 256 | 256 | 1855 | 1745 | 1527 | — | t1s |
| avx2 | 256 | 264 | 3290 | 1986 | 1963 | — | t1s |
| avx2 | 256 | 3328 | 3082 | 2615 | 2671 | — | log3 |
| avx2 | 256 | 8192 | 4079 | 4013 | 3445 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 109 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 53 |
| avx2 | `t1_dit` | 8 | 104 | `ct_t1_dit` | 53 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 55 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 102 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 97 |
| avx2 | `t1_dit` | 16 | 208 | `ct_t1_dit` | 101 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 253 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 195 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 183 |
| avx2 | `t1_dit` | 32 | 416 | `ct_t1_dit` | 210 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 472 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 343 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 358 |
| avx2 | `t1_dit` | 64 | 832 | `ct_t1_dit` | 421 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 957 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 876 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 715 |
| avx2 | `t1_dit` | 128 | 1664 | `ct_t1_dit` | 889 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2103 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1855 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3290 |
| avx2 | `t1_dit` | 256 | 3328 | `ct_t1_dit` | 3082 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 4079 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 59 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 56 |
| avx2 | `t1_dit_log3` | 8 | 104 | `ct_t1_dit_log3` | 55 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 58 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 107 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 104 |
| avx2 | `t1_dit_log3` | 16 | 208 | `ct_t1_dit_log3` | 102 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 259 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 194 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 199 |
| avx2 | `t1_dit_log3` | 32 | 416 | `ct_t1_dit_log3` | 223 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 508 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 404 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 427 |
| avx2 | `t1_dit_log3` | 64 | 832 | `ct_t1_dit_log3` | 442 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 877 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 839 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 807 |
| avx2 | `t1_dit_log3` | 128 | 1664 | `ct_t1_dit_log3` | 829 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1996 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1745 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1986 |
| avx2 | `t1_dit_log3` | 256 | 3328 | `ct_t1_dit_log3` | 2615 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 4013 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 52 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 75 |
| avx2 | `t1s_dit` | 8 | 104 | `ct_t1s_dit` | 80 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 70 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 169 |
| avx2 | `t1s_dit` | 16 | 208 | `ct_t1s_dit` | 189 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 235 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 173 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 179 |
| avx2 | `t1s_dit` | 32 | 416 | `ct_t1s_dit` | 313 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 491 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 429 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 365 |
| avx2 | `t1s_dit` | 64 | 832 | `ct_t1s_dit` | 820 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 920 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 687 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 854 |
| avx2 | `t1s_dit` | 128 | 1664 | `ct_t1s_dit` | 857 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1834 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1527 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1963 |
| avx2 | `t1s_dit` | 256 | 3328 | `ct_t1s_dit` | 2671 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 3445 |