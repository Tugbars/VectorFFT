# VectorFFT R=10 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 25 | 32 | 27 | — | flat |
| avx2 | 8 | 16 | 25 | 32 | 27 | — | flat |
| avx2 | 8 | 80 | 25 | 32 | 28 | — | flat |
| avx2 | 8 | 256 | 25 | 32 | 27 | — | flat |
| avx2 | 16 | 16 | 46 | 61 | 49 | — | flat |
| avx2 | 16 | 24 | 46 | 61 | 54 | — | flat |
| avx2 | 16 | 160 | 46 | 61 | 58 | — | flat |
| avx2 | 16 | 512 | 141 | 156 | 149 | — | flat |
| avx2 | 32 | 32 | 89 | 118 | 91 | — | flat |
| avx2 | 32 | 40 | 87 | 117 | 91 | — | flat |
| avx2 | 32 | 320 | 88 | 117 | 92 | — | flat |
| avx2 | 32 | 1024 | 259 | 240 | 198 | — | t1s |
| avx2 | 64 | 64 | 171 | 231 | 175 | — | flat |
| avx2 | 64 | 72 | 172 | 230 | 177 | — | flat |
| avx2 | 64 | 640 | 179 | 234 | 182 | — | flat |
| avx2 | 64 | 2048 | 440 | 411 | 335 | — | t1s |
| avx2 | 128 | 128 | 340 | 461 | 344 | — | flat |
| avx2 | 128 | 136 | 342 | 459 | 346 | — | flat |
| avx2 | 128 | 1280 | 388 | 460 | 358 | — | t1s |
| avx2 | 128 | 4096 | 897 | 1056 | 641 | — | t1s |
| avx2 | 256 | 256 | 867 | 934 | 694 | — | t1s |
| avx2 | 256 | 264 | 791 | 916 | 745 | — | t1s |
| avx2 | 256 | 2560 | 2130 | 1565 | 1423 | — | t1s |
| avx2 | 256 | 8192 | 1922 | 1014 | 824 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 80 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 25 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 16 | 160 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 141 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 89 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 87 |
| avx2 | `t1_dit` | 32 | 320 | `ct_t1_dit` | 88 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 259 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 171 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 172 |
| avx2 | `t1_dit` | 64 | 640 | `ct_t1_dit` | 179 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 440 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 340 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 342 |
| avx2 | `t1_dit` | 128 | 1280 | `ct_t1_dit` | 388 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 897 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 867 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 791 |
| avx2 | `t1_dit` | 256 | 2560 | `ct_t1_dit` | 2130 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 1922 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 32 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 32 |
| avx2 | `t1_dit_log3` | 8 | 80 | `ct_t1_dit_log3` | 32 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 32 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 61 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 61 |
| avx2 | `t1_dit_log3` | 16 | 160 | `ct_t1_dit_log3` | 61 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 156 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 118 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 117 |
| avx2 | `t1_dit_log3` | 32 | 320 | `ct_t1_dit_log3` | 117 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 240 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 231 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 230 |
| avx2 | `t1_dit_log3` | 64 | 640 | `ct_t1_dit_log3` | 234 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 411 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 461 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 459 |
| avx2 | `t1_dit_log3` | 128 | 1280 | `ct_t1_dit_log3` | 460 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1056 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 934 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 916 |
| avx2 | `t1_dit_log3` | 256 | 2560 | `ct_t1_dit_log3` | 1565 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1014 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 27 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 27 |
| avx2 | `t1s_dit` | 8 | 80 | `ct_t1s_dit` | 28 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 27 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 49 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 54 |
| avx2 | `t1s_dit` | 16 | 160 | `ct_t1s_dit` | 58 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 149 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 91 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 91 |
| avx2 | `t1s_dit` | 32 | 320 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 198 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 175 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 177 |
| avx2 | `t1s_dit` | 64 | 640 | `ct_t1s_dit` | 182 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 335 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 344 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 346 |
| avx2 | `t1s_dit` | 128 | 1280 | `ct_t1s_dit` | 358 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 641 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 694 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 745 |
| avx2 | `t1s_dit` | 256 | 2560 | `ct_t1s_dit` | 1423 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 824 |