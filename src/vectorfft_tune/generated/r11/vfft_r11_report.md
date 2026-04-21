# VectorFFT R=11 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 39 | 49 | 49 | — | flat |
| avx2 | 8 | 16 | 41 | 44 | 50 | — | flat |
| avx2 | 8 | 88 | 46 | 50 | 52 | — | flat |
| avx2 | 8 | 256 | 40 | 52 | 43 | — | flat |
| avx2 | 16 | 16 | 73 | 92 | 75 | — | flat |
| avx2 | 16 | 24 | 76 | 85 | 73 | — | t1s |
| avx2 | 16 | 176 | 70 | 86 | 80 | — | flat |
| avx2 | 16 | 512 | 200 | 202 | 262 | — | flat |
| avx2 | 32 | 32 | 139 | 176 | 150 | — | flat |
| avx2 | 32 | 40 | 135 | 161 | 141 | — | flat |
| avx2 | 32 | 352 | 138 | 185 | 166 | — | flat |
| avx2 | 32 | 1024 | 353 | 314 | 551 | — | log3 |
| avx2 | 64 | 64 | 292 | 324 | 328 | — | flat |
| avx2 | 64 | 72 | 264 | 307 | 279 | — | flat |
| avx2 | 64 | 704 | 319 | 336 | 276 | — | t1s |
| avx2 | 64 | 2048 | 849 | 669 | 777 | — | log3 |
| avx2 | 128 | 128 | 538 | 631 | 555 | — | flat |
| avx2 | 128 | 136 | 503 | 638 | 673 | — | flat |
| avx2 | 128 | 1408 | 598 | 646 | 595 | — | t1s |
| avx2 | 128 | 4096 | 1626 | 1390 | 1371 | — | t1s |
| avx2 | 256 | 256 | 1384 | 1244 | 1134 | — | t1s |
| avx2 | 256 | 264 | 1288 | 1256 | 1919 | — | log3 |
| avx2 | 256 | 2816 | 1662 | 1331 | 1238 | — | t1s |
| avx2 | 256 | 8192 | 2098 | 1472 | 2823 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 39 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 41 |
| avx2 | `t1_dit` | 8 | 88 | `ct_t1_dit` | 46 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 73 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 16 | 176 | `ct_t1_dit` | 70 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 200 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 139 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 135 |
| avx2 | `t1_dit` | 32 | 352 | `ct_t1_dit` | 138 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 353 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 292 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 264 |
| avx2 | `t1_dit` | 64 | 704 | `ct_t1_dit` | 319 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 849 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 538 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 503 |
| avx2 | `t1_dit` | 128 | 1408 | `ct_t1_dit` | 598 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1626 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1384 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1288 |
| avx2 | `t1_dit` | 256 | 2816 | `ct_t1_dit` | 1662 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2098 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 44 |
| avx2 | `t1_dit_log3` | 8 | 88 | `ct_t1_dit_log3` | 50 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 52 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 92 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 85 |
| avx2 | `t1_dit_log3` | 16 | 176 | `ct_t1_dit_log3` | 86 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 202 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 176 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 161 |
| avx2 | `t1_dit_log3` | 32 | 352 | `ct_t1_dit_log3` | 185 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 314 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 324 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 307 |
| avx2 | `t1_dit_log3` | 64 | 704 | `ct_t1_dit_log3` | 336 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 669 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 631 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 638 |
| avx2 | `t1_dit_log3` | 128 | 1408 | `ct_t1_dit_log3` | 646 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1390 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1244 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1256 |
| avx2 | `t1_dit_log3` | 256 | 2816 | `ct_t1_dit_log3` | 1331 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1472 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 49 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 50 |
| avx2 | `t1s_dit` | 8 | 88 | `ct_t1s_dit` | 52 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 43 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 75 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 73 |
| avx2 | `t1s_dit` | 16 | 176 | `ct_t1s_dit` | 80 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 262 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 150 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 141 |
| avx2 | `t1s_dit` | 32 | 352 | `ct_t1s_dit` | 166 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 551 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 328 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 279 |
| avx2 | `t1s_dit` | 64 | 704 | `ct_t1s_dit` | 276 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 777 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 555 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 673 |
| avx2 | `t1s_dit` | 128 | 1408 | `ct_t1s_dit` | 595 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1371 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1134 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1919 |
| avx2 | `t1s_dit` | 256 | 2816 | `ct_t1s_dit` | 1238 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2823 |