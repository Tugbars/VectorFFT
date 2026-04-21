# VectorFFT R=11 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 38 | 45 | 59 | — | flat |
| avx2 | 8 | 16 | 41 | 45 | 50 | — | flat |
| avx2 | 8 | 88 | 43 | 49 | 86 | — | flat |
| avx2 | 8 | 256 | 40 | 48 | 43 | — | flat |
| avx2 | 16 | 16 | 70 | 84 | 71 | — | flat |
| avx2 | 16 | 24 | 74 | 84 | 84 | — | flat |
| avx2 | 16 | 176 | 76 | 82 | 70 | — | t1s |
| avx2 | 16 | 512 | 240 | 207 | 237 | — | log3 |
| avx2 | 32 | 32 | 128 | 164 | 246 | — | flat |
| avx2 | 32 | 40 | 148 | 155 | 183 | — | flat |
| avx2 | 32 | 352 | 141 | 178 | 163 | — | flat |
| avx2 | 32 | 1024 | 342 | 312 | 373 | — | log3 |
| avx2 | 64 | 64 | 293 | 321 | 513 | — | flat |
| avx2 | 64 | 72 | 254 | 318 | 321 | — | flat |
| avx2 | 64 | 704 | 264 | 324 | 270 | — | flat |
| avx2 | 64 | 2048 | 764 | 780 | 663 | — | t1s |
| avx2 | 128 | 128 | 558 | 632 | 623 | — | flat |
| avx2 | 128 | 136 | 492 | 619 | 626 | — | flat |
| avx2 | 128 | 1408 | 591 | 612 | 959 | — | flat |
| avx2 | 128 | 4096 | 1482 | 1386 | 1396 | — | log3 |
| avx2 | 256 | 256 | 1410 | 1307 | 1167 | — | t1s |
| avx2 | 256 | 264 | 1221 | 1337 | 1012 | — | t1s |
| avx2 | 256 | 2816 | 1408 | 1288 | 1220 | — | t1s |
| avx2 | 256 | 8192 | 3342 | 1804 | 2194 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 41 |
| avx2 | `t1_dit` | 8 | 88 | `ct_t1_dit` | 43 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 40 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 70 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 74 |
| avx2 | `t1_dit` | 16 | 176 | `ct_t1_dit` | 76 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 240 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 128 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 148 |
| avx2 | `t1_dit` | 32 | 352 | `ct_t1_dit` | 141 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 342 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 293 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 254 |
| avx2 | `t1_dit` | 64 | 704 | `ct_t1_dit` | 264 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 764 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 558 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 492 |
| avx2 | `t1_dit` | 128 | 1408 | `ct_t1_dit` | 591 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1482 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1410 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1221 |
| avx2 | `t1_dit` | 256 | 2816 | `ct_t1_dit` | 1408 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 3342 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 45 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 45 |
| avx2 | `t1_dit_log3` | 8 | 88 | `ct_t1_dit_log3` | 49 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 48 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 84 |
| avx2 | `t1_dit_log3` | 16 | 176 | `ct_t1_dit_log3` | 82 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 207 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 155 |
| avx2 | `t1_dit_log3` | 32 | 352 | `ct_t1_dit_log3` | 178 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 312 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 321 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 318 |
| avx2 | `t1_dit_log3` | 64 | 704 | `ct_t1_dit_log3` | 324 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 780 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 632 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 619 |
| avx2 | `t1_dit_log3` | 128 | 1408 | `ct_t1_dit_log3` | 612 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1386 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1307 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1337 |
| avx2 | `t1_dit_log3` | 256 | 2816 | `ct_t1_dit_log3` | 1288 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1804 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 59 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 50 |
| avx2 | `t1s_dit` | 8 | 88 | `ct_t1s_dit` | 86 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 43 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 71 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 84 |
| avx2 | `t1s_dit` | 16 | 176 | `ct_t1s_dit` | 70 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 237 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 246 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 183 |
| avx2 | `t1s_dit` | 32 | 352 | `ct_t1s_dit` | 163 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 373 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 513 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 321 |
| avx2 | `t1s_dit` | 64 | 704 | `ct_t1s_dit` | 270 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 663 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 623 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 626 |
| avx2 | `t1s_dit` | 128 | 1408 | `ct_t1s_dit` | 959 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1396 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1167 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1012 |
| avx2 | `t1s_dit` | 256 | 2816 | `ct_t1s_dit` | 1220 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2194 |