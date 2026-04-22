# VectorFFT R=11 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 38 | 45 | 39 | — | flat |
| avx2 | 8 | 16 | 38 | 44 | 39 | — | flat |
| avx2 | 8 | 88 | 39 | 45 | 39 | — | t1s |
| avx2 | 8 | 256 | 38 | 48 | 46 | — | flat |
| avx2 | 16 | 16 | 68 | 89 | 69 | — | flat |
| avx2 | 16 | 24 | 67 | 81 | 116 | — | flat |
| avx2 | 16 | 176 | 71 | 83 | 69 | — | t1s |
| avx2 | 16 | 512 | 210 | 225 | 174 | — | t1s |
| avx2 | 32 | 32 | 127 | 156 | 244 | — | flat |
| avx2 | 32 | 40 | 127 | 161 | 133 | — | flat |
| avx2 | 32 | 352 | 129 | 166 | 148 | — | flat |
| avx2 | 32 | 1024 | 303 | 300 | 448 | — | log3 |
| avx2 | 64 | 64 | 253 | 319 | 252 | — | t1s |
| avx2 | 64 | 72 | 264 | 322 | 305 | — | flat |
| avx2 | 64 | 704 | 258 | 346 | 262 | — | flat |
| avx2 | 64 | 2048 | 690 | 717 | 776 | — | flat |
| avx2 | 128 | 128 | 506 | 615 | 585 | — | flat |
| avx2 | 128 | 136 | 504 | 618 | 488 | — | t1s |
| avx2 | 128 | 1408 | 541 | 638 | 502 | — | t1s |
| avx2 | 128 | 4096 | 1414 | 1354 | 1464 | — | log3 |
| avx2 | 256 | 256 | 1176 | 1217 | 1124 | — | t1s |
| avx2 | 256 | 264 | 1266 | 1247 | 1957 | — | log3 |
| avx2 | 256 | 2816 | 1278 | 1303 | 1222 | — | t1s |
| avx2 | 256 | 8192 | 2062 | 1784 | 1395 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 8 | 88 | `ct_t1_dit` | 39 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 38 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 68 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 67 |
| avx2 | `t1_dit` | 16 | 176 | `ct_t1_dit` | 71 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 210 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 127 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 127 |
| avx2 | `t1_dit` | 32 | 352 | `ct_t1_dit` | 129 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 303 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 253 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 264 |
| avx2 | `t1_dit` | 64 | 704 | `ct_t1_dit` | 258 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 690 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 506 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 504 |
| avx2 | `t1_dit` | 128 | 1408 | `ct_t1_dit` | 541 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1414 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1176 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1266 |
| avx2 | `t1_dit` | 256 | 2816 | `ct_t1_dit` | 1278 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2062 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 45 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 44 |
| avx2 | `t1_dit_log3` | 8 | 88 | `ct_t1_dit_log3` | 45 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 48 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 89 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 81 |
| avx2 | `t1_dit_log3` | 16 | 176 | `ct_t1_dit_log3` | 83 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 225 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 156 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 161 |
| avx2 | `t1_dit_log3` | 32 | 352 | `ct_t1_dit_log3` | 166 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 300 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 319 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 322 |
| avx2 | `t1_dit_log3` | 64 | 704 | `ct_t1_dit_log3` | 346 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 717 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 615 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 618 |
| avx2 | `t1_dit_log3` | 128 | 1408 | `ct_t1_dit_log3` | 638 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1354 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1217 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1247 |
| avx2 | `t1_dit_log3` | 256 | 2816 | `ct_t1_dit_log3` | 1303 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1784 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 39 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 39 |
| avx2 | `t1s_dit` | 8 | 88 | `ct_t1s_dit` | 39 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 46 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 69 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 116 |
| avx2 | `t1s_dit` | 16 | 176 | `ct_t1s_dit` | 69 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 174 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 244 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 133 |
| avx2 | `t1s_dit` | 32 | 352 | `ct_t1s_dit` | 148 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 448 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 252 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 305 |
| avx2 | `t1s_dit` | 64 | 704 | `ct_t1s_dit` | 262 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 776 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 585 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 488 |
| avx2 | `t1s_dit` | 128 | 1408 | `ct_t1s_dit` | 502 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1464 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1124 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1957 |
| avx2 | `t1s_dit` | 256 | 2816 | `ct_t1s_dit` | 1222 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 1395 |