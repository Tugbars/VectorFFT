# VectorFFT R=12 tuning report

Total measurements: **448**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 80 | 81 | 65 | — | t1s |
| avx2 | 8 | 16 | 88 | 79 | 64 | — | t1s |
| avx2 | 8 | 96 | 150 | 85 | 64 | — | t1s |
| avx2 | 8 | 256 | 88 | 80 | 63 | — | t1s |
| avx2 | 16 | 16 | 160 | 216 | 123 | — | t1s |
| avx2 | 16 | 24 | 149 | 159 | 131 | — | t1s |
| avx2 | 16 | 192 | 154 | 294 | 126 | — | t1s |
| avx2 | 16 | 512 | 306 | 336 | 326 | — | flat |
| avx2 | 32 | 32 | 333 | 399 | 267 | — | t1s |
| avx2 | 32 | 40 | 306 | 315 | 248 | — | t1s |
| avx2 | 32 | 384 | 309 | 316 | 247 | — | t1s |
| avx2 | 32 | 1024 | 612 | 623 | 660 | — | flat |
| avx2 | 64 | 64 | 608 | 647 | 516 | — | t1s |
| avx2 | 64 | 72 | 612 | 682 | 498 | — | t1s |
| avx2 | 64 | 768 | 718 | 642 | 505 | — | t1s |
| avx2 | 64 | 2048 | 1830 | 1585 | 1646 | — | log3 |
| avx2 | 128 | 128 | 1254 | 1239 | 970 | — | t1s |
| avx2 | 128 | 136 | 1195 | 1257 | 1066 | — | t1s |
| avx2 | 128 | 1536 | 3446 | 3106 | 3299 | — | log3 |
| avx2 | 128 | 4096 | 3511 | 2879 | 3261 | — | log3 |
| avx2 | 256 | 256 | 2918 | 2538 | 1990 | — | t1s |
| avx2 | 256 | 264 | 2774 | 2617 | 2071 | — | t1s |
| avx2 | 256 | 3072 | 7142 | 5933 | 6344 | — | log3 |
| avx2 | 256 | 8192 | 7106 | 6127 | 6376 | — | log3 |
| avx512 | 8 | 8 | 50 | 46 | 47 | — | log3 |
| avx512 | 8 | 16 | 49 | 42 | 47 | — | log3 |
| avx512 | 8 | 96 | 53 | 43 | 44 | — | log3 |
| avx512 | 8 | 256 | 57 | 45 | 44 | — | t1s |
| avx512 | 16 | 16 | 85 | 77 | 78 | — | log3 |
| avx512 | 16 | 24 | 84 | 78 | 78 | — | t1s |
| avx512 | 16 | 192 | 88 | 79 | 76 | — | t1s |
| avx512 | 16 | 512 | 179 | 120 | 168 | — | log3 |
| avx512 | 32 | 32 | 169 | 156 | 150 | — | t1s |
| avx512 | 32 | 40 | 162 | 153 | 135 | — | t1s |
| avx512 | 32 | 384 | 173 | 165 | 153 | — | t1s |
| avx512 | 32 | 1024 | 318 | 296 | 351 | — | log3 |
| avx512 | 64 | 64 | 309 | 320 | 291 | — | t1s |
| avx512 | 64 | 72 | 309 | 320 | 284 | — | t1s |
| avx512 | 64 | 768 | 445 | 320 | 290 | — | t1s |
| avx512 | 64 | 2048 | 818 | 629 | 757 | — | log3 |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 80 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 88 |
| avx2 | `t1_dit` | 8 | 96 | `ct_t1_dit` | 150 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 88 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 160 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 149 |
| avx2 | `t1_dit` | 16 | 192 | `ct_t1_dit` | 154 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 306 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 333 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 306 |
| avx2 | `t1_dit` | 32 | 384 | `ct_t1_dit` | 309 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 612 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 608 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 612 |
| avx2 | `t1_dit` | 64 | 768 | `ct_t1_dit` | 718 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1830 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1254 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1195 |
| avx2 | `t1_dit` | 128 | 1536 | `ct_t1_dit` | 3446 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3511 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 2918 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 2774 |
| avx2 | `t1_dit` | 256 | 3072 | `ct_t1_dit` | 7142 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 7106 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 81 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 79 |
| avx2 | `t1_dit_log3` | 8 | 96 | `ct_t1_dit_log3` | 85 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 80 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 216 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 159 |
| avx2 | `t1_dit_log3` | 16 | 192 | `ct_t1_dit_log3` | 294 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 336 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 399 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 315 |
| avx2 | `t1_dit_log3` | 32 | 384 | `ct_t1_dit_log3` | 316 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 623 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 647 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 682 |
| avx2 | `t1_dit_log3` | 64 | 768 | `ct_t1_dit_log3` | 642 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1585 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1239 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1257 |
| avx2 | `t1_dit_log3` | 128 | 1536 | `ct_t1_dit_log3` | 3106 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2879 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 2538 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 2617 |
| avx2 | `t1_dit_log3` | 256 | 3072 | `ct_t1_dit_log3` | 5933 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 6127 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 65 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 64 |
| avx2 | `t1s_dit` | 8 | 96 | `ct_t1s_dit` | 64 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 63 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 123 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 131 |
| avx2 | `t1s_dit` | 16 | 192 | `ct_t1s_dit` | 126 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 326 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 267 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 248 |
| avx2 | `t1s_dit` | 32 | 384 | `ct_t1s_dit` | 247 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 660 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 516 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 498 |
| avx2 | `t1s_dit` | 64 | 768 | `ct_t1s_dit` | 505 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1646 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 970 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1066 |
| avx2 | `t1s_dit` | 128 | 1536 | `ct_t1s_dit` | 3299 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 3261 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1990 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2071 |
| avx2 | `t1s_dit` | 256 | 3072 | `ct_t1s_dit` | 6344 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 6376 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 50 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 49 |
| avx512 | `t1_dit` | 8 | 96 | `ct_t1_dit` | 53 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 57 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit_u2b` | 85 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit_u2b` | 84 |
| avx512 | `t1_dit` | 16 | 192 | `ct_t1_dit_u2a` | 88 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 179 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 169 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 162 |
| avx512 | `t1_dit` | 32 | 384 | `ct_t1_dit` | 173 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 318 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit_u2b` | 309 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit_u2b` | 309 |
| avx512 | `t1_dit` | 64 | 768 | `ct_t1_dit` | 445 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 818 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit_u2b` | 606 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit_u2b` | 584 |
| avx512 | `t1_dit` | 128 | 1536 | `ct_t1_dit` | 1614 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1587 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit_u2b` | 1971 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_u2b` | 1574 |
| avx512 | `t1_dit` | 256 | 3072 | `ct_t1_dit` | 3317 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 3413 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 46 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 42 |
| avx512 | `t1_dit_log3` | 8 | 96 | `ct_t1_dit_log3` | 43 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 45 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3_u2b` | 77 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3_u2a` | 78 |
| avx512 | `t1_dit_log3` | 16 | 192 | `ct_t1_dit_log3_u2b` | 79 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3_u2b` | 120 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3_u2b` | 156 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3_u2b` | 153 |
| avx512 | `t1_dit_log3` | 32 | 384 | `ct_t1_dit_log3_u2b` | 165 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3_u2b` | 296 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3_u2b` | 320 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3_u2a` | 320 |
| avx512 | `t1_dit_log3` | 64 | 768 | `ct_t1_dit_log3_u2b` | 320 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3_u2b` | 629 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3_u2a` | 612 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3_u2a` | 631 |
| avx512 | `t1_dit_log3` | 128 | 1536 | `ct_t1_dit_log3_u2a` | 1279 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3_u2b` | 1310 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3_u2a` | 1282 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3_u2a` | 1278 |
| avx512 | `t1_dit_log3` | 256 | 3072 | `ct_t1_dit_log3_u2a` | 2684 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3_u2b` | 2749 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 47 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 47 |
| avx512 | `t1s_dit` | 8 | 96 | `ct_t1s_dit` | 44 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 44 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 78 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 78 |
| avx512 | `t1s_dit` | 16 | 192 | `ct_t1s_dit` | 76 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 168 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 150 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 135 |
| avx512 | `t1s_dit` | 32 | 384 | `ct_t1s_dit` | 153 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 351 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 291 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 284 |
| avx512 | `t1s_dit` | 64 | 768 | `ct_t1s_dit` | 290 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 757 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 559 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 642 |
| avx512 | `t1s_dit` | 128 | 1536 | `ct_t1s_dit` | 1427 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1445 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 1301 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 1087 |
| avx512 | `t1s_dit` | 256 | 3072 | `ct_t1s_dit` | 2714 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 2691 |