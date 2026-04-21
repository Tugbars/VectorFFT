# VectorFFT R=12 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 29 | 40 | 36 | — | flat |
| avx2 | 8 | 16 | 29 | 40 | 41 | — | flat |
| avx2 | 8 | 96 | 31 | 39 | 32 | — | flat |
| avx2 | 8 | 256 | 29 | 39 | 30 | — | flat |
| avx2 | 16 | 16 | 54 | 78 | 55 | — | flat |
| avx2 | 16 | 24 | 55 | 75 | 53 | — | t1s |
| avx2 | 16 | 192 | 59 | 76 | 53 | — | t1s |
| avx2 | 16 | 512 | 230 | 209 | 203 | — | t1s |
| avx2 | 32 | 32 | 111 | 147 | 103 | — | t1s |
| avx2 | 32 | 40 | 115 | 152 | 101 | — | t1s |
| avx2 | 32 | 384 | 109 | 156 | 103 | — | t1s |
| avx2 | 32 | 1024 | 371 | 307 | 265 | — | t1s |
| avx2 | 64 | 64 | 209 | 305 | 201 | — | t1s |
| avx2 | 64 | 72 | 239 | 296 | 199 | — | t1s |
| avx2 | 64 | 768 | 495 | 326 | 211 | — | t1s |
| avx2 | 64 | 2048 | 610 | 617 | 680 | — | flat |
| avx2 | 128 | 128 | 437 | 592 | 388 | — | t1s |
| avx2 | 128 | 136 | 434 | 588 | 385 | — | t1s |
| avx2 | 128 | 1536 | 1331 | 1278 | 1334 | — | log3 |
| avx2 | 128 | 4096 | 1226 | 1337 | 1329 | — | flat |
| avx2 | 256 | 256 | 1122 | 1221 | 797 | — | t1s |
| avx2 | 256 | 264 | 1111 | 1193 | 819 | — | t1s |
| avx2 | 256 | 3072 | 2745 | 2597 | 2606 | — | log3 |
| avx2 | 256 | 8192 | 2373 | 1521 | 1611 | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 29 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 29 |
| avx2 | `t1_dit` | 8 | 96 | `ct_t1_dit` | 31 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 29 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 54 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 55 |
| avx2 | `t1_dit` | 16 | 192 | `ct_t1_dit` | 59 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 230 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 111 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 115 |
| avx2 | `t1_dit` | 32 | 384 | `ct_t1_dit` | 109 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 371 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 209 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 239 |
| avx2 | `t1_dit` | 64 | 768 | `ct_t1_dit` | 495 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 610 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 437 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 434 |
| avx2 | `t1_dit` | 128 | 1536 | `ct_t1_dit` | 1331 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 1226 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1122 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1111 |
| avx2 | `t1_dit` | 256 | 3072 | `ct_t1_dit` | 2745 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 2373 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 40 |
| avx2 | `t1_dit_log3` | 8 | 96 | `ct_t1_dit_log3` | 39 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 39 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 78 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 75 |
| avx2 | `t1_dit_log3` | 16 | 192 | `ct_t1_dit_log3` | 76 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 209 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 147 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 152 |
| avx2 | `t1_dit_log3` | 32 | 384 | `ct_t1_dit_log3` | 156 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 307 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 305 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 296 |
| avx2 | `t1_dit_log3` | 64 | 768 | `ct_t1_dit_log3` | 326 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 617 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 592 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 588 |
| avx2 | `t1_dit_log3` | 128 | 1536 | `ct_t1_dit_log3` | 1278 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 1337 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 1221 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 1193 |
| avx2 | `t1_dit_log3` | 256 | 3072 | `ct_t1_dit_log3` | 2597 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 1521 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 36 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 41 |
| avx2 | `t1s_dit` | 8 | 96 | `ct_t1s_dit` | 32 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 30 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 55 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 53 |
| avx2 | `t1s_dit` | 16 | 192 | `ct_t1s_dit` | 53 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 203 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 103 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 101 |
| avx2 | `t1s_dit` | 32 | 384 | `ct_t1s_dit` | 103 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 265 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 201 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 199 |
| avx2 | `t1s_dit` | 64 | 768 | `ct_t1s_dit` | 211 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 680 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 388 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 385 |
| avx2 | `t1s_dit` | 128 | 1536 | `ct_t1s_dit` | 1334 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 1329 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 797 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 819 |
| avx2 | `t1s_dit` | 256 | 3072 | `ct_t1s_dit` | 2606 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 1611 |