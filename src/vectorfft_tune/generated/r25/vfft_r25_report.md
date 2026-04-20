# VectorFFT R=25 tuning report

Total measurements: **144**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 78 | 103 | 91 | — | flat |
| avx2 | 8 | 16 | 80 | 91 | 92 | — | flat |
| avx2 | 8 | 200 | 79 | 92 | 87 | — | flat |
| avx2 | 8 | 256 | 125 | 156 | 135 | — | flat |
| avx2 | 16 | 16 | 152 | 177 | 165 | — | flat |
| avx2 | 16 | 24 | 194 | 180 | 166 | — | t1s |
| avx2 | 16 | 400 | 159 | 181 | 171 | — | flat |
| avx2 | 16 | 512 | 339 | 354 | 339 | — | t1s |
| avx2 | 32 | 32 | 299 | 353 | 313 | — | flat |
| avx2 | 32 | 40 | 296 | 344 | 300 | — | flat |
| avx2 | 32 | 800 | 300 | 359 | 313 | — | flat |
| avx2 | 32 | 1024 | 683 | 701 | 653 | — | t1s |
| avx2 | 64 | 64 | 614 | 693 | 588 | — | t1s |
| avx2 | 64 | 72 | 680 | 689 | 610 | — | t1s |
| avx2 | 64 | 1600 | 675 | 755 | 685 | — | flat |
| avx2 | 64 | 2048 | 1578 | 1434 | 1427 | — | t1s |
| avx2 | 128 | 128 | 1578 | 1394 | 1215 | — | t1s |
| avx2 | 128 | 136 | 1608 | 1755 | 1225 | — | t1s |
| avx2 | 128 | 3200 | 2096 | 1548 | 1306 | — | t1s |
| avx2 | 128 | 4096 | 3232 | 3630 | 2702 | — | t1s |
| avx2 | 256 | 256 | 5391 | 6063 | 5083 | — | t1s |
| avx2 | 256 | 264 | 3443 | 3049 | 2607 | — | t1s |
| avx2 | 256 | 6400 | 7255 | 5919 | 5576 | — | t1s |
| avx2 | 256 | 8192 | 11395 | 7914 | 6697 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 78 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 80 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 79 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 125 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 152 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 194 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 159 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 339 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 299 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 296 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 300 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 683 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 614 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 680 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 675 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 1578 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1578 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1608 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 2096 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 3232 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5391 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3443 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_dit` | 7255 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 11395 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 103 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 91 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 92 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 156 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 177 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 180 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 181 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 354 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 353 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 344 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 359 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 701 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 693 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 689 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 755 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1434 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1394 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1755 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 1548 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3630 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6063 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3049 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 5919 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 7914 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 91 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 92 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 87 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 135 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 165 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 166 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 171 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 339 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 313 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 300 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 313 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 653 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 588 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 610 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 685 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1427 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1215 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1225 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 1306 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2702 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 5083 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2607 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 5576 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 6697 |