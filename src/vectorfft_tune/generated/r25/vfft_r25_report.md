# VectorFFT R=25 tuning report

Total measurements: **256**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 79 | 108 | 90 | — | flat |
| avx2 | 8 | 16 | 80 | 105 | 88 | — | flat |
| avx2 | 8 | 200 | 111 | 92 | 94 | — | log3 |
| avx2 | 8 | 256 | 146 | 145 | 178 | — | log3 |
| avx2 | 16 | 16 | 151 | 177 | 163 | — | flat |
| avx2 | 16 | 24 | 156 | 181 | 157 | — | flat |
| avx2 | 16 | 400 | 154 | 181 | 175 | — | flat |
| avx2 | 16 | 512 | 335 | 370 | 384 | — | flat |
| avx2 | 32 | 32 | 307 | 348 | 309 | — | flat |
| avx2 | 32 | 40 | 298 | 345 | 303 | — | flat |
| avx2 | 32 | 800 | 380 | 387 | 337 | — | t1s |
| avx2 | 32 | 1024 | 735 | 868 | 700 | — | t1s |
| avx2 | 64 | 64 | 623 | 691 | 604 | — | t1s |
| avx2 | 64 | 72 | 626 | 737 | 632 | — | flat |
| avx2 | 64 | 1600 | 657 | 769 | 616 | — | t1s |
| avx2 | 64 | 2048 | 1623 | 1469 | 1390 | — | t1s |
| avx2 | 128 | 128 | 1843 | 1735 | 1258 | — | t1s |
| avx2 | 128 | 136 | 1641 | 1444 | 1279 | — | t1s |
| avx2 | 128 | 3200 | 1646 | 1417 | 1394 | — | t1s |
| avx2 | 128 | 4096 | 2760 | 2964 | 2781 | — | flat |
| avx2 | 256 | 256 | 5446 | 6810 | 4806 | — | t1s |
| avx2 | 256 | 264 | 4148 | 3249 | 2752 | — | t1s |
| avx2 | 256 | 6400 | 6259 | 7167 | 5932 | — | t1s |
| avx2 | 256 | 8192 | 7100 | 8266 | 7405 | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 79 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 80 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 111 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 146 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 151 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 156 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 154 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 335 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 307 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 298 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_buf_dit_tile32_temporal` | 380 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 735 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 623 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 626 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 657 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_buf_dit_tile32_temporal` | 1623 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1843 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1641 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 1646 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 2760 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 5446 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4148 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_dit` | 6259 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 7100 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 108 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 105 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 92 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 145 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 177 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 181 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 181 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 370 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 348 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 345 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 387 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 868 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 691 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 737 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 769 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1469 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1735 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1444 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 1417 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 2964 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6810 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3249 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 7167 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 8266 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 90 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 88 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 94 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 178 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 163 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 157 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 175 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 384 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 309 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 303 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 337 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 700 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 604 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 632 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 616 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1390 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1258 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1279 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 1394 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2781 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4806 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 2752 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 5932 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 7405 |