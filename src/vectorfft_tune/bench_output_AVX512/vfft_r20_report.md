# VectorFFT R=20 tuning report

Total measurements: **288**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 235 | 211 | 187 | — | t1s |
| avx2 | 8 | 16 | 225 | 209 | 191 | — | t1s |
| avx2 | 8 | 160 | 228 | 203 | 185 | — | t1s |
| avx2 | 8 | 256 | 301 | 253 | 252 | — | t1s |
| avx2 | 16 | 16 | 446 | 429 | 359 | — | t1s |
| avx2 | 16 | 24 | 441 | 415 | 351 | — | t1s |
| avx2 | 16 | 320 | 458 | 419 | 359 | — | t1s |
| avx2 | 16 | 512 | 806 | 759 | 663 | — | t1s |
| avx2 | 32 | 32 | 829 | 815 | 674 | — | t1s |
| avx2 | 32 | 40 | 905 | 781 | 693 | — | t1s |
| avx2 | 32 | 640 | 900 | 866 | 734 | — | t1s |
| avx2 | 32 | 1024 | 1860 | 1668 | 1478 | — | t1s |
| avx2 | 64 | 64 | 1804 | 1572 | 1377 | — | t1s |
| avx2 | 64 | 72 | 1832 | 1555 | 1332 | — | t1s |
| avx2 | 64 | 1280 | 3156 | 3107 | 2393 | — | t1s |
| avx2 | 64 | 2048 | 3757 | 3379 | 2932 | — | t1s |
| avx2 | 128 | 128 | 3786 | 3477 | 2868 | — | t1s |
| avx2 | 128 | 136 | 3774 | 3404 | 2761 | — | t1s |
| avx2 | 128 | 2560 | 7303 | 7439 | 5957 | — | t1s |
| avx2 | 128 | 4096 | 7412 | 7307 | 5885 | — | t1s |
| avx2 | 256 | 256 | 10164 | 9085 | 8421 | — | t1s |
| avx2 | 256 | 264 | 7798 | 6584 | 5534 | — | t1s |
| avx2 | 256 | 5120 | 15684 | 14158 | 11890 | — | t1s |
| avx2 | 256 | 8192 | 15742 | 15068 | 12061 | — | t1s |
| avx512 | 8 | 8 | 155 | 124 | 125 | — | log3 |
| avx512 | 8 | 16 | 162 | 124 | 126 | — | log3 |
| avx512 | 8 | 160 | 162 | 126 | 123 | — | t1s |
| avx512 | 8 | 256 | 225 | 161 | 195 | — | log3 |
| avx512 | 16 | 16 | 293 | 236 | 212 | — | t1s |
| avx512 | 16 | 24 | 288 | 239 | 215 | — | t1s |
| avx512 | 16 | 320 | 278 | 221 | 215 | — | t1s |
| avx512 | 16 | 512 | 523 | 441 | 410 | — | t1s |
| avx512 | 32 | 32 | 516 | 460 | 409 | — | t1s |
| avx512 | 32 | 40 | 527 | 459 | 390 | — | t1s |
| avx512 | 32 | 640 | 622 | 501 | 510 | — | log3 |
| avx512 | 32 | 1024 | 1061 | 953 | 768 | — | t1s |
| avx512 | 64 | 64 | 1143 | 896 | 763 | — | t1s |
| avx512 | 64 | 72 | 1140 | 904 | 781 | — | t1s |
| avx512 | 64 | 1280 | 1931 | 1542 | 1347 | — | t1s |
| avx512 | 64 | 2048 | 2143 | 1921 | 1404 | — | t1s |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 235 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 225 |
| avx2 | `t1_dit` | 8 | 160 | `ct_t1_dit` | 228 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 301 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 446 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 441 |
| avx2 | `t1_dit` | 16 | 320 | `ct_t1_dit` | 458 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 806 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 829 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 905 |
| avx2 | `t1_dit` | 32 | 640 | `ct_t1_dit` | 900 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1860 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1804 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1832 |
| avx2 | `t1_dit` | 64 | 1280 | `ct_t1_dit` | 3156 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 3757 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3786 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 3774 |
| avx2 | `t1_dit` | 128 | 2560 | `ct_t1_dit` | 7303 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 7412 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 10164 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 7798 |
| avx2 | `t1_dit` | 256 | 5120 | `ct_t1_dit` | 15684 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 15742 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 211 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 209 |
| avx2 | `t1_dit_log3` | 8 | 160 | `ct_t1_dit_log3` | 203 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 253 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 429 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 415 |
| avx2 | `t1_dit_log3` | 16 | 320 | `ct_t1_dit_log3` | 419 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 759 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 815 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 781 |
| avx2 | `t1_dit_log3` | 32 | 640 | `ct_t1_dit_log3` | 866 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 1668 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1572 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1555 |
| avx2 | `t1_dit_log3` | 64 | 1280 | `ct_t1_dit_log3` | 3107 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 3379 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3477 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 3404 |
| avx2 | `t1_dit_log3` | 128 | 2560 | `ct_t1_dit_log3` | 7439 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 7307 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 9085 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 6584 |
| avx2 | `t1_dit_log3` | 256 | 5120 | `ct_t1_dit_log3` | 14158 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 15068 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 187 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 191 |
| avx2 | `t1s_dit` | 8 | 160 | `ct_t1s_dit` | 185 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 252 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 359 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 351 |
| avx2 | `t1s_dit` | 16 | 320 | `ct_t1s_dit` | 359 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 663 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 674 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 693 |
| avx2 | `t1s_dit` | 32 | 640 | `ct_t1s_dit` | 734 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 1478 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1377 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1332 |
| avx2 | `t1s_dit` | 64 | 1280 | `ct_t1s_dit` | 2393 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 2932 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2868 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2761 |
| avx2 | `t1s_dit` | 128 | 2560 | `ct_t1s_dit` | 5957 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 5885 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 8421 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 5534 |
| avx2 | `t1s_dit` | 256 | 5120 | `ct_t1s_dit` | 11890 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 12061 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 155 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 162 |
| avx512 | `t1_dit` | 8 | 160 | `ct_t1_dit` | 162 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 225 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 293 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 288 |
| avx512 | `t1_dit` | 16 | 320 | `ct_t1_dit` | 278 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 523 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 516 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 527 |
| avx512 | `t1_dit` | 32 | 640 | `ct_t1_dit` | 622 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1061 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1143 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1140 |
| avx512 | `t1_dit` | 64 | 1280 | `ct_t1_dit` | 1931 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 2143 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2427 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2396 |
| avx512 | `t1_dit` | 128 | 2560 | `ct_t1_dit` | 4242 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 4562 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6238 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4918 |
| avx512 | `t1_dit` | 256 | 5120 | `ct_t1_dit` | 8717 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 9089 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 124 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 124 |
| avx512 | `t1_dit_log3` | 8 | 160 | `ct_t1_dit_log3` | 126 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 161 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 236 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 239 |
| avx512 | `t1_dit_log3` | 16 | 320 | `ct_t1_dit_log3` | 221 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 441 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 460 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 459 |
| avx512 | `t1_dit_log3` | 32 | 640 | `ct_t1_dit_log3` | 501 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 953 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 896 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 904 |
| avx512 | `t1_dit_log3` | 64 | 1280 | `ct_t1_dit_log3` | 1542 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1921 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1966 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1870 |
| avx512 | `t1_dit_log3` | 128 | 2560 | `ct_t1_dit_log3` | 3836 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3694 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 5522 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3946 |
| avx512 | `t1_dit_log3` | 256 | 5120 | `ct_t1_dit_log3` | 7074 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 7381 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 125 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 126 |
| avx512 | `t1s_dit` | 8 | 160 | `ct_t1s_dit` | 123 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 195 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 212 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 215 |
| avx512 | `t1s_dit` | 16 | 320 | `ct_t1s_dit` | 215 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 410 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 409 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 390 |
| avx512 | `t1s_dit` | 32 | 640 | `ct_t1s_dit` | 510 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 768 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 763 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 781 |
| avx512 | `t1s_dit` | 64 | 1280 | `ct_t1s_dit` | 1347 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1404 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1859 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1694 |
| avx512 | `t1s_dit` | 128 | 2560 | `ct_t1s_dit` | 2924 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 2913 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 5084 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 3496 |
| avx512 | `t1s_dit` | 256 | 5120 | `ct_t1s_dit` | 5767 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 6086 |