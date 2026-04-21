# VectorFFT R=3 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 24 | 24 | 9 | 11 | 10 | — | flat |
| avx2 | 24 | 32 | 9 | 11 | 9 | — | flat |
| avx2 | 24 | 72 | 10 | 11 | 10 | — | flat |
| avx2 | 24 | 192 | 10 | 11 | 11 | — | flat |
| avx2 | 48 | 48 | 18 | 21 | 20 | — | flat |
| avx2 | 48 | 56 | 18 | 21 | 17 | — | t1s |
| avx2 | 48 | 144 | 19 | 22 | 19 | — | flat |
| avx2 | 48 | 384 | 18 | 20 | 19 | — | flat |
| avx2 | 96 | 96 | 35 | 41 | 34 | — | t1s |
| avx2 | 96 | 104 | 35 | 40 | 40 | — | flat |
| avx2 | 96 | 288 | 34 | 40 | 36 | — | flat |
| avx2 | 96 | 768 | 35 | 41 | 35 | — | t1s |
| avx2 | 192 | 192 | 72 | 77 | 68 | — | t1s |
| avx2 | 192 | 200 | 69 | 85 | 71 | — | flat |
| avx2 | 192 | 576 | 66 | 82 | 68 | — | flat |
| avx2 | 192 | 1536 | 66 | 80 | 177 | — | flat |
| avx2 | 384 | 384 | 135 | 160 | 137 | — | flat |
| avx2 | 384 | 392 | 130 | 167 | 130 | — | flat |
| avx2 | 384 | 1152 | 136 | 167 | 146 | — | flat |
| avx2 | 384 | 3072 | 134 | 164 | 139 | — | flat |
| avx2 | 768 | 768 | 409 | 327 | 259 | — | t1s |
| avx2 | 768 | 776 | 443 | 329 | 258 | — | t1s |
| avx2 | 768 | 2304 | 376 | 352 | 268 | — | t1s |
| avx2 | 768 | 6144 | 330 | 358 | 276 | — | t1s |
| avx2 | 1536 | 1536 | 936 | 925 | 840 | — | t1s |
| avx2 | 1536 | 1544 | 916 | 846 | 849 | — | log3 |
| avx2 | 1536 | 4608 | 1016 | 947 | 851 | — | t1s |
| avx2 | 1536 | 12288 | 932 | 933 | 779 | — | t1s |
| avx2 | 3072 | 3072 | 1875 | 1762 | 1731 | — | t1s |
| avx2 | 3072 | 3080 | 1813 | 1787 | 1672 | — | t1s |
| avx2 | 3072 | 9216 | 1886 | 1958 | 1599 | — | t1s |
| avx2 | 3072 | 24576 | 1905 | 1683 | 1636 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 24 | 72 | `ct_t1_dit_u1` | 10 |
| avx2 | `t1_dit` | 24 | 192 | `ct_t1_dit_u1` | 10 |
| avx2 | `t1_dit` | 48 | 48 | `ct_t1_dit_u1` | 18 |
| avx2 | `t1_dit` | 48 | 56 | `ct_t1_dit_u1` | 18 |
| avx2 | `t1_dit` | 48 | 144 | `ct_t1_dit_u1` | 19 |
| avx2 | `t1_dit` | 48 | 384 | `ct_t1_dit_u1` | 18 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit_u1` | 35 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit_u1` | 35 |
| avx2 | `t1_dit` | 96 | 288 | `ct_t1_dit_u1` | 34 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit_u1` | 35 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit_u1` | 72 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit_u1` | 69 |
| avx2 | `t1_dit` | 192 | 576 | `ct_t1_dit_u1` | 66 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u1` | 66 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit_u1` | 135 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit_u1` | 130 |
| avx2 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u1` | 136 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u1` | 134 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit_u1` | 409 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit_u1` | 443 |
| avx2 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u1` | 376 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u1` | 330 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u1` | 936 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u1` | 916 |
| avx2 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u1` | 1016 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u1` | 932 |
| avx2 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u1` | 1875 |
| avx2 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u1` | 1813 |
| avx2 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u1` | 1886 |
| avx2 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u1` | 1905 |
| avx2 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 11 |
| avx2 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 11 |
| avx2 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u1` | 11 |
| avx2 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u1` | 11 |
| avx2 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u1` | 21 |
| avx2 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u1` | 21 |
| avx2 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u1` | 22 |
| avx2 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u1` | 20 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u1` | 41 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u1` | 40 |
| avx2 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u1` | 40 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u1` | 41 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u1` | 77 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u1` | 85 |
| avx2 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u1` | 82 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u1` | 80 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u1` | 160 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u1` | 167 |
| avx2 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u1` | 167 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u1` | 164 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u1` | 327 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u1` | 329 |
| avx2 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u1` | 352 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u1` | 358 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 925 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u1` | 846 |
| avx2 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u1` | 947 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u1` | 933 |
| avx2 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u1` | 1762 |
| avx2 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 1787 |
| avx2 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u1` | 1958 |
| avx2 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u1` | 1683 |
| avx2 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u1` | 10 |
| avx2 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u1` | 10 |
| avx2 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 11 |
| avx2 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u1` | 20 |
| avx2 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u1` | 19 |
| avx2 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u1` | 19 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u1` | 34 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u1` | 40 |
| avx2 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u1` | 36 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u1` | 35 |
| avx2 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u1` | 68 |
| avx2 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u1` | 71 |
| avx2 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u1` | 68 |
| avx2 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u1` | 177 |
| avx2 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u1` | 137 |
| avx2 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u1` | 130 |
| avx2 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u1` | 146 |
| avx2 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u1` | 139 |
| avx2 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u1` | 259 |
| avx2 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u1` | 258 |
| avx2 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u1` | 268 |
| avx2 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u1` | 276 |
| avx2 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u1` | 840 |
| avx2 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u1` | 849 |
| avx2 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u1` | 851 |
| avx2 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 779 |
| avx2 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 1731 |
| avx2 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u1` | 1672 |
| avx2 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 1599 |
| avx2 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 1636 |