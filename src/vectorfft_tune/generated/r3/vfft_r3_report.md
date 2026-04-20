# VectorFFT R=3 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 24 | 24 | 9 | 10 | 9 | — | t1s |
| avx2 | 24 | 32 | 9 | 10 | 9 | — | t1s |
| avx2 | 24 | 72 | 9 | 13 | 9 | — | t1s |
| avx2 | 24 | 192 | 9 | 10 | 9 | — | t1s |
| avx2 | 48 | 48 | 17 | 20 | 17 | — | t1s |
| avx2 | 48 | 56 | 17 | 19 | 17 | — | t1s |
| avx2 | 48 | 144 | 17 | 20 | 17 | — | t1s |
| avx2 | 48 | 384 | 17 | 19 | 17 | — | flat |
| avx2 | 96 | 96 | 33 | 40 | 34 | — | flat |
| avx2 | 96 | 104 | 35 | 41 | 33 | — | t1s |
| avx2 | 96 | 288 | 33 | 42 | 33 | — | t1s |
| avx2 | 96 | 768 | 33 | 42 | 33 | — | flat |
| avx2 | 192 | 192 | 65 | 80 | 66 | — | flat |
| avx2 | 192 | 200 | 64 | 85 | 65 | — | flat |
| avx2 | 192 | 576 | 64 | 78 | 66 | — | flat |
| avx2 | 192 | 1536 | 65 | 79 | 173 | — | flat |
| avx2 | 384 | 384 | 128 | 210 | 128 | — | flat |
| avx2 | 384 | 392 | 131 | 159 | 129 | — | t1s |
| avx2 | 384 | 1152 | 128 | 157 | 128 | — | flat |
| avx2 | 384 | 3072 | 131 | 170 | 128 | — | t1s |
| avx2 | 768 | 768 | 358 | 320 | 257 | — | t1s |
| avx2 | 768 | 776 | 385 | 318 | 254 | — | t1s |
| avx2 | 768 | 2304 | 368 | 329 | 259 | — | t1s |
| avx2 | 768 | 6144 | 321 | 333 | 260 | — | t1s |
| avx2 | 1536 | 1536 | 913 | 882 | 742 | — | t1s |
| avx2 | 1536 | 1544 | 897 | 862 | 770 | — | t1s |
| avx2 | 1536 | 4608 | 918 | 856 | 750 | — | t1s |
| avx2 | 1536 | 12288 | 913 | 884 | 751 | — | t1s |
| avx2 | 3072 | 3072 | 1775 | 1727 | 1586 | — | t1s |
| avx2 | 3072 | 3080 | 1753 | 1824 | 1586 | — | t1s |
| avx2 | 3072 | 9216 | 1771 | 1719 | 2025 | — | log3 |
| avx2 | 3072 | 24576 | 1765 | 1721 | 1591 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 24 | 72 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 24 | 192 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 48 | 48 | `ct_t1_dit_u1` | 17 |
| avx2 | `t1_dit` | 48 | 56 | `ct_t1_dit_u1` | 17 |
| avx2 | `t1_dit` | 48 | 144 | `ct_t1_dit_u1` | 17 |
| avx2 | `t1_dit` | 48 | 384 | `ct_t1_dit_u1` | 17 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit_u1` | 33 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit_u1` | 35 |
| avx2 | `t1_dit` | 96 | 288 | `ct_t1_dit_u1` | 33 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit_u1` | 33 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit_u1` | 65 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit_u1` | 64 |
| avx2 | `t1_dit` | 192 | 576 | `ct_t1_dit_u1` | 64 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u1` | 65 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit_u1` | 128 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit_u1` | 131 |
| avx2 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u1` | 128 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u1` | 131 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit_u1` | 358 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit_u1` | 385 |
| avx2 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u1` | 368 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u1` | 321 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u1` | 913 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u1` | 897 |
| avx2 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u1` | 918 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u1` | 913 |
| avx2 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u1` | 1775 |
| avx2 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u1` | 1753 |
| avx2 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u1` | 1771 |
| avx2 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u1` | 1765 |
| avx2 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 10 |
| avx2 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 10 |
| avx2 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u1` | 13 |
| avx2 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u1` | 10 |
| avx2 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u1` | 20 |
| avx2 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u1` | 19 |
| avx2 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u1` | 20 |
| avx2 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u1` | 19 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u1` | 40 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u1` | 41 |
| avx2 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u1` | 42 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u1` | 42 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u1` | 80 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u1` | 85 |
| avx2 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u1` | 78 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u1` | 79 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u1` | 210 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u1` | 159 |
| avx2 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u1` | 157 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u1` | 170 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u1` | 320 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u1` | 318 |
| avx2 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u1` | 329 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u1` | 333 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 882 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u1` | 862 |
| avx2 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u1` | 856 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u1` | 884 |
| avx2 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u1` | 1727 |
| avx2 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 1824 |
| avx2 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u1` | 1719 |
| avx2 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u1` | 1721 |
| avx2 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u1` | 34 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u1` | 33 |
| avx2 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u1` | 33 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u1` | 33 |
| avx2 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u1` | 66 |
| avx2 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u1` | 65 |
| avx2 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u1` | 66 |
| avx2 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u1` | 173 |
| avx2 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u1` | 128 |
| avx2 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u1` | 129 |
| avx2 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u1` | 128 |
| avx2 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u1` | 128 |
| avx2 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u1` | 257 |
| avx2 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u1` | 254 |
| avx2 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u1` | 259 |
| avx2 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u1` | 260 |
| avx2 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u1` | 742 |
| avx2 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u1` | 770 |
| avx2 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u1` | 750 |
| avx2 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 751 |
| avx2 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 1586 |
| avx2 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u1` | 1586 |
| avx2 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 2025 |
| avx2 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 1591 |