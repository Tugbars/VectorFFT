# VectorFFT R=3 tuning report

Total measurements: **192**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 24 | 24 | 10 | 12 | 9 | — | t1s |
| avx2 | 24 | 32 | 10 | 12 | 9 | — | t1s |
| avx2 | 24 | 72 | 9 | 11 | 9 | — | t1s |
| avx2 | 24 | 192 | 11 | 11 | 9 | — | t1s |
| avx2 | 48 | 48 | 18 | 21 | 18 | — | flat |
| avx2 | 48 | 56 | 18 | 23 | 17 | — | t1s |
| avx2 | 48 | 144 | 17 | 21 | 17 | — | t1s |
| avx2 | 48 | 384 | 18 | 20 | 17 | — | t1s |
| avx2 | 96 | 96 | 33 | 41 | 35 | — | flat |
| avx2 | 96 | 104 | 33 | 40 | 34 | — | flat |
| avx2 | 96 | 288 | 34 | 40 | 34 | — | t1s |
| avx2 | 96 | 768 | 32 | 41 | 34 | — | flat |
| avx2 | 192 | 192 | 65 | 81 | 65 | — | t1s |
| avx2 | 192 | 200 | 65 | 80 | 65 | — | flat |
| avx2 | 192 | 576 | 64 | 81 | 67 | — | flat |
| avx2 | 192 | 1536 | 64 | 81 | 68 | — | flat |
| avx2 | 384 | 384 | 142 | 163 | 155 | — | flat |
| avx2 | 384 | 392 | 125 | 160 | 132 | — | flat |
| avx2 | 384 | 1152 | 126 | 161 | 139 | — | flat |
| avx2 | 384 | 3072 | 126 | 160 | 155 | — | flat |
| avx2 | 768 | 768 | 387 | 329 | 280 | — | t1s |
| avx2 | 768 | 776 | 427 | 334 | 293 | — | t1s |
| avx2 | 768 | 2304 | 370 | 345 | 282 | — | t1s |
| avx2 | 768 | 6144 | 321 | 342 | 274 | — | t1s |
| avx2 | 1536 | 1536 | 878 | 867 | 797 | — | t1s |
| avx2 | 1536 | 1544 | 877 | 873 | 854 | — | t1s |
| avx2 | 1536 | 4608 | 882 | 886 | 809 | — | t1s |
| avx2 | 1536 | 12288 | 883 | 883 | 814 | — | t1s |
| avx2 | 3072 | 3072 | 1758 | 1709 | 1702 | — | t1s |
| avx2 | 3072 | 3080 | 1755 | 2123 | 1724 | — | t1s |
| avx2 | 3072 | 9216 | 1761 | 1760 | 1798 | — | log3 |
| avx2 | 3072 | 24576 | 1768 | 1751 | 1592 | — | t1s |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 10 |
| avx2 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 10 |
| avx2 | `t1_dit` | 24 | 72 | `ct_t1_dit_u1` | 9 |
| avx2 | `t1_dit` | 24 | 192 | `ct_t1_dit_u1` | 11 |
| avx2 | `t1_dit` | 48 | 48 | `ct_t1_dit_u1` | 18 |
| avx2 | `t1_dit` | 48 | 56 | `ct_t1_dit_u1` | 18 |
| avx2 | `t1_dit` | 48 | 144 | `ct_t1_dit_u1` | 17 |
| avx2 | `t1_dit` | 48 | 384 | `ct_t1_dit_u1` | 18 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit_u1` | 33 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit_u1` | 33 |
| avx2 | `t1_dit` | 96 | 288 | `ct_t1_dit_u1` | 34 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit_u1` | 32 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit_u1` | 65 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit_u1` | 65 |
| avx2 | `t1_dit` | 192 | 576 | `ct_t1_dit_u1` | 64 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u1` | 64 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit_u1` | 142 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit_u1` | 125 |
| avx2 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u1` | 126 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u1` | 126 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit_u1` | 387 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit_u1` | 427 |
| avx2 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u1` | 370 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u1` | 321 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u1` | 878 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u1` | 877 |
| avx2 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u1` | 882 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u1` | 883 |
| avx2 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u1` | 1758 |
| avx2 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u1` | 1755 |
| avx2 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u1` | 1761 |
| avx2 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u1` | 1768 |
| avx2 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 12 |
| avx2 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 12 |
| avx2 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u1` | 11 |
| avx2 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u1` | 11 |
| avx2 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u1` | 21 |
| avx2 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u1` | 23 |
| avx2 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u1` | 21 |
| avx2 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u1` | 20 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u1` | 41 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u1` | 40 |
| avx2 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u1` | 40 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u1` | 41 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u1` | 81 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u1` | 80 |
| avx2 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u1` | 81 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u1` | 81 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u1` | 163 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u1` | 160 |
| avx2 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u1` | 161 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u1` | 160 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u1` | 329 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u1` | 334 |
| avx2 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u1` | 345 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u1` | 342 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 867 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u1` | 873 |
| avx2 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u1` | 886 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u1` | 883 |
| avx2 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u1` | 1709 |
| avx2 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 2123 |
| avx2 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u1` | 1760 |
| avx2 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u1` | 1751 |
| avx2 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 9 |
| avx2 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u1` | 18 |
| avx2 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u1` | 17 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u1` | 35 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u1` | 34 |
| avx2 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u1` | 34 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u1` | 34 |
| avx2 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u1` | 65 |
| avx2 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u1` | 65 |
| avx2 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u1` | 67 |
| avx2 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u1` | 68 |
| avx2 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u1` | 155 |
| avx2 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u1` | 132 |
| avx2 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u1` | 139 |
| avx2 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u1` | 155 |
| avx2 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u1` | 280 |
| avx2 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u1` | 293 |
| avx2 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u1` | 282 |
| avx2 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u1` | 274 |
| avx2 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u1` | 797 |
| avx2 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u1` | 854 |
| avx2 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u1` | 809 |
| avx2 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 814 |
| avx2 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 1702 |
| avx2 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u1` | 1724 |
| avx2 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 1798 |
| avx2 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 1592 |