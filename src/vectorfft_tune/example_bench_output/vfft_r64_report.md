# VectorFFT R=64 tuning report

Total measurements: **240**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 5431 | 5016 | 4632 | — | t1s |
| avx2 | 64 | 72 | 5484 | 4942 | 4242 | — | t1s |
| avx2 | 64 | 512 | 10975 | 11649 | 11287 | — | flat |
| avx2 | 128 | 128 | 18331 | 16726 | 15594 | — | t1s |
| avx2 | 128 | 136 | 11217 | 9593 | 9147 | — | t1s |
| avx2 | 128 | 1024 | 22340 | 23088 | 22652 | — | flat |
| avx2 | 256 | 256 | 40808 | 42191 | — | — | flat |
| avx2 | 256 | 264 | 25176 | 20559 | — | — | log3 |
| avx2 | 256 | 2048 | 44630 | 47040 | — | — | flat |
| avx2 | 512 | 512 | 98101 | 93405 | — | — | log3 |
| avx2 | 512 | 520 | 49612 | 42808 | — | — | log3 |
| avx2 | 512 | 4096 | 100863 | 94286 | — | — | log3 |
| avx2 | 1024 | 1024 | 202447 | 183148 | — | — | log3 |
| avx2 | 1024 | 1032 | 120003 | 84783 | — | — | log3 |
| avx2 | 1024 | 8192 | 199325 | 195421 | — | — | log3 |
| avx2 | 2048 | 2048 | 535425 | 390967 | — | — | log3 |
| avx2 | 2048 | 2056 | 327501 | 175419 | — | — | log3 |
| avx2 | 2048 | 16384 | 553515 | 385223 | — | — | log3 |
| avx512 | 64 | 64 | 3561 | 2983 | 3724 | — | log3 |
| avx512 | 64 | 72 | 3373 | 2681 | 2707 | — | log3 |
| avx512 | 64 | 512 | 6058 | 7823 | 5905 | — | t1s |
| avx512 | 128 | 128 | 9732 | 9105 | 8958 | — | t1s |
| avx512 | 128 | 136 | 6397 | 5576 | 5370 | — | t1s |
| avx512 | 128 | 1024 | 12167 | 12369 | 11879 | — | t1s |
| avx512 | 256 | 256 | 21643 | 21324 | — | — | log3 |
| avx512 | 256 | 264 | 14727 | 11170 | — | — | log3 |
| avx512 | 256 | 2048 | 24114 | 24560 | — | — | flat |
| avx512 | 512 | 512 | 51681 | 48838 | — | — | log3 |
| avx512 | 512 | 520 | 28340 | 23613 | — | — | log3 |
| avx512 | 512 | 4096 | 48687 | 48762 | — | — | flat |
| avx512 | 1024 | 1024 | 115609 | 97031 | — | — | log3 |
| avx512 | 1024 | 1032 | 78709 | 46783 | — | — | log3 |
| avx512 | 1024 | 8192 | 116014 | 100422 | — | — | log3 |
| avx512 | 2048 | 2048 | 300109 | 220162 | — | — | log3 |
| avx512 | 2048 | 2056 | 205335 | 140556 | — | — | log3 |
| avx512 | 2048 | 16384 | 294512 | 231495 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 5900 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 6108 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 10975 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 18331 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 12184 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 22340 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 40808 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 25176 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 44630 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 98101 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 54915 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 102891 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 202447 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 131186 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 199325 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 556228 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 327501 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 553515 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 5431 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 5484 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 11656 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 19117 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 11217 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 24266 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 45440 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 31158 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 48695 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 101423 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 49612 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 100863 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 230111 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 120003 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 234500 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 535425 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 343089 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 574954 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 5016 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 4942 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 11649 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 16726 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 9593 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 23088 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 42191 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 20559 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 47040 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 93405 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 42808 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 94286 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 183148 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 84783 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 195421 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 390967 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 175419 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 385223 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 4632 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 4242 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 11287 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 15594 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 9147 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 22652 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 4015 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 3515 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 6092 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 9732 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 7575 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12167 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 21643 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 14667 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 24114 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 51681 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 31096 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 48687 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 115609 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 78893 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 116014 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 297581 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 205335 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 296837 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 3561 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 3373 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 6058 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 10159 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 6397 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 12877 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 22434 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 14727 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 25442 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 52916 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 28340 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 52272 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 118985 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 78709 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 122857 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 300109 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 210198 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 294512 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2983 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2681 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 7823 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9105 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 5576 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 12369 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 21324 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 11170 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 24560 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 48838 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 23613 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 48762 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 97031 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 46783 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 100422 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 220162 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 140556 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 231495 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 3724 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 2707 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 5905 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 8958 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 5370 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 11879 |