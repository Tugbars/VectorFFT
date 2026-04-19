# VectorFFT R=64 tuning report

Total measurements: **120**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 2837 | 2232 | 1907 | — | t1s |
| avx2 | 64 | 72 | 2402 | 2077 | 1698 | — | t1s |
| avx2 | 64 | 512 | 4875 | 5192 | 4805 | — | t1s |
| avx2 | 128 | 128 | 10455 | 9891 | 8800 | — | t1s |
| avx2 | 128 | 136 | 6606 | 4981 | 3811 | — | t1s |
| avx2 | 128 | 1024 | 10397 | 9955 | 10099 | — | log3 |
| avx2 | 256 | 256 | 19369 | 17923 | — | — | log3 |
| avx2 | 256 | 264 | 14495 | 8716 | — | — | log3 |
| avx2 | 256 | 2048 | 23813 | 24826 | — | — | flat |
| avx2 | 512 | 512 | 41422 | 43053 | — | — | flat |
| avx2 | 512 | 520 | 20476 | 19918 | — | — | log3 |
| avx2 | 512 | 4096 | 55912 | 85438 | — | — | flat |
| avx2 | 1024 | 1024 | 88688 | 78509 | — | — | log3 |
| avx2 | 1024 | 1032 | 47812 | 39218 | — | — | log3 |
| avx2 | 1024 | 8192 | 168981 | 178286 | — | — | flat |
| avx2 | 2048 | 2048 | 233466 | 188091 | — | — | log3 |
| avx2 | 2048 | 2056 | 148902 | 81211 | — | — | log3 |
| avx2 | 2048 | 16384 | 350512 | 380081 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 3876 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 3987 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 5687 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 10335 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 9768 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12251 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 22544 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 14399 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 27160 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 52135 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 26826 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 64628 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 97143 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 69790 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 168981 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 284291 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 201842 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 350512 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 2837 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 2402 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 4875 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 10455 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 6606 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 10397 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 19369 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 14495 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 23813 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 41422 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 20476 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 55912 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 88688 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 47812 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 175962 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 233466 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 148902 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 388584 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 2232 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2077 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 5192 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9891 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 4981 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 9955 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 17923 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 8716 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 24826 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 43053 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 19918 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 85438 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 78509 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 39218 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 178286 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 188091 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 81211 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 380081 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1907 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1698 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 4805 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 8800 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3811 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 10099 |