# Codelet Selection Report (R=32)

Tie threshold: **2.0%**
Total decisions: **72**
Distinct winners: **28**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1_dit__avx2__tpf16r2` | 11 |
| `ct_t1_dit_log3__avx512` | 7 |
| `ct_t1_dit__avx512__tpf32r1` | 5 |
| `ct_t1_dit_log3__avx2` | 5 |
| `ct_t1_dit_log3__avx512__tpf4r1` | 3 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1` | 3 |
| `ct_t1_dit__avx512` | 3 |
| `ct_t1_dit_log3__avx512__tpf32r1` | 3 |
| `ct_t1_dit_log3__avx2__tpf8r1` | 3 |
| `ct_t1_dit_log3__avx2__tpf32r1` | 3 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r2` | 2 |
| `ct_t1_dit__avx512__tpf4r1` | 2 |
| `ct_t1_ladder_dit__avx512` | 2 |
| `ct_t1_dit__avx2` | 2 |
| `ct_t1_dit_log3__avx512__tpf16r2` | 2 |
| `ct_t1_dit_log3__avx512__tpf8r1` | 2 |
| `ct_t1_dit_log3__avx2__tpf16r1` | 2 |
| `ct_t1_dit_log3__avx512__tpf32r2` | 2 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r2` | 1 |
| `ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf4r1` | 1 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf8r2` | 1 |
| `ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf8r2` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw` | 1 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r2` | 1 |
| `ct_t1_dit__avx512__tpf8r2` | 1 |
| `ct_t1_buf_dit__avx512__tile256__draintemporal__tpf4r1` | 1 |
| `ct_t1_dit_log3__avx2__tpf8r2` | 1 |
| `ct_t1_dit_log3__avx512__tpf16r1` | 1 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r2` | 1359 |
| 72 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf8r2` | 1403 |
| 128 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r2` | 1555 |
| 128 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1` | 3283 |
| 136 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1` | 3161 |
| 192 | 128 | `ct_t1_dit__avx2__tpf16r2` | 3083 |
| 256 | 256 | `ct_t1_dit__avx2__tpf16r2` | 10406 |
| 264 | 256 | `ct_t1_dit__avx2__tpf16r2` | 6614 |
| 320 | 256 | `ct_t1_dit__avx2__tpf16r2` | 6785 |
| 512 | 512 | `ct_t1_dit__avx2__tpf16r2` | 23463 |
| 520 | 512 | `ct_t1_dit__avx2` | 14417 |
| 576 | 512 | `ct_t1_dit_log3__avx2` | 13338 |
| 1024 | 1024 | `ct_t1_dit_log3__avx2__tpf8r1` | 44516 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2__tpf8r2` | 32058 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 27376 |
| 2048 | 2048 | `ct_t1_dit_log3__avx2__tpf32r1` | 99538 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2__tpf32r1` | 62522 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2__tpf32r1` | 60563 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dit__avx2__tpf16r2` | 1437 |
| 72 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r2` | 1347 |
| 128 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw` | 1344 |
| 128 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r2` | 3434 |
| 136 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1` | 2911 |
| 192 | 128 | `ct_t1_dit__avx2__tpf16r2` | 3399 |
| 256 | 256 | `ct_t1_dit__avx2__tpf16r2` | 10399 |
| 264 | 256 | `ct_t1_dit__avx2__tpf16r2` | 6589 |
| 320 | 256 | `ct_t1_dit__avx2__tpf16r2` | 6384 |
| 512 | 512 | `ct_t1_dit__avx2__tpf16r2` | 23530 |
| 520 | 512 | `ct_t1_dit__avx2` | 14111 |
| 576 | 512 | `ct_t1_dit_log3__avx2` | 13636 |
| 1024 | 1024 | `ct_t1_dit_log3__avx2__tpf8r1` | 50699 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2__tpf8r1` | 29453 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 27061 |
| 2048 | 2048 | `ct_t1_dit_log3__avx2` | 89382 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2__tpf16r1` | 61085 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2__tpf16r1` | 55846 |

### avx512 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf4r1` | 853 |
| 72 | 64 | `ct_t1_dit_log3__avx512__tpf4r1` | 871 |
| 128 | 64 | `ct_t1_dit__avx512__tpf8r2` | 1220 |
| 128 | 128 | `ct_t1_dit_log3__avx512__tpf4r1` | 2298 |
| 136 | 128 | `ct_t1_buf_dit__avx512__tile256__draintemporal__tpf4r1` | 2171 |
| 192 | 128 | `ct_t1_dit__avx512__tpf32r1` | 1689 |
| 256 | 256 | `ct_t1_ladder_dit__avx512` | 5639 |
| 264 | 256 | `ct_t1_dit__avx512__tpf32r1` | 4413 |
| 320 | 256 | `ct_t1_dit_log3__avx512__tpf32r1` | 4104 |
| 512 | 512 | `ct_t1_dit_log3__avx512__tpf32r1` | 12629 |
| 520 | 512 | `ct_t1_dit_log3__avx512` | 10192 |
| 576 | 512 | `ct_t1_dit__avx512__tpf32r1` | 8590 |
| 1024 | 1024 | `ct_t1_dit__avx512` | 27912 |
| 1032 | 1024 | `ct_t1_dit_log3__avx512` | 16794 |
| 1088 | 1024 | `ct_t1_dit_log3__avx512` | 13583 |
| 2048 | 2048 | `ct_t1_dit_log3__avx512` | 46744 |
| 2056 | 2048 | `ct_t1_dit_log3__avx512` | 29851 |
| 2112 | 2048 | `ct_t1_dit_log3__avx512__tpf32r2` | 36272 |

### avx512 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dit_log3__avx512__tpf4r1` | 759 |
| 72 | 64 | `ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf8r2` | 877 |
| 128 | 64 | `ct_t1_dit_log3__avx512` | 1175 |
| 128 | 128 | `ct_t1_dit__avx512__tpf4r1` | 2396 |
| 136 | 128 | `ct_t1_dit__avx512__tpf4r1` | 2057 |
| 192 | 128 | `ct_t1_dit__avx512__tpf32r1` | 2125 |
| 256 | 256 | `ct_t1_ladder_dit__avx512` | 5576 |
| 264 | 256 | `ct_t1_dit__avx512__tpf32r1` | 4240 |
| 320 | 256 | `ct_t1_dit__avx512` | 4864 |
| 512 | 512 | `ct_t1_dit_log3__avx512__tpf32r1` | 11127 |
| 520 | 512 | `ct_t1_dit__avx512` | 10359 |
| 576 | 512 | `ct_t1_dit_log3__avx512__tpf16r2` | 7633 |
| 1024 | 1024 | `ct_t1_dit_log3__avx512__tpf16r2` | 25781 |
| 1032 | 1024 | `ct_t1_dit_log3__avx512__tpf8r1` | 19253 |
| 1088 | 1024 | `ct_t1_dit_log3__avx512__tpf16r1` | 18886 |
| 2048 | 2048 | `ct_t1_dit_log3__avx512__tpf8r1` | 47824 |
| 2056 | 2048 | `ct_t1_dit_log3__avx512` | 33442 |
| 2112 | 2048 | `ct_t1_dit_log3__avx512__tpf32r2` | 33359 |
