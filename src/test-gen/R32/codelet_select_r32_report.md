# Codelet Selection Report (R=32)

Tie threshold: **2.0%**
Total decisions: **36**
Distinct winners: **22**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1_dit_log3__avx2__tpf32r1` | 4 |
| `ct_t1_dit_log3__avx2__tpf16r2` | 4 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal` | 3 |
| `ct_t1_dit_log3__avx2__tpf4r1` | 3 |
| `ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf4r1` | 2 |
| `ct_t1_buf_dit__avx2__tile32__draintemporal__tpf32r1` | 2 |
| `ct_t1_dit_log3__avx2__tpf32r2` | 2 |
| `ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf4r1` | 2 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r1` | 1 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r2` | 1 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf8r1` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf8r2` | 1 |
| `ct_t1_dit_log3__avx2__tpf16r1` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf16r2` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r1` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r2` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r1` | 1 |
| `ct_t1_buf_dit__avx2__tile64__draintemporal__tpf32r2` | 1 |
| `ct_t1_dit__avx2__tpf8r2` | 1 |
| `ct_t1_dit_log3__avx2__tpf8r1` | 1 |
| `ct_t1_buf_dit__avx2__tile32__draintemporal__tpf16r2` | 1 |
| `ct_t1_dit_log3__avx2__tpf8r2` | 1 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r2` | 1090 |
| 72 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf8r1` | 1022 |
| 128 | 64 | `ct_t1_dit_log3__avx2__tpf16r1` | 1997 |
| 128 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r1` | 2747 |
| 136 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal` | 2336 |
| 192 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal` | 2673 |
| 256 | 256 | `ct_t1_buf_dit__avx2__tile64__draintemporal__tpf32r2` | 8046 |
| 264 | 256 | `ct_t1_dit_log3__avx2__tpf4r1` | 5903 |
| 320 | 256 | `ct_t1_dit_log3__avx2__tpf4r1` | 5651 |
| 512 | 512 | `ct_t1_buf_dit__avx2__tile32__draintemporal__tpf32r1` | 19149 |
| 520 | 512 | `ct_t1_dit_log3__avx2__tpf8r1` | 12192 |
| 576 | 512 | `ct_t1_dit_log3__avx2__tpf4r1` | 11966 |
| 1024 | 1024 | `ct_t1_buf_dit__avx2__tile32__draintemporal__tpf16r2` | 35240 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2__tpf16r2` | 22553 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2__tpf8r2` | 21983 |
| 2048 | 2048 | `ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf4r1` | 79530 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2__tpf32r1` | 52740 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2__tpf32r2` | 49790 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r1` | 1061 |
| 72 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal` | 1103 |
| 128 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf8r2` | 2034 |
| 128 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf16r2` | 2742 |
| 136 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r2` | 2321 |
| 192 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r1` | 2675 |
| 256 | 256 | `ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf4r1` | 8494 |
| 264 | 256 | `ct_t1_dit_log3__avx2__tpf32r1` | 6375 |
| 320 | 256 | `ct_t1_dit__avx2__tpf8r2` | 6708 |
| 512 | 512 | `ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf4r1` | 18718 |
| 520 | 512 | `ct_t1_dit_log3__avx2__tpf32r1` | 11915 |
| 576 | 512 | `ct_t1_dit_log3__avx2__tpf16r2` | 12652 |
| 1024 | 1024 | `ct_t1_buf_dit__avx2__tile32__draintemporal__tpf32r1` | 34150 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2__tpf32r2` | 23237 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2__tpf32r1` | 23333 |
| 2048 | 2048 | `ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf4r1` | 80690 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2__tpf16r2` | 55925 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2__tpf16r2` | 54130 |
