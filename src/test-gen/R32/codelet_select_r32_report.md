# Codelet Selection Report (R=32)

Tie threshold: **2.0%**
Total decisions: **36**
Distinct winners: **9**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1s_dit__avx2` | 24 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal` | 3 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r2` | 2 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r1` | 2 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r2` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1` | 1 |
| `ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf4r1` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf8r1` | 1 |
| `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r1` | 1 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal` | 1115 |
| 72 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal` | 1094 |
| 128 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf4r1` | 1743 |
| 128 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r2` | 2686 |
| 136 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r1` | 2364 |
| 192 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r1` | 2689 |
| 256 | 256 | `ct_t1s_dit__avx2` | 8094 |
| 264 | 256 | `ct_t1s_dit__avx2` | 4954 |
| 320 | 256 | `ct_t1s_dit__avx2` | 5380 |
| 512 | 512 | `ct_t1s_dit__avx2` | 17848 |
| 520 | 512 | `ct_t1s_dit__avx2` | 10984 |
| 576 | 512 | `ct_t1s_dit__avx2` | 9961 |
| 1024 | 1024 | `ct_t1s_dit__avx2` | 36683 |
| 1032 | 1024 | `ct_t1s_dit__avx2` | 20700 |
| 1088 | 1024 | `ct_t1s_dit__avx2` | 19850 |
| 2048 | 2048 | `ct_t1s_dit__avx2` | 76330 |
| 2056 | 2048 | `ct_t1s_dit__avx2` | 41385 |
| 2112 | 2048 | `ct_t1s_dit__avx2` | 42845 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r2` | 1100 |
| 72 | 64 | `ct_t1_buf_dit__avx2__tile128__draintemporal` | 1095 |
| 128 | 64 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1` | 1748 |
| 128 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf8r1` | 2806 |
| 136 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r1` | 2379 |
| 192 | 128 | `ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r2` | 2663 |
| 256 | 256 | `ct_t1s_dit__avx2` | 8150 |
| 264 | 256 | `ct_t1s_dit__avx2` | 4890 |
| 320 | 256 | `ct_t1s_dit__avx2` | 5291 |
| 512 | 512 | `ct_t1s_dit__avx2` | 18074 |
| 520 | 512 | `ct_t1s_dit__avx2` | 9508 |
| 576 | 512 | `ct_t1s_dit__avx2` | 9997 |
| 1024 | 1024 | `ct_t1s_dit__avx2` | 36927 |
| 1032 | 1024 | `ct_t1s_dit__avx2` | 20690 |
| 1088 | 1024 | `ct_t1s_dit__avx2` | 19847 |
| 2048 | 2048 | `ct_t1s_dit__avx2` | 77185 |
| 2056 | 2048 | `ct_t1s_dit__avx2` | 41920 |
| 2112 | 2048 | `ct_t1s_dit__avx2` | 43090 |
