# The re-raced rows of src/wisdom/14900KF, before and after

The eleven in-place natural four-step rows (recalibrated one by one with a clean f6e4a970 build), and the five
scrambled in-place rows the same recalibrations re-banked: the planner races both order classes of a cell in one
race. An 8-thread row's banked time is its one-thread row's.

| precision | cell | plan before | plan after | banked ms before | after |
| --- | --- | --- | --- | ---: | ---: |
| FP32 | N = 2097152, natural, in place | 512 x 4096 chain 8.8.8 | 2048 x 1024 chain 8.8.4.8 | 5.803 | 3.736 |
| FP32 | N = 4194304, natural, in place | 1024 x 4096 chain 16.8.8 | 4096 x 1024 chain 8.8.8.8 | 12.754 | 8.622 |
| FP32 | N = 8388608, natural, in place | 2048 x 4096 chain 4.8.8.8 | 4096 x 2048 chain 16.8.8.4 | 27.278 | 19.828 |
| FP32 | N = 16777216, natural, in place | 4096 x 4096 chain 8.8.8.8 | 4096 x 4096 chain 8.8.8.8 | 54.893 | 42.394 |
| FP64 | N = 524288, natural, in place | 512 x 1024 chain (no child on the row) | 1024 x 512 chain 8.4.4.8 | 1.931 | 1.496 |
| FP64 | N = 1048576, natural, in place | 512 x 2048 chain (no child on the row) | 4096 x 256 chain 8.8.8.8 | 4.207 | 3.413 |
| FP64 | N = 2097152, natural, in place | 2048 x 1024 chain (no child on the row) | 4096 x 512 chain 8.8.8.8 | 9.252 | 8.149 |
| FP64 | N = 4194304, natural, in place | 1024 x 4096 chain (no child on the row) | 4096 x 1024 chain 8.8.8.8 | 20.776 | 18.935 |
| FP64 | N = 524288, natural, in place, 8 threads | 512 x 1024 chain 8.8.4.8, threaded split 2048 | 1024 x 512 chain 8.4.8, threaded split 256 | 1.931 | 1.496 |
| FP64 | N = 1000000, natural, in place | 2000 x 500 chain 25.16.5 | 2000 x 500 chain 16.5.5.5 | 4.505 | 3.142 |
| FP64 | N = 1000000, natural, in place, 8 threads | 2000 x 500 chain 25.5.5, threaded split 625 | 2000 x 500 chain 25.5.5, threaded split 625 | 4.505 | 3.142 |
| FP64 | N = 524288, scrambled, in place | 1024 x 512 chain (no child on the row) | 512 x 1024 chain 8.8.8 | 1.046 | 1.005 |
| FP64 | N = 1048576, scrambled, in place | 512 x 2048 chain (no child on the row) | 4096 x 256 chain 8.8.8.8 | 2.567 | 2.104 |
| FP64 | N = 2097152, scrambled, in place | 2048 x 1024 chain (no child on the row) | 4096 x 512 chain 8.8.8.8 | 5.877 | 5.241 |
| FP64 | N = 4194304, scrambled, in place | 1024 x 4096 chain (no child on the row) | 4096 x 1024 chain 8.8.8.8 | 14.088 | 12.647 |
| FP64 | N = 1000000, scrambled, in place | 2000 x 500 chain 25.16.5 | 2000 x 500 chain 16.5.5.5 | 2.595 | 2.221 |

16 rows changed; no row was added or removed.
