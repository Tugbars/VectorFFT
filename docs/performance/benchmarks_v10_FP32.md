# VectorFFT v1.0 — FP32 results, **AVX-512 hosts**

Single precision, 1D complex-to-complex, powers of two from 64 to 4M points, interleaved, natural order, out of
place, one thread. Speedup = the other library's time / VectorFFT's time: above 1 means VectorFFT is faster. Each
pass takes the worse of the two engine orders. Two hosts, each in its own section, and **not comparable** with
each other:

- [Emerald Rapids](#emerald-rapids): against FFTW and MKL.
- [Granite Rapids (6980P)](#granite-rapids-6980p): against MKL.

## Emerald Rapids

Each figure is the median of 3 passes.

Comparators:

- **FFTW 3.3.10**, built from the fftw.org source with the same configuration as the repository's bundled DLL
  (`fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512`), planned with `FFTW_MEASURE`.
- **Intel oneMKL 2026.0.2** (the pip build), sequential.

Plans: VectorFFT's planner ran once on this machine and saved all 17 FP32 plans in its wisdom; every timed pass
read them back, with nothing planned during the timing.

Neither host is the i9-14900KF of [v1_0_results.md](v1_0_results.md) or the Zen 4 laptop of
[v1_0_Zen4_results.md](v1_0_Zen4_results.md): the numbers are **not comparable** across these pages.

### The machine

| Property | Value |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz. Family 6, model 207 (0xCF), stepping 2, which is Emerald Rapids. The 260 MiB L3 points to a Xeon Platinum 8581C (Google Cloud C4 class). |
| vCPUs | 2 (1 socket, 2 cores, 1 thread per core as the guest sees it) |
| Hypervisor | KVM, full virtualization |
| L1d / L1i | 48 KiB / 32 KiB per core |
| L2 | 2 MiB per core |
| L3 | 260 MiB shared (the host's whole L3, not a share reserved for this machine) |
| NUMA | 1 node |
| RAM | 7.8 GiB, no swap |
| Address sizes | 52-bit physical, 57-bit virtual (5-level paging) |

### 1D C2C, single precision

| N | VectorFFT | FFTW | vs FFTW | MKL | vs MKL |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.02 µs | 0.04 µs | **1.62×** | 0.03 µs | **1.26×** |
| 128 | 0.05 µs | 0.07 µs | **1.36×** | 0.05 µs | 1.02× |
| 256 | 0.13 µs | 0.12 µs | 1.10× | 0.14 µs | **1.38×** |
| 512 | 0.23 µs | 0.27 µs | 1.15× | 0.27 µs | 1.09× |
| 1024 | 0.56 µs | 0.63 µs | 1.14× | 0.63 µs | 1.07× |
| 2048 | 1.71 µs | 1.51 µs | 0.89× | 1.50 µs | 0.90× |
| 4096 | 3.70 µs | 3.45 µs | 0.95× | 3.43 µs | 0.93× |
| 8192 | 8.84 µs | 10.6 µs | 1.20× | 9.34 µs | 1.06× |
| 16K | 18.2 µs | 26.1 µs | **1.44×** | 18.8 µs | 1.05× |
| 32K | 40.8 µs | 56.2 µs | **1.34×** | 44.9 µs | 1.10× |
| 64K | 99.5 µs | 126 µs | 1.26× | 112 µs | 1.09× |
| 128K | 300 µs | 365 µs | 1.22× | 462 µs | **1.55×** |
| 256K | 744 µs | 827 µs | 1.11× | 1.35 ms | **1.65×** |
| 512K | 1.99 ms | 4.12 ms | **2.11×** | 2.97 ms | **1.48×** |
| 1M | 3.08 ms | 9.25 ms | **2.93×** | 6.85 ms | **2.12×** |
| 2M | 8.05 ms | 24.5 ms | **3.11×** | 15.5 ms | **2.10×** |
| 4M | 16.4 ms | 70.9 ms | **4.34×** | 40.0 ms | **2.15×** |
| **Geometric mean** | | | **1.49×** | | **1.30×** |

VectorFFT is faster on 15 of 17 lengths against each library. The two it loses against both are N = 2048 and
4096.

### Reading the numbers

- **Large lengths:** from 512K points up VectorFFT leads by 2.1× to 4.3× against FFTW and 1.5× to 2.2× against
  MKL. From 1M points the planner uses the four-step route (`fs`).
- **Noise:** this is a shared virtual machine with 2 vCPUs. Small lengths moved by up to ±30 % between passes
  (128 against MKL read 0.75× to 1.17×, 2048 read 0.60× to 0.92×), so a difference under about 10 % is a tie
  here. The losses at 2048 and 4096 appeared in every pass.
- **Accuracy:** the worst round-trip error was 2.7e-7, as expected for single precision.

## Granite Rapids (6980P)

Comparator: **Intel oneMKL 2026.0.1** (conda-forge), sequential, through `mkl_rt`. Built with GCC 16.2 on Linux.

Plans: the library recognised this CPU as the 6980P and served from its wisdom folder, `src/wisdom/6980P/`. Its
planner had saved all 17 FP32 plans there in an earlier run; this run read them back and planned nothing. One run.

### The machine

| Property | Value |
|---|---|
| CPU | Intel Xeon 6980P (Granite Rapids) |
| vCPUs | 2, a shared KVM slice; no turbo visible to the guest |

### 1D C2C, single precision

| N | Speedup vs MKL |
|---:|---:|
| 64 | 1.20× |
| 128 | 1.09× |
| 256 | **1.33×** |
| 512 | 1.13× |
| 1024 | 0.61× |
| 2048 | 0.91× |
| 4096 | 0.92× |
| 8192 | 1.13× |
| 16K | 0.98× |
| 32K | 1.01× |
| 64K | 1.24× |
| 128K | 1.23× |
| 256K | **1.49×** |
| 512K | **1.51×** |
| 1M | **1.69×** |
| 2M | **1.62×** |
| 4M | **1.63×** |
| **Geometric mean** | **1.18×** |

VectorFFT is faster on 13 of 17 lengths.

### Reading the numbers

- **Large lengths:** from 256K points up VectorFFT leads by 1.5× to 1.7×.
- **2048 and 4096** lose in every run on this machine (about 0.9×), as they do on the Emerald Rapids host.
- **Noise:** the control cell drifted to 0.90×–0.91× during the run, so a difference under about 10 % is a tie.
  The 1024 reading (0.61×) is one engine order where VectorFFT ran 761 ns against 437 ns in the other order on
  the same plan; in the run before this one, 1024 read 1.13×. Likewise 16K read 0.66× in that earlier run and
  0.98× here.
