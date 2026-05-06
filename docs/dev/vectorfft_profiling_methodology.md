# VectorFFT Profiling Methodology

Reference doc for building the microarchitectural comparison section of the VectorFFT white paper. Covers which counters to collect, how to collect them without fighting noise, and how to structure the comparison so it survives reviewer scrutiny.

---

## The goal

For each kernel of interest (yours and FFTW/MKL at matched N, K), produce:

1. **Per-codelet counter tables** — one row per counter, one column per library, with interpretation paragraphs pointing at the specific rows that explain the headline gap.
2. **Spill-count tables** — direct disassembly counts, one row per (codelet, library).
3. **A roofline plot** — one point per (library, radix, N), showing your points closer to the compute ceiling than FFTW's.

Two separate experiments: **isolated codelet microbenchmarks** for microarch claims, **full-FFT benchmarks** for wallclock headlines. Don't mix them.

---

## Counter groups

Modern Intel cores have 4–8 general-purpose PMU counters. You cannot measure all of these at once — you must run the benchmark multiple times, each run measuring one group. Always include `cycles` and `instructions` in every group (they use fixed-function counters, free slots) so you can sanity-check that each run was identical.

### Group 1 — Throughput ("am I fast?")

| Counter | Purpose |
|---|---|
| `cycles` | Normalization baseline |
| `instructions` | IPC denominator |
| `fp_arith_inst_retired.512b_packed_double` | AVX-512 FMA count |
| `fp_arith_inst_retired.256b_packed_double` | AVX2 FMA count (shows SIMD width mix) |

Derived: **CPE** = cycles / (N·K) · (1 / num_iterations). **IPC** = instructions / cycles. **FLOPs/cycle** = (8·512b_count + 4·256b_count) / cycles, times 2 if counting FMAs as 2 FLOPs.

### Group 2 — Front-end / decode

| Counter | Purpose |
|---|---|
| `cycles`, `instructions` | Normalization |
| `idq_uops_not_delivered.core` | Front-end starvation |
| `idq.dsb_uops` | Uops from decoded-uop cache |
| `idq.mite_uops` | Uops from legacy decoder |

**DSB coverage** = `dsb_uops / (dsb_uops + mite_uops)`. Should be >80% for a well-laid-out codelet. If FFTW drops to 40–60%, that's a paragraph.

### Group 3 — Back-end stalls

| Counter | Purpose |
|---|---|
| `cycles`, `instructions` | Normalization |
| `cycle_activity.stalls_total` | Total back-end stall cycles |
| `cycle_activity.stalls_mem_any` | Memory-related stalls |
| `cycle_activity.stalls_l1d_miss` | L1D miss stalls specifically |

Stall percentages = stall_count / cycles. FFTW's large codelets should show significantly more mem_any stalls if the spill-into-L1 story is real.

### Group 4 — Memory hierarchy

| Counter | Purpose |
|---|---|
| `cycles`, `instructions` | Normalization |
| `mem_load_retired.l1_hit` | L1 hits |
| `mem_load_retired.l2_hit` | L2 hits |
| `mem_load_retired.l3_hit` | L3 hits |
| `l1d.replacement` | L1 eviction rate — where the twiddle-thrashing story lives |

**Arithmetic intensity** (derived across groups) = FLOPs / bytes_loaded, where bytes_loaded ≈ (l1_hit + l2_hit + l3_hit) · 64. Exported to the roofline plot x-axis.

### Group 5 — Port pressure (the FMA saturation story)

| Counter | Purpose |
|---|---|
| `cycles` | Normalization |
| `uops_dispatched.port_0` | Port 0 utilization (AVX-512 FMA unit 1) |
| `uops_dispatched.port_1` | Port 1 |
| `uops_dispatched.port_5` | Port 5 (AVX-512 FMA unit 2 when active) |
| `uops_dispatched.port_6` | Port 6 |

**Port utilization** = port_N / cycles. On Skylake-X through Emerald Rapids, AVX-512 FMA units live on ports 0 and 5. Target: both ports near 1.0 uops/cycle = fully saturated. FFTW typically sits at 0.6–0.7; you should be at 0.9+.

### CPU-specific event names

These names are for Skylake-X through Sapphire Rapids. **Always verify for your exact CPU:**

```bash
perf list | grep -i fp_arith
perf list | grep -i cycle_activity
perf list | grep -i mem_load_retired
perf list | grep -i uops_dispatched
```

Know your CPU exactly: `lscpu` gives you family/model/stepping. Look up the matching Intel PMU events reference document.

---

## Machine lockdown (do once, benefit forever)

Before collecting any numbers for the paper:

```bash
# Lock CPU frequency to base clock
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -d 3.5GHz -u 3.5GHz   # adjust to your base clock

# Disable turbo
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Disable SMT (optional, reduces noise)
echo off | sudo tee /sys/devices/system/cpu/smt/control

# Disable ASLR
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Allow perf without root
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Pin every run to the same physical core:

```bash
taskset -c 2 perf stat -e ... ./bench
```

**Verification:** run the same benchmark 10 times. Cycle count variance must be under 1%. If not, hunt down the cause (background processes, thermal throttling, wrong governor) before trusting any numbers.

---

## Tool division of labor

### VTune — use once per kernel for diagnosis

```bash
vtune -collect uarch-exploration -result-dir vtune_result ./your_benchmark
vtune -report summary -result-dir vtune_result
```

VTune's strength is the **top-down breakdown**: frontend-bound / backend-bound / bad-speculation / retiring. Run this once per kernel to diagnose *where* the bottleneck lives. Then switch to perf for the quantitative tables.

**Gotchas:**
- Don't trust VTune's top-down on hybrid cores (Alder/Raptor/Meteor Lake). Pin to a P-core explicitly.
- Sampling skew can misattribute counts by 1–2 instructions on tight loops. Cross-check aggregate numbers against `perf stat` on the same binary.

### perf stat — use for all quantitative tables

Faster, scriptable, easy to automate. One command per counter group:

```bash
# Throughput group
taskset -c 2 perf stat -e cycles,instructions,\
fp_arith_inst_retired.512b_packed_double,\
fp_arith_inst_retired.256b_packed_double \
./microbench 2>&1 | tee throughput.log

# Front-end group
taskset -c 2 perf stat -e cycles,instructions,\
idq_uops_not_delivered.core,idq.dsb_uops,idq.mite_uops \
./microbench 2>&1 | tee frontend.log

# Back-end group
taskset -c 2 perf stat -e cycles,instructions,\
cycle_activity.stalls_total,cycle_activity.stalls_mem_any,\
cycle_activity.stalls_l1d_miss \
./microbench 2>&1 | tee backend.log

# Memory hierarchy group
taskset -c 2 perf stat -e cycles,instructions,\
mem_load_retired.l1_hit,mem_load_retired.l2_hit,\
mem_load_retired.l3_hit,l1d.replacement \
./microbench 2>&1 | tee memory.log

# Port pressure group
taskset -c 2 perf stat -e cycles,\
uops_dispatched.port_0,uops_dispatched.port_1,\
uops_dispatched.port_5,uops_dispatched.port_6 \
./microbench 2>&1 | tee ports.log
```

Run each group 3 times, take the median. Automate this in a shell script.

---

## Isolating a single codelet

Whole-program counters are useless for codelet-level claims — they aggregate across stages, harness, and init. Two techniques:

### Technique 1 — Tight-loop microbenchmark (preferred)

```c
#include "vectorfft.h"
// ... setup: allocate re, im, twiddle tables, etc.

int main() {
    // Warmup
    for (int i = 0; i < 100; i++)
        codelet_r20_fwd(re, im, W_re, W_im, stride, K);

    // Measured loop — dominates counter activity
    for (int i = 0; i < 1000000; i++)
        codelet_r20_fwd(re, im, W_re, W_im, stride, K);

    return 0;
}
```

99%+ of counter activity is inside the codelet. Divide by iteration count for per-call numbers. Build one such microbenchmark per codelet you want to report on. 20 lines each, tedious but clean.

### Technique 2 — Attach at runtime

```bash
perf stat --delay=1000 -- ./bench      # skip first 1s of startup
perf stat -p <pid>                      # attach to running process
```

Useful for real-FFT profiling without startup noise, but technique 1 is cleaner for codelet-level claims.

---

## Fair comparison checklist

- **Same compiler, same flags** for both libraries. Build FFTW from source with `-O3 -march=native --enable-avx512`.
- **Same buffer, alignment, core affinity, N, K.**
- **FFTW planning level:** use `FFTW_PATIENT` or `FFTW_EXHAUSTIVE`. "Beats FFTW MEASURE" is blog-post territory; "beats FFTW PATIENT" is what reviewers demand.
- **MKL threading:** `mkl_set_num_threads(1)` for single-thread microarch runs. Match threaded setup for headline numbers.
- **Which FFTW codelet ran:** use `fftw_print_plan` after planning to see exactly which codelet FFTW picked, then microbenchmark *that function* by name.
- **Cold vs steady-state:** report both, labeled. FFTW will look better on steady-state than cold because of icache footprint; show this honestly.
- **Prefetchers:** disable for the microarch analysis run, re-enable for the production wallclock number. Report both.

---

## Spill counting via disassembly

Counters can't directly count register spills. Disassembly can.

```bash
# Compile with debug + optimization
gcc -O3 -march=native -g -S -o codelet.s codelet.c

# Or disassemble existing binary
objdump -d --disassembler-options=intel libfftw3.so > fftw_disasm.txt
```

Within the hot function body, count:

- **Spill reloads:** loads from `[rsp + offset]` or `[rbp - offset]` where the address is the stack frame.
- **Spill stores:** stores to the same addresses.

Function entry/exit push/pop of callee-saved registers are **not** spills — ignore them. Real spills are interleaved with the arithmetic body.

### Semi-automated counting

```bash
# Approximate spill reload count for FFTW's t1_16 codelet
objdump -d libfftw3.so | awk '/<t1_16>:/,/^$/' | \
  grep -cE 'vmov[au]p[ds].*\(%rsp\)|\(%rbp\)'
```

This greps for 256b/512b moves against a stack-relative address. The count is approximately the spill traffic per call. Do it for each codelet in each library. Tabulate as "spill reloads per butterfly."

**Expected pattern:** your codelets zero or near-zero; FFTW's large codelets in double digits on AVX-512. If the numbers come out this way, it's devastating evidence and unimpeachable because it's a direct property of the emitted binary.

---

## Deriving arithmetic intensity from counters

For the roofline plot x-axis:

```
FLOPs_per_call = 2 × (8 × count(512b_packed_double) + 4 × count(256b_packed_double))
                 × (1 / iterations)
                 # factor 2: each FMA counts as 2 FLOPs

bytes_per_call = 64 × (count(l1_hit) + count(l2_hit) + count(l3_hit) + count(dram_hit))
                 × (1 / iterations)
                 # 64 bytes per cache line

arithmetic_intensity = FLOPs_per_call / bytes_per_call   # FLOPs/byte
GFLOPs_per_sec       = FLOPs_per_call / (cycles_per_call / clock_freq_Hz) / 1e9
```

Plot `GFLOPs_per_sec` (y) vs `arithmetic_intensity` (x) for each (library, radix, N) point, with the DRAM/L3/L2/L1 bandwidth rooflines and the peak FMA ceiling drawn in. Your points should cluster **right of and above** FFTW's.

---

## Results table template

One table per radix of interest (R = 8, 16, 20, 32). Columns: counter, VectorFFT, FFTW, MKL. Followed by a 2–3 sentence interpretation pointing at the specific row.

Example:

**R=20, K=128, N=2560, Sapphire Rapids, 1 core @ 3.5 GHz:**

| Metric | VectorFFT | FFTW PATIENT | MKL |
|---|---|---|---|
| CPE | 0.42 | 0.71 | 0.58 |
| IPC | 4.8 | 3.1 | 3.6 |
| Port 0 utilization | 0.94 | 0.64 | 0.78 |
| Port 5 utilization | 0.91 | 0.62 | 0.74 |
| STALLS_MEM_ANY (% cycles) | 1.8 | 18.4 | 9.2 |
| L1D replacements / call | 12 | 184 | 73 |
| DSB coverage | 0.92 | 0.54 | 0.81 |
| Spill reloads / butterfly | 0 | 6.3 | 2.1 |

> *VectorFFT achieves 0.42 CPE versus FFTW's 0.71. The gap is fully explained by back-end pressure: FFTW executes 6.3 spill reloads per butterfly (from disassembly) and shows STALLS_MEM_ANY at 18% of cycles, while VectorFFT executes zero spills and stalls on memory under 2% of cycles. Port 0/5 utilization reaches 0.94/0.91 in VectorFFT versus 0.64/0.62 in FFTW, consistent with spill-induced dispatch gaps.*

---

## Order of operations

1. **Lock down the machine.** One afternoon, done forever.
2. **Write one codelet microbenchmark.** Verify the loop body in disassembly.
3. **Get a baseline `perf stat cycles,instructions`.** Run 10 times, check variance <1%.
4. **Add counter groups one at a time.** Run each group 3 times, take medians. Script it.
5. **Do the same for FFTW's equivalent codelet.** Use `fftw_print_plan` to identify which codelet to isolate.
6. **Disassemble both, count spills manually.**
7. **Tabulate one table per radix.** 4–6 tables total.
8. **Roofline plot last** — counters already give you FLOPs/sec and bytes/sec.

Budget: ~2 weeks of focused work if methodical, 3 if the tooling surprises you once (it will).

---

## Critical advice

Start with **one codelet, one counter group, end-to-end** before scaling. Nothing wastes a week faster than collecting 50 data points and discovering a silly microbenchmark bug. The first clean result is the hardest; the rest are copy-paste.

When in doubt, trust `perf stat` over VTune for quantitative numbers, and trust disassembly over any counter for spill claims.
