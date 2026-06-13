# Intel MKL FFT Internals: Evidence from the Installed Binary

A record of what the installed MKL actually is, gathered by inspecting the
binary directly rather than relying on reputation. The trigger was a claim made
repeatedly in discussion, that MKL's FFT is hand-written assembly, which turned
out to be unsupported and then contradicted by the evidence below. This document
separates direct evidence (bytes in the binary) from inference (what those bytes
imply), and ends with the implications for VectorFFT.

Investigated against Intel oneMKL 2026, installed in the working container,
`libmkl_avx512.so.3` (80 MB) and siblings under `/usr/local/lib`.

## 1. Question

Two things were unknown:

1. Is MKL's FFT hand-written assembly, compiler-generated from intrinsics, or
   machine-generated (a code generator in the FFTW or SPIRAL family)?
2. For the power-of-two double-complex path that competes with VectorFFT
   (N = 1024), what does the actual kernel look like, and why is MKL hard to
   beat there?

MKL is closed-source and Intel documents none of this; every official source
only says "highly optimized." So the binary is the only ground truth available.

## 2. Method and environment

Tools available in the container: `nm`, `objdump`, `file`, `ar`. Not available:
`perf`, `vtune`, `llvm-mca`. The host is a single-vCPU virtualized Xeon, so
absolute timing is noisy (plus or minus 5 percent with occasional 2x
descheduling spikes); only interleaved back-to-back min-of-N rankings are
reliable. Binary inspection is unaffected by VM noise.

Steps: enumerate the distribution form (static archive versus shared object),
check whether the libraries are stripped, grep the symbol table for FFT and DFT
kernels to recover the architecture by naming scheme, then locate and disassemble
a representative `STEP_RADIX8` to judge code origin, vector width, instruction
mix, spilling, and scheduling.

## 3. Finding: packaging and runtime behavior

* This install ships only linked shared objects. No `libmkl*.a` static archive
  exists anywhere on the system. A `.so` is already linked, so the individual
  `.o` objects that went into it are fused and not recoverable with `ar x`.
  (The full oneAPI toolkit ships `.a` versions, which are archives of `.o` you
  could list and extract; the pip wheel used here is `.so` only.)
* MKL does not emit object files at runtime. The DFTI create/commit/compute flow
  selects from precompiled kernels already inside the `.so`; commit does not
  invoke a compiler. MKL's only runtime code generation is the GEMM JIT
  (`mkl_jit_*`), which writes machine code straight into executable memory, not
  to `.o`. The FFT path has no such JIT.
* The kernel libraries are not stripped. `libmkl_avx512.so.3` exposes 11,051
  dynamic symbols and 17,069 total (local plus global), so the kernel functions
  retain names and can be located and disassembled individually despite the
  absence of `.o`.

Implication: you cannot see per-codelet boundaries on disk the way you can in
FFTW's or VectorFFT's build trees, but the intact symbol table makes the
internal structure fully recoverable.

## 4. Finding: the architecture is a generated mixed-radix codelet library

Direct evidence from the symbol table of `libmkl_avx512.so.3`:

| keyword | matches | reading |
|---|---|---|
| `dft` | 2700 | DFT machinery throughout |
| `dfti` | 615 | the DFTI descriptor API surface |
| `fft` | 497 | FFT driver and helper routines |
| `radix` | 242 | radix-step primitives (also IPP radix-sort, see below) |
| `cfft` | 202 | complex FFT |
| `twiddle` / `codelet` | 6 / 6 | small but present, names not obfuscated |
| `butterfly`, `spiral`, `genfft` | 0 | not named in the binary |

The decisive evidence is the naming scheme:

* Fixed-size complex codelets `cDFTfwd_N` exist only for the non-power-of-two
  factors: N in {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}. The largest is 15.
  There is no `cDFTfwd_2/4/8/16`.
* Powers of two are handled by butterfly-step primitives `STEP_RADIX4` and
  `STEP_RADIX8`, of which there are several specialized variants (sizes 0x6b6,
  0x830, 0x82e, 0x751 bytes, etc.).
* Reorganization is factored into separate passes: `_row_1d_dft_twist` and
  `_col_1d_dft_twist` (the twiddle multiply) and `_row_1d_dft_twist_transpose`
  (the transpose).

So N = 1024 = 2^10 is not a single kernel. It is composed from radix-8 and
radix-4 steps with twiddle and transpose passes between stages. This is the
classic mixed-radix codelet decomposition, the same family as FFTW (whose
codelets come from the `genfft` generator) and SPIRAL. The per-size leaf
codelets plus radix-step primitives plus separate twist/transpose passes are the
signature of a generator, not of a hand-written monolith.

Note: many `radix` matches are `mkl_dft_avx512_ippsSortRadix...`, which is
radix-sort plumbing from IPP, unrelated to FFT butterflies. The FFT radix
primitives are specifically the `STEP_RADIX4/8` symbols.

## 5. Finding: STEP_RADIX8 disassembly

One representative double-precision variant was disassembled
(`STEP_RADIX8` at 0x4b38a20, 1718 bytes).

### 5.1 Reads as compiler output, not hand assembly (inference)

* Prologue is a textbook callee-save frame: `push r15; push r14; push r12;
  push rbx`, followed by ordinary index arithmetic (`mov`, `shl` by 4/5/6 for
  the times-16/32/64 byte strides of complex doubles, `lea`, `movslq`).
* The body is a real counted loop with backward conditional branches
  (`jne STEP_RADIX8+0x60`, `jg STEP_RADIX8+0x50`, and a second loop region near
  +0x380), not fully unrolled straight-line code.

Hand-written FFT codelets are typically fully unrolled with idiosyncratic frames.
A counted loop behind a standard frame is what a compiler emits from
well-structured source. This does not prove the source was intrinsics versus
generated C, and it does not identify the compiler, but it does positively
contradict "hand-written assembly." The sophistication is in the decomposition,
the source quality, and the compiler's scheduling, not in assembly authorship.

### 5.2 Vector width is 256-bit, by deliberate choice (direct evidence)

| register class | references |
|---|---|
| `%zmm` (512-bit) | 0 |
| `%ymm` (256-bit) | 600 |
| `%xmm` (128-bit) | 0 |

It uses `ymm0` through `ymm25`. Registers `ymm16` through `ymm31` exist only
under AVX-512 EVEX encoding, and the EVEX prefix is visible in the instruction
bytes (the `62 ...` prefix, with the length field encoding 256-bit). No `k1..k7`
mask registers appear in this kernel.

So this is genuinely an AVX-512 kernel by encoding and by register file, run
deliberately at 256-bit width. This is the well-known throughput-versus-frequency
tradeoff: take the 16 extra architectural registers and EVEX features, but keep
256-bit width to avoid the AVX-512 frequency license that downclocks the core
under sustained heavy 512-bit load.

### 5.3 FMA fusion is selective, not maximal (direct evidence)

Instruction mix over the 347 instructions:

| class | count |
|---|---|
| FMA (`vfmadd*`, `vfnmadd*`) | 28 |
| `vmulpd` (standalone) | 36 |
| `vaddpd` / `vsubpd` | 104 |
| `vmovupd` / `vmovapd` | 92 |
| `vperm` / `vshuf` / `vunpck` | 0 |

In the loop body, standalone `vmulpd` feeds `vfmadd231pd` and `vfnmadd231pd`:
the canonical complex multiply (one multiply, two FMAs) rather than fusing every
multiply-add. MKL keeps products standalone so they retire early instead of
lengthening the dependency chain. This is the opposite of maximizing FMA
utilization, and it matches the hypothesis raised earlier in discussion that on
a latency-bound kernel, full fusion can be locally suboptimal.

### 5.4 Zero stack spills, transpose factored out (direct evidence)

* Of the 92 vector moves, 64 are pointer-relative data loads and stores, and
  zero are `rsp`/`rbp`-relative. The radix-8 step holds its entire working set
  live across the 26 `ymm` registers it uses, with no stack spilling.
* There are no shuffle or permute instructions in the step. The transpose lives
  in the separate `_row_1d_dft_twist_transpose` pass, not inside the butterfly.

### 5.5 Scheduling (inference)

The loop body interleaves the next iteration's loads among the current FMAs,
alternates independent multiplies and FMAs across spread-out destination
registers, and reaches into the extended register file (`ymm16+`) specifically to
keep enough independent work in flight. This is deep software-pipelining and is
well balanced against the two FP pipes, whether produced by the compiler or by
hand. It is the "balanced issue order" that distinguishes a tuned kernel from a
naive one.

## 6. Verdict on the original question

MKL's FFT is a generated mixed-radix codelet library. The power-of-two path is
built from radix-4 and radix-8 step primitives plus separate twiddle and
transpose passes, with per-size leaf codelets for the non-power-of-two factors.
The one kernel examined reads as compiler-generated code (standard frame, real
loop, balanced schedule), runs at 256-bit EVEX width by deliberate choice, and
uses selective rather than maximal FMA fusion.

"Hand-written assembly, optimized to the end by humans" is not what the evidence
shows. The optimization is real but lives in the algorithm and decomposition, the
source and compiler, the width strategy, and the schedule, not in hand-authored
assembly. This is a more interesting and more reproducible kind of optimized than
the reputation suggests.

## 7. Implications for VectorFFT

### 7.1 The 256-bit choice explains the measured thermal behavior

VectorFFT's codelets are 512-bit (`zmm`, 8 doubles per register). The earlier
observation was a roughly 2.35x advantage over MKL on cold runs that narrows
under sustained thermal load. The width finding gives a concrete mechanism:

* Cold or short: 512-bit does twice the lanes per instruction, so VectorFFT
  leads.
* Sustained: 512-bit trips the AVX-512 frequency drop and the core clocks down,
  while MKL at 256-bit holds its clock, so the lead narrows.

The single-shot, cache-resident container benchmark (not thermally saturated)
showed VectorFFT log3 ahead of MKL by about 6 percent, consistent with 512-bit
winning when the clock is not throttled. This is a plausible explanation tied to
direct evidence, not a proven causal claim; confirming it requires power and
frequency telemetry on real hardware.

### 7.2 The fusion finding is a clean experiment

MKL's selective fusion (28 FMA against 36 standalone multiplies) suggests that
VectorFFT's max-FMA codelets may sit at a locally suboptimal corner at this size
if the kernel is latency-bound. The experiment is isolated: selectively un-fuse
the multiplies that sit on the critical path so they retire early, and measure.

### 7.3 The two designs sit at opposite corners

| axis | MKL (this kernel) | VectorFFT |
|---|---|---|
| radix granularity | small steps (4, 8), looped | large codelets (16, 32, 64), unrolled |
| transpose | separate pass | fused into special codelet variants |
| vector width | 256-bit EVEX | 512-bit |
| stack spills | none (fits in 26 ymm) | some at large radix |
| frequency under load | sustains (256-bit) | throttles (512-bit) |
| per-uarch variants | dispatched per ISA | one AVX-512 target |

Each corner buys something. VectorFFT's transpose elision is a real advantage MKL
does not have, since MKL pays separate transpose passes. MKL's 256-bit width is a
real advantage under sustained load that VectorFFT does not have. The near-parity
in the cache-resident benchmark is these tradeoffs roughly cancelling.

### 7.4 Where the residual gap actually lives

VectorFFT has already maximized the static-count axes: arithmetic near the
Winograd minimum, spills near the minimum the 2-pass structure allows, maximal
FMA fusion, transpose-reducing codelets, and even control over GCC via the
M-project fences. Those are necessary and largely captured. The residual is in a
different domain: schedule quality, the latency-versus-throughput balance, vector
width strategy under thermal load, and per-microarchitecture tuning. The concrete
lever is the scheduler. The current selective-unpinning plus Graham-style greedy
list scheduler is locally myopic and starts costing at the latency-bound
crossover where instruction order decides whether the kernel hits the throughput
floor or stalls on latency. Because the schedule is a pass and not the
architecture, it can be replaced with a port-and-latency-modeling scheduler
without rewriting anything underneath it.

## 8. Benchmark context

For reference, the comparison that motivated this (single-thread, out-of-place,
batched K transforms of N = 1024, interleaved rdtsc min-of-120, on the noisy
single-vCPU VM):

* Cache-resident regime, K = 128 (about 6 MB working set, codelet-bound), stable
  over five repetitions: FFTW/MKL approximately 1.12 to 1.28 (MKL beats FFTW, as
  expected), and MKL/VF_log3 approximately 1.05 to 1.07 (VectorFFT log3 a few
  percent ahead of MKL).
* Larger K is memory-bandwidth bound (at K = 512 the three buffer sets total
  about 48 MB and FFTW alone runs 2.6x slower than in cache); the ratios there
  reflect layout, not codelet quality.
* This contradicts an earlier session on the same VM that had MKL well ahead.
  The robust, regime-independent statement is that the gap closed from tens of
  percent to single digits; the sign of the small remaining margin is unsettled
  until confirmed on quiet hardware.

## 9. Caveats and open questions

* Only one representative double-precision `STEP_RADIX8` was disassembled. It was
  not exhaustively confirmed that no FFT path or precision uses 512-bit `zmm`. A
  scan of the FFT kernels for any 512-bit-width code would tell us whether
  256-bit is universal in MKL's FFT or specific to this path.
* The exact kernel dispatched for a committed N = 1024 double-complex descriptor
  was not pinpointed through the dispatch indirection. `STEP_RADIX8` is on the
  power-of-two path and is double-precision (`pd` operands), so it is
  representative, but "this is literally the 1024 inner loop" was not proven.
* The compiler (icc versus icx) and whether the source is intrinsics or generated
  C were not determined. The "compiler output" verdict rests on the frame, the
  loop, and the schedule, which is strong but not a source-level proof.
* All timing is from a single-vCPU virtualized host. Rankings from interleaved
  min-of-N are reliable; absolute cross-binary times are not. The AVX2 deploy
  target and any real-hardware frequency behavior remain unmeasured.

## 10. Bottom line

MKL's FFT is a generated mixed-radix codelet library, compiled rather than
hand-written in assembly, run deliberately at 256-bit EVEX width, with selective
FMA fusion and no stack spills in its small looped radix steps. It is hard to
beat because of algorithmic decomposition, compiler schedule quality, a width
strategy that sustains clock under load, and per-microarchitecture dispatch, not
because of hand-authored assembly. VectorFFT already matches or slightly beats it
in the cache-resident regime by occupying the opposite design corner (large
512-bit codelets with fused transposes). The remaining work is not more of the
static-count optimization that is already maxed; it is scheduling, the
latency-throughput balance, and width strategy under thermal load.

## MKL FFT ALGORITHM DETERMINED (binary reverse-engineering, container)

Method: no PMU in the KVM guest (perf_event_open -> ENOENT), so used gdb PC-sampling
to find the hot kernel + objdump on libmkl_avx512.so.3 (symbols present in .symtab).

HOT KERNEL (batched c2c-256, K=256): mkl_dft_avx512_mg_rowbatch_twidl_fwd_016_d
=> radix-16 twiddle codelet => N=256 runs as (16,16) COOLEY-TUKEY.

CODELET FAMILY (libmkl_avx512.so.3 symtab): per-radix codelets for
8,10,12,16,24,32,36,48,60,64,72,96, each as {plain (no-twiddle stage 0),
twidl (twiddle stages), scale}. This plain+twidl-per-radix structure IS
Cooley-Tukey mixed-radix (same organization as FFTW genfft, and as our CT path).
ZERO split-radix naming in DFT symbols. (The cFft_BlkSplit_32/64 symbols are
cache "block-split", not the split-radix butterfly.)

VERDICT: MKL = Cooley-Tukey mixed-radix, NOT split-radix. High confidence (3 lines:
hot-kernel name, codelet family, disassembly). MKL uses large fused radices up to 96
(few passes / low memory traffic) - same lever as our (32,32)-for-1024 finding.

RADIX-16 CODELET HEAD-TO-HEAD (MKL final asm vs our t1 intrinsics):
| codelet         | fma | mul | add/sub | perm/shuf | layout                        |
| MKL twidl_016   | 16  | 52  | 168     | 167       | interleaved, vec WITHIN xform |
| VFFT t1_dit_16  | 70  | 30  | 104     | 0         | split-complex, vec ACROSS K   |

THE REFRAME: MKL and VFFT-CT use the SAME algorithm (CT). The MKL gap is NOT SR-vs-CT;
it is the SIMD AXIS. MKL's interleaved-complex pays 167 permutes + 237 moves to align
re/im across lanes; our split-complex K-batch pays 0 permutes (contiguous loads). That
0-vs-167 shuffle tax is why we win 1.29x in the batched layout and MKL wins in its
contiguous-per-transform layout. Confirms section 68 SIMD-axes taxonomy in MKL machine code.
=> SR is orthogonal to the MKL gap. Our edge vs MKL is the split-complex K-batch axis.
