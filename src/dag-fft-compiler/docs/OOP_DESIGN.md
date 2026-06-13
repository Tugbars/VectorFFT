# The out-of-place FFT design

This document explains how VectorFFT computes an out-of-place (input preserving),
natural-order FFT, and why the design is structured the way it is. It assumes
familiarity with Cooley-Tukey and with FFTW's general shape. The companion
document `SUPPORTED.md` describes which ordering modes the engine exposes and
when to use each one.

The short version: VectorFFT borrows FFTW's data flow exactly. There is one
out-of-place pass, concentrated in a single no-twiddle leaf, and every twiddle
stage after it runs in place on the output buffer. There is no separate work
buffer and no out-of-place twiddle stage. The codelets are emitted by an OCaml
DAG compiler and are individually faster per call than FFTW's genfft leaves.


## 1. The core idea: copy FFTW's data flow, not its codelets

FFTW does not move data out of place at every stage. It does it once. Reading
FFTW's own plan tree for N=1024 (out-of-place, PATIENT) makes this explicit:

```
N=1024, K=512:
(dft-ct-dit/64
  (dftw-direct-64 "t2fv_64")        <- twiddle radix-64 step, IN-PLACE (kdftw)
  (dft-vrank>=1
    (dft-direct-16 "n2fv_16")))     <- size-16 no-twiddle leaf, OUT-OF-PLACE (kdft)
```

Two codelet ABIs are in play, and the distinction is the whole point:

* `kdft` (the notw leaves, `n2fv_R`) is **out-of-place**. It takes separate
  `const R *ri, *ii` and `R *ro, *io` pointers with independent input and output
  strides. This is where the one and only out-of-place move happens: it reads the
  input (with the digit reversal folded into the read stride) and writes the
  output buffer.
* `kdftw` (the twiddle codelets, `t1fv_R`, `t2fv_R`) is **in-place**. It takes a
  single `R *rio` array and operates on it in place.

So FFTW reads input into the output buffer once through the notw leaf, then runs
every twiddle stage in place on that buffer. No work buffer, no second shuffle.
Natural order falls out of the recursion's input/output strides.

VectorFFT's earlier engine did the opposite: it moved data out of place twice
(input to a block-local work buffer through a notw leaf, then work buffer to
output through an out-of-place twiddle stage). The codelets were already faster
than FFTW's per call, but that advantage was being spent on a data move FFTW
never makes. Removing the second shuffle (running the twiddle stage in place on
the output, deleting the work buffer) is the central structural fix and is worth
roughly 1.17 to 1.20x over the work-buffer engine in the cache-resident regime.


## 2. The codelet model (FFTW-like, two families plus a per-position variant)

All codelets are straight-line DFTs emitted by the OCaml DAG compiler in
`generator/`. There are three families, mapping onto FFTW's leaf/twiddle split:

| VectorFFT family | role | placement | FFTW analogue |
|------------------|------|-----------|---------------|
| `n1` (no twiddle) | the leaf: a size-R DFT with no twiddles | out-of-place capable | `kdft` / `n2fv_R` |
| `t1s` (scalar twiddle) | pre-multiply each input leg by a per-leg broadcast twiddle, then size-R DFT | in-place | `kdftw` / `t1fv_R` |
| `t1` (per-position twiddle) | same, but the twiddle is a per-position vector, so one call covers all positions | in-place | `t1fv_R` with a vector twiddle |

The leaf does the out-of-place move. The twiddle codelets are decimation-in-time:
each input leg is multiplied by its twiddle, then the radix butterfly runs. Leg 0
gets the identity twiddle.

### The ABI

The out-of-place codelets use a single uniform signature, richer than FFTW's
kdft, with four strides:

```c
void radixR_KIND_oop_fwd_avx512_UG_UG(
    const double *in_re,  const double *in_im,
    double       *out_re, double       *out_im,
    const double *tw_re,  const double *tw_im,   /* NULL for n1 */
    size_t in_leg_stride,  size_t in_group_stride,
    size_t out_leg_stride, size_t out_group_stride,
    size_t me);                                   /* number of groups = transforms */
```

* `me` is the number of groups processed, which is the number of transforms (or
  SIMD lanes) handled in this call. The codelet vectorizes 8 groups at a time
  (one AVX-512 ZMM holds 8 doubles).
* `UG_UG` means unit group stride on both input and output: the batch is
  contiguous, so the SIMD dimension is the batch and the layout is element-major.
* The four strides (`in_leg`, `in_group`, `out_leg`, `out_group`) are the reason
  one codelet can sit in a leaf, a transposed, or a twiddle-step position purely
  by how it is called. FFTW needs distinct codelet variants for some of these;
  here it is a stride choice.

The in-place stride codelets used by the batched executor in `core/` take a
narrower signature, `radixR_KIND_dit_fwd_avx512(rio_re, rio_im, tw_re, tw_im, ios, me)`,
operating on a single `rio` array at inter-leg stride `ios`.

### In-place aliasing is safe

The twiddle stage runs in place on the output buffer, which means the codelet is
called with `in == out` and matching strides. This is correct because the
straight-line codelets load all R input legs into registers before storing any
output. Register spills go to the stack, not to the in/out buffer, so the buffer
is read cleanly and then written cleanly. This holds up to radix-64 (verified:
the in-place twiddle-64 stage of the worked example below is bit-for-bit
identical run in place versus out of place).


## 3. Worked example: N = 1024 = 16 x 64, decimation-in-time

This is the validated two-stage FFTW-method engine. Let M = 16 (the notw leaf
size, the cofactor) and R = 64 (the in-place twiddle radix). V = 8 (lanes).
Layout is element-major within a block: `buf[element*V + lane]`.

**Stage 1: notw leaf, size 16, out-of-place (64 calls).**
For each input sub-sequence `n1` in `[0, 64)`, run a size-16 DFT that gathers the
sub-sequence from the input (leg stride `64*V`) and scatters it contiguously to
the output (leg stride `V`, base `16*n1*V`):

```c
for (int n1 = 0; n1 < 64; n1++)
    radix16_n1_oop(in + n1*V, ..., out + 16*n1*V, ..., NULL, NULL,
                   /*in_leg*/ 64*V, /*in_grp*/ 1, /*out_leg*/ V, /*out_grp*/ 1,
                   /*me*/ V);
```

This is the single out-of-place move. Input is read, output is written, nothing
else is touched.

**Stage 2: twiddle radix-64, in-place on the output (16 calls).**
For each output position `k2` in `[0, 16)`, run a size-64 twiddle butterfly over
the 64 sub-results at stride `16*V`, in place (`in == out`):

```c
for (int k2 = 0; k2 < 16; k2++)
    radix64_t1s_oop(out + k2*V, ..., out + k2*V, ...,   /* in == out */
                    tw[k2], ..., /*in_leg*/ 16*V, /*in_grp*/ 1,
                    /*out_leg*/ 16*V, /*out_grp*/ 1, /*me*/ V);
```

The twiddle for position `k2` is `W_1024^{n1 * k2}` on leg `n1` (broadcast across
the V lanes). The result `X[16*k1 + k2]` lands at element `16*k1 + k2`, which is
natural order. Input is preserved, no work buffer is used, and the result is
correct to machine precision (relative error around 1e-14).

The same recursion generalizes: split the leaf or the twiddle radix further to
get more stages. Whether that helps depends entirely on batching, which is the
subject of section 6.


## 4. Why the old engine was slower, and what the fix removes

The work-buffer engine for the same N=1024 did:

1. notw leaf, size 64, input to a block-local work buffer (out-of-place move 1),
2. twiddle stage, size 16, work buffer to output through an out-of-place twiddle
   that also performed the natural-order transpose (out-of-place move 2).

That is two out-of-place passes plus a work buffer. At K=2048 both engines are
DRAM-bound and at parity, because the extra pass is hidden behind memory latency.
At K=512 the working set is L3-resident, the extra round trip through the work
buffer is exposed, and the old engine loses 10 to 20 percent. The in-place
twiddle design deletes move 2 and the work buffer entirely, which is the measured
1.17 to 1.20x improvement and the bulk of the gap to FFTW.


## 5. Vector blocking is the single biggest lever

Independent of the structural fix above, how the K transforms are batched
dominates everything. The natural-order engine runs the transforms in blocks of
V = 8 (one AVX-512 vector), with the block laid out element-major:
`buf[block*N*V + element*V + lane]`. A block is `V * N * 16` bytes = 128 KB and
stays resident in L2, so the intermediate reads and writes are L2 hits and the
only DRAM traffic is the input block read and the output block write, each once.

The naive unblocked recursion scatters the intermediate across the entire working
set and runs at roughly 0.21x of FFTW. Blocking alone brings it to parity at
scale. Widening V past 8 makes it worse, not better (0.77x at V=8 down toward
0.60x as V grows), because cache residency matters more than the extra
instruction-level parallelism. V=8 is the sweet spot.


## 6. Batching determines whether small radices help

A separate exhaustive search over factorizations (run inside the `core/` batched
stride executor) found that small radices win there: 4x4x4x4x4 reaches FFTW
parity or beyond, while 64x16 is the worst row. That result does **not** transfer
to the per-block natural-order engine, and the reason is batching.

* The batched stride executor calls each codelet with `me = K` (the whole batch),
  so a tiny radix-4 butterfly still does K transforms of work per call and the
  many calls amortize.
* A stage can instead be run as a SINGLE call with `me = positions*lanes` over
  the whole block. This is the one-call form, and it is the fast path. Each
  twiddle stage needs a per-(leg,position) twiddle broadcast across the V lanes,
  which is exactly the `t1p` codelet (per-position broadcast). The earlier `t1`
  (per-position vector load) replicated each twiddle V=8 times; `t1s` (one
  twiddle per call) cannot vary the twiddle within a call. `t1p` does both: one
  call, compact (R-1)*(me/V) table. See engine_natural_oop_onecall.c.

One-call measured results (AVX-512, this VM, FFTW PATIENT, interleaved min-of-60),
2-factor sweep for N=1024 (the split forces one radix >= 32):

| split   | big radix is  | rel to FFTW   |
|---------|---------------|---------------|
| 32x32   | balanced      | 1.07 to 1.15x |  <- best
| 16x64   | twiddle stage | 1.01 to 1.04x |
| 64x16   | leaf stage    | ~1.02x        |
| 16x16x4 | 3-stage small | 0.99 to 1.01x |

32x32 wins because spill grows faster than linearly with radix on 32 ZMM
registers: a radix-64 stage wants 128 registers (263 stack refs in the emitted
code), a radix-32 stage wants 64 (135 refs). Balancing into two radix-32 stages
minimizes total spill, and the twiddle stage was the bottleneck. Placing the big
radix as the leaf (64x16) costs about the same as placing it as the twiddle
(16x64), both ~1.02x, so balance is what wins, not placement. FFTW's PATIENT
planner also picks 32x32 for this size. The small-radix 16x16x4 loses: natural
order forces a multi-pass data flow, so each extra stage is another full array
pass that the radix-4 register savings do not recover. Small radices are a
property of the `me = K` batched executor, which produces permuted output
(section 7); the advantage does not transfer to natural order. For contrast, the
16x64 at the per-block `me = 8` granularity (80 calls/block) runs 0.87-0.95x; the
one-call collapse to 2 calls/block is what moves it above FFTW.

Twiddle policy: log3 on the 32x32 twiddle stage. The radix-32 twiddle stage spends
real time on broadcast twiddle loads (31 `set1_pd` per position). The log3 policy
loads only the 5 base twiddles W^{1,2,4,8,16} (slots 0,1,3,7,15) and derives the
other 26 by complex multiply. It trades 26 loads for ~52 mul/fma ops, and on
AVX-512 that nets out 2 to 4% faster (log3/flat ~0.94 to 0.99 across K, both runs),
because the FMA ports were underused while the load path was a bottleneck. The
twiddle table is unchanged: log3's slot indexing matches the flat layout exactly,
so it reads a sparse subset of the same array. The compiler has special log3
inline-asm scheduling gated on R<=32 AND AVX-512, which is exactly this stage.
This is generated by adding `--log3` to the t1p generation command, which routes
`Dft.TP_Log3` into the OOP per-position expansion. So the recommended engine is
32x32 with a log3 twiddle stage, at ~1.03 to 1.12x FFTW.

Stage-count study (why few stages win for OOP natural order). Extending the sweep
by depth, with the general multi-stage DIT one-call rule (leaf decimates with
stride N/R0 and scatters its output by digit weight; stage s>=1 does N/P_{s+1}
blocks of P_s positions with legs at stride P_s*V and twiddle
W_{P_{s+1}}^{leg*position}; inner stages run in-place):

| factorization | stages | calls/block | rel to FFTW   |
|---------------|--------|-------------|---------------|
| 32x32         | 2      | 2           | 1.07 to 1.11x |  <- best
| 16x16x4       | 3      | 9           | 0.99 to 1.01x |
| 4x4x4x16      | 4      | 97          | 0.80 to 0.83x |  <- first below FFTW

The degradation is monotonic, and the call count is the cause. The outermost
stage is always a single block, so it is one clean t1p call, but inner stages with
small radices form many small blocks (4x4x4x16 has 64 then 16), and the block
stride does not equal the position stride, so they cannot collapse to one call
each. Stage 1 of 4x4x4x16 is 64 calls each doing only 4 groups, a poor
work-to-overhead ratio, on top of 4 array passes versus 2. The arithmetic per
stage shrinks but per-call overhead and extra passes dominate. The one gain is
precision: 4x4x4x16 is ~2x more accurate (3.4-4.0e-15 vs 32x32's ~8e-15) because
smaller radices accumulate less roundoff per butterfly, which is the axis MKL
leads on. The same 4x4x4x16 runs ~0.84 to 0.95x in the in-place stride executor,
better than the 0.83x here, because in-place halves the passes and inlines the
codelet in a tight per-group loop instead of paying the 97-way call fragmentation
an OOP engine is forced into. That gap is why in-place wins and OOP is the harder
mode for deep factorizations. See engine_natural_oop_4stage.c.

Note on a corrected earlier claim: a previous version of this document stated the
one-call `t1` path was blocked by a `register ... asm volatile` pinning
miscompile at large `me`. That was wrong. The radix-4 `t1` codelet is correct at
`me = 2048` (relerr 5.68e-15, bit-identical to `t1s`). The earlier relerr ~1.19
was a twiddle-table off-by-one in the engine, not a codegen bug: `t1`/`t1p` index
`tw[j]` for `j` in `[0, R-1)` meaning legs 1..R-1, since leg 0 is the identity and
has no twiddle slot. The failing table included a leg-0 entry, shifting every
twiddle by one leg. With the table built leg-1-based, the one-call path works and
needs no pinning fix.


## 7. Ordering: natural versus permuted

The per-block engine in this design produces **natural order** directly. The leaf
gathers the input with the digit reversal folded into the read stride, and the
in-place twiddle stage writes results to their natural element positions. No
separate bit-reversal pass.

The batched stride executor in `core/` is a standard iterative decimation-in-time
and produces output in a fixed mixed-radix **permuted** order. It is numerically
correct (Parseval energy matches FFTW exactly), just reordered. Turning that into
natural order requires either an explicit permutation pass (a full N x K shuffle,
whose cost roughly cancels the small-radix advantage) or folding the digit
reversal into the leaf with a different per-stage stride structure than the
permuted executor uses. The latter is FFTW's full recursive natural-order
machinery and is not built here. `SUPPORTED.md` covers when permuted output is
fine (it usually is, for convolution and filtering) and when it is not.


## 8. The generator (OCaml DAG compiler)

Every codelet in `codelets/` is emitted by the dune project in `generator/`:

```
cd generator && dune build
# binary at generator/_build/default/bin/gen_radix.exe
```

The pipeline is expressions -> CSE and algebraic simplification -> scheduling ->
register allocation -> C emission:

* `dft.ml`, `split_radix.ml`: the FFT algebra (the DAG of complex operations).
* `codelet_oop.ml`: the out-of-place codelet family used by the natural-order
  engine.
* `schedule.ml`, `regalloc.ml`: instruction scheduling and register allocation.
* `emit_c.ml`: C emission (this is where the `register asm` pinning lives).
* `simd_ir.ml`, `isa.ml`: the AVX2 and AVX-512 backends.
* `uarch.ml`: the cost model.
* `bin/gen_radix.ml`: the CLI driver.

The exact invocations that emit the shipped codelets are in the top-level
`README.md`. The relevant flags: `--oop` selects the out-of-place family,
`--oop-load`/`--oop-store UG` set unit group strides, `--twiddled-scalar` emits
`t1s`, `--twiddled` emits the per-position `t1`, and `--isa avx512` or
`--isa avx2` selects the backend.


## 9. Register pressure and stride specialization

The radix-32 codelets are 2-pass blocked (`dft_expand_n1_blocked`), so each pass
fits in registers (measured peak_live 17 and 10 against a 32-ZMM budget, versus
120 for the naive monolithic DAG). Three further levers reduce the vector moves
around that:

* **M-project** (`--fuse N`): keep the last N PASS-2 sub-DFTs register-resident
  across the pass boundary instead of round-tripping through the spill arrays.
  Now wired into the OOP emitter (previously a no-op there).
* **Stride specialization** (`--oop-strides L,G,OL,OG`): bake the four strides as
  compile-time constants. Folds the per-leg address arithmetic to constant
  displacements and frees the four argument registers. Independent of and
  additive to M-project.
* **Store-on-compute** (`--oop-store-fused`): for the UnitGroup store, write each
  output to memory the instant it is computed instead of accumulating all of them
  into `out_lane_*` registers and storing at the end. Removes the 2R output
  accumulators. The biggest of the three.

Together they take the radix-32 leaf from 162 stack spills / 342 vmovapd to
90 / 189, below even the in-place codelet (107 / 238). The specialized 32x32
engine (`engine_natural_oop_onecall_spec.c`, built as `engine_onecall_spec`,
codelets regenerated by `codelets/regen_spec_r32.sh`) is 6 to 10 percent faster
than the general one-call engine on AVX-512 at identical numerics, and the log3
variant beats FFTW by roughly 1.15 to 1.23x.

After this spill work, a re-measurement against Intel MKL (DFTI, single-thread,
out-of-place) shows the gap has closed: in the cache-resident regime on the test
machine MKL no longer leads, and VectorFFT log3 is a few percent ahead, versus
the earlier 15 to 35 percent MKL advantage. That result is regime-sensitive and
not yet confirmed on quiet hardware; full analysis and caveats are in
`docs/mkl_comparison.md`. Interleaving the two passes (to drop the spill arrays
entirely, as FFTW does) remains the open lever and would need an OOP-scheduler
rewrite. Full analysis and the measurement method for the codelet work are in
`docs/oop_stride_specialization.md`.



* One out-of-place move (the notw leaf), then in-place twiddle stages, no work
  buffer. This is FFTW's data flow and the main structural fix over the older
  engine (about 1.18x).
* Three codelet families: `n1` (leaf, out-of-place), `t1s` (in-place twiddle,
  broadcast), `t1` (in-place twiddle, per-position, currently large-`me`
  limited). One uniform four-stride ABI.
* Vector blocking at V=8, element-major, L2-resident, is the largest single lever
  (about 3x over naive).
* Small radices help only at `me = K` batching (the permuted-output stride
  executor), not in the per-block natural-order engine.
* The per-block engine gives natural order directly; the batched executor gives
  fast permuted order.
