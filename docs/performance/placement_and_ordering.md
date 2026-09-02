# Placement and ordering: how the engines are built, and what each one pays

**Scope.** 1D complex-to-complex. Why VectorFFT ships more than one c2c engine, what each
engine gets for free, what each one has to construct, and what the construction costs.

**Audience.** An engineer who knows FFTs and has to decide where a new feature belongs.

**Reading rules used throughout.**

* Every non-obvious claim carries a `file:line`. Where a shipped comment disagrees with the
  code, the code wins and the comment is listed in [Appendix A](#appendix-a--comments-that-have-outlived-their-code).
* Measured figures carry a source **and a vintage**. The host (i9-14900KF Raptor Lake, AVX2
  only, ~5.7 GHz, 48 KB L1d per P-core) is thermally noisy: only same-run arms are
  comparable, and third digits are not stable. Cross-vintage division of two ratio columns
  is not a measurement.
* Figures whose source file is absent from the tree, or which live under `docs/research/`
  (ignored by `.gitignore:100`, so absent from a fresh clone), are labelled as such.
* **Regime warning, load-bearing:** every interleaved number in this document is `K = 1`.
  Every split natural-order number is batched `K > 1`. The two halves were measured in
  different regimes and are not cell-comparable.

---

## 1. The question, and the answer

A user of this library picks two axes independently: **placement** (in-place, `z -> z`, or
out-of-place, `in -> out`) and **ordering** (natural `X[k]`, or the engine's scrambled
digit-reversed permutation). Four combinations. Three of the four are, for at least one
engine, something the engine had to be *built* to do rather than something it does anyway.

The answer is not "the layout decides". The answer is one sentence about buffers:

> **A pass can fold a permutation into its stores exactly when it already owns a destination
> buffer to scatter into. An engine can run in place exactly when it already owns a scratch
> buffer that decouples the last read of the input from the first write of the output. Both
> properties are bought with the same asset — a second buffer — and each engine's founding
> contract either forbids that buffer or requires it.**

The split (Method-C) engine's founding contract forbids it: one buffer, every stage strides
differently over it, no stage moves an element (stated in the header of the since-removed `engine/stride_executor.h`; the live split engine is `executor_generic.h` + `proto_stride_compat.h`).
So in-place and scrambled are not features it supports — they are the only things its
codelets can express, and natural order and out-of-place are both constructions that end up
buying a second buffer by other means. The four-step (Bailey) interleaved engine's contract
requires it: its leaf's stores are a corner turn, i.e. a scatter, so a destination plane must
exist, and once it does, natural order is that turn and in-place is nearly free.

The confirmation is inside the split family itself: `oop_plan.h` ships a four-step kind
(BAILEY2) with the transpose fused into the stores and **natural** output — but only inside
the *out-of-place* plan kind, the one that has a destination. The general kind (MODEB), which
has no destination of its own, is scrambled (`src/core/oop/oop_plan.h:5-18`). The turn appears
exactly where a destination buffer exists, in either layout.

Two corrections that the rest of this document depends on:

1. **There are three engines, not two.** The library's two "interleaved engines" have
   *opposite* founding orders. The Bailey tier (`il2p.h`, `il3p`) is natural-native. The
   cascade tier (`zsplit.h`, `zturn.h`) is **scrambled**-native — `zsplit.h:9-11` states the
   digit-reversed comb as its contract — because its interior is Method C running in place on
   a scratch plane. The ordering tax therefore lands on opposite sides of N = 2048.
2. **The scratch plane is forced by the in-place contract, not by the four-step shape.** Only
   one of Bailey's two passes is turned (§3.2). A strictly out-of-place four-step could write
   the leaf straight into `zout`; `mid` exists because the API allows `zin == zout`.

---

## 2. The two layouts, physically

**Split (column) layout.** Two `double` arrays. Element `e` of transform `t` lives at
`re[e*K + t]` and `im[e*K + t]` — a real plane and an imaginary plane, transform-minor
(`src/core/oop/oop_plan.h:37-39`). At `K = 1` this is two contiguous arrays of `N` doubles.

```
re: [ x0.re x1.re x2.re x3.re ... ]        two planes, 8N bytes each
im: [ x0.im x1.im x2.im x3.im ... ]
```

**Interleaved (`z`) layout.** One `double` array, `z[2*e] = re`, `z[2*e+1] = im`.

```
z:  [ x0.re x0.im x1.re x1.im ... ]        one plane, 16N bytes for N complex
```

Why both exist is not aesthetic. It is the SIMD unit:

* Split lets a `ymm` hold **4 consecutive points of the same component**. A complex multiply
  is `fnmadd`/`fmadd` on four separate register operands, with no lane shuffling anywhere.
  This is why split codelets are written against a `me` (multiple-element) count and why
  batching `K > 1` is natural for them.
* Interleaved packs **2 complex per `ymm`** with `re` and `im` in alternating lanes. A complex
  multiply now needs a lane flip: the twiddle apply is `BYTW2 = fmadd(c, x, mul(s, cflip x))`
  where `cflip` is `_mm256_permute_pd(x, 0x5)` (`src/core/oop/il2p.h:41-43`; emitted form at
  `codelets/zil/avx2/pure_il/radix4_z_t2_avx2.c:43-47`). That per-multiply permute is a real,
  permanent tax on the interleaved layout — and it is the mechanism that decides §3.3.

The twiddle *tables* inherit the layout. Split's table is `cos[]` and `sin[]`. Interleaved's
VTW2 record covers a **column pair** and is eight doubles wide:
`[c(k), c(k), c(k+1), c(k+1)][-s(k), +s(k), -s(k+1), +s(k+1)]`, at
`tw + (pp*(R1-1) + (l-1))*8` (`src/core/oop/il2p.h:38-45`, filled at `:661-686`). The
duplicated cosine and the sign-alternating sine *are* the interleaved layout showing up in the
constant pool.

The library serves interleaved callers for a layout reason, not a performance one: MKL's
default DFTI complex layout is interleaved, and so is most calling code. A converter exists
(`_exec_c2c_interleaved`, `src/core/vfft.c:6086-6239`) and costs two full passes over the data
plus running the wrong engine for the layout — the ILP race measured that at 4–9× (§5.4).

---

## 3. The three founding contracts

### 3.1 Method C (split): one buffer, no motion

Read the plan's `N*K` buffer as an `nf`-digit mixed-radix array with a trailing batch
dimension. The positional weight of digit `s` is

```
dim_stride[s] = K * PROD_{d > s} factors[d]                 (twiddle.h:31-40)
offset(d0..d_{nf-1}) = SUM_s d_s * dim_stride[s]
```

Stage `s` owns digit `s`: radix `R = factors[s]`, leg stride `dim_stride[s]`,
`num_groups = N/R`, and `group_base[g]` is a mixed-radix odometer over the *other* digits, so
the `N/R` groups tile the buffer exactly (`src/core/engine/twiddle.h:41-70`).

```
for s in 0..nf-1:                          # the entire executor
    for g in 0..N/R_s-1:
        base = group_base[g]
        legs at base + j*dim_stride[s],  j = 0..R_s-1
        codelet(re+base, im+base, tw, ios = dim_stride[s], me = slice_K)
        #   load  -> butterfly -> store back to THE SAME addresses
```

`src/core/engine/executor_generic.h:60-76` is literally that loop. There is no copy, gather,
scatter or transpose in it. The only writes in the engine are the codelets' own stores.

**The contract is legible in one line of the codelet.** The twiddled workhorse loads
`rio_re[3*ios + k]` and stores `rio_re[3*ios + k]` — the same address expression on both sides
(`src/dag-fft-compiler/codelets/inplace/avx2/r4_t1_dit_fwd.c:28,30` vs `:64-71`). And the ABI
makes it structural rather than incidental: the twiddled codelet type has **six arguments, one
buffer pair, no destination and no output stride**:

```c
typedef void (*vfft_proto_codelet_fn)(double *, double *,
                                      const double *, const double *,
                                      size_t, size_t);          /* plan_executors.h:23-25 */
typedef void (*vfft_proto_n1_fn)(const double *, const double *,
                                 double *, double *,
                                 size_t, size_t, size_t);       /* plan_executors.h:26-28 */
```

Only the *untwiddled* kind (`n1`, seven arguments) can address a destination at all. This
asymmetry is the whole reason split's out-of-place story reduces to "whichever stage happens to
be untwiddled" (§5.2).

**Why "no permutation" forces digit-reversed output.** Stage `s` butterflies along digit `s`
and writes each result back to the slot it read from. After stage `s`, digit-position `s` no
longer holds an input index — it holds a frequency index, at the same positional weight.
Cooley–Tukey consumes input digits from the fastest-varying dimension inward but produces
frequency digits with the opposite significance ordering, so with nothing ever moving, the
frequency digits land in slots whose weights are reversed relative to the frequency's own radix
expansion. That map is exactly what the natural-order machinery has to undo, and the builder
states it: *"little-endian digits of n become big-endian slot digits in the SAME factor order"*
(`src/core/transforms/natorder/natorder_perm.h:24-37`). Scrambled output is not a design choice
with a cost — it is the absence of any mechanism that could produce anything else.

**Twiddles.** Method C splits the cross-stage exponent into a leg-independent common factor
`cf0[g] = W_N^{k_prev*ow_prev*lower_data_pos}` and a leg-linear `per_leg[j]`, bakes their
product into the table for legs `1..R-1`, and applies `cf0` to leg 0 at execute time
(`src/core/engine/twiddle.h:81-107`, fills at `:224-291`). Values are constant across the `K`
batch lanes, so the table is K-replicated; a `K`-slice narrower than `plan->K` must re-broadcast
into a `this_K`-strided block buffer because the codelet indexes `W[(j-1)*me + m]`
(`src/core/engine/executor_generic.h:88-113`). Four variants per stage (FLAT / LOG3 / T1S / BUF)
are raced and persisted in wisdom (`plan_executors.h:73-77`).

**Backward is free.** `IDFT(re,im) = swap(DFT(im,re))` on the *same forward plan*, unnormalized,
same ordering semantics (`src/core/oop/oop_execute.h:88-95`). This falls out of the layout: with
`re` and `im` in separate planes, conjugation is a pointer permutation, not an arithmetic sign
flip inside a packed vector. The split OOP codelet corpus is consequently **forward-only** — 90
`_fwd_` entry points to 1 `_bwd_` (and that one is an interleaved monolith) against the in-place
corpus's symmetric 162/162 (counted in-tree 2026-08-22, HEAD `d1ea8ff`). The trap the source
records: a supplied JIT inner must be the *forward*-resolved executor, because the swap lives in
the data pointers, not in the direction.

### 3.2 Bailey / four-step (interleaved): the turn is the reordering

`N = R1 * R2`. `vfft_il2p_execute_fwd` is two calls (`src/core/oop/il2p.h:692-698`):

```
             R1 columns                                 R2 columns
        +--------------------+                     +--------------------+
  R2    | zin, leg-major     |   leaf n1t(R2)      |  mid, col-major    |   mid t2(R1)
 legs   | leg p, col k       | ----------------->  |  leg k, col p      | ---------> zout
        | at zin[2*(p*R1+k)] |  TURN in the stores | at mid[2*(k*R2+p)] |  identity map
        +--------------------+                     +--------------------+
          Ls = R1, count = R1                        Ls = OLs = count = R2
                                                     + streamed VTW2 twiddles
```

Read it as: lay `N` complex out as `R2 x R1`, DFT the columns *while transposing*, twiddle, DFT
the rows. Output is natural, `X[k2 + R2*k1]`, with no reordering pass anywhere — the turn
already did it.

**The contract is again one line of address arithmetic.** The leaf loads `zin[2*(j*Ls + k)]` and
stores `zout[2*(k*OLs + j)]` — the maps differ by `(p,k) -> (k,p)`
(`codelets/zil/avx2/pure_il/radix4_z_n1t_avx2.c:38-41` vs `:50-53`; the stores recombine through
`permute2f128`). Two consequences follow with no further argument: it is a scatter, so it cannot
run in place; and the permutation it performs *is* the four-step corner turn.

**But only one of the two passes is turned.** The mid codelet loads `zin[2*(j*Ls + k)]` and
stores `zout[2*(j*OLs + k)]` — identical maps — and `il2p.h:697` calls it with `Ls == OLs == R2`
(`codelets/zil/avx2/pure_il/radix4_z_t2_avx2.c:38-41` vs `:55-58`). Stage 2 is in-place-capable.
The entire scatter in the route lives in the leaf's stores. Therefore:

> For a strictly out-of-place call, the leaf could write directly into `zout` and `t2` could run
> in place on `zout`. `mid` (2N doubles, `il2p.h:239`, allocated unconditionally at `:667`)
> exists because the API allows `zin == zout`, and with aliased buffers the leaf's scatter has
> nowhere to land.

This is a derivation from the index maps, not a measurement — no mid-free out-of-place arm exists
in the tree — but it is load-bearing, because that third plane is exactly what puts N = 1024 at
48 KB (§6).

**Backward is a different decomposition, not a conjugate.** With `re` and `im` in one word, the
split engine's free pointer-swap inverse does not exist. `t2t` runs the `R1` butterfly *first*
(the forward runs `R2` first):

```
x[a*R1+b] = SUM_k e^{+2*pi*i*a*k/R2} * e^{+2*pi*i*b*k/N}
                 * [ SUM_j X[j*R2+k] e^{+2*pi*i*b*j/R1} ]
```

Stage 1 is `t2t` at radix `R1` (post-twiddle, turned store), stage 2 is plain `n1_bwd` at radix
**`R2`** — using the `R1` twin measures 1.1e+00 error and is flagged in-source as a trap
(`src/core/oop/il2p.h:765-798`). Three properties are pinned by the derivation and each was
verified by a one-argument-at-a-time control sweep that produced O(1) error (0.54..1.37) for every
perturbation: twiddle post not pre, store turned not straight, and the exact `(Ls, OLs, count)`
triple. Direction costs a *table*, not a codelet: the kernel arithmetic is bit-for-bit the
forward's, and a separately built conjugated stream (`twb`) is supplied
(`il2p.h:38-45`, `:799-807`).

### 3.3 The cascade (interleaved, N >= 2048): Method C on a plane

The cascade is not "a bigger Bailey". It differs by **amortization**:

```
z --s0s/s0t--> plane --msg x (nf-2)--> plane --sterm/stf--> z
   ONE conversion         in-place, shuffle-free      ONE conversion
   (deinterleave paid     split mids, group-constant  (re-interleave fused
    once; turn fused       splat-pair twiddles         into the stores)
    into the stores)
```

`src/core/oop/zsplit.h:5-13` and `zturn.h:11-18`. The economics are stated in-source, quoting the
MKL RE doc: *"2 passes can't amortize a conversion; the high-N cascade converts because log-many
[passes can]"* (`src/core/oop/il2p.h:24-27`).

The `plane` is 2N doubles addressed as **four sections** at byte offsets `{0, 4N, 8N, 12N}`;
ingest position `p` emits one 64-byte record `[re x4][im x4]` into section `bitrev2(p mod 4)` at
granule `p div 4`, lanes = the radix-4 butterfly's *output* digit (`zturn.h:4-9`). So a "section"
is one of four interleaved sub-planes and a "record" is 4 complex points stored **split** within a
cache line. Two structural payoffs: the terminator reads 4 section taps, 128 B contiguous per tap,
with no load shuffles; and the mid stages are shuffle-free because `re` and `im` are already
separated inside a record. The geometry is copied from measured MKL, which is also why
`chain[0] == 4` is a hard scope fence — the four-section geometry is baked into the `_r4` kernels
(`zturn.h:33-40`).

**And here is the unification.** The mid stages run **in place on the plane**:
`f(p->sp, 0, p->sp, 0, p->twsp[s], ...)` (`src/core/oop/zsplit.h:228-236`; `zturn.h:797-805`).
That is Method C — one buffer, one stage per digit, no element moves — hosted on a scratch plane
instead of on the user's array. So the cascade inherits Method C's *ordering* behaviour
(scrambled-native: `zsplit.h:9-11` declares `out[l*(N/8)+g] = X[drev(g*8+l)]` as the contract)
while inheriting the four-step's *placement* behaviour (both placements free, because the plane
decouples: *"the terminator is the only writer of `zout` and reads only the scratch plane"*,
`zsplit.h:16-17`).

That the interior is *split* inside an interleaved-in/interleaved-out route is not an accident
either, and it was raced: an interleaved (`z`) interior **lost** to the block-split interior by
-8.9 % at 4096 (5609 vs 6157 ns) and -12.6 % at 16384 (23086 vs 26410 ns); even an all-radix-4
split chain beat every `z`-interior arm (`docs/roadmap/z_cascade_plan.md:325-328`, vintage
2026-07-24). The mechanism is §2's `cflip`: over log-many mid passes the per-multiply permute tax
dominates the one-time conversion. Granularity mattered too — *full* split planes won at 4096 but
lost at 16384 (+7.7 %, two streams per leg row), while the 64-byte block-split record (one stream
per leg row, addressing = `z` plus `+4` for `im`) swung 16384 by +29 %.

---

## 4. The symmetry, as a table

The framing "each engine gets one placement and one ordering free" is exactly right for Method C
and needs refining for the other two, because owning a scratch plane collapses the placement axis
entirely.

| engine | placement free | ordering free | must construct | derivation |
|---|---|---|---|---|
| **Method-C split** (`engine/`, general N, any K) | in-place only | scrambled only | **both** the other placement and the other ordering | one buffer, no motion (§3.1) |
| **Bailey IL** (`il2p.h`/`il3p`, K=1, 128–1024) | **both** — the plane decouples | **natural**; scrambled served by the identity permutation | nothing on either axis | the turn needs a destination, and the destination gives in-place (§3.2) |
| **Cascade IL** (`zsplit.h`/`zturn.h`, K=1, N>=2048) | **both** — the plane decouples | **scrambled** (Method-C interior) | natural order (§5.4) | Method C hosted on a plane (§3.3) |

The law behind the table:

```
owns a destination buffer  ==>  a permutation can be fused into stores      (ordering is cheap)
owns a scratch plane       ==>  reads of `in` decoupled from writes of `out` (placement is free)
```

Both are the same asset. Method C's contract forbids it and therefore pays on both axes; the
four-step's contract mandates it and therefore pays on neither. The cascade owns the asset but
spends its ordering freedom on the *interior* (Method C on the plane), so it pays a small amount
at the boundary to get natural order back.

---

## 5. The four taxes

### 5.1 Natural order on the split engine

**Mechanism.** `VFFT_ORDER_NATURAL` needs a plan-time permutation build, an execute-time reorder
pass, and — decisively — an **auto-detect** step, because the *orientation* of the permutation
(forward vs reversed factor order, `perm` vs `iperm`) is not derivable from the plan. Detection
FFTs an impulse at plan time and checks four candidate maps against the closed form
`X[k] = e^{-2*pi*i*k*n0/N}` at 12 probe points with a 1e-12 gate; a wrong map errs at O(1), so the
test is exact rather than a tolerance game, and if no candidate matches the plan **refuses**
`order=NATURAL` rather than returning wrong data (`natorder_perm.h:4-12`, `:41-76`). A system that
must empirically discover its own output permutation is the signature of a property that was
constructed rather than designed in.

The pass runs on **both** directions: forward FFT then unscramble; backward unscramble-inverse
*first*, then the zero-permutation DIF backward (`src/core/vfft.c:5599-5623`). Either way it reads
and writes all `N*K` doubles in both planes.

**Two shipped mechanisms, chosen per cell.** `PURE_CYCLE` follows the permutation's cycles with a
plan-time flattened move list, `_mm_prefetch`, and AVX row moves. `PSWAP` is the degenerate case
where the permutation is an **involution** (a palindromic factor chain), so the pass becomes
independent pair swaps. Neither needs a second plane — only `(pool_size+1)` slots of `2*K` doubles,
one complex row per worker (`src/core/vfft.c:381,4113`). Split's natural order is therefore
*memory*-in-place, which is the whole point of doing it this way.

**Why per-cell and not one mechanism.** Two independent regime flips.

* *Row width.* On the same involution with no FFT in the loop, at 128-byte rows `PSWAP` beats
  cycle-follow 2.73–4.68× (64×16: 396 vs 85 ns; 256×16: 1699 vs 621 ns) because cycle-follow is a
  serial dependency chain; at >=512-byte rows both are bandwidth-bound at ~120 GB/s and tie
  (0.93–1.06×). `docs/roadmap/natorder_2d_status.md:133-152`, 2026-07-05, core-pinned best-of-5.
* *The chain itself is a candidate.* A palindromic factorization makes the reversal an involution
  — but the DP scores chains under *scrambled* economics and prunes palindromes before
  natural-order cost exists. So the race **injects** its own candidates: `(a,a)`, `(a,b,a)`,
  uniform `a^m`, the single-stage `[N]` leaf (whose reorder is the identity), and the calibrated
  chain if it is already palindromic (`natorder_calibrate.h:44-71,126-146`). Natural order is not
  a post-processing decision; it can change which FFT runs.

**The tax is a distribution, not a constant.** Shipped winner map (T11 clean run, 2026-07-04/05;
warm-up + 5 rounds, 150 ms cool-down pacing, averaged not best-of, 400 ms between cells, pinned
core 0 — `docs/roadmap/natural_order_inplace_design.md:367-378`; batched `K > 1`):

| cell | winner | tax |
|---|---|---|
| 16/4, 32/4, 64/4 | FREE (`nf == 1`, no reorder at all) | **0 %** |
| 128/64 | PSWAP, injected 4·8·4 | **-7.3 %** (negative in 5 of 6 runs) |
| 4096/32 | PURE-cycle-UB | +18.5 % |
| 1024/32 | PURE-cycle-UB | +22 % |
| 4096/256, 64/64, 256/256 | PURE-cycle-UB | +26 % … +30 % |

> The range "-21 % to +36 %" that circulates for this tax comes from §2d of the same document,
> which `:365` explicitly supersedes. Do not quote it. Likewise, rows in which `SCR` wins are
> historical: SCR is deactivated (below).

And in the **shipped bank** the injection lever almost never fires. Of 62 banked `@nat` records,
52 are split-engine (34 `pcyc` + 18 `pswap`) and 10 are escapes to the other engine (5 `zcasc`,
5 `ilp`). Comparing each split record's deployed chain against the scrambled record for the same
`(N,K)`: **49 identical, 3 injected**. Fourteen of the 18 `pswap` cells are *opportunistic* — the
calibrated chain was already palindromic, so the verdict is banked with `nat_ns 0.00` and no race
ever ran (`src/core/vfft.c:4220-4227`). Source: `generated/wisdom2_stride.txt`, frozen v8 wisdom,
2026-08-20.

> **`nat_ns` is not a tax.** It is 0.00 on every opportunistic cell, and where populated it comes
> from the create-time race (4-execute chunks, 3 interleaved rounds), not the paced benchmark
> methodology. Different instrument; the two are not interchangeable.

**Against MKL** (whose DFTI is natively natural), public-API natural forward, 9 cells: FREE (64/4,
radix-64 leaf) 2.28×; PSWAP (100/4, palindrome 10·10) 2.72×; PURE (128/4) 1.27×; median ~1.68×,
9/9 wins (`docs/performance/v1_0_results.md:231-262`). The ranking FREE > PSWAP > PURE *is* the tax
ranking read off the ratio column. The source carries its own honesty note: measured on a live
host, ratios directional; the correctness gate (elementwise vs naive DFT, plus roundtrip, all
1e-14/1e-15) is exact.

**Inherent vs incidental.**

*Inherent.* The fused single-buffer scatter is provably impossible, and this is the load-bearing
claim. Last stage, radix `R`, `P = N/R` groups: group `q` reads `R` **contiguous** rows but its
natural targets are the stride-`P` comb `{q + j*P}`. Read units are blocks, write units are combs;
a wavefront must hold Theta(N*K) live rows before targets free up, and there are no small closed
cycle sets (`docs/roadmap/natural_order_inplace_design.md:221-229`, which supersedes an earlier
committed design that had specified exactly such a codelet). Two survivors: shrink `N` until
registers *are* the live set (LEAF-IP, ditched), or give the live set a home — a second plane.
**This is the same geometric fact that forces `mid` into existence on the IL side. Split pays it
at the end and gets nothing else; IL pays it up front and gets natural order thrown in.**

*Inherent.* The kernel is at its measured ceiling. Four separate attacks were refuted: 8-way
interleaved cycles (uniformly slower — the move list has no data-dependent addressing, so OoO
already overlaps the loads and there was no serial chain to break); `transpose.h` cache-oblivious
recursion at cell granularity (+45 %…+172 %, it costs 2 passes as a bolt-on); COBRA L1-tiled
scatter (staging copy exceeds the pattern win, every cell); full mixed-radix Stockham (1.86–2.40×
the in-place time even fully L2-resident, 3.18× on the L3 control — the hypothesis had confused
bandwidth with memory-op throughput). Residual gap to the in-place-pass floor is ~1.45× and
TLB/line-miss bound, huge pages already refuted.
`natural_order_inplace_design.md:180-209,374-390`, T7–T11, 2026-07-04/05.

*Incidental.* The **carrying cost**. `SCR` — the mode that buys the second plane and fuses the
reversal into the last butterfly stage, `nf` passes instead of `nf+1` — lost its own race by 2.76×
(4096/4: SCR 79.5 us vs PURE 28.8 us, and 0.45× vs MKL) because it must inject an uncalibrated
forced-DIT uniform-T1S plan, pay a double-footprint scratch fill, and take 0.40×-rate scattered
stores. It is deactivated: re-entry needs `-DVFFT_NATORDER_RACE_SCR`, defined nowhere in the tree,
and zero banked cells carry `mode=scr` (`natorder_calibrate.h:147-165`). ~180 lines plus
consume/build/MT wiring in `vfft.c` are carried, not used. `LEAF_IP` burned an enum slot that is
documented "never reused" (`wisdom_reader.h:55`).

*Structural, and worth naming.* Two hard constraints the tax imposes downstream. (1) MT must split
the reorder by **row ranges, never by K** — a K-split manufactures 64-byte sub-rows, the regime
gate G2 measured at 34–218 % of FFT time, per-row-overhead-bound at ~30 GB/s against 150+ GB/s
streaming (`natural_order_inplace_design.md:108-110`). The kernels are therefore written as
cycle/pair *range* functions. (2) A natural MEASURE cell pays **two** full calibrations at create
— the c2c chain race and then up to 5 candidate chains, each needing its own plan build, impulse
probe, JIT resolve, and 3 interleaved rounds (`natorder_calibrate.h:167-240`).

**2D proves the mechanism from the other side.** 2D natural undoes two commuting reversals.
`dim2` (within-row) is applied to the tile scratch immediately after the inner FFT, while the data
is L1-hot and already in the shape `cycle_pass` wants — it costs nothing measurable (16×64,
dim2-only 0.99×). `dim1` (whole matrix rows) has no such host and is a standalone strided pass —
1.49× at 64×16, 1.17× at 256×16; squares 1.22–1.60×, turning 2D from a 1.26–1.42× win over MKL into
roughly parity (`docs/roadmap/natorder_2d_status.md:99-131`, 2026-07-05). Split gets natural order
cheap exactly where it already owns a buffer to fuse into.

**The escape hatch is the thesis in miniature.** The natural-order verdict enum has 8 values, but
only 4 (`FREE`, `PURE_CYCLE`, `PSWAP`, `SCR`) are split mechanisms. `ZCASC` and `ILP` abandon the
split path entirely and attach an interleaved engine that emits natural order with **no reorder
pass** (`wisdom_reader.h:55-75`). The cheapest way to get natural order out of the split engine is
to not use the split engine — and the create-time race measured it: ZCASC beat the tape incumbent
**4.7–7.1×** at every cell (`docs/roadmap/cascade_natural_inplace_plan.md:140-148`, 2026-08-03).
Natural order also disqualifies split's fused `z -> z` adapters, which are gated on
`nat_mode == 0`, so an interleaved caller wanting natural order through split pays a layout
conversion *on top of* the reorder pass (`src/core/vfft.c:6131,5626-5636`).

### 5.2 Out-of-place on the split engine

**Mechanism.** MODEB — the workhorse kind, the one that carries every wisdom factorization —
reaches out-of-place through a **stage-0 graft**: stage 0 of a DIT plan is untwiddled in every
group, so run stage 0's `n1` codelet with `src` in and `dst` out (same strides, same group
geometry), then run stages 1.. unchanged in place on `dst`. `group_base` offsets are absolute, so
resuming at stage 1 needs no shifted sub-plan; output is bit-identical to the in-place run on a
copy (`src/core/oop/oop_execute.h:1-14,46-77`).

Split's out-of-place is therefore **not a placement the engine has. It is a property of one stage
that the planner may or may not have left untwiddled.**

**The construction is a whole parallel stack**, not a wrapper (all counts measured in-tree
2026-08-22, HEAD `d1ea8ff`):

| | split in-place | split out-of-place |
|---|---|---|
| engine headers | `core/engine/` 4,390 LOC | `core/oop/` split-specific 1,861 LOC |
| planner | `dp_planner.h` 822 LOC | `dp_planner_split_oop.h` 619 LOC |
| wisdom code / data | `wisdom2_stride_reader.h` 433 LOC / 312 cells | `wisdom2_oop.h` + reader 1,207 LOC / 92 cells |
| codelets (avx2) | 324 files, 383,548 LOC | 91 files, 133,956 LOC |
| direction coverage | 162 fwd / 162 bwd | 90 fwd / 1 bwd (pointer-swap dividend, §3.1) |

The two corpora have a **provably disjoint ABI**: 324/324 in-place files contain `rio_re` and 0
contain `in_re`; 89/91 OOP files contain `in_re` and 0 contain `rio_re` (the 2 exceptions are
interleaved monoliths). Zero shared entry-point symbols. The OOP form carries independent in/out
leg *and group* strides — an 11-argument shape — which is what lets a four-step turn be fused into
its stores.

**But that corpus does not serve the general-N out-of-place path.** MODEB binds
`st->n1_fwd = reg->n1_fwd[R]` (`src/core/engine/planner.h:99`), and in the shipped stride registry
that function pointer is this:

```c
if (in_re != out_re || in_im != out_im) {
    for (int j = 0; j < R; j++) {
        memcpy(out_re + j*os, in_re + j*is, vl*sizeof(double));
        memcpy(out_im + j*os, in_im + j*is, vl*sizeof(double));
    }
}
radix##R##_n1_##dir##_##isa(out_re, out_im, NULL, NULL, os, vl);
                                    /* registry_avx2.h:1101-1113 */
```

Copy, then the **six-argument in-place butterfly on the destination**. `oop_execute.h:62` calls it
with `src != dst` on every MODEB forward. `grep -c '_oop_' registry_avx2.h` = **0**: the stride
registry contains no genuinely out-of-place c2c codelet; `codelets/oop/` is bound only through
`oop_leaf_registry.h`, never through the stride executor. Stage 0 has `num_groups = N/R` groups,
each copying `R` legs of `slice_K` doubles, so the forward moves `N*K` doubles per plane — a
complete copy of both planes, issued as `N` memcpys of `K*8` bytes.

*Calibrate this correctly.* The copy is **fused into the same group loop** as the butterfly, so the
butterfly re-reads L1-hot bytes. The cost is one extra store stream plus an L1 re-load per group —
**not** a full extra pass over memory. And nobody has measured it (see below).

The **backward** copy is not fused and not hidden: `memcpy(dr, sr, NK*8); memcpy(di, si, NK*8);`
before an in-place DIF backward (`src/core/oop/oop_plan.h:912-913`). That one is **inherent given
the scrambled forward**: MODEB's output is digit-scrambled, so the pointer-swap identity — which
holds only for natural order — is invalid, and the only exact inverse is the in-place DIF backward,
which needs its input in the destination (`oop_plan.h:899-905`). The MT slice path does the same
copy as `2N` strided memcpys of `S*8` bytes, i.e. 64-byte memcpys at the minimum slab `S = 8`
(`src/core/vfft.c:2099-2103`). 2D..4D split OOP is memcpy-then-in-place outright, stated plainly in
the code (`src/core/vfft.c:6560-6562,6729-6732`) — there is no ND out-of-place engine.

**Three structural limits on top.**

1. **The graft refuses two plan classes.** `if (plan->use_dif_forward) return -1;` then a per-group
   `needs_tw` check (`oop_execute.h:52-57`), because in a DIF-oriented plan stage 0 carries
   twiddles and no twiddled codelet has a destination argument. Of 312 banked in-place c2c cells,
   **80 (25.6 %) carry `dif=1`** (`generated/wisdom2_stride.txt`, 2026-08-20). Those cells still
   run out-of-place — `oop_auto.h:9-11` builds them DIT regardless — so the honest statement is:
   **26 % of banked in-place cells are served by a non-champion dataflow when run out-of-place,
   and the size of that penalty is unmeasured.**
2. **Reach.** The OOP twiddled-mid registry covers 7 radices `{4,7,8,13,16,32,64}` against the
   in-place registry's 19 (`oop_leaf_registry.h:63-79`). BAILEY2's rule spine needs *both* an OOP
   leaf and an OOP twiddled mid, so the 7-radix mid set is the binding constraint on how often the
   natural-order OOP kinds fire at all; everything else drops to MODEB, i.e. to the graft.
3. **Prime `N` is refused outright.** `vfft_create` warns and returns NULL: *"the OOP kinds need a
   radix factorization of N; prime and other Rader/Bluestein-class sizes are served IN-PLACE
   only"* (`src/core/vfft.c:5246-5250`). The in-place engine serves those through `core/primes/`.

**At K = 1 the tax inverts.** Nine of the ten raced split routes are `-ip`
(`dp_planner_split_oop.h:105-114`, `VFFT_SP_RAXIS = {0,1,1,1,1,1,1,1,1,1}`), and every BAILEY2V
route allocates `p->col_re`/`p->col_im` at `N` doubles each (`oop_plan.h:401-402`) — **2N doubles
of scratch, exactly the size of `il2p`'s `mid` and `zturn`'s `plane`**. The entry points say so:
*"no transpose sweep, 2 passes, 1 scratch pair; `dst == src` is safe for both routes since the t1
pass reads only the scratch"* (`oop_plan.h:823-825`). At K = 1 split has already bought the
interleaved engine's decoupling buffer, so out-of-place costs nothing beyond re-pointing `dr/di` —
`src/core/vfft.c:6310-6330` dispatches the identical route function for both placements. What is
missing there is not work but an **axis**: the K=1 wisdom line is keyed on `N` alone and ships
`place=*`, the planner races both placements and prints both, and `win_oop` is discarded
(`dp_planner_split_oop.h:578-600`). 18 of 92 `wisdom2_oop.txt` cells are `eng=k1`, all `place=*`.
So K=1 out-of-place is served by a route that was raced *in-place*.

**Inherent vs incidental — this is the most-mislabelled cost in the system.**

| cost | class | why |
|---|---|---|
| twiddled codelet has no destination argument | **inherent** | direct consequence of Method C (§3.1); it is why the graft must exist and why it can `return -1` |
| MODEB backward's two-plane memcpy | **inherent given scrambled output** | swap identity requires natural order (`oop_plan.h:899-905`); a natural-order OOP kind escapes it entirely |
| the stage-0 **memcpy wrapper** | **incidental** | nothing forbids emitting a genuine 7-arg untwiddled `n1` that reads `src` and writes `dst`. The same emitter already produces 91 eleven-argument out-of-place codelets for a different consumer. Migration debt wearing an inherent costume. |
| 2D..4D copy-then-in-place | incidental | no ND OOP engine has been built; the copy is a placeholder, not a derivation |

**No number exists for this tax.** There is no measurement anywhere in the repo of split
out-of-place against split in-place at the same cell. The two `v1_0_results.md` tables are ratios
against *MKL's* in-place and *MKL's* out-of-place respectively and cannot be divided; the K=1
table's own footnote (double-dagger) says its sub-2048 columns measure the IL axis, not split. The
one place a split IP-vs-OOP delta is computed writes it to stdout and drops it. **State the split
out-of-place tax as structural and unmeasured.**

Three residual percentages circulate for this path. Quote them only with their caveats: 5–6 % for
the missing OOP-aware tier-1 fast path — its cited source `docs/61` **does not exist** in the tree
(`src/core/engine/executor.h:10-12`); 15–20 % transposed-intermediate tax at single-codelet `N`,
which is why LEAF is a rule and not a race (`oop_plan.h:6-8`); 6–8 % BAILEY2 divisor-pair order
residue (`oop_plan.h:30-31`). None carries a vintage, protocol, or artifact.

The only banked figure that *sizes* a copy on this host is MKL's, from a `memcpy 2N doubles`
control arm run under the full protocol (pinned core 2, HIGH priority, 17 rounds, alternating arm
order, 200 ms pace, 32 MB cachebust, one arena with 64-B-skewed planes): 56.3 ns @ N=1024, 423.1 @
2048, 847.3 @ 4096, 2,828.3 @ 8192, 9,580.3 @ 16384, 23,100 @ 32768, 48,583.3 @ 65536. Source
`docs/research/mkl_inplace_campaign/results/mkl_ip_timed_census.txt`, campaign closed 2026-08-02 —
**untracked** (`.gitignore:100` ignores `/docs/research`), and it is MKL's K=1 interleaved harness,
not our split engine. Use it to scale MODEB's backward copy (which moves exactly `2*N*K` doubles)
and for nothing else.

### 5.3 Out-of-place on IL

**At execute, the tax is one ternary.** The in-place branch normalizes the destination and calls
the same engine function on the same plan:

```c
double *zo = dre ? dre : (double *)sre;         /* vfft.c:6801  (in-place)     */
vfft_il2p_execute_fwd(h->k1il2p, sre, zo);
...
vfft_il2p_execute_fwd(h->k1il2p, sre, dre);     /* vfft.c:6875  (out-of-place) */
```

Same plan object, same two kernel calls. The cascade is blunter still: `_exec_zcascade` takes
`(sre, dre)` and never inspects `h->placement`. There is no placement branch anywhere below the
dispatcher, no second plan, no second buffer, no second codelet. `grep memcpy` over `il2p.h`,
`zturn.h`, `zsplit.h` returns **zero**.

**One corpus serves both.** Every IL stage takes `(const in, out)` through a single 11-argument
typedef (`il2p.h:68-70`), and 282/282 zil codelets match it exactly — 282/282 declare
`const double * __restrict__ zin` first, 0/282 declare an aliased `rio_*` pair, and 0 carry a
placement or gauge suffix. The kind names there (`n1`, `n1t`, `t2`, `t2t`, `t2tg`) are **math**
kinds — which butterfly, where the twiddle sits, whether the store is turned — not placement
gauges. Contrast the split OOP corpus, which carries 12 distinct gauge/variant tags (`UG_UG`,
`UG_UG_log3`, `UL_UG_twl`, `UG_UL`, …): the UL/UG axis is load-gauge vs store-gauge, because
split's OOP path must fuse its transpose into one side or the other and pays a variant for each
choice. IL's turn is fused into the stores once, in the math, so there is nothing to gauge. Two
vestigial arguments (`zin_unused`, `zout_unused`) preserve the split ABI's plane-pair shape for a
layout with one plane — 2 dead arguments per call, the only incidental item in the ABI.

**The one real cost: +1 plane of working set.** The engine always routes through its owned scratch
regardless of placement (`mid` allocated unconditionally, `il2p.h:667`). So:

```
in-place        {z, mid}          = 2 planes x 16N bytes
out-of-place    {in, mid, out}    = 3 planes x 16N bytes
il3p            3 planes in-place (mid1, mid2), 4 out-of-place
```

`il2p.h:21-24` writes its crossover analysis for the **out-of-place** case and names that third
plane: *"N=1024 is in+mid+out = 3*16 KB = 48 KB = exactly this machine's L1d, and that cell measures
a dead wash — the crossover sits precisely where the mechanism predicts."* So IL out-of-place is not
paying instructions or copies; it is hitting the L1 wall at 2/3 of the `N` its in-place twin would.
Note the comment says *wash*, not loss.

And note §3.2: that third plane is **avoidable in principle**. `mid` is needed only when
`zin == zout` — precisely the case in which it is free (2 planes total) — and is carried in the
out-of-place case where it buys nothing. No mid-free out-of-place arm exists in the tree, so this
is a derivation, not a result. It is also the most concrete piece of headroom this document
identifies (§8).

**Not measured.** There is no valid same-vintage IL in-place-vs-out-of-place number sub-2048, and
the repo says so: `v1_0_results.md`'s double-dagger footnote marks the sub-2048 in-place column
*"pre-tangent plan … the banked kind-3 row for this N CHANGED on 2026-08-12, so the in-place figure
no longer describes what ships … it has not been re-measured, and is not quoted as if it had"*. The
apparent sub-2048 in-place penalty (0.91 vs 1.05 at 128; 0.85–0.86 vs 1.00 at 256; 0.78–0.80 vs
0.98–1.00 at 512) is a **vintage artifact**: the in-place rows are 2026-08-04/08-06 and the OOP rows
are 2026-08-16 (the closed tangent/wing32/TURNED campaign). Structurally they cannot differ — both
placements attach the same `il2p` object through the same `_k1_il2p_apply_kv`, consuming the same
banked pair and `il_kv` verdicts. The N>=2048 rows *are* same-run, and show in-place ~5 points above
OOP at every N (2048: 1.09–1.16 vs 0.99–1.11; 4096: 0.96–0.99 vs 0.91–0.94; 8192: 1.00–1.03 vs
0.95–0.98; 16384: 1.02–1.03 vs 0.94–0.98; 32768: 0.94–0.97 vs 0.88–0.91) — but these are ratios
against MKL, and the repo attributes the offset to MKL's own strength out-of-place, not to us.

**Coverage inverts here, twice** (developed in §7):

* IL out-of-place **serves prime N** via Rader/Bluestein on IL inner plans (`il_prime.h`, route
  `VFFT_K1_IL_PRIME`, attached at `src/core/vfft.c:4869-4881`, band `[5, 2048]`) — exactly the
  cells split out-of-place refuses at create. The K=1 handle returns at `vfft.c:5127`, before the
  refusal at `:5246`. `include/vfft.h:383-387` still says those sizes are "REJECTED loudly …
  served IN-PLACE only"; that line is **stale** for `layout=INTERLEAVED`.
* IL's *constrained* placement is **in-place**, not out-of-place (§5.4).

### 5.4 In-place, and scrambled order, on IL

**In-place is structurally free, and the signatures prove it.** No IL create function takes a
placement parameter: `vfft_il2p_create(int N, int R1, int R2)` (`il2p.h:635`) always allocates
`mid` (`:667`); `vfft_zsplit_create(int N, const int *chain, int nf)` (`zsplit.h:143`) always
allocates `sp` (`:198`); `vfft_zturn2_create_chain` always allocates `plane` (`zturn.h:629`). The
buffer is demanded by the math, not by aliasing, so in-place adds zero allocation, zero passes,
zero copies.

Alias safety is asserted by **construction**, one invariant per engine: *"each stage fully consumes
its input before the next writes"* (`il2p.h:866`); *"the terminator is the only writer of `zout` and
reads only the scratch plane"* (`zsplit.h:16-17`); *"fwd `zout` is written only by `stf`, which
reads only the plane; bwd `zin` is read only by `stfb`, the first stage"* (`zturn.h:43-45`). The
invariant is fragile and the code knows it — `zturn.h:976-977` carries a warning that fusing `s0tb`
into the tile loop would break in-place.

**In-place is in fact *better* than out-of-place for IL**, by exactly one plane (§5.3). That is
what puts the tier gate where it is: at N=1024 in-place is 2 × 16 KB = 32 KB (inside L1d), at
N=2048 it is 64 KB (outside), and the shipped ILP gate is a literal `N < 2048`
(`src/core/vfft.c:4064,4519`). Derivation from a banked measurement, not a separate measurement:
no experiment isolates two-plane vs three-plane.

**Where IL in-place is genuinely incomplete: one route, and the blocker is a type qualifier.**
`_k1_il_candidate` comments *"MONO is deliberately absent — its kernels are `__restrict__` and
refuse aliasing"* (`src/core/vfft.c:2521-2523`). Reading the generated body shows this is a
declaration, not a dataflow hazard: `vfft_k1_mono64_il_bwd_avx2.c` loads both input blocks (`:51`,
`:228`) and only then stores both output blocks (`:544`, `:723`) — all loads precede all stores.
The monolith already owns a full-N scratch; it just lives on the stack frame rather than in a plan
field. The hole is **one cell wide**: `vfft_k1_mono_il_fn` resolves only `N == 64`
(`oop_leaf_registry.h:352-365`). At N=64 out-of-place routes to `VFFT_K1_IL_MONO` while in-place
routes to the `il2p` 8×8 pair — **a placement-dependent engine divergence**, whose cost is not
measured anywhere (every sub-2048 table starts at 128). Lifting the fence is a codelet-signature
change on a spilling monolith: removable at a price nobody has measured.

**A real remaining hole: scrambled in-place is HIT-ONLY.** The Phase-B3 attach requires a banked
`@nat` record with `mode == VFFT_NAT_ILP`, and the comment states the rule: *"hit-only keeps `@nat`
single-writer (only NATURAL creates measure/bank)"* (`src/core/vfft.c:4514-4527`). So a program
that only ever asks for SCRAMBLED in-place interleaved below 2048 never causes anyone to measure,
never banks, and is served forever by the convert path — the path the ILP race beat by
**9.1× / 7.2× / 5.7× / 4.0×** at N = 128/256/512/1024 (`docs/roadmap/il_coverage_plan.md:64-70`,
create race, 2026-08-03). The single-writer discipline is deliberate wisdom hygiene, but the
measured win is realised only by callers who created a NATURAL handle at that cell at some point.

**Ordering, Bailey tier: zero cost and zero benefit.** Because the turn *is* the reordering,
`il2p`/`il3p` emit natural order as their only order, and a SCRAMBLED request below 2048 is
honoured by the same engine with the identity permutation — which the scrambled contract admits.
The gate proves it as **bit equality**, not tolerance: `scr == nat` memcmp-EXACT at 128–1024 at
0.95–1.05× the speed (`il_coverage_plan.md:35-39`, `vfft_k1scr_gate.c`, 2026-08-03; the results
table records this as "= NAT bits"). Worth stating plainly: a scrambled caller at this tier
surrenders a freedom the engine cannot spend.

**Ordering, cascade tier: this is where the ordering tax lives.** The cascade's native output is the
digit-reversed comb. Natural is *purchased* by `vfft_zturn2_set_natord`, which builds `rho` /
`rho^-1` block tables the terminator addresses its **loads** through, and forces `p->tfuse = 0`
because `rho` spans a section (`zturn.h:343-370`).

The mechanism is a load/store asymmetry, measured before the design: permuting the terminator's
64-byte **stores** costs +29–50 % of the pass, while permuting its **reads** is free (0.96–1.12×)
(`docs/research/natterm_spec.md:9-15`, P0c, `zturn_natscatter_probe.c`; untracked). Natural order is
therefore bought by moving an *index* on the load side of a kernel that reads the hot scratch plane
and stores contiguously into user memory — same arithmetic, same shuffles, same store instructions,
one indexed table load per four columns.

The B4 falsifier holds everything else constant and prices it:

| arm | forward | backward |
|---|---|---|
| cascade + in-place 64-B-block PURE-cycle reorder pass | **+13.4 … +23.5 %** | +17.4 … +26.6 % |
| cascade with the natural **terminator** | **+2.5 … +5.7 %** | +3.0 … +4.9 % |

Nine arms in one run, same-plan control S'/S = 0.996–1.020, 17 paced rounds, alternated order,
pinned core 2 HIGH, two runs, ±3 % thermal drift
(`docs/roadmap/cascade_natural_inplace_plan.md:130-138`, 2026-08-03; raw at
`docs/research/natord_falsifier_b4_run2.txt`, untracked). End-to-end confirmation the next day
through the canonical bench: natural-vs-scrambled delta **2–6 %** at N = 2048…32768, 3 runs/cell,
cell-per-process, pinned core 2, per-cell spread 1–3 %, cross-engine elementwise error ~7e-16
against MKL (`docs/performance/zturn_cascade_tiling.md:340-365`). The tiled arms came out at parity
(Nt/St = 0.98–1.04), so the `tfuse` legality the natural contract gives up costs ~0 in practice.

The falsification that makes this a *mechanism* rather than a result: DIT-natural, which moves the
`rho` scatter to the ingest's **reads on user memory** instead of the hot plane, **lost at every
cell** (D/N = 1.05–1.18; 1.26–1.37 against tiled DIF-natural), and a store-side-permuted ingest
reclaimed nothing (D2/D = 0.92–1.06) (`cascade_natural_inplace_plan.md:191-217`).

**Read the two tiers together and the symmetry is exact.** Split pays +13–30 % for natural order
because it must run a whole extra pass. The cascade pays +2.5–5.7 % because it only has to index a
load it was already performing. Same permutation, same cells, same run, ~4× different price — and
the entire difference is which engine already owned a buffer to fuse into. **B4 is the only
coverage-immune evidence in this document that the founding choice, not the feature set, is doing
the work.**

**Coupling on the real path.** For `zr2c`, ordering and placement are not independent. The forward
fold pairs bin `f` with bin `N/2 - f`, reads both windows before writing either (loop guard
`2*(f+3) < half`), and saves DC/Nyquist into locals up front — so it is in-place-safe **by
involution** (`src/core/transforms/real/zr2c.h:46-50,86-160`). The obvious optimisation — consume
the cascade's scrambled-native output directly — exists as `_zr2c_fold_*_perm`, and the header
refuses in-place for it with the mechanism spelled out: *"reads scattered mirrors after writes would
collide across slot order"* (`zr2c.h:278-293`). Scrambling the slots destroys the involution the
alias-safety rests on. Those perm folds are gated and benched but appear in **no production path**;
the production child is always created with `order = VFFT_ORDER_NATURAL` (`src/core/vfft.c:2206`),
so real transforms at N >= 4096 pay the cascade's natural-terminator tax. Alias-safety of the fold
is necessary but not sufficient — the composite pays too: route 0 (OOP-IL child) allocates a second
`N+2`-double plane to serve in-place R2C, route 1 (NAT-IP cascade child) pays an `N`-double memcpy
when `dre != sre` (`src/core/vfft.c:2218-2295`).

---

## 6. Why there are three tiers

The tiers are not a taxonomy imposed on the size axis. Each fence has a **different physical
mechanism**, and they do not coincide.

```
   N:   ...64        128 ......... 1024        2048 ....................  8192 ...
        [ mono ]     [ Bailey two-pass (il2p / il3p) ]  [ CT cascade (zturn/zsplit) ]
             ^                                     ^                            ^
     register file                          working set                   registry reach
```

**Fence 1 — the register file (~64).** A monolith has zero pass overhead and still loses above 64,
because `N`-sized function-scope state on 16 `ymm` registers becomes a spill storm. For the
interleaved axis the mono tier is literally `{64}`: `vfft_k1_mono_il_fn` resolves `N == 64` and
nothing else (`oop_leaf_registry.h:352-365`), and the DP can only offer `VFFT_K1_IL_MONO` when that
resolver answers (`dp_planner_il.h:1046-1050`). The same register-file argument is the shipped
default for R >= 32 blocked kernels: *"a monolithic R>=32 body holds ~40–64 live values against
AVX2's 16 registers"* — a structural rule, not a per-cell race, worth -27 % at 1024 and -8 % at 512
in a same-run A/B with the pair pinned by wisdom (`v1_0_results.md`, triangle footnote, 2026-08-06).

**Fence 2 — the working set (2048).** This is the derivation in §5.3/§5.4, and it is the strongest
form of evidence available for a boundary: not *"we measured a boundary"* but *"we predicted a
boundary from a working-set identity and the measurement landed on it."* Out-of-place at N=1024 is
`in + mid + out = 3 × 16 KB = 48 KB`, exactly `VFFT_L1D_PCORE_BYTES`
(`src/core/support/cpu_cache.h:15-16,69`, cross-checked against the Raptor Cove 48 KB / 12-way
geometry), and that cell measures a **dead wash**: pure IL vs the hybrid two-pass route, 0.558× @64
(60.4 → 33.7 ns), 0.765× @256 (248.5 → 190.2), 0.956× @1024 (1796.4 → 1717.3) — `il2p.h:12-16`,
vintage in-header 2026-07-26, both arms gated against a scalar DFT. In-place at 1024 is 32 KB and
sits inside L1; in-place at 2048 is 64 KB and does not. The shipped ILP gate is `N < 2048`.

Above the fence, only the cascade amortizes: two passes cannot pay back a layout conversion,
log-many passes can (`il2p.h:24-27`). That is a statement about pass count, not about size, which is
why the same engine family cannot serve both bands with one shape.

**Fence 3 — registry reach (8192), which is NOT the working-set fence.** These two are frequently
conflated; they are independent. The natural-IL DP candidate census collapses 23 / 8 / 1 / 0 / 0 /
0 / 0 at N = 1024 / 2048 / 4096 / 8192 / 16384 / 32768 / 65536, and the stated reason is pair
coverage: *"At 4096 the only legal pair is 64×64, and above that no pair of powers of two both <= 64
can multiply to N"* — a consequence of the IL registry stopping at R = 64, itself a register-file
limit (`docs/search_space/il_c2c.md:251-274`; counts declared measured, **no run date stamped** —
vintage unverified). That wall bites at 8192. The working-set wall bites at 1024 (out-of-place) /
between 1024 and 2048 (in-place), where candidates still exist in quantity but *lose*. Quoting the
census as confirmation of the working-set story overstates it.

**The fences are planner gates, not capability limits — deliberately.** Nothing in `il2p` refuses
N >= 2048; its create refuses only on factorization and registry availability (`il2p.h:635-645`).
The boundary is `_vfft_zcasc_min_n()`, whose default is 2048 and which reads `VFFT_NAT_ZCASC_MINN`
so that the boundary itself can be raced (`src/core/vfft.c:93-117`). Accordingly, cold-start seed
chains exist at 1024 and 32768 *only* so the tier boundary is raceable — 1024's is explicitly
*"below the production ZCASC gate … unreachable unless `VFFT_NAT_ZCASC_MINN` lowers it"*
(`zsplit.h:88-98`). **The tiering is falsifiable by construction.** A gate that is never crossed can
never be shown to be in the right place.

> Minor inconsistency worth fixing: the cascade floor is env-overridable but the two ILP attach
> sites hard-code `N < 2048` (`src/core/vfft.c:4064,4519`). *Lowering* the cascade floor is harmless
> (both candidates exist and race); *raising* it opens a band where neither attaches and in-place
> interleaved silently falls back to convert.

---

## 7. The honest counter-argument: how much of IL's cheapness is coverage?

A large part. State it in full before claiming anything.

**IL's native out-of-place reach is ~600 of 4095 sizes in [2, 4096], at K = 1 only.** Executing the
create-time predicates over the shipped registry: the `il2p` pair search needs `N = R1*R2` with
`R2` in [4,64], `R1` in [3,64], both radices in the registry's 20-radix set
`{3,4,5,6,7,8,9,10,11,12,13,15,16,17,19,21,25,27,32,64}` — **170** of 4095 (all powers of two from
16 to 4096; 168 below 2048). `vfft_il3p_default_chain` reaches 170 more, of which **121** are new.
`vfft_ilprime_create` covers primes in [5, 2048] — **307**. Union ~ **600**; **1488 of the 2046
sizes below 2048 have no native IL route and convert.** Beyond `N`: IL converts for lane-major
`K > 1` and for 2D..4D, and **rejects padded batches outright** (*"config.batch + layout=INTERLEAVED
is unsupported — padded batches are split-plane by construction"*, `src/core/vfft.c:3160-3165`).
Split MODEB, by contrast, is general-N through the stride executor at any K, and serves padded
batches. Computed 2026-08-22 by executing the predicates over the shipped registry.

So every "IL in-place and out-of-place are both ~free" claim is measured over roughly 15 % of 1D
sizes at a single batch width. That is a real and large qualification and it belongs on page one,
not in a footnote.

**But coverage does not explain the mechanism, and it inverts in three places.**

1. **IL out-of-place covers *more* than split out-of-place at prime N.** Split OOP refuses those
   cells at create with a warning and a NULL handle; IL serves them natively through `il_prime.h`,
   explicitly *"NOT a memcpy wrapper"* (`il_prime.h:1-9`; attach at `src/core/vfft.c:4869-4881`,
   whose own comment reads *"the OOP INTERLEAVED prime coverage the split OOP path refuses"*).
2. **IL's constrained placement is in-place, not out-of-place.** Out-of-place is IL's *default*
   route; in-place requires K == 1, no padded batch, `layout == INTERLEAVED`, `N < 2048`, no cascade
   attached, `!recalibrate`, and a wisdom hit on `VFFT_NAT_ILP` — and on the natural path it must
   additionally **win a create-time race** against the convert incumbent
   (`src/core/vfft.c:4064-4081,4514-4527`). In-place was bolted on in Phase B, 2026-08-03;
   out-of-place was native from the start. So the axis on which IL "looks cheap" is not the axis the
   library had to construct for it.
3. **B4 holds coverage constant.** Same cells, same permutation, same run, two mechanisms:
   +13.4–23.5 % for the reorder pass against +2.5–5.7 % for the fused terminator (§5.4). Nothing
   about N-reach, K-reach or feature set varies between the two arms.

**Verdict the evidence supports.** Coverage explains most of the *aggregate engineering-cost* gap —
1,861 vs 4,390 engine LOC, 91 vs 324 codelets, two planners, two wisdom stores, one corpus vs
two-plus-twelve-gauges. It explains approximately **none** of the *per-cell* tax gap that B4
isolates. Lead any argument about founding choices with B4, never with a ratio-against-MKL table.

---

## 8. What this means for future work

**Structural — do not attack.**

* Split's scrambled output and in-place-only twiddled codelet. Both follow directly from Method C
  (§3.1). The fused single-buffer natural-order scatter is *provably* impossible
  (`natural_order_inplace_design.md:221-229`); the reorder-pass kernel is at its measured ceiling
  with four refuted attacks behind it (§5.1).
* MODEB's backward copy, *given* a scrambled forward. The escape is not a faster copy; it is a
  natural-order OOP kind, which is BAILEY2, which needs the OOP twiddled-mid registry widened
  (below).
* IL's two-plane floor. You cannot four-step without a destination.
* The interleaved `cflip` per complex multiply. It is why the cascade's interior is split (§3.3),
  and that decision was raced, not assumed.

**Worth attacking, in rough order of leverage.**

1. **Emit a genuine out-of-place untwiddled `n1` for the stride registry.** The stage-0 memcpy
   wrapper (`registry_avx2.h:1101-1113`) is migration debt, not physics — the same emitter already
   produces 91 eleven-argument out-of-place codelets for a different consumer. This is the largest
   mislabelled "inherent" cost in the system. It needs a number first (item 6).
2. **Give the K=1 split wisdom line a placement axis.** The planner already races both and prints
   both; `win_oop` is discarded and the record ships `place=*` (`dp_planner_split_oop.h:578-600`).
   Today K=1 out-of-place runs a route that was raced in-place. The store's own legend reserves
   `sp_kv` for exactly this class of widening.
3. **A mid-free out-of-place `il2p`.** §3.2 shows only the leaf is turned, so a strictly
   out-of-place call could write leaf -> `zout` and run `t2` in place on `zout`. That removes the
   third plane and moves IL's out-of-place L1 crossover from N=1024 upward. It is the one change
   this document identifies that would move a *fence* rather than a constant. Build it as an arm
   and race it; do not assume it.
4. **Widen the OOP twiddled-mid registry** from 7 radices toward the in-place 19. That is the
   binding constraint on how often BAILEY2 (natural, out-of-place, no graft, no backward copy) can
   fire instead of MODEB (§5.2).
5. **Retire the SCR carrying cost** or re-race it: ~180 lines plus wiring for a mode with zero
   banked cells and a compile flag defined nowhere (§5.1).
6. **Measure the two holes this document could not fill**: split in-place vs split out-of-place at
   the same cell (the only place it is computed, stdout, drops it), and the N=64 mono-vs-`il2p` 8×8
   divergence (a placement-dependent engine choice with no datum, §5.4).

**Where a new engine should sit.** Ask two questions, in this order.

1. *Does it own a destination buffer?* If yes, fuse every permutation you need into it — but only
   where the permutation lands on the **load** side of the hot pass: P0c measured permuted stores at
   +29–50 % and permuted loads at free (§5.4). If no, expect to construct both a placement and an
   ordering, and budget for a parallel planner, a parallel wisdom store, and a corpus.
2. *How many passes will amortize the buffer?* Two cannot pay back a layout conversion; log-many
   can (`il2p.h:24-27`). That single question, not the size axis, is what separates the Bailey tier
   from the cascade tier — and it is why the cascade's interior is Method C on a plane rather than
   more four-step.

**Do not** build hybrid IL-boundary / split-interior *codelets*. The test is the signature: a
codelet taking `in_re` **and** `in_im` inside an interleaved route is a hybrid, and the pattern has
been refuted twice. Note the distinction the cascade draws: converting at the *route* boundary and
running a split interior is exactly what `zsplit` does, and it is the measured winner at N >= 2048
(§3.3); what is refuted is mixing the two layouts *inside* one kernel's ABI.

---

## Appendix A — comments that have outlived their code

A document that teaches "prefer the code" should ship the list. Each of these is currently wrong.

| location | says | reality |
|---|---|---|
| `registry_avx2.h:1096-1099` | *"no current caller hits that — the executor always passes in==out"* | `oop_execute.h:62` passes `in != out` on every MODEB forward |
| `src/core/oop/README.md:35` | MODEB stage 0 runs OOP *"— no extra copy"* | stage 0 **is** the copy (`registry_avx2.h:1101-1113`) |
| `oop_plan.h:34-36` | *"Backward for every kind: pointer-swap identity"* | contradicted by its own file at `:899-921` (MODEB memcpys) |
| `include/vfft.h:383-387` | prime OOP *"REJECTED loudly … IN-PLACE only"* | true for split; **false for `layout=INTERLEAVED`** (`vfft.c:4871`, handle returns at `:5127`) |
| `natorder_exec.h:3` | *"SCRATCH terminator + LEAF-IP arrive in P1b"* | SCRATCH landed then was deactivated; LEAF-IP was ditched |
| `src/core/vfft.c:3998-4000` | *"SCR/LEAF-IP still degrade to PURE until their executors land"* | SCR's executor landed; LEAF-IP was removed. Current statement: `natorder_calibrate.h:1-3` |
| `src/core/vfft.c:4497-4500` | *"Mono/Bailey IL tiers stay OOP-only until their alias-safety is verified"* | true for MONO only; the next block attaches Bailey in-place (`:4514-4527`) |
| `src/core/engine/executor.h:10-12` | 5–6 % figure *"per the spike measurements in docs/61"* | `docs/61` does not exist in the tree |
| six sites in `core/engine/` | point at `src/core/executor.h` | that path does not exist; the file was `engine/stride_executor.h`, removed 2026-09-01 as a zero-includer stale generation |

**Where the doctrine lives.** the header of the since-removed `src/core/engine/stride_executor.h` was the clearest statement of
the Method-C contract in the tree — and it is **included by nothing**. `src/core/vfft.c:13-14` says
why: *"we don't include stride_executor.h (it redefines executor symbols)"*. The shipping split
engine is `plan_executors.h` (the real `stride_plan_t`), `twiddle.h` (layout + Method-C twiddles),
`executor_generic.h` (the cold-cell loop) and `executor.h` (tier-1 dispatch). The prose in the dead
header is faithful — `twiddle.h:31-44` reproduces the `dim_stride` formula and `:81-107` the
`cf0`/`per_leg` factorization — so quote the doctrine from there and **cite the shipping files for
the mechanism**, or readers will patch a file that is not compiled.

## Appendix B — `__restrict__` contracts the code does not honour

"IL in-place is safe by construction" is a dataflow argument that the type system does not license.
Two live instances, both currently correct and gated, both worth naming:

* **Cascade mids.** `zsplit.h:228-236` calls `f(p->sp, 0, p->sp, 0, …)` and `zturn.h:797-805` does
  the same with `p->plane`, while the `msg` codelet declares
  `const double * __restrict__ zin, double * __restrict__ zout`
  (`codelets/zil/avx2/boundary_split/radix4_z_msg_avx2.c:19-21`). The same address bound to two
  restrict-qualified pointers is UB by the C contract. It works because each iteration loads all `R`
  lanes of a group before storing any. The signature is honest; the **call** is the exception.
* **`zr2c` folds.** Declared `__restrict__` on both pointers while the header two lines above says
  *"X may alias Z (in-place)"*, and `_exec_zr2c` calls them aliased in three of its four shapes
  (`zr2c.h:48-50`; `src/core/vfft.c:2259,2264,2280,2292`). The scalar tail (store `o[2*f]`, then
  load `z[2*m]` on the next iteration) makes a hoist reachable in principle.

A related recorded trap on the same path: gating the route-1 copy on `placement` instead of on
`dre != sre` made an in-place plan called with a distinct `dre` transform stale data at relerr
1.000, **silently** (`src/core/vfft.c:2270-2278`).

## Appendix C — provenance of every measured figure quoted here

| figure | source | vintage | notes |
|---|---|---|---|
| pure IL vs hybrid 0.558× / 0.765× / 0.956× | `src/core/oop/il2p.h:12-16` | 2026-07-26 | both arms gated vs scalar DFT |
| z-interior vs block-split -8.9 % / -12.6 % | `docs/roadmap/z_cascade_plan.md:325-328` | 2026-07-24 | identical-chain control |
| natural-order winner map (0 %, -7.3 %, +18.5…+30 %) | `natural_order_inplace_design.md:367-378` | 2026-07-04/05 | T11; **K > 1 batched**; §2d's range is superseded |
| kernel-campaign refutations (Stockham 1.86–2.40×, cell-transpose +45…+172 %) | same, `:180-209,374-390` | 2026-07-04/05 | |
| SCR 79.5 vs PURE 28.8 us @4096/4 | `natorder_calibrate.h:148-165` | 2026-07-05 | forced-mode, paced + locked |
| PSWAP/cycle 4.68× @64×16, tie at >=512 B rows | `natorder_2d_status.md:133-152` | 2026-07-05 | no FFT in the loop |
| 2D dim2 0.99× vs dim1 1.49× | `natorder_2d_status.md:99-131` | 2026-07-05 | |
| natural vs MKL 1.23–2.72×, median ~1.68×, 9/9 | `v1_0_results.md:231-262` | v1.0 | live host; directional; correctness exact |
| ZCASC vs tape 4.7–7.1× | `cascade_natural_inplace_plan.md:140-148` | 2026-08-03 | create-time race |
| B4: +2.5–5.7 % vs +13.4–23.5 % | `cascade_natural_inplace_plan.md:130-138` | 2026-08-03 | control S'/S = 0.996–1.020; raw file **untracked** |
| end-to-end natural delta 2–6 % @2048–32768 | `zturn_cascade_tiling.md:340-365` | 2026-08-04 | pinned core 2 |
| P0c stores +29–50 %, reads 0.96–1.12× | `docs/research/natterm_spec.md:9-15` | pre-2026-08-03 | **untracked** |
| DIT-natural D/N = 1.05–1.18 | `cascade_natural_inplace_plan.md:191-217` | 2026-08-03 | |
| ILP vs convert 9.1× / 7.2× / 5.7× / 4.0× | `il_coverage_plan.md:64-70` | 2026-08-03 | create race |
| `scr == nat` memcmp-EXACT, 0.95–1.05× | `il_coverage_plan.md:35-39` | 2026-08-03 | bit equality, not tolerance |
| N>=2048 in-place vs OOP ratios vs MKL | `v1_0_results.md:304-310` | 2026-08-16 (OOP), mixed (IP) | ratios **against MKL**; not our tax |
| sub-2048 in-place column | `v1_0_results.md`, double-dagger footnote | pre-2026-08-12 | **VOID** for IP-vs-OOP comparison |
| blocked R>=32: 1024 -27 %, 512 -8 % | `v1_0_results.md`, triangle footnote | 2026-08-06 | same-run A/B, pair pinned by wisdom |
| memcpy 2N doubles, 56.3 → 48,583.3 ns | `docs/research/mkl_inplace_campaign/results/mkl_ip_timed_census.txt` | closed 2026-08-02 | **untracked**; MKL's harness, not ours |
| IL candidate census 23/8/1/0/0/0/0 | `docs/search_space/il_c2c.md:253-262` | **no date stamped** | vintage unverified |
| corpus / wisdom counts (324 vs 91 files, 312 vs 92 cells, 80 `dif=1`, 90 fwd / 1 bwd, 282 zil) | in-tree counts | 2026-08-22, HEAD `d1ea8ff` | reproducible by `grep`/`find` |
| IL coverage ~600 of 4095 | create-time predicates executed over the shipped registry | 2026-08-22 | reproducible |

**Figures deliberately NOT quoted**, and why: the three-tier ladder (mono 30 ns @64, Bailey 0.54×
@2048, cascade 0.46 -> 0.81×) — its cited source `docs/performance/k1_optimization_campaign.md`
does not exist in the tree; the 5–6 % tier-1 gap — source `docs/61` does not exist; the 15–20 %
transposed-intermediate tax and the 6–8 % pair-order residue — in-source assertions with no
artifact, protocol, or date; the "-21 % to +36 %" natural-order range — retracted by its own
document at `natural_order_inplace_design.md:365`.
