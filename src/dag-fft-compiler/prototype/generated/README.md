# generator/generated/

Auto-emitted headers, regenerated and promoted into the source tree by
`dune build` (the `(mode promote)` rules in `dune`). **Do not edit these by
hand** — they are overwritten on the next build. Each one is emitted from
`generator/lib/coverage.ml` (the single source of truth for codelet coverage),
so a coverage change re-emits the matching registry automatically. The one
exception is `registry.h`, which is hand-written (see below).

The norm across all registries: **one ABI-typed slot per codelet identity**
(kind × variant × direction × ISA). The slot's function-pointer type *is* that
codelet's ABI, so wiring a wrong-ABI codelet is a compile error. Slots for
codelets a given ISA does not generate stay NULL (caller `calloc`s the struct).

## Codelet registries (one struct + one `*_register_all_<isa>()` each)

### `registry_avx2.h` / `registry_avx512.h` — c2c in-place
The complex-to-complex in-place codelet family. Struct `vfft_proto_registry_t`,
filled by `vfft_proto_registry_init_<isa>()`. Slots, radix-indexed:
- **n1** — no-twiddle leaf butterfly (first/last stage of a c2c plan).
- **t1 / t1s** — twiddled stage. `t1` reads vector twiddles; `t1s` reads
  scalar-broadcast twiddles. × {dit, dif} orientation × {fwd, bwd} direction
  × {flat, log3} twiddle policy (log3 = fewer twiddle loads, more FMA).
The workhorse registry: ~324 codelets per ISA across the 18-family matrix.
(n1 slots are exposed via a 7-arg OOP wrapper so they match production's
stride ABI.)

### `oop_registry_avx2.h` / `oop_registry_avx512.h` — c2c out-of-place
Struct `oop_codelets_t`, filled by `oop_register_all_<isa>()`. Two ABIs:
- **11-arg** (runtime strides): `n1`, `t1p` (flat twiddle, the default),
  `t1p_log3` (port-rebalance opt-in — spends idle FMA-port slack to relieve
  load-port pressure; the planner picks it only for load-bound stages).
- **7-arg** (`*_spec`, strides baked at codegen as `rv = R*V`): `n1_spec`,
  `t1p_spec`, `t1p_log3_spec`. ~6-10% faster on the fixed-geometry one-call
  engine by letting the compiler fold strides into addressing.
Used by the Bailey fused four-step OOP engine (leaf = n1, stage 2 = t1p).

### `rfft_registry_avx2.h` / `rfft_registry_avx512.h` — real-to-complex (forward)
Struct `rfft_codelets_t`, filled by `rfft_register_all_<isa>()`. FFTW-style
real FFT cascade — no pack stage, no separate Hermitian terminator:
- **r2cf** — real-input leaf (the first stage; produces packed halfcomplex).
- **hc2hc / hc2hc_dif** — interior twiddle stage (Hermitian-packed in/out),
  DIT and DIF orientations in distinct slots (DIF generated but not yet called
  by the executor).
- **hc2c** — natural-split terminator (6-ptr mirror-pair ABI: Rp/Ip/Rm/Im).
- **\*_log3** — log3 twiddle-policy variants of hc2hc and hc2c (planner-
  preferred when registered).
- **\*_rng** — "ranged" variants that process a span of interior columns per
  call (fewer call boundaries).
Forward only; the backward (c2r) cascade is a future phase.

### `trig_registry_avx2.h` / `trig_registry_avx512.h` — real-to-real trig
Struct `trig_codelets_t`, filled by `trig_register_all_<isa>()`. One uniform
3-arg ABI `(in, out, K)`; the kind selects the slot, N is the index:
- **dct1..dct4, dst1..dst4** — the eight DCT/DST variants (Makhoul reduction
  for DCT-II, Lee 1984 for DCT-IV, odd/even extension for the boundary kinds
  DCT-I/DST-I). Forward direction only (II/III are an inverse pair, IV/DHT
  self-inverse up to scaling).
- **dht** — discrete Hartley transform.
Sizes N ∈ {8,16,32,64} for the main kinds; boundary kinds run at logical-
extension sizes (DCT-I at 2^k+1, DST-I at 2^k-1).

### `strided_registry_avx2.h` / `strided_registry_avx512.h` — 2D Design-C
Struct `strided_codelets_t`, filled by `strided_register_all_<isa>()`. The
batched 2D row-FFT codelets: matrix → register transpose → butterfly DAG →
inverse transpose → matrix, with no scratch buffer (doc 56). One uniform 6-arg
in-place ABI:
- **n1_fwd / n1_bwd** — single-stage no-twiddle 2D row FFT, forward/backward.
Single-stage by design. Radix sets differ per ISA (avx2 {4,8,12,16,20,32,64};
avx512 {8,16,32,64}) — avx2's 16-register file spills past R=256, so the high
radices are avx512-only; the registry mirrors that (missing slots stay NULL).

## Other files

### `registry.h` — hand-written ISA dispatcher
NOT auto-generated. A small, stable shell that selects the avx2 or avx512 c2c
registry at compile time. Has no promote rule; never overwritten.

### `plan_executors.h` — wisdom-bound executor specializations
Auto-emitted by `bin/emit_executor_h.ml` from `spike_wisdom.txt`. One
straight-line executor function per wisdom entry, replacing the generic
per-stage dispatch when the runtime plan matches a known shape (measured
5-10% on high multi-stage factorizations).

### `spike_wisdom.txt` — input to `plan_executors.h`
The committed wisdom lines that drive executor specialization. Append a line
and rebuild to add a specialization.

### `dune` — the promote rules
Declares every auto-emitted header above as a `(mode promote)` target, so
`dune build` regenerates and writes them into this folder whenever the
emitters or `coverage.ml` / `spike_wisdom.txt` change.

## Coverage map (which families are auto-wired)

| family | registry | status |
|--------|----------|--------|
| c2c in-place | `registry_*.h` | wired |
| c2c out-of-place | `oop_registry_*.h` | wired |
| r2c forward (rfft) | `rfft_registry_*.h` | wired |
| trig (dct/dst/dht) | `trig_registry_*.h` | wired |
| 2D strided (Design C) | `strided_registry_*.h` | wired |
| c2r (inverse real) | — | future phase (monolithic construction exists; cascade + coverage not yet built) |
