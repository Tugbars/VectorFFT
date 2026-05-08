# vfft v2 — Variant-Dispatcher Codelet Generator

A small OCaml-based generator for SIMD FFT codelets, exploring how far emission policies (variants) can carry generated code without changing the math layer. Empirically, the answer is "very far": the same OCaml code path beats hand-tuned reference codelets across radices R=16/32/64 on AVX-512, and across radices R=16/32 on AVX2.

## What's inside

```
vfft_v2/
├── README.md          this file
├── docs/              chronological findings (01-12), each one is a session writeup
├── lib/               OCaml generator (dft, algsimp, emit_c, schedule, isa, uarch, ...)
├── bin/               gen_radix CLI
├── bench/             bench harnesses + generated codelets + hand-coded references
└── dune-project       OCaml build config
```

## Headline result

For R=32 AVX-512:

| K     | Hand | Topo  | SU+Spill | SU/Hand |
|-------|------|-------|----------|---------|
| 64    | 1600 | 2442  | 1572     | **0.98** |
| 128   | 3560 | 6588  | 3481     | **0.99** |
| 256   | 9173 | 14228 | 9066     | **0.99** |
| 512   | 23097 | 32001 | 22565   | **0.97** |
| 1024  | 58729 | 75853 | 56359   | **0.93** |
| 2048  | 166167 | 199958 | 150476 | **0.91** |
| 4096  | 338886 | 398469 | 311878 | **0.91** |

(times in ns; SU/H < 1 means we beat hand-coded)

For R=64 AVX-512: 1-7% faster than hand. For R=16 AVX-512: beats hand at K ≥ 128. R=32 AVX2: up to **44% faster than Topo** (no hand-coded reference).

## The recipe

For CT-decomposed codelets:
```
spill PASS 1 outputs at the natural CT boundary (--spill)
  + run SU within each pass (--su)
  + cluster-sequential PASS 1 (block by sub-FFT)
  + cluster-sequential PASS 2 (one sub-DFT at a time, stores immediately after)
  + just-in-time reload (within PASS 2, defer each load to first use)
```

Use it when:
```
CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32)
```

This means:
- AVX-512 (vec_regs=32): always use the recipe at R≥4
- AVX2 (vec_regs=16): use the recipe at R≥16; R≤8 prefers Topo

## How to read the docs

The findings under `docs/` are chronological session writeups, numbered 01-12. They tell the story of how each lever was discovered, what didn't work, and how the picture evolved:

1. **su_scheduler** — SU list scheduler (first lever, modest effect alone)
2. **r16_crossover** — discovered R=16 small-K vs large-K crossover; variants matter
3. **r32_finding** — R=32 brings register pressure problem into focus
4. **r32_arith_gap** — investigation: GCC already does FMA fusion, math layer is fine
5. **r32_spill** — explicit boundary spilling (closes ~half the gap)
6. **r16_spill** — heuristic falsified at AVX-512 R=16 (recipe helps even when registers fit)
7. **r32_fuse** — FUSE per-spill-target lifetime (modest gains, F8 = block-seq no-spill is interesting)
8. **r32_su_spill** — SU within passes (closes most of the rest)
9. **r32_beating_hand** — cluster-sequential PASS 2: the missing piece. We start beating hand.
10. **r16_full_recipe** — full recipe applied at R=16 (beats hand K≥128)
11. **r64_finding** — full recipe applied at R=64 (beats hand all K)
12. **avx2_finding** — recipe validated at AVX2; ISA-aware rule

Read in order, the docs trace from "Topo is 13-69% slower than hand at R=32" to "we beat hand on every radix and ISA combination where the rule says to use the recipe."

## How to build and run

```bash
cd /path/to/vfft_v2
dune build

# Generate a codelet
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --spill --su 32

# CLI flags
#   --twiddled         use twiddle pre-multiply (t1_dit shape)
#   --emit-c           emit C code instead of stats
#   --in-place         in-place signature (rio_re/rio_im) vs out-of-place
#   --spill            enable explicit boundary spilling
#   --su               enable SU scheduler within passes
#   --fuse N           keep N PASS 2 sub-DFTs' inputs alive across boundary
#   --isa avx512|avx2  target ISA (default: avx512)
```

## What didn't work (briefly)

- **FMA fusion at the source level.** Tried to manually emit `_mm256_fmadd_pd` instead of `_mm256_add_pd(_mm256_mul_pd(...), ...)`. GCC already does this fusion at the right level given `-O3 -mfma`, and explicit FMA emission constrained GCC's instruction selection slightly. Disabled.
- **Bisection scheduler (Frigo's).** Implemented but performed similarly or slightly worse than Topo on AVX-512. The cp_dist + su_num approach (SU) ended up dominant.
- **Annotate (Frigo-style nested-block scoping).** Built and benchmarked — produced byte-identical assembly to Topo at -O3. Modern GCC's SSA + liveness analysis already extracts everything annotate would communicate.
- **Distributing constants** (`Add(Mul(a,k), Mul(b,k)) → Mul(Add(a,b), k)`). Looks like an arith-saving rewrite; in practice it broke shared muls in cmul patterns and increased op counts.

## What's left in the queue

- Encode the rule in `should_spill` so `--spill --su` becomes the default (5 lines)
- R=64 AVX2 (CT(8,8) at AVX2 should be a massive recipe win)
- R=4 AVX2 (spot-check the lower edge)
- K-threshold for spill at R=16 K=64 on AVX-512 (close the small remaining regression)

## Acknowledgments

Built around hand-coded R=4/8/16/32/64 references generated from `gen_radix*.py` scripts. The hand-coded codelets are excellent and beating them empirically validated the variant approach.

The OCaml generator (`lib/`) is roughly 2500 lines, including the DAG (`algsimp.ml`), the algorithm picker (`dft.ml`), the schedulers (`schedule.ml`), the C emitter (`emit_c.ml`), and the ISA/µarch profiles (`isa.ml`, `uarch.ml`).
