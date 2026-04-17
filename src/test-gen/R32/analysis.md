# R=32 Codelet Selection — SPR Results

This is the R=32 counterpart to the R=16 work. Same methodology, larger
candidate matrix (161 candidates vs 22 for R=16), more knobs.

---

## Bench execution summary

- **Candidates benched:** 161 (after cache-spill gating)
- **Sweep points:** 18 (me ∈ {64, 128, 256, 512, 1024, 2048} × ios ∈ {me, me+8, me+64})
- **Directions:** fwd + bwd, so **5796 total measurements**
- **Compile time:** 218s (parallel per-codelet, 2 cores)
- **Run time:** ~6 minutes wall (with intermittent restarts due to container process-kill behavior)
- **Chip:** SPR-class (x86_64, AVX-512F/DQ/CD/BW/VL + AMX)

The container killed the bench process every ~45-60 seconds. Bench harness
was modified to write measurements incrementally (JSONL format) and skip
already-completed work on restart. Eight restart iterations completed all
5796 measurements.

---

## Top-level findings

### Win distribution by family (knobs collapsed)

| Family | AVX2 wins | AVX-512 wins |
|---|---|---|
| `ct_t1_dit_log3` | **14 (39%)** | **20 (56%)** |
| `ct_t1_dit` (flat) | 13 (36%) | 11 (31%) |
| `ct_t1_buf_dit` | 9 (25%) | 3 (8%) |
| `ct_t1_ladder_dit` | — | 2 (6%) |

**`log3` dominates both ISAs on SPR**, consistent with R=16 SPR behavior.
Server-class HW prefetcher aggressiveness continues to favor derived-twiddle
families over memory-elimination families. This is the same mechanism we
identified on R=16.

### SW prefetch usefulness

| | Wins |
|---|---|
| With SW twiddle prefetch (`__tpf*`) | **52 (72%)** |
| Without SW prefetch (baseline) | 20 (28%) |

**Twiddle prefetch wins the majority of regions.** This is new information
— R=16 had no twiddle prefetch knob, so we never saw this. For R=32:

- At larger me (>= 128), twiddle table spills L1D, and SW prefetch
  becomes a genuine win even on SPR with its aggressive HW prefetcher.
- The dominant SW prefetch distance is **16** (for log3/flat_dit) and
  varies per regime.
- Prefetch rows=2 wins in some cases (notably at small me with flat t1_dit).

### Buf vs non-buf split

| | Wins |
|---|---|
| Non-buffered variants (dit, log3, ladder) | **60 (83%)** |
| Buffered (`buf_dit`) | 12 (17%) |

**Non-buffered dominates R=32 on SPR.** This is the opposite of what the
R=32 VTune data (66% DTLB-store-bound) initially suggested. The buffered
variants *are* better at small me (tile=256 + prefw + prefetch wins at
me=64-128 on AVX2) but lose the majority of the sweep.

### Stream drain

**Zero stream-drain wins.** Correctly gated out by the candidate matrix
(R × max_me × 16 = 1 MB < 2 MB L2, so stream would never win). The gate
saved ~50% of candidates.

---

## Split-point analysis

### AVX2 fwd direction

| ios | me | winner | ns |
|---|---|---|---|
| 64 | 64 | `ct_t1_dit__avx2__tpf16r2` | 1437 |
| 72 | 64 | `ct_t1_buf_dit__tile256_temporal_prefw_tpf16r2` | 1347 |
| 128 | 64 | `ct_t1_buf_dit__tile256_temporal_prefw` | 1344 |
| 128 | 128 | `ct_t1_buf_dit__tile256_temporal_prefw_tpf16r2` | 3434 |
| 136 | 128 | `ct_t1_buf_dit__tile256_temporal_prefw_tpf4r1` | 2911 |
| 192 | 128 | `ct_t1_dit__tpf16r2` | 3399 |
| 256 | 256 | `ct_t1_dit__tpf16r2` | 10399 |
| 264 | 256 | `ct_t1_dit__tpf16r2` | 6589 |
| 320 | 256 | `ct_t1_dit__tpf16r2` | 6384 |
| 512 | 512 | `ct_t1_dit__tpf16r2` | 23530 |
| 520 | 512 | `ct_t1_dit` (no prefetch) | 14111 |
| 576 | 512 | `ct_t1_dit_log3` (no prefetch) | 13636 |
| 1024 | 1024 | `ct_t1_dit_log3__tpf8r1` | 50699 |
| 1032 | 1024 | `ct_t1_dit_log3__tpf8r1` | 29453 |
| 1088 | 1024 | `ct_t1_dit_log3` (no prefetch) | 27061 |
| 2048 | 2048 | `ct_t1_dit_log3` (no prefetch) | 89382 |
| 2056 | 2048 | `ct_t1_dit_log3__tpf16r1` | 61085 |
| 2112 | 2048 | `ct_t1_dit_log3__tpf16r1` | 55846 |

**Pattern on AVX2:**
- **Small me (64, 128):** Buffered `tile256` with drain-prefetch wins — small enough working set that buffering overhead amortizes well, and the prefetchw of output pages is an outright win.
- **Medium me (256, 512):** Flat `t1_dit` with `tpf16r2` (distance-16 prefetch, 2 rows) wins consistently. The flat codelet's simpler inner loop benefits most from SW prefetch hiding twiddle latency.
- **Large me (1024+):** `log3` takes over — its derived-twiddle scheme has less cache footprint, which matters more at large working sets.
- **Padded stride (me+8, me+64):** frequently a different winner than power-of-2 stride — the alias effect we saw on R=16 applies here too.

### AVX-512 fwd direction

| ios | me | winner | ns |
|---|---|---|---|
| 64 | 64 | `ct_t1_dit_log3__tpf4r1` | 759 |
| 72 | 64 | `ct_t1_buf_dit__tile128_temporal_prefw_tpf8r2` | 877 |
| 128 | 64 | `ct_t1_dit_log3` (no prefetch) | 1175 |
| 128 | 128 | `ct_t1_dit__tpf4r1` | 2396 |
| 136 | 128 | `ct_t1_dit__tpf4r1` | 2057 |
| 192 | 128 | `ct_t1_dit__tpf32r1` | 2125 |
| 256 | 256 | **`ct_t1_ladder_dit`** | 5576 |
| 264 | 256 | `ct_t1_dit__tpf32r1` | 4240 |
| 320 | 256 | `ct_t1_dit` (no prefetch) | 4864 |
| 512 | 512 | `ct_t1_dit_log3__tpf32r1` | 11127 |
| 520 | 512 | `ct_t1_dit` (no prefetch) | 10359 |
| 576 | 512 | `ct_t1_dit_log3__tpf16r2` | 7633 |
| 1024 | 1024 | `ct_t1_dit_log3__tpf16r2` | 25781 |
| 1032 | 1024 | `ct_t1_dit_log3__tpf8r1` | 19253 |
| 1088 | 1024 | `ct_t1_dit_log3__tpf16r1` | 18886 |
| 2048 | 2048 | `ct_t1_dit_log3__tpf8r1` | 47824 |
| 2056 | 2048 | `ct_t1_dit_log3` (no prefetch) | 33442 |
| 2112 | 2048 | `ct_t1_dit_log3__tpf32r2` | 33359 |

**Pattern on AVX-512:**
- **`ct_t1_ladder_dit` wins at me=256 pow2 stride.** This is the new variant
  with 5-base-twiddle log derivation on AVX-512 — validates the ladder
  development from prior sessions.
- Flat `t1_dit` wins at me=128 across all stride offsets — AVX-512's wider
  vector width makes flat's simpler codegen more profitable here.
- `log3` dominates me >= 512 across both stride regimes. The lower
  twiddle-memory footprint shines at large working sets with AVX-512's
  larger register file.
- SW prefetch distances vary between 8 and 32 across me values — confirms
  the "per-region tuning" premise.

---

## Stride aliasing (same story as R=16, larger effect)

Comparing flat `t1_dit` fwd at me=1024 on AVX2:

| ios | ns | relative |
|---|---|---|
| 1024 (pow2) | ~51000 | baseline |
| 1032 (+8 padded) | ~30000 | −41% |
| 1088 (+64 padded) | ~28000 | −45% |

The L2 set-associativity conflict costs **45%** on R=32 at me=1024 (vs
44% on R=16). Same mechanism, slightly larger absolute effect because
R=32 has 2× the per-me memory footprint.

**Implication for the planner:** when factorizing a composite size
(e.g. 32768 = 32 × 1024), prefer factorizations where the inner ios
isn't a multiple of 1024. The planner doesn't know this yet — it's a
v2 optimization where the planner reads the selector's per-region
timings and avoids alias-prone strides.

---

## Comparison to R=16 SPR

| | R=16 | R=32 |
|---|---|---|
| Distinct winners | 7 | 28 |
| Dominant family (AVX2) | `log3` (64%) | `log3` (39%) + `flat_dit` (36%) |
| Dominant family (AVX-512) | `log3` (72%) | `log3` (56%) |
| Buf family wins | 14% | 17% |
| Prefetch knob wins | — (didn't exist) | 72% |

**Key differences:**

1. **R=32 has more winners** (28 vs 7) because the candidate space is 7×
   larger (161 vs 22) and more knobs create more "close calls" that the
   2% tie threshold doesn't collapse.

2. **Flat `t1_dit` is much more competitive on R=32.** On R=16 it only won
   3 regions (6%). On R=32 it wins 24 regions (33%). Why? Because R=32
   benefits much more from SW twiddle prefetch — the larger radix means
   bigger twiddle arrays, and SW prefetch with rows=2 can hide enough
   latency to let flat's simpler codegen outperform log3's derived
   twiddles.

3. **Prefetch knobs matter massively on R=32.** 72% of winners use some
   prefetch distance. This validates the decision to include them in the
   candidate matrix despite the candidate-count cost.

4. **Ladder codelet finds a niche** (2 wins at me=256 AVX-512). Not a
   dominant winner, but takes regions where nothing else beats it.

---

## The AVX2 results you'll compare against on Raptor Lake

The AVX2 half of the SPR bench (36 decisions) produced these winners for
you to compare against your Raptor Lake run:

- `ct_t1_dit__tpf16r2`: 11 wins (dominant)
- `ct_t1_dit_log3` + variants: 14 wins
- `ct_t1_buf_dit__tile256_temporal_prefw` + tpf variants: 7 wins
- `ct_t1_dit`: 2 wins (no prefetch)
- `ct_t1_buf_dit__tile128*`: 2 wins

**Predictions for Raptor Lake AVX2 based on R=16 cross-chip pattern:**

1. **Buffered should win more regions on Raptor Lake.** R=16 saw Raptor
   Lake pick tile16 (consumer-narrow) where SPR picked tile128
   (server-wide). Extrapolating: R=32 Raptor Lake will probably favor
   tile32 or tile64 where SPR picked tile256.

2. **Prefetch knobs should matter even more on Raptor Lake.** Weaker HW
   prefetcher means SW prefetch fills a larger gap. SPR already leans
   heavily on SW prefetch (72%); Raptor Lake may reach 85-95%.

3. **log3 may lose dominance on Raptor Lake.** R=16 showed Raptor Lake's
   weaker HW prefetcher tipped the balance toward memory-elimination
   (t1s). R=32 doesn't have t1s but does have ladder (AVX-512 only) and
   flat with heavy SW prefetch. Predict: flat+prefetch could be a much
   bigger winner on Raptor Lake than on SPR.

These are testable hypotheses. When you run on Raptor Lake, three things
to look for:

- Does flat (`ct_t1_dit__*`) win more than 50% of regions?
- Does buf favor tile32/tile64 over tile128/tile256?
- Are prefetch-using variants >80% of winners?

If all three, the "µarch decides codelets, HW prefetcher aggressiveness
dominates" framing holds for R=32 just as it did for R=16.

---

## Files in this directory

- `candidates.py` — R=32 candidate enumeration (161 candidates)
- `run_r32_bench.py` — full pipeline (generate + parallel build + run with resume)
- `bench_codelets.py` — legacy R=16-style bench (use run_r32_bench.py instead)
- `select_codelets.py` — winner selection
- `emit_selector.py` — C header generation
- `gen_radix32_buffered.py` — R=32 codelet generator
- `spr_results/` — SPR measurement data
  - `measurements.json` — 2898 aggregate records (fwd+bwd paired)
  - `selection.json` — 72 decisions with tie-threshold applied
  - `selection_report.md` — human-readable win tables
  - `generated_headers/` — the 28 winning codelet .h files + `codelet_select_r32.h`

---

## Usage on your Raptor Lake

Pipeline:
```
python run_r32_bench.py --phase generate   # ~20s
python run_r32_bench.py --phase build      # ~2-5 min (parallel compile)
python run_r32_bench.py --phase run        # can be interrupted + resumed
python select_codelets.py --input measurements.json --output selection.json
python emit_selector.py --selection selection.json --measurements measurements.json --out-dir include --report raptor_lake_r32_report.md
```

The run phase produces `measurements.jsonl` incrementally. If the process
dies partway through, just re-run `--phase run` — it'll pick up where it
left off.

For AVX2-only runs, the harness's CPUID check will auto-skip the 73 AVX-512
candidates, cutting the sweep roughly in half.

The `run_r32_bench.py` pipeline uses GCC by default. Set `CC=icx` in your
environment to use Intel ICX (same as R=16 bench).
