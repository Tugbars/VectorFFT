# Large fully-odd N: the engine question

**STATUS: SHIPPED IN THE FRONT DOOR (2026-09-05).** The flat mixed-radix
DIT (`src/core/oop/il_flatdit.h`, route `VFFT_K1_IL_FLAT`) serves odd N to
2¹⁸ in both placements and directions, raced per cell by the K=1 planner
(chains and per-stage kernel forms), banked as `il_flat=`/`il_forms=` on
the kind-3 row; results in docs/performance/v1_0_results.md ("the K=1 IL
tier for odd N"). The four-step bridge (`il_flat.h`) it was raced against
was deleted the same day: the flat DIT matched or beat it at every cell.
Threading and banding are the levers left. The option analysis below is
kept as the record of how the design was chosen.

## v1: what was built and what it measures

Structure (every kernel a shipped one): corner-turned leaf (R₀ legs at
stride N/R₀, full lanes) → one rotor sweep for the outer twiddle
(AVX2, table of N/R₀ entries) → the 2D column chain over N/R₀ rows ×
R₀ lanes (per-digit tables, any depth, natural output by the leaf
scatter; no ordering pass). The truly flat interior with compact tables
needs the two-group "arrange halves" kinds (MKL's Fact kernels); with
the per-column-pair `t2` kernels a flat chain would carry ~8N doubles
of table per stage, so v1 takes this bridge form.

Raced per cell (leaf × interior), single thread, same run vs MKL:

| N | raced chain | vs MKL |
|---|---|---|
| 405 | 9 \| 5·9 | 1.06× |
| 1215 | 9 \| 9·3·5 | 1.10× |
| 4095 | 9 \| 7·5·13 | 0.94× |
| 6561 | 9 \| 9·9·9 | 1.09× |
| 19 683 | 9 \| 9·3·9·9 | 1.14× |
| 59 049 | 9 \| 9·9·9·9 | 1.18× |
| 98 415 | 15 \| 9·9·9·9 | 0.96× |
| 137 781 | 9 \| 21·9·9·9 | 0.98× |

The seed chain (radix 27 first) is 1.5–2× slower than the raced winner
at every cell: radix 27 is twiddle-load bound, and the winners avoid it.

## The requirement

Odd N (no factor of 2) served natively with cascade-class performance up
to at least N ≈ 131 000. Today's ceiling for a native odd plan is the
three-stage Bailey chain, 27³ = 19 683.

## What exists today

| N class                    | serving                                   | vs MKL (2026-09-04) |
|----------------------------|-------------------------------------------|---------------------|
| odd, ≤ 27·27 as a pair     | il2p pair (two stages)                    | 1.0–2.0×            |
| odd, ≤ 27³ as a chain      | il3p chain3 (three stages, odd-legal)     | 0.79–1.24×          |
| 2^a·odd, N % 16 == 0       | z-cascade with odd mids (r0 = 4 ingest)   | 1.46–1.62×          |
| odd, > 27³                 | Bluestein at the next power of two        | ~0.3×               |

The chain3 numbers: 1215 → 0.81×, 2187 → 0.85×, 3645 → 0.79×,
4095 → 1.24×, 6561 → 0.87× (same-run, single thread).

## The constraint that shapes every option

The cascade's ingest radix is the vector width. The `s0t` kernel stores
the radix-4 butterfly's four outputs — one per sub-transform — as a
single `[re×4][im×4]` record, and the whole interior then runs the four
sub-transforms lane-parallel; the terminator un-turns them. The emitter
asserts `r0 == vw` at every section edge for this reason. An odd ingest
therefore cannot fill the vector: r0 = 3 leaves one dead lane in every
interior instruction (−25% by construction), r0 = 9 needs 2¼ vectors.
"Extend ingest/terminator to odd radices" is not a codelet addition; it
is a different, lane-padded geometry.

## Options

### A. chainK — a deeper Bailey chain (driver-only)

Generalize il3p from three stages to K, N = R2·A·B·C…, each stage a full
pass with the existing ≤27-point monolithic kernels and per-stage twiddle
tables (the chain3 odd-legal table layout, per block with a ceiling pair
count, generalizes directly). Reach: 27⁴ = 531 441. Every lane busy.
Open question: whether 4–5 passes hold chain3's 4095-class performance
at 59 049 / 98 415, where the planes leave L2 — the cascade's banded
walk (tcut) is the known remedy and could be laid over the chain later.
Cost: days. Risk: low (no new kernels).

### B. Lane-padded odd cascade (new tier geometry)

An r0 = 3 (or 5) ingest producing padded 4-lane records, the interior
unchanged apart from a dead lane, odd terminators tapping three sections.
Touches: emitter section addressing (`E_sect_tr4`/`E_sect_tap` at r0 ≠ 4),
zsplit.h's plan builder, zturn.h create/execute and count math, the t2q
calibrator, the natural terminator's rho tables, the MT phases. Keeps the
cascade's streaming shape at large N but pays 25% on every interior
instruction. Cost: an arc several times the odd-mids step. Risk: medium.

### C. Bluestein over the cascade (exists)

The current fallback. Correct anywhere; ~3–4× slower than a native plan.

## The decision the owner is weighing

Whether the size target is met by A (with the cascade's banding as a
later optimization) or requires B.

**Ruling (owner, 2026-09-04): Bluestein is never raced against a native
chain.** A native chain serves wherever N is expressible in the kernel
corpus; Bluestein serves only where it is not (a factor above the largest
emitted radix — and there the prime engine's own Rader-vs-Bluestein race
applies). The only race A or B needs is among chains: factorisation and
forms, the plan race chain3 already runs. The measurement that matters is
A (and B, if built) against MKL at 19 683 < N ≤ 131 072.

Note on the "batch-lane" geometry in
[`docs/design/odd_n_cascade_geometry.md`](../design/odd_n_cascade_geometry.md):
the interior family it describes — radix across registers, lanes holding
consecutive elements, one permute per twiddle pair — is the existing IL
chain corpus (cil `t2c`/`n1c`, the il2p/il3p mids). Option A *is* that
geometry built from shipped kernels; its genuinely new piece is banding
for planes beyond L2.

## References

- `src/core/oop/il2p.h` — il2p / il3p (chain3 odd-legal, 2026-09-04)
- `src/core/planning/dp_planner_il.h` — the K=1 IL plan race and its
  chain3 pool (odd leaves, race admitted above 2048 for N % 4 ≠ 0)
- `src/dag-fft-compiler/generator/lib/gen/cascade_z.ml` — the ZTURN-S
  section edges and the `r0 == vw` assertions
- `docs/roadmap/cascade_load_path_restructure.md` — Amendment ZTURN-S,
  the r0 = 4 geometry of record
- `docs/performance/v1_0_results.md` — "1D ODD c2c" for the measured state
