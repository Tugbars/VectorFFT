# 73 — butterfly_share_mul revived: the orphan pass, measured

## Provenance

`Algsimp.butterfly_share_mul` (algsimp.ml, ~185 lines) was found
fully implemented — header analysis, safety argument, frozen-tag
remap support, its own `BSM_TRACE` debug gate — but wired into
nothing. This doc records its first measurement. The pass recognizes
swap-pair butterflies (two FMAs holding each other's product in the
addend slot), rewrites one via commutativity + sign-flag swap so both
share a single declared Mul, and orphans the twin. The original
header predicted "2 such pairs at R=32, saving 2 ops."

## Wiring

Env-gated `VFFT_BUTTERFLY_SHARE=1`, default off, byte-identical off
(10-radix gate green). Inserted after the mfl4 pass and BEFORE
flatten_fma_mul_addend (flatten absorbs the Mul-addends this pass
pairs) in BOTH cascades: `gen_main.ml` (production path, remap spliced
into the marker walk after mfl4) and `pipeline.ml` (via its `step`
helper, same position).

## Results (gcc 13.3, -march=raptorlake, main loop, insns/spills)

| R | baseline | BSM on | pairs fired | verdict |
|---|---|---|---|---|
| 4–23 (all mono) | — | identical | 0 | pattern absent monolithic |
| 25 | 1000/171 | 1014/185 | 4 | LOSES — keep off |
| 32 | 1031/183 | **1022/181** | 2 (exactly as the header predicted) | win |
| 64 | 2543/594 | **2529/590** (mul 32→24, fma unchanged — textbook) | 8 | win |

## Numerical status — FIRST NON-BIT-EXACT TRANSFORM

The sign-flag swap changes which product is FMA-fused (exact) vs
separately rounded per FMA. Mathematically identical; bitwise not:

| R | values differing | max abs | max rel |
|---|---|---|---|
| 32 | 27 / 4096 | 8.9e-16 | 1.3e-14 |
| 64 | 22 / 8192 | 4.4e-16 | 8.6e-16 |

Last-ulp noise — and MEASURED AGAINST TRUTH (40-digit mpmath
reference DFT of the verify input, tools/truth_gate.py):

| | max rel vs truth | rms vs truth | elements closer to truth |
|---|---|---|---|
| R=32 baseline | 3.61e-13 | 2.5e-14 | 11 |
| R=32 BSM | 3.61e-13 (identical) | 2.5e-14 (identical) | 12 |
| R=64 baseline | 3.97e-13 | 1.98e-14 | 9 |
| R=64 BSM | 3.97e-13 (identical) | 1.98e-14 (identical) | 10 |

ACCURACY IS UNCHANGED — the trade is bit-REPRODUCIBILITY against
previous binaries, not correctness. Every prior transform was
bit-exact (a free proxy for "numerically unchanged"); BSM is the
first where the proxy and the property diverge, and the property
holds. The policy decision (owner's) is therefore purely about
reproducibility guarantees, with truth_gate.py as the permanent
instrument for any future non-bit-exact transform.

## Composition with wisdom — the per-cluster dagsig system at work

BSM changes only the clusters containing pairs, so per-cluster
dagsigs partially survive: at R=32, 11/12 annealed entries still
inject (one refuses); at R=64, 16/20 (four refuse). Measured:

| R | wisdom only | BSM only | BSM + surviving wisdom | BSM + 22-iter re-anneal |
|---|---|---|---|---|
| 32 | **996/168** (incumbent) | 1022/181 | 999/172 | **999/172** (22-iter campaign; shipped) |
| 64 | **2490/585** (incumbent) | 2529/590 | 2543/597 (4 SR-fallback clusters cost +53) | not yet run |

R=32 CAMPAIGN RECORD: the 22-iter stacked run's incumbents, scored
through the shipping mechanism, land at 999/172 — dagsig-verified and
shipped as `generator/sched_wisdom_bsm/` (11/12 verified, one
snapshot entry unverified; activation = VFFT_BUTTERFLY_SHARE=1 +
VFFT_SCHED_WISDOM=<that dir>). A follow-up 50-iter run at seed 2 did
NOT resume from incumbents (tool restarts fresh per seed-keyed output
dir — resume support is a noted tool gap) and stayed at baseline.
VERDICT TODAY: the baseline-DAG incumbent 996/168 still leads the
stack by 3 insns / 4 spills at this search budget. The unlock is a
resume-capable long anneal (and the R=64 campaign); until stacked
beats incumbent, BSM stays default-off on PERFORMANCE grounds — the
ulp question is already resolved in its favor.

Interpretation: BSM alone is dominated by existing wisdom at both
radices TODAY; its value is locked behind re-annealing the changed
clusters on the new (smaller) DAG. The 22-iter probe covered a
quarter of the original campaign length and closed 14 of the 26-insn
gap to the incumbent from the BSM baseline — trajectory points at or
below 996/168 on a full run. R=64 full re-anneal (4 clusters)
prescribed, not claimed.

## Status and the two open decisions (owner's)

1. **Ulp policy — RESOLVED by owner (2026-07-02)**: truth-gated
   ulp-neutral transforms are admissible; users needing bit-exactness
   disable the flag; disclosure lives in the dag-fft-compiler
   README ("Numerical reproducibility policy"). truth_gate.py PASS is
   mandatory for this transform class.
2. **Full stacked campaign** — unblocked by (1). R=32 full-length run
   recorded below; R=64 (20 clusters, ~4x cost) prescribed for a
   dedicated session, then the i9 runtime gate.

R=25 stays off regardless (recorded negative). The pass itself is no
longer orphaned: wired, gated, raced, and its own years-old R=32
prediction confirmed to the pair.
