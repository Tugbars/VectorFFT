# 74. Sum-skew / reassociation — investigated, NEGATIVE (dominated by dup)

2026-07-02. Status: closed as a deployment lever; probe retained as a
tool (`tools/skew_probe.py`). No OCaml pass will be built.

## 1. The question

Prime codelets are dominated by product-chains
`F(a,b, F(a,b, ... mul(a,b)))` (24 at R=13, each 6 products; the tail
mul computes FIRST — the textually-first operand is consumed LAST)
plus two DC add-nests. Reassociation freedom: any product permutation,
any accumulator count. Two sub-levers were hypothesized:
(a) ORDER — stagger consumption across chains so pair-sum live ranges
    shorten (the same material dup attacks);
(b) SHAPE — k accumulators per chain (split2/split2c/split3), paying
    +1..2 adds per chain for halved depth and earlier operand death.

## 2. Probe

`tools/skew_probe.py FILE STRAT [--dc]` — text-space rewrite of
parsed chains; `id` strategy is a parser gate (byte-exact on all five
prime baselines). Strategies: rev, alt, stag (order); split2,
split2c, split3 (shape); --dc (balanced-tree DC sums). Same
gcc/count harness as everything else (13.3, raptorlake, main loop).

## 3. Finding 1 — ORDER IS INERT (proven, not assumed)

rev/alt/stag produce DIFFERENT assembly (md5s of the .s differ) yet
IDENTICAL insns/spills at R=11/13/17/19. gcc's RA absorbs
consumption-order permutations of a fixed linear-chain shape. The
"stagger the last consumer" idea is dead at the gcc level. (R=23
orders regress — consistent with its general fragility.)

## 4. Finding 2 — SHAPE wins standalone, small

| R | base | split2 | split2c | split3 | s2+dc |
|---|---|---|---|---|---|
| 11 | 317/42 | **310/40** | 315/44 | = | 312/44 |
| 13 | 446/70 | 430/63 | **429/62** | 433/**59** | 442/66 |
| 17 | 756/175 | 761/180 | **754/175** | 764/175 | 770/190 |
| 19 | 876/192 | 884/185 | 889/194 | 895/186 | 903/204 |
| 23 | 1275/294 | 1291/311 | 1291/306 | 1333/334 | 1282/304 |

DC-tree alone wins at 13 (433/62) but INTERFERES with split2
(combo 442/66 — worse than either). Wins are real at 11/13, marginal
at 17, absent at 19/23.

## 5. Finding 3 — DOMINATED BY DUP (the verdict)

Composition, both orders (skew then dup-probe; dup output then skew):

| R | skew2>dup best | dup-only | dup+affinity |
|---|---|---|---|
| 11 | 272/18 | 269/17 | **267/16** |
| 13 | 385/35 | **377/31** | 377/31 |
| 17 | 713/127 | 713/118 | **693/108** |
| 19 | 851/146 | **849/139** | — |
| 23 | 1373/367 | (dup off) | — |

Skew and dup are SUBSTITUTES: both shorten pair-sum live ranges (skew
by earlier consumption via a second accumulator, dup by
rematerialization at the last consumer). Dup wins at every prime AND
is bit-exact, while skew changes rounding (truth-gate class). Weaker
wins + worse numerics class = dominated on both axes. No truth-gate
run was needed: nothing deploys.

## 6. Honest caveats

Static counts are the metric. split2 halves chain depth, so a
LATENCY-bound context (tiny K, single call, no k-loop overlap) could
in principle prefer it; the codelets are throughput-shaped (24
independent chains x independent k iterations), so the expectation is
no. Listed on the i9 manifest as a curiosity-priority A/B, not a
blocker. Pow2/mono-16 chains were not raced (different structure;
primes were the motivated target).
