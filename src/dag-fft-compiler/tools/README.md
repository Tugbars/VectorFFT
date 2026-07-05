# tools/ — measurement instruments for the DAG compiler

These are not part of the compiler. They are the instruments that
decide what becomes part of the compiler. All of them score the same
way the project scores everything: emit C, compile with the pinned
flags (gcc, `-O3 -mavx2 -mfma -march=raptorlake`, main loop via
`VFFT_NO_ANYK_TAIL=1`), count instructions and stack spills. A change
wins only if insns and spills are both <= baseline with at least one
strict (the "win rule").

## The probe-first method (why these exist)

An optimization idea has two costs: implementing it in the OCaml IR
(days, plus risk to a certified tree) and finding out whether it
works (which should be minutes). The probes separate the two: each is
a ~150-line Python rewriter that applies a candidate transform to
the EMITTED C TEXT, recompiles, and reports the counts. If the probe
wins, it becomes the convergence oracle for the real pass — the pass
is debugged until its output matches the probe's, then raced against
it. If the probe loses, no pass is ever built and the negative is
recorded with numbers.

Track record: two transforms probed, one pass built (duplication —
converged to probe parity, then exceeded it), one pass avoided
(reassociation — dominated, closed for the cost of a parser).

## dup_probe.py — selective un-CSE (duplication)

**Problem.** Hashconsing CSEs everything, globally. For values with
long live ranges and many consumers (the pair-sums of a direct-DFT
prime codelet), keeping one copy alive across the whole body costs
more in register pressure — spills — than recomputing it near its
last consumer would cost in arithmetic. The question: which values,
where, and does gcc actually reward it?

**Tool.** `dup_probe.py FILE S CAP` parses temp definitions, finds
multi-use cheap values whose textual span exceeds S, clones the top
CAP of them immediately before their last consumer (with a compiler
barrier so gcc cannot re-CSE the clone), redirects that one use, and
scores.

**Evidence.** Wins on every direct-DFT prime; the numbers below were
then reproduced exactly by the IR-level pass (`VFFT_DUP=1`), which
the probe served as oracle for:

| R | baseline | duplication | with affinity tie-break |
|---|---|---|---|
| 11 | 317/42 | 269/17 | 267/16 (best known) |
| 13 | 446/70 | 377/31 | — |
| 17 | 756/175 | 711/115 | 693/108 (best known) |
| 19 | 876/192 | 849/139 | — |

The probe also mapped the transform's boundaries: blocked emission
has zero candidates (explicit spilling already pays the pressure),
and CT-factored codelets have none (stage outputs die locally). The
material is a property of direct-DFT dense-sum structure — knowledge
that came from probing six radices in minutes each.

## skew_probe.py — reassociation / sum-skew

**Problem.** Prime codelets are dominated by linear FMA
product-chains (the tail mul computes first; the textually-first
operand dies last). Two hypotheses: (a) permuting per-chain
consumption ORDER staggers live ranges; (b) splitting each chain
across k accumulators (SHAPE) halves depth and kills operands
earlier. Worth an IR pass?

**Tool.** `skew_probe.py FILE STRAT [--dc]` parses the chains and DC
add-nests and rebuilds them under a strategy: `rev/alt/stag` (order),
`split2/split2c/split3` (shape), `--dc` (balanced DC tree). `id`
must reproduce the input byte-exactly — the parser is self-gating.

**Evidence.** Both hypotheses answered with finality. Order is
provably inert: rev/alt/stag produce DIFFERENT assembly (md5s
differ) with IDENTICAL insn/spill counts at R=11-19 — gcc's register
allocator absorbs order permutations of a fixed chain shape. Shape
wins standalone (R=11 310/40, R=13 429/62) but is DOMINATED by
duplication at every prime in both composition orders (e.g. R=13:
skew-then-dup 385/35 vs dup alone 377/31), and it changes rounding
where duplication is bit-exact. Weaker wins + worse numerics class =
no pass built. Full tables in docs/74.

## truth_gate.py — accuracy adjudication

**Problem.** Some transforms change floating-point rounding
(reassociation, FMA-fusion swaps). "Bit-exact or nothing" would ban
them dogmatically; "looks close enough" is not a criterion. Needed:
an objective admissibility gate.

**Tool.** Computes the reference DFT at 40-digit precision (mpmath)
and compares both the baseline codelet and the candidate against
truth. PASS iff the candidate's max error <= the baseline's. This is
the codified numerics policy: a non-bit-exact transform is
admissible iff it truth-gate PASSES and the README discloses it.

**Evidence.** Adjudicated butterfly-share-mul: against truth, both
versions are IDENTICAL to the digit (R=32 max 3.61e-13, rms 2.5e-14,
both; per-element differences are a coin-flip in the last ulp). That
measurement converted a philosophical blocker into a shipped policy.

## blocked_anneal.py — schedule search for blocked codelets

**Problem.** The SR scheduler is certified optimal for its objective,
but its objective is a proxy; for the big blocked codelets (R=32/64)
the mapping from schedule to gcc's final counts has exploitable slack
that only search finds.

**Tool.** Per-cluster simulated annealing over intra-cluster orders,
scoring through the real generator + gcc each trial. Incumbents
persist across runs (a run resumes from the best of all previous
runs); `VFFT_WARM=<dir>` seeds from any wisdom directory, enabling
warm-started campaigns (e.g. from affinity orders).

**Evidence.** R=32: 1031/183 -> 996/168. R=64: 2543/594 -> 2490/585
(the shipped `sched_wisdom/` entries). Resume verified by chaining:
consecutive seeds progressed 1020 -> 1017 -> 1015 without losing
ground — long campaigns are now cumulative instead of amnesiac.

## Boundaries

The probes operate on emitted C and share nothing with the OCaml
tree; they run anywhere. `blocked_anneal.py` is the exception: it
drives `VFFT_SCHED_ORDER` injection and is inert without the
schedule-wisdom machinery in `generator/lib`. `truth_gate.py` needs
`mpmath`. All results above are static counts on the pinned
toolchain; runtime confirmation on the target machine
(REPRODUCE.md, recipe C8) is the standing final gate.
