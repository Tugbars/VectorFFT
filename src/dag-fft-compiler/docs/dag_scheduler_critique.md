# DAG construction and scheduler: the critique of record

The criticizing-goggles review (lab notebook sections 44-45), preserved
with the design owner's verdicts and the post-A1 status of each charge.
Actionables live in dag_scheduler_program.md; this document is the
argument behind them, kept so future-us knows WHY each item exists and
which charges died on measurement.

## DAG construction

### C1. Construction treats liveness as the scheduler's problem
The CT(4,4) seam mathematically forces 32 live reals; the list
scheduler achieves 35 against that floor; FFTW sits at 8. The
scheduler is nearly optimal against a wall the constructor built, and
weeks of scheduler tuning (bisection repair: real effort, zero wins)
were spent against a bound it could never beat. FFTW's trick is that
construction order is the schedule seed: streaming emission with a
bounded frontier by birth.
**Owner's verdict:** split-radix arithmetic is NO better than CT+CSE
(SR module is well behind CT today), so the prize is the emission
ORDER, not the algorithm. Folded into A7: CT-streaming emission, SR
held in reserve (A8) behind an op-parity entry gate.

### C2. doc-58 institutionalized the seam instead of attacking it
The designed/overflow split was diagnostically essential, but spill
arrays make seam traffic legible and PERMANENT: a treaty with the
enemy. Candidate B and pass-1 localization (250 of 326 at R=64) are
negotiations over the tax rate; a streaming construction abolishes
the tax. Status: live; serviced by A6/A7.

### C3. CT factor choice is unraced at codegen time
Plan-layer evidence (the 1.5x stale-wisdom find) suggested the same
argument one level down: CT(4,16) vs CT(8,8) vs SR per (R, ISA) never
raced as a portfolio; VFFT_CT_FACTOR exists as an override, not a
searched dimension.
**Owner's verdict: DECLINED** ("we have no problem with that").
Recorded so the ledger shows a decision, not an omission.

### C4. The blocked construction knows one shape
Uniform-depth markers only: the recipe IR can express
[block][block][block][combine] and nothing else. No way to say "this
combine slice may start now", hence no fusion across the seam, which
is exactly the move streaming constructions make. Candidate B being
blocked on the marker format is the tell that the IR is the rigid
layer. Accessible version: prep-everything-then-assemble cooking; the
seam is the counter space holding every bowl at once; FFTW washes
bowls as it goes (liveness 8); our recipe grammar cannot express
washing early. Status: live; A6 is the grammar fix.

### C5. U is hardcoded
U=2 everywhere; U=3 pending for weeks. Unroll depth is a
schedule/ILP decision frozen at the construction layer. Status: live,
minor; rides the i9 audit (U=3 vs U=2 bench).

## Scheduler

### S1. It optimizes the wrong objective (HALF-WITHDRAWN by A1)
The charge: minimizing liveness conflicts with maximizing ILP, and the
MKL 1024 post-mortem credits their gap-closing to ILP scheduling.
A1's first measurement (sched_analyze.py): every kernel on BOTH sides
is p01 PORT-bound, CP slack 0.13-0.50; our critical paths are
SHORTER than FFTW's (8.7-10.3 vs 14.9 per 100 flops). Latency is not
binding, so the proposed dual-objective ILP scheduler buys ~nothing:
charge half-withdrawn, pre-registered bet falsified both ways. The
surviving half: our p01 pressure per flop is HIGHER (35.8-37.2 vs
32.9) via FMA fusion rate, 53% of flops through FMA vs FFTW's 68%.
A2 re-scoped to the fusion audit.

### S2. The schedule->regalloc phase split is a phase-ordering trap
Spills are inserted after the order is frozen, so spill placement
cannot reshape the order; the pin allocator is gated by a
hand-maintained whitelist (avx2 && n1 && radix>=16), and the t1s
family at 225 spill movs on avx512 was found by ACCIDENT days later.
Manual whitelists discovered by archaeology are not a policy. Status:
live; A3 (spills as schedulable IR nodes: allocator chooses WHAT,
scheduler chooses WHERE) + A4 (generate-both-count-keep replaces the
whitelist).

### S3. Zero feedback at the codelet layer
Wisdom exists for plans; codelets are one-shot feed-forward; no
schedule decision has ever been validated by its own measured CPE.
Status: live; A4 is the first step, full codelet wisdom is the
structural fix below.

### S4. Addressing isn't scheduled at all
308 GPR spills vs FFTW's zero at R=64: the scheduler and allocator
agree address arithmetic is nobody's job; it is loop-invariant code
motion we simply don't do. Owner: "good find." Status: live; A5.

## The structural charge
We built a world-class falsification apparatus around a ONE-SHOT
compiler: construction -> schedule -> allocate, each phase locally
decent, no phase informed by measurement, all empiricism applied at
the plan layer. The 9-second regeneration pipeline makes codelet
wisdom suddenly cheap: race {construction style x scheduler variant x
U} per (R, family, ISA), ship winners with provenance, the
architecture that already won the plan layer. Under this reframe A7
is not "implement one new construction" but "make construction a
searched dimension."

## Counter-goggles (calibration)
The 8-vs-35 liveness gap did not stop our leaves beating FFTW 2x on
split format, so liveness binds only for spill-heavy families and the
interleaved-parity fight, not everywhere. Per-codelet search risks
uarch overfitting; per-ISA codegen and provenance stamps are the
existing containment. And A1's first table is the proof the gates
work: the review's own biggest scheduler charge was half-killed by
the review's own first instrument.
