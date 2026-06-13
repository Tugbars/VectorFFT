# The DAG and scheduler program

Outcome of the criticizing-goggles session (lab notebook sections
42-44). Each item carries its evidence, its gate, and its decision
rule. Container items are runnable now; i9 items wait on hardware.
Standing discipline applies: pre-register predictions, counts rank,
races decide, bit-exactness is a hard gate, identity-gate every
migration.

## P0 — measure before changing (container, days)

### A1. Schedule quality analyzer (gate for everything below)
Build the missing instrument: critical-path length and port-pressure
histogram of an emitted schedule, computed from the uarch tables we
already carry. Run it on our r32/r64 (in-place and t1s) vs FFTW's
t1fv_32-avx512 and MKL's 1024 kernel (disasm-level for theirs).
Evidence behind it: the MKL post-mortem attributes their gap-closing
to ILP scheduling; this session's port arithmetic was the first
port-level look ever taken. Pre-register before running: whose
critical path is shorter at r32, ours or MKL's. No scheduler change
ships without before/after numbers from this tool.

### A2. FMA fusion audit (RE-SCOPED by A1's first measurement)
A1 v1 results (generator/cost_model/sched_analyze.py): every kernel
on both sides is p01 PORT-bound with CP slack 0.13-0.50; latency is
not binding anywhere, so the dual-objective ILP scheduler this item
originally proposed would buy ~nothing (charge 1 half-withdrawn; my
pre-registered CP bet falsified BOTH ways: our CP/100fl 8.7-10.3 vs
FFTW 14.9, and our p01/100fl 35.8-37.2 vs their 32.9). The real
kernel-layer deficiency: FMA fusion rate, ours 53% of flops via FMA
vs FFTW's 68%; unfused mul+add pairs waste slots on the binding port,
worth ~10-12% of p01 pressure. New scope: find where algsimp/emit
declines to fuse (association order, CSE splitting mul-add pairs,
missing fnmadd/fmsub patterns); gates: fma-fraction count toward
~68%, bit-exact NOT required here (re-association changes rounding;
gate becomes max-ulp vs naive DFT at machine-precision class), races.
Pressure-aware scheduling survives only as support inside A3.
STATUS: COMPLETE. The count: ZERO fusable single-consumer mul->add
pairs at r32_n1 and r32_t1s; the apparent mul/add pool is 6 muls +
240 bare add/subs (n1) — every fusable multiply is already fused, the
strays are CSE-shared multi-consumer (correctly unfused). Emission is
optimal; the gap is CONSTRUCTIONAL. Family-mismatch correction: like
for like the FMA-fraction gap is 57% vs 68% (twiddled) and 53% vs
~57% (n1). The metric that matters at p01-bound kernels:
binding-port uops per flop, ours 0.716-0.734 vs FFTW 0.658. Routed
to A7 as a second gate metric (p01-uops/flop toward 0.66 alongside
liveness); A8's entry gate sharpened: SR earns its reserve slot only
if its DAG buys FMA shapes CT+CSE cannot.

### A3. Spills as IR nodes + reschedule round
Allocator chooses the spill SET and emits spill-store/load as
first-class DAG ops; scheduler runs again on the augmented DAG and
owns spill PLACEMENT. Two-round convergence. Gates: spill counts <=
current at equal liveness, races, bit-exact. Completes the
phase-separation answer to charge 2.

### A4. Kill the regalloc whitelist
Replace `is_avx2 && is_n1 && radix >= 16` with generate-both,
count-statically, keep-winner, stamp-choice-in-provenance, per
codelet. The 9-second tree makes this free. Gates: tree regeneration
identity except intended flips; a counts table in the doc; the t1s
left-on-the-table failure mode becomes impossible by construction.
Note: the t1-family pin itself stays env-gated until the i9 A/B
(section 43 decision rule, >=2% on 32x32 TT at K>=256).

### A9. Escape the pow2-stride L1 aliasing (NEW, measured +16%)
Found while prosecuting S1 to Tugbars's "attack it committedly": the
4x gap between r16/r32 measured cycles and their port floors
decomposes as ~2.4x L1 SET-ALIASING (pow2 ios folds 16-32 element
lines onto 2 cache sets, past 8-way associativity; ios=264/320/8 all
run ~128 cy/group vs 305-310 at ios=256, floor 72) times ~1.8x
residual (avx512 downclock + TSC accounting + window/latency, i9
measures the true split). PRODUCTION CONTEXT VERIFIED: 32x32 TT plan
at K=264 runs 15.8-28.6% faster per point than K=256. FFTW's
dft-buffered + memcpy wrapper stands REINTERPRETED: the copy pass is
the aliasing dodge, not overhead. Plausibly explains the parked
K=256 L3-dip (aliasing bites exactly where data is L1-resident;
moot at K=1024 which streams from L2/L3). Options by invasiveness:
(a) padded-K guidance/layout in libvfft (the split layout is ours to
define), (b) internal lda=K+pad, (c) leaf staging buffer. Scope:
L1-resident K. Gates: per-point ns at matched work, correctness at
machine precision, no regression at large K.

### A10. Def-use-spacing scheduler (S1's residual, demoted, i9-gated)
The committed S1 prosecution's full record: MKL's ILP = def-use
spacing median 12 vs our 4 (colbatch interleaves batch columns),
carried by deliberately worse static profiles (twiddle kernel
p5-BOUND at 51.9/100fl, FMA:mul 0.4-1.3). Our scheduler exonerated
three times (efficiency 0.24 identical for SU and gcc-free orders;
defencing flat with +90 spills; short-body r16 flat). Surviving
headroom = whatever of the ~1.8x residual is window/schedule on quiet
hardware with clean strides (A9 first). Gate TIGHTENED after the fuse-M probe: fuse-2 (the M
project, emit_c ?fuse=M interleaving pass-2 sub-DFTs, threaded into
the in-place blocked path; bit-exact, 113-line diff, spills 103 vs
105) ran <=2% at clean stride ios=264, so window starvation is
triply dead in container (short bodies flat, gcc reorder flat,
fuse-2 flat) and the residual attributes toward downclock/TSC
accounting + latency model. A10 fires only if the i9, AFTER A9
padding and downclock-corrected cycle accounting, shows >10%
residual; first implementation attempt is a fuse-M sweep (existing
machinery), then VFFT_SCHED_OBJ=spacing (Goodman-Hsu, median
def-use >= 10 at vec-liveness <= 28). Terminology fixed on record:
VFFT_COLLECT_M is an unrelated algsimp multiplication-collection
pass (default-off pending measurement), not the M project.

## P1 — the construction floor (container, days-to-week)

### A5. GPR addressing lane — MERGED INTO A7 by experiment
Diagnosis: our emitted `e*ios + k` indices get CSE'd by gcc into ~64
long-lived scalars; 14 GPRs cannot hold them; 298 spills at r64.
FFTW: 0 imul, 131 lea, 4 spills. Both standalone fixes FALSIFIED:
flag sweep (best -fno-ivopts, -13% gpr for +7% vec, a trade) and the
root-sum transform (imul 0, lea 165, FFTW's exact mix, yet gpr
298->404 because gcc CSE'd the sums instead). The representation was
never the problem: LIFETIMES are. Our emission spreads 64 loads
across the whole body so all addresses are live at once; FFTW's
wave-ordered DAG keeps ~8 live. GPR pressure is therefore an
emission-order property, the same defect as the vector seam one
register file down. A5 folds into A7 as its third gate metric (GPR
spills toward ~0). Parked curiosity: the root-sum transform cut VEC
spills 323->279 via reassociation; possible future lever.

### A6. Candidate B: mixed-depth marker format
The IR grammar extension that lets a combine slice start before all
blocks finish. Prerequisite for any seam fusion; its absence is why
the construction knows one shape ([block]...[combine], uniform depth).
Estimated half-day in emit_c classify_passes per the earlier sizing.

### A7. CT-streaming emission (the liveness prize, on CT arithmetic)
Tugbars's correction folded in: SR's arithmetic advantage is ~nil vs
CT+CSE, so the prize was never the algorithm, it is the
narrow-frontier EMISSION ORDER. Reshape CT construction to emit
depth-first along dependence chains with combine slices fused at the
seam (via A6). Targets, with measured ceilings attached: the t1s
construction share (~114 spill movs after pin), R=64 pass-1 overflow
(250 of 326), the 32-real CT(4,4) seam vs FFTW's 8-liveness frontier.
Gates: liveness profile from A1, spill counts, races, bit-exact.

## P2 — exploratory, only if P1 stalls

### A8. SR module rehabilitation
Tugbars's offer ("fully commit and try to make it rise to CT's
level") held in reserve: only if A7 cannot reach frontier targets on
CT. Entry gate: SR must first match CT+CSE op counts at matched radix.

## Declined by design owner
CT-factor portfolio search per (R, ISA) at codegen time: raised in
review, declined by Tugbars ("we have no problem with that").
Recorded so the ledger shows a decision, not an omission.

## Standing i9 queue (unchanged by this program, runs first)
0. Wisdom regeneration on current codelets (~1.5x stale, the
   116-candidate race harness is the tool).
1. Phase-6 hardware audit: avx2 leaf three-deep A/B, t1s pin A/B,
   plan-level magnitudes, U=3 vs U=2, estimate_plan recalibration.
