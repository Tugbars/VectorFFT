
## Tier A FALSIFIED (and what it proved)

Extracted the tan-factored const_cmul into a shared arith.ml so SR and
newsplit could use it instead of split_radix.ml's naive local copy.
Gated correct (SR vs CT 8.9e-16, newsplit vs CT 8.9e-16). Result:

  SR-16 flops 170 -> 170 (NO change); SR-32 461 -> 464; SR-64 1170 -> 1186
  newsplit-16 220 -> 233 (WORSE); -32 600 -> 668; -64 1594 -> 1846
  CT (production) byte-identical.

Net negative -> REVERTED (arith.ml deleted, both modules restored to
naive cmul, dune restored). Back to baseline (SR 170, newsplit 220).

THE DECISIVE EVIDENCE: after Tier A, CT and SR call the IDENTICAL
const_cmul, yet CT fuses to 40 FMA / 0 mul while SR stays at 4 FMA /
22 mul. Same multiply primitive, opposite fusion. This RULES OUT the
cmul form (and the module-layering hypothesis) as the cause, and pins
the blocker precisely on SR's COMBINE STRUCTURE: T1 and T3 each feed
BOTH U=T1+T3 and V=T1-T3 (two consumers), and a 2-consumer multiply
result cannot fuse no matter how it is emitted. This is the use-count
problem on shared intermediates, not a primitive problem.

CONSEQUENCE FOR TIER B (sharpened): the target is exactly the shared
T1/T3 -> U/V. Two options, narrow-first:
  B1 (narrow): use-count-aware UN-sharing of T1/T3 -- duplicate the
      shared twiddle-mul results so each feeds one add/sub and fuses,
      trading a few flops for instruction-count. Bounded; directly
      targets the proven blocker.
  B2 (deep): the full n-ary Plus rewrite + constant collection across
      sums (the codebase's named unlock for the R=25/R=64 gap too).
      Major algsimp surgery, full re-gate.
Recommend B1 first: it is the minimal change that tests whether
un-sharing the EXACT proven blocker recovers fusion, before committing
to the large B2 project.

## Update (this session): CT's fma_lift IS applicable to SR (doc-28 refinement)

Discriminating experiment (VFFT_FORCE_FMA_LIFT = remove the single_use gate = make CT's
fusion fire on SR), measured on SR n1 codelets:

| SR-16        | instr | fma | mul | flops | cyc/DFT-16 | max_err  |
|--------------|-------|-----|-----|-------|------------|----------|
| default      | 167   | 3   | 23  | 170   | 10.0       | 2.55e-14 |
| forced       | 152   | 24  | 8   | 176   | 9.7        | 2.55e-14 |
| CT-16/FFTW   | 144   | 40  | 0   | 184   | -          | -        |

Static instruction reduction holds at all sizes: N16 167->152 (-9%), N32 451->404 (-10%),
N64 1144->1016 (-11%). Correctness identical (2.55e-14). Runtime -3% at N=16 (latency-bound
ceiling; this session's r16 cliff finding limits the throughput payoff).

KEY: doc 28's +27% regression was R=32 *t1 (twiddled)* - heavily-shared precomputed-twiddle
muls with non-Add consumers => duplication GROWS op-count (910 vs 717). SR *n1* is the
opposite: light structured sharing => fusion SHRINKS op-count. The global single_use gate is
tuned for the t1 worst case and is needlessly suppressing a correct, free, modest SR n1 win.
=> Refines earlier "B1 pre-falsified": B1-style forced lift is a REGRESSION for twiddled t1
   but a (modest) IMPROVEMENT for no-twiddle SR. Action: codelet-class-aware lift gate
   (force-lift no-twiddle SR; keep single_use for twiddled t1). Full fusion to 40 fma (the
   multi-use shared blocks like t128 x3) still needs B2 (global fma), as FFTW demonstrates.

## BENCH VERDICT (supersedes static-count optimism above)

Hot bench, SR n1 default vs forced-lift, cyc/DFT-N, min of 2 passes x 3000 reps (L1-hot..mem):

| N  | K8 Δ%  | K32 Δ% | K128 Δ% | verdict           |
|----|--------|--------|---------|-------------------|
| 16 | -5.0   | -4.3   | -4.1    | FORCED (clear)    |
| 32 | +0.8   | +0.1   | -3.3    | tie               |
| 64 | -1.2   | +0.8   | -0.2    | tie               |

CLEAR WINNER: forced-lift at N=16 only (reproducible -4..-5% all K). N>=32 is a wash.
Static instr reductions (-9/-10/-11% at 16/32/64) did NOT translate at 32/64: forced lift
raises FMA-port-op count via duplicated products (N=64: 288 fma+mul vs 258), and that port
pressure cancels the instruction savings. Crossover ~N=16->32 (doc-28 mechanism, shifted to
larger radix for n1 than for t1). NOTE: a first-pass run showed N=32/K8 -22%, which was a
default outlier; the 2-pass re-run corrected it to tie. Bench, not static counts, decides.

ACTION: force-lift SR n1 at N=16, single_use for N>=32. Larger-radix gap to FFTW's 144-instr
fully-fused codelet remains B2 (global fma, fuses without duplication).

## CT vs SR HEAD-TO-HEAD BENCH (the decisive comparison)

cyc/DFT-N, best of passes, L1-hot..memory. Δ = best-SR vs CT (neg ⇒ SR beats CT):

| N  | K8     | K64    | K256  | K1024 | K4096 |
|----|--------|--------|-------|-------|-------|
| 16 | -1.0   | +7.3   | +1.1  | -0.7  | +1.0  |  (CT~SR-fma tie)
| 32 | +22.2  | -13.5  | +3.9  | +2.8  | +7.9  |  (CT wins, SR sweet spot @K64)
| 64 | +32.6  | -6.5   | +8.6  | -0.7  | +9.9  |  (CT wins big, SR sweet spot @K64)

WINNER: CT, clearly. Margin GROWS with radix at compute-bound K8 (-1% / +22% / +33% for
N=16/32/64) => this is SPILLS (SR-64 507 vs CT-64 307, no cut topology). High K does NOT
rescue SR: still +8..+10% behind CT at N=32/64 (memory-bound, extra spill/instr traffic rides
along). "SR wins at high K" = FALSIFIED. SR's only win = K=64 L2-residency band (-6.5..-13.5%),
reproducible but narrow (gone by K=256).

REFRAME: the SR<->CT gap is SPILLS, not FMA. Forced-lift closed most of the fma deficit yet SR
still loses by up to 33%. The real SR lever is CUT TOPOLOGY (dft_split_radix_spill) to bound
peak-live like CT's recipe cut, NOT more fusion. fma-fusion (this session) is a secondary term.
