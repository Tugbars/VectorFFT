# Large-N pass-minimization plan: escape the per-pass cliff

GOAL: cut memory traffic at large N by minimizing FFT passes (large radix), and fix the
per-pass cliff so the traffic cut converts to wall-clock. Origin: c2c-1024 (32,32) beat
(4,16,16) by 16%; MKL ships radices up to 96 for the same reason.

## The prize (exact, traffic = passes x 2N x 16B; total flops are INVARIANT to factorization)
| N     | r4 passes | best large-radix | traffic cut |
| 1024  | 5         | 2 (32x32)        | -60%        |
| 4096  | 6         | 2 (64x64)        | -67%        |
| 16384 | 7         | 2 (128x128)      | -71%        |
| 65536 | 8         | 3                | -62%        |
| 1M    | 10        | 3                | -70%        |
At 1024 the -60% traffic cut yielded only +16% wall-clock; the rest was eaten by the cliff
(radix-32 ~43.5 GB/s vs radix-4 ~136). So two halves, both required.

## Mechanism
total_time = #passes x per_pass_time.
  Lever A (#passes): ceil(log2 N / log2 R), fewer with bigger R.
  per_pass_time = max(FP-port compute, exposed memory latency).
  The 43.5 cliff = heavy codelet LATENCY-bound at large N (0.63 IPC, ports idle, loads not
  hidden). Lever B = hide per-pass latency -> per-pass BW toward 136.
A alone = the partial (32,32) win; A+B converts the full 50-71%.

## Phases (Phase 0 gates all; container = directional, metal = binding)
0. DIAGNOSE + CURVE. Traffic model done. Measure per-radix per-pass BW (4/8/16/32/64) at
   large N -> per_pass_time(R) curve + knee per N. Diagnose cliff: critical-path-vs-floor
   with L2/L3/DRAM load latencies + IPC/mem-ops-per-cyc per radix -> compute vs latency
   bound. Predict latency-bound => Phase 2 essential.
1. PASS-MIN FACTORIZER (biggest lever, radices <=64 already exist). Replace op-count plan
   selection in pick_algorithm/planner with cost = #passes x per_pass_time(R). Pick (32,32)
   @1024, (64,64)/(64,16) @4096, etc. Validate vs the 16%; sweep N.
2. LATENCY HIDING in heavy codelets (converts traffic cut -> speedup). If latency-bound:
   sw prefetch (next-tile/pass) + U=2->U=3/4 pipelining, enough independent loads in flight
   to cover L2/L3/DRAM latency. Goal radix-32/64 ~43.5 -> ~136 GB/s; raises the knee. Folds
   in the pending U=3-vs-U=2 bench.
3. LARGER/COMPOSITE RADICES (conditional). If knee for N>=16K exceeds r64: build r128 +
   composite fused radices (MKL 48/96 style) for the last-stage n1 hot path. Gate + BW each.
4. WISDOM/AUTOTUNE + METAL. Record best (factorization, prefetch, U) per (N,K) (generalize
   pending (32,32) entry; regen stride-wisdom). Binding numbers on i9-14900KF / EPYC 9575F;
   container memory regime is +-25% noisy + cache-confounded.

## Caveats
- Flops invariant to factorization => purely a memory-system optimization.
- Realized win bounded by how much of the traffic cut the cliff lets through => Phase 2 is
  the unlock, not a nicety.
- This is orthogonal to SR (dead end vs MKL) and to scheduling (codelets are port-bound at
  small K; this is the large-N memory regime).

## PHASE 0 RESULT (container, directional): heavy radices are LATENCY-bound at scale

Streaming ceiling (read+write, bytes/TSC-cyc): L1 ~50-100, L2 ~40-50, L3~=DRAM ~13.
(constrained 1-vCPU guest: L3 ~= DRAM; metal will differ. Direction is what's actionable.)

Codelet achieved B/cyc vs ceiling at matching working set:
| codelet @K (wkset)   | achieved | ceiling | verdict        |
| CT-16 @4096 (2MB L3) | 13.7     | 13.0    | BANDWIDTH-bound (at ceiling) |
| CT-32 @4096 (4MB)    | 6.9      | ~13     | LATENCY-bound (1/2 ceiling)  |
| CT-64 @4096 (8MB)    | 6.2      | 12.8    | LATENCY-bound (1/2 ceiling)  |
(L1 = compute-bound for all; L2 = below ceiling, compute/latency.)

DIAGNOSIS: light radix (16) hits the L3 bandwidth ceiling; HEAVY radices (32/64) stall at
~half the ceiling (below ceiling AND >> compute floor: CT-64 328 vs 49 cyc/DFT) => LATENCY-
bound, insufficient loads in flight to cover L3/DRAM latency, worsening with radix. CT-16's
13.7 B/cyc x ~3GHz ~= 41 GB/s ~= the remembered "43.5 cliff" (= L3 ceiling for light radix).

=> This is why (32,32) paid only 16% vs the 60% traffic cut: radix-32 stages latency-bound,
   half the bandwidth wasted. PHASE 2 (prefetch + U=2->U=3/4 deeper pipelining) is the UNLOCK,
   not polish: target pulling radix-32/64 from ~6 toward the ~13 ceiling (~2x). Phase 2
   confirmed as the lever; fewer passes only pays once the heavy codelets reach the ceiling.

NEXT: Phase 2 on radix-32/64 (sw prefetch next-group/next-pass + extend pipelining depth),
re-measure B/cyc at large working set, goal ~ceiling. Plus Phase 3 R128 validation (the
codelet exists as CT(8,16); needs correctness gate + B/cyc). Binding numbers on metal.

## PHASE 2/3 RESULTS + REDIRECT (container, directional)

PREFETCH (Phase 2 as framed) = DEAD LEVER:
- Generator has NO prefetch/unroll/pipelining knobs (plain `for k+=8` codelet loop; the only
  load-ordering is for the HW prefetcher in schedule.ml).
- Hand-injected sw prefetch into radix-64 @ K=4096, swept distance:
  | PFD       | cyc/DFT-64 | B/cyc |
  | off       | 289        | 7.1   |
  | 64        | 274        | 7.5 (+5%) |
  | 128/256/512 | ~277/277/286 | 7.4/7.4/7.2 |
  => +5% only. Heavy-radix stall is the ACCESS PATTERN (radix-64 = 128 strided streams;
     radix-128 = 256), not fetch timing. Prefetch can't fix stream-count/page/TLB pressure.
  Caveat: container L3~=DRAM; prefetch may help somewhat more on metal but won't close 2x.

RADIX-128 (Phase 3) = WORKS, but pass-count play not bandwidth play:
- radix128_n1_fwd_avx512: CORRECT (max_err 3.9e-13..5.4e-13). Massive: 2482 reg decls,
  864 fma + 136 mul + 1328 add/sub. Heavily spilled (peak live >> 32 ZMM).
- B/cyc: K8(L1) 22.5, K256 13.3, K4096(DRAM) 6.2  -- SAME as radix-64 at DRAM. No per-pass
  bandwidth advantage. Value = FEWER PASSES only (16384 = 128x128 = 2 vs 64x64x4 = 3).
  Available + correct for the 2-pass plan once per-pass cost is good (= blocking).

THE REAL LEVER = BLOCKED EXECUTOR (UNBUILT):
- stride_executor.h has plan hooks use_blocked/split_stage/block_groups and references
  core/planner_blocked.h -- which DOES NOT EXIST. Cross-pass-reuse cache blocking
  (four-step/six-step FFT = MKL's cFft_BlkSplit_32/64) was scaffolded but never built.
- Mechanism: heavy radices stall AND fewer-passes only banked 16% for the SAME reason --
  every pass re-streams the whole array from DRAM. Blocking (tile stays in L2, all passes
  on it, then next tile) removes the repeated DRAM traffic. This is what MKL does.

REDIRECT: drop Phase 2-as-prefetch. Large-N prize = build the BLOCKED EXECUTOR (four-step).
Implementation + correctness is container-doable; perf validation needs metal. R128 is
correct and waiting to feed the 2-pass 16384 plan once blocking makes heavy radices
cache-resident. Prefetch is off the table.

## R128 generation: the M-on / M5 recipe (CORRECTION to earlier bench)

Earlier R128 was generated as `128 --isa avx512 --su` with NO M-system env vars.
That left R128 OUTSIDE the regalloc gate (emit_c.ml: `log3 & avx512 & R<=32`
OR `avx2 & n1 & R>=16`), so the M-series allocator never ran and gcc spilled all
2740 __m512d values itself. The 6.2 B/cyc @K=4096 reported for that codelet is
the M-OFF variant and is SCRAPPED (never a real R128).

Correct R128 (M-on, 2-pass, M5 pre-spilling):
    VFFT_PIN_FORCE=1 VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1 \
      $GEN 128 --in-place --isa avx512 --su --emit-c
  - PIN_FORCE bypasses the R<=32/avx2 gate (force_pin in emit_c.ml ~1015)
  - USE_REGALLOC runs the M-series allocator (M3a/M5/M6)
  - USE_REGALLOC_M5 engages M5 pre-spilling for over-budget passes
  Confirmed engaged: log says spill_pass1 regalloc 1408 tags bound,
  spill_pass2 1040 tags bound; output has 2480 asm("zmm") bindings.

M-mode mapping (from emit_c.ml / regalloc.ml):
  M-off   = VFFT_NO_REGALLOC=1     (no pin, no fence; "true M-off")
  M-fence = regalloc gated off, fence on (current_fence_only path)
  M-on    = VFFT_USE_REGALLOC=1 (+ _M5 for overflow); pin + fence + allocator
  R128 requires M-on (fence-only cannot bind 2480 values to registers).

BLOCKER on re-bench: the M-on codelet is IN-PLACE, signature
(rio_re, rio_im, tw_re, tw_im, ios, me) — FFTW twiddle-codelet style, uses rio
as its own pass-1/pass-2 scratch (hence no big stack arrays). The standalone
out-of-place driver (in_re,in_im,out_re,out_im,...) mismatches args -> segfault.
A valid R128 number needs the executor context or a faithful in-place harness
(rio layout + internal CT(8,16) twiddle table + correct ios/me). Defer to metal
+ executor; container-standalone cannot measure M-on R128 cleanly.

## Codelet register-budget audit (peak-live, M-off production gen, avx512)

Triggered by: R128 was found generated M-off (overflow). Question: were other
codelets in MKL comparisons also over-budget under M-off? Method: VFFT_PEAK_LIVE=1
per pass, plus M-off-vs-M-on static stack-store counts to separate intentional
2-pass scratch from gcc pressure-spills.

Results (avx512, --in-place --su):
  R16 n1   single-pass  peak_live 66/32  OVERFLOW   <-- LATENT BUG
  R16 t1   2-pass       12/32, 10/32     fits
  R16 t1s  2-pass       12/32, 10/32     fits
  R32 n1   2-pass       17/32, 10/32     fits
  R64 n1   2-pass       17/32, 25/32     fits
  R128 n1  2-pass       34/32, 27/32     OVERFLOW (pass 1; needs M5)
  R16 n1 AVX2: 2-pass 9/16,10/16 fits (regalloc fires: in-gate avx2 arm)

M-off vs M-on stack stores (R32: 58 vs 54; R64: 173 vs 147) confirm the 2-pass
radices' stores are intentional scratch, NOT pressure spills -> M-off is correct
for R32/R64. So c2c-1024 (32,32), r2c-256 (8,32), and R16-as-twiddle in (4,16,16)
used correctly-allocated codelets. Headline MKL comparisons stand.

LATENT BUG: R16 n1 avx512 is emitted SINGLE-PASS (su_s1) and overflows 32 ZMM at
66 live (gcc spills ~10). The avx2 R16 n1 and the R16 t1/t1s variants are all
2-pass and fit. Fix: force R16 n1 avx512 to the 2-pass CT(4,4) decomposition like
its avx2/twiddle siblings (or bring it into the regalloc gate). Only affects
transforms using R16 as the FIRST stage on avx512 (e.g. (16,8)); moderate severity
(VFFT understated, not flattered, in any such case). CHECK c2c-128 factorization.

## Executor wired in on Linux/avx512 (build recipe) + R16 reframe

The core/ stride-executor + planner is the proper path (NOT standalone harnesses).
Linux/gcc build recipe (the demos' build scripts are Windows/icx):
  1. prototype/generated/ must have registry*.h (already present) + plan_executors.h.
  2. Compile codelets BOTH ISAs (executor always links the avx2 fallback lookup):
       codelets/inplace/avx512/*.c  with -O2 -mavx512f -mavx512dq -mfma
       codelets/inplace/avx2/*.c    with -O2 -mavx2 -mfma
     -> ar into one libcodelets.a (648 objects).
  3. gcc-13 -O3 -mavx512f -mavx512dq -mfma -I core -I prototype/generated \
       core/demo/demo_planner.c libcodelets.a -lm
  4. Run with K a MULTIPLE OF 8 (avx512 width). The demos hardcode K=4 (avx2 width)
     -> heap corruption on avx512 (8-wide ops overrun a K=4 buffer). Patch K=4->8.
  Result: all cells PASS correctness through the real planner+executor on avx512.

c2c-128 factorization (item #2): greedy AVAILABLE_RADIXES is largest-first
[64,32,25,20,19,17,16,...] -> 128 = (64,2). reorder_pow2_innermost keeps (64,2).
Wisdom (core/vfft_wisdom_tuned.txt): "128 4 2 16 8" -> at K=4 ONLY, wisdom uses
(16,8) = R16 n1 first stage. K=32 -> (4,32); K=256 -> (4,4,8). So the R16-n1
codelet is exercised by c2c-128 only at K=4.

R16 n1 avx512 (item #1) REFRAME — NOT a bug:
should_block_n1(n,vec_regs): default_min = (vec_regs<=16 ? 16 : 25); CT -> n>=min.
So R16 avx512 (32 regs) -> 16>=25 = false -> MONOLITHIC by design; R16 avx2 -> blocked.
Doc 58 calibrated this: monolithic R16 wins until peak_live > ~1.5x budget; threshold
set to R>=25 (R25 blocked measured +47% avx512). My "latent bug" claim was WRONG.
OPEN (empirical): doc 58 assumed R16 peak ~40 ("fits in 32"), but the CURRENT codelet
measures 66/32 (179 __m512d). If the codelet grew past the ~1.5x-budget crossover, the
R>=25 threshold may be stale -> test with VFFT_N1_BLOCK_MIN=16 (forces R16 avx512 to
block) and A/B via the executor. Container perf unreliable -> directional; metal binds.

R128 via executor (item #3): registry/plan_executors do NOT reference radix128_n1.
Wiring needed: generate M-on R128 codelet (VFFT_PIN_FORCE=1 VFFT_USE_REGALLOC=1
VFFT_USE_REGALLOC_M5=1 ... --in-place), add registry extern + a plan/lookup entry.

## Step 2 progress: emit_c mirror de-duplication (regression-gated)

Goal: collapse the 7 "Mirror of emit_c.ml line NNNN" hand-copies in
codelet_oop.ml into shared emit_c helpers (the emit_strided_* precedent),
move the one DAG-prep mirror to its proper home, delete stale comments.
Each extraction is gated by: (1) byte-for-byte regeneration of the 36-codelet
baseline (aggregate sha256 c3a6b57c3fd3a785bf14b76166d7c1bb992d696db318ecb0b09c19a24da0aaa7,
per-file manifest /tmp/step2_baseline.manifest), (2) benchmarks/run_t1_twiddle_gate.sh 24/24 PASS.

Mirror inventory (codelet_oop.ml line : description):
  [DONE] :664  inline_set cross-pass filter  -> Emit_c.filter_inline_set_cross_pass
  [DONE] :636  topo-sort  -> Algsimp.topo_sort_reachable (preds-based, NK_Plus-tolerant)
  [DONE] :724  emit_body_spill — decomposed (thin orchestrator over shared helpers; nothing left to extract; header de-staled)
  [DONE] :811  cluster-SU sub-block A (min_slot) -> Emit_c.compute_min_slot_pass1
  [DONE] :887  cluster-SU sub-block B (splitter) -> Emit_c.cluster_split_schedule
  [DONE] :926  fused-tag predicate -> Emit_c.is_fused_tag (3 copies collapsed; emission per-caller)
  [NOEXTRACT] :1177 PASS-2 store flush — divergent by design (regalloc+store-on-compute absent on OOP); comment de-rotted, not unified
  [DONE] :1291 spill-array decl — comment de-rotted (4-line decl, too small to centralize; left as code)

Module dependency note: Pipeline -> Emit_c (pipeline.ml uses Emit_c.spill_info,
Emit_c.make_spill_info); emit_c has ZERO Pipeline refs. So shared helpers whose
primitives live in emit_c (compute_inline_set/is_spilled/classify_passes) belong
in emit_c, NOT Pipeline — moving them to Pipeline would invert nothing but would
make Pipeline reach back into emit_c for 4 primitives. The layering-correct home
is emit_c as exported pure helpers. (Corrects the earlier "move inline_set to
Pipeline" plan: that was written before confirming the primitives are emit_c-resident.)


### :636 topo-sort extraction — NK_Plus policy subtlety (resolved)
The "Mirrors emit_c.topo_sort_reachable" comment was inaccurate. codelet_oop's
copy traversed via Algsimp.preds (NK_Plus-TOLERANT); emit_c.topo_sort_reachable
hardcodes the match and calls nk_plus_unreachable on NK_Plus (NK_Plus-FATAL,
the repo-wide "fail loud during migration" guard, also at algsimp.ml 764/840/900/1375).
Naively pointing codelet_oop at emit_c's version would have INJECTED a migration
guard into the OOP path. Resolution: homed a preds-based topo_sort_reachable in
Algsimp (clean base layer, no emit_c/codelet_oop deps; both already depend on it).
emit_c's 11 guarded call sites are UNTOUCHED. No behavior change (byte-identical regen).


### :926 fused-tag extraction (the highest-drift-risk one — live M-project semantics)
GATE COVERAGE FIX FIRST: the original 36-codelet baseline had fuse=0 only, so
the fused-tag path (forward-declare / no-spill-store / skip-reload) was DEAD CODE
and the gate gave it ZERO coverage. A broken extraction would have passed both
gates. Extended baseline to 48 codelets (+12 --fuse 2 R>=25: OOP t1p fwd/bwd,
in-place n1/t1), confirmed to engage the fused path (16 forward-declared tags
each). fuse=2 verified numerically correct AND result-identical to fuse=0
(fuse is perf-only). Also fixed a latent gate flaw the extension surfaced: byte-diff
was invocation-PATH-sensitive (provenance stamps argv[0]); now normalized out via
benchmarks/diff_step2_baseline.sh. Locked normalized aggregate: 5e7f772c...

EXTRACTION: there were THREE copies of is_fused_tag — emit_c.ml:786 (an already-
exported one nobody used), emit_c.ml:1784 (a local shadow in emission code), and
codelet_oop.ml:927 (the mirror). All three now resolve to Emit_c.is_fused_tag.
The forward-declare EMISSION is intentionally NOT unified: emit_c emits one
combined decl line (Isa.forward_decl), codelet_oop emits one-per-line; they
produce different C by design. Per the policy/mechanism split: extract the
DECISION (predicate), leave the printf sites per-caller. PASS-2 skip-reload
already routed through the shared Emit_c.is_fused_slot. Gates: 48/48 byte-identical
(incl fuse=2), t1 24/24, fuse2 n1 numeric PASS.

New durable gate infra (benchmarks/): snapshot_step2_baseline.sh (48-codelet regen),
diff_step2_baseline.sh (provenance-normalized byte diff), gate_fuse_n1.c (fuse numeric).


### :1177 store flush — NOT extracted (designed divergence, not drift)
Read both sides in full. emit_c's flush_cluster_stores is entangled with two
optimizations the OOP path does NOT have: M5 regalloc (current_regalloc
spill_sites/reload_sites emission at each position + final_pos + clear_alloc) and
store-on-compute (soc_assigns_by_tag / soc_stored inline stores). Genuinely-shared
logic = ~10 lines (assigns_by_cluster grouping + the `prev <> now -> flush prev`
boundary-detect pattern). The flush bodies, tail sweeps, and idempotent guards
(flushed_tags vs soc_stored) diverge for real reasons. Extraction shape would be
core + 2 closures (emit_reload_if_needed, emit_output_store) + 3 feature-flags
(regalloc?, soc?, which-guard?) — highest extraction risk in the file for ~10 lines
of yield, while FIGHTING an intentional capability gap rather than preventing
accidental drift. Same situation as uarch. Resolution: replaced the rotting
"mirror of emit_c.ml lines 2114-2140" comment with an accurate divergence note
stating the relationship + the convergence condition (when OOP gains
store-on-compute/regalloc, the feature gets built once). Comment-only change;
48/48 byte-identical. The analysis ranked this #2-by-value; reading the source
overturned that — the payoff was smaller than the labels suggested because the
divergence is a designed gap, visible only in code. (Prioritize-by-drift-risk
compass was right; this specific target's yield was not.)

Revised remaining order: cheap-half cluster-SU (min_slot + groups-fold splitter,
~50 verbatim lines, no closures/flags — now the best-remaining-yield) -> emit_body_spill
decompose last -> :1332 leave -> end-pass comment sweep (convert surviving
line-number mirror refs to function-name refs).


### Cluster-SU PASS-1 extraction (sub-blocks A + B) — DONE, 2 gated commits
Decision flipped from my B-only lean by a second-Claude review. The deciding
arguments: (1) the latent ordering dependence (List.rev pass1_nodes) lives in
emit_c TODAY and is the WORSE copy; extracting sub-block A with an explicit
descending sort DELETES the dependence rather than preserving it, and since the
producer (topo_sort_reachable) is already single-sourced in Algsimp, leaving A
forked means a unified producer with a forked consumer. (2) The de-dup unit is
the cross-file mirror (emit_c PASS-1 vs codelet_oop PASS-1) — those two ARE one
function; PASS-2's different shape is a separate candidate pair, not a reason to
skip PASS-1. (3) THE STATIC REGIME IS ENDING: the SR-seam blocked-newsplit work
modifies exactly sub-blocks A and B (non-uniform cluster ranges, a cluster_of
that isn't min_slot/ct_n2, possibly different sink predicates). Single-sourcing
BEFORE that edit is the whole point — otherwise SR lands in emit_c, codelet_oop
keeps old semantics, M-gate class strikes in the executor's hot codelets.

Two shared helpers in emit_c (home: needs spill_info which is emit_c-resident;
both callers already depend on Schedule):
  compute_min_slot_pass1 sp pass1_nodes -> (min_slot, pass1_blocked_topo)
    [sub-block A; explicit descending sort, not List.rev; Hashtbl.replace]
  cluster_split_schedule sp ~pass1_blocked_topo ~min_slot ~schedule_cluster
    [sub-block B; ct_n2<=0 guard INSIDE; one closure = the su-vs-bb policy]
Design notes honored: cluster_of NOT parameterized (kept min_slot/ct_n2 inline —
tomorrow's SR parameterization becomes a one-place change with the real
requirement in hand); ct_n2<=0 guard moved inside the helper so no future caller
(SR path) can forget it. uarch/gh selection stays per-caller (differs by design).

Both commits: 48/48 byte-identical (incl fuse=2), t1 24/24, fuse2 numeric PASS.
The byte gate on commit A is load-bearing: it confirms List.rev->explicit-sort is
byte-invisible across all 48 codelets, i.e. the tag-ordering invariant holds.

### Rule for remaining mirrors (write-it-down distinction)
- Divergence by DESIGN (different capabilities/interfaces; unifying fights an
  intentional gap): document + leave. Examples: :1177 store flush (regalloc +
  store-on-compute absent on OOP), uarch selection (CLI vs hardcoded default).
- Divergence by ACCIDENT (identical semantics under cosmetic texture): unify.
  Example: cluster-SU PASS-1 (just done).


### :724 emit_body_spill + end-pass comment sweep — DONE
:724 resolved as the reviewer predicted: nothing left to extract. emit_body_spill
is now a thin orchestrator calling the five shared helpers (classify_passes,
filter_inline_set_cross_pass, compute_min_slot_pass1, cluster_split_schedule,
is_fused_tag/is_fused_slot) plus OOP-specific emission (render_node_def, spill
store/reload, OOP output stores). What remains is divergence-by-design (no
regalloc, no store-on-compute, OOP store patterns) — correctly NOT shared.

Comment sweep (comments contradicting freshly-shared code died with the extraction
that touched them; line-number refs converted to function-name/behavioral refs):
  - emit_body_spill header: removed false "no SU scheduler" / "fuse=0" bullets
    (these fooled an earlier careful read); now lists the shared helpers + the
    OOP-specific divergences explicitly.
  - phase-1 BUTTERFLY BODY header: removed false "NO scheduling, NO register
    allocation, NO fence/pin emission"; states fence + Tier-C SU are wired,
    regalloc/pinning genuinely deferred (render-convention blocker).
  - t1 twiddle "may need fixup / needs verification": removed; states VERIFIED
    (flat/log3/t1s/t1p x fwd/bwd x R16/32/64 to ~1e-12, run_t1_twiddle_gate.sh).
  - two leftover sub-block headers (min_slot, cluster-split): now reference
    Emit_c.compute_min_slot_pass1 / cluster_split_schedule by name, no line nums.
  - spill-array-decl "Mirror of emit_c.ml lines 1322-..." -> behavioral note.
Result: ZERO surviving line-number mirror refs in codelet_oop.ml; zero stale
"NO scheduling"/"needs verification" comments. All comment-only (48/48 byte-identical).

## STEP 2 SUMMARY: 5 mirrors collapsed, 2 designed-divergences documented+left, 1 decomposed
Shared helpers created: Algsimp.topo_sort_reachable; Emit_c.{filter_inline_set_cross_pass,
is_fused_tag (3 copies->1), compute_min_slot_pass1, cluster_split_schedule}.
Gate infra: benchmarks/{snapshot_step2_baseline.sh, diff_step2_baseline.sh (provenance-
normalized), gate_fuse_n1.c, run_t1_twiddle_gate.sh, gate_t1_twiddle.c, gate_t1p_twiddle.c}.
Locked baseline: 48 codelets (incl fuse=2 covering the fused-tag path),
normalized aggregate 5e7f772c0268bdb89631139a709efc191d393d4def8c0d7c9c9969b3d70d6e87.
Left by design: :1177 store flush, uarch selection, gh predicate, spill-array decl.
Remaining future work (separate candidate, NOT step 2): PASS-2 cluster scheduling
cross-file mirror (emit_c vs codelet_oop PASS-2) — evaluate on its own merits.

## STEP 3 (designated, NOT yet done): the construction-selection chooser
THE live drift risk standing between current state and safely landing the SR seam.

Problem: codelet_oop's prepare_butterfly and gen_main's recipe_applicable +
construction branch independently select the math-layer construction
(dft_expand / dft_expand_n1_blocked / dft_expand_twiddled / _spill / _il2 / hc
routes) from the same Dft.should_spill / Dft.should_block_n1 predicates. This is
a cross-file mirror, and gen_main's branch has ALREADY grown cases codelet_oop's
copy lacks (dft_expand_twiddled_il2, hc2hc/hc2c cascade routes, the --strided
exclusion). The newsplit-blocked / SR prototype patches gen_main's
recipe_applicable + construction branch but NOT prepare_butterfly — so the moment
any SR-blocked construction gating lands in gen_main, the OOP path silently emits
SR-monolithic where the in-place path emits SR-blocked: M-gate failure class, in
the math-layer dispatch, on already-benched work.

Fix (cheap, no closures): a single
  Dft.select_expansion ~twiddles ~radix ~vec_regs ~policy ~direction ~sign
    -> (assigns, spill_markers, ct)
consumed by both gen_main and prepare_butterfly. Home = Dft (the chooser runs
UPSTREAM of Pipeline's input; not Pipeline). Policy predicates already live in
Dft; only the dispatch is duplicated. Gate with the same 48-codelet regen
baseline. DO THIS BEFORE TOUCHING THE SR SEAM BRANCH AGAIN.
When it lands, it also dissolves the codelet_oop construction-selector comments
(now de-rotted to behavioral refs + flagged as the step-3 target).

## Minor verdicts written down (so they don't need re-deriving)
- PASS-2 cluster-SU (min_input_slot fixpoint + array-bucket SU, codelet_oop
  ~:995): ASSESSED in Q2 and left unshared — a SEPARATE candidate pair from the
  PASS-1 cluster-SU (which is shared). PASS 2 uses a different mechanism, so the
  PASS-1 helpers don't serve it. Whether emit_c/codelet_oop PASS-2 copies are
  near-verbatim is its own evaluation, likely alongside step 3. (Now annotated
  in-code at the PASS-2 region.)
- All remaining ".ml lines NNNN" line-number references in codelet_oop.ml are
  GONE (full sweep, including gen_radix/gen_main/emit_c refs and the fuse-default
  ref). Comments reference functions by name or describe behavior.

## NEWSPLIT integration (scaled conjugate-pair split-radix) — DONE, additive + gated
Integrated the uploaded SR change set (split_radix_v3_blocked.ml + dft/gen_main patches):
  - lib/split_radix.ml: whole-file replace with the superset. Confirmed dft_split_radix
    body byte-identical to prior; const_cmul upgraded to the tan-factored / K-shared form
    (matches Dft.const_cmul). Adds newsplit_core (factored newsplit0/S/S2/S4 + combine0)
    and dft_newsplit_blocked (E/O1/O3 spill seam, fake ct=(4,n/4)).
  - lib/dft.ml: hunk 1 (SR dispatch under VFFT_NEWSPLIT) was ALREADY present via the
    newsplit_enabled () helper (more robust than the patch's inline getenv — treats ""
    as off). hunk 2 applied: added dft_expand_newsplit_blocked (marker-contract wrapper).
  - lib/gen_main.ml: both hunks applied — recipe_applicable + construction branch gate
    the blocked construction on (Split_radix && n>=32 && VFFT_NEWSPLIT).

Gated behind VFFT_NEWSPLIT=1. REGRESSION: with the flag OFF, the 48-codelet baseline is
byte-for-byte identical (integration is purely additive) and t1 gate stays 24/24.
CORRECTNESS (flag ON): newsplit monolithic AND blocked both numerically correct —
R=32 1.28e-12, R=64 4.44e-12, R=128 1.16e-11 (fwd), bwd compiles. Blocked path engages
the spill+cluster machinery via the fake ct (130/258/514 spill refs at R=32/64/128).
New durable gate: benchmarks/run_newsplit_gate.sh.

Perf note (from uploaded bench, directional/KVM): newsplit MONOLITHIC beats CT-monolithic
by 23-42% (fewer ops) but LOSES to the CT production recipe by 27-45% — the spill recipe
dominates instruction count. At R=32 newsplit emits 0 FMA / 56 mul vs CT's 62 FMA / 6 mul
(tangent scaling shifts the mix toward standalone mul+add, which the FMA-lift can't fuse).
The blocked-newsplit-vs-CT-recipe fight (both blocked) is the measurement that decides
whether newsplit is worth carrying to production; not yet benched on metal.

## STILL OUTSTANDING: the construction-chooser drift (step 3) is now LIVE in the tree
The gen_main patch added the newsplit blocked construction to gen_main's recipe_applicable
+ construction branch but NOT to codelet_oop's prepare_butterfly. So under VFFT_NEWSPLIT,
the in-place path emits newsplit-blocked at Split_radix R>=32 while the OOP path's
prepare_butterfly has no newsplit branch and emits CT. This is exactly the predicted
divergence, now materialized. Dft.select_expansion (the step-3 chooser) should be landed
to close it before relying on newsplit through the OOP/executor path.

## M-PROJECT DEFAULT FLIPPED TO OFF (2026-06-09)
Decision: pin and fence default to OFF; machinery kept as opt-in. No metal sweep
(evidence consistent across t1/t1s/log3/n1 x avx2/avx512 x R=4..128 on gcc-13;
a confirmation sweep judged ceremony, not information). No deletion.

WHY: M-on is net-negative or tie in every gcc-13 cell measured. Mechanisms:
  - pin: round-robin zmm/ymm assignment fights gcc-13 coalescing (log3 R=4: 10
    reg-reg copies on an 8-reg working set, ~10% cost; avx2 R=32: ~25%).
  - fence: fragments live ranges AND defeats operand folding (t1s: fence un-folds
    an embedded {1to8} broadcast into a named register, +9%; R=64 avx512: ~11%).
M's original protective function (force FMA contraction, block remat on gcc-11)
is obsolete: we IR-lift FMAs and gcc-13 fuses unaided. "The opponent moved."

CHANGE (lib/emit_c.ml gate): regalloc_enabled = not opt_out && force_pin;
fence_enabled = not opt_out && (force_pin || force_fence). Retired auto-pin
niches (log3+avx512+R<=32; avx2+n1+R>=16) and the blanket fence default.
Opt-in preserved:
  VFFT_PIN_FORCE=1   -> pin+fence (deterministic cross-compiler asm: the one
                        surviving M motivation, at measured runtime cost)
  VFFT_FORCE_FENCE=1 -> fence-only (diagnostic / 3-state A-B); NEW knob
  VFFT_NO_REGALLOC=1 -> all off (now also the default)

GATE FOR THIS CHANGE: NOT byte-identity (it intentionally changes output). The
flip changed exactly 36/48 baseline cells (every R>=16 that used to pin/fence);
12 small-radix cells the gate never touched stayed byte-identical (scope correct).
Numeric correctness preserved across the flip: t1 gate 24/24, n1 fuse2 R32/64
1.44e-12/4.10e-12, previously-pinned avx2 R32 1.44e-12 — all PASS. Re-locked the
M-off baseline: normalized aggregate 99967a5031dc72c863f1b63540e930f621cc56c9179ccc5abeca48bb522de51e.
The pre-flip (M-on) aggregate was 5e7f772c...; superseded.
