# MKL comparison — geometry contracts (the baseline-complete record)

*2026-07-15, bench container, matched ISA (MKL_ENABLE_INSTRUCTIONS=AVX2),
canonical bench_1d_vs_mkl protocol (isolated single-cell, order-neutralized,
best-of-5, paced/cooled). Same instrument for every row; only the MKL
descriptor differs. Container ratios are directional; the 14900KF re-run is
the arbiter for absolutes.*

## Library identity (positioning summary)

A **lane-major-geometry** FFT engine — locked there by the vectorization
axis (lanes span the batch, buying shuffle-free butterfly arithmetic) —
supporting **both split and interleaved storage** — split native in the
arithmetic interior; IL by integration layer: (i) SIMD converters
(il2sp/sp2il, any plan, the unfolded fallback), (ii) **integrated 1D DIT
boundary adapters** (fwd_ilin entry fold + bwd_ilout exit fold, all 18
stage-0 radices, il_execute.h — the layer behind every measured IL result
incl. the ~6% roundtrip tax and the vs-MKL-IL wins), (iii) **all boundary folds
WIRED** (variant-aware DIT exits, DIF entry/exit, bwd entries; il2il z->z
orchestrators; adapter gate 7 plans ALL BIT; fwd fold gain measured
+3.4-6.7% = the sweep; MKL-IL-lane odd small-N measured strong, 0.44-0.53x
at N=100/1000 — see section 6a16 below), (iv) public-API layout flag and
fftnd/2D/3D executor IL **not developed** (multiD zero-sweep compositions
exist as bench code only) and **both scrambled and natural output
order** (scrambled native per the convolution contract; natural productized
via FREE/PURE/PSWAP + @nat wisdom, bin-for-bin gated). Transform-major
geometry is consumed (strided in-register-transposing codelets, the
2D/3D/ND row passes) but never the engine's home; small-K/K=1 is the
geometry lock's cost regime, with four-step decomposition as the in-house
remedy. Fine print: natural-order x IL output composition is currently
unfused (reorder runs on split).

VectorFFT is a **lane-major** engine by construction: SIMD lanes span the
batch axis (element i of K transforms per vector), which is what makes the
butterfly arithmetic shuffle-free. MKL's native geometry is **transform-major
contiguous** (lanes span consecutive elements of one transform). One geometry
per design, dictated by the vectorization axis. Any comparison therefore
carries a contract:

| (N,K) | vfft split fwd | vs MKL split (v1.0 protocol) | vs MKL IL, lane geometry | vs MKL IL, contiguous |
|---|---|---|---|---|
| 1024, 4  | 13.0-13.2 us | 1.48-1.53x | **1.20x** | 0.69x |
| 1024, 32 | 92-95 us     | 1.65-1.69x | **1.45x** | 0.78x |

- **Lane-batched contract** (caller's data lane-major — this library's
  design center): the fight is column 3. VectorFFT beats MKL's native
  interleaved kernels at equal geometry, and the margin grows with K.
  The folded-IL path adds ~6% roundtrip on top (measured; fwd currently
  ~+29% pending the DIF-forward exit adapter, projected ~+6% after — the
  n1 fwd il_out codelets are generated and gated, wiring only).
- **Transform-major contract** (caller holds contiguous per-transform
  arrays): column 4 is MKL-as-users-run-it. The 0.69-0.78x contains NO
  repack cost on either side (both engines on native buffers); a
  ctg-contracted caller adopting this engine would pay a repack on top.
  Serving this contract natively is a design decision, not a tuning gap —
  the strided codelet family (in-register transposing consumers) is the
  existing ingredient if it is ever wanted.
- **K=1** is the one cell where the geometries coincide and no contract
  caveat applies; the lane-batched design is structurally quarter-filled
  there (measured 0.19x raw). The in-house remedy is four-step
  decomposition (manufactures internal batch); the alternative is a
  within-transform codelet family.

Historical note: the v1.0 §1 protocol ran MKL in REAL_REAL split storage at
identical layout (bench source, line ~288) — fair as a same-layout kernel
comparison, but MKL's split path is a neglected mode measured 2.3-8.5x below
its CCE-interleaved kernels (see §2b correction + il_architecture 6a14). The
three-column table above is the baseline-complete replacement: same-geometry
native (we win), MKL-best-geometry native (we lose, shrinking with K), and
the legacy split column for continuity.

## 6a16 — Adapter wiring: variant-aware boundary folds + il2il orchestrators

Every boundary fold is now WIRED in il_execute.h (was: codelets gated,
adapters absent). Fold matrix, all resolved from the stage struct
(st->use_log3 / st->t1s_fwd) — no wisdom parsing:

| boundary | DIT | DIF |
|---|---|---|
| fwd entry (z->split) | n1 il_in (pre-existing) | t1_dif[_log3] il_in (NEW) |
| fwd exit (split->z) | t1s / t1 / t1_log3 il_out by variant (NEW) | n1 il_out (NEW) |
| bwd entry (z->split) | n1_bwd il_in + conj(cf_all) on split (NEW) | n1_bwd il_in (NEW) |
| bwd exit (split->z) | n1 il_out (pre-existing) | t1_dif[_log3]_bwd il_out (NEW) |

Plus fully-folded z->z orchestrators: vfft_proto_execute_fwd_il2il_core /
bwd_il2il_core (entry fold + split interior + exit fold; z_in==z_out safe;
single-stage plans fall back with -1). Interior runs LOCAL generic-range
copies (_vfft_il_{fwd,bwd}[_dif]_range, confined to il_execute.h) — jit
executors are start_stage-gated only, so a stop gate (DIT exit / bwd-entry
resume at jit tier) is the standing follow-up. Executor fixup collision
audit: all fwd fixups (cf0 / cf-all-legs) apply on SPLIT before codelets ->
il_out folds clean; bwd conj(cf_all) applies on SPLIT after n1_bwd -> il_in
entry folds clean; DIF has cf0=1 universally, no fixups anywhere.

**The log3 lesson.** emit_jit's "bwd is variant-agnostic" is true of the
STAGE macro, NOT the fn pointer: the plan binds st->t1_bwd per variant, and
t1_dif_log3_bwd addresses W differently from plain t1_dif_bwd. The adapter
gate caught it precisely: DIF-bwd exit fold failed ONLY on LOG3 plans
((100,4): 40 BAD; (100,67): 843 BAD; FLAT: BIT). Fix: t1_dif_log3_bwd
il_out family (generator names log3 BEFORE bwd), resolver selected by
st->use_log3. Gate-first doctrine paying for itself.

**Gate evidence.** Codelet layer this campaign: +144 (t1_dit / t1_dit_log3
fwd il_out, t1_dif_log3 fwd il_in; r{4,5,8,10,16,20,25,32} x 2 ISAs x
me{64,65,67}) +36 (r10/r20 t1s/dif/difb backfill) +48 (t1_dif_log3_bwd
il_out) — all BIT under the 4ULP/cancellation-floor tail doctrine.
Adapter layer: 7 forced plans (DIT T1S/LOG3/FLAT exits, DIF LOG3/FLAT,
K=67 tails both types) x {fwd_ilout, fwd_ilin, bwd_ilout, bwd_ilin,
fwd_il2il, fwd_il2il in-place-z, bwd_il2il} vs the generic executors:
ALL BIT, zero ULP tolerance consumed. Sources: build_tuned/benches/gate_tw2.c,
gate_1020.c, gate_dbl3.c, gate_adapt.c (+ build_tuned/benches/wad.txt forced wisdom).

**Measured (this container, matched-ISA AVX2, same-process medians of 5;
old path = fwd_ilin + sp2il sweep [tier CORRECTED in 6a17: generic, not jit]; build_tuned/benches/bench_fwdfold.c):**

| cell | il2il | old path | MKL-IL-lane | fold gain | new vs MKL | old vs MKL |
|---|---|---|---|---|---|---|
| (1024,4) DIT T1S | 16.2us | 16.8us | 16.0us | 1.034x | 0.987x | 0.955x |
| (100,4) DIF L3 | 1.48us | 1.58us | 0.70us | 1.067x | 0.473x | 0.444x |
| (1000,4) DIF | 22.0us | 23.0us | 11.6us | 1.047x | 0.528x | 0.504x |

Interpretation: (1) the fold gain equals the sweep it deletes — +3.4-6.7%
fwd-only at these sizes, every cell, same-process; the earlier "+29% exit
tax / ~1.14x projection" conflated tier effects, and this before/after is
the trustworthy shape. (2) At (1024,4) the DIT exit fold runs the
generic-tier group loop where the old path ran the same generic stage [CORRECTED in 6a17: h.exec_fwd was NULL — no jit ran] — near-parity
with MKL-IL (0.987x) anyway; the jit stop-gate is the remaining DIT upside.
(3) NEW honest datum: MKL's native-interleaved odd small-N kernels are
STRONG (we lose at 0.44-0.53x, N=100/1000) — a different opponent than the
MKL-split arm we beat 2.4-4.5x. Odd-composite dominance claims are vs
MKL-split and free-dispatch, as documented; the IL-native column now has
measured entries.

Deferred, in order: jit stop-gate emission; t1s_dit_bwd il_in wiring (jit
fused-bwd tier only); public vfft_config layout flag; fftnd executor IL.


## 6a17 — jit stop-gate + fused t1s bwd entry (runtime jit proven in-container)

### Stop-gate design (emit_jit.py + jit_runtime.h, VFFT_PROTO_JIT_VERSION 3→4)
`emit_jit.py` now emits TWO exported symbols per plan TU:

    vfft_proto_jit_exec_range(plan, re, im, slice_K, full_K, start_stage, stop_stage)
    vfft_proto_jit_exec(...)   /* classic 6-arg = range(start, 0x7fffffff) wrapper */

Each stage line in the range body is wrapped `if (S < stop_stage) { STAGE_XXX(S,R,isa) }`
with S a per-line literal (compile-time-folded). The start gate stays INSIDE the
STAGE macros (untouched generated plan_executors.h); the stop guard composes
externally. Semantics uniform in both directions: run stages S with
start_stage <= S < stop_stage, in the direction's natural walk order — so a bwd
entry fold at the top stage hands the remainder to `range(0, num_stages-1)`.
Cache compatibility: old .so files simply lack the range symbol → dlsym NULL →
resolver returns NULL → adapters fall back to _core. Version bumped to 4 anyway
so caches regenerate. New resolvers `vfft_proto_plan_jit_{fwd,bwd}_range()` SKIP
the baked lookup (baked fns have no range flavor) and always go through the jit
cache — legitimate per the emit doctrine (same macros → same machine code).

### Runtime jit in-container: FIRST time, and why it never worked before
`VFFT_PROTO_JIT_REPO` defaults to the WSL path `/mnt/c/Users/...` on Linux —
absent in the container, so `compile_load`'s system() always failed silently and
every prior "jit-tier" arm silently ran generic. Working recipe (all -D at the
consumer TU): `-DVFFT_PROTO_JIT_REPO='"<tree>/src/dag-fft-compiler"'`
`-DVFFT_PROTO_JIT_DIR='"/tmp/jitcache"'`
`-DVFFT_PROTO_JIT_CODELETS='"@/tmp/jitcache/codelets_linux.rsp"'` + `-ldl` +
`-I src/dag-fft-compiler/jit`, where the rsp lists -fPIC-compiled avx2 codelet
objects (90 objects: radices {2,4,5,8,10,16,20,25,32,64} × families n1_{fwd,bwd},
t1_dit_fwd[_log3], t1s_dit_fwd, t1_dif_fwd[_log3], t1s_dit_bwd, t1_dif_bwd).
Probe result: range fns resolved for (100,4)+(1024,4) both dirs; ~2MB .so per
plan; 14 cold compiles for the 7-plan gate, cached thereafter.

**CORRECTION to §6a16 fwd-fold bench labels:** the "old path = jit fwd_ilin +
sp2il sweep" arm was actually GENERIC-tier — `h.exec_fwd` was NULL in all prior
benches because (a) orchestrator jit resolution sits behind `#ifdef VFFT_USE_JIT`
(never defined in bench builds) and (b) even resolved directly, compile_load
failed on the WSL default path, and the baked lookup missed these cells. The
fold-gain numbers stand unchanged as generic-vs-generic (the gain was the
deleted sweep); only the tier label was wrong.

### Fused t1s bwd entry (mirrors VFFT_PROTO_STAGE_BWD at the top stage)
STAGE_BWD per group: `!needs_tw → radix{R}_n1_bwd`; else fused
`radix{R}_t1s_dit_bwd(base, tw_scalar[g], ...)` THEN
`VFFT_PROTO_BWD_LEG0_CONJ(base, cf0[g], slice_K)` — leg0-conj applied AFTER the
codelet, on split output, so the il_in entry fold composes with zero z-space
work: `t1s_dit_bwd_il_in(zb → split)` + leg0-conj on split leg 0.
`_vfft_il_bwd_leg0_conj` replicates the macro's exact FMA grouping
(`nr = fmadd(vi,cfi, mul(vr,cfr))`, `ni = fnmadd(vr,cfi, mul(vi,cfr))`, same
scalar tail, same cf==1 skip) — copied verbatim, NOT routed through
`_stride_cmul_scalar_inplace`, because bit parity with the jit reference demands
the grouping. t1s_dit_bwd il_in backfilled at {2,10,20,64} (existing {4,5,8,16,
25,32}), 24/24 BIT vs originals, both ISAs.

### il_execute.h consolidation (6a17 fold helpers)
Boundary folds extracted into resolve/apply pairs (`vfft_il_infold_t`/
`_outfold_t`; `_vfft_il_{resolve,apply}_{fwd_entry,fwd_exit,bwd_exit,
bwd_entry_gen,bwd_entry_jit}`), resolve-before-touch preserved. The five _core
fns are now compositions (entry → interior range → exit); `fwd_ilin`/`bwd_ilout`
DIF branches keep inline copies (accepted duplication to cap refactor risk).
New jit-tier wrappers take `vfft_proto_exec_range_fn`:
`fwd_ilout_jit` (re-signatured), `fwd_il2il_jit`, `bwd_ilin_jit2`,
`bwd_il2il_jit`. Tier purity: NULL range_fn or any resolve gap (e.g. t1sb radix
hole) falls back to the WHOLE _core path — never a mixed-tier pipeline.

### Flavor lesson #2 (gate-caught): STAGE_DIF_BWD is variant-agnostic
First gate run: `bwd_il2il(jitT)` FAILED (BAD=49/945) at exactly the DIF LOG3
cells while DIF FLAT passed and `bwd_ilin(jitT)` passed everywhere. Cause: the
jit macro always calls PLAIN `t1_dif_bwd`; the generic executor binds
`st->t1_bwd` per variant (log3 for LOG3 plans). My jit exit fold mirrored the
GENERIC flavor → ULP divergence vs the full-jit reference. Fix:
`_vfft_il_resolve_bwd_exit_jit` resolves plain `difb_ilo` unconditionally
(core keeps variant-bound). After fix: ADAPTER GATE ALL PASS — 7 plans ×
{core-vs-generic, jit-vs-full-jit} arms, everything BIT including in-place-z
jit il2il, and the first-ever-firing dif jit-resume arms.

### Bench (bench_bwdfold.c, same-process medians, spike wisdom, MKL OOP lane)

| cell | b.sweep | b.core | b.jit | b.mkl | f.core | f.jit | f.mkl |
|---|---|---|---|---|---|---|---|
| (1024,4) DIT [64,16] | 20.39 | 15.10 | **11.56** | 7.27 | 11.13 | 11.14 | 7.19 |
| (100,4) DIF [10,10] | 1.297 | 0.996 | 1.019 | 0.825 | 1.078 | 0.972 | 0.823 |
| (1000,4) DIF 3-stage | 21.11 | 15.34 | 17.15 | 11.81 | 16.45 | 16.82 | 11.91 |

Within-session structure (the reliable signal):
- **DIT bwd: fused t1s entry is the headline win** — 1.31× over core, 1.76×
  over the sweep path at (1024,4). At 2 stages the interior is empty, so the
  entire gain is entry-fold quality (fused t1s vs n1_bwd + cf_all conj traffic).
- **DIF bwd jit tier loses on log3 plans** (17.15 vs 15.34 at 3-stage = the
  plain-flavor penalty made manifest in time, not just bits). Selection rule:
  bwd z→z DIT → jit tier; DIF → core tier. fwd z→z: core (jit adds nothing
  measurable; the codelets do the work, loop overhead was never the cost).
- Cross-session vs-MKL ratios are NOT comparable (container phase shifts the
  memory-bound/compute-bound balance; §6a16 measured 0.99× fwd at (1024,4),
  today 0.65× same code) — per-session same-process ratios only, as ever.

### Deferred (updated)
#1 stop-gate + #2 t1s bwd entry: DONE. Remaining: public vfft_config layout
flag; fftnd executor IL; small-N odd IL kernels; NEW: plan-time tier chooser
(encode DIT→jit / DIF→core bwd rule, or measure at plan time); optional: emit
variant-bound DIF bwd stages in emit_jit (would recover log3 speed at jit tier
and obsolete the selection rule — touches macro mapping only, not macros).


## 6a18 — bwd_oop_jit (OOP split→split symmetry restored)

`oop_execute.h` gains `vfft_proto_execute_bwd_oop_jit`: the pointer-swap
identity IDFT(re,im) = swap(DFT(im,re)) composed with the JIT forward.
`bwd_oop` now delegates to it with NULL. THE resolution contract (loud in the
header comment): the swap lives in the data pointers, not the direction — the
plan executes its FORWARD dataflow, so callers pass the FORWARD-resolved
executor (`vfft_proto_plan_jit_fwd`), never `_bwd`; the real backward executor
applied to swapped data is a different transform. Deliberately NO range-fn
plumbing: stage 0 is the fused OOP boundary, the remainder is exactly classic
start_stage=1 — the 6a17 stop gate has no consumer here.

Gate T11 (gate_adapt.c): 4 DIT cells × {fwd,bwd}_oop jit-inner vs generic-inner
ALL BIT, src-preservation OK, 3 DIF cells reject -1 both directions (stage-0
twiddle physics, unchanged).


## 6a19 — public API convergence: interleaved z contract + fft3d wisdom + dims=3

Both features land WITHOUT new interfaces — the constraint was extending the
domain of existing entry points, not adding functions.

### A. INTERLEAVED z on vfft_execute (the "public layout flag", solved as a contract)
New buffer-table row in vfft.h: 1D tight in-place C2C with **sim==dim==NULL**
means sre/dre are interleaved complex (2*N*K doubles; dre may equal sre) —
the same NULL-halves pattern R2C/trig already use. Internally (vfft.c):
the 1D in-place body was extracted verbatim to `_exec_c2c_inplace` and a new
`_exec_c2c_interleaved` routes: fast path = the 6a16/6a17 folded z→z adapters
under the tier rule (fwd → il2il core; bwd → DIT `bwd_il2il_jit` fused-t1s /
DIF core), taken when order=DEFAULT and the pool is single-threaded; NATURAL /
MT / prime-override / <2-stage / padded all fall back to convert →
`_exec_c2c_inplace` → convert (always correct, never silent; padded excluded
by contract — z is tight-only). Lazy NK-complex split scratch + one-shot
`vfft_proto_plan_jit_bwd_range` resolve, guarded `#ifdef VFFT_USE_JIT`
(non-jit builds: il_rfb NULL → the wrapper's internal core fallback). This
makes the public convolution pattern (fwd → pointwise → bwd on z buffers) the
FIRST production consumer of the il2il surface.

Gate (build_tuned/benches/gate_vfft_il.c, public API, ALL PASS): DIT 64×8 + DIF 100×8
fast paths BIT vs split; prime 101 (Bluestein override), NATURAL, and
nthreads=2 all BIT through the fallback; conv pattern BIT end-to-end;
roundtrip/N OK everywhere.

### B. fft3d wisdom + public dims==3
`fft3d_wisdom.h` (new, mirrors fft2d_c2c_wisdom.h): one entry per (N1,N2,N3)
storing B + a_block + all three inner chains (nf/factors/variants/dif).
`vfft_fft3d_plan_create_wisdom`: HIT → `plan_create_ex ×3` +
`stride_plan_3d_from` — the path fft3d.h's own header comment requested;
MISS → the greedy per-axis exhaustive body with the inners kept visible,
banked by direct plan extraction (radix / use_log3 / t1s_fwd per stage; BUF
not round-tripped — banked FLAT, correctness-identical; override/prime axes
never banked, creates still succeed greedy). Bundle membership: path +
table in vfft_wisdom_s, `_bundle_paths`/`_bundle_load` rows
(fft3d_c2c_wisdom.txt).

Public dims==3 via the SAME create/execute/destroy: vfft.h `n[2]`→`n[3]`,
dims 1/2/3; constraints (rejected with NULL): C2C only, howmany==1 (the wrap
is a K=1 override plan), order DEFAULT/SCRAMBLED (3D natural = the
nat_col_list follow-up). Execute reuses the 2D branch with an N3-aware plane
size — the 3D wrap IS a stride_plan_t.

Gate (build_tuned/benches/gate_vfft_3d.c, ALL PASS): 16×20×8 — greedy create 3.81 s,
banked entry, fresh-bundle wisdom-hit create **34 μs (≈110,000× faster)**,
delta-spectrum |X|²==1 and roundtrip/N identical on both paths; K>1 /
NATURAL / non-C2C rejects hold. Feature-A gate re-run post-patch: ALL PASS.

Known shared gap (pre-existing, now inherited): public vfft_wisdom_free/save
do not cover the fft2d tables and now likewise not fft3d — the bundle
auto-save at create is the persistence path. Parity follow-up queued.

Feature-matrix deltas vs the 6a18 audit: conv consumers ✓ il2il (via the
public z contract); fft3d ✓ wisdom; dims=3 ✓ public.


## 6a20 — r2c packing-tax attribution bench (pointer)

Full findings + revised design in `docs/roadmap/r2c_c2r_il_design.md`.
Headlines: fused-first-stage already live (DIT+aligned only — the tax
survives at DIF-inner/high-K + B-tails); model-(b) last-stage fusion is
built, correct (activated live, BAD=0), has NO setter anywhere, and is 17%
SLOWER as-is (scalar scaffold); rfft-native non-pow2 low-K already BEATS MKL
(1.22–1.62×); inner-cell wisdom quality moves r2c ~30%. Bench:
`build_tuned/benches/bench_r2c_tax.c` (same-TU prof + stub-delta + live activation).
rfft.h:523 UB warning logged as debt.


## 6a21 — variant-bound DIF bwd emission (jit ver5)

emit_jit.py now maps DIF bwd stages per variant: LOG3 stages emit
`VFFT_PROTO_STAGE_DIF_BWD_LOG3` — a verbatim twin of the generated
STAGE_DIF_BWD (extracted at patch time, only the codelet symbol swapped to
`t1_dif_log3_bwd`), defined INSIDE the emitted TU so plan_executors.h stays
untouched. Externs gain the log3 names; VFFT_PROTO_JIT_VERSION 4→5 (cache
regeneration). DIT bwd stays plain t1s by design — generic DIT bwd has no
variant counterpart to match.

il_execute.h simplification: `_vfft_il_resolve_bwd_exit_jit` (the
plain-forced fold from the 6a17 flavor lesson) is DELETED; bwd_il2il_jit uses
the variant-bound resolver like core. ADAPTER GATE: ALL PASS with the special
case gone — jit and generic tiers now agree on DIF bwd flavors bit-for-bit.

Bench (same-process): 2-stage DIF (100,4) flipped to a slight jit win
(1.508 vs 1.554 µs). 3-stage DIF (1000,4) retains an ~8.7% jit deficit
(23.41 vs 21.55) — flavor binding was NECESSARY but NOT SUFFICIENT there;
residual attributed next (candidate: tw_buf block-broadcast hoisting
differences between the STAGE macro and the generic dif bwd range loop).
Consequence: the bwd tier-selection rule (DIT→jit, DIF→core) STAYS for
multi-stage DIF until the residual closes; vfft.c's interleaved routing is
unchanged. DIT bwd win intact (15.1 vs 20.9 this session).


## 6a22 — public wisdom free/save parity (#7)

`vfft_wisdom_free` / `vfft_wisdom_save` now cover the FULL `_bundle_load` set:
+fft2d_c2c, +fft2d_r2c, +fft2d_c2r, +fft3d_c2c (frees + saves), +bluestein
(save; table is fixed-size, no free — like oop). `c2r_path` stays excluded on
both sides: it loads into a c2r_dispatch file-static and persists via its own
decision-time writer, not through w. Gate (`build_tuned/benches/gate_w7.c`, ASAN build):
3× load+free leak-clean (LSAN silent — pre-patch four tables leaked per
free); save writes all 8 files; save→load→save byte-identical per file.
Closes the persistence trap where "vfft_wisdom_save succeeded" silently
dropped every 2D/3D entry (auto-save at create was the only real writer).


## 6a23 — r2c fused-first-stage coverage (#6): B-tail closed, DIF spec'd

**Gap B (B-tail) CLOSED.** The `(B & 3u)==0` guards on both
`_r2c_fused_first_stage` call sites were STALE and are removed: the OOP n1
family is rem-aware by construction (generator anyk-tail — masked group
loads/stores, codelet_oop.ml emit_codelet preamble +
docs/performance/arbitrary_k_tail_handling.md), and the engine gates already
ran it at me=65/67. Gate `build_tuned/benches/gate_r2c_tail.c` (same-TU, internal
stride_r2c_plan with forced block_K, naive O(N²) real-DFT reference):
DIT-fused B∈{64,65,67} and DIF-explicit B∈{64,67} all ≤3e-13 maxrel — ALL
PASS. Misaligned-B DIT cells no longer pay the explicit pack. Deliberately
UNTOUCHED: r2c.h:973 (`n1_scaled_bwd` c2r fused-unpack gate — separate
family, tail status unverified) and r2c.h:1527 (`term_fwd` step-2 — dormant).

**Gap A (DIF-inner) findings — fused DIF needs a codelet that does not
exist.** The generator's OOP twiddled modes are ALL PRE-twiddle: `t1_oop`
(--twiddled --oop, UG/UG) pre-multiplies legs 1..R-1 by tw rows (j-1)
(standard non-conj cmul) then DFTs — verified by reading the emitted body;
`--dif` is cosmetic for this family (identical math); `--twiddled-pos` gives
`t1p` ("per-position, second-stage"), also ≠ post (probe distances ~7 vs
both conventions). DIF stage-0 requires POST (out = tw ⊙ DFT(in)); DFT and
diagonal don't commute, so no pre-flavor can serve. The 8 generated pre-tw
codelets (misleadingly named radix{5,10,20,25}_t1_dif_oop*) were gated
(total mismatch), then DELETED from the tree. The n1-oop + separate
twiddle-multiply bypass was analyzed: equal memory traffic to explicit-pack,
no win. Gap A therefore ships as a generator spec (r2c_c2r_il_design.md §6);
the dormant `stride_t1_oop_fn t1_oop_fwd` stage slot (stride_executor.h,
"For R2C fused pack") is the wiring target once the family exists.

**Dual-ABI n1 landmine (documented hazard).** `radix{R}_n1_fwd_{isa}` names
TWO families with different ABIs: the engine's `stride_n1_fn` 7-arg
(in/out + is/os/vl — archive editions, main link) and codelets/inplace 6-arg
(rio/tw/ios/me — jit .so universe via the PIC rsp). Same symbols, segregated
by link set. Any build-system change that mixes the sets will link the wrong
arity silently.


## 6a24 — R2C/C2R interleaved-z public contract + native z terminators (#5)

**Contract (NULL-halves, completing the §6a19 C2C convention).** R2C with
`dim==NULL`: `dre` receives the INTERLEAVED (CCE) spectrum — (N/2+1)*howmany
complex pairs at `dre[2*(f*howmany+t)]`. C2R with `sim==NULL`: `sre` is the
interleaved spectrum input, same layout. Documented in include/vfft.h rows.
Gate `build_tuned/benches/gate_vfft_rz.c`: cells (512,256) stride / (2000,4) + (200,4)
rfft / (128,67) odd — fwd z-vs-split BIT, bwd z-vs-split BIT, roundtrip
≤2.2e-14: ALL PASS (12/12).

**Mechanism.** r2c.h gained loop-invariant store/load boundary helpers
(_r2c_st1/4/8, _r2c_ld*/ldr*v; avx2 unpack+permute2f128, avx512
permutex2var) and every postprocess store / preprocess load was converted
(8+4 store sites, 8+6 load sites; raw-access asserts = 0). stride_r2c_data_t
carries zo/zi (NULL = split). FWD stride path rides `_r2c_execute_fwd_oop`
with real_in read-only — zero staging copy. BWD stride path passes the real
output as the in-place `re` with `im=NULL` (audited: worker_bwd touches im
only via _r2c_preprocess, which reads zi in z mode). RFFT / NATURAL paths
convert-around via a lazy per-plan ztmp (native rfft z terminator = tracked
follow-up). Dormant fused-post branches (ls_fwd/term_fwd) are z-guarded.
2D R2C/C2R z contract is deliberately out of scope (follow-up).

**Perf — an honest-measurement story.** First public run (sequential arms)
read z as 1.5% SLOWER; the same-TU 3-arm isolation showed phase movement in
pack/inner — phases the store variant cannot touch — flagging container
drift. Drift-proof reruns (arms round-robined within each trial):

| arm (512,256 fwd) | µs | delta |
|---|---|---|
| internal: inplace+copy split (old default path) | 355.96 | — |
| internal: oop split (no staging copy) | 354.08 | −0.5% |
| internal: oop z (native interleaved stores) | 341.92 | −3.9% vs default; **−3.4% pure store-variant** |
| public: split | 364.58 | — |
| public: z | 344.55 | **−5.5%** |
| MKL CCE (contract-equal) | 248.66 | z/MKL 0.722×, split/MKL 0.682× |

Post-phase isolation: 108.3 -> 93.5µs (−13.7% of post) — exceeding the
design doc's "~10% of post" estimate. The staging-copy deletion is nearly
free in context (−0.5%; the 25µs memcpy microbench bound is not
representative). The MKL line is the first same-layout comparison
(CCE↔CCE) and sits at MKL's strongest r2c geometry; rfft-native cells beat
MKL outright per §6a20. Bycatch: the OOP worker's stale pre-6a23 odd-B
comment was corrected. Benches: build_tuned/benches/bench_rz.c (public, drift-proof),
build_tuned/benches/bench_rz_iso.c (same-TU 3-arm attribution).


## 6a25 — Model-B fused last-stage terminator (#8): CLOSED, measured negative

**Verdict: model-B does not pay. The ls_fwd setter is NOT wired; the
machinery stays dormant in tree.**

**Attribution first (correcting §6a20).** The §6a20 diagnosis blamed the
"scalar group-pair scaffold" for the −17%. Measured today with a stub-ls
arm: the scaffold + caller specials floor is **11.8µs** at (512,256) —
essentially free. The actual deficit was the codelet: the avx2 edition costs
~260µs to do what model-A's last stage + postprocess do in ~179µs (4-wide
fold math vs A's 8-wide postprocess, plus strided leg access).

**The avx512 edition (emitted this session:
`gen_radix.exe 256 --r2c-term-ls --r2c-term-ls-r 8 --emit-c --isa avx512`,
now at codelets/rfft/avx512/, correctness 8.9e-15 vs model-A) closes the
width gap but not the structural one:**

| cell | inner shape | A (µs) | B-avx512 (µs) | delta |
|---|---|---|---|---|
| (512,256) ×5 process instances | r2 r4 r4 r8 | 344–359 | 350–363 | −1.8, +3.7, +2.0, +0.3, +1.7% → median **+1.7%**, lone win = noise-band outlier |
| (256,256) | r4 r4 r8 | 132.9 | 141.6 | **+6.6%** |
| (1024,256) | r4 r4 r4 r8 | 815.8 | 834.4 | **+2.3%** |
| (4096,32) | r8 r4 r8 r8 | 491.9 | 532.1 | **+8.2%** |

Phase evidence at (512,256): B's inner is genuinely −49µs (the deleted
last-stage round-trip is real), but B's post (codelet + 12µs scaffold) runs
138µs vs A's 116µs postprocess — and A's postprocess streams scratch
row-contiguously with hoisted mirror twiddles while the ls codelet gathers
strided legs per group pair under 32-zmm pressure (the generator's own
(2,r) spill-seam note at gen_main.ml:299). The avx2 edition is +24.5%
total; avx512 lands at parity on the best cell and regresses elsewhere.

**Cross-process noise honesty:** same-process interleaved medians still
disagree by up to 5.5pp across process instances on this container; single
runs below ~±3% are not decisions. The (512,256) table row shows the full
band, per doctrine.

**What stays / structural liabilities recorded:** scaffold
(_r2c_laststage_fused), dispatch branch, typedef — all dormant (no setter).
Both codelet editions in tree. Decision harness:
build_tuned/benches/bench_ab_modelb.c (argv cells; link the avx512 codelet TU compiled
with -mavx512f -mavx512dq). slice_until has no jit counterpart (jit-bound
inners would additionally forfeit jit on stages 1..nf-2 — a further
handicap never even reached in these numbers). Revisit conditions: a
z-native ls variant + jit end_stage + a demonstrated geometry where fold
traffic dominates could shift the balance; no active plan.


## 6a26 — rfft native z terminator (fwd): SHIPPED, BIT-exact via chunked hcnr

**Contract completion:** `vfft_r2c_execute_fwd_z` on the RFFT/SPLIT path now
runs a native interleaved stage-0 terminator instead of the §6a24
convert-around (which cost +19–23% on rfft cells and surrendered the MKL
lead at (2000,4): 1.145× → 0.960×).

**Design (rfft.h `_rfft_stage0_z` + zo threading):** both natural executors
(plain + lane-range) and `rfft_natural_mt` take a `double *zo`; stage 0
branches to the z helper, everything upstream shared. k0 specials interleave
from `nat_k0`; the terminator codelet lands in a plan-owned scratch
(`p->zscr`, 4 planes sized `r * zch * K`) and rows are interleaved to z
while L1-hot; mid stores through `rfft_mid_column`'s new zo mode. **Chunked
RANGED terminator:** when `p->hcnr` is bound (VFFT_RFFT_RANGED — defined
in-source at vfft.c:37), z-mode calls the SAME ranged codelet as split's
single kcount sweep, in `zch`-column chunks (plan-time L1 budget
`768/(r*K)`, capped at kmax; mirror-side bases at `+(cw-1)*K` matching the
codelet's internal −cs_out walk). Splitting the codelet's column loop across
calls preserves per-column arithmetic → **gate_vfft_rz 12/12 BIT PASS, fwd
and bwd,** all four cells.

**Debugging record (doctrine lesson):** the initial per-k `hcn` z-mode
diverged from split by 1–2 ULP in hcn-family rows only. The isolation
harness proved the codelet stride-agnostic and bit-stable at every k;
entry tracers then showed the split arm skipping the per-k loop entirely —
split runs `p->hcnr`. The "refutation" that ruled hcnr out was doubly
invalid: `-U` cannot undo an in-source `#define`, and the define grep
covered flags and two headers, not the tree. **Lesson: compile-time-feature
questions are answered by grepping the tree for the define, never by flag
archaeology.** The D2 one-sided slot partition (low slots via Rp at row f,
upper conjugated via Rm at row r*m−f) is documented at the interleave.

**Measured tax curve, z vs split (public API, generic cascade both arms —
see jit note below; drift-proof in-process, container weather poor today):**

| cell | z vs split | z/MKL-CCE | split/MKL-CCE |
|---|---|---|---|
| (200,4) ×3 | +34.8, +35.2, +40.8% | 1.18–1.25× | 1.59–1.76× |
| (2000,4) | +21.5% | 1.15× | 1.40× |
| (1000,8) | +16.4% | 1.26× | 1.46× |
| (20000,4) | +2.5…+6.3% | 0.63× | 0.66× |
| (50000,4) | −1.0% | 0.93× | 0.92× |
| (100000,4) | −12.8…−31.7% | 0.67× | 0.53–0.59× |

Tiny cells are L1-resident: there is no memory round-trip to fuse away, so
the interleave's ALU cost is an inherent floor (the convert-around paid the
same +19–23% there). The design's payoff is structural at DRAM scale: one
interleaved output stream has half the distinct rows of two split planes
(TLB/prefetch density) — at (100000,4) native z beats split outright.
z/MKL stays ≥1 on the small-N rfft lead cells (vs 0.96× under
convert-around at (2000,4)).

**Route:** STRIDE → native zo postprocess (§6a24). RFFT/SPLIT + jit_natural
bound + single-thread → jit cascade into ztmp + interleave (keeps jit speed
AND bit parity; the jit emits its own stage-0 arithmetic). RFFT/SPLIT
otherwise → native (mt-capable; K<16 falls back to the folded ST executor
as before). PACKED (publicly unreachable) → packed→CCE unpack via lazy
ztmp, which also fixes the pre-§6a26 latent bug where the convert-around
assumed split planes for packed plans. C2R z-in stays deinterleave-around
(v1 scope; native z-in initiator is a documented follow-up).

**JIT emitter repair + discovery:** the zo arity change broke
`emit_rfft_jit.py`'s emitted fallback/mid calls (runtime jit compiles
failed silently → generic). Fixed (natural fallback + both mid_column
emissions now pass the zo NULL; VFFT_PROTO_JIT_VERSION 5→6 orphans the
poisoned cache). Discovery while verifying: the freshly compiled ver6 .so
dlopen-fails with `undefined symbol: radix5_hc2hc_dit_fwd_avx2` — the PIC
codelet rsp has NO rfft codelet family AT ALL (zero hc2hc/r2cf entries —
the rsp carries only the c2c/DAG set), so **the rfft natural/packed jit
has never bound for any multi-stage shape in this container** (ver5 failed
identically).
All rfft-cell history in this doc is generic-cascade and remains valid;
jit-coverage gap recorded as new debt.

**Bycatch:** the rfft.h:523 -Waggressive-loop-optimizations warning is gone
(mid scalar tail restructured by the zo mode); vfft.c -Wall is clean of
rfft.h warnings.


## 6a27 — rfft PIC codelet set + natural-jit verdict: rsp fixed, jit CLOSED negative

**PIC set + rsp (shipped, tools/build_rfft_pic_rsp.sh):** 115 rfft/c2r avx2
codelets compiled -fPIC and appended to codelets_linux.rsp (94→209 lines).
This made the rfft natural jit bind for the first time in this container
(the resolver had been dlopen-failing silently on missing symbols and
falling back to generic since forever).

**Then the measurements, same-process four-arm (split-jit / split-generic /
z-jit_z / z-native):**

| cell | path | split generic vs jit | z native vs z jit_z |
|---|---|---|---|
| (2000,4) | RFFT | generic **−8.9%** (12.94 vs 14.21) | native 15.69 vs jit_z 24.10 |
| (2000,16) | RFFT | generic −1.0% | native 55.5 vs jit_z 66.0 |
| (2000,64) | STRIDE | rfft jit n/a (fields nil) | n/a |
| (128,67) | STRIDE | n/a | n/a |

The rfft path serves only small K (larger K routes to STRIDE), and across
that entire domain the natural jit is a net negative: the emitted per-k
terminator's call overhead at small vl exceeds the cascade gain over the
generic executor's single ranged hcnr sweep. **Verdict: jit_natural and
jit_natural_z are NOT bound (explicit NULLs + rationale at the bind site);
the fwd_z route is native always.** Resolvers, emitter modes (natural,
natural-z), and the PIC rsp remain in place for revisit.

**The natural-z jit (built and closed this session):** emit_rfft_jit.py
gained --mode natural-z — same cascade and per-k log3 terminator as the
natural mode with stores redirected through p->zscr + _rfft_zrow interleave.
It is **bit-exact against the natural jit** (gate 12/12 BIT with the
jit↔jit_z pairing — store-redirect-only emission works exactly as the
§6a26 isolation predicted) but slow: +63% over jit-split at (2000,4).
Variant surgery on the emitted .c (v1 = codelet→scratch only, v2 =
interleave only) attributed it: codelet-to-scratch is 0.7µs FASTER than
direct scatter; **the interleave loop alone costs ~11µs in the jit TU vs
2.2µs for the identical static-inline helper in the main build** (~5×;
-O3/-march=haswell TU, cause not root-caused — mode closed, observation
recorded). An intermediate batched-k emission (RAW-stall hypothesis) made
it worse and is kept in the emitter as the shipped shape of the closed mode.

**Doctrine reinforcement (the weather ghost):** the initial "jit −19%/−29%
wins" at 20K/100K were cross-process comparisons on a day the container
swung ±25%; the same-process four-arm showed generic ahead. Cross-run
deltas are not evidence at ANY magnitude when the noise regime is unknown —
the ±3% band is a floor, not a ceiling. Same-process interleaved arms only.

**Shipped configuration, public API, contract-equal (gate ALL PASS,
generic↔native pairing):**

| cell | split | z | z tax | z/MKL-CCE | split/MKL-CCE |
|---|---|---|---|---|---|
| (2000,4) | 12.69 | 14.97 | +18.0% | **1.036×** | 1.222× |
| (200,4) | 0.73 | 0.97 | +33.2% | **1.297×** | 1.727× |
| (1000,8) | 10.92 | 12.92 | +18.3% | **1.155×** | 1.367× |

**The §6a26 motivation is closed: z/MKL ≥ 1.0 at contract equality on
every rfft lead cell** (was 0.960× at (2000,4) under convert-around), with
the §6a26 large-N structural win unchanged (native z −15..−30% at
(100000,4) vs any split arm).

**Bycatch:** jit cflags += -Wno-nonnull (emitted fallback's provably-benign
NULL split args); decomposition-by-variant-surgery on emitted .c files
recorded as a reusable attribution technique.


### 6a27 addendum — post-ship regression audit (did the jit work regress anything?)

Question asked directly after ship; answered with measurement, not inspection.

| surface | verdict |
|---|---|
| shipped fwd hot paths (split, z) | no change vs pre-campaign — natural/natural-z jits unbound; behavior identical to the never-bound history; plan create slightly faster (doomed resolve attempt removed) |
| c2c jit after VERSION 5→6 | steady-state unregressed ((512,256) stride matches §6a24 history); one-time recompile per shape from the orphaned ver5 cache |
| §6a26 zo plumbing on split | predictable-branch guards; split numbers match historical bands all session (no rigorous A/B; sub-noise by evidence) |
| **c2r jit (packed low-K path)** | **newly ACTIVE** — pre-existing bind that dlopen-failed until the rsp rebuild. Same-process A/B: (2000,4) −0.4%, (200,4) +0.6%, (2000,16) +0.3%, (1000,8) +0.4..+4.0% across 4 runs — **parity within the ±3% band, no wins, no proven loss.** Left bound: pre-existing designed behavior, bit-exact vs generic, and live jit compiles exercise the emitter (how the §6a26 arity breakage was caught). Revisit if a c2r cell regression is ever suspected. |
| r2c packed fwd jit (jit_packed) | **unreachable** — every r2c create site in the tree passes VFFT_R2C_SPLIT; the bind is dead code, no live change |


## 6a28 — c2r native z-in initiator: SHIPPED, BIT-exact, 1D z contract complete

**The bwd mirror of §6a26.** vfft_c2r_disp_execute_z on the NATURAL layout now
feeds the interleaved (CCE) input natively into the stage-0 initiator instead
of the §6a24 deinterleave-around (measured +24..+35% and z/MKL 0.965× at
(2000,4) before this change).

**Design (c2r.h `_c2r_stage0_zin` + zi threading, one-patch land, gate
first-attempt ALL PASS):** the chunk's z rows are deinterleaved through the
base rfft plan's zscr planes (P/M families, slot stride cw*K, column stride
K — the §6a26 layout mirrored, reusing zscr/zch since c2r_plan_t wraps
rfft_plan_t) filling EXACTLY the cells the fwd terminator wrote (the same
one-sided D2 predicate — the initiator reads only those); the SAME nat_init
codelet then runs per column on scratch pointers (stride-agnostic,
§6a26-proven). DC gathers become fused `_rfft_zldrow` deinterleaves (new
helpers, exact shuffle inverses of `_rfft_zst4`); mid reads via a zi mode on
c2r_mid_inv_column_natural. Both executors + c2r_natural_mt thread `zi`
(NULL = split, split path untouched inside its branch). **gate_vfft_rz
12/12 BIT, fwd and bwd** — native z-in and split-in run the same generic
machinery, so parity is structural.

**Also fixed:** the PACKED-input layout's z entry now packs CCE -> packed
halfcomplex properly (the pre-§6a28 convert-around fed split planes to the
packed-input entry — the bwd twin of the fwd latent bug fixed in §6a26).

**Measured (public API, drift-proof, medians of repeated runs):**

| cell | z-in tax (was, convert) | z/MKL-CCE-bwd (was) | split/MKL |
|---|---|---|---|
| (200,4) | ~+43% (**was +35%** — see floor note) | 1.13–1.18× (was 1.26×) | 1.62–1.71× |
| (2000,4) | +17.9..+20.9% (was +24.1%) | **0.99–1.01× parity** (was 0.965×) | 1.19–1.20× |
| (1000,8) | +16.7% (was +31.7%) | **1.227×** (was 1.100×) | 1.432× |
| (20000,4) | +11.7% | 0.718× | 0.802× |
| (100000,4) | **−10.9%** | 0.756× | 0.673× |

**Micro-cell floor (honest note, mirrors fwd):** at (200,4) the native
row-structured gather is ~0.1µs WORSE than the branch-free streaming
convert it replaced — the same trade the fwd side accepted at the same
cell. A size gate (convert below ~H*K≈2K) is a recorded option, not taken:
route simplicity + symmetry win at 0.1µs absolute stakes. Everywhere else
native wins, structurally at DRAM scale (−10.9% at (100000,4), mirroring
the fwd interleaved-layout advantage).

**1D z contract status: COMPLETE both directions.** fwd z-out native
(§6a26), bwd z-in native (§6a28), STRIDE path native both ways (§6a24),
PACKED both ways correct via proper pack/unpack. Remaining z-family debt:
the 2D r2c/c2r z contract.


## 6a29 — 2D r2c/c2r z contract: SHIPPED (v1 convert-around); 2D-vs-MKL gap surfaced

**Correctness first: the 2D z sentinel SEGFAULTED before this change** —
vfft.c passed dim/sim straight into stride_execute_2d_r2c/c2r, dereferencing
the NULL plane inside the tiled row pass. §6a29 defines and implements the
contract: dims==2 + dim==NULL (fwd) / sim==NULL (bwd) means the buffer is
interleaved CCE over the 2D half-spectrum, z[2*(i*H2+f)] for i=0..N1-1,
f=0..N2/2 — MKL's 2D CCE shape exactly (howmany==1 by 2D design; flat
M=N1*H2 interleave, no lane structure).

**v1 = convert-around** (lazy h->z2tmp planes + vectorized flat
interleave/deinterleave via _rfft_zst4/_rfft_zld4d). Gate extended with 2D
cells (64x128, 96x200): **18/18 ALL PASS** — fwd and bwd z-vs-split BIT
plus roundtrips, alongside the four 1D cells.

**Measured tax and — the real finding — the 2D baseline itself:**

| cell | dir | z tax | z/MKL | split/MKL |
|---|---|---|---|---|
| (256x256) | fwd | +16.2% | 0.529x | **0.615x** |
| (256x256) | bwd | +18.0% | 0.651x | 0.768x |
| (512x512) | fwd | +12.7% | 0.533x | **0.601x** |
| (512x512) | bwd | +16.6% | 0.658x | 0.768x |

**[CORRECTED — see the §6a29 addendum below.]** The original text here
claimed these were "the first MKL comparison recorded for the 2D path" and
framed the gap as "2D trails MKL 1.6-1.7x". Both framings were wrong:
v1_0_results.md §2 records 2D C2C beating MKL 1.26-1.41x (i9-14900KF,
PATIENT, cooled), and the 0.60x here compares against MKL's REAL-CCE
baseline, which is a much stronger MKL than the complex-split baseline the
v1.0 win was earned against. The corrected finding is in the addendum.

**Native 2D z (phase-2 c2c z-store) — recorded, deliberately subordinated:**
unlike 1D, the final 2D pass is an in-place c2c over the split planes (no
real-transform terminator to redirect), so native z means c2c-engine store
surgery. Recovering the 13-18% convert tax on a path sitting at 0.60x of
MKL is priority inversion; the item is parked behind the 2D perf campaign.

**1D+2D z contract status: correct and gated everywhere.** 1D native both
directions (§6a26/§6a28); 2D convert-around, bit-exact, crash fixed.


### 6a29 addendum — reconciliation vs v1_0_results.md (claim challenged and corrected)

The §6a29 headline was challenged against the prior record (v1_0_results.md
§2: 2D C2C beats MKL complex-split 1.26x at 256², 1.29x at 512², i9-14900KF,
PATIENT + fft2d_c2c_wisdom, cooled best-of-5). Reconciliation control run on
THIS container, same-process interleaved, (256×256):

| arm | µs | ratio |
|---|---|---|
| vfft c2c-2D (in-place scrambled, PATIENT, copy-corrected) | 452.0 | — |
| MKL complex SPLIT (the v1.0 config) | 467.0 | **dag/MKL 1.033×** |
| MKL complex interleaved | 419.4 | 0.928× |
| MKL real CCE (the §6a29 r2c baseline) | 208.4 | MKL real/complex-split = **2.24×** |

**Verdicts:**
1. **No regression.** The c2c-2D path still ties/beats MKL on the v1.0
   configuration today (1.03×). The 1.26→1.03 residue is cross-host
   (14900KF vs this 1-vCPU container), no cooling, and container weather —
   cross-host ratio deltas are not regression evidence per doctrine.
2. **"First 2D-vs-MKL numbers": FALSE** — v1.0 §2 predates this; withdrawn.
3. **The corrected finding:** MKL extracts **2.24×** from real-vs-complex 2D;
   VectorFFT's r2c-2D extracts only **1.31×** over its own c2c-2D
   (452 → 345.7µs). The r2c-2D composition under-harvests the real-transform
   advantage — that is the debt, precisely stated.
4. **Phase attribution (VFFT_2D_PROFILE counters, pre-existing in tree;
   /tmp/f2dprof harness), (256×256) fwd, 98.8% accounted:** wrapper memcpys
   14.7% (OOP-convenience layer), p1 transpose-in 8.4%, p1 inner-r2c 37.3%,
   p1 transpose-out+pad 12.4%, p2 col-c2c 17.0%, p3 perm-pack 9.0%
   (the natural-ordering pass — v1.0's c2c comparison was scrambled and
   legitimately skipped this class of cost). Copies lower bound 23.7%.
   Stripping wrapper+pack puts the core at **0.79× vs MKL-real** — the
   campaign targets, in order: wrapper elimination (in-place-capable public
   route + the z path interleaving directly from re_pad), perm-pack fusion
   into phase 3 reads, then the inner-r2c 37% share.
5. Methodology deltas vs v1.0 recorded for future comparisons: this
   container has no bench_1d_vs_mkl.c (tool not in archive), no cooling,
   MEASURE-vs-PATIENT wisdom states differ, and the §6a29 r2c rows used
   MKL-native-CCE deliberately (the contract-equal framing of §6a24+) — a
   harsher and different question than v1.0's identical-split-layout
   framing. Both framings are valid; they must never be conflated again.


## 6a30 — 2D wrapper elimination + fused z: copy-free OOP path, z tax ELIMINATED

**First slice of the 2D campaign, straight off the §6a29-addendum phase
table.** The OOP entries (stride_execute_2d_r2c/c2r) paid a full-plane
memcpy in + half-plane out purely because they reused the in-place ABI where
one pointer is both input and output — while the core's phases never needed
to mutate the input (phase 1 reads it, phases 2-3 live in the pad scratch,
phase 3 writes the output). §6a30 adds copy-free OOP-native executors
(_fft2d_r2c_execute_fwd_oop/_bwd_oop) and rewires the wrappers; the in-place
override stays untouched for its own contract.

**The fused z (the "parked" §6a29 native-2D-z, shipped for free):** the _z
variants fold the interleave/deinterleave into the existing phase-3 pack /
phase-1 unpack perm loops (_f2d_zil4/_f2d_zde4). vfft.c's 2D z branches call
them directly; the §6a29 z2tmp convert machinery is REMOVED (field, allocs,
frees, convert loops — one day old, cleanly retired). Gate 18/18 ALL PASS
(bit-identical: same phases, same pad bytes).

| cell/dir | split before → after | z tax before → after | split/MKL-real | z/MKL-real |
|---|---|---|---|---|
| 256² fwd | 332.4 → **283.1 (−14.8%)** | +16.2% → **−0.7%** | 0.615 → **0.741×** | 0.529 → **0.747×** |
| 256² bwd | 312.0 → 282.7 (−9.4%) | +18.0% → +2.3% | 0.768 → 0.859× | 0.651 → 0.840× |
| 512² fwd | 1652.3 → **1382.2 (−16.3%)** | +12.7% → **−0.5%** | 0.601 → **0.715×** | 0.533 → **0.719×** |
| 512² bwd | 1482.3 → 1322.5 (−10.8%) | +16.6% → −0.1% | 0.768 → 0.870× | 0.658 → 0.871× |

**The 2D z tax is eliminated — z runs at split parity everywhere** (the
interleaved single-stream write ≈ the two-plane write). Real-vs-complex
harvest moves 1.31× → **1.60×** (our r2c 283.1 vs our c2c 452.0 at 256²)
against MKL's 2.24×. Post-change profile (256² fwd): total 293.6µs, wrapper
0%, p1 transpose-in 10.9%, **inner-r2c 43.3%** (the next campaign slice),
transpose-out+pad 12.7%; the OOP path lacks p2/p3 probes (re-instrument
before the next attribution round — noted, cosmetic).

Campaign remaining, in the §6a29-addendum order: inner-r2c share (43%),
transpose pair (~24%), p2/p3 re-attribution. The 2D z contract is now
native, gated, and free.


## 6a31 — 2D row pass: measured inner-engine selection (rfft vs stride)

**Second slice of the 2D campaign, off the refreshed phase table (inner-r2c
44.2%/40.5% at 256²/512²).** B-sweep first eliminated tile width as a lever
(B=16 worse, B=32 parity — the default 8 sits at a local optimum). The real
finding: the tile's transposed layout (N2×B lane-contiguous) is exactly the
rfft natural engine's input shape, and isolated measurement showed the rfft
engine 27% faster than the hardwired stride inner at the tile shape
((256,8): 2.885 vs 3.969 µs/call).

**Shipped:** the vfft layer force-creates an rfft-path 1D plan at (N2, B)
(the existing decouple_min_k forcing trick), injects the raw rfft_plan_t
into the 2D data, and the tile worker uses it when the row pass runs
single-threaded (in-place-safe: the leaf fully consumes x before the
terminator writes; the stride inner keeps per-tid scratch for MT).

**The lesson inside the slice: "rfft wins at low K" did not survive
N-scaling.** Unconditional adoption regressed (512,8) by +66% (2302 vs
1382 µs — plane working-set vs tile scratch, mechanism unattributed).
**Adoption is therefore MEASURED at plan create**: both inners A/B'd on tile
scratch, same-process, 64 reps each (~sub-ms), winner kept. 256² adopts
rfft; 512² re-picks stride; the gate is self-calibrating per cell/host.

| cell fwd | §6a29 baseline | §6a30 (wrapper) | §6a31 (inner) | split/MKL-real |
|---|---|---|---|---|
| 256² | 332.4 | 283.1 | **262.1 (−21.1% cum.)** | 0.615 → **0.783×** |
| 512² | 1652.3 | 1382.2 | 1363.8 (stride re-picked) | 0.601 → 0.723× |

z stays at split parity (+0.3%/−0.7%). Gate 18/18 (roundtrip epsilons
shifted e-16 → e-15 at the rfft-adopted cells — different codelet sequence,
still clean — proving the engine actually runs). Real-vs-complex harvest:
1.60× → **1.72×** at 256² vs MKL's 2.24×.

**Open notes:** the (512,8) rfft-slower mechanism is unattributed (plane
thrash vs forced factorization — a future micro-study); the bwd row pass
(c2r engine, §6a28-familiar) is the symmetric next slice; col-c2c (22%) and
the transpose pair (~24%) follow. Profile-build totals run ~10% above
bench totals (timer overhead per tile) — compare within builds only.


## 6a32 — bwd row inner (c2r mirror): machinery shipped, adopts nowhere measured

**The §6a31 mirror:** c2r natural-engine row inner for the C2R 2D plan —
same injection (vfft layer owns a NATURAL-layout c2r dispatch plan, raw
c2r_plan_t into the 2D data), same in-place-safety argument (the initiator
consumes all input rows in stage 0, output written last), same ST-only rule,
same **measured adoption at create**. Both A/B gates also gained per-rep
refill from a saved pattern (unnormalized reps compound to inf otherwise —
x86 inf arithmetic is full-speed so prior decisions were likely right, but
the hygiene hole is closed; both arms equally handicapped).

**Verdict (same-process forced-arm A/B, build_tuned/benches/bench_2d_row_engine_ab.c):**

| cell/dir | adoption | engine delta (adopted vs stride) |
|---|---|---|
| 256² fwd | RFFT | **−3.2%** (noise floor ±1.4% on NULL-vs-NULL arms) |
| 256² bwd | **stride kept** | +0.2% = floor (same engine both arms) |
| 512² fwd | stride kept (§6a31 gate) | +1.4% = floor |
| 512² bwd | stride kept | +1.0% = floor |

**The c2r natural inner is slower than the stride bwd inner at every
measured tile shape on this container** — the gate adopts it nowhere. The
machinery stays (it self-enables on any host/shape where it measures
faster); outcome recorded as negative-for-now, zero risk by construction.
Asymmetry note: the fwd rfft engine wins at (256,8) while the bwd c2r
natural loses at the same shape — the bwd stride inner's fused structure
holds up better than its fwd counterpart did; mechanism unattributed.

**Weather-ghost specimen #2:** the cross-run bench during this slice showed
"bwd −31%" and MKL itself "speeding up 45%" — a container regime shift
mid-session (23:00+). The forced-arm same-process A/B showed both bwd arms
identical. The §6a31 fwd win also re-quantifies honestly: **−3.2%**
same-process, not the −7% cross-run figure — the §6a31 table's cumulative
claims should be read with that correction (absolute µs rows are
regime-stamped; engine deltas above are the durable numbers).


## 6a33 — 2D campaign checkpoint: compute beats MKL, movement IS the gap

**No code in this section — the analysis that closes the slicing campaign at
a principled boundary.**

**Machinery status after §6a29–§6a32, per phase:**

| phase | share (256² fwd) | machinery | headroom under this architecture |
|---|---|---|---|
| wrapper | 0% | eliminated (§6a30) | none |
| transpose-in | ~11% | 8×8 ZMM / 8×4 YMM line-filling kernels, L1/L2/large regime-tiered | engineered; none material |
| inner-r2c | ~40% | measured-best engine per plan (rfft vs stride, §6a31 gate) | 1D-engine-class work only |
| transpose-out+pad | ~13% | same engineered kernels + K_pad zeroing | none material |
| col-c2c | ~21% | **c2c jit BOUND** at both cells | c2c-jit-class work only |
| pack / z-interleave | ~11% | fused perm loop (§6a30) | none material |

**The decomposition that matters (256², regime-stamped):** compute phases
(inner ~113 µs + col ~60 µs) ≈ **173 µs — already below MKL-real's entire
205 µs**. Data movement (transposes + pack) ≈ **101 µs ≈ the whole gap**.
The tiled-transpose composition pays two full transposes and a pack pass
that MKL's fused 2D real design does not; our kernels for those passes are
at engineering quality — the passes themselves are the cost.

**Therefore: the remaining 0.72–0.78× vs MKL-real is ARCHITECTURAL.**
Closing it means a movement-free composition ("2D v2"): strided-lane leaf
and terminator codelet families that consume row-major real input and
produce the pad/output layouts directly — generator work, adjacent to the
Gap-A post-tw OOP mode already specced (design doc §6). Multi-session
project; parked as a designed direction, not a slice.

**Campaign ledger (§6a29 baseline → §6a32, 256² fwd):** 332.4 → ~262 µs
regime-adjusted (−21%), split/MKL 0.615 → ~0.78×, z tax +16% → 0, harvest
1.31× → 1.72× vs MKL's 2.24×. Open mechanism studies carried: (512,8)
rfft-slower, bwd stride-inner robustness asymmetry.


## 6a34 — v2 spike + two v1 bycatches: skinny transpose, gate hysteresis

**The v2 feasibility spike** (benches-grade micro, register-block vs
engineered vs copy at tile shapes) validated the fused-IO thesis and
produced the design doc: **docs/roadmap/fft2d_v2_design.md** — block-
transposing codelet IO (leaf-bt / terminator-bt emission modes), staged,
budgeted to ~0.92–0.95× MKL-real at 256², v1 fallback by construction.

**Bycatch 1 — skinny transpose fast path (shipped):** the spike showed the
regime-tiered kernels lose 35% to a plain 8×4 register sweep at the skinny
tile shape (256×8, L1). stride_transpose now takes the 8×4 path when one
dim ≤ 8, the grid divides, and the working set is L1-resident (the same
spike measured the sweep LOSING at (512×8) — the guard is shape AND size).
Standalone-proven −35%; in-2D contribution (~−6–10 µs/plane/direction at
256²) is projected, not re-measured — container weather entered a third
regime this session (MKL absolutes swung 205→167 and 986→790) and
cross-build totals are mush. Gate 18/18 (transpose is value-preserving).

**Bycatch 2 — adoption-gate hysteresis (shipped):** the same weather flip
showed both §6a31/32 create-time gates choosing OPPOSITE winners across
regimes — the true engine deltas (≤3%) sit inside create-time measurement
noise. Both gates now require the challenger to beat the stride incumbent
by **>5%** (t*20 < t*19). Near-ties keep the incumbent; churn eliminated;
the v2 design carries the rule as mandatory for all future gates, with
wisdom-persistence of decisions as the follow-on.


### 6a34 addendum — v2 §4 corrected by stage-1 recon + prior art

Stage-1 recon (emitted-leaf load patterns) invalidated the original v2 §4
mechanism: cascade leaf/terminator bt-preludes die on the DIT fold's
S-strided columns. The governing prior art — found after the design was
written, the same class of process failure as the §6a29 v1.0 incident —
is the **strided codelet quadrant** (strided_rows_case_study.md, Design C):
mono bt-IO codelets, wired and gated, 1.72×/1.40× measured on row passes,
naturally ordered for free, capped at N=64 and c2c-only. v2 is now
redefined as the quadrant's own named growth directions: strided r2c mono
emission + strided twiddle-stage (DIF-front) codelets for large N2. The
design doc carries the correction in place; the check-the-record-before-
designing rule now has two specimens.


## 6a35 — v2 stage-1 probe: strided r2c mechanism VALIDATED, emission spec forced

**The probe** (build_tuned/benches/probe_strided_r2c*.c): strided r2c composed from the
EXISTING c2c strided mono via the two-for-one trick — call
`fwd(base, base+row_stride, 2*row_stride, R/2)` packing row-pairs as complex
lanes, then a conjugate-symmetry epilogue splitting each Z into two
half-spectra. Correctness: 1.7e-14 vs the native path at every R (different
arithmetic route; eps gate, not BIT); the vectorized epilogue is
BIT-identical to the scalar one.

**The composition losses and their attribution (R=4096, N=16, same-process):**

| arm / component | µs |
|---|---|
| A — v1 tiled (transp-in + (16,8) inner + transp-out) | 79.7 |
| B — wrapper composition (copy + mono + scalar epilogue) | 99.6 (+25%) |
| ... memcpy (in-place-only mono artifact) | 13.5 |
| ... **mono — the actual FFT** | **31.9** |
| ... epilogue as a separate pass | 48.9 scalar / **43.9 vectorized** |

**The two findings:**
1. **The mechanism wins decisively**: the mono ALONE (31.9 µs) beats the
   entire v1 composition (79.7). The §6a34-corrected v2 direction is
   confirmed with a number.
2. **The wrapper's parasites are both bandwidth passes** — vectorizing the
   epilogue bought only 10% because it is memory-bound (~1.3 MB extra
   traffic), the exact disease v2 exists to cure, reproduced in miniature
   inside the probe. Wrapper composition is therefore NOT the production
   shape.

**The emission spec, forced by measurement:** strided r2c = the c2c strided
body + **OOP IO** (kills the 13.5 µs copy) + **in-register conjugate-split
fused at the store lattice** (Z[f] is already in registers at store time;
the split's flops hide in store slots, zero extra traffic — measured flop
content ≤5 µs by the vec-vs-scalar delta). Projected: **~35 µs vs 79.7 =
−55%** on covered-N row passes, with every term in the projection
individually measured. Two-for-one is the internal mechanism (even lane
pairs; odd-R tail rides the quadrant's existing padded-tail machinery).

**Reach:** covers N2 ≤ 64 today (mono ceiling). The campaign cells
(N2 = 256/512) still require the strided twiddle-stage (DIF-front)
direction; the fused-split store lattice transfers to it directly.
Generator work items, in order: (1) --strided-r2c emission mode (OOP +
fused split), (2) bwd mirror, (3) the twiddle-stage family.


## 6a36 — hand-written fused strided r2c codelet: the emission spec VALIDATED

**r16_r2c_fwd_strided.c** (codelets/strided/avx2/, hand-written, marked as
the emission reference): the §6a35 spec realized — the emitted c2c strided
mono's DAG body and load lattice untouched; two-for-one via load addressing
(rio_re = rio, rio_im = rio + row_stride, pair stride 2·rs); the store
lattice replaced by the in-register conjugate split (`_SPL` on the lane
vectors, wrap-mirror g=(16−f)&15 handling DC/Nyquist automatically) feeding
even/odd-row inverse-transpose store groups + a Nyquist column.

| R | v1 tiled composition | fused strided r2c (ONE call) | delta |
|---|---|---|---|
| 256 | 7.06 µs | **3.41** | **−51.7%** |
| 4096 | 122.0 µs | **67.3** | **−44.9%** |

Gate 1.9–2.5e-14 vs the native path at both scales (eps, not BIT — different
arithmetic route, as established in §6a35). The §6a35 projection (−55%)
essentially achieved; absolutes are regime-stamped (weather moved A itself
79.7→122 between sessions), the same-process deltas are the durable claim.

**What this buys the program:** the --strided-r2c OCaml mode is now
de-risked to a transcription task — its acceptance gate is byte-level
equivalence to this reference (modulo naming), and every design decision in
the emission (OOP, fused split at the lattice, two-for-one addressing,
even/odd store groups) carries a measured justification from §6a35/36.
Remaining emission work: the mode itself (fwd), bwd mirror, N ∈
{4,8,12,20,32,64} coverage via the same recipe, then the twiddle-stage
family for the N2 ≥ 128 campaign cells.


## 6a37 — --strided-r2c generator mode SHIPPED: emitted ≡ hand reference, BIT

**The OCaml mode** (emit_state.ml ref + gen_main.ml flag/wiring/name-suffix
+ emit_c.ml signature/prologue/postamble branches): --strided-r2c rides the
--strided machinery — the DAG body and load lattice untouched; the
signature becomes (rio, out_re, out_im, tw_re, tw_im, row_stride_in,
out_stride, me) with tw kept for family uniformity; the prologue's local
`row_stride = 2*row_stride_in` makes the pair stride flow through the load
lattice with ZERO load-emission changes; the postamble is the parameterized
fused split (per-bin _SPL vectors with mirror g=(n−f) mod n, h/4 full
4-bin store-group chunks, scalar tail bins incl. Nyquist). Every out_lane
is consumed via the mirrors — no dead stores by construction.

**Acceptance (three-arm same-process, probe_strided_r2c_fused):**

| | R=256 | R=4096 |
|---|---|---|
| gate vs native | 1.9e-14 PASS | 2.5e-14 PASS |
| **emitted vs hand reference** | **BIT-IDENTICAL** | **BIT-IDENTICAL** |
| v1 tiled | 5.06 µs | 79.71 µs |
| emitted | 2.33 (−54.0%) | **37.06 (−53.5%)** |

(+7.1% vs hand at R=256 = 0.16 µs absolute, code-layout class; exact
parity at R=4096. One ABI lesson en route: the family-uniform tw args made
the emitted signature 8-arg vs the hand file's 6 — a probe segfault until
the extern matched. The emitted ABI is the production one.)

**Coverage: N ∈ {8,12,16,20,32,64} emitted into codelets/strided/avx2/
(rN_n1_fwd_strided_r2c.c), each gated** — r16 BIT vs the reference;
the rest vs an independent naive O(N²) DFT: r8 1.2e-15, r12 8.9e-15,
r20 9.3e-15, r32 3.6e-14, r64 6.9e-14 (chunk/tail logic exercised at
full-chunks+Nyquist, chunk+multi-scalar-tail, and small-N shapes). The
hand-written r16 reference is SUPERSEDED and removed (bit-identical;
generator is the single source of truth; its history lives in §6a36).

**Remaining per the v2 plan:** bwd mirror (--strided-r2c --bwd: c2r
two-for-one — merge epilogue becomes a PROLOGUE building Z from two
half-spectra before the c2c bwd body), integration into the 2D row pass
behind the §6a31 gate machinery, then the twiddle-stage family for
N2 ≥ 128.


## 6a38 — --strided-r2c --bwd (c2r mirror): SHIPPED, roundtrip machine-epsilon

The bwd mirror confirmed the "easy" prediction: the store lattice needed
ZERO changes (the c2c bwd body's Re/Im lanes ARE the even/odd real rows —
the same pair-shadow trick applied on the out side), the body is the
existing --bwd machinery, and the only new emission is the **merge
prologue** replacing the load lattice: transposing-load groups for the
four half-planes (X1/X2 × re/im, even/odd row addressing) + scalar set_pd
tail gathers, then Z[f] = X1[f] + i·X2[f] and the Hermitian mirror
Z[n−f] = conj(X1[f]) + i·conj(X2[f]) — the general formula absorbing
DC/Nyquist via the zero-imag contract, all n lane vectors covered.

Signature: (in_re, in_im, out, tw_re, tw_im, in_stride, row_stride_in, me);
output unnormalized (rows = N·x), me = PAIRS. Plumbing: one new ref
(strided_r2c_bwd := strided_r2c && bwd), fwd guards tightened, the
postamble falling through to the untouched plain store lattice.

**Roundtrip gates |bwd(fwd(x))/N − x|, emitted fwd × emitted bwd:**
r16 at R=256: **4.44e-16**; r8/12/20/32/64 at R=64: all PASS at
machine-epsilon scale. Coverage installed:
codelets/strided/avx2/rN_n1_bwd_strided_r2c.c, N ∈ {8,12,16,20,32,64}.

**The strided r2c/c2r codelet family is COMPLETE both directions.**
Remaining per the v2 plan: 2D row-pass integration (§6a31 gate machinery +
hysteresis; the bwd side slots into the same worker branch the §6a32
c2r-natural experiment used), then the twiddle-stage family for N2 ≥ 128.


### 6a38 addendum — old-mode emission stability audit (user-prompted)

Question: are the r2c/c2r strided codelets a replacement, overwrite, or
addition? **Addition on every axis**, with proof: the patched generator
re-emits the pre-existing modes BYTE-IDENTICAL to the checked-in files
(`--strided` fwd and bwd both verified; the only diffs ever observed were
the provenance header's argv echo — path and flag-order, not code). Zero
old codelet files modified or removed; the new family is a separate
domain (r2c/c2r vs c2c), separate ABI, and not yet wired into any runtime
path — production behavior today is byte-for-byte pre-§6a37.


## 6a39 — strided r2c/c2r integrated into the 2D row pass: MKL PARITY at a covered cell

**The integration** (fft2d_r2c.h, self-contained — no vfft-layer wiring
needed this time): per-N resolver over the §6a37/38 family; struct fields
strided_fwd/strided_bwd; the executor branches replace the ENTIRE tiled row
pass with one strided sweep (fwd: user real plane → re_pad/im_pad at
out_stride=K_pad + pad-tail zeroing; bwd: pads → real rows). Eligibility:
N2 ∈ {8,12,16,20,32,64}, N1 % 8 == 0, execute re-guards T ≤ 1. **Adoption
is measured at plan create with the §6a34 >5% hysteresis** — fwd arms
refill-free (preserved input, pads as output), bwd arms read pads
read-only; MT-tiled-vs-ST-strided at create noted as conservative
under-adoption. The z contract inherits the engines automatically (the z
variants route through the same OOP executors).

**Gate: 21/21 ALL PASS** including the new covered cell (64×64, roundtrip
1.0e-15); the uncovered cells never adopt — regression-proof by
construction.

**Payoff (same-process forced arms, build_tuned/benches/bench_2d_strided_ab.c):**

| cell | adoption | fwd (strided vs tiled) | bwd | split/MKL-real |
|---|---|---|---|---|
| (256×32) | both | **−20.2%** (19.5 vs 24.4 µs) | −16.4% | 0.801 → **1.004× — PARITY** |
| (4096×64) | both | **−29.8%** (995 vs 1418 µs) | −10.0% | 0.632 → **0.901×** |

**(256×32) is the first 2D real cell where VectorFFT reaches MKL-real
parity** — the campaign target achieved wherever the strided family
covers. The (4096×64) fwd −29.8% mirrors the case study's c2c-side 3D
wins on the r2c side. Container note: the strided r2c objects live at
/tmp/osr2c/*.o and must precede the archives in the link TAIL.

**v2 remaining: the twiddle-stage family** (N2 ≥ 128) — the single piece
between this result and the 256²/512² campaign cells.


### 6a39 addendum — vs MKL's INTERLEAVED solutions, both domains (user-prompted)

All within-run same-process ratios; this container regime runs ~25-35%
faster than the §6a29 one (every arm moved together).

**Complex domain (256², vfft split-native in-place scrambled,
copy-corrected 271.9 µs):** vs MKL complex-SPLIT 382.7 → **1.41×**; vs MKL
complex-INTERLEAVED (their native preferred format) 319.1 → **1.17×**.
Cross-regime honesty: the §6a29 regime read 0.93× vs IL — the band is
parity-to-winning against MKL's own complex layout, clear wins against the
identical-split config.

**Real domain, interleaved-out vs interleaved-out (our z vs MKL real-CCE),
strided-covered cells:**

| cell | our z | MKL CCE | z/MKL |
|---|---|---|---|
| (256×32) fwd | 17.2 µs | 18.1 | **1.057×** |
| (256×32) bwd | 17.7 | 22.0 | **1.240×** |
| (4096×64) fwd | 1204.8 | 878.7 | 0.729× (bench doesn't print adoption; regime swung our side 995→1220 across runs — within-run only) |
| (4096×64) bwd | 1137.9 | 1037.3 | 0.912× |

**Bonus finding at (256×32): z now BEATS split (−11.6% fwd, −5.2% bwd)** —
with the strided engine writing pads, the fused z-interleave (ONE output
stream) is cheaper than the split pack (TWO plane streams). The §6a26
single-stream lesson, reappearing at pack scale.

**Architectural comparison in one paragraph:** MKL is interleaved-native
end to end — kernels consume and produce (re,im) pairs, paying pair-shuffle
arithmetic inside the math for a one-stream memory profile. VectorFFT is
split-native in the math (zero shuffles in any butterfly — the lane-batched
doctrine, docs/design/lane_batched_simd.md) and interleaves only at the
boundary via the fused pack. The covered-cell results are the doctrine's
vindication: shuffle-free math + shuffles-at-the-door now beats MKL's
native-interleaved real path outright at (256×32), both directions.


## 6a40 — twiddle-stage design study: the composition law, measured

**The probe** (build_tuned/benches/probe_dif_front.c): a hand DIF radix-2 front
(contiguous half-span butterflies, vectorized ALONG the row — no
transposes, no row batching needed) composed with the EXISTING r64 strided
monos via base-offset sub-rows = a complete N=128 strided c2c row pass with
zero new emission. Multiset gate 1.1e-14 (ordering deferred by design, the
case study's own gate style).

**The findings (same-process, R=4096 DRAM-scale / R=256 cache-scale):**

| arm | R=4096 | R=256 |
|---|---|---|
| A tiled (transp + inner + transp) | 1697 µs | 72 |
| B plain 2-sweep (OOP front, then monos) | 1801 (+6%) | 57 (−21%) |
| **C row-blocked fused (front + monos per 8-row block)** | **1541 (−9%)** | **56 (−22%)** |

Attribution en route (R=4096): memcpy 621 (in-place-mono artifact),
front_ip 315, front_OOP 725 (write-allocate RFO doubles the store cost),
monos 910.

**The composition law:** (1) multi-sweep strided composition LOSES at DRAM
scale — each stage is a full-plane pass, while the tiled incumbent reads
each row once and works cache-resident; pass-count is the tax. (2)
**Row-blocked fusion restores the one-DRAM-pass structure and wins both
regimes.** (3) At cache scale even the naive composition wins (passes are
free in cache).

**Emission target, now numbers-backed:** a fused strided twiddle-stage
codelet — front + sub-FFTs in ONE body per row-block — eliminating arm C's
remaining per-block call overhead and the L1 store/reload between front and
monos. C's −9%/−22% is the floor, not the ceiling. Remaining design work
for the r2c edition: the Sorensen real front (real row → real half +
complex half), r2c mono + c2c mono leaves, and the half-spectrum assembly/
ordering map — the one genuinely open design question.


## 6a41 — twiddle-stage engine SHIPPED (correct, integrated, dormant) + the gate-fidelity lesson

**Built** (src/core/transforms/fft2d/strided_tw.h + fft2d_r2c.h wiring):
DIF r2/r4 fronts both directions (engine kernels, vectorized along the
row), the DIF ordering map (Z[r·k+j] at col j·64+k, nothing reordered in
memory), mapped conjugate split/merge, and row-blocked r2c/c2r
compositions for N2 ∈ {128,256} per the §6a40 law (one DRAM pass).
Correctness: exact-position vs naive DFT 3.4e-13/8.2e-13 (N=128/256),
roundtrips 8.9e-16/1.0e-15; full gate **24/24** with new covered cell
(64,256) at 1.1e-14.

**The gate-fidelity lesson (measured the hard way):** the first adoption
gate — isolated hot-looped row-pass A/B, the §6a39 pattern — MISADOPTED
stw at 256²: create measured >5% better, the execute-context forced-arm
race measured **+15.2% WORSE**. Root cause: create context (row pass
alone, hot) vs execute context (phase-interleaved with col+pack). Fix
shipped: **the stw A/B now runs the FULL fwd/bwd executors with the flag
toggled** — same phase interleaving as production. Re-gated: 256² and 128²
both correctly decline (+1.3%/+1.9% and −0.4%/−0.4% forced-arm — ties,
properly rejected). The §6a39 mono gates keep their pattern (their
adoptions were independently validated by forced-arm benches); migrating
them to full-executor arms is a noted refinement.

**The honest campaign verdict:** the hand composition TIES the production
tiled row pass everywhere tested — §6a40's −9% was against a
dispatch-heavy stand-in, and the real incumbent (measured-selected
engines) absorbs the composition's remaining overheads: per-block call
structure, the front→mono L1 reload, and the SCALAR mapped split (the map
scatters f-adjacent bins 64 columns apart, blocking the §6a36 vector
split at this layer). **The fused emitted codelet is confirmed necessary
for the campaign cells**, with its spec sharpened by this session:
split-before-map — compute the §6a36 vector split on lane vectors BEFORE
the ordering map, then let the map live in the store addressing where it
is free. The §6a41 machinery (resolver, full-executor gates, execute
branches, gate cells) is the finished integration bed it slots into.

**Taxonomy guide shipped:** docs/design/strided_codelet_families.md — the
four families (c2c monos + IL variants; r2c/c2r monos; stw engine
kernels; the planned fused family), ABIs, coverage, call sites,
provenance flags, and a caller decision table.


## 6a42 — the fused family already existed: large-N strided r2c monos at the campaign cell

**The discovery:** the §6a41 spec called for a fused single-body
twiddle-stage codelet. Testing before building: gen_radix.exe {128,256}
--strided-r2c [--bwd] EMITS IT — the N=64 mono ceiling was convention, not
capability. Monolithic construction absorbs the "front stages" as DAG
depth; the §6a37 split/merge postambles consume out_lane_0..N-1 unchanged;
split-before-map holds by construction (the mono IS natural order — there
is no map). Zero new OCaml. 314/681 KB emissions, 7-13 s compiles.

**Gates:** naive-DFT 3.2e-13 (128) / 7.6e-13 (256); roundtrips 8.9e-16 /
7.8e-16; resolver extended to 128/256; full 2D gate **24/24**.

**Measured verdict at 256² (forced arms, same regime):** **bwd −7.3% vs
tiled — ADOPTED by the create gate**; fwd +2.5% — a tie, correctly
declined (the 16-reg monolithic's spills tie the §6a31-tuned fwd
incumbent; the bwd incumbent, where §6a32's mirror adopted nowhere, falls).

**Campaign table, 256² this regime (bench_rz2d, same-process):**

| | split/MKL-real | z/MKL-real |
|---|---|---|
| fwd | 0.872× | 0.886× |
| **bwd** | **0.974×** | 0.956× |

From 0.60× at the §6a29 start to 0.87×/0.97× — bwd a hair from parity with
the fused mono live. **The remaining fwd gap is precisely located:** the
monolithic 256-DAG's register pressure. Candidates in order:
regalloc/pinning gate extension to this size class, GH pressure mode, a
CT-blocked strided construction. N2=512 emission untested. Files
installed: codelets/strided/avx2/r{128,256}_n1_{fwd,bwd}_strided_r2c.c.


## 6a43 — M-knobs measured out at R=256; the family extended to N2=512

**The fwd-gap experiment:** the r256 fwd mono's provenance showed GH
pressure mode already ON (auto-rule) and Regalloc+pinning OFF — but the
gate string in the provenance is STALE TEXT: the M-project note
(emit_c.ml, 2026-06-09) retired the auto-pin policy after measurement
("net-negative or a tie in every cell", R=4..128, gcc-13) and left the
knobs opt-in (VFFT_PIN_FORCE / VFFT_FORCE_FENCE) with an explicit
fresh-measurement warning. R=256 was out-of-sample; measured now
(same-process, row-pass level, R=256 rows): **PIN +14.4%, FENCE +6.8% —
both lose.** The M-retirement extends to this size class. (The 4454
value-diffs vs baseline are the documented pinned-form FMA-contraction
difference, not an error.) The fwd mono stands as-is at its local floor;
the create gate's decline at 256² is the correct steady state.

**N2=512 family shipped:** gen_radix 512 --strided-r2c [--bwd] emits
(1.47/1.45 MB); naive-DFT gate 2.9e-12, roundtrip 1.0e-15; installed
(r512_n1_{fwd,bwd}_strided_r2c.c), objects in /tmp/osr2c, resolver
extended, new gate cell (64,512) at 1.2e-15 — full gate **27/27**.
Coverage is now N2 ∈ {8..64, 128, 256, 512}: every power-of-2 campaign
row length. The 512² whole-cell bench (adoption + forced arms + MKL) is
deferred to a fresh session — the 2D MEASURE create at N1=512 runs
minutes and deserves its own timeout budget; based on the 256² pattern
the bwd mono is the likely adopter there.

**Where the remaining fwd deficit lives** is the open question §6a42/43
sharpened: with fwd row engines tying each other, the 0.87× fwd gap at
256² sits in the phase distribution (col pass, pack, or the tie itself)
— a bench_f2d_profile refresh under current engines is the named next
diagnostic.


### 6a43 addendum — GH measured out too (user-corrected)

The §6a43 text treated "GH pressure mode: true" as a live optimization on
the large-N emissions. User correction: GH is obsoleted alongside the
M-knobs, and the provenance's "+4-8% documented" is stale for this size
class (its measured basis is R={32,64}). Measured now with a new
VFFT_NO_GH env guard on the auto-force (gen_main.ml, matching the M-note's
opt-in env pattern): r256 fwd **GH-off is BIT-IDENTICAL (0 diffs — pure
scheduling) and −0.9% — a no-op.** No re-emission (tie = no churn); the
guard stays as diagnostic infrastructure.

**Complete knob ledger at the r256-fwd size class:** PIN +14.4%, FENCE
+6.8%, GH-off −0.9%. The fwd mono is schedule-knob-exhausted; remaining
paths are structural (CT-blocked strided construction — parked, real
emitter work: the blocked recipe's spill markers are not understood by the
strided emitter, gen_main ~290) or elsewhere in the phase budget (profile
refresh). Provenance staleness now documented for BOTH the regalloc gate
string AND the GH benefit claim.


## 6a44 — strided MT: range-split shipped, BIT-invariant by construction

The mono tier's ST-only guards are lifted. _f2d_sr2c_{fwd,bwd}_run
(fft2d_r2c.h) mirror the tiled path's pool-dispatch pattern but simpler —
the codelets are scratch-free, so no per-thread slot rationing. Chunks are
masked to 4-pair (8-row) multiples: every thread executes exactly the
blocks ST would, so **MT output is BIT-IDENTICAL to ST by construction**.
The §6a39 adoption arms now call the wrappers (MT-faithful A/B at any
create-time T). The stw tier stays ST (shared work buffer; dormant tier).

Gate (build_tuned/benches/gate_strided_mt.c, real dispatched workers — pool 1 and 3):
T ∈ {1,2,4} at (64,64), fwd and bwd, **BIT at every T**; full 2D gate
27/27 unchanged. MT SPEED is unmeasurable in this 1-vCPU container
(threads oversubscribe one core) — correctness is proven here, the
speedup claim waits for a multicore host (14900KF race noted in the
strided case study's §5 applies to this family too).


## 6a45 — avx512 8×8 strided-r2c editions: emitted, gated, wired, winning

**Emission (emit_c.ml):** the r2c split postamble and merge prologue gained
width-8 branches built from the tree's own 3-stage lattice vocabulary
(unpack pairs → permutex2var _tp_idx_lo/hi → shuffle_f64x2 0x44/0xEE):
fwd = _SPL on __m512d lane vectors (8 lanes = 8 pairs = 16 rows/block) +
four 3-stage store groups per 8-bin chunk + 32-double scalar tails incl.
Nyquist; bwd = 8×8 transposing loads of the four half-planes + 8-arg
reversed set_pd tails + the merge. The _hx declarations went ISA-typed.
Coverage {8,16,32,64,128,256,512} both directions (N=12/20 excluded:
radix % 8 ≠ 0 — avx2-only, resolver falls back).

**Gates: 14/14 standalone PASS** (naive-DFT 1.6e-15…2.9e-12, roundtrips
2.2e-16…1.0e-15, native on this avx512 host). **ISA race r256 fwd: the
avx512 edition is BIT-IDENTICAL to avx2 (same DAG, wider lanes — zero
value drift) and −9.0% isolated.**

**Two build-convention discoveries, now doctrine:** (1) tree-wide avx512
codelets (old c2c and new r2c alike) carry target("avx512f") ONLY, while
bodies use DQ (_mm512_xor_pd) and the il family uses BMI2 (_pdep_u32) —
**avx512 codelet objects MUST be compiled with -mavx512f -mavx512dq
-mbmi2**; the per-function attr does not stand alone. Upgrading
isa.target_attr to "avx512f,avx512dq" is a recorded improvement (needs a
full-quadrant regeneration audit). (2) codelets/il/avx512 contains TWO
filename generations (*_emit.c and *_avx512.c) defining IDENTICAL
symbols — never bulk-compile that dir; per-symbol first-match picks only.
Dedupe audit queued.

**Integration:** fft2d_r2c.h resolvers prefer avx512 under
`#if defined(__AVX512F__) && defined(__AVX512DQ__)` (build-target
selection, the strided_rows.h convention). The avx512-flagged full build
converges with /tmp/ox512 (462 objects: inplace/avx512 complete + il
per-symbol picks) + /tmp/osr512 (the 14 editions) in the link tail.

**stw demoted to fallback-only (structural fix):** under the avx512 build
the stw create A/B misadopted (+20.4% forced-arm regression) despite the
§6a41 full-executor arms. Rather than a third fidelity round: family 4
supersedes family 3 at every covered N2, so stw adoption is now gated on
`!_f2d_sr2c_fwd_resolve(N2)` — eligible only where the monos have no
coverage (currently nowhere). The misadoption class is closed by
construction.

**Campaign results under the avx512 build (same-process forced arms):**

| cell | fwd | bwd | split/MKL-real |
|---|---|---|---|
| (256,32) | MONO **−45.6%** | MONO −31.2% | **1.097× — BEATS MKL** |
| 256² | MONO **−9.3%** (avx2 edition tied; avx512 wins) | MONO **−15.1%** | 0.853× adopted, this regime |

Both gate variants green: **avx2 build 27/27, avx512 build 27/27.**


## 6a46/Q0 — avx512 row-multiple hazard: found by queue review, fixed, gated

Latent §6a45 bug: avx512 editions consume 8 pairs (16 rows) per block but
eligibility still checked N1 %% 8. At pairs %% 8 != 0 the tail block
OVERRUNS (demo at me=20: heap corruption, malloc abort — worse than wrong
values). Every existing gate cell was a lucky multiple of 16 rows.
**Fix:** resolvers are (N2, pairs)-aware — avx512 editions only when
pairs %% 8 == 0, else the avx2 editions (4-pair blocks). MT chunk mask
widened to 8-pair alignment (valid for both editions; boundaries remain
block-aligned for each, so MT stays BIT-identical to ST). New gate cell
(40, 32) — 20 pairs — exercises the fallback. **avx2 build 28/28, avx512
build 28/28.**


## 6a47/Q1 — 3D real transforms lit up (the dark engine, dispatched, gated, engine-integrated)

**The discovery:** fftnd_r2c.h was a COMPLETE dark implementation — rank-N
real transforms with pads, phases, MT, JIT hooks — referenced by nothing.
vfft_create rejected dims==3 for anything but C2C. Q1's row-pass
integration became the larger feature: light the engine up.

**Dispatch (vfft.c):** dims==3 + R2C/C2R (K==1, OOP, N[2] even) builds via
stride_plan_nd_r2c(3, n, reg). Execute bridges the phases directly (the
override's buffer contract is in-place-ish; vfft's R2C is OOP): fwd =
rows → axes → unpack(dre,dim); bwd = pack → axes → rows. The §6a24 z
contract rides an il_out flip per execute (pack/unpack read it at run
time; pads are layout-agnostic).

**Strided engines pre-integrated:** the §6a47 wiring inside the builder —
snd_fwd/snd_bwd via the pairs-aware fft2d resolvers, MT-faithful adoption
arms toggling the SAME _fndr_rows_mt entry, >5% hysteresis. The engines
adopt across covered cells (MONO both directions at (8,24,32),
(16,16,256), (4,6,8); tiled correctly holds NL=10/uncovered and the
avx512 build falls back at pairs %% 8 != 0).

**Latent bug found by first-ever gating:** axis N=9 (log3 family) —
roundtrip-perfect, naive-WRONG: the c2c axis passes inherit the dag
convention's digit-scrambled output (fftnd.h's documented c2c contract),
unacceptable for a real spectrum. **Fix: per-axis naturalization ported**
— EMPIRICAL detection at build (impulse at row 1, angle-identified bins,
bijection-verified, fail-safe: any anomaly fails the build rather than
ship a scrambled spectrum; identity ⇒ no list, zero overhead for the
even-N common case), vfft_natorder_mk_cycles + cycle_pass/_inv sweeps
inside _fndr_axis_mt (fwd: naturalize after; bwd: inverse before). The
sweeps are ST v1 (light permutation; MT-ization noted).

**Gates (build_tuned/benches/gate_fndr_q1.c):** rank-3 cells — naive ND-DFT verified
(4,6,8) and (6,10,64) and the fixed (8,9,10); roundtrips 5.6e-16…1.4e-15;
z-vs-split BIT; il roundtrip; (16,16,256) meaty cell. **ALL PASS on both
builds. 2D regression: avx2 28/28, avx512 28/28.**


## 6a48/Q2 — row tails: any row count, both editions, staged

**The feature:** eligibility relaxes from rows %% 8 == 0 to rows ≥ 8, both
2D and ND. Rows-based wrappers (_f2d_sr2c_{fwd,bwd}_rows) run full blocks
through the MT path and stage the remainder (rows %% (2·blk), including an
odd lone row) through a zeroed pad block. The lone row's zero partner makes
X2 a zero spectrum by the two-for-one algebra — discarded fwd, ignored bwd.
**This retires the Q0 pairs constraint**: staging absorbs any remainder, so
the resolvers now ALWAYS prefer the avx512 editions when built and return
(fn, blk). Gate cells: (27,32) odd+ragged both editions, (44,64), (9,128),
ND (3,9,32) R=27. **Full matrix ALL PASS: 2D+ND × avx2+avx512.**

**Findings along the way (each its own keeper):**
- One un-migrated adoption arm (string drift) fed raw pair counts to the
  codelets — heap overflow caught by ASAN at fft2d_r2c.h:999 after pipe
  buffering had hidden the crash cell. Doctrine reinforced: stdbuf -oL for
  crash hunts; grep-verify every multi-site replace.
- **Prime N1 in 2D r2c is broken pre-existing**: the column pass needs a
  length-N1 c2c and prime lengths (41, 43) are outside col coverage —
  create returns NULL when the process is warm, and COLD-FIRST it half-
  succeeds with WRONG values (rt ≈ 1.0). The cold path is a fail-safe
  violation (create must never succeed with wrong math) — filed as a
  must-fix. Odd composite N1 (9, 27) is fine.
- STRIDE_ALIGNED_ALLOC now rounds size to the alignment (ASAN-strict
  conformance; glibc tolerated the old form).

## 6a49/Q3 — adoption wisdom: persisted verdicts, warm creates skip the A/Bs

src/core/planning/adopt_wisdom.h: a human-readable versioned sidecar
(strided_adopt.wis) keyed (kind, rows, NL, blk) — blk keys the build
edition, so avx2/avx512 builds bank separate records. Lookup-hit applies
both verdicts and skips the A/B blocks; miss runs them and records
(tmp+rename writes; corrupt lines skipped; caps at 512 records).

**Env decoupling lesson (measured the hard way):** the first design shared
VFFT_WISDOM_DIR — pointing it at an empty dir activates the engine's
bundle-calibration machinery and creates took **31–46 SECONDS**. The
sidecar now has its own env, **VFFT_ADOPT_WISDOM_DIR**; unset = fully off
(the default), and the bundle system is untouched.

**Gate (build_tuned/benches/gate_adopt_wisdom.c):** cold run records (file contents
verified: `2d 64 64 4 1 1` …), warm run **decision-match PASS** with create
times 1.4→0.9 ms and 1.1→0.4 ms at the small cells (the skip scales to
tens of ms at campaign cells). ND records verified (5 banked across the ND
gate's cells; double-run ALL PASS through the warm path). Full env-unset
regression: **all four gate matrices ALL PASS.**


## 6a48/Q2 — row tails: ragged and odd row counts across the strided family

**The machinery:** rows-based, tail-capable engine entries
(_f2d_sr2c_{fwd,bwd}_rows in fft2d_r2c.h) — full blocks through the MT
_run path, the remainder (rows %% (2·blk), including an odd lone row)
staged through a zeroed block. The lone row's zero partner makes X2 a zero
spectrum by the two-for-one algebra: discarded fwd, ignored bwd — odd row
counts cost nothing conceptually. **Resolvers refactored to (fn, blk):**
staging absorbs any remainder, so the Q0 pairs%%8 constraint is RETIRED —
avx512 preferred whenever built. Eligibility relaxed to rows >= 8 in both
2D and ND; tail staging buffers plan-owned, allocated only when ragged.

**The debugging arc (all filed):** (1) one adoption arm escaped the
call-site migration (string drift) — ASAN named it exactly
(fft2d_r2c.h:999) after plain runs showed only downstream heap-corruption
aborts; migrated. (2) STRIDE_ALIGNED_ALLOC now rounds size up to the
alignment (glibc tolerated the old form; ASAN-strict conformance, tree
hygiene win). (3) **Pre-existing, out of scope, MUST-FIX filed: prime N1
in 2D r2c** — the column pass needs a length-N1 c2c; at prime N1 (41, 43)
create returns NULL when the process is warm and — worse — cold-first it
can HALF-SUCCEED WITH WRONG VALUES (rt error ~1.0), a fail-safe
violation. Odd-composite N1 (9, 27) is fully healthy through the new tail
path (e-16).

**Gates:** new 2D cells (27,32) [odd + ragged both editions], (44,64),
(9,128), plus the Q0 cell now exercising staged tails under avx512; ND
cell (3,9,32) [R=27]. **All four matrices ALL PASS: 2D+ND × avx2+avx512.**

## 6a49/Q3 — adoption wisdom: persisted verdicts, warm creates skip the A/Bs

**Design:** src/core/planning/adopt_wisdom.h — a human-readable versioned
sidecar (vfftaw1; lines "kind rows NL blk fwd bwd") keyed per edition via
blk, atomic tmp+rename writes, corrupt-line tolerant, capped in-memory
table. Integrated in both consumers: 2D (the combined sf&&sb gate) and ND
— lookup hit applies both verdicts and skips the arms; miss runs and
records.

**The env lesson (measured the hard way):** the first cut shared
VFFT_WISDOM_DIR — pointing that at an empty dir activates the engine's
bundle-calibration machinery and creates took **31–47 SECONDS**. The
sidecar now owns **VFFT_ADOPT_WISDOM_DIR** (unset = fully off), completely
decoupled from bundle semantics.

**Gate (build_tuned/benches/gate_adopt_wisdom.c):** cold run records (file contents
verified: 2d 64 64 4 1 1 / 2d 256 32 4 1 1), warm run **decision-match
PASS** with create times 1.4→0.9 ms and 1.1→0.4 ms at the small cells
(the skipped A/Bs are tens of ms at campaign-size cells). ND records
persist through the ND gate (5 keys). Env-unset regression: **all four
gate matrices ALL PASS unchanged.**


## 6a50/Q4 — 2D howmany hole closed

Probe demonstrated the hazard live: dims==2 with howmany=2 CREATED
SUCCESSFULLY (both r2c and c2c) while the 2D executors are K-blind — the
silent-wrong class the padding gate's own contract forbids. Fix: dims==2
now rejects K != 1 up front (same contract as 3D). Permanent gate line
([2D K=2] reject) added; 2D avx2 matrix ALL PASS. Sequential-plane 2D/3D
batching remains a designed feature (needs its own dist convention — the
documented howmany is the 1D lane-batched layout). Full 4-matrix regate
of this early-return queued for next session (no existing cell uses
K > 1; risk minimal).


## 6a51 — the prime-N1 fail-safe: empirical col-path verification at create

**Root cause found:** the 2D r2c pack permutation is computed BLIND from
plan_col->factors as standard mixed-radix digit-reversal. For any col plan
whose true output ordering differs — prime N1 was the demonstrated case —
the perm is silently wrong: cold-first creates half-succeeded with rt~1.0
WRONG spectra (the library's only known silent-wrong path); warm creates
failed NULL upstream (order-dependent, untagged). The perm predates the
§6a47b lesson; the cure is the same pattern.

**Fix:** create now runs an EMPIRICAL verification right after the perm
build — impulse at row 1 through the PRODUCTION col call (jit-or-generic),
all N1 bins checked through the perm against the closed form; plus the
row-inner probe (lane-batched impulse, N2 bins) as cheap insurance for its
class (N2-even already excludes primes there). Any mismatch destroys the
plan and returns NULL with a stderr tag — never a silently wrong spectrum.
Cost: two memsets + one col FFT + one inner call at create.

**Gates:** build_tuned/benches/gate_cold_prime.c — a FRESH-PROCESS first-create at
(41,32) must be NULL (the exact cold path that used to succeed-wrong) and
(27,32) must stay healthy (1.1e-15) — PASS under both builds. Permanent
in-gate line [2D prime-N1] reject. **All four matrices ALL PASS** (2D now
33 checks). ND was already immune via the §6a47b axis detector.

The REAL prime-N1 support (Bluestein/Rader col path + a verified perm)
remains a designed feature; this closes the safety hole that made its
absence dangerous.


## 6a52 — old-debt #2 CLOSED: the DIF-bwd jit residual is gone (inverted)

The §6a21-era ledger recorded the 3-stage DIF bwd jit ~8.7% behind core at
(1000,4), with the selection rule (bwd DIT→jit, DIF→core) "retained until
closed" and a tw_buf-hoisting suspicion. Measured today, same-process,
BIT-verified both forms:

| pair at (1000,4), factors {10,10,10}, DIF | core | jit | delta |
|---|---|---|---|
| full executors (generic bwd vs baked) | 24.66 µs | 22.63 µs | **jit −8.3%** |
| ilin pipelines (the brief's exact pair: core vs jit2+range) | 23.51 µs | 22.66 µs | **jit −3.6%** |

Both jit forms are BIT-IDENTICAL to core. The residual did not merely
close — it inverted, at roughly the recorded magnitude, somewhere in the
§6a21→now jit regenerations (ver5 emission + the §6a26-27 repo/RSP
rebuilds). The tw_buf suspicion is moot.

**Rule audit:** no active production gate excludes DIF-bwd from the jit
tier today — the resolver is variant-blind, vfft.c assigns the bwd range
unconditionally, and the single remaining `use_dif_forward` guard
(oop_execute.h:52, fwd side) protects an OOP path that is DIT-by-
construction (oop_dp builds use_dif_forward=0 only) — vestigial and
correctly left in place. **Zero tree changes; the debt was ledger
staleness.** Benches preserved: build_tuned/benches/bench_dif_bwd_jit_{full,ilin}.c.

Adjacent open mechanism "jit-TU interleave 5×" remains a separate item.


## 6a53 — Gap-A CLOSED: the post-twiddle OOP family exists, is wired, and is gated

**The family (generator).** New emission mode `--post-tw` in codelet_oop.ml:
the body expands as a PURE DFT (NoTwiddles math) under the t1 ABI, and a
cmul postamble multiplies output legs 1..R-1 by W[(j-1)*me+m] just before
the UnitGroup store — out = tw (.) DFT(in), the OOP twin of
radix{R}_t1_dif_fwd, leg 0 untwiddled. Blocked n1 construction composes
(PASS-2 outputs land in out_lane accumulators, exactly what the postamble
consumes). The postamble is emission-context aware — the same store edge is
re-emitted for the main vector loop (__m256d/__m512d), the avx2 SSE2 tail
(__m128d), and scalar contexts, so the cmul renders per width (the bug that
broke twice before the width-derived prefix + scalar branch landed).
Emitted+INSTALLED: radix{5,10,20,25}_t1_dif_oop_{avx2,avx512} — overwriting
the mislabeled §6a23-era pre-tw files, resolving the naming-collision
hazard by replacement (symbols now carry the orientation). 16 standalone
closed-form gates PASS (me=7/11 through the anyk tails), ≤6.2e-14.

**The wiring (r2c.h).** _r2c_fused_first_stage_dif at BOTH fwd worker
sites: direct 11-arg calls (the engine's 7-arg n1 slot is the OTHER family
— dual-ABI landmine respected), untwiddled groups via the n1_oop siblings,
kb-blocked broadcast of the per-leg grp_tw scalars, atomic -1 on uncovered
radix (explicit-pack fallback continues untouched), site-2's Model-(b)
fork mirrored. **Variant-independent: log3-bound plans fuse too** — log3
changes tw DERIVATION only; the table rows read here are identical. The
spec's log3-parity requirement is met by construction, no log3 emission
needed.

**Measured verdict → DEFAULT OFF (opt-in VFFT_DIF_FUSED=1).** Same-process
env-toggled arms, med9:

| cell | explicit | fused | delta |
|---|---|---|---|
| (200,{10,10},K=64)  | 20.11 µs | 20.23 µs | +0.6% (wash) |
| (200,{10,10},K=16)  |  4.93 µs |  5.33 µs | +8.2% |
| (200,{10,10},K=256) | 116.1 µs | 105.4 µs | **-9.3%** |
| (250,{25,5},K=64)   | 27.50 µs | 29.51 µs | +7.3% |
| (160,{5,16},K=64)   | 15.07 µs | 13.89 µs | **-7.9%** |

Coherent physics: DIT fusion won by folding the pack into a free no-tw
leaf; DIF fusion trades the STREAMING pack for leg-strided gathers inside
the OOP codelet — it wins where pack-elimination dominates (large K, small
stage-0 radix) and loses where the gather does (small K, R=25). Mixed at
±10% ⇒ per-plan measured adoption is the NAMED FOLLOW-UP (the r2c create
has no A/B scaffold yet; Q3's adopt-wisdom can cache it); until then the
default must not regress anyone. Numerics fused-vs-explicit 3.8e-15.

**Gates.** build_tuned/benches/gate_r2c_tail.c: 11 cells — dif-defoff polarity (must
NOT fire without the env), 3 dit-fused, 2 dif-expl radix-8 (uncovered
fallback), 5 dif-FUSED with fired-assertions ({10,10} B=64/67, {25,5},
{20,8} B=65, {5,16}) ≤1.3e-12 — ALL PASS under BOTH builds (the avx512
resolver arm proven). All four big matrices ALL PASS with the new objects
linked. A/B bench preserved: build_tuned/benches/bench_dif_fused_ab.c (env-driven
BN/BK/BF0/BF1).


## 6a54 — tail-handling doctrine ported to the July features: pad-to-8

**Audit first (what was ALREADY compliant):** the 1D r2c dispatch requires
block_K to DIVIDE K and be a multiple of 8 — no partial lane block ever
exists, so the whole 1D path (including §6a53's fused-DIF codelets in
production) runs full-width; the anyk hybrid only ever fired in the
deliberately-odd B=65/67 gate cells. fft2d tiles are B=8. The one real gap:
**K_pad was roundup4** — on avx512 (VW=8) every plan whose hp1 rounded to a
4-not-8 multiple (e.g. N2=256: hp1=129 → K_pad=132) ran the anyk masked
tail on EVERY column-pass call.

**The port (4 sites):** K_pad = roundup8(hp1) in fft2d_r2c_planner,
fft2d_c2r_planner, fft2d_r2c_wisdom, and fftnd_r2c. The ND win compounds
for free: per-axis lane counts Kc[m] are products of padded extents that
include K_pad, so K_pad ≡ 0 (mod 8) makes EVERY axis pass full-width —
no per-axis padding design needed.

**Doctrine basis** (docs/roadmap/tail_handling/): padding wins when the
allocator owns the buffer (pad_vs_tail 2026-06-29, with the copy-to-pad
caveat — inapplicable here, the pads ARE the allocation); pad-vs-tail
bit-exactness proven at 21/21 K cells (arbitrary_k_tail_handling §8), so
this is a pure code-path simplification, not a numerics change. Masked
tails remain only where they belong per the padding_design_decision:
user-extent memory (the anyk hybrid in the OOP codelets, unreached in
production lane blocking).

**Gates:** all four matrices ALL PASS. Perf A/B old-vs-new K_pad deferred
honestly: it needs a runtime pad-width knob for same-process arms
(cross-process comparison banned as evidence); the doctrine docs carry the
underlying pad-vs-tail measurements.


## 6a55 — IL padded arm: built, gated, verdict env-only (tail-handling port, 1D c2c interleaved)

**Design (Target B, approved):** the interleaved z pipeline gains a padded
arm riding the SAME machinery as the public padded-batch path — plan side
only, buffers stay engine-internal (il_wr/il_wi). Kp = roundup8(K);
deinterleave into Kp-strided work (slack zeroed ONCE at alloc — linear,
lane-independent stages keep zero lanes zero both directions), full split
execute at Kp on cplan_il (aligned wisdom chain when present, auto
otherwise; jit tier resolved — vfft_proto_plan_jit_{fwd,bwd}), interleave-
out at true K. Plain-C boundary helpers, compiler-vectorized, ZERO masks.
Arm decision is FIRST-EXECUTE-lazy (matching il_wr's lazy pattern);
VFFT_IL_PAD=0/1 forces it.

**Correctness (build_tuned/benches/gate_il_pad.c, ALL PASS):** per-arm roundtrips
4.4e-16..8.9e-16; cross-arm BIT-IDENTICAL where the two arms' planners
picked the same chain ((100,5),(200,12) — lane independence makes equal
chains bit-equal); sorted-magnitude spectrum equality at 2.4e-15/4.9e-15
where chains diverge. **Contract observation recorded:** default-order z
bin order is CHAIN-DEFINED — wisdom recalibration already changes it
across plan creations today, so a padded arm choosing a different chain is
contract-equivalent; gates must compare order-free unless chains match.

**Perf (same-process, arm-locked, jit-fair, med9):**

| cell | fused-K | padded-Kp | pad delta |
|---|---|---|---|
| (100,7)   | 1.92 µs | 1.87 µs | **-2.4%** |
| (512,7)   | 14.78 µs | 15.73 µs | +6.4% |
| (1000,12) | 35.86 µs | 66.76 µs | +86% |

The +86% is DIAGNOSED, not shrugged: (1000,12) rides the calibrated
wisdom chain {25,5,8}; (1000,16) is a wisdom MISS -> cold auto chain
{25,20,2} (radix-2 last stage). Chain quality, not padding physics.
First cut also proved the jit-fairness lesson: the pad arm on generic
core executors read +102% before the jit tier was wired.

**Verdict:** exec_me auto-engage REMOVED — it was measured for the split
padded-batch arms; IL's fused folds change the trade, and engaging it
would ship users onto uncalibrated aligned cells (the §6a41 cross-context
sin, now with numbers). VFFT_IL_PAD stays as the experimental opt-in.
Named follow-up: an IL-specific verdict (own A/B at create, hysteresis,
adopt-wisdom cacheable) whose measurement would also CALIBRATE the Kp
cell — closing both gaps at once. All four big matrices ALL PASS.


## 6a56 — Target A (vectorize the IL convert-around) CLOSED NEGATIVE, measured

The premise: the fallback's element-by-element z<->split loops looked
scalar and worth hand-vectorizing. The measurement (same-process, BIT-
verified movement, build_tuned/benches/bench_il_convert_vec.c at (64,512)):

| arm | both converts |
|---|---|
| compiler loop (-O2 -mavx2) | 46.03 µs |
| hand AVX2 (unpack/perm 4-complex) | 44.44 µs (**-3.4%**) |

gcc -O2 already vectorizes the deinterleave essentially optimally; the
passes are bandwidth-bound (~5 MB touched per round here), so shuffle
work has nothing to buy. BIT diffs = 0 both directions. Converts alone
~46 µs against a ~59 µs full z execute at this cell — the cost is REAL
but STRUCTURAL: the passes exist at all, and they run single-threaded
under MT (the largest fallback population). Both structural angles file
under Target C (MT convert-split + lifting the il2il MT gate), not A.

Verdict: no code. The compiler is sufficient; the doctrine's "measure
before designing" saved an afternoon of shuffle engineering.


## 6a57 — Target A applied after all: explicit-intrinsic converts, for compiler independence

The §6a56 verdict stands as measured (gcc -O2 auto-vectorizes at parity)
but the DEPENDENCE was the problem: other toolchains and -O1 builds carry
no guarantee. The bench-proven patterns are now production:
_vfft_z_dein/_vfft_z_inter in vfft.c — AVX-512 8-complex/iter via
_mm512_permutex2var_pd (the tree's own IL-store vocabulary), AVX2
4-complex via unpack+perm2f128, plain-C floor, scalar epilogue, NO masks.

**Unification:** one flat primitive, two consumers — the IL fallback's
convert-around AND §6a55's padded-arm helpers (now per-row calls).

**Gates (both builds ALL PASS):** BIT residue sweep vs the scalar
reference across 13 lane counts covering every epilogue class (0 diffs);
natural-order z cells (100,7)/(256,5) exercising the REAL fallback path
through the new converts at 5.6e-16/6.7e-16; all §6a55 cells hold; all
four big matrices ALL PASS. The avx512 permutex2var arms verified under
the avx512 build.


## 6a58 — Target C SHIPPED: the interleaved path goes multithreaded

**C2 — il2il lane-slab dispatch.** The nthreads<=1 gate is LIFTED. The
dispatcher is _c2c_mt's pattern verbatim (S = ceil(K/T) rounded to 8,
main thread slab 0, worker offsets are pure base adds in the lane-major
layout: z±2k0, wr/wi±k0). Resolvability is PRE-FLIGHTED once — rc is
plan-deterministic, not slab-dependent — so dispatch is all-or-nothing
and a failed pre-flight falls to the convert-around with z_in untouched.
mt_unsafe routes to the fallback (same stage-codelet hazard class the
_c2c_mt self-check exists for). ST and K<8 keep the exact single-call
path. The bwd DIT tier keeps its per-slab jit2-with-core-fallback
selection (tier purity preserved per slab).

**C1 — the fallback's converts slab too.** Flat element-range slabs over
_vfft_z_dein/inter with barriers around the (already-MT) inplace
transform; NK < 4096 stays ST (dispatch overhead floor).

**Gates (build_tuned/benches/gate_il_mt.c, ALL PASS both builds):** MT-vs-ST
BIT-IDENTICAL — 0 diffs fwd AND bwd — at (200,12,T4), (256,64,T8),
(100,67,T4 ragged slabs 24/24/19), (504,40,T8), the K<8 single-slab
boundary, and the natural-order fallback cell (100,96,T4) through
slabbed converts + MT inplace. Lane independence makes slabbing
bit-invariant, and the gate proves it. Creates run at T=1 in the gate
(MEASURE calibration under T=8 on this 1-vCPU container thrashes —
recorded; plans are normally created once and executed under varying T,
so the gate's shape matches real usage). All four big matrices ALL PASS.

**Not claimed:** speedup numbers — this container has one vCPU; MT wall
gains are host-territory. What IS claimed and proven: MT interleaved now
runs the REAL pipeline (not the scalar convert-around), bit-identically.
The §6a55 padded arm stays ST (env-only experimental; slabbing it at Kp
is mechanical if it ever earns a default).


## 6a59 — the IL per-cell verdict SHIPPED: fused-vs-padded raced at create-time, exec_me lifecycle complete

**The mechanism (mirroring the split A/B at vfft.c:~455 faithfully):**
wisdom entry gains the v7 trailing field `il_me` (0 = unmeasured, K =
fused won, Kp = padded won; v5/v6 files load 0; the reader tolerates
trailing tokens, so forward-compat both ways). At the first-execute
decision on an unmeasured misaligned cell, _il_ab_race runs BOTH
production arms on PRIVATE scratch — alternating order per round, med9,
reps from a ~10 ms budget, **3% hysteresis toward the FUSED incumbent**,
winner roundtrip-gated (failure → K). Losing Kp frees cplan_il. The
verdict stamps te->il_me in-memory and persists with the bundle save.
VFFT_IL_PAD still forces (gates/benches); exec_me is never read
(§6a41/§6a55 cross-context).

**Self-protection proven:** the (1000,12) cell — §6a55's +86% cold-chain
hazard — RACED and verdicted K. The hazard that forced auto-engage
removal is now a measured outcome, not a shipped regression.

**Gate (extended build_tuned/benches/gate_il_pad.c, ALL PASS both builds):**
[verdict] cells (200,20) and (1000,12): A/B ran exactly ONCE (counter
hook), verdict valid, roundtrip e-16, and the second plan on the same
cell REUSED the stamp without re-racing (te HIT). All prior cells hold;
all four big matrices ALL PASS.

**The IL tail-handling arc is now structurally identical to split's:**
built (§6a55) → measured honestly (§6a55/56) → applied where
deterministic (§6a57) → MT (§6a58) → per-cell measured defaults
(§6a59). Padding becomes the IL default exactly where it wins, exactly
how 1D c2c split was built.


## 6a60 — the ND partial-tile port: measured, guard shipped, fft2d verdict inverted

**The ask:** port fft2d's fixed-K tile pattern (S1 compute-and-discard) to
the c2c ND engines' partial-tile lane calls, on suspicion fft2d was ahead.

**The site audit first:** fftnd:311/fft3d:228 = tile-scratch calls at
this_B (slack exists, S1 applicable). fftnd:173 (fused-group axes) runs
IN-PLACE at offsets into live pads — no owned slack, full-C would
overwrite adjacent live lanes: **S3 correct-by-necessity there**, its S1
would require padding the c2c pad allocation itself (a §6a54-analog,
filed). fft2d c2c: no this_B-into-plan site exists.

**The measurement (build_tuned/benches/bench_tile_partial.c, same-process, jit, med9)
inverted the premise:**

| this_B/B | full-B vs hybrid |
|---|---|
| 1..B/2   | +61% .. **+819%** (waste dominates) |
| B-1      | **-11.5% .. -31.8%** (the only win region) |

fftnd's this_B choice was ALREADY RIGHT — unconditional S1 would have
shipped up to 8x regressions at small remainders. The hybrid's straggler
only loses when the remainder is one short of full width.

**What shipped:** the measured guard `run_B = (B - this_B <= 1) ? B :
this_B` at both tile sites (fftnd.h, fft3d.h) — captures the proven edge,
zero risk elsewhere; slack lanes are stale scratch, lane-independent,
discarded at scatter. All four matrices ALL PASS.

**The inverted fft2d finding:** fft2d's r2c tile inner ALWAYS runs the
fixed K=B plan — it eats the small-remainder waste this measurement just
quantified (bounded: one tile per execute, worst ~+4% on the row pass at
this_B<<B shapes). Width-plumbing through _fft2d_r2c_inner (whose
helpers take no lane-count today) is the named follow-up — invasive
enough for its own session, now with the numbers that justify it.


## 6a61 — the featureset parity sweep + the last crash class eliminated

**The sweep (build_tuned/benches/sweep_featureset.c):** dims{1..4} x {c2c, r2c+c2r}
x howmany{1,5} x order{default,natural} x layout{split,z} through the
PUBLIC API, 80 cells, each classified OK / REJECT / FAIL, plus a T=4 MT
pass. Headline: **zero wrong-number cells anywhere** — every gap is a
clean create-reject except one class the sweep caught:

**c2c interleaved at dims>=2 SEGFAULTED** (unwired: NULL im flowed from
vfft_execute's 2D/3D c2c branch straight into the split executors; the
§6a47 z work had been real-side only). FIXED: a convert-around branch at
the top of that dispatch — §6a57's _vfft_z_dein/inter into lazily-
allocated work halves, the existing split engines (incl. _natorder_2d on
the halves, so 2D NATURAL z works too), interleave out. Post-fix sweep:
d=2 def/nat z and d=3 def z all OK at e-16, ST and T=4; d=4 z now
REJECTs cleanly at create instead of crashing. Correct at convert cost;
native ND c2c z is filed. All four big matrices ALL PASS.

**Header contract un-staled:** vfft.h:91 claimed "3D: C2C only" — now
reads "3D: C2C + R2C/C2R (§6a47)".

**The parity map after the fix** (OK = e-16/e-15 roundtrip incl. MT):

| capability | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|
| c2c split default | OK K=1,5 | OK K=1 | OK K=1 | REJECT |
| c2c split natural | OK | OK | REJECT | REJECT |
| c2c z (default+2D natural) | OK | **OK (was crash)** | **OK (was crash)** | REJECT (was crash) |
| r2c/c2r split + z | OK K=1,5 | OK K=1 | OK K=1 | REJECT |
| real natural | REJECT (all dims — order knob is c2c-only) ||||
| howmany>1 | 1D only (2D = deliberate §6a50) ||||

**Filed findings:** (2) dims==4 publicly unexposed while fftnd is rank-
general internally — the vfft_create dispatch stops at 3; (3) real-side
natural order has no public contract at any rank (§6a47b naturalization
is internal chain machinery); (4) howmany>1 beyond 1D. Each is a
contract-extension session, none is a correctness issue.


## 6a62 — dims==4 exposed: the rank-general engines go public

**The finding held exactly:** the engines were rank-4-ready all along
(FFTND_MAX_RANK=4; stride_plan_nd and stride_plan_nd_r2c both accept
rank 4 and built clean at every probed shape) — the public dispatch just
stopped at 3. The exposure was two guards (dims>3 -> dims>4; the order-
gate dims<=3 -> <=4) plus one dims==4 create block mirroring the 3D
contracts verbatim: K==1, order DEFAULT/SCRAMBLED, real = OOP with even
last dim; c2c via the generic fftnd wrap, real via fndr rank 4. h gains
N4; the plane products (OOP memcpy + the §6a61 z convert-around) extend
— so 4D c2c interleaved works on day one through the same fallback.

**Proof (build_tuned/benches/gate_4d.c + the sweep, ALL PASS):** 4D r2c vs the
brute-force naive real-DFT at 20 random natural bins: 3.2e-14 — the
strong check, possible because fndr's output order is defined; c2r
roundtrip 6.5e-16; c2c Parseval 1.9e-15 + roundtrip 6.7e-16. Sweep d=4:
c2c split+z and r2c/c2r split+z all OK at e-16 incl. the T=4 MT pass;
nat/K>1 REJECT per contract. All four big matrices ALL PASS.

**Featureset parity after §6a61+§6a62:** every rank 1..4 now has c2c
split+z and r2c/c2r split+z, all MT, all at machine precision. The
remaining inequalities are exactly two contracts, uniformly absent:
real-side natural order (all ranks) and howmany>1 beyond 1D — filed,
neither a correctness issue.
