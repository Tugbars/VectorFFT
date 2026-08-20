# wisdom2 campaign — execution checklist

**STATUS: APPROVED DESIGN, wave 0 ready to start (owner go 2026-08-19).**
Design of record: `src/core/wisdom2/README.md`. Decision trail:
`docs/research/wisdom_redesign/ARCHITECTURE_VERDICT.md` (+ SURFACES.md census).
Go through slowly, one gate at a time. No step involves re-timing; every gate
is zero-timing. The README's status ledger is updated at every wave exit —
an out-of-date README is a gate failure.

**The per-wave ritual** (applies to waves 1–4, referenced below as RITUAL):
enumerate the wave's writer fleet from the registry → git tag + snapshot →
migrate (Gate A: lossless accounting + idempotency; "skipped" counts ONLY
non-record lines — comments/banners/blanks; every data row is migrated or
quarantined, no third bucket) → **owner reviews the quarantine** → Gate B:
plan-equivalence over the full consumer matrix (for RE-KEYED records — trig —
the gate resolves old-key-via-old-path vs new-key-via-wisdom2 using the
migrator's key map) → flip readers behind kill switch
`VFFT_WISDOM2_OFF=<family>` → rebuild EVERY consumer exe on the fleet list →
flip writers + delete old writer paths → sentinel-canary gate → freeze-stamp
the old file(s) + baseline checksums → bake window with freeze watch →
update README status ledger → close wave, remove kill switch.
Meanwhile rule: until a family's wave closes, its legacy reader/writer path
stays live and untouched; the only cross-cutting rule from wave 0 on is D9
(no new field/axis ever lands in a legacy file).

---

## Wave 0 — infrastructure (no runtime behavior change)

- [x] 0.1 **DONE 2026-08-19.** README-verification workflow (4 adversarial
      lenses) folded in: illegal flagship example fixed (was N=8192 data
      labeled n=512 with fresh-race wildcards), env-preemption added to the
      lookup order, bluestein removed from the t= vocabulary (route never a
      key), lexical rules + ref= failure semantics + dir= reservation +
      legend format pinned, adopt sidecar scoped out, guard mechanism marked
      OPEN (owner), header evolution-rules block added.
- [x] 0.2 **DONE 2026-08-19.** `wisdom2.h` codec shipped as specified.
- [x] 0.3 **DONE 2026-08-19.** Store API shipped: open/lookup/scan/bank/
      save/update_field/quarantine_append; bank MOVES the record on success.
- [x] 0.4 **DONE (module half) 2026-08-19.** Explicit writable flag at open;
      unset-env "." forces read-only with one loud line. ⏳ The config-field/
      env shape of the guard = OPEN owner decision (README §2.2); wiring
      lands with wave 1's vfft.c integration.
- [x] 0.5 **DONE 2026-08-19.** Env-law table as data in-module (four shapes,
      VFFT_SP_ROUTE row, dead-name blacklist comment). Enforcement lives in
      the create path at wave 1+.
- [x] 0.6 **DONE 2026-08-19.** Field registry with classes; `sp_kv` reserved
      LOCAL.
- [x] 0.7 **DONE 2026-08-19.** .gitattributes eol=lf pin + .gitignore
      explicit negation landed.
- [x] 0.8 **G0 ALL PASS 2026-08-20 — 113 checks** (wisdom2_g0_gate.exe, thin
      driver over module-owned `wisdom2_selftest.h`). The first 55-check
      green was then adversarially reviewed by a 3-lens workflow
      (spec-fidelity with a compiled probe, C/memory safety, grammar edges)
      which found 2 blockers + 8 bugs + gaps — ALL fixed and gated: save
      emits by RESIDENCY (re-route scrubs the old shard; update_field never
      re-routes; cross-shard duplicate keys dedup at load); every write
      checked (a failed emit can no longer rename a truncated tmp over a
      good file); pid-suffixed tmps swept at OPEN only; merge law reordered
      RANK-FIRST (a real race always displaces env/seed; metric refusal
      applies at equal rank; dated beats dateless; measure-less challengers
      refused; unknown src ranks lowest); seeds excluded from BOTH lookup
      tiers; empty sections carry the `-` marker (round-trip corruption
      fixed); wildcard tier prefers q=*-only; dangling ref ⇒ loud MISS in
      lookup; request wildcards refused; growing line reader (no 4096-byte
      split); bare/duplicate tokens ⇒ whole-line opaque carry; lexical law
      enforced at the write API; n=/q= overflow + dangling-'x' rejected;
      zero-byte/headerless file ⇒ poison; @meta round-trips (+ setter);
      quarantine guarded + versioned; zr_kv reclassified LOCAL. Coverage
      added: routing (prime/trig), unknown KEY token carry, minor-version
      accept, CRLF, update_field, re-route migration, 6KB lines. Lint leg
      green. 🔴 TOOLCHAIN TRAP found en route: mingw15.2 at the repo's race
      flags miscompiles `va_start` after a BY-VALUE STRUCT parameter —
      never put an aggregate before `...` (fixed by pass-by-pointer;
      repro'd, minimal case isolated).
- [x] 0.9 **DONE 2026-08-20 — [mig-gate] ALL PASS on a scratch copy of the
      shipped oop_wisdom.txt.** `src/core/wisdom2/wisdom2_migrate.h` (module
      owns ALL logic; `benches/wisdom2_migrate.c` = 12-line thin driver).
      Links the SHIPPED legacy parser verbatim (per-line probe-file parse ⇒
      exact line numbers; legacy silent drops become quarantine entries).
      Full oop kinds 0–5 mapping: kinds 0/1/2 → classic records (`units=cyc`
      honesty); kind-3 → wildcard dual records (`ran=` per K-rule, decoded
      CCOL chains+vars, il_kv carried, cascade il_route would emit the ref=
      signpost); kind-4 → zturn/zsplit records (`metric=joint2` route lines
      vs `fwd1` legacy); kind-5 → per-(transform,placement) real-shard
      records (line ns not slot-attributable ⇒ omitted, refuse-don't-guess).
      TWO-PASS bank (verdicts, then seeds) makes collisions deterministic:
      the K%8≠0 bank-only warts migrate as `src=seed`; a seed whose key is
      owned by a live verdict quarantines as shadowed-by-live-verdict —
      the single-pass version once quarantined a LIVE kind-4 2048 verdict in
      favor of a wart, by file order. Results on the shipped file: 122 lines
      = 94 migrated + 28 quarantined (24 garbage-variant + 4 shadowed) →
      121 records; runs 2 and 3 byte-identical; persistence and quarantine
      counts machine-checked (non-vacuous: the first gate version passed on
      an empty store because the out dir didn't exist — fixed with loud
      probe-I/O fatality + persist-failure accounting).
- [x] 0.10 **Gate B's DATA HALF DONE 2026-08-20, pre-flip.**
      (a) Gate-A field verify: every legacy entry compared field-by-field
      against the saved store through the codecs — 0 mismatches.
      (b) `src/core/wisdom2/wisdom2_oop_reader.h` (PERMANENT module code):
      the production read side — resolves wisdom2 records back into the
      EXACT `vfft_oop_wisdom_entry_t` the existing plan constructors
      consume; one twin per legacy lookup (lookup_k1 / lookup_zsplit /
      lookup_ord / lookup_zr2c), encodings via the SHIPPED codecs
      (cc_chain/cc_vars encode, zr_kv reassembled via vfft_zr2c_kv_set);
      name tables owned here, migrator includes them (forward and inverse
      maps cannot drift).
      (c) **[reader-gate] ALL PASS — 122 cells, 0 mismatches**: for every
      verdict the legacy lookups would serve, the wisdom2 reader produces a
      field-identical entry from the migrated store; every UNSERVABLE row
      (K%8 warts, sub-2048 kind-4, quarantined garbage) correctly MISSES;
      kind-5 kv bit-identical; non-vacuous (0-cells = FAIL).
      Consequence for wave 1: the reader flip in vfft.c is now mechanical —
      swap each legacy lookup call for its vw2_oop_* twin; the constructors
      consume identical entries, so the remaining Gate-B half (resolved-plan
      diff through create) is near-tautological. The migrator/registry are
      linguist-vendored (transitional, die at wave 5); the reader is shipped
      library code.
- [x] 0.11 **DONE 2026-08-20.** `tools/wisdom_migrate/WRITER_REGISTRY.md`
      checked in — agent-authored, every entry verified against source
      (file:line cited): 4 direct writers + 16 front-door bankers + library
      banking sites by family + read-only consumers + per-wave rebuild
      checklists. Corrections vs older lists: zr2c_gate is NOT a writer;
      calibrate_k1's spike banner and zr2c_fd_gate's never-persists comment
      are stale; the bench flag is `--c2rcalib`.
- [ ] 0.12 Gate stderr taps (`_*.log`) repointed out of wisdom dirs.

## Wave 1 — oop family (APPROVED TO START; unblocks Phase C / sp_kv)

- [ ] 1.1 Migrate `oop_wisdom.txt` kinds 0–5 → `wisdom2_oop.txt`:
      kind-0/1/2 with `units=cyc`, `q=bK`; kind-3 as `q=*/ord=*/pl=*`
      wildcard records, `ran=` from the K column (owner K-rule),
      `metric=fwd1`, il_kv carried, `sp_kv` slot ready; kind-4 `metric=joint2`
      with route/t2q/zt_tw/zt_l1, sub-2048 inert law enforced at lookup;
      kind-5 zr_kv split into per-(transform,placement) records — these are
      t=r2c/c2r keys, so they ROUTE to wisdom2_real.txt (created in wave 1,
      extended in wave 2; the key decides the shard, never the wave).
      ⚠ kind-5's CONSUMER flip touches the r2c/c2r create's storage calls
      only (4 sites, zero route logic) — included here so oop_wisdom.txt
      freezes whole; owner confirms alongside #5b, else kind-5 flip defers to
      wave 2 with oop_wisdom.txt in partial freeze (flagged, not preferred).
- [x] 1.2 Quarantine REVIEWED BY OWNER 2026-08-20 — verdict: DISPOSE.
      28 rows = 24 garbage-variant-token (kind-2 MODEB, uninitialized
      variant ints; source found + closed same day: champions bank skips
      bK%8!=0, codec refuses the class) + 4 shadowed-by-live-verdict (K=1
      classic rows at 2048/16384/131072/262144, double-dead: K%8-unservable
      AND cascade-owned keys). Owner: "you can get rid of them, I will run
      most of the IL/ModeB cells again" ⇒ wisdom2_quarantine.txt is NOT
      promoted into generated/ (git history of the frozen legacy file is
      the archive); promotion set = wisdom2_oop.txt + wisdom2_real.txt.
- [x] 1.3 **READ+BANK FLIP LANDED 2026-08-20, behind the kill switch.**
      All five lookup sites flipped to the vw2_oop_* twins (lookup_k1 ×2,
      lookup_zsplit, lookup_zr2c, lookup_ord), guarded by
      `VFFT_WISDOM2_OFF=oop` (reads fall back to the legacy table for the
      bake window; writes go to wisdom2 either way). All four bank sites
      rewired through the ONE family constructor (`vw2_oop_bank_entry` /
      the per-slot zr2c bank — the packed RMW is gone); persistence behind
      the owner's guard: **`config.wisdom_write` field added to
      vfft_config_t** (default 0 = serving mode: banks stay in memory, one
      loud line per process; 1 = measurement mode persists). Colony law
      intact (a "."-defaulted bundle opens the store read-only).
      `_oop_wisdom_put_and_save` DELETED with a tombstone; vfft_wisdom_save
      persists the wisdom2 store (repoint + all-dirty) instead of rewriting
      the frozen file; vfft_wisdom_free closes the store.
      **SMOKE GREEN (dual scratch: legacy + migrated files):**
      sp_ccol_decode_gate 4/4 PASS with legacy-line expectations vs
      wisdom2-served plans (relerr ≤7.5e-16); tangent_frontdoor_gate ALL
      CORRECT on BOTH arms with bit-identical worst-error values
      (wisdom2 reads ≡ kill-switch legacy reads).
- [x] 1.4 **DRIFT SURFACE COLLAPSED 2026-08-20.** All four kind-4 builders
      gone: `_sp_merge_bank` (+ k1_bank_tmp.txt dance) deleted with a
      tombstone; calibrate_zchain's `bank()` + hand-built entry deleted
      (thin driver → `vfft_il_dp_bank_scr_top`, new planner-owned banker);
      width-gate line surgery deleted (logic moved to module-owned
      `src/core/oop/oop_width_gate.h`; injections via `vw2_oop_bank_entry`,
      store byte-snapshot/restored — 6/6 PASS); `_oop_wisdom_put_and_save`
      + vfft_wisdom_save's raw oop rewrite already dead per 1.3.
      **Found by the width gate — kind-4 METRIC LAW fixed:** metric was
      route-inferred (zturn→joint2, zsplit→fwd1), which mislabeled fresh
      dp verdicts (the dp races BOTH routes joint) and let a zturn incumbent
      refuse a fresh zsplit winner under metric identity. Now: measured
      kind-4 = always `joint2` (only the dp banks measured kind-4);
      `fwd1` survives only as migrated vintage (route-inference kept for
      src=migrated → migration stays byte-idempotent, gates re-verified);
      the create-time t2q race banks MEASURE-LESS (ns=0 — its fwd-only
      median is placement luck, and a measure-less row can always be
      replaced by the planner's measured verdict, never the reverse).
      **Wart sunset:** the champions bank site now skips `bK%8 != 0`
      cells (they could never replay — `vfft_oop_plan_from_entry` hard-
      gates K%8; legacy wrote them as the write-only garbage rows the
      migration quarantined, incl. the uninitialized-variants K=1 MODEB
      class the codec refuses).
- [x] 1.5 dp_planner_il emit + dp_planner_split_oop bank → wisdom2 API
      (2026-08-20): `vfft_il_dp_emit_wisdom(vw2_store_t*, ...)` banks via
      `vw2_oop_bank_entry`; `vfft_sp_dp_plan_and_bank` opens/saves/closes
      the store itself (no FILE*, no tmp). Thin-driver law intact;
      calibrate_k1 unchanged as driver (stale spike banner fixed).
      lookup_k1 twin gained exact-beats-wildcard (a fresh concrete race
      verdict outranks the migrated wildcard at the same N — probe-verified;
      migrated wildcards sunset naturally as cells re-race).
      Smoke: calibrate_k1 128 + calibrate_zchain 2048 on the dual scratch —
      legacy oop_wisdom.txt md5-unchanged through BOTH full runs (freeze
      proof), fresh verdicts banked (concrete kind-3 key; kind-4 joint2).
- [x] 1.6 **Gate B SWEEP GREEN 2026-08-20** (dual scratch for hit/replay
      gates; EMPTY scratch for the cold-start race+bank gates — their
      documented shape; a populated store correctly yields NO-RACE rows):
      sp_ccol_decode 4/4 · tangent front door ALL CORRECT on BOTH read arms
      (bit-identical) · k1z_inplace ALL PASS · width gate 6/6 ·
      replay probe PASS · zturn_tcut 63 PASS / 0 FAIL / 17 refused-by-fence
      (spec F2) · zr2c front door ALL CORRECT · natural_front COLD ALL PASS
      (fresh bank → save/reload → bitwise replay through wisdom2) ·
      ilp_front COLD ALL PASS.
      **Known-fail, PRE-EXISTING (not a flip regression): k1scr arm-1 at
      N=1024** — fwd(scr) vs fwd(nat) bitwise DIFF, both individually
      correct (1.78e-15 / 9.99e-16), deterministic across six runs spanning
      {cold, dirty, populated} × {wisdom2 reads, kill-switch legacy reads}.
      Identical behavior in BOTH read worlds = the plan-equivalence law
      holds; the Phase-A identity contract at 1024 broke engine-side after
      the gate shipped (1024 is the owner-locked contested cell). Filed for
      the owner's planned IL/ModeB re-race; 128/256/512 + the 4096 A3
      non-identity arm all PASS.
- [x] 1.7 RITUAL DONE 2026-08-20. Fleet rebuilt (15 front-door + 3 direct
      writers + module gates; traps: sp_ccol_decode_gate + cil_ab include
      vfft.c wholesale → build WITHOUT --vfft; cil_ab only via
      benches/build_cil_ab.py). Sentinel canary PASSED: a future payload
      token (zz_future=canary7 on an untouched record) and a whole
      unknown-key @cell line (zfut=7) survived every store rewrite of the
      sweep — carry-unknown-forward proven through the real fleet.
      oop_wisdom_roundtrip_gate.c DELETED (retired → G0 twin). Final
      reruns: G0 113 / mig-gate / reader-gate ALL PASS; migration output
      REPRODUCED byte-identical from the frozen file.
      **PROMOTED + FROZEN**: generated/wisdom2_oop.txt
      (md5 9992355517a08abd3121fbd632e6f1ec) + wisdom2_real.txt
      (e2388397d06bbc543ba7971ee932e171) from the pristine migration;
      oop_wisdom.txt freeze-stamped as its LAST-EVER write —
      pre-stamp md5 40ea35f3f298bb615e317c680057b1e5 (byte-identical
      through the whole campaign), FROZEN baseline
      cc529724e140bcbaefa5264426771bb5. Post-promotion serving verified on
      a copy: probe + sp_ccol 4/4 + tangent ALL CORRECT on BOTH read arms
      (the stamp is invisible to the legacy reader, as designed).
- [ ] 1.8 CLOSE — **BAKE WINDOW OPEN since 2026-08-20**: reads run on
      wisdom2; `VFFT_WISDOM2_OFF=oop` flips reads back losslessly. Close
      after normal work bakes clean → Phase C (sp_kv) starts in the
      IL–split campaign. (Commit of the promoted files left to the owner,
      who is committing this campaign's work themselves.)

## Wave 2 — minis 🔴 PARKED (owner 2026-08-20: c2r IL session still live
## in the create seams — "skip it and I will tell you when it is done";
## wave 3 runs BEFORE wave 2)

- [ ] 2.1 Bluestein: migrate the 40 orphaned rows → `wisdom2_prime.txt`;
      flip vfft.c sites; prime-N creates now HIT (structurally impossible
      before); Rader stays heuristic-B (future route enters the same
      enumeration per planner charter).
- [ ] 2.2 c2r_path absorbed → `wisdom2_real.txt`; REAL banking hook in the
      c2r create (off-grid cells persist); bench --c2rcalib clobber retired
      (mode ports to wisdom2).
- [ ] 2.3 DELETE `c2r_wisdom.txt` (owner order; git = archive); recalibrate
      its 3 cells fresh at next create.
- [ ] 2.4 DELETE `fftnd_wisdom.h` (dormant, zero callers).
- [ ] 2.5 RITUAL: Gate B over migrated cells; zr2c fd-gate 38/38 rerun;
      freeze-stamp c2r_path.txt + vfft_bluestein_wisdom.txt; fleet
      (zr2c_prewarm — its argless shipped-tree default now writes the new
      store via explicit path; zr2c_fd_gate; bench real modes); README.

## Wave 3 — 2D/3D

- [x] 3.1 MIGRATED 2026-08-20: --2d migrator leg (probe-parse via the three
      shipped loaders; @nat2d = data, other @ = headers). [mig-gate-2d]
      ALL PASS: 41 lines = 7 skipped + 34 migrated + 0 quarantined → 34
      records in wisdom2_2d.txt; byte-idempotent ×3; [reader-gate-2d]
      34/34 field-identical incl. every vars/dif field.
- [x] 3.2 3D BORN IN WISDOM2 2026-08-20: store-first lookup at the dims=3
      create; the legacy creator runs greedy+extract against an in-process
      SCRATCH table (no load, no save, no path — disk contact removed);
      extraction harvested via vw2_3d_bank_entry (measure-less src=race);
      guarded persist. [3d-born-gate] PASS: 16³ create banked → persisted
      → re-served bitwise from a fresh load.
- [x] 3.3 FLIP LANDED 2026-08-20: wisdom2_2d_reader.h (4 twins + 4 bank
      constructors; canonical-key law place=oop / real ord=nat dissolves
      design Q1 — no plumbing, no duplicate calibration); from-entry
      builders extracted in all three legacy headers (pure refactor);
      vfft.c c2c scr/nat lookups + post-bank re-serves + r2c/c2r lookup +
      bank flipped; kill switch VFFT_WISDOM2_OFF=2d; the
      save-even-on-FAILED-create rewrites deleted (persist = guarded,
      success-only); explicit-save 2D/3D lines dropped from
      vfft_wisdom_save (vw2_save covers). [2d-flip-gate]: c2c scr/nat ×4 +
      r2c ×2 + c2r ×2 all arms-bitwise-identical + naive-DFT-anchored;
      freeze held on all three files through every run.
      🔴 GATE-HARNESS LESSON: compare only the transform-DEFINED output
      region — beyond it live padding lanes whose garbage is plan-shaped
      (an r2c 128x128 "failure" was the comparison window covering
      padding, not a real divergence; window fixed to halfcomplex/real
      plane per direction).
- [x] 3.x OWNER-REQUESTED TOUCH-POINT AUDIT (2026-08-20, 18-agent
      sweep+verify): **library CLEAN — zero missed flips, zero live
      writers, zero broken paths in the IL/split machinery** (both
      planners, oop_dp, il2p, zsplit, zturn, every K=1 engine arm read via
      vw2 or behind the kill switch). 14 candidates → 3 real
      stale-consumers, all out-of-front-door tools: (1)
      sp_ccol_decode_gate enumerated cells from the FROZEN file — FIXED
      (re-hosted: vw2_scan enumeration + production-twin resolve, vacuity
      guard); (2) zr2c_prewarm + zr2c_fd_gate dir probes keyed on the
      frozen filename — FIXED (probe wisdom2_oop.txt); (3)
      bench_1d_vs_mkl --oop / k1z-view / --zr2c-fold arms read the frozen
      snapshot as if live — WAVE-4 debt per the registry, ACCEPTED for the
      bake with a 🔴 CAVEAT: after any recalibration lands fresh wisdom2
      verdicts, those bench arms show freeze-time data (labels/cell
      selection/fold permutations) until the wave-4 re-host; do not quote
      their wisdom-view columns against post-freeze banks.
      WRITER SWEEP same day: 8 orphaned legacy writers deleted with
      tombstones (oop line encoder; 2D saves/adds; 3D save/load) — zero
      callers verified, 7/7 consumer classes compile, full regression
      battery green (G0 · mig-gate 122 · mig-gate-2d 34 · sp_ccol ·
      tangent both arms · 2d flip gate ALL PASS).
- [x] 3.y FOLDER CONSOLIDATION (owner directive 2026-08-20: "well ordered
      folder structure" — wisdom code lives under wisdom2/):
      fft2d_c2c_wisdom.h + fft2d_r2c_wisdom.h + fft3d_wisdom.h DELETED
      from the transform folders; ALL contents consolidated into
      **src/core/wisdom2/wisdom2_fftnd.h**, sectioned by LIFETIME TIER
      (PERMANENT: entry structs + plan_from_entry builders + the 3D
      scratch/extract/create · LEGACY: fft2d loaders/lookups/creators —
      die with the =2d kill switch at bake close, then migrator-only).
      Named wisdom2_fftnd.h not fftnd_wisdom.h: transforms/fftnd owns a
      live same-named header (ND module, wave-2 scope) and bare includes
      would ambiguate. Rewired: wisdom2_2d_reader.h, vfft.c (3 includes),
      both fft2d planner headers, bench_1d_vs_mkl. Transform folders are
      now wisdom-free for the shipped families.
- [ ] 3.4 adopt_wisdom.h: explicitly untouched (healthy sidecar).
- [ ] 3.5 RITUAL: Gate B over all (N1,N2) cells incl. @nat2d; freeze-stamp
      the three fft2d files; fleet (bench 2D read modes may keep reading
      frozen copies until ported — read-only is safe); README.

### Wave-3 DESIGN OF RECORD (2026-08-20 — 5-agent recon, load-bearing
### claims independently re-verified: placement-blind `_build_2d` signature,
### save-before-`!tp`-check at vfft.c:3183-3189, lossy README:115 example)

Key mapping: 2D c2c scrambled table → `t=c2c n=N1xN2 q=1 ord=scr place=*`
(migration wildcard — the legacy row genuinely served both placements;
`_build_2d` takes no placement); @nat2d table → same key `ord=nat`.
Real 2D → `t=r2c` / `t=c2r` keyed by which FILE the row lived in (the c2r
file self-stamps the r2c version tag — direction exists only as file
membership; it becomes the transform tag, no dir= token), `ord=*` (the
real-2D branch never reads order). 3D → `t=c2c n=N1xN2xN3`, born in
wisdom2: fresh concrete banks only, MEASURE-LESS (`best_ns` is a dead
always-0.000 field — never encode ns=0).

Payload tokens: `rowplan= rowvars= rowdif= colplan= colvars= coldif= b=
k_pad=` (2D); 3D adds `ax0plan/ax0vars/ax0dif ax1plan/ax1vars/ax1dif` and
reuses row* ; `ablock=` LOCAL, absent when -1/heuristic. Variant names
flat/log3/t1s/buf (the 2D vocabulary has BUF=3 that 1D kind-2 lacks).
`k_pad=` stored VERBATIM never re-derived (three pad conventions coexist);
serve-side validation stays `(k_pad&3)==0 && >=hp1`. ns= metric=fwd1
units=ns on all migrated rows ("per call of the keyed transform" — the c2r
column's backward-measure trap dissolves into the t= split). All 34 data
rows parse clean: expected quarantine count 0 (classes defensive only:
truncated-row, chain-too-long, v2-trailing-tokens).

Helper-cell law: NO ref= from composites to 1D cells ever — composite
chains are verdicts measured in the 2D memory regime (dedicated 2D
planner), not copies; they stay inline. The `_inner_c2c` miss-path banks
SPIKE cells (vfft.c:1363-1368, 1440-1442, saved :3180-3181) — wave-4
family, untouched and live through wave 3 (meanwhile rule; the freeze
watch must ignore spike churn). Kind-5/zr2c coupling: none. Checklist
3.3's "helper cells remain plain c2c records" clause EXECUTES AT WAVE 4
with the spike migration, not here.

Migration: probe-parse through the shipped loaders (c2c loader parses both
tables; the r2c loader runs twice, direction stamped into t= at migrate
time). Accounting: 27 = 3 skipped + 24 migrated (13 scr + 11 nat);
7 = 2 + 5; 7 = 2 + 5 → 34 records, all → wisdom2_2d.txt; wisdom2_3d.txt
created empty by the first 3D bank. No two-pass needed (no seeds; scr/nat
split by ord=). fft3d flips instead of migrating: lookup/bank inside
vfft_fft3d_plan_create_wisdom → vw2 twins; banked-flag save →
vw2_save under config.wisdom_write; explicit-save line :6987 dropped;
NO kill-switch value for 3d (nothing to fall back to); 3D's
ignore-recalibrate behavior preserved (flip ≠ behavior change);
prime-axis 3D cells stay unbanked-by-design.

Flip plan: reader twins in src/core/wisdom2/wisdom2_2d_reader.h
(vw2_2d_c2c_lookup_scr/_nat → exact legacy entry structs;
vw2_2d_r2c_lookup(t); vw2_3d_lookup) + bank constructors
(vw2_2d_c2c_bank_entry with the overwrite=0 nat regime-separation as a
constructor flag; vw2_2d_r2c_bank_entry(t); vw2_3d_bank_entry). Call
sites: lookups vfft.c:1350/:1353/:1410/:1433; banks :1405/:1407/:1470 +
fft3d via :3141. The cal_ns<fb_ns speed gates and natural-decoupling rule
are upstream of banking and stay verbatim. Kill switch: extend
VFFT_WISDOM2_OFF vocabulary with `2d`. Dirty-flag save: DELETE the
unconditional rewrites :3183-3188 (today they run even when the create
FAILED — before the !tp check); persistence = dirty shards + vw2_save
under config.wisdom_write. Fleet = the standing 15 front-door exes (no
direct writers this wave).

Gates: [mig-gate-2d] (accounting + byte-idempotency ×3) and
[reader-gate-2d] (34/34 field-identical incl. every vars/dif field — the
exact lossiness the README:115 prototype had) ride the existing migrator
driver; [2d-flip-gate] = NEW thin driver wisdom2_2d_gate.exe over
module-owned logic (dual-scratch, both kill-switch arms, bit-identical
errors + identical served-plan identity; serving mode writes nothing) —
needed because the bench 2D modes bypass vfft_create and cannot drive
Gate B; [3d-born-gate] = same driver, 3D leg (first create banks
measure-less src=race, second re-serves plan-identical, prime-axis banks
nothing); sentinel canary through the fleet; [bench-2d-continuity]
(bench --2d modes still read their frozen copies); freeze watch ignores
spike churn + _*.log taps.

Freeze list: the three fft2d files (27/7/7 lines) stamp + checksum at
wave close; nothing to freeze for 3D (no file ever existed — the
path_3d_c2c stamp dies at flip); strided_adopt.wis machinery untouched by
declaration; registry line-number drift corrections land with the wave.

OWNER QUESTIONS (recommended defaults in parentheses):
- Q1 placement stamping for FRESH rank≥2 banks — wildcards are
  migration-only, but the 2D plan is placement-blind; stamping the
  creating cfg's placement means the other placement re-pays calibration
  once. (Recommended: stamp concrete, accept the one-time duplicate; a
  "rank≥2 does not key placement" module rule is cleaner but is an
  owner-only grammar change.)
- Q2 README:115 + selftest worked example are lossy (drop vars/dif) and
  stamp place=oop — update both to this design's shape. (Recommended: yes.)
- Q3 bench 2D read modes keep frozen copies until wave 4. (Recommended:
  yes — registry-sanctioned read-only-safe.)
- Q4 b= stays STRUCTURAL this wave (degenerate 8 everywhere on disk).
- Q5 prime-axis 3D cells stay unbanked. Q6 3D keeps ignoring
  cfg->recalibrate at flip (honoring it = separate follow-up). Q7
  quarantine disposal same as wave 1 (expected count 0 anyway).

## Wave 4 — stride/spike LAST (the triple-role file)


### 4.4 BENCH RE-HOSTED 2026-08-20 — the divergence is gone

`bench_1d_vs_mkl` no longer serves verdicts from the frozen files. All four
sites now read the LIVE store, so what the bench measures and labels is
what the front door serves:

- `--oop` arm: `run_oop_cell` takes a `vw2_store_t *` and builds via
  `vw2_oop_lookup_ord` + the shipped `vfft_oop_plan_from_entry` (was a pure
  lookup in the frozen legacy table).
- k1z kind-4 arm and the `--zr2c` cascade arm: `vw2_oop_lookup_zsplit`.
- The main strided c2c arm: the row's factorization is REPLACED by the
  store's verdict for that cell when present (fallback to the parsed row
  when the store does not carry it, so coverage never shrinks).
- 🔴 THE BASENAME CONTRACT IS DELETED. It existed so the bundle would read
  the very file the bench parsed; bundle and bench now agree by
  construction (same store, same directory), not by filename coincidence.

HONEST SCOPE NOTE (registry said "raw spike strtok walk DELETED"): the walk
SURVIVES, demoted to pure ENUMERATION — "which cells exist to visit". It no
longer supplies a single verdict. Enumerating from a frozen file yields
freeze-time cells, the same benign limitation `run_bench.py` has, and the
exporter is the forward path if it should ever enumerate live cells.

SMOKE: N=4096 K=1 prints `z:4x4x4x4x4x4/R1 zturn` (matches the store's
kind-4 record) and N=1024 K=4 prints `64x16/DIT` (matches `chain=64.16
dif=0`), both with ~8e-16 roundtrip error.

### 4.5 REPOINT SURVEY DONE 2026-08-20 — almost nothing to repoint

Consequence of keeping the FROZEN file as the build input (the correction
above): the external consumers need no changes at all.

- `scripts/bootstrap.sh:173-178` regenerates plan_executors.h from
  `generated/spike_wisdom.txt` and sanity-counts its rows — reading a
  frozen, immutable snapshot is exactly right; the count still matches.
- `build_tuned/run_bench.py:38` reads the same file to ENUMERATE cells to
  bench. Read-only and correct; it enumerates freeze-time cells, the same
  documented limitation as bench_1d_vs_mkl's wisdom-view arms, and the
  exporter is the forward path when it should see live cells.
- `build_tuned/build.py:44` only names the path in a comment.

🔴 DEAD TOOLING FOUND: `build_tuned/calibrate.py` orchestrates
`src/dag-fft-compiler/calibrator/calibrate.exe` — that directory does not
exist (removed in an earlier restructure) — and passes the wisdom path via
`VFFT_PROTO_WIS`, which has NO reader anywhere in the tree (it is on
wisdom2.h's dead-env-name list, correctly). The script cannot run. It is
not a freeze hazard, but it is a trap for a future session that tries to
calibrate with it. Recommend deletion (owner's call — it is their
orchestration tooling, and the powercfg/thermal discipline in its header
is worth preserving somewhere if the driver is ever rebuilt).

### 🔴 4.5 PLAN CORRECTED 2026-08-20 — the dune input is the FROZEN FILE,
### not an export (found by reading emit_executor_h.ml, pre-empting the
### byte-oracle)

`emit_executor_h.ml:125-145` dedups on the WHOLE PLAN TUPLE
`(n, k, factors, variants, use_dif_forward)` — NOT on `(N,K)`. So rows the
legacy READER can never serve (intra-file duplicates shadowed by
first-match) are NOT duplicates to the emitter: it emits a separate
specialized executor for each. Verified in the shipped header: cell
(64,64) has three shapes `164_v02 / 88_v02 / 416_v10` from its three
duplicate rows; (64,4096) three; (128,32) and (128,64) two each.

Consequence: exporting from the store and regenerating would SILENTLY DROP
6 specialized executors — the store legitimately holds one record per key,
so the shadowed rows are gone. Executors are matched by plan SHAPE, so a
greedy/inner-cell plan of that shape would lose its specialization
(slower, never wrong).

THEREFORE: at the wave-4 freeze the dune rule keeps depending on the
FROZEN `spike_wisdom.txt` itself — byte-identical `plan_executors.h` by
construction, zero risk, and an immutable file is the right shape for a
build input. The EXPORTER is the FORWARD bridge: when new cells are
calibrated into the store and codegen should see them, export + regenerate
(and then plan_executors.h SHOULD change; the 6 shadowed shapes drop out
as a documented, deliberate cleanup).

EXPORTER SHIPPED + GATED 2026-08-20: `vw2_export_stride` (store -> legacy
v8 via the SHIPPED writer + the SHIPPED twins; rows re-ordered into
ORIGINAL FILE ORDER from the `from=<file>:<line>` provenance, no-provenance
rows last by (N,K)). `[export-gate] ALL PASS`: 336 cells first-match
field-identical across all three tables, byte-reproducible x2. Diff vs the
original: exactly the 8 unservable rows (6 shadowed duplicates + 2 junk
N=0/N=2 cells) + the header count line; the 62 natural entries and every
surviving row keep their ORIGINAL LINE POSITION.

### Wave-4 DESIGN NOTES (2026-08-20, first-hand codec read)

Source of truth: `src/core/planning/wisdom_reader.h` (v8; THREE tables in
one file) + samples read from the shipped spike/rfft files.

KEY/PAYLOAD MAPPING (all `eng=stride`):
- scrambled row `N K nf f.. ns blocked split groups dif v.. exec_me il_me`
  → `t=c2c n=N q=K ord=scr place=ip | eng=stride chain= vars= dif=
  [blocked= bsplit= bgroups=] [pad_me=] [il_me=] | ran=K ns= metric=fwd1
  units=ns` — the blocked triple only when use_blocked!=0; pad_me/il_me
  only when nonzero (absent = unmeasured, the "absent = legacy behavior"
  law; exec_me's wisdom2 name is pad_me per the registry).
- `@nat N K mode nf f.. v.. dif ns` → same key `ord=nat place=ip` +
  `mode=` (name table free/leafip/scr/pcyc/pswap/zcasc/ilp; leafip retired
  but files may carry it — migrate verbatim, never reuse). Chain carried
  VERBATIM for every row EXCEPT the dummy-chain shape (nf==1 &&
  factors[0]==N — the @natoop zcasc placeholder): the dummy is dropped and
  the record carries `ref=cell(t=c2c,n=N,q=1,ord=scr,place=oop)` instead
  (the README flagship; the reader twin reconstructs the dummy when
  filling the legacy struct — deterministic, reader-gate-checkable).
  Mode census on the shipped file: pcyc×34 pswap×18 zcasc×5 ilp×5.
- `@natoop` → same as @nat with `place=oop`.
- rfft rows (same v8 grammar, own file) → `t=r2c n=N q=K ord=scr place=ip`
  — the ROUTER puts them in wisdom2_real.txt (the key decides the shard;
  the old "rfft → wisdom2_stride.txt" line in 4.1 is superseded by the
  router law, exactly the kind-5 precedent).
- spike_wisdom_padded.txt (@version 6 fossil, 25 rows) → quarantined
  `reason=superseded-fossil` at migration, never parsed for records.
- Canonical fresh keys: place=ip (the stride family identity; @natoop
  banks place=oop). K semantics: ran=K verbatim (the batch that ran).
- New registry rows: `dif blocked bsplit bgroups` STRUCTURAL (pad_me/
  il_me/mode already registered).
- TRIG RE-KEY identification (4.2) comes from the trig creates' inner-cell
  geometry (three classes: DCT2/3 → inner r2c cells; DCT4 → inner c2c at
  N/2; DCT1/DST1 → N±1 c2c) — enumerated from vfft.c at the flip step;
  the t= vocabulary already carries dct1-4/dst1-4/dht (wave-0 foresight).

- [ ] 4.1 Migrate spike scrambled + @nat + @natoop + rfft →
      `wisdom2_stride.txt`: pad verdicts as named `pad_me=`/`il_me=` fields;
      @nat/@natoop as `ord=nat` records with SIGNPOST `ref=` to component
      records (never inlined); duplicates/junk to quarantine;
      spike_wisdom_padded.txt fossil → quarantine as `superseded`.
- [x] 4.2 TRIG RE-KEY DONE 2026-08-20 (owner override #8). Trig helper
      cells are now keyed under their OWNING transform at the OUTER size
      (`t=dct1 n=1025 q=4`), not as plain c2c cells. Inner-size derivation
      lives in ONE place (`vw2_stride_trig_inner_n`: DCT1 -> N-1,
      DST1 -> N+1, else N/2) and is used by read and write, and the write
      side REFUSES a key whose derived inner size disagrees with the entry
      being banked (`trig-inner-size-mismatch`).
      🔴 THE COLLISION, CONCRETELY: DCT-I at 1025 and DST-I at 1023 BOTH
      drive an inner c2c of 1024 — and 1024 is one of the most common
      genuine c2c cells. Under the old scheme those three verdicts shared
      one slot; now they are three keys.
      NO MIGRATION IS POSSIBLE for these rows: on disk a helper row and a
      genuine c2c row at the same (N,K) are indistinguishable, so the trig
      cells start cold under the new keys and re-race (small, cheap).
      Under `VFFT_WISDOM2_OFF=stride` the old behavior is exact (inner
      looked up as a plain c2c cell in the legacy table).
      GATE: DCT1@1025, DST1@1023, DCT2@1024 — records land under
      t=dct1/dst1/dct2, and the wisdom2 arm vs the kill-switch arm are
      BITWISE-IDENTICAL on all three (plan equivalence).
- [ ] 4.3 il_me RELOCATION (D6, approved): the fused-vs-padded A/B moves
      into create, banks immediately into the plan's OWN bundle; the
      execute-time stamp is deleted; exec-purity audit goes fully green.
- [ ] 4.4 Exporter (links the frozen legacy writer code VERBATIM) emits
      `spike_wisdom_frozen.txt`; repoint `generated/dune` promote rule +
      `bootstrap.sh`; run_bench.py gains its small wisdom2 branch;
      bench_1d_vs_mkl's raw strtok walk + basename==oop_wisdom.txt contract
      deleted (canonical-bench law: a --mode, never a new harness).
- [ ] 4.5 **Byte-oracle gate**: `plan_executors.h` generated from the frozen
      snapshot must be byte-identical to the one from frozen
      spike_wisdom.txt; bootstrap entry count matches; dune full-tree build.
- [ ] 4.6 Flip the ~12 spike autosave sites + @nat/@natoop banks + r2c/c2r/
      trig inner-cell calibrates; RITUAL (largest fleet); freeze-stamp
      spike_wisdom.txt + rfft_wisdom.txt; seam gate rerun; README.

## Cross-wave standing gates (run at EVERY wave)

- [ ] Freeze watch: frozen-file checksums unchanged across the whole gate
      run and bake window.
- [ ] Sentinel canary through that wave's full writer registry.
- [ ] Seam gate (from wave 1 on): legacy-spike @nat mode-6/7 replay against
      wisdom2-served kind-4/kind-3 components.
- [ ] README status ledger updated (gate item, not a chore).
- [ ] Scratch-wisdir law for every unported writer; promotions field-scoped
      with dated backups + exact git diff.

### WAVE-4 FREEZE DONE 2026-08-20 + CHECKSUM BASELINE

`spike_wisdom.txt` and `rfft_wisdom.txt` took their one-time LAST-EVER
write (the `# FROZEN` stamp, naming the live shards and recording that the
file is STILL a deliberate build input); `spike_wisdom_padded.txt` is
stamped as the superseded @version-6 fossil it always was.

Row count across the stamp: **258 before, 258 after** — `bootstrap.sh`'s
`grep -cE "^[0-9]"` check, the legacy loaders, and the OCaml emitter all
skip `#` lines, so `plan_executors.h` is unaffected by construction.

FROZEN BASELINE (every legacy family, all four waves):

| file | md5 |
|---|---|
| oop_wisdom.txt | cc529724e140bcbaefa5264426771bb5 |
| fft2d_c2c_wisdom.txt | 021829d0476e7a4514ebcfc60e841693 |
| fft2d_r2c_wisdom.txt | 480447c1ed8f26458c1a8287ded61c0d |
| fft2d_c2r_wisdom.txt | 0e2aa3be2f834e4e5e9a91ba4f93d5b6 |
| spike_wisdom.txt | 976a18a7770344935a04b92989f903c6 |
| rfft_wisdom.txt | ebc7b983506c974858cd3193a7a58105 |
| spike_wisdom_padded.txt | 3957a85d4f3b118b1af1a4ad7020548d |

LIVE STORE: wisdom2_oop 27d946c1 · wisdom2_stride f9328cdf ·
wisdom2_real 1d7a8b72 · wisdom2_2d d6b0fbba · wisdom2_3d born on first bank.

## Wave 5 — post-cutover (hygiene + chartered follow-ups, not correctness)

- [ ] 5.1 Delete legacy reader/writer code paths per pool-sunset (legacy
      parsers survive ONLY inside the migrator tool until v1.0, then die
      with old-lib).
- [ ] 5.2 D5 follow-up: flip K=1 classic-OOP rows to replayable (engine-side
      from_entry gate lift) — ends the perpetual champions re-race.
- [ ] 5.3 Retire kill switches; remove transitional wildcards as re-races
      overlay concrete cells.
- [ ] 5.4 Phase C (sp_kv) executes in the IL–split campaign on the new
      schema (chartered after wave 1, listed here for the ledger).
- [ ] 5.5 ONE-FILE COLLAPSE (owner end state): concatenate per-family files,
      bind one path — mechanical by construction; owner picks the moment.
- [ ] 5.6 Throughput table: `(n, q, ord)` keys + strategy `ref=`s to
      component records — its OWN record kind in its OWN file (one concern,
      one file — owner law; the record-kind separation survives the eventual
      one-file collapse). Separate chartered campaign (§5d); lands on this
      schema with no re-keying.

Freeze schedule per legacy file (a file freezes only when its LAST writer
cuts over): oop_wisdom.txt = wave 1 · c2r_path.txt + vfft_bluestein_wisdom.txt
= wave 2 (c2r_wisdom.txt deleted, not frozen) · fft2d_*_wisdom.txt = wave 3 ·
spike_wisdom.txt + rfft_wisdom.txt = wave 4 TOGETHER (shared v8 writer —
rfft cannot freeze earlier) · spike_wisdom_padded.txt = already dead,
stamped in wave 4.

## Repo-root `wisdom/` move — SEQUENCING CORRECTED 2026-08-20

Owner approved the destination. An earlier note here claimed the move was
unblocked once the dune rule stayed on the frozen file. **That was wrong**,
and two facts contradict it:

1. `vfft.c:467-510` resolves the WHOLE bundle from ONE directory `d`:
   `path_c2c = d/spike_wisdom.txt` … and `vw2_open(&W->vw2, d)`. While any
   `VFFT_WISDOM2_OFF=<family>` kill switch exists, the store and its legacy
   fallbacks MUST be co-located.
2. `spike_wisdom.txt` cannot leave `generated/`: it is a dune dep declared
   by bare filename, and the dune workspace root is
   `src/dag-fft-compiler/generator/` — a repo-root `wisdom/` is outside it,
   and dune refuses deps outside its root.

Together: moving the store alone breaks the kill switches; moving both is
impossible while spike is a build dep. **So the move waits for the bakes to
close and the kill switches (with their legacy loads) to be deleted.** At
that point the store moves alone and the frozen files stay in `generated/`
as build input plus archive — which is where a build input belongs anyway.

Order of operations, once bakes are clean:
1. delete each family's `VFFT_WISDOM2_OFF` branch + its legacy load;
2. `git mv` the `wisdom2_*.txt` shards to repo-root `wisdom/`;
3. change the ONE default in `_bundle_paths` (the library already takes the
   directory as a parameter, so this is a one-line default);
4. repoint the bench/gate scratch-dir probes;
5. re-run the sweep; the frozen files never move.

### Dead weight removed 2026-08-20

`build_tuned/calibrate.py` (orchestrated a driver whose directory no longer
exists, via an env nothing reads), 5 `.bak`/`.pre-il` snapshots of wisdom
files (git history is the archive), and a stray `calibrate_nat_grid.log`
tap output. `generated/README.md` now states that the directory holds two
kinds of file with opposite recovery stories, and that a `.txt` must never
be regenerated, hand-edited, or deleted to tidy up.

KEPT (still referenced): `c2r_wisdom.txt` — `bench_1d_vs_mkl.c:3592` reads
it; it dies in wave 2 with the c2r work, per owner decision #10.

## Kill switches RETIRED + five legacy files DELETED (owner go, 2026-08-20)

Owner's argument, accepted: the reader gates already proved old and new
serve IDENTICAL values (122 oop + 34 2D + 338 stride cells, zero
mismatches, plus bitwise-identical front-door output), so the fallback
carried no information the store does not carry.

DELETED: `oop_wisdom.txt`, `rfft_wisdom.txt`, `fft2d_{c2c,r2c,c2r}_wisdom.txt`
(+ `spike_wisdom_padded.txt`, the fossil, deleted earlier the same day).
`generated/` now holds four live shards, `spike_wisdom.txt` (build input),
and the three not-yet-migrated wave-2 files.

Switches and files went TOGETHER — deleting files while the branches
remained would make `VFFT_WISDOM2_OFF` silently serve an EMPTY table, which
is worse than having no switch. Removed from vfft.c: the env parse, the
five legacy loads, the five dead path fields. The env NAME stays RESERVED
and now prints one line saying it is retired and ignored.

### 🔴 DEFECT FOUND IN OUR OWN VERIFICATION (same day)

`vw2_off_stride` was NEVER ASSIGNED — an earlier scripted edit silently did
not match and was not checked. Consequences, stated plainly:

- `VFFT_WISDOM2_OFF=stride` never worked; the stride family had no kill
  switch from the wave-4 flip onward.
- The "stride kill switch reads the stamped spike file" check was VACUOUS
  (it ran the wisdom2 path twice and compared it to itself).
- The 4.2 TRIG GATE was VACUOUS for the same reason — both "arms" took the
  same path, so its bitwise-identical result proved nothing.

REPLACEMENT GATE (real, and stronger than what it replaces): forward then
backward through each trig handle must return the input scaled by ONE
constant, the constant read off the data so no convention is assumed. A
wrong inner plan cannot produce a uniform scale.

    DCT-I  N=257  scale=512.0  uniform to 5.14e-16   PASS
    DST-I  N=255  scale=512.0  uniform to 4.44e-16   PASS
    DCT-II  N=256 scale=512.0  uniform to 6.67e-16   PASS
    DCT-III N=256 scale=512.0  uniform to 1.00e-15   PASS
    DCT-IV  N=256 scale=512.0  uniform to 5.55e-16   PASS
    DHT     N=256 scale=256.0  uniform to 6.67e-16   PASS

The SCALE VALUES are the real result: 2N for the DCT/DST family, N for the
Hartley — and for the two collision cells, 2(N-1) at DCT-I 257 and 2(N+1)
at DST-I 255. Those constants independently pin the inner-size derivation
(`vw2_stride_trig_inner_n`) the whole re-key depends on; a sign error there
could not produce them.

LESSON (banked): a scripted source edit MUST assert its match count. An
unchecked `replace()` that silently no-ops produces a feature that looks
present (the struct field, the branches, the env name) and is never
exercised — and any gate built on it passes vacuously.

### 4.4 AMENDED — k1z enumeration moved to the store (2026-08-20)

Deleting `oop_wisdom.txt` exposed the half of 4.4 that had been deferred:
the bench SERVED from the store but still ENUMERATED kind-4 cells by
walking that file, so with the file gone the k1z arm benched 0 cells.

Fixed properly rather than by restoring the file: a store-driven pass
(`vw2_scan` over kind-4 records, each resolved through the production twin
`vw2_oop_lookup_zsplit`) now enumerates the cells to visit, honouring the
`target_N` filter. The file walk keeps its K=1 lines SKIPPED — that guard
must stay, it is what prevents the historical out-of-bounds parse of
kind-lines-as-factors.

Strictly better than the file enumeration it replaces: a re-raced or newly
banked cell is visible to the bench with no file to re-parse. VERIFIED:
single cell N=4096 restored, and a full pass enumerates all 8 cascade cells
(2048 · 4096 · 8192 · 16384 · 32768 · 65536 · 131072 · 262144), 91 cells
benched total, roundtrip errors 7e-16 … 1.8e-15.

LESSON: re-hosting a consumer's SERVING path without its ENUMERATION path
leaves a dependency that only fails when the old file is finally removed —
and it fails silently, as "0 cells benched".
