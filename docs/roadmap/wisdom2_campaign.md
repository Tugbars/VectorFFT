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

## Wave 2 — minis 🔴 BLOCKED on #5b (c2r boundary handshake, owner)

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

- [ ] 3.1 Migrate fft2d c2c (+@nat2d) / r2c / c2r → `wisdom2_2d.txt`:
      `n=N1xN2`, ordered extents, `q=1` explicit, `k_pad=` named (both
      pad-to-4 vintage and pad-to-8 legal), per-direction records replace the
      shared-version-tag file pair.
- [ ] 3.2 3D binds natively → `wisdom2_3d.txt` (`n=N1xN2xN3`); the legacy
      fft3d grammar never materializes on disk; `n=AxBxCxD` stays reserved.
- [ ] 3.3 Flip _build_2d lookups/banks; dirty-only saves end the
      unconditional per-create rewrites; 2D helper 1D c2c cells remain plain
      c2c records with `from=` provenance (design default, owner informed and
      did not override — unlike trig N±1 rows these are genuinely servable
      c2c cells a 1D caller may share).
- [ ] 3.4 adopt_wisdom.h: explicitly untouched (healthy sidecar).
- [ ] 3.5 RITUAL: Gate B over all (N1,N2) cells incl. @nat2d; freeze-stamp
      the three fft2d files; fleet (bench 2D read modes may keep reading
      frozen copies until ported — read-only is safe); README.

## Wave 4 — stride/spike LAST (the triple-role file)

- [ ] 4.1 Migrate spike scrambled + @nat + @natoop + rfft →
      `wisdom2_stride.txt`: pad verdicts as named `pad_me=`/`il_me=` fields;
      @nat/@natoop as `ord=nat` records with SIGNPOST `ref=` to component
      records (never inlined); duplicates/junk to quarantine;
      spike_wisdom_padded.txt fossil → quarantine as `superseded`.
- [ ] 4.2 ❗ TRIG RE-KEY (owner override): DCT1/DST1 (N±1) and sibling helper
      rows re-keyed under their owning transform; the trig creates' inner
      lookups flip IN THIS WAVE; Gate B covers trig cells explicitly
      (old (N±1,K) c2c path vs new trig-key path → identical plans).
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
