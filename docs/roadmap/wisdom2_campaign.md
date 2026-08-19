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
- [ ] 0.9 Migrator skeleton linking the FROZEN legacy parsers verbatim;
      row-conservation accounting (migrated + quarantined + skipped ==
      source, machine-checked); idempotency harness; quarantine emitter.
- [ ] 0.10 Plan-equivalence gate harness: both stacks in one binary; diff by
      executor registry id + args (chains, variants, t2q, kv, widths, routes,
      pad geometry), never raw fn pointers across binaries.
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
- [ ] 1.2 Quarantine: N-garbage MODEB variant rows (~24), duplicates
      (first-match copy migrates), junk rows — owner reviews before flip.
- [ ] 1.3 Flip readers: lookup_ord path (vfft.c:4745 region), lookup_k1's
      four consumer sites, kind-4 replay, kind-5 route resolution;
      `q=*` rule serves the kind-3 fan-out.
- [ ] 1.4 Collapse the drift surface: ONE `from_plan` constructor replaces
      the four kind-4 builders; ONE dedup replaces the three; delete
      `_sp_merge_bank` (+ k1_bank_tmp.txt dance), calibrate_zchain `bank()`,
      width-gate sscanf/snprintf surgery (→ field-update API),
      `_oop_wisdom_put_and_save`, vfft_wisdom_save's raw oop rewrite.
- [ ] 1.5 dp_planner_il emit + dp_planner_split_oop bank → wisdom2 API
      (thin-driver law intact; calibrate_k1 unchanged as driver).
- [ ] 1.6 **Gate B**: full kind-3 consumer matrix (4 consumers × layouts ×
      placements × 3 orders), kind-4 four caller configs × both routes,
      kind-5 four slots, kinds 0/1/2 lookup_ord classes — zero diffs.
- [ ] 1.7 RITUAL remainder: fleet rebuild (registry list), sentinel canary,
      freeze-stamp oop_wisdom.txt (last-ever write) + checksum baseline,
      bake, existing gates rerun (sp_ccol_decode_gate, tangent_frontdoor_gate
      re-hosted on wisdom2; oop_wisdom_roundtrip_gate retired→G0 twin),
      README ledger.
- [ ] 1.8 CLOSE → Phase C (sp_kv) may start in the IL–split campaign.

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
