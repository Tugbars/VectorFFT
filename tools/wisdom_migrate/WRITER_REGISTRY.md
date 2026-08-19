# WRITER_REGISTRY.md — wisdom writer fleet (strip-cycle rebuild manifest)

**Campaign item 0.11** (`docs/roadmap/wisdom2_campaign.md`). This registry is the source of truth for the per-wave fleet rebuild — not memory, not the PROPOSALS list. Sources: SURFACES.md §3 + risk-2, the banking/benches census agents, and the code. **Every entry below verified against source 2026-08-19** (file:line cited). A binary absent from this registry that can write wisdom is a registry bug and a gate failure.

**Laws this registry encodes**
- Latent-writer law: EVERY binary that calls `vfft_create` is a wisdom writer — calibrate-on-miss persists into the bundle dir at any rigor, for the default bundle (`$VFFT_WISDOM_DIR` else `"."`, `vfft.c:480`) and caller-owned bundles alike (`vfft.c:508-536`, `include/vfft.h:168-176`). "Read-only on hits" is a property of the data, never of the binary.
- Rebuild rule: every wave flips banking sites inside `vfft.c` → **every front-door binary (§2) rebuilds at EVERY wave**, plus that wave's direct writers (§1) and external repoints (§4).
- Sentinel-canary gate (G4 graft): each wave banks a future-minor-token record through every binary on that wave's checklist (§6) and asserts token survival.
- Scratch-wisdir law stands for every unported writer until its wave closes.

---

## 1 · Direct writers (write wisdom files without the front door)

| binary | source | writes | how | disposition / wave |
|---|---|---|---|---|
| `calibrate_k1.exe` | `build_tuned/benches/calibrate_k1.c` | `<argv1>/oop_wisdom.txt` (kind-3 + kind-4) via `<argv1>/k1_bank_tmp.txt` | thin driver → `vfft_sp_dp_plan_and_bank` (`calibrate_k1.c:71`, `dp_planner_split_oop.h:603`): fresh lines to `k1_bank_tmp.txt`, `_sp_merge_bank` rewrite of the main file. ⚠ its banner (`:22`) still claims spike_wisdom.txt is written — STALE post-B2.2; spike is decoupled (`dp_planner_split_oop.h:397`) | **Wave 1**: ports — stays the driver (thin-driver law, 1.5), planner headers emit via wisdom2 API |
| `calibrate_zchain.exe` | `build_tuned/benches/calibrate_zchain.c` | `<argv1>/oop_wisdom.txt` (kind-4) | own `bank()`: shipped reader load (`:61`), dedup (N, K, kind==ZSPLIT) (`:63-71`), hand-built entry in `main` (`:160-183`, incl. zt_tw/zt_l1 — the silent-width-drop site, `:174-181`), whole-file `fopen(path,"w")` rewrite (`:73-77`). No sub-2048 filter | **Wave 1**: `bank()` DELETED (1.4); entry construction collapses into the one `from_plan` |
| `zturn_wisdom_width_gate.exe` | `build_tuned/benches/zturn_wisdom_width_gate.c` | `<--wisdir>/oop_wisdom.txt` (kind-4 line surgery) + `_wisdom_width_gate.log` tap | bypasses shipped reader/writer: raw sscanf line find (`:68-86`), hand-snprintf rewrite `N 1 4 0 <cc_chain> <ns> [tail]` (`:89-98`), whole-file rewrite (`:94`), restore at exit (`:211-214`). The one surviving hand-encoder | **Wave 1**: surgery replaced by the field-update API (1.4) |
| `bench_1d_vs_mkl.exe` (`--c2rcalib`) | `build_tuned/benches/bench_1d_vs_mkl.c` | `../../src/dag-fft-compiler/generator/generated/c2r_path.txt` (SHIPPED TREE, hardcoded `:3589`) | flag `--c2rcalib` (`:3179-3182`); `fopen(path,"w")` clobber `:3595`, fixed grid N∈{256,512,1024}×K∈{8..256} — off-grid cells LOST. The ONLY writer of a file the library depends on (`vfft.c:470`) | **Wave 2**: retired — mode ports to wisdom2, c2r create gains the real banking hook (2.2) |

Direct writer of the NEW store (registry member for G4 completeness):

| binary | source | writes | how | disposition |
|---|---|---|---|---|
| `wisdom2_g0_gate.exe` | `build_tuned/benches/wisdom2_g0_gate.c` | wisdom2 scratch store (`argv[1]` else `./w2_g0_scratch`) | thin driver over `src/core/wisdom2/wisdom2_selftest.h` (`:8-12`) | permanent (G0 gate); absorbs `oop_wisdom_roundtrip_gate`'s role at wave 1 (1.7) |

---

## 2 · Front-door bankers (latent writers via `vfft_create` calibrate-on-miss)

Every binary here can write ANY bundle file its creates touch (spike, oop, rfft, 2D×3, 3D, bluestein). "primary files" = what its cells deliberately exercise.

| binary | source | dir resolution | primary files / modes | disposition / wave |
|---|---|---|---|---|
| `bench_1d_vs_mkl.exe` | `benches/bench_1d_vs_mkl.c` | positional argv = spike FILE (default hardcoded shipped `spike_wisdom.txt`, `:3278-3279`); k1z bundle derived from wpath dirname, basename MUST be `oop_wisdom.txt` (`:447-478`); sibling files from hardcoded shipped-tree paths | banking modes: K=1 kind-dispatch cells (any wpath run with K=1 lines, `:545ff`), `--k1zip` (`:3188`), `--k1nat` (sub-2048 race+bank), `--kzb` (rigor=MEASURE champions, `:712ff,:856`), `--zr2c` cascade arm (`:2626ff`) → oop + spike into wpath dir | rebuilds EVERY wave (canonical bench, `--mode` law); raw spike strtok walk + basename contract DELETED wave 4 (4.4) |
| `zr2c_prewarm.exe` | `benches/zr2c_prewarm.c` | argv[1] else probes `../src/.../generated` then `../../src/.../generated` (`:59-62`) — **argless default writes the SHIPPED tree by design** | oop kind-5 (zr_kv route race per slot, autosave per bank); `vfft_wisdom_load` `:76` | **Wave 1** (kind-5 storage flip, 1.1; records route to `wisdom2_real.txt`); wave 2 keeps shipped-tree write via explicit path (2.5) |
| `zr2c_fd_gate.exe` | `benches/zr2c_fd_gate.c` | same 3-candidate probe (`:225-243`); refuses empty wisdom | oop kind-5 banks-on-miss (route −1 deletes `VFFT_ZR2C_ROUTE`, `:61-63`; banks incl. N=510). ⚠ header `:13-15` "never auto-persisted" is STALE | **Wave 1** flip + **Wave 2** rerun (fd-gate 38/38, 2.5) |
| `sp_ccol_decode_gate.exe` | `benches/sp_ccol_decode_gate.c` | `_putenv VFFT_WISDOM_DIR=argv[1]` before first create (`:105-106`) + shipped-reader enumeration of `<argv1>/oop_wisdom.txt` (`:107-110`); `#include "vfft.c"` wholesale | oop kind-3 (expects HITS; a miss banks into argv[1]) | **Wave 1**: re-hosted on wisdom2 (1.7) |
| `tangent_frontdoor_gate.exe` | `benches/tangent_frontdoor_gate.c` | no argv, no cfg.wisdom → library default bundle (`$VFFT_WISDOM_DIR` else `"."`) | oop kind-3 (128/256/512/1024); wrong-cwd run banks a colony | **Wave 1**: re-hosted on wisdom2 (1.7) |
| `cil_ab.exe` | `benches/cil_ab.c` | `$VFFT_WISDOM_DIR`, else SETS it to `cil_ab_wis` (`:204-205`); ABORTS unless `<dir>/spike_wisdom.txt` opens (`:207-217` — has destroyed shipped rows before) | spike (in-place IL natural c2c creates write back); `#include "vfft.c"` wholesale | rebuild every wave; spike write-back ports **wave 4** |
| `vfft_natural_front_gate.exe` | `benches/vfft_natural_front_gate.c` | `--wisdir` (`:275`), `vfft_wisdom_load` (`:283`, reload `:351`) | DELIBERATE race+bank: oop kind-4 + spike @nat (mode 6) + @natoop; tap `_nat_front_gate.log` (`:51`) | **Wave 1** (kind-4 leg) + **Wave 4** (@nat/@natoop leg); rebuilt both |
| `vfft_ilp_front_gate.exe` | `benches/vfft_ilp_front_gate.c` | `--wisdir` (`:131`), `vfft_wisdom_load` (`:137`) | DELIBERATE @nat ILP race+bank; tap `_ilp_gate.log` (`:37`) | **Wave 1** (kind-3 components) + **Wave 4** (@nat leg) |
| `k1z_inplace_gate.exe` | `benches/k1z_inplace_gate.c` | `--wisdir` (`:104`), `vfft_wisdom_load` (`:114`) | oop kind-4 in-place cells (banks on miss); sets/clears VFFT_TCUT* + FORCE_ZROUTE (`:61,putenv`); tap `_k1z_ip_gate.log` (`:35`) | **Wave 1** rebuild |
| `mt_c2c_gate.exe` | `benches/mt_c2c_gate.c` | `--wisdir` (`:128`), `vfft_wisdom_load` (`:130`) | spike in-place split cells (banks on scratch miss) | rebuild every wave; G5 gate |
| `vfft_k1scr_gate.exe` | `benches/vfft_k1scr_gate.c` | `--wisdir` (`:124`), `vfft_wisdom_load` (`:126`) | spike + oop (scrambled vs natural identity route; banks on miss) | rebuild every wave |
| `vfft_tcbatch_gate.exe` | `benches/vfft_tcbatch_gate.c` | `--wisdir` (`:362`), `vfft_wisdom_load` (`:365`) | spike (TRANSFORM_CONTIGUOUS batch cells; banks on miss) | rebuild every wave |
| `zturn_tcut_gate.exe` | `benches/zturn_tcut_gate.c` | `--wisdir` (`:255`), `vfft_wisdom_load` (`:268`) | oop kind-4 (miss-banking is its named hazard, `:39` scratch mandate); putenv VFFT_TCUT* (`:88`); tap `_tcut_gate_stderr.log` (`:58`) | **Wave 1** rebuild |
| `zturn_tcut_ab.exe` | `benches/zturn_tcut_ab.c` | `--wisdir` + optional `--wisdir2` (`:163-181`) | oop kind-4 (banks on miss, both banks); putenv VFFT_TCUT (`:119`); stderr tap (`:132`) | **Wave 1** rebuild |
| `zturn_wisdom_replay_probe.exe` | `benches/zturn_wisdom_replay_probe.c` | `--wisdir` (`:105`), `vfft_wisdom_load` (`:113`) | oop kind-4 ("read-only" claim `:16` true only on HITS); putenv (`:59`); tap `_replay_probe_stderr.log` (`:32`) | **Wave 1** rebuild |
| `oop_wisdom_roundtrip_gate.exe` | `benches/oop_wisdom_roundtrip_gate.c` | argv[1] else `"."` | `_roundtrip_oop_wisdom.txt` scratch only (`:117`, removed `:167`); 216-byte entry-size canary (`:41-44`) | **Wave 1**: RETIRED → G0 twin in wisdom2 selftest (1.7) |

---

## 3 · Library-side writer entry points (by family)

Every exe linking `vfft.c` compiles these in; a stale build of ANY §2 binary carries stale versions of ALL of them.

### oop_wisdom.txt (freezes wave 1)

| entry point | location | trigger | wave-1 fate |
|---|---|---|---|
| `vfft_oop_wisdom_write_entry` (the ONE line encoder) | `src/core/oop/oop_wisdom.h` | called by all merge sites below | replaced by wisdom2 codec |
| `_oop_wisdom_put_and_save` (dedup key N,K,kind-CLASS) | `vfft.c:508` | call sites: `:2038` (kind-5 `_bank_zr2c`, RMW of zr_kv), `:4239` (kind-4 create race, hand-built entry `:4208-4238`), `:4766`+`:4772` (kinds 0/1/2 champions) | DELETED (1.4) |
| `vfft_wisdom_save` raw oop rewrite loop | `vfft.c:6911-6917` | explicit API call | DELETED (1.4) |
| `_sp_merge_bank` (dedup N,kind; kind-3 any-K; sub-2048 kind-4 filter) + `k1_bank_tmp.txt` dance | `dp_planner_split_oop.h:563`, rewrite `:591` | `vfft_sp_dp_plan_and_bank` (`:603`), i.e. calibrate_k1 | DELETED (1.4) |
| `vfft_il_dp_emit_wisdom` (kind-3 + kind-4 constructor) | `dp_planner_il.h:1249` | planner pipeline via `:1356` | ports to wisdom2 API (1.5) |

### spike_wisdom.txt + rfft_wisdom.txt (shared v8 writer; freeze wave 4 TOGETHER)

| entry point | location | trigger |
|---|---|---|
| `vfft_proto_wisdom_save` (the ONE v8 writer fn, clobber `fopen(path,"w")`) | `wisdom_reader.h:301` | all sites below |
| @nat bank `_bank_nat_1d` / @natoop `_bank_natoop_1d` | saves `vfft.c:2358` / `:2383` | natural-order races at create |
| 2D inner-cell calibrate | `vfft.c:3146` | every 2D create (unconditional) |
| padded-batch calibrate / pad exec_me A/B stamp | `vfft.c:3390` / `:3431` | create |
| in-place c2c calibrate (regime-guarded scr cell) | `vfft.c:3552` | create |
| r2c inner / c2r inner / trig inner cells | `vfft.c:4909` / `:5009` / `:5072` | create |
| rfft table (same fn, own file; bK≤64 gate) | `vfft.c:4925`, `:6910` | r2c rfft-path create |
| `_pad_stride_c2c` ALLOC-time bank | `vfft.c:6702` | `vfft_alloc_batch` |
| `vfft_wisdom_save` | `vfft.c:6909` | explicit API |
| ⚠ `il_me` EXECUTE-time in-memory stamp (persists only via a later save; exec-purity outlier) | `vfft.c:5550` | first execute of misaligned-K IL plan — relocates to create at **wave 4** (4.3, D6 approved) |

### 2D / 3D (freeze wave 3)

| entry point | location | trigger | note |
|---|---|---|---|
| `vfft_fft2d_c2c_wisdom_save` | `vfft.c:3149`, `:6920` | UNCONDITIONAL after every 2D c2c create; bank rules at `vfft.c:1396-1398` | dirty-flag save lands wave 3 (3.3) |
| `vfft_fft2d_r2c_wisdom_save` (ONE fn, TWO files r2c/c2r) | `vfft.c:3151`+`:3153`, `:6921-6922` | unconditional after 2D create; bank `vfft.c:1455-1461` | |
| `vfft_fft3d_wisdom_save` | `vfft.c:3108`, `:6923` | dims=3 create, banked-flag guarded | file has NO disk instance; 3D born in wisdom2 (3.2) |

### minis (freeze/delete wave 2)

| entry point | location | trigger | note |
|---|---|---|---|
| `bluestein_wisdom_save` (faster-only merge — unique) | `vfft.c:3530`, `:6924`; `bluestein_wisdom.h:103` | prime-cell create calibrate-on-miss | filename mismatch: bundle `bluestein_wisdom.txt` vs shipped `vfft_bluestein_wisdom.txt` (D1) |
| c2r_path.txt | **NO library writer** (reader only: `vfft.c:470`, `c2r_dispatch.h`) | — | wave 2 adds the banking hook (2.2, D2) |
| `fftnd_wis_append` (append-only, atomic-per-line) | `fftnd_wisdom.h:158` | DEAD — zero callers | DELETED wave 2 (2.4, D4) |
| `vfft_adopt_record` (tmp+rename — the only atomic writer) | `adopt_wisdom.h:82` | 2D/ND r2c adoption A/B, own env `$VFFT_ADOPT_WISDOM_DIR` | explicitly UNTOUCHED (3.4); ⚠ its plain `rename()` must not be copied verbatim on Windows |

---

## 4 · Read-only consumers (need rebuild/repoint; cannot strip, CAN go silently stale)

| consumer | family | coupling | wave action |
|---|---|---|---|
| `emit_executor_h.ml` (OCaml, dune BUILD INPUT → `plan_executors.h`) | spike | positional walk, HARD-FAILS on truncation/unknown variant | **Wave 4**: repointed at frozen snapshot / exporter; byte-oracle gate (4.4-4.5) |
| `bootstrap.sh` | spike | `grep -cE '^[0-9]'` entry count | **Wave 4** repoint; G3 count-identical |
| `run_bench.py` | spike | cell enumeration | **Wave 4**: gains wisdom2 branch |
| `bench_1d_vs_mkl` raw spike strtok walk + K=1 kind dispatch | spike/oop | `:3828-3925` (read side) | **Wave 4**: DELETED (4.4) |
| `bench_1d_vs_mkl --oop` | oop | `$VFFT_OOP_WIS` else shipped path, pure lookup (`:3648-3651`) | wave 1 rebuild |
| `bench_1d_vs_mkl` 2D read modes (`--2d`/`--r2c2d`/`--2dc2r`) | 2D | hardcoded shipped-tree paths (`:3384,:3419,:3452`), read-only | may keep reading frozen copies until ported (3.5) |
| `bench_1d_vs_mkl` c2r reads | c2r | `c2r_wisdom.txt` orphan (`:3578`) + `c2r_path.txt` (`:3612`) | wave 2: c2r_wisdom.txt DELETED (owner order, 2.3) |
| `bench_1d_vs_mkl` bluestein read | prime | `$VFFT_PROTO_BLUE_WIS` (`:4046`) | wave 2 |
| the nine C parsers (wisdom_reader.h, oop_wisdom.h, fft2d_c2c/r2c, fft3d, bluestein, c2r_dispatch, adopt, fftnd) | all | compiled into every front-door exe | replaced per wave; legacy parsers survive only inside the migrator until v1.0 |

**Out of scope — verified wisdom-free (no wisdom I/O, no `vfft_create`; the redesign must not break them, none need fleet rebuild):** `cpu_l1_probe`, `il2p_tangent_gate`, `il_dp_cand_census`, `il_dp_overflow_gate`, `il_inplace_probe`, `r32exp_reorder_gate`, `smallre_mkl_probe`, `smallre_pairs_probe`, `smallre_split_probe`, `tangent_gate`, `zr2c_gate`, `zturn_dit_diffprobe`, `zturn_dit_gate`, `zturn_dit_pipe_gate`, `zturn_dit_race`, `zturn_dit_stageprobe`, `zturn_inplace_probe`, `zturn_natmap_probe`, `zturn_natord_falsifier`, `zturn_natord_gate`, `zturn_natscatter_probe`, `zturn_stfn_gate`, `zturn_tile_census` (23 files; `cil_ab_arms/*.c` are arm sources, no main).

---

## 5 · Side-effect writers into wisdom dirs (item 0.12 scope)

| binary | file dropped in wisdir | location |
|---|---|---|
| `k1z_inplace_gate` | `_k1z_ip_gate.log` | `:35` freopen |
| `vfft_ilp_front_gate` | `_ilp_gate.log` | `:37` |
| `vfft_natural_front_gate` | `_nat_front_gate.log` | `:51` |
| `zturn_wisdom_width_gate` | `_wisdom_width_gate.log` | `:50` region |
| `zturn_wisdom_replay_probe` | `_replay_probe_stderr.log` | `:32` |
| `zturn_tcut_gate` | `_tcut_gate_stderr.log` | `:58` |
| `zturn_tcut_ab` | stderr tap | `:132` |
| `calibrate_k1` (via `_sp_merge_bank`) | `k1_bank_tmp.txt` residue (never deleted) | `dp_planner_split_oop.h:638` |
| `oop_wisdom_roundtrip_gate` | `_roundtrip_oop_wisdom.txt` (removed on exit) | `:117,:167` |

All taps repoint out of wisdom dirs at 0.12; a frozen dir's checksum watch must not see them.

---

## 6 · Per-wave rebuild checklists (RITUAL step "rebuild EVERY consumer exe")

**Standing rule**: the FULL front-door fleet rebuilds at every wave because `vfft.c` banking sites change every wave. Front-door fleet =
`bench_1d_vs_mkl.exe` · `zr2c_prewarm.exe` · `zr2c_fd_gate.exe` · `sp_ccol_decode_gate.exe` · `tangent_frontdoor_gate.exe` · `cil_ab.exe` · `vfft_natural_front_gate.exe` · `vfft_ilp_front_gate.exe` · `k1z_inplace_gate.exe` · `mt_c2c_gate.exe` · `vfft_k1scr_gate.exe` · `vfft_tcbatch_gate.exe` · `zturn_tcut_gate.exe` · `zturn_tcut_ab.exe` · `zturn_wisdom_replay_probe.exe` (15 exes).

### Wave 1 — oop (freeze `oop_wisdom.txt`)
- [ ] Front-door fleet (all 15 above)
- [ ] `calibrate_k1.exe` (ported driver; fix stale `:22` banner)
- [ ] `calibrate_zchain.exe` (`bank()` deleted → wisdom2)
- [ ] `zturn_wisdom_width_gate.exe` (surgery → field-update API)
- [ ] `oop_wisdom_roundtrip_gate.exe` → RETIRE (G0 twin in `wisdom2_g0_gate.exe`)
- [ ] `wisdom2_g0_gate.exe` (rerun G0)
- [ ] Sentinel canary through every box above; freeze-stamp + checksum baseline `oop_wisdom.txt`

### Wave 2 — minis (freeze `c2r_path.txt`, `vfft_bluestein_wisdom.txt`; DELETE `c2r_wisdom.txt`) — 🔴 blocked on #5b c2r handshake
- [ ] Front-door fleet (all 15)
- [ ] `bench_1d_vs_mkl.exe` — `--c2rcalib` mode ported (shipped-tree clobber retired); c2r_wisdom.txt read path removed
- [ ] `zr2c_prewarm.exe` (argless shipped-tree default → explicit path to new store)
- [ ] `zr2c_fd_gate.exe` + rerun 38/38
- [ ] fftnd writer deleted (library, no exe)
- [ ] Sentinel canary; freeze-stamps + checksums

### Wave 3 — 2D/3D (freeze `fft2d_{c2c,r2c,c2r}_wisdom.txt`)
- [ ] Front-door fleet (all 15) — 2D/3D writers are library-side only
- [ ] `bench_1d_vs_mkl.exe` 2D read modes: may keep reading frozen copies (read-only safe) or port
- [ ] `adopt_wisdom.h` verified untouched
- [ ] Sentinel canary; freeze-stamps + checksums

### Wave 4 — stride/spike + rfft (freeze `spike_wisdom.txt` + `rfft_wisdom.txt` together; stamp `spike_wisdom_padded.txt` fossil)
- [ ] Front-door fleet (all 15) — largest blast radius: ~12 spike autosave sites + @nat/@natoop banks + r2c/c2r/trig inner cells + il_me relocation (D6) + trig re-key flip all in `vfft.c`
- [ ] `bench_1d_vs_mkl.exe` — raw spike strtok walk + `basename==oop_wisdom.txt` contract DELETED; wisdom2 `--mode` path
- [ ] `cil_ab.exe` — spike write-back on wisdom2; empty-dir refusal retargeted
- [ ] Exporter tool (links frozen legacy writer VERBATIM) → `spike_wisdom_frozen.txt`
- [ ] Repoints: `generated/dune` promote rule · `bootstrap.sh` · `run_bench.py` — byte-oracle gate on `plan_executors.h` (4.5)
- [ ] Sentinel canary; freeze-stamps + checksums; seam gate rerun

---

## 7 · Registry corrections and open items (verified against source)

| item | status |
|---|---|
| `zr2c_gate.exe` named in PROPOSALS_FULL.md's fleet list | **NOT a writer** — verified wisdom-free (no fopen/wisdom_load/putenv/front door in `zr2c_gate.c`); registry omits it from all checklists |
| `calibrate_k1.c:22` banner "oop_wisdom.txt AND spike_wisdom.txt written" | STALE post-B2.2 — writes oop only (`dp_planner_split_oop.h:397` spike decoupled) |
| `zr2c_fd_gate.c:13-15` "never auto-persisted / read-safe" | STALE — banks kind-5 on miss (`:55-57,63`) |
| banking-census flag name "--c2r --calibrate" | WRONG — real flag is single token `--c2rcalib` (`bench_1d_vs_mkl.c:3179`) |
| gitignored research trees (docs/research/*, scratch scripts) | UNSWEPT for additional oop/2D parsers — one-time sweep required before wave-1 G4 (PROPOSALS open question) |
| `build_tuned/test/` | does not exist (census-confirmed); benches/ is the whole fleet surface |
