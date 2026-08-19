# VectorFFT Wisdom System (vw2)

**STATUS: DESIGN OF RECORD — approved decisions, no code yet.**
Owner-approved 2026-08-19 (decision trail: `docs/research/wisdom_redesign/ARCHITECTURE_VERDICT.md`;
system census: `docs/research/wisdom_redesign/SURFACES.md`). Where this README
and code disagree after implementation begins, the code wins and this README
must be corrected in the same change — an out-of-date README is a gate failure.

**READ THIS FIRST — the prime rule of this folder.** This README is the ONE
complete declaration of how VectorFFT's wisdom system works. If you are an AI
session or a new contributor: do not touch any wisdom file, reader, writer, or
banking site until you have read this whole file. The previous wisdom system
became a mess precisely because sessions read partial information and imposed
their own conventions; every extension you make must be derived from the rules
below, and when the structure genuinely has no answer, the gap-fill is an
OWNER decision presented as options — never a silent default of yours.

---

## 1 · What wisdom is

Wisdom is the library's store of **measured verdicts**: for a given problem,
which plan won a real race on this machine. It is never a heuristic; a banked
record reads back as a VERDICT and is replayed as truth. The lifecycle law:

> **Planning races and banks. Create resolves wisdom into function pointers
> and arguments on the handle. Execute reads nothing.**

There is no exception. (The old system had one — `il_me` stamped at first
execute — it is removed by design: that race runs at create and banks
immediately, into the plan's own bundle.)

## 2 · The store

### 2.1 Files

```
src/dag-fft-compiler/generator/generated/
  wisdom2_oop.txt      wave 1 — c2c OOP verdicts (old kinds 0–5 re-expressed)
  wisdom2_prime.txt    wave 2 — Bluestein (+ future Rader) verdicts
  wisdom2_real.txt     wave 2 — r2c/c2r route + factorization verdicts
  wisdom2_2d.txt       wave 3 — 2D composite verdicts
  wisdom2_3d.txt       wave 3 — 3D composite verdicts (born here; no legacy file ever existed)
  wisdom2_stride.txt   wave 4 — stride/spike scrambled + natural + rfft cells
  wisdom2_quarantine.txt   migration rejects, verbatim + reason; owner-reviewed; never deleted
```

**Transitional shape.** The owner's end state is ONE wisdom file. Per-family
files exist only to keep cutover blast radii small; because every file shares
one grammar and one module, the eventual collapse is mechanical (concatenate
records, bind one path). Therefore: **family identity lives in the record's
KEY, never in the filename.** No file may grow file-specific semantics.

**Frozen legacy files.** Every pre-vw2 wisdom file (`spike_wisdom.txt`,
`oop_wisdom.txt`, `rfft_wisdom.txt`, `c2r_path.txt`, `fft2d_*_wisdom.txt`,
`vfft_bluestein_wisdom.txt`, `spike_wisdom_padded.txt`) is FROZEN at its
wave's cutover: one final `#`-comment header stamp ("FROZEN <date> — never
edit, never add fields; new axes land in the wisdom2 store; see this README")
is written as the file's last-ever write, then its checksum is baselined and
watched by the gates. A frozen-file checksum change means a stale writer
escaped the fleet rebuild — that is a gate failure. Exception by owner order:
`c2r_wisdom.txt` is DELETED, not frozen (3 rows, unverifiable vintage,
recalibrated fresh; git history is the archive).

**Tracking / EOL contract.** New files end `.txt` directly in `generated/`
(the `.gitignore:89` negation covers only that exact pattern — never a
subfolder). `.gitattributes` pins `wisdom2_*.txt text eol=lf`; writers open
`"wb"` and emit `\n`; readers tolerate `\r\n`. This kills the Git-Bash-sed
CRLF trap mechanically.

### 2.2 Directory resolution and the write guard

- One directory concept: `VFFT_WISDOM_DIR`, else `"."` — REUSED from the old
  system, frozen and live files side by side.
- **The library default is READ-ONLY wisdom** (owner directive). Serving
  mode (default): hits are served; misses race in memory for process
  coherence; NOTHING is written to disk. Measurement mode (explicit opt-in
  guard — config field and/or env, both, config wins): calibrate-on-miss
  persists. Calibrators, benches, gates, and prewarm tools set the guard;
  an application can never accidentally bank. Banking into an unset-env
  default directory is refused with one loud stderr line (the wrong-cwd
  wisdom-colony trap is dead).

## 3 · Record grammar

Line-oriented text. `#` = comment. First line is the version header and it is
**actually checked**: bad magic or unsupported major ⇒ the file is refused,
one loud stderr line, and banking/saving to that path is POISONED (a save can
never clobber a file we could not read). Never silently empty.

```
@vw2 1.0
@legend <writer-emitted digest of the key semantic rules — see §3.4>
@meta host=i9-14900KF isa=avx2 l1d=49152

@cell t=c2c n=512 q=* ord=* place=oop | eng=split_oop sp_route=ccol chain=8.32.32 vars=t1s.flat.log3 | ran=4 ns=19289.3 metric=fwd1 units=ns arms=2 src=race bin=calibrate_k1@ce40f78d date=2026-08-18
@cell t=c2c n=4096 q=1 ord=nat place=ip | mode=zcasc ref=cell(t=c2c,n=4096,q=1,ord=scr,place=oop) | ran=1 ns=8891.0 metric=fwd1 units=ns arms=2 src=race ...
@cell t=c2c n=64x64 q=1 ord=nat place=oop | rowplan=... colplan=... b=8 k_pad=36 | ran=1 ns=7485.2 metric=fwd1 units=ns src=migrated from=fft2d_c2c_wisdom.txt:19
@quarantined reason=garbage-variant-token from=oop_wisdom.txt:45 raw="4096 1 2 3 32 16 8 690 -1435841587 32761 81219.0"
```

### 3.1 KEY — the problem (before the first `|`)

MKL-descriptor-shaped: the key states what the caller asked for, never how it
was served.

| token | meaning |
|---|---|
| `t=` | the mathematical transform: `c2c r2c c2r dct1..dct4 dst1..dst4 dht bluestein` … Never fused with rank (no `fft2d_c2c`); never a route. |
| `n=` | the domain shape. **Rank lives here**: `n=1024` (1D), `n=64x64` (2D), `n=32x32x32` (3D), `n=AxBxCxD` reserved (4D/ND). Extents are ORDERED (anisotropic law: `64x32 ≠ 32x64`). Real transforms key the REAL length. |
| `q=` | quantity REQUESTED (howmany). The caller's number, never the executed width. `q=*` = axis-agnostic record, legal ONLY on migrated records (`from=` required); new banks stamp concrete values. Exact `q` beats `*` at lookup, so wildcards sunset naturally. |
| `ord=` | `nat` \| `scr` — explicit in EVERY record. Order is a key, never a ranking axis; lookups never cross order classes. |
| `place=` | `ip` \| `oop`. |
| `dir=` | `fwd` \| `bwd` \| absent where the verdict is direction-agnostic (defined per transform in the spec table, never inferred). |

**Layout (split/interleaved) is NEVER a key token.** It is a strategy
property — an OUTPUT of planning — and appears only in the payload (§5d
throughput charter). An unknown token in the KEY section makes the record
invisible to lookup and opaque-carried on resave — refuse-don't-guess applied
to keys, which is how future key axes (`isa=`, `nthreads=`) land as additive
minor versions that stale binaries can neither serve, strip, nor collapse.

### 3.2 PAYLOAD — the verdict (between the pipes)

Open vocabulary, registered in the module's field table (§4.3). Typical:
`eng=` (engine/strategy: mono, il2p, il3p, zcasc, zsplit, bailey, split_oop,
classic, stride, zr2c, bluestein, rader…), `route=`, `chain=` (dot-joined
factors), `vars=` (dot-joined variant names), knobs (`t2q= kv= sp_kv= zt_tw=
zt_l1= zr_kv= b= k_pad= path= mode=`), pad verdicts as NAMED fields (`pad_me=`,
`il_me=` — the old `exec_me` dual meaning is retired), and `ref=` (§3.3).
`sp_kv` is reserved NOW: Phase C of the IL–split campaign lands it here, never
in a frozen file (rule D9 — a new axis is added to the old format zero times).

### 3.3 Signpost rule (`ref=`)

A verdict that says "strategy X won" does NOT copy X's recipe; it carries a
`ref=` pointing at the component record that owns the recipe (e.g. the
natural-order verdict referencing the cascade record). One recipe, one place:
when the component re-races to a better recipe, every referencing verdict is
automatically current. Copies are forbidden — a copy is stale the moment its
source improves. This is the same mechanism the future throughput table uses
(strategy verdicts referencing component wisdom).

### 3.4 MEASURE + PROVENANCE (after the second pipe)

| token | meaning |
|---|---|
| `ran=` | the EXECUTED batch geometry of the run that produced the number (owner K-rule 2026-08-19: the batch count of the run that produced the banked time — e.g. `q=1` served by the split engine ⇒ `ran=4` lane batch; IL single transform ⇒ `ran=1`). Never conflated with `q`. |
| `ns=` + `metric=` + `units=` | the number, with EXPLICIT identity: `metric=fwd1` (forward-only per call) \| `joint2` (joint fwd+bwd); `units=ns` \| `cyc` (the old kind-0/1/2 rdtsc rows keep `cyc` — never converted, never compared across metric/units; the module's compare helper refuses mismatches). Informational rows have NO `ns=` token (absent ≠ 0.0). |
| `arms=` | how many arms the race had. `arms=1` + `src=env:<VAR>` marks an env-shaped one-armed verdict; a later real race outranks it at merge. |
| `src=` | `race` \| `env:<VAR>` \| `migrated` \| `seed` (proposes candidates for a future race, never served as a verdict). |
| `bin= date= host= l1d= from=` | writer binary + vintage, date, host stamps (the zt_l1 refuse-on-mismatch precedent), and migration lineage (`from=<file>:<line>` — every migrated verdict auditable to its frozen source forever). |

### 3.5 Header legend (owner directive)

The `@legend` block is a compact digest of the key semantic rules — the
signpost rule, the q-vs-ran rule, the metric-identity rule, the layout-never-
in-key rule, the wildcard rule — so the file explains itself to anyone who
opens it. **It is emitted by the writer on every save, generated from the
module's own rule tables**, so it cannot drift from the code (the old spike
banner advertising modes 1/3/4/5 while the data carried 6/7 is the rot class
this kills).

### 3.6 Forward compatibility

- Unknown tokens are carried VERBATIM through load→bank→save (per-record
  extras). Unknown records likewise. The stale-binary strip cycle — which
  fired three times in the old system — is structurally unrepresentable.
- Absent token = legacy behavior; no sentinel to forget. Retired names are
  never reused (blacklist in the module).
- Minor version bumps = additive tokens only. Major bump = reader refusal.

## 4 · The module — one definition point

`src/core/wisdom2/wisdom2.h` (+ its TU) is the ONLY parser and the ONLY
writer of this grammar, everywhere: library, calibrators, benches, gates.
A lint gate greps the tree for `@cell` emitters outside this folder.

### 4.1 Lookup resolution order (encoded once, here)

exact key → migration wildcard (`q=*`, then `ord=*`/`place=*`) → caller falls
to env override → default spine. **Neighbor cells are never served as
verdicts** — a separate seed iterator exposes them as race PROPOSALS only
(never-heuristic law: serving q=8's plan for q=6 would be a heuristic wearing
a verdict's clothes).

### 4.2 Banking, merge, atomicity

- One dedup: upsert on the full key tuple (replaces the old system's three
  divergent dedup keys). Merge rank: `race` > `env:*` > `seed`/info; among
  equals, newer date wins. Cross-metric replacement is refused with a log.
- Saves are dirty-only (no more unconditional rewrites), merge-on-save
  (reload disk, upsert this process's delta — concurrent sessions banking
  different cells both survive), then atomic replace: write `.tmp`, fsync,
  `MoveFileEx(MOVEFILE_REPLACE_EXISTING)` on Windows / `rename()` on POSIX.
  (Do NOT copy adopt_wisdom.h's bare `rename()` — it fails over an existing
  target on Windows CRTs.) Stale `.tmp` swept on load. Tables are dynamic;
  capacity exhaustion is loud (the old 1024-row silent drop is retired).
- One constructor per verdict shape (`from_plan`) replaces the old system's
  four independent kind-4 entry builders — the zt_tw silent-drop bug class
  becomes unrepresentable.
- Field-scoped promotion is a supported API (replacing hand line-surgery on
  wisdom files).

### 4.3 Field registry and portability classes

Every payload field is registered with a class:
**STRUCTURAL** (route, chain — ports across machines) ·
**LOCAL** (placement-luck: `t2q`, `kv`, `il_kv`, `sp_kv`, tile widths — on a
host/L1 mismatch, THAT FIELD degrades to re-race while the structural rest of
the record still serves) · **INFO** (never decision-load-bearing).
This is the plan-sharing charter and the zt_l1 fence, mechanized. Adding a
field = one registry line + the producer stamp + the consumer read.

## 5 · Env law (one table in the module)

| shape | contract | members |
|---|---|---|
| force, never bank | env wins for this process; nothing persists while set | `VFFT_ZR2C_ROUTE`, `VFFT_IL_PAD`, `VFFT_SP_ROUTE` (new), **`VFFT_FORCE_ZROUTE`, `VFFT_NO_ZTURN`** (owner 2026-08-19: demoted to pure debugging switches — zturn-dev artifacts; they retire with legacy zsplit) |
| force + bank, stamped | the banked plan really ran the forced value; record carries `arms=1 src=env:<VAR>`; real races outrank it | `VFFT_TCUT` (tile width — "never banked as untiled when it was not") |
| suppress banked field | env set (any value) masks that field at lookup | `VFFT_TCUT` (masks `zt_tw` replay) |
| wisdom beats env | the ONE inversion, documented at the field | banked `il_kv` over `VFFT_NO_ILBLK` |

Kill switches (`VFFT_NO_IL2P`, `VFFT_NO_T2B`, `VFFT_NO_NAT_*`, `VFFT_NO_TCMT`)
are availability gates outside this table; none bank under the switch. Env is
read at CREATE time only (exec-purity invariant). Dead names never reused:
`VFFT_PROTO_WIS`, `VFFT_PROTO_PAD_WIS`, `VFFT_WISDOM`, `VFFT_WARM`,
`VFFT_NO_K1`, `VFFT_PROTO_WISDOM_OVERWRITE`.

## 6 · Migration (one-shot, lossless, zero re-timing)

The migrator links the FROZEN legacy parsers verbatim (no old grammar is ever
re-implemented) and is table-driven per family. Non-negotiables:

- Row conservation, machine-checked: `migrated + quarantined + skipped ==
  source rows`, per file. Idempotent (re-run ⇒ byte-identical output).
- No re-timing anywhere — banked numbers are hours of racing on a thermally
  hostile machine; they are carried as data with `from=` lineage.
- Metric honesty: old kind-0/1/2 rows stamped `units=cyc`; kind-3 `metric=
  fwd1`; kind-4 `metric=joint2` (the ~2× fwd-vs-joint gap becomes explicit).
- Owner-decided special cases: trig helper rows (DCT1/DST1 banking c2c cells
  at N±1) are RE-KEYED under their owning transform, with the trig creates'
  lookups flipped in the same wave (collision hazard with genuine odd-N c2c
  cells); the 40 orphaned Bluestein rows are revived into `wisdom2_prime.txt`
  (the runtime filename mismatch dies with the old binding); `c2r_wisdom.txt`
  is deleted and its 3 cells recalibrated; 2D helper 1D cells stay honest c2c
  records with `from=` provenance; junk (N=0/N=2 rows, ~24 garbage-variant
  rows, shadowed duplicates — keeping the copy first-match served) goes to
  quarantine with reasons, reviewed by the owner per wave, never thrown away,
  never salvage-guessed.

## 7 · Gates (all zero-timing)

1. **Plan-equivalence** — the acceptance gate for every wave: for every
   banked cell, across its FULL consumer matrix (e.g. old kind-3's four
   consumers × layouts × placements × orders), resolve the plan via the old
   path and via vw2 and diff the resolved plans (executor registry IDs +
   args, chains, variants, strides, pad geometry). Zero mismatches or the
   wave stops; fixes go into the migrator table, never into frozen data.
2. **Sentinel canary** — per wave: bank a record carrying an unknown
   future-minor token through EVERY binary in that wave's checked-in writer
   registry; assert the token survives all of them (the proven-3× strip
   cycle, turned into a mechanical gate).
3. **Freeze watch** — frozen-file checksums verified before/after every gate
   run and during bake windows; any change = an escaped stale writer.
4. **Seam gate** (after wave 1): legacy spike @nat mode-6/7 replay resolving
   against vw2-served component records — the one mixed-resolution state the
   waves create.
5. **Header/lint** — `@legend` regenerated-equals-committed; no `@cell`
   emitter outside this folder; migrator idempotency; atomic-replace
   crash-injection; export byte-oracle at wave 4 (`plan_executors.h` from the
   frozen spike snapshot must be byte-identical — the OCaml emitter's
   hard-fail strictness is the oracle).

## 8 · Cutover waves and status

| wave | scope | status |
|---|---|---|
| 0 | module + migrator + gates + this README + gitattributes/gitignore lines | PENDING (approved to start) |
| 1 | oop family (kinds 0–5 → wisdom2_oop.txt); kills the 4-constructor/3-dedup drift surface; `sp_kv` reserved; unblocks Phase C | PENDING (owner: "starts now") |
| 2 | minis: bluestein revival, c2r_path absorption + real banking hook, c2r_wisdom deletion, fftnd deletion | BLOCKED on the c2r boundary handshake with the parallel session (owner to arrange) |
| 3 | 2D + 3D | — |
| 4 | stride/spike LAST (dune build input): frozen snapshot + exporter linking legacy writers verbatim; repoint dune/bootstrap/run_bench | — |
| later | collapse per-family files into ONE file (owner end state); throughput table (`(n, q, ord)` keys + strategy `ref=`s) | chartered |

Each wave: enumerate writer fleet → snapshot/tag → migrate → gates → flip
readers (kill switch `VFFT_WISDOM2_OFF=<family>` during bake) → rebuild every
consumer → flip writers + delete old writer paths → bake with freeze watch →
update THIS README → close.

## 9 · For future sessions (the meta-rule)

The old wisdom system accreted ten grammars, nine parsers, four constructors
for one record type, and three dedup policies because each extension was
locally reasonable and globally foreign. The countermeasures are structural
(one module, carry-unknown, writer-emitted legend) — but the first line of
defense is you: **derive every change from this README and the module's rule
tables; state the derived rule before implementing; when the structure has no
answer, the owner decides.** Repeated owner pushback on one point means a
system law is being stated — stop defending and re-read.
