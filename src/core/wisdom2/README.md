# VectorFFT Wisdom System

*(Status: design of record; implementation in progress — see
`docs/roadmap/wisdom2_campaign.md` for the transition plan. That plan is the
only place transition details live; this file describes how the system works.
Until a family's cutover closes, that family's legacy reader/writer path
remains the live system; the one rule in force everywhere from day zero is:
no new field or axis is ever added to a legacy wisdom file.)*

**Read this file before touching anything wisdom-related.** It is the one
complete declaration of the system. Every change is derived from the rules
below; if the rules genuinely have no answer, the gap is an owner decision —
never a convention you invent.

---

## 1 · What wisdom is

Wisdom is the library's store of **measured verdicts**: for a given problem,
which plan won a real race on this machine. It is never a heuristic — a
banked record reads back as a VERDICT and is replayed as truth.

The lifecycle law, with no exceptions:

> **Planning races and banks. Create resolves wisdom into function pointers
> and arguments on the handle. Execute reads nothing.**

## 2 · The store

### 2.1 Files

```text
src/dag-fft-compiler/generator/generated/
  wisdom2_oop.txt         1D c2c out-of-place verdicts
  wisdom2_stride.txt      1D c2c in-place verdicts + trig (dct/dst/dht) verdicts
  wisdom2_real.txt        1D r2c / c2r verdicts (routes + factorizations)
  wisdom2_prime.txt       Bluestein / Rader engine verdicts
  wisdom2_2d.txt          2D composite verdicts
  wisdom2_3d.txt          3D (and higher-rank) composite verdicts
  wisdom2_quarantine.txt  records rejected with a stated reason; kept, never deleted
```

All files share one grammar and one module.

**Read model:** lookup always consults the UNION of every `wisdom2_*.txt`
present in the wisdom directory — the per-family split is a write-side
sharding choice only. **Write model:** the module owns one key→file routing
table (by transform, rank of `n=`, and placement; the prime shard
additionally routes on the engine field — sharding may peek at payload,
semantics never do). **Family identity lives in the record's key, never in
the filename**; the shards can be collapsed into one file at any time
without a format change.

Any other `*wisdom*.txt` in that folder is a frozen legacy store: read-only
history. Never edit one, never add fields to one; new axes land here.

**Out of scope by decision:** `strided_adopt.wis` (the adoption sidecar under
`$VFFT_ADOPT_WISDOM_DIR`, with its own grammar and module,
`adopt_wisdom.h`) stays live and untouched — not migrated, not frozen.
`k1_bank_tmp.txt` and `_*.log` files found near wisdom dirs are tool
droppings, not wisdom — never migration inputs.

### 2.2 Directory and the write guard

- One directory holds the store: `$VFFT_WISDOM_DIR`, else the current
  directory.
- **The library default is read-only wisdom.** In serving mode (default),
  hits are served, and a miss races in memory for process coherence but
  writes nothing to disk. In measurement mode — an explicit opt-in guard —
  calibrate-on-miss persists. Calibrators, benches, and gates set the guard;
  an application can never accidentally bank. Banking with the env unset is
  refused with one loud stderr line.
- ⏳ OPEN (owner decision pending): the guard's concrete shape — config
  field, env variable, or both, and which wins. Until pinned, the module
  exposes an explicit writable flag at open; tools pass it deliberately.

### 2.3 Text contract

Files are LF-only (pinned by `.gitattributes`; writers open `"wb"` and emit
`\n`; readers tolerate `\r\n`). New wisdom filenames must end `.txt`
directly in `generated/` — the `.gitignore` negation covers exactly that
pattern, never a subfolder.

## 3 · Record grammar

Line-oriented text. `#` = comment. The first line is the version header and
it is **checked**: bad magic or an unsupported major version means the file
is refused — one loud stderr line, and banking to that path is disabled so a
save can never clobber a file the reader could not understand. A wisdom file
is never silently empty.

**Lexical rules:** tokens are whitespace-separated `key=value` pairs; values
are bare (no whitespace, no quotes, no escapes). Sections are separated by
`" | "` (pipe with one space each side); pipes never appear in values. The
single exception: `raw=` in `@quarantined` records is always the LAST token
and its value runs to end-of-line (escape-free, so any legacy line survives
verbatim).

```text
@vw2 1.0
@legend <one line per rule — writer-emitted, §3.5>
@meta host=i9-14900KF isa=avx2 l1d=49152

@cell t=c2c n=8192 q=* ord=* place=* | eng=split_oop sp_route=ccol chain=8.32.32 vars=t1s.flat.log3 | ran=4 ns=19289.3 metric=fwd1 units=ns src=migrated from=oop_wisdom.txt:119
@cell t=c2c n=4096 q=1 ord=nat place=ip | mode=zcasc ref=cell(t=c2c,n=4096,q=1,ord=scr,place=oop) | ran=1 ns=8891.0 metric=fwd1 units=ns arms=2 src=race bin=vfft_create@ce40f78d date=2026-08-19
@cell t=c2c n=64x64 q=1 ord=nat place=oop | rowplan=4.16 colplan=2.16.2 b=8 | ran=1 ns=7485.2 metric=fwd1 units=ns src=migrated from=fft2d_c2c_wisdom.txt:19
@quarantined reason=garbage-variant-token from=oop_wisdom.txt:45 raw=4096 1 2 3 32 16 8 690 -1435841587 32761 81219.0
```

(The first example is a migrated record — wildcards and `from=` belong
together. The second is a fresh bank — concrete axes, race provenance. A
fresh bank carrying a wildcard is illegal and the writer refuses it.)

### 3.1 KEY — the problem (before the first `|`)

The key states what the caller asked for, never how it was served
(MKL-descriptor-shaped).

| token | meaning |
|---|---|
| `t=` | the mathematical transform: `c2c r2c c2r dct1..dct4 dst1..dst4 dht`. Never a route, never an algorithm (Bluestein/Rader are engines in the payload; a prime-N request is still `t=c2c`), never fused with rank (no `fft2d_*` tags). |
| `n=` | the domain shape. **Rank lives here**: `n=1024` (1D), `n=64x64` (2D), `n=32x32x32` (3D), `n=AxBxCxD` (4D/ND). Extents are ORDERED (`64x32 ≠ 32x64`). Real transforms key the REAL length. |
| `q=` | the REQUESTED quantity (howmany) — the caller's number, never the executed width. |
| `ord=` | `nat` \| `scr` — explicit in EVERY record. Order is a key, never a ranking axis; lookups never cross order classes. |
| `place=` | `ip` \| `oop`. |
| `dir=` | reserved. No transform keys direction today (every verdict serves both directions from one plan; what a number measured is stated by `metric=`, §3.4). If a future transform needs per-direction verdicts, `dir=fwd`/`dir=bwd` becomes a key token for that transform via the module's rule table. |

**Wildcards:** `q=*`, `ord=*`, `place=*` mark an axis-agnostic record. They
are legal ONLY on migrated records (`from=` required) — an explicit statement
that the legacy verdict genuinely served every value of that axis. New banks
stamp concrete values; the writer refuses a wildcard without `from=`. An
exact-key hit always beats a wildcard record, so wildcards retire naturally
as cells re-race.

**Layout (split vs interleaved) is never a key token.** It is a strategy
property — an *output* of planning — and appears only in the payload.
An unknown token in the KEY section makes the record invisible to lookup and
is carried opaquely on resave: future key axes (`isa=`, `nthreads=`) land as
additive minor versions that older binaries can neither serve, strip, nor
collapse.

### 3.2 PAYLOAD — the verdict (between the pipes)

Open vocabulary, registered in the module's field table (§4.3). Typical
fields: `eng=` (winning engine/strategy: mono, il2p, il3p, zcasc, zsplit,
bailey, split_oop, classic, stride, zr2c, bluestein, rader …), `route=`,
`chain=` (dot-joined factors), `vars=` (dot-joined variants), per-engine
knobs (`t2q= kv= sp_kv= zt_tw= zt_l1= zr_kv= b= k_pad= path= mode= m=`), and
the pad verdicts as named fields (`pad_me=`, `il_me=`).

### 3.3 The signpost rule (`ref=`)

A verdict that says "strategy X won" does **not** copy X's recipe — it
carries a `ref=` pointing at the component record that owns the recipe. One
recipe, one place: when the component re-races to a better recipe, every
referencing verdict is automatically current. Copies are forbidden — a copy
is stale the moment its source improves.

Syntax: `ref=cell(<complete key tuple, comma-joined>)` — always the full key,
never partial. Resolution scope: the union of all loaded files. A dangling
ref (target deleted, quarantined, or not yet present) makes the referencing
verdict a MISS — one loud stderr line, then the normal miss path (re-race in
memory; re-bank in measurement mode). Never a hard error, never a silent
default.

### 3.4 MEASURE + PROVENANCE (after the second pipe)

| token | meaning |
|---|---|
| `ran=` | the EXECUTED batch geometry of the run that produced the number — e.g. `q=1` served by the split engine runs a 4-lane batch ⇒ `ran=4`; an interleaved single transform ⇒ `ran=1`. Never conflated with `q`. |
| `ns=` `metric=` `units=` | the number with EXPLICIT identity: `metric=fwd1` (forward-only per call) \| `joint2` (joint fwd+bwd); `units=ns` \| `cyc`. Numbers with different metric/units are never compared or converted — the module's compare helper refuses. Informational records carry no `ns=` token (absent ≠ 0.0). |
| `arms=` | how many arms the race had. `arms=1` + `src=env:<VAR>` marks an env-shaped one-armed verdict; a later real race outranks it at merge. |
| `src=` | `race` \| `env:<VAR>` \| `migrated` \| `seed` (a seed proposes candidates for a future race and is never served as a verdict). `migrated` serves like `race` (the number WAS raced); a fresh same-metric race outranks it. |
| `bin= date= host= l1d= from=` | writer binary + vintage, date (migrated records keep the original race date when known), host stamps, and lineage (`from=<file>:<line>` for records carried in from legacy stores). |

### 3.5 The header: legend + evolution rules

The writer emits the header block on every save, generated from the module's
own rule tables, so it cannot drift from the code. It contains:

- `@legend` lines — ONE line per rule, in the module's rule-table order,
  each a fixed string constant (signpost, q-vs-ran, metric identity,
  layout-never-in-key, wildcards). The gate compares the writer's
  regeneration byte-for-byte against the committed file.
- The evolution rules: the reserved-field list (`sp_kv` first), the rule
  "new axes are added here as additive minor versions, never to frozen
  files", and the pointer `see src/core/wisdom2/README.md`.

### 3.6 Evolution rules

- Unknown tokens and unknown records are carried VERBATIM through
  load→bank→save. No code path parses-and-truncates; a stale binary cannot
  strip a newer field.
- Absent token = legacy behavior. No sentinels. Retired names are never
  reused (blacklist in the module).
- Minor version = additive tokens only. Major version = reader refusal.
- New fields and new axes are added HERE, never to a frozen legacy file.

## 4 · The module — one definition point

`src/core/wisdom2/wisdom2.h` is the ONLY parser and the ONLY writer of this
grammar, everywhere: library, calibrators, benches, gates. A lint gate keeps
it that way (no `@cell` emitter outside this folder).

### 4.1 Resolution order

Two layers, encoded once:

0. **Force-shaped env overrides preempt everything** (per the §5 table):
   a force env decides its route/field for this process before wisdom is
   consulted — env beats wisdom, the standing law. Suppress-shaped envs mask
   their single field inside the steps below. The one documented inversion:
   banked `il_kv` beats `VFFT_NO_ILBLK`.
1. exact key →
2. migration wildcard (`q=*`, then `ord=*`/`place=*`) →
3. MISS: the caller falls to the default spine (serving mode) or races and
   banks (measurement mode).

**Neighbor cells are never served as verdicts** — a separate seed iterator
exposes them as race PROPOSALS only.

### 4.2 Banking and saving

- ONE dedup: upsert on the full key tuple. Merge rank: `race`/`migrated` >
  `env:*` > `seed`; among equal rank, newer date wins; cross-metric
  replacement refused.
- Saves are dirty-only and merge-on-save (the on-disk file is re-read and
  this process's delta upserted, so concurrent sessions banking different
  cells both survive), then atomic: write `.tmp`, flush,
  `MoveFileEx(MOVEFILE_REPLACE_EXISTING)` on Windows / `rename()` on POSIX.
  Stale `.tmp` files are swept on load. Tables are dynamic; exhaustion is
  loud.
- One constructor per verdict shape builds records from plans; hand-formatted
  lines do not exist. Field-scoped promotion is a supported API call —
  never line surgery on a wisdom file.

### 4.3 Field registry and portability

Every payload field is registered with a class:

- **STRUCTURAL** — ports across machines (routes, chains).
- **LOCAL** — placement-luck, machine-tied (`t2q`, `kv`, `il_kv`, `sp_kv`,
  tile widths): on a host/L1 mismatch that FIELD degrades to re-race while
  the structural rest of the record still serves.
- **INFO** — never decision-load-bearing.

Adding a field = one registry line + the producer stamp + the consumer read.

## 5 · Env law

One table in the module defines every override. The four shapes:

| shape | contract | members |
|---|---|---|
| force, never bank | env wins for this process (preempts wisdom); nothing persists while set | `VFFT_ZR2C_ROUTE`, `VFFT_IL_PAD`, `VFFT_SP_ROUTE`, `VFFT_FORCE_ZROUTE`, `VFFT_NO_ZTURN` (debug switches) |
| force + bank, stamped | the banked plan really ran the forced value; record carries `arms=1 src=env:<VAR>`; real races outrank it | `VFFT_TCUT` |
| suppress banked field | env set (any value) masks that field at lookup | `VFFT_TCUT` (masks `zt_tw` replay) |
| wisdom beats env | the one inversion, documented at the field | banked `il_kv` over `VFFT_NO_ILBLK` |

Kill switches (`VFFT_NO_IL2P`, `VFFT_NO_T2B`, `VFFT_NO_NAT_*`, `VFFT_NO_TCMT`)
are availability gates outside this table; none bank while set. Env is read
at CREATE time only. Dead names are never reused: `VFFT_PROTO_WIS`,
`VFFT_PROTO_PAD_WIS`, `VFFT_WISDOM`, `VFFT_WARM`, `VFFT_NO_K1`,
`VFFT_PROTO_WISDOM_OVERWRITE`.

## 6 · Extending the system

1. Read this file and the module's rule tables; derive the change from them.
2. A new payload field: one registry line (+class), producer, consumer;
   absent = legacy behavior.
3. A new key axis: an additive minor-version token (old records default to
   the legacy value; old binaries carry the token opaquely).
4. A new engine/route: a payload `eng=`/`route=` value entering the existing
   enumeration — never a new transform tag, never a new file, never a second
   parser.
5. If these rules give no answer, the decision is the owner's — present
   options, don't default.
