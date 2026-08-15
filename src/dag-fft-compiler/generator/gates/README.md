# generator/gates — the acceptance machinery (TRACKED)

**What this folder is for.** The generator emits 1,432 `.c` codelets that ship
in `src/dag-fft-compiler/codelets`. Those files are the product. Any change to
`generator/lib` — a refactor, a module split, a new abstraction — must be
provably **non-destructive**: the emitter has to keep producing the same bytes
it produced before. This folder is the machinery that proves it, and the
recorded evidence it proves it against.

Its job is one sentence: **DO NOT REGRESS.** It does not check that the FFT math
is correct (see *What this folder does NOT do*, at the bottom) — it checks that
nothing changed that you did not intend to change.

Moved here from the gitignored `docs/research/generator_arch/` at M0: the
acceptance criterion for a multi-week campaign must not be one `git clean -xfd`
away from gone. Everything the gate needs now lives beside it, tracked, so
every re-record is a reviewable diff.

---

## The files

| file | kind | role |
|---|---|---|
| `full_corpus_gate.sh` | the instrument | THE gate. Replays the whole corpus and compares bytes. ~60 s in WSL. |
| `recipes.tsv` | data | 1,432 rows: how to regenerate each file, and what its bytes must be. |
| `baseline_manifest.tsv` | data | sha256 of every shipped file — detects hand edits to the corpus. |
| `baseline_verdicts.tsv` | data | the expected *verdict class* per file — the pass/fail reference. |
| `layout_smoke.sh` | compensating gate | 17 cases over the IL-layout arms the corpus cannot see. **17/17 green** — see below. |
| `cil_matrix.sh` | tool | the 183-case cx emission matrix (~10 s). Exercises off-default `VFFT_CX_*` knobs. |
| `regen_cil.sh` | tool (legacy) | in-place regen of cil-owned files. Superseded by `gen_set --root codelets zil-pure`. |
| `.gitattributes` | config | forces LF in this folder on every platform. Three lines that prevent a whole class of false failures. |

---

## `full_corpus_gate.sh` — how it actually works

Run it from WSL. Five phases:

**1. Corpus integrity (G4).** Hashes every shipped `.c` and compares against
`baseline_manifest.tsv`. This asks a different question from the rest of the
gate: *did someone edit a codelet by hand?* A generator change and a hand edit
both show up as "bytes differ" downstream, and only this phase can tell them
apart. **Drift here is fatal in every mode**, including `record` — you cannot
accidentally bless a corpus edit by re-recording.

**2. Scoped build (G3).** Builds all 15 executables — not just `gen_radix` — and
then *runs* `cx_pipeline_test`. Without this, a break in a chain-tail consumer
(`dbg_eval`, `dump_ir`, `dbg_zil_math`) or in the cx stack is invisible and the gate
reports PASS on a broken tree. Scoped targets only; see the `dune build` hazard
below.

**3. Regeneration.** Every file is re-emitted into a scratch tree, by one of
five *arms* recorded per row in `recipes.tsv`:

| arm | count | how it regenerates |
|---|---:|---|
| `genset` | 1,074 | one warm `gen_set.exe --root <tmp> all` process for the whole corpus |
| `derive` | 221 | one fork per file: zil/pure_il filename grammar + the `--cil-split` radix table |
| `replay` | 125 | one fork per file, using the file's own recorded argv |
| `k1` | 6 | the `--k1-mono` family |
| `ship` | 6 | tangent recipes: env overrides + a post-emit symbol rename |

Two details that look odd but are load-bearing: `argv[0]` is reproduced with
`exec -a` (some recorded stamps are not the plain binary name), and empty
fields are written `-` in the TSV, because bash's `read` collapses runs of tab
and would silently shift every field after the first empty one.

**4. Classification.** Each regenerated file is compared against the LF-canonical
shipped file and gets a verdict:

| verdict | meaning | current count |
|---|---|---:|
| `IDENTICAL` | byte-for-byte equal | 1,403 |
| `PROLOGUE_ONLY` | only the comment header before `#include` differs — code identical | 11 |
| `PROLOGUE+EOF_NL` | prologue plus a trailing-newline difference | 1 |
| `BODY_DIFFERS` | actual code differs (count of differing lines attached) | 17 |
| `MISSING` / `EMIT_FAILED` / `NO_RECIPE` | regeneration produced nothing usable | 0 |

**5. Verdict diff.** The gate does **not** require everything to be `IDENTICAL`.
29 files legitimately do not reproduce (dead-era orphans, sunset copies, cells
whose bodies drifted). Demanding IDENTICAL of them would mean living with a red
gate forever, which trains everyone to ignore it. Instead each file is pinned to
its *recorded class* in `baseline_verdicts.tsv`, and the gate fails if any file
moves in **either direction** — a non-reproducer that suddenly starts matching is
just as much a signal as the reverse.

### Modes

```bash
bash full_corpus_gate.sh verify     # the gate: exit != 0 on any drift   (default)
bash full_corpus_gate.sh report     # verify + print every non-IDENTICAL file in full
bash full_corpus_gate.sh record     # rewrite baseline_verdicts.tsv from the current run
bash full_corpus_gate.sh manifest   # rewrite baseline_manifest.tsv from the current corpus
```

`verify` and `report` are read-only and safe to run any time. `record` and
`manifest` **change the reference** — they are how you bless an intentional
change, and they belong in their own reviewed commit, never folded into a
structural step. The blessing ritual is:

```bash
bash full_corpus_gate.sh manifest   # corpus bytes are now the reference
bash full_corpus_gate.sh record     # verdict classes are now the reference
bash full_corpus_gate.sh verify     # must come back green
```

That sequence, plus a per-file audit that the *bodies* did not move, is exactly
how the two announced regens (M12b's provenance headers, M10b's coverage raise)
were landed.

---

## `layout_smoke.sh` — the compensating gate

The byte gate can only test inputs the corpus contains. The five IL-layout arms
have **zero corpus representatives**, so byte-identity says nothing about them.
This smoke drives every layout shape through the real emitter, compiles the
output with `gcc -Werror`, and asserts that the declared pointer parameters are
exactly the referenced ones.

It also tests the **negative space**, which nothing else does: illegal layout
combinations must fail *loudly* at emission (the anti-hybrid law raises) rather
than silently emitting uncompilable C. 17 cases. Weaker than byte-identity —
stated plainly, not oversold — but it is the only net over the error paths.

> 🟢 **STATUS 2026-08-15: 17/17 GREEN** — restored the same day it went red.
>
> It had been RED at 14/3: `neg_ip_both`, `neg_strided_both` and
> `neg_il_plus_r2c` (e.g. `--strided-il-in --strided-il-out` together) were
> **emitted silently instead of refused**. Cause: since M5 the illegal pair is
> *structurally unrepresentable* — the `Codelet` descriptor carries one
> three-way `il` field per axis, not two booleans — so `of_argv` took the last
> flag and dropped the other **before** the Layout law or `emit_body`'s strided
> guards could ever see it. The hybrid stayed impossible to emit (the law's
> purpose held), but **"refuse loudly" had degraded to "silently pick one"** —
> the exact silent-acceptance class this repo hunts — and the provenance header
> still stamped both flags, so the artifact misrepresented itself.
>
> **Fix:** the exclusivity check moved UP to where the conflict is still
> visible. `Codelet.of_argv` now raises `Parse_error` when a second,
> *disagreeing* flag claims an axis that is already set, naming both sides
> (`--ip-il-in and --ip-il-out are mutually exclusive: …`); repeating the same
> flag stays legal. All four axes are covered (ip / strided / oop-il-in /
> oop-il-out) plus the cross-family `--strided-il-* + --strided-r2c` pair.
> Verified safe first: 0 conflicts across all 1,433 recipe rows and no Corpus
> matrix emits these flags — and the corpus gate re-ran green at 1403/1432,
> drift 0. Error strings are deliberately **ASCII**: OCaml's uncaught-exception
> printer escapes non-ASCII bytes, so an em-dash reaches the user as
> `\226\128\148`.
>
> **The process lesson is bigger than the bug, and it stands.** The corpus gate
> ran after every campaign step; this smoke did not run once between M3 and
> 2026-08-15. A compensating gate covers the blind spot *by construction* —
> which means the main gate can never tell you it has gone red. **Run both.**

---

## The data files

**`recipes.tsv`** — 1,432 rows, 9 columns:

```text
path  arm  argv0  env  args  sed_from  sed_to  sha256_lf  bytes_lf
```

This is the corpus's identity: for each file, *how to rebuild it* (arm, argv,
env, post-sed) and *what it must weigh* (sha + size, over LF-canonical bytes).
It is also the recovery record — before the cil family carried provenance
headers, this file was the only place its 221 recipes existed.

**`baseline_manifest.tsv`** — `path <TAB> sha256`, one row per shipped file. The
hand-edit tripwire (phase 1).

**`baseline_verdicts.tsv`** — `path <TAB> arm <TAB> verdict <TAB> detail`. The
pass/fail reference (phase 5).

---

## Rules that exist because they were learned the hard way

- 🔴 **Never run a bare `dune build`** in the generator. `@default` **promotes
  tracked `generated/` headers even when the build fails** — reproduced: rc=1
  and `plan_executors.h` rewritten anyway. Use scoped targets.
- 🔴 **Comparisons are against LF-canonical bytes**, never worktree bytes.
  `core.autocrlf=true` leaves ~99 worktree files CRLF; comparing raw bytes
  reports spurious failures that measure your checkout, not the emitter.
  This is also why `.gitattributes` pins this folder to LF.
- **Run from WSL** (`~/.opam/5.2.0`, dune 3.23, `DUNE_CACHE=disabled`), and keep
  the scratch dir `WORK` on a Linux filesystem — `/mnt/c` costs roughly 8× the
  wall time.
- **A gate failure is a question, not a verdict.** Read *what* changed before
  concluding anything: in M12b, 329 files went non-IDENTICAL and every one was a
  comment header. The transition matrix (which verdict moved to which) is the
  first thing to look at, and `report` mode prints the detail.
- **Editing gate scripts from a heredoc eats backslashes.** Line-continuation
  needles must be built with `chr(92)`, or you will silently break the build
  list.

---

## What this folder does NOT do

It never compiles a codelet for correctness, never runs one, and never times
anything. A codelet could compute complete garbage and still pass every check
here — it only has to be the *same* garbage as before.

Numerical correctness lives elsewhere: the fd-gate (38/38), `mt_c2c_gate`
(MT == ST bitwise), the c2r matrix gate, and the bench correctness checks.
Performance lives elsewhere again (`build_tuned/benches`, quiet-machine
protocol). Those gates and this one are complementary, and neither substitutes
for the other.

---

## Usage from anywhere

```bash
wsl bash -lc 'bash /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/gates/full_corpus_gate.sh verify'
```

Context and the full milestone record: `docs/roadmap/generator_lib_architecture.md` §14.
