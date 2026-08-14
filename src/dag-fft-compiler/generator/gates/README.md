# generator/gates — the corpus reproducibility gate (TRACKED)

The acceptance machinery for any change to `generator/lib`: the emitter must
keep reproducing the shipped codelet corpus (`src/dag-fft-compiler/codelets`,
1,432 `.c` files) **byte-verbatim at the recorded baseline** — 1403/1432 =
97.97% as of 2026-08-14. The job of every restructure step is DO NOT REGRESS
(`docs/roadmap/generator_lib_architecture.md`, §14).

Moved here from the gitignored `docs/research/generator_arch/` at **M0** (G1):
the acceptance criterion for a multi-week campaign must not be one
`git clean -xfd` from gone. Everything the gate needs now lives beside it,
tracked; every re-record (G2) is a reviewable diff.

| file | role |
|---|---|
| `full_corpus_gate.sh` | THE gate. `verify` (exit≠0 on any drift) / `record` / `report` / `manifest`. ~60 s in WSL. |
| `recipes.tsv` | 1,432 rows: per-file regeneration recipe (arm, argv0, env, args, post-sed). |
| `baseline_manifest.tsv` | sha256 of every LF-canonical corpus file — the corpus statement of record. |
| `baseline_verdicts.tsv` | the recorded per-file verdict baseline the gate diffs against. |
| `cil_matrix.sh` | the 183-case cx inner loop (~10 s). NOT sufficient for acceptance (12.8% of corpus). |
| `regen_cil.sh` | the only home of the `--cil-split` radix table the zil/pure_il recipes derive from. |

Hard rules (M0 hardening, `generator_lib_architecture.md` §14.1):

- **G4 — corpus drift is FATAL in every mode.** An intentional corpus change is
  its own reviewed commit: `bash full_corpus_gate.sh manifest && bash
  full_corpus_gate.sh record`, never folded into a structural step.
- **G3 — the gate builds all 19 executables and runs `cx_pipeline_test`**, so a
  chain-tail or cx break cannot hide behind a 2-exe build.
- 🔴 **Never run a bare `dune build`** in the generator: `@default` PROMOTES
  tracked `generated/` headers **even when the build fails** (reproduced:
  rc=1 and `plan_executors.h` rewritten). Scoped targets only.
- Run from WSL (`~/.opam/5.2.0`, dune 3.23, `DUNE_CACHE=disabled`); keep
  `WORK` on a Linux fs (`/mnt/c` costs ~8× wall time).
- Comparisons are against the **LF-canonical blob** (`git cat-file` semantics),
  never worktree bytes — `core.autocrlf=true` makes 99 worktree files CRLF.

Usage, from anywhere:

```bash
wsl bash -lc 'bash /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/gates/full_corpus_gate.sh verify'
```
