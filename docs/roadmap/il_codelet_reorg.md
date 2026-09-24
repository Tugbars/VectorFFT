# Interleaved codelet tree — inventory and reorganisation proposal

**Status: proposal.** Nothing has been moved, renamed or deleted. Owner rulings: propose
first; report unused files, do not delete them.

Scope: the 746 kernels in `codelets/zil/avx2/` (672 `pure_il`, 20 `pure_il/tangent`,
54 `boundary_split`) plus the 30 fused drivers in `generator/generated/fused_codelets/`.

---

## 1. What is there, and what each family is for

Grouped by the engine that uses it — which is also the proposed folder layout.

| proposed folder | kinds | files | what it is for |
|---|---|---|---|
| `shared/` | `n1` | 62 | A whole small transform in registers: no scratch buffer, no twiddle table. Forward = the MONO route (small N); backward = the leaf of the pair and of chain3. |
| `mono/` | `mono64_il` fwd/bwd *(moved in from `oop/avx2`)* | 2 | N=64 as an 8×8 with both halves fused in registers. Kept as a raced form; lost to the pair on this machine. |
| `pair2p/` | `n1t`, `t2`, `t2t` | 135 | The two-pass Bailey pair. The transpose between the passes is folded into the store addresses — R lane swaps, no extra memory pass. Wins while input + scratch + output fit in L1. |
| `pair2p/` | `n1_ct`, `n1t_ct`, `t2_ct`, `t2t_ct` | 26 | Odd composite radices (15, 21, 25, 27) done as a small Cooley–Tukey inside the kernel, so the kernel stops spilling registers. |
| `pair2p/blocked/` | `n1b*`, `t2b*`, `n1tb*`, `t2bt*` | 46 | Large power-of-two radices (16, 32, 64) run as two passes through a small parked buffer, because the one-pass form spills. |
| `pair2p/tangent/` | `tan`, `bw32`, `t256`, `m128` | 20 | Rotations rewritten as cos·(1 − i·tan) so plain adds become FMAs, moving work off the busiest execution ports. Wins at radix 8 and 16. |
| `pair2p/log3/` | `t2_log3`, `t2b_log3` | 52 | Load a few twiddles and derive the rest. Loses in IL, because registers — not loads — are the limit. **Unused.** |
| `chain3/` | `t2tg` | 27 | A three-pass chain so odd factors can be used at all: a two-pass pair needs both pass counts even. `t2tg` does the backward pass's strided scatter. |
| `flat/` | `t2cp`, `t2cs`, `t2csg`, `t2csgn`, `t2csgt`, `t2csgnt` | 198 | Odd N. ZTURN-T's first stage needs a power-of-two item count, so odd N runs one sweep per factor instead; these kinds keep the short last sweep fast by putting the loop inside the kernel. |
| `flat/odd_mid/` | `msz`, `mszt` | 15 | The odd middle radices (3, 5, 7, 9, 15) for the flat engine. |
| `ztt/` | `t0tp`, `tmg`, `tlf`, `tlfi` · `t0d`, `tmgd`, `tld` | 39 | ZTURN-T, N = 2048 to 262144. Natural order with no reorder pass; re and im are split inside the kernel, which removes the lane shuffles of the complex multiply. The scrambled set (`t0d`/`tmgd`/`tld`) skips one full sweep of the array and wins in place only. |
| `shared/col/` | `n1c`, `t2c` | 110 | The 2D column stage. Twiddles are hoisted out of the column loop. The kernels allow input = output, so they run in place — which is also how the 1D in-place MONO path uses `n1c`. |
| `shared/col/blocked/` | `n1cb*`, `t2cb*` (`b48`, `b84`, `b88`, `b416`) | 16 | Blocked 2D column forms at radix 32 and 64, raced per cell. |

Total 748 = 746 today + the 2 `mono64_il` files moved in. The fused drivers stay where they
are: they are derived output with their own build rule, and `ztt/README.md` points to them.

**The rule behind most of this:** splitting a radix into passes pays exactly when the
one-pass kernel runs out of registers (AVX2 has 16). Where it doesn't, the extra pass is
pure loss. So radix ≤ 8 is never blocked, 16 is raced, 32 and 64 are always blocked.

---

## 2. What the banked wisdom uses

The shipped store (`src/wisdom/wisdom2_*.txt`) has 11,686 rows. IL routes: prime 3097,
chain3 2070, pair 1130, flat 1002, ZTURN-T 85, mono 76, four-step 8.

Kernel choices across the 2,250 banked pair slots:

| kernel form | share |
|---|---|
| default (one-pass) | 80% |
| `_ct` | 10% |
| tangent | 9% |
| blocked | 1% |

2D column forms banked: `b88` 12, `b48` 4, `b84` 1, `b416` **0**.

---

## 3. Unused files (reported, not deleted)

| files | count | why unused |
|---|---|---|
| `t2_log3`, `t2b_log3` | 52 | No consumer anywhere in `src/`. Loses in IL. |
| odd-radix `n1b`, `t2b` (9, 15, 21, 25, 27, 45, 49) | 28 | No resolver uses them. |
| `t2cs` | 22 | Reachable only through the `VFFT_ILFD_NO_GEN2` env pin. Forward-only fallback. |
| `t2tan`, `n1ttan` at radix 6, 10, 12 | 6 | No resolver arm; only `tangent_gate.c` builds them. |
| `b416` column form | 4 | In the race pool, but has never won a banked cell. |
| route 4 `VFFT_K1_IL_CASCADE` | — | Engine deleted 2026-09-15; the enum value remains. |

---

## 4. The one real problem: the tangent kernels cannot be regenerated

- **20 files**, and 199 banked pair slots depend on them.
- They are kept out of the corpus **on purpose** (`corpus.ml:80-93`): 18 of the 20 were
  generated with environment variables (`VFFT_CX_WING`, `VFFT_CX_LAZYLOAD`,
  `VFFT_CX_LAZYSTORE`, `VFFT_CX_SCHED=asis`, `VFFT_CX_ROTFMA`), and a corpus row cannot carry
  environment variables. Only `radix8_z_n1tan_bwd` and `radix8_z_t2ttan_bwd` need none.
- So `gen_set` will not recreate them, and the only record of how they were made is a comment
  at the top of each file.
- **Fix:** turn the five env reads into `--cil-*` flags (they are `Sys.getenv_opt` calls in
  `cx_ir.ml`, `cx_cpl.ml`, `cx_render.ml` and `c2c_il.ml`), keeping the env path working.
  Then replay the 20 recorded command lines and diff. Files that match become normal corpus
  rows; any that don't are listed by name.

---

## 5. How to do the move

- **`corpus.ml`** — keep the two zil quadrants (`zil-pure`, `zil-boundary`). Add one function
  that maps a file name to its subfolder, and have `gen_set` place each file by it.
  `emit_il_registry.ml` needs no change: it reads the corpus tables, not the disk.
- **`build.py` and `CMakeLists.txt`** — list the new folders explicitly. Do **not** switch to
  recursive globbing: CMake's per-folder count check exists because a single total once hid
  265 missing zil files for weeks.
- **The three gate TSVs** (`recipes.tsv`, `baseline_manifest.tsv`, `baseline_verdicts.tsv`) —
  re-path rows by matching their sha256, not by editing path strings.

**Order:**

1. Teach the build the new folders while the tree is unchanged. The library must come out
   identical.
2. `git mv` the files.
3. Re-path the TSVs.
4. Write one `README.md` per folder: which route uses it, what each kind is, which wisdom
   token selects it, what it is for, and what in it is unused.

**Check after each step:** the symbol list of `libdagcodelets.a` (`nm --defined-only`) must
match the list taken before step 1. A file count can hide a dropped kernel; a symbol list
can't. Re-run CMake configure after step 2 — its globs do not refresh on their own.

---

## 6. Recommended order of work

1. Tangent env → flags (§4). It protects files that are in use; the move only makes files
   easier to find.
2. The move (§5).
3. The READMEs.
4. Decisions on the unused files (§3), in any order.

---

## 7. Decisions needed

1. Do the tangent env → flag work?
2. Execute the move?
3. Unused files: delete the 52 log3 files? The 28 odd `n1b`/`t2b`? Retire `t2cs` and its env
   pin? Re-race then drop `b416`? Keep the route-4 enum value as a reserved number?

---

## 8. Open questions

- **`_ct` may be out of date.** Its banked wins were raced against the old odd kernels. The
  2026-09-23 odd-kernel fix made the plain kernels much faster (radix 25: 1.35 → 0.73 ns per
  point), so some `_ct` verdicts may flip on a re-race.
- **Two stale documents.** `docs/design/il_codelet_design.md` §3 describes the deleted cascade
  as if it were live. `CODELET_TAXONOMY.md` marks forward `n1` as dead (MONO has used it since
  2026-09-04) and marks `t2c` as a dead probe (that was a different kind; the name is now the
  2D column stage).

---

The detailed study behind this — the per-family optimization analysis, the verification
notes and the corrections — is in git at commit `8f07d86a`.
