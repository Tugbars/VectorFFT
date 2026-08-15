# plan_executors.h re-promotion — 🟢 DONE (executed 2026-08-15, commit `0352c6ed`)

STATUS: **CLOSED.** Taken deliberately, gated, and committed alone as
"Add K=1 FFT plan executors". Nothing below is outstanding; the file is kept
as the record of what was done and how it was verified.

## The fact (as it stood)

`generator/generated/plan_executors.h` was **stale against its own input**:

| file | last touched | commit |
|---|---|---|
| `generated/spike_wisdom.txt` (the input) | 2026-08-13 | `a1e9d932` — "Add kind-5 wisdom for zr2c route selection" |
| `generated/plan_executors.h` (the output) | 2026-07-22 | `40c48156` — "new dag integration" |

It had been re-promoted and **deliberately reverted** during M10a and again at
M10b: the corpus inversion's discipline is that no tracked-file byte change
rides a structural step. The staleness predated all restructure work.

## What the re-promotion actually produced

`dune build @default` (WSL, rc=0 — so the promote is trustworthy, not the
"promotes even on failure" hazard). **Exactly one file changed:**
`plan_executors.h`. All 12 registry headers came back **byte-identical**,
confirming M10a/M10b's finding that the corpus work left the C build untouched.

Measured **CR-normalized** (git's own `--stat` pairs lines across the CRLF
boundary and lies about removals):

```
old: 26,159 lines      new: 27,279 lines
added: 1,024           removed: 0        ← PURELY ADDITIVE
```

⚠ **Correction to this doc's earlier figure.** It previously recorded
"+1120 lines, 0 removals". The measured value on the day it was taken is
**+1024**. The load-bearing property — *zero removals* — held in both
measurements; only the magnitude was off. (The still-earlier "+1172/−52"
reading was the CRLF artifact and was already retracted.)

The additions are **48 plan-shaped executor specializations** plus their
dispatch arms and the extern declarations they need — including the
radix17/19-class `t1_dit` kernels that ship in the tree but were undeclared in
the header. Shape of a typical addition:

```c
/* Plan-shaped executor specialization
 *   N=2048 K=1  factors=64,32  variants=FLAT,T1S
 *   orient=DIT dir=FWD isa=avx2 */
static void exec_n2048_k1_6432_v02_dit_fwd_avx2(...)
```

Nothing was deleted or rewritten, so no previously-selected executor changed
behaviour; the dispatcher gained arms it did not have.

## The C-side gate (all run AFTER the promote, against the new header)

Rebuilt through `build_tuned/build.py --vfft` (mingw 15.2, the 863-codelet
cached lib). Every gate binary was **deleted before rebuilding** — a stale
`.exe` makes a failed build look like a pass.

| gate | result |
|---|---|
| `zr2c_fd_gate` | **ALL CORRECT** — r2c+c2r × OOP+in-place × routes 0/1/W × N=512/2048/4096, plus the K=4 regression cell and non-pow2 N=510 |
| `mt_c2c_gate` | **ALL PASS** — MT==ST bitwise EXACT, 12 cells (default + NATURAL order) |
| `k1z_inplace_gate` | **ALL PASS** — memcmp-EXACT fwd+bwd, N=2048…32768, incl. `VFFT_TCUT=off` |
| `bench_1d_vs_mkl --zr2c` | route **resolves**; cross-check `xerr 4.1e-16` vs MKL |

`k1z_inplace_gate` is the most on-point of these: the additions are all **K=1**
executors in exactly the 2048–32768 range it covers memcmp-exact.

🔴 **No performance claim is attached to this change.** The machine was
compiling throughout; per the standing protocol those ratios are not data.
Whether the new specializations are *faster* than the generic path they now
pre-empt is an open question for a quiet-machine race, not something this
re-promotion established.

## Hygiene that held

- Every gate was pointed at a **scratch wisdom dir** (a copy of the real
  `generated/*.txt`), never the real one. `zr2c_fd_gate` *defaults* to the real
  `generated/` and races-and-banks on a miss; `mt_c2c_gate` and
  `k1z_inplace_gate` require `--wisdir` and their own headers say SCRATCH.
- The canonical bench got a **scratch CSV path** — running it bare overwrites
  the banked `vfft_perf_tuned_1d_zr2c.csv`.
- Verified after: all 10 real `generated/*.txt` md5s **unchanged**, and
  `git status` clean apart from the intended header.
- ⚠ Two argument contracts worth remembering: the bench's `[wisdom]` positional
  is a **file** (and must be `oop_wisdom.txt`, not `spike_wisdom.txt`, or the
  K=1 kind-4 front-door cells silently SKIP), while the gates take a **dir**
  via `--wisdir`. Running a gate `.exe` outside `build.py` also needs
  `C:\mingw152\mingw64\bin` on PATH or it dies with a bare exit 53.

## What this unblocks

`dune build @default` no longer resurfaces `plan_executors.h` as a modified
tracked file, so a dirty `generated/` once again means something real.
