# scripts/

Production codelet generation and compilation scripts. These encode the
best-known configurations from docs 09-42 — what variant flags, what
compiler, what target ISA produce the best codelets per radix.

## CSE / Algsimp passes are gated by algorithm class

This is a subtle but important point. The generator does NOT apply the
same optimization pipeline to every radix — several passes are gated by
whether the codelet uses Direct DFT (primes) or Cooley-Tukey
decomposition (composites, pow2). Passes that help one class actively
hurt the other; the gating is empirically derived from docs 23, 28, 30.

The dispatch happens inside `gen_radix.ml` via the `aggressive` flag
(true ⇔ `pick_algorithm n = Direct`, i.e., n=2 or odd prime ≥3). You
don't need to think about this when invoking the generator — the same
`gen_radix.exe N --twiddled ...` command works for all radixes — but
it matters when reasoning about why a given codelet looks the way it
does.

### Primes-only passes (aggressive=true)

| Pass | What it does | Why pow2 doesn't get it |
|------|--------------|------------------------|
| `factor_common_muls` | Winograd-style: factor out constants shared across output bins | Either no-op or breaks the FMA-friendly butterfly structure |
| `factor_by_atom` | Factor expressions by shared atom subexpressions | Same |
| `fma_lift` | Explicit `Add(Mul,c) → NK_Fma` lift (~1-2% prime win) | **Regresses composites** (doc 28): R=32 t1_dit llvm-mca SKX 312→226 cycles when DISABLED. Explicit FMA atoms constrain GCC's register allocator more than letting it auto-fuse. |
| Conjugate-pair DFT (in `dft.ml`) | Pair sums/diffs + shared `p_re`/`q_im`/etc. | Only applies to direct-DFT primes ≥3. Doc 23 brought R=11 from 300→190 ops. |

### Pow2/composite-only passes (aggressive=false)

| Pass | What it does | Why primes don't get it |
|------|--------------|------------------------|
| `share_subsums` | Factor common partial-sums across output bins | **Hurts primes** (doc 23): splits unified mixed-sign FMA chain into separate +/- sub-chains, costs 4 extra ops per pair output |
| Transposition fixed-point loop | Frigo's network transposition, iterated up to 6× with factor/share between passes | Inner step uses share_subsums; without it the loop offers no benefit on primes. Also skipped for twiddled codelets (t1_dit, t1_dif): Cmul nodes wrap symbolic twiddle loads, making the network non-linear in our representation |

### Universal passes (apply to both classes)

- `dedup_sub_pairs` — canonicalize `Sub(a,b)` vs `Sub(b,a)` and merge
- `Sub(Neg(Mul(a,b)),c) → fnmsub` peephole at construction time via
  `mk_sub_binary` (doc 30). Emits `vfnmsub` which GCC won't auto-derive.
- Single-use inlining (doc 24) — values with one consumer get inlined
  into their consumer's expression rather than as separate
  `const __m512d t<N> = ...` declarations. Closes the nested-intrinsic
  gap to hand-written codelets. Applies in the SU emit path universally.
- Spill recipe + SU scheduler — auto-applied for any twiddled CT
  codelet meeting `should_spill` (n ≥ 5 in practice). AVX2 with R ≥ 32
  also auto-fires GH (Goodman-Hsu pressure-aware mode). Doc 13: the
  cost-model rule "if CT-decomposed AND (n+6 > vec_regs OR vec_regs ≥ 32),
  use the full recipe."

### Why this matters for adding new radixes

If you add a new radix to `generate_codelets.sh`, the generator's
internal dispatch handles algorithm-class selection automatically based
on `pick_algorithm`. You don't need to specify which CSE passes apply.

The family separation in this script reflects two different things:
1. **Variant matrices** (log3 only applies to pow2, twidsq exists for
   square-block codelets, etc.)
2. **Per-family wisdom from docs 33-42** about optimal CT factorizations,
   compiler interactions, runtime crossovers

It does NOT reflect different optimization-pass gating — that's handled
by `pick_algorithm` and the `aggressive` flag inside the generator.

## Quick start

```bash
# One-time: build the generator (requires OCaml + dune)
dune build

# Generate all production codelets for both ISAs (~3 min)
ISA=both ./scripts/generate_codelets.sh

# Compile them all with gcc-11 + -flive-range-shrinkage
./scripts/compile_codelets.sh

# Output: codelets/<isa>/<family>/r<N>_<variant>.{c,o}
```

After both scripts complete, the `codelets/` tree contains 468 `.c` files
and 468 `.o` files, organized by ISA and family. The planner consumes
these — each one is a separate, independently-compilable codelet.

## Windows usage

Three paths, in order of recommendation:

### Option 1: WSL (Windows Subsystem for Linux)

If you have WSL set up (Windows 10/11), the bash scripts work as-is.
This is the most frictionless path because OCaml/dune/gcc-11 all install
cleanly via apt:

```bash
# Inside WSL
sudo apt install opam gcc-11
opam init -y && opam install dune
dune build
ISA=both ./scripts/generate_codelets.sh
./scripts/compile_codelets.sh
```

The `.o` files emit to the WSL filesystem and are linkable from there.
For mixed Windows/WSL workflows, generate inside WSL and access the
codelets/ tree from Windows via `\\wsl$\<distro>\path\to\project`.

### Option 2: Native PowerShell + MSYS2

PowerShell ports of both scripts live alongside the bash versions
(`scripts\generate_codelets.ps1` and `scripts\compile_codelets.ps1`).

```powershell
# One-time setup
# 1. Install OCaml for Windows from https://fdopen.github.io/opam-repository-mingw/
#    or use WinGet's OCaml package
# 2. Install MSYS2 from https://www.msys2.org/
# 3. In MSYS2 shell: pacman -S mingw-w64-x86_64-gcc
# 4. Add C:\msys64\mingw64\bin to your Windows PATH

# Build the generator
dune build

# Generate codelets — same env-var interface as the bash version
$env:ISA = "both"
.\scripts\generate_codelets.ps1

# Compile with gcc-11 + shrinkage
.\scripts\compile_codelets.ps1
```

PowerShell script parameters mirror the bash env vars:

```powershell
$env:ISA = "avx512"           # default; or "avx2", or "both"
$env:OUTDIR = "D:\codelets"   # override output location
$env:GEN = "C:\path\to\gen_radix.exe"   # override generator binary

# Family selection via positional args (same as bash):
.\scripts\generate_codelets.ps1 primes mid_pow2

# Compile script env vars:
$env:CC = "gcc-13"            # use different compiler
$env:EXTRA_CFLAGS = ""        # disable shrinkage
$env:JOBS = "8"               # override parallelism (default: $env:NUMBER_OF_PROCESSORS)
$env:CODELETS_DIR = "D:\codelets"

# Verify-only mode (compile but don't save .o files):
.\scripts\compile_codelets.ps1 -VerifyOnly
```

Notes:
- **PowerShell 5.1 vs 7+**: both scripts work on PowerShell 5.1 (the
  default on Windows 10/11). On PS5.1, `compile_codelets.ps1` falls
  back to sequential compilation; on PS7+, it uses `ForEach-Object
  -Parallel` (throttled to `$env:NUMBER_OF_PROCESSORS`). Install PS7
  from https://aka.ms/powershell for the parallel speedup.
- The PowerShell scripts fall back to plain `gcc` if `gcc-11` isn't on
  PATH, with a clear warning. For best results match the bash setup
  (production config = `gcc-11 + -flive-range-shrinkage` from doc 38).
- Output extension is `.o` to match the bash script. Windows GCC accepts
  both `.o` and `.obj`; we use `.o` for cross-platform consistency.
- The PowerShell scripts encode the same CSE/Algsimp algorithm-class
  gating notes in their help blocks (`Get-Help .\generate_codelets.ps1
  -Full` to view), citing the same docs as the bash versions.

### Option 3: Git Bash / MSYS2 bash

The bash scripts also work in Git Bash or an MSYS2 shell with the same
command-line interface as Linux. Use this if you want bash semantics on
Windows without WSL. PowerShell scripts are still the cleaner choice for
pure Windows automation (Task Scheduler, CI runners, etc.).

## Output layout

```
codelets/
├── avx512/
│   ├── primes/          ← R = 2, 3, 5, 7, 11, 13, 17, 19
│   ├── small_pow2/      ← R = 4, 8
│   ├── mid_pow2/        ← R = 16, 32, 64
│   ├── large_pow2/      ← R = 128, 256, 512
│   ├── xl_pow2/         ← R = 1024 (research-only; planner prefers cascade)
│   └── composites/      ← R = 6, 10, 12, 20, 25
└── avx2/                ← same structure
```

Per radix, the variant matrix is `{t1, t1s} × {dit, dif} × {fwd, bwd}`
(8 variants), with `log3` doubling that for pow2 sizes where it applies
(16 variants total).

## generate_codelets.sh

Drives `gen_radix.exe` over all (R, ISA, variant) combinations, emitting
`.c` files but NOT compiling them.

Options (env vars):
- `ISA=avx512|avx2|both` — target ISA (default avx512)
- `OUTDIR=path` — output directory (default `$ROOT/codelets`)
- `GEN=path` — override generator binary location

Positional args (optional) select families:
```bash
./generate_codelets.sh                          # all families
./generate_codelets.sh primes                   # primes only
./generate_codelets.sh pow2 large_pow2          # multiple
```

Available families: `primes`, `small_pow2`, `mid_pow2`, `large_pow2`,
`xl_pow2`, `composites`.

### Per-family wisdom

Each family corresponds to a band of radixes with shared characteristics
from the docs:

**primes** (R ∈ {2,3,5,7,11,13,17,19})
Recipe auto-fires for R≥5 via the `n>=5` clause (doc 29). Conjugate-pair
construction for odd primes ≥3 produces compact DAGs (doc 23). fma_lift
is gated to primes only (doc 28). 8 variants per radix; log3 doesn't
apply to primes.

**small_pow2** (R ∈ {4,8})
Used as building blocks in mixed-radix cascades. Recipe applies on AVX-512
(vec_regs ≥ 32 clause); may over-spill at small R but compiler trims it
back. 16 variants per radix.

**mid_pow2** (R ∈ {16,32,64})
The workhorse codelets. Recipe + SU active (docs 09-13: 13-69% wins over
Topo). On AVX2 with R≥32, GH (Goodman-Hsu pressure-aware mode) auto-fires
giving +4-8% on top of recipe (doc 21). 16 variants per radix.

**large_pow2** (R ∈ {128,256,512})
Monolithic codelets dominate up to R=512 (docs 33, 34). R=512 has a log3
crossover at B≈128 — generate both flat and log3 variants and let the
planner pick (doc 42). 16 variants per radix.

**xl_pow2** (R = 1024)
Monolithic at R=1024 loses to multi-stage cascade by ~50% (doc 41) due
to super-linear spill scaling. Generated only for research/repro;
the planner should never select this in production. Just 2 variants
(t1_dit_fwd, t1_dit_fwd_log3).

**composites** (R ∈ {6,10,12,20,25})
Mixed-radix composites for non-power-of-two transforms. Recipe applies.
8 variants per radix (log3 only applies to pow2).

## compile_codelets.sh

Walks the `codelets/` tree and compiles each `.c` → `.o` using the
production compiler config from doc 38: **gcc-11 + -flive-range-shrinkage**.

Options (env vars):
- `CC=gcc-11|gcc-13|...` — compiler (default gcc-11)
- `EXTRA_CFLAGS='-flive-range-shrinkage'` — additional flags (default applies shrinkage)
- `JOBS=N` — parallelism (default `nproc`)
- `CODELETS_DIR=path` — input tree (default `$ROOT/codelets`)
- `VERIFY_ONLY=yes` — check sources compile but don't save `.o` files

### Compiler choice rationale (doc 38)

- **gcc-11 + -flive-range-shrinkage**: 29% fewer stack ops at R=512 AVX-512
  vs gcc-13 default; 5-8% runtime gain at moderate B.
- **gcc-12** introduced an AVX-512 register allocator regression vs
  gcc-11 (-9.4% to -14% worse). gcc-13 inherits this.
- **Clang-18** is significantly worse on AVX-512: 3× more spills at R=512.
- The `-flive-range-shrinkage` flag is asymmetric on AVX2 (helps small R,
  mildly hurts large R) but production deployment uses it everywhere
  for CI simplicity.

To verify on different compilers:
```bash
CC=gcc-13 EXTRA_CFLAGS='' ./compile_codelets.sh   # gcc-13 default
CC=clang  EXTRA_CFLAGS='' ./compile_codelets.sh   # clang baseline
```

## Wall time on i9-14900K (estimated)

| Step | Container CPU | i9-14900K (estimated) |
|------|---------------|----------------------|
| Generate AVX-512 (234 codelets) | ~74s | ~30s |
| Generate AVX2 (234 codelets) | ~86s | ~35s |
| Generate both (468 codelets) | ~160s | ~65s |
| Compile all (468 .o files) | ~30 min* | ~12-15 min |

*Most of the compile time is in R=512 and R=1024 — each takes 4-5 min
on the container due to the codelet's size (~30-40K lines). With
parallel jobs (`JOBS=$(nproc)`), wall time is much better on multi-core
machines.

## Customization

To add a new radix or variant: edit `generate_codelets.sh`. Each family's
radix set is a bash variable near the top. The `emit_variants` helper
already covers the {t1, t1s} × {dit, dif} × {fwd, bwd} ± {log3} matrix —
just add the new R to the family's list.

To use different compiler flags per family: extend `compile_codelets.sh`
to dispatch on the directory path. Currently AVX-512 and AVX2 get
different `-m*` flags via path matching.
