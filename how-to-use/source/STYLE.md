# How the VectorFFT User Guide is written

Every chapter follows this file. `sample/sample.md` uses each construct once; build it to see
the source beside the result (section 9).

## 1. Files and numbers

`book.yml` fixes the order of the book and numbers it: chapters 1 to 19, appendices A and B.
A chapter file never states its own number. Each file name starts with the file's **key**.

| Key | File in `chapters/` | Title (the file's `#` line) | Part |
| --- | --- | --- | --- |
| fm1 | `fm1_using-this-guide.md` | How to use this guide | front matter |
| fm2 | `fm2_notation.md` | Notation | front matter |
| ch01 | `ch01_building.md` | Building and installing VectorFFT | I Getting started |
| ch02 | `ch02_first-transform.md` | Your first transform | I |
| ch03 | `ch03_describing.md` | Describing a transform | II Using the API |
| ch04 | `ch04_plans.md` | Plans: create, execute, destroy | II |
| ch05 | `ch05_buffers.md` | Buffers and placement | II |
| ch06 | `ch06_complex.md` | Complex transforms in one, two and three dimensions | II |
| ch07 | `ch07_real.md` | Real transforms | II |
| ch08 | `ch08_batches.md` | Batches | II |
| ch09 | `ch09_threads.md` | Threads | II |
| ch10 | `ch10_wisdom.md` | Planning and wisdom | II |
| ch11 | `ch11_errors.md` | Errors and messages | II |
| ch12 | `ch12_poisson.md` | Solving a periodic Poisson equation | III Worked examples |
| ch13 | `ch13_convolution.md` | Linear convolution of 3D data | III |
| ch14 | `ch14_derivatives.md` | Spectral derivatives in place | III |
| ch15 | `ch15_wavepacket.md` | Propagating a wave packet | III |
| ch16 | `ch16_dfti-switch.md` | Switching a DFTI program to VectorFFT | IV Moving a DFTI program |
| ch17 | `ch17_dfti-support.md` | What the DFTI interface supports | IV |
| ch18 | `ch18_vfft-h.md` | `vfft.h` reference | V Reference |
| ch19 | `ch19_environment.md` | Environment variables | V |
| appa | `appA_windows.md` | Notes for Windows and MinGW | V (Appendix A) |
| appb | `appB_troubleshooting.md` | Troubleshooting | V (Appendix B) |

The title page ("VectorFFT", "User Guide", "v0.6 beta") and the contents come from
`book.yml`; nobody writes them. Names a chapter owns, all lower case:

- **labels**: `<key>-<kind>-<name>`, kind one of `sec`, `eq`, `fig`, `tbl`:
  `ch07-sec-in-place`, `ch07-eq-half`, `ch07-fig-bins`, `ch07-tbl-sizes`. The build refuses a
  label that does not start with the file's key and a hyphen.
- **listings**: `listings/<key>_<name>.c`, name in letters, digits and underscores:
  `listings/ch07_r2c_in_place.c`. Its outputs are `listings/out/ch07_r2c_in_place.out`
  (stdout), `.err` (stderr) and `.json` (the run).
- **figures**: `figures/<key>_<name>.py`, which writes `figures/<key>_<name>.svg`.

Touch only your own files. Never edit `how-to-use/examples/*.c`, `book.yml`, `tools/` or
another chapter's files; report what you need instead.

## 2. Voice

- **A how-to-use guide.** Each chapter starts from something a programmer does and ends in
  working code: a listing and what it printed. Open with one or two sentences that say what
  the reader does in the chapter. Title sections by the task (`## Run a real transform in
  place`). No "Introduction" or "Summary" sections, no exercises, no Example blocks. Part V
  (reference) is for looking things up: tables, `{api}` blocks, short code lines.
- **Plain words, second person, present tense, active voice.** "You pass the buffer twice."
  No future tense, no history, no "we".
- **Not a math class.** Mathematics only where a call depends on it: the forward and backward
  definitions with the sign and the unnormalized scale ($N$, $N_1N_2$, $N_1N_2N_3$), the
  storage index of a 2D or 3D array, the half-spectrum size $N/2+1$ and where negative
  frequencies sit, the padded in-place row length, batch addressing. One equation where a
  convention matters, then code. No derivations, proofs, DTFT or Dirichlet pictures, butterfly
  graphs or flow graphs.
- **No coined labels.** Say what the reader does. Code things by their code names, exactly
  spelled, in code font, never plural or possessive: `vfft_create()`, `VFFT_FLAG_PATIENT`,
  `n[0]`. Never "source mode".
- **Scope.** The native API (`vfft.h`), the DFTI interface (`src/shims/dfti/`), the four 3D
  examples. Never in a task chapter: `VFFT_LAYOUT_SPLIT`, `vfft_execute_split()`, `VFFT_DCT1`
  to `VFFT_DHT`, rank 4, `VFFT_FLAG_EXHAUSTIVE`, `vfft_diagnostics.h`. Chapter 18 lists them
  once, as not part of this release; Table 1.2 also names the installed `vfft_diagnostics.h`,
  as not part of this release.
- **Planning.** Every create plans PATIENT: `VFFT_FLAG_PATIENT` names the default,
  `VFFT_FLAG_EXHAUSTIVE` is refused, `VFFT_FLAG_RECALIBRATE` measures again. A create that
  misses wisdom measures for seconds, up to about a minute for some configurations, and saves
  the winner; later creates are served at once.
- **Wisdom folder.** The library comes with its own wisdom folder, one folder per CPU
  (`src/wisdom/`; `<prefix>/share/vectorfft/wisdom/` installed), and with no store named it
  picks the folder of this CPU by itself, serves from it and saves into it. A program is run
  with no setup: never tell the reader to create an empty directory or to set
  `VFFT_WISDOM_DIR` in order to run a program. Naming a store (`wisdom_dir`,
  `VFFT_WISDOM_DIR`) is a special case that Chapter 10 explains once
  (`ch10-sec-own-store`); other chapters say it in one sentence and refer there.
- **Version.** Write "v0.6 beta". No listing prints `vfft_info().version`.
- **Building.** The route is the library target: CMake `find_package(VectorFFT)`,
  `add_subdirectory`, `FetchContent`, or one compiler line against an installed package.
  `gauntlet/build.py` is never a user's route; at most one plain line says the gauntlet is the
  tool for testing the library on your own machine.
- **Correct use, not defects.** Describe what works and how to use it: one plan per thread at
  a time, non-`NULL` outputs. Never present a defect as behaviour and never invent a
  workaround for one. A defect that blocks a task goes in your report, one line, not in the
  guide.
- **No attribution.** Name Intel MKL's DFTI interface as the API being replaced; say nothing
  about how MKL, FFTW or any other library works inside.
- **Words.** Use the terms of section 8. Never: race, bank, verdict, rigor, codelet, engine
  names; for the header's "cell" write "configuration".
- **References by number.** "Eq. (2.1)", "Figure 7.1", "Table 3.2", "Section 10.4",
  "Chapter 12", "Appendix A". Never "above", "below" or a page number.

## 3. Headings and labels

```text
# Real transforms                  the title: once, first line, exactly as in section 1
(ch07-sec-in-place)=               a label: the line directly before the heading it names
## Run a real transform in place   a section: numbered 7.3 by the build
### The padded rows                a subsection: 7.3.1 (no deeper headings)
```

Headings are in sentence case and carry no math (code names are fine). Front matter
headings are not numbered. A section without a label gets one (`ch07-s3`), but only a
labelled section can be referenced.

## 4. Cross-references

| You write | The page shows | For |
| --- | --- | --- |
| ``{ref}`ch07-sec-in-place` `` | Section 7.3 | a `##` or `###` heading |
| ``{ref}`ch07` `` / ``{ref}`appa` `` / ``{ref}`part2` `` | Chapter 7 / Appendix A / Part II | a chapter, appendix, part |
| ``{ref}`fm2` `` | Notation | front matter (its title) |
| ``{numref}`ch07-fig-bins` `` | Figure 7.1 | a figure |
| ``{numref}`ch07-tbl-sizes` `` | Table 7.2 | a table |
| ``Eq. {eq}`ch07-eq-half` `` | Eq. (7.3) | an equation |
| ``Eqs. {eq}`ch02-eq-pair` `` | Eqs. (2.1) | a group of equations |

In the book every reference is a link. A reference to a chapter that is not written yet still
shows its number; a label that exists nowhere fails the build.

## 5. Equations

```text
$$
%lead: Forward:
X[k] = \sum_{n=0}^{N-1} x[n]\, W_N^{kn}, \qquad 0 \le k \le N-1,
$$ (ch02-eq-forward)
```

- A label after the closing `$$` numbers the equation per chapter: (2.1), (A.1); front
  matter (0.1). A display without a label is not numbered; number every display.
- `%lead: Forward:` puts a lead word at the left, as in Oppenheim and Schafer.
- `%group: ch02-eq-pair` on consecutive equations numbers them (2.1a), (2.1b); ``{eq}`ch02-eq-pair` ``
  cites them together as (2.1).
- `formula, \qquad condition` sets the condition beside the formula, or below it on a phone.
- `%narrow:` followed by the same equation broken by hand (`\begin{aligned}`) is the form a
  phone shows, never the PDF; use it for a display longer than one short line (a triple sum).
- Inline math is `$...$`. A `$` followed by a digit is not math, so "$5" is safe; a shell
  variable always goes in code (`$PATH`, `$env:VFFT_WISDOM_DIR`), never bare. Inside a table a
  `|` ends the cell, even in math: write $|x|$ there as `$\lvert x\rvert$`.
- The faces lack ≤ ≥ ∑ ∞ ⋅ ℓ ⌊ ≈: these go in math only (`$\le$`). The check of the PDF
  fails on a character the fonts lack.

## 6. Figures, tables, notes, steps, API blocks

```text
:::{figure} ch07_bins.svg
:name: ch07-fig-bins
:alt: What the figure shows, in words, for a reader who cannot see it.
The caption: full sentences that state the fact the figure shows.
:::

:::{table} The caption: one sentence, above the table.
:name: ch07-tbl-sizes
| Transform | Input | Output |
| --- | --- | --- |
| `VFFT_R2C` | $N$ | $2(N/2+1)$ |
:::

:::{note}
Something the reader needs at this point.
:::

:::{warning}
Only what breaks a program or its results. Never two notices in a row.
:::

:::{steps}
1. Configure.
2. Build.
:::
```

`{steps}` shows "Step 1.", "Step 2.". A table has one header row; code names in backticks.
In Chapter 18, `{api}` shows a signature verbatim from `include/vfft.h`, then its parameters:

```text
:::{api}
vfft_plan vfft_create(const vfft_config_t *config);

`config`
: The configuration, Table 3.1.

Returns the plan, or `NULL`; `vfft_error()` says why.
:::
```

Figure scripts use `figures/style.py` (the site's ink #101010, muted #6a6a6a, rule #dcdcdc,
on white, and the two accents of the Colour section; 140 mm wide, 8 pt lettering, every glyph
outlined in Computer Modern):

```text
import style
f = style.Figure(height=100)                          # pt; the width is W = 396.85 pt
xs = f.cells(66, 38, 8, w=40, h=20, labels=["$X[%d]$" % k for k in range(8)],
             fills=[None] * 5 + ["rule"] * 3)          # "rule": the light grey
f.brace(66, 266, 34, "stored", below=False)
f.text("frequency", 56, 89, anchor="r")                # words: cmr10; $...$: math
f.text("in_doubles", 200, 20, font="tt")               # code names: typewriter
f.save()                                               # writes figures/<this script>.svg
```

`f.save()` keeps the full width and crops the figure to what it draws, 6 pt above and below,
so every figure sits at the same distance from the text; `height` only sizes a figure that
draws nothing. `style.py` also has `line`, `rect` (with `hatch=True`), `arrow`, `grid`.
Figures show only what a user reads or does: array layouts, which bin is which frequency,
padded rows, batch layouts. Colour follows the Colour section. `figures/sample_half_spectrum.py`
is a complete script. The build runs a script when its SVG is missing or older than the script
or `style.py`.

### Colour

The book has two accent colours, each with one meaning everywhere, in `style.py` as `BLUE`
#2457c5 with its tint `BLUE_T` and `RED` #c8322b with its tint `RED_T`:

- **Blue is yours: the input, before.** The samples you supply, the input buffer, the real
  part, the forward direction, a create served from wisdom.
- **Red is the library's: the output, after.** The bins it writes, the output buffer, the
  imaginary part, the backward direction, a miss that measures.

Grey (`"rule"`) stays for what is not read, padding and what is not stored; ink draws the
structure. A tint fills a cell; the full colour strokes it (`strokes=`) and letters the labels
outside cells: a brace's words, a row's name. Lettering inside a tinted cell stays ink. Colour
never carries meaning alone: a coloured cell also carries a word, a glyph or a hatch, so a
greyscale print loses nothing. The notices carry the same rules: a `{note}` has a blue border
and label word, a `{warning}` a red one, and their text stays ink. Nothing else on a page is
coloured.

## 7. Code

### Code blocks

````text
```sh
cmake --build build
```
````

The languages are `sh` (Git Bash or Linux), `powershell`, `cmake`, `text`, `ini`, `json`.
A `text` block is what a program printed, quoted by hand (the store line with its
placeholder, Section 10.2): the page shows it as an output, never as code to type.
C is never typed into a chapter: it comes from a listing or an excerpt of an example. Lines
stay within 80 columns (the build warns; the PDF wraps longer ones). Indented code blocks
and raw HTML are not used.

### Listings

A listing is a complete C11 program in `listings/`. Include all of it, or one region:

```text
:::{listing} ch07_r2c_in_place.c
:::

:::{listing} ch07_r2c_in_place.c
:region: execute
:::
```

A region is marked inside the listing, on lines of their own, and the page never shows them:

```text
    /* region execute */
    vfft_execute(plan, VFFT_FORWARD, x, x);
    /* end region execute */
```

Also on lines of their own, for the harness only: `/* harness: env VFFT_VERBOSE=1 */`
(repeatable; never `VFFT_WISDOM_DIR`), `/* harness: exit 1 */` (the exit status it must
end with), `/* harness: args A B */`, `/* harness: timeout 300 */`.

The C style is the examples' style, and the harness compiles with
`-std=c11 -Wall -Wextra -pedantic -Werror`:

- at most 80 columns (the build refuses longer lines), spaces not tabs, Allman braces;
- a first comment line `name.c: what it does`; sizes in an `enum`; one file-scope
  `static const double TOL`;
- `#include "vfft.h"` only (a DFTI program: `#include "mkl_dfti.h"`); never
  `vfft_diagnostics.h`;
- a zeroed configuration (`vfft_config_t cfg = {0};`), and every create checked:
  `if (!plan)` prints `vfft_error()` at once and returns 1;
- buffers sized from `vfft_info()` (`in_doubles`, `out_doubles`; `stride` from every in-place
  2D or 3D real plan), allocated with `vfft_malloc()`, tested, released with `vfft_free()`;
  plans released with `vfft_destroy()`;
- a check against a stated reference with a stated tolerance, and a NaN fails it:
  `if (d > err || isnan(d)) err = d;` then `pass = err <= TOL;`;
- a last line that ends in `PASS` or `FAIL`, and `return pass ? 0 : 1;`;
- no timings, addresses, paths, user names or `vfft_info().version` in the output;
- no `wisdom_dir`: a listing uses the library's wisdom folder, as a reader's program does (the
  harness runs it on a fresh copy, section 9); naming a store is Section 10.3's, in words.

A point about C++ is made by a `.cpp` listing (C++11, `vfft_config_t cfg = {};`, the
`vfft_malloc()` result cast).

### What a program printed

```text
:::{output} ch07_r2c_in_place
:::

:::{output} ch07_r2c_in_place
:stream: stderr
:::

It prints {output}`ch07_r2c_in_place`.
```

The block quotes standard output (or standard error) verbatim; the inline form is for an
output of one line. Quote standard error only where what the library prints is the subject
(Chapters 10 and 11, Appendix B). Standard error holds the store line, which names the
harness's copy of the store, not the reader's: where you show it, write the lines in a `text`
block, copied from the `.err` file, with the store line as `[wisdom2] <the folder of this
CPU>: <n> record(s) loaded, writable` (Section 10.2 explains it). The build fails when a
listing changed after its last harness run.

### Excerpts of the examples

The worked examples (Chapters 12 to 15) quote `how-to-use/examples/*.c` in place. Copy the
lines exactly, indentation included; a line of three dots, indented as the line after it,
leaves lines out:

````text
```c excerpt=poisson3d.c
        vfft_execute(r2c, VFFT_FORWARD, f, F);
        ...
        vfft_execute(c2r, VFFT_BACKWARD, F, phi);
```
````

The build checks that every piece still occurs in the file, in order, and labels the block
with the file and its line numbers. Never add markers to the examples. The harness also builds
and runs the four examples as they are, so their output embeds like a listing's:
`:::{output} poisson3d` (`convolve3d`, `gradient3d`, `wavepacket3d`); `--only poisson3d` runs
one. An example that measures prints rounding-level numbers that can differ between runs.

## 8. Notation and terms

| Symbol | Meaning | In the code |
| --- | --- | --- |
| $j$ | the imaginary unit | `vfft.h` writes `i`; C99 code writes `I` |
| $W_N = e^{-j2\pi/N}$ | the kernel of the forward transform | |
| $x[n]$, $0 \le n \le N-1$ | the input of a 1D transform | `in`: $\mathcal{R}e\,x[n]$ at `in[2n]`, $\mathcal{I}m\,x[n]$ at `in[2n+1]` |
| $N$ | the length | `n[0]`, `rank` 0 or 1 |
| $X[k] = \sum_n x[n] W_N^{kn}$ | the forward transform | `VFFT_FORWARD` |
| $\sum_k X[k] W_N^{-kn} = N\,x[n]$ | the backward transform: no $1/N$ | `VFFT_BACKWARD` |
| $N_1, N_2, N_3$ | the extents, $N_1$ slowest | `n[0]`, `n[1]`, `n[2]` (contiguous) |
| $x[n_1,n_2,n_3]$, $X[k_1,k_2,k_3]$ | a 3D array and its transform | `rank = 3` |
| $\ell = (n_1N_2+n_2)N_3+n_3$ | where a sample is stored | row-major, the last axis contiguous |
| $((n))_N$ | $n$ modulo $N$ | |
| $X[k] = X^*[((-k))_N]$ | the symmetry of a real sequence's transform | why `VFFT_R2C` keeps half |
| $\lfloor N/2\rfloor + 1$ | the bins of the half spectrum | `N/2 + 1` in C |
| $K$, $t$ | the batch count; one transform of a batch | `howmany`; $0 \le t \le K-1$ |

- **Sign and scale.** The forward kernel is $e^{-j2\pi kn/N}$. A forward then a backward
  transform returns the input times $N$ (1D), $N_1N_2$ (2D), $N_1N_2N_3$ (3D); `VFFT_C2R`
  after `VFFT_R2C` the same. The program divides, best inside a pass it makes anyway.
- **Frequencies.** Bin $k \le N/2$ holds frequency $k$; bin $k > N/2$ holds $k - N$. At an
  even $N$, bin $N/2$ is both $N/2$ and $-N/2$.
- **Index letters.** $n$ for samples, $k$ for bins, $t$ for the transform of a batch.
  Never $e$ as an index, although `vfft.h` uses it.
- **Batches** in the header's addressing, written with $n$: contiguous, element $n$ of
  transform $t$ at `z[2(t N + n)]`; lane-major, at `z[2(n K + t)]`.

The terms, in the header's words; use each with this meaning only:

| Term | Meaning |
| --- | --- |
| plan | what `vfft_create()` returns (`vfft_plan`): one configuration committed to one measured plan, execute-ready; it owns what it allocated until `vfft_destroy()` |
| create | the call `vfft_create()`: it checks the configuration, serves it from wisdom or measures it on a miss and saves the winner, then builds the plan |
| execute | the call `vfft_execute()`: it runs a plan on your buffers, and allocates, measures and decides nothing |
| wisdom | the record of the plan that won when a configuration was measured |
| wisdom folder | the folder of wisdom that comes with the library, with one folder in it for each CPU: `src/wisdom/` in the sources, `<prefix>/share/vectorfft/wisdom/` once installed |
| store | the directory a create reads wisdom from and saves winners into: the folder of this CPU in the library's wisdom folder, or a directory that `wisdom_dir` or `VFFT_WISDOM_DIR` names |
| miss | a create whose configuration has no wisdom in the store: it measures, then saves the winner |
| refusal | a configuration or call the library does not serve: `vfft_create()` returns `NULL`, and `vfft_error()` says why; nothing is converted and nothing falls back |
| natural order | `VFFT_ORDER_NATURAL`, the default: bin $k$ at position $k$ |
| half spectrum | what `VFFT_R2C` writes and `VFFT_C2R` reads: bins $0$ to $N/2$ of the last axis |
| lane-major | `VFFT_BATCH_LANE_MAJOR`: the transforms of a batch side by side, sample by sample |
| pitch | the doubles from one row to the next of an in-place 2D or 3D real array: `vfft_info().stride`; `VFFT_FLAG_OWN_PITCH` lets the plan choose it |

## 9. Commands

From `how-to-use/source`, in Git Bash with the MinGW-w64 `bin` folder first on `PATH`;
`<prefix>` is the installed VectorFFT, and `<work>` a folder of your own outside the
repository.

```sh
export PATH=/c/mingw152/mingw64/bin:$PATH
py -3.13 -E -B tools/listings.py --prefix <prefix> --only ch07 --work <work>/listings
py -3.13 -E -B tools/build.py --chapter ch07 --pdf --out-dir <work>/build
powershell -NoProfile -ExecutionPolicy Bypass -File tools/pdfpages.ps1 -Pdf <work>/build/ch07.pdf -OutPrefix <work>/build/ch07
py -3.13 -E -B tools/listings.py --status
py -3.13 -E -B tools/build.py --draft --out-dir <work>/book
py -3.13 -E -B tools/build.py --book sample/book.yml --out-dir <work>/sample
```

The first runs your listings, each on a fresh copy of `src/wisdom/14900KF` named `wisdom`
(`VFFT_WISDOM_DIR=wisdom`), and says for each whether the store served it or it measured; a
program that measures pays those seconds on every run. The second builds your chapter alone
(with every other chapter's numbers), checks its PDF and writes `<work>/build/ch07.check.json`;
`pdfpages.ps1` turns its pages into PNG files you can look at. A full build without
`--out-dir` writes `how-to-use/VectorFFT-Guide.html` and `.pdf`; it runs once all chapters
exist, not while chapters are being written. Run a VectorFFT program only through the harness,
never without `VFFT_WISDOM_DIR` at a fresh copy of `src/wisdom/14900KF`.
