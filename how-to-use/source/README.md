# The sources of the VectorFFT User Guide

This folder builds the guide, v0.7 beta, into two files one folder up:

- `how-to-use/VectorFFT-Guide.html`: the whole guide on one self-contained page (fonts,
  formulas and figures inside it, no JavaScript), with the contents in a sidebar;
- `how-to-use/VectorFFT-Guide.pdf`: the same pages printed on A4 by headless Chrome, with
  bookmarks, a contents page with page numbers, and running heads.

Every program in the guide is a file in `listings/`, or one of the four examples in
`how-to-use/examples/`, that is compiled against an installed VectorFFT and run before the
guide quotes what it printed. Writers follow `STYLE.md`.

## What the build needs

| Tool | Used for | Tested with |
| --- | --- | --- |
| Python 3.10 or newer, with markdown-it-py, PyYAML and Pygments | `tools/build.py` | Python 3.13.1, markdown-it-py 4.0.0, PyYAML 6.0.3, Pygments 2.19.2 |
| matplotlib | the figure scripts in `figures/` | 3.10.7 |
| pypdf, and pdftotext when present | the checks of the PDF | pypdf 6.16.2; pdftotext 4.00 from Git for Windows |
| Node.js | KaTeX, run once per build | 22.13.1 |
| Google Chrome | printing the PDF (Edge is not used) | 155 |
| CMake 3.16 or newer, Ninja, GCC | building the listings | CMake 4.3.1, MinGW-w64 GCC 15.2.0 |
| an installed VectorFFT | linking the listings (`find_package(VectorFFT)`) | the library built from this tree |

KaTeX and the fonts are in `vendor/`; nothing is fetched while building.

## Rebuild the guide

Run these from this folder (`how-to-use/source`). On Windows use Git Bash with the MinGW-w64
`bin` folder first on `PATH`; `py -3.13` starts Python 3.13 (on Linux, write `python3`).

```sh
# 1. compile and run every listing and the four examples against the installed library at <prefix>
py -3.13 -E -B tools/listings.py --prefix <prefix>

# 2. build the HTML and the PDF, and check them
py -3.13 -E -B tools/build.py
```

Step 1 builds in a folder outside the repository (`--work DIR` names it) and runs each
program on a fresh copy of `src/wisdom/14900KF` (`--cpu-folder` names another), so the
outputs never depend on a store the machine has gathered. It writes `listings/out/<name>.out`
(standard output), `<name>.err` (standard error) and `<name>.json` (the run). A program whose
output holds an absolute path or the user's name is refused. Run step 1 again after installing
a new build of the library: the build cannot see that the outputs came from an older one.

Step 2 fails when a listing changed after its last run, when an excerpt of an example no
longer matches `how-to-use/examples/`, when a reference names no label, or when a check of the
PDF fails; `_build/VectorFFT-Guide.check.json` lists the checks.

## Build one chapter

```sh
py -3.13 -E -B tools/listings.py --prefix <prefix> --only ch07     # the listings ch07_*
py -3.13 -E -B tools/build.py --chapter ch07 --pdf                 # _build/ch07.html and _build/ch07.pdf
```

A chapter build knows the numbers of every other chapter, so its references read as in the
book. `--draft` builds the whole book while chapters are missing, with a line in their place.
`tools/listings.py --status` lists the listings whose output is missing or out of date.

On Windows, `tools/pdfpages.ps1` renders PDF pages to PNG with the PDF renderer Windows ships:

```sh
powershell -NoProfile -ExecutionPolicy Bypass -File tools/pdfpages.ps1 -Pdf _build/ch07.pdf -OutPrefix _build/ch07
```

## The folders

| Path | What it holds |
| --- | --- |
| `book.yml` | the order of the book; it numbers the chapters |
| `chapters/` | the chapters, one Markdown file each |
| `listings/` | the programs, `listings/CMakeLists.txt`, and `out/`, what they printed |
| `figures/` | the figure scripts, their SVG files, and `style.py`, the figure helper |
| `sample/` | a one-chapter book that uses every construct once (not part of the guide) |
| `tools/` | the build, the listings harness, the checks, the design (`guide.css`) |
| `vendor/` | KaTeX 0.16.21 and the Computer Modern Unicode fonts, with their licences |
| `STYLE.md` | the conventions every chapter follows |

## The vendored files

- `vendor/katex-0.16.21/`: KaTeX 0.16.21, MIT licence (`LICENSE`), the copy that VS Code ships
  in its markdown-math extension, and the KaTeX fonts, SIL Open Font License 1.1
  (`fonts/OFL.txt`). `tools/katex_render.js` also loads a plain `katex.min.js` from npm.
- `vendor/cmu-0.7.0/`: Computer Modern Unicode 0.7.0 by Andrey V. Panov, SIL Open Font
  License 1.1 (`OFL.txt`): the upstream TrueType files of CMU Serif and CMU Typewriter Text,
  each wrapped in WOFF 1.0 with every table unchanged.

The HTML is the same for the same sources; the PDF also carries the date it was printed.
