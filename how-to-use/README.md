# VectorFFT User Guide

This folder holds the user guide of VectorFFT v0.6 beta. The guide shows you how to build and
install the library, how to call it from C and C++ through `vfft.h`, and how to move a program
written for Intel MKL's DFTI interface to it. Every program in the guide was compiled against
the installed library and run before the guide quotes what it printed.

## What the folder holds

| Path | What it holds |
| --- | --- |
| `VectorFFT-Guide.html` | the guide on one page, with its fonts, formulas and figures inside; open it in a browser |
| `VectorFFT-Guide.pdf` | the same guide printed on A4, with bookmarks and a contents page |
| `examples/` | the four 3D programs of Part III, and the `CMakeLists.txt` that builds them |
| `source/` | what the guide is built from: the chapters, the programs, the figures and the build |

## Read the guide

Open `VectorFFT-Guide.html` in a browser, or `VectorFFT-Guide.pdf` in a PDF reader. The guide
has five parts:

| Part | Chapters | What you do there |
| --- | --- | --- |
| I Getting started | 1 and 2 | build and install the library, and run a first transform |
| II Using the API | 3 to 11 | describe a transform, run plans, size buffers, transform real data and batches, use threads, keep wisdom, read errors |
| III Worked examples | 12 to 15 | follow the four programs of `examples/`, one chapter each |
| IV Moving a DFTI program | 16 and 17 | build a DFTI program against VectorFFT, and see what the interface serves |
| V Reference | 18, 19, A and B | look up `vfft.h`, the environment variables, notes for Windows and MinGW, and troubleshooting |

If you are new to the library, read Chapters 1 and 2, then the chapters of Part II that your
program needs.

## Build and run the examples

Install the library first, as Chapter 1 of the guide describes, and note the folder you
installed it to, `<prefix>`. Then build the four examples against it, from the root of your
VectorFFT clone:

```sh
cmake -S how-to-use/examples -B build-examples -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build-examples
```

Run each one from the build folder:

```sh
cd build-examples
./poisson3d
```

In PowerShell, run `.\poisson3d.exe`. Each example checks its own result, and its last line ends
in `PASS`. The library keeps the wisdom it measures in its wisdom folder,
`<prefix>/share/vectorfft/wisdom/`, which holds one folder for each CPU, and it picks the folder
of your CPU by itself. On a CPU without shipped wisdom, the first run of an example measures
each configuration it creates, for seconds, up to about a minute for some configurations, and
saves the winners there; later runs are served at once. Saving needs a `<prefix>` you can
write: under a prefix only an administrator can write, nothing is saved and every run measures
again (Chapter 10 of the guide). On Windows, a program built with MinGW-w64 needs
`libwinpthread-1.dll`, from the compiler's `bin` folder, on `PATH` or beside it, unless it is
linked with `-static` (Appendix A of the guide).

## Rebuild the guide

The guide is built from `source/`: Markdown chapters, C programs that a harness compiles against
an installed VectorFFT and runs, and a Python build that makes the HTML page and prints the PDF
with headless Chrome. From `how-to-use/source`, two commands rebuild both files (on Linux, write
`python3` for `py -3.13`):

```sh
py -3.13 -E -B tools/listings.py --prefix <prefix>
py -3.13 -E -B tools/build.py
```

The first runs every program of the guide on a fresh copy of the shipped wisdom of one CPU
(`src/wisdom/14900KF`), so the wisdom folder stays as it is, and keeps what each printed; the
second builds `VectorFFT-Guide.html` and `VectorFFT-Guide.pdf` in this folder and
checks them. `source/README.md` lists the tools the build needs, and `source/STYLE.md` gives
the rules every chapter follows.
