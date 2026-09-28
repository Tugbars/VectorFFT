# Architecture of `src/core`

Diagrams and a text map of the library's source, generated from the code by
`src/tools/archgraph.py`. Everything under `generated/` is rewritten by that tool
and never edited by hand, so it always matches the tree it was generated from.

## What is here

| file | what it shows |
|---|---|
| [`generated/zones.md`](generated/zones.md) | The six zones and the include edges between them, checked against the dependency rules. Start here. |
| [`generated/folders.md`](generated/folders.md) | One diagram per zone: its folders and the includes among them. |
| [`generated/folders/`](generated/folders/) | One diagram per folder: its files, what each is for, and what they include. |
| [`generated/calls/`](generated/calls/) | Call graphs: `create.md` and `execute.md` from the entry points, then one per side of the layout fork (`create_split`, `create_il`, `create_real`, `execute_split`, `execute_il`, `execute_real`). |
| [`generated/map.md`](generated/map.md) | Plain text, no rendering needed: every file with its one-line role, its includes and who includes it. |
| [`generated/functions.md`](generated/functions.md) | Plain text: every function with its file, line, callees, the functions it references by name, and its callers. |
| [`generated/graph.json`](generated/graph.json) | The same data, for other tools. |

The diagrams are [Mermaid](https://mermaid.js.org/); GitHub, GitLab and VS Code
render them inline. The same diagrams are also rendered as images in
[`svg/`](svg/), same layout (`svg/zones.svg`, `svg/calls/create.svg`,
`svg/folders/split__real.svg`; the per-zone sections of `folders.md` are
`svg/folders-split.svg` and so on), for reading in any browser or image viewer.

## The zones in one paragraph

The library holds two complete FFT libraries side by side: **split** (separate
`re[]` and `im[]` planes) and **il**, interleaved (`z[]` of `(re, im)` pairs).
They never include each other. What both need lives in **common** (ABI types,
math, support, the wisdom store core, the plan struct). The **front door**
(`vfft.c`) is one translation unit: it validates a request and forks once into
`split/split_create.h` or `il/il_create.h` (and `*_execute.h` at execute time).
**bridge** is the one temporary place where the two layouts meet (1D real
transforms, until the interleaved real engine exists), and **wisdom2** holds
wisdom glue that spans both layouts (legacy readers, the migrator). The rules
are enforced by `src/tools/baseline/hygiene.py`.

## Regenerating

```
python src/tools/archgraph.py            # rewrite docs/architecture/generated/
python src/tools/archgraph.py --check    # exit 1 and list what is stale
```

To refresh the SVG images too:

```
python src/tools/archgraph.py --svg
```

This needs [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) and a Chromium:
set `MMDC` to its `mmdc`, put `mmdc` on PATH, or let the tool run
`npx -y @mermaid-js/mermaid-cli` (needs Node). Set `CHROME` to a Chromium binary
if puppeteer cannot find one. `svg/manifest.json` holds a hash of each diagram's
source, so `--check` also reports SVGs that need re-rendering, without needing
mermaid-cli itself.

`hygiene.py` runs the check as part of the dependency check, so a change that
moves or re-wires files reports stale graphs until they are regenerated.
Python 3.8+, standard library only.

## Asking for one part of the picture

```
python src/tools/archgraph.py --focus split/real        # a folder
python src/tools/archgraph.py --focus ztt.h             # a file
python src/tools/archgraph.py --focus bridge --depth 2  # a zone, two steps out
```

prints one Mermaid diagram of that part and its neighbours, plus the roles of the
files it was asked about. For functions:

```
python src/tools/archgraph.py --calls _vfft_split_create            # what it calls
python src/tools/archgraph.py --calls vfft_cs2pi_exact --up         # who calls it
python src/tools/archgraph.py --calls _vfft_il_execute --depth 2    # exactly two levels
```

Paste the output into any Mermaid renderer or a markdown file.

The call graph is best effort. It reads the source text, not the compiled program:
both sides of every `#if` count, and calls made through a function pointer or a
macro are not followed. Where a function is handed over by name (a thread-pool
task, a dispatch table, a plan's execute pointer) it shows as a dashed edge, which
is how most of the execute-time dispatch stays visible. Helpers called from six
or more places are left out of the call views and listed under each diagram.

## Using it with an AI assistant

`generated/map.md` is written to be read by a model: one line of purpose per
file and its full include lists, grouped by zone and folder; `generated/functions.md`
does the same for functions. To get a diagram of something the generated views
do not show (a data flow, the path one transform takes, what a change would
touch), give the assistant `generated/map.md` and `generated/functions.md`, with
`generated/zones.md` for the rules, and ask for a Mermaid diagram of it. For a
scoped question, the output of `--focus` or `--calls` is a smaller starting point.
The source headers each open with a comment that explains the file; the map's
roles are their first sentences.
