# src/wisdom/ — the wisdom store

This folder is what the library plans from. Every plan VectorFFT executes was
chosen by a race on a real machine, and the winner is kept here as a record:
the engine, the chain of radices, the tile, the row route, the thread split.
A `vfft_create` that finds its cell here builds the banked plan and never
measures; a create that misses races the candidates at `config.rigor`, banks
the winner, saves it here, and serves it.

These files are **measured, not generated**. Nothing in the build produces
them, nothing can rebuild them, and a deleted row costs the race that made it.

## One CPU, one folder

A race winner is a measurement of the machine that raced it, so the store keeps
one folder per CPU and a row is served only on the CPU that raced it. Nothing is
taken from another CPU's folder.

| folder | holds |
|---|---|
| `14900KF/` | the rows raced on the Intel i9-14900KF |
| `Zen4/` | the rows raced on the AMD Zen 4 laptop (Ryzen 5 PRO 8640HS) |
| `new/` | the six shard files with headers only: the folder of the next CPU this store has not seen |

**A folder belongs to the CPU its files are stamped with**, not to its name.
Each shard opens with an `@meta` line carrying the CPU identity:

```
@meta host=intel-f6m183 isa=avx2 l1d=49152 l2=2097152 l3=37748736 pcores=8 ecores=16
```

vendor and model, the build's instruction set, the P-core's L1d and L2 and the
L3 in bytes, and the number of P- and E-cores. Two machines share a folder only
when every field is equal: a 14900K, a 14900KF and a 13900K are one identity; an
i7-14700K (12 E-cores, 33 MB) is another. `vfft_wisdom_identity()` returns this
machine's, `vfft_wisdom_folder()` the folder in use.

The library selects the folder by itself:

1. the folder whose stamp equals this CPU's identity;
2. otherwise `new/`, while it is unstamped: the first saved winner stamps it,
   and it is this CPU's folder from then on. Rename it as you like;
3. otherwise (`new/` already belongs to another CPU) a folder named after the
   identity, created on the spot.

When no folder matched, one line names the identity that was looked for and the
folder taken. On the machines above the library always finds its own folder, so
`new/` stays empty in the repository.

## Files

Every folder holds the same shards:

| file | holds |
|---|---|
| `wisdom2_oop.txt` | 1D complex out-of-place verdicts, and every 1D complex order verdict (natural output) for both placements |
| `wisdom2_scr.txt` | 1D complex in-place scrambled-output chains, and the trig transforms (DCT, DST, DHT) |
| `wisdom2_real.txt` | 1D real-input and real-output verdicts: routes and factorizations |
| `wisdom2_prime.txt` | the prime route: Bluestein and Rader engine verdicts, including the inner transform each one raced |
| `wisdom2_2d.txt` | 2D verdicts: the column chain, the row route, the tile, the turn and skewed passes |
| `wisdom2_3d.txt` | 3D and higher-rank verdicts |
| `wisdom2_quarantine.txt` | written beside the shards when a record is rejected with a stated reason; append-only, never loaded |

The record grammar, the key fields and the laws of the writer are in
[`src/core/wisdom2/README.md`](../core/wisdom2/README.md). Files are LF-only,
pinned by `.gitattributes`.

## How the library finds it

`vfft_wisdom_load(dir)` opens the store directory it is given, as it is.
`config.wisdom = NULL` means the library opens its own store:

1. `VFFT_WISDOM_DIR`, if set: that directory is the store, with no selection;
2. otherwise this CPU's folder under the root the build compiled in as
   `VFFT_WISDOM_DIR_DEFAULT`, which is this folder for an in-tree build;
3. otherwise no store: winners are kept in memory for the process.

A create that races saves its winner before it returns, under the store lock
(`wisdom2.lock`). `VFFT_WISDOM_WRITE=0` turns saving off for a process. A
directory named by the caller that was raced on another CPU is served as it is,
and the library says so once.

## The frozen bundle is elsewhere

`spike_wisdom.txt`, `bluestein_wisdom.txt` and `c2r_path.txt` are the frozen
wisdom of the split library's stride family. They are not part of this store:
they stay in `src/dag-fft-compiler/generator/generated/`, where
`spike_wisdom.txt` is also a build input of `plan_executors.h`, and the
library reads them from there whatever store directory it is given
(`VFFT_FROZEN_WISDOM_DIR`, set by the build).

## Changing the store

- **Tests, gates, benches and probes take a scratch copy.** They point
  `VFFT_WISDOM_DIR` at a copy of this CPU's folder; the gauntlet makes that
  copy itself (`gauntlet/results/<run>/store/`). Tools ask the library which
  folder is this CPU's (`python gauntlet/wisdom_folder.py`).
- **A library that is simply used saves here.** A `vfft_create` with no store
  named races a missing cell once and saves the winner into this CPU's folder.
- **Gauntlet verdicts arrive by merge.** `python gauntlet/gauntlet.py run ... --merge`
  (or the `merge` verb on a finished run) copies the run's rows into this CPU's
  folder: a row with the same key replaces the shipped one, a new row is added,
  rows the run did not race are kept. The merge writes a backup beside each shard.
- **Re-measure, don't edit.** A verdict that looks wrong is re-raced
  (a gauntlet run of the cell, or `config.recalibrate = 1` on it); the file
  is never hand-edited, regenerated, or trimmed to tidy it.
- **Add a CPU, not a file.** Another machine's rows go in that machine's folder
  with the same six shards, never in new file names.
