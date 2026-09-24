# src/wisdom/ — the wisdom store

This folder is what the library plans from. Every plan VectorFFT executes was
chosen by a race on a real machine, and the winner is kept here as a record:
the engine, the chain of radices, the tile, the row route, the thread split.
A `vfft_create` that finds its cell here builds the banked plan and never
measures; a create that misses races the candidates at `config.rigor`, banks
the winner, and serves it.

These files are **measured, not generated**. Nothing in the build produces
them, nothing can rebuild them, and a deleted row costs the race that made it.

## Files

| file | holds |
|---|---|
| `wisdom2_oop.txt` | 1D complex out-of-place verdicts, and every 1D complex order verdict (natural output) for both placements |
| `wisdom2_scr.txt` | 1D complex in-place scrambled-output chains, and the trig transforms (DCT, DST, DHT) |
| `wisdom2_real.txt` | 1D real-input and real-output verdicts: routes and factorizations |
| `wisdom2_prime.txt` | the prime route: Bluestein and Rader engine verdicts, including the inner transform each one raced |
| `wisdom2_2d.txt` | 2D verdicts: the column chain, the row route, the tile, the turn and skewed passes |
| `wisdom2_3d.txt` | 3D and higher-rank verdicts |
| `wisdom2_quarantine.txt` | written beside the shards when a record is rejected with a stated reason; append-only, never loaded |
| `<host>/` | another machine's store, same layout: `Zen4/` is the AMD Zen 4 calibration host |

Each shard opens with an `@meta` line naming the host it was raced on (host
tag, ISA, L1 size). A store opened on a different machine still serves its
structural verdicts (routes, chains), and the library says so once: the
placement-sensitive fields are only valid where they were measured. For full
performance on another host, race into that host's own directory and keep it
beside this one, as `Zen4/` is.

The record grammar, the key fields and the laws of the writer are in
[`src/core/wisdom2/README.md`](../core/wisdom2/README.md). Files are LF-only,
pinned by `.gitattributes`.

## How the library finds it

`vfft_wisdom_load(dir)` opens a store directory; `config.wisdom = NULL` means
the library opens one itself, resolved in this order:

1. `VFFT_WISDOM_DIR`, if set: opened writable, so a miss is banked and, with
   `config.wisdom_write = 1`, persisted;
2. otherwise the directory the build compiled in as `VFFT_WISDOM_DIR_DEFAULT`,
   which is this folder for an in-tree build: opened **read-only**, so a miss
   is raced and served but never written;
3. otherwise the current directory, read-only.

Only an explicit directory or the environment variable can bank. That is
deliberate: no process that merely runs the library can write the shipped
store.

## The frozen bundle is elsewhere

`spike_wisdom.txt`, `bluestein_wisdom.txt` and `c2r_path.txt` are the frozen
wisdom of the split library's stride family. They are not part of this store:
they stay in `src/dag-fft-compiler/generator/generated/`, where
`spike_wisdom.txt` is also a build input of `plan_executors.h`, and the
library reads them from there whatever store directory it is given
(`VFFT_FROZEN_WISDOM_DIR`, set by the build).

## Changing the store

- **Never point a racing tool at this folder.** The gauntlet, the gates, the
  benches and the probes all take a scratch copy; a create on a writable copy
  of this folder banks into it. The gauntlet makes that copy itself
  (`gauntlet/results/<run>/store/`).
- **New verdicts arrive by merge.** `python gauntlet/gauntlet.py run ... --merge`,
  or the `merge` verb on a finished run, copies a run's rows into these shards:
  a row with the same key replaces the shipped one, a new row is added, rows the
  run did not race are kept. The merge writes a backup beside each shard.
- **Re-measure, don't edit.** A verdict that looks wrong is re-raced
  (`--calibrate`, or `config.recalibrate = 1` on the cell) and merged; the file
  is never hand-edited, regenerated, or trimmed to tidy it.
- **Add a host, not a file.** Another machine's store goes in a subfolder with
  the same six shards, never in new file names.
