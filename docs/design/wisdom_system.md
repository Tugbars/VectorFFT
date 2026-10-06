# The wisdom system

**Status:** decided 2026-10-04 (owner). The six steps of the build order (§11) are
built. §12 lists what is still open. §12 lists what is still open. Scope: the wisdom system only. Engines, the split library and the
public defaults stay as they are (§9).

Wisdom is the record of race winners that `vfft_create` plans from. This document
declares how winners are measured, kept, stored per CPU, and served.

## 1. The rules

1. **A winner is always kept and always saved.** A create that races serves the winner
   and writes it to the store before it returns. Execution never waits on planning.
2. **A recalibration always overwrites** the cell's verdicts. On a row two plans share
   (the 2D real row: one chain for both directions), a recalibrate rewrites its own
   direction's verdicts in place and leaves the other direction's standing; only a
   chain that changes replaces the row.
3. **A row that no longer builds is empty.** The cell races from scratch and the winner
   replaces the row.
4. **One CPU, one folder.** A row is served only on the CPU identity that raced it.
   Nothing is borrowed from another CPU's folder.
5. **The library measures; tools call the library.** Pinning, the sibling guard, the
   priority raise and the measurement lock are library code. The gauntlet is a test
   tool: it compares the library against others and holds no measurement logic of its
   own.
6. **A measurement the library knows is bad is served, not saved:** a race that could
   not pin, or that timed out waiting for the measurement lock.
7. **A noisy race is the user's to fix:** they recalibrate.

## 2. The store

```
src/wisdom/
  new/        wisdom files with headers only: the next unknown CPU's folder
  14900KF/    the i9-14900KF's rows (today's root files, moved)
  Zen4/       the Zen 4 laptop's rows
```

- **Selection** (`common/wisdom/wisdom2_folders.h`). The library reads the identity
  stamp (§3) inside each folder's files and uses the folder whose stamp equals this
  CPU's identity. Folder names carry no meaning; a user may rename theirs.
- **No match.** The library uses `new/` and stamps it with this CPU's identity at the
  first save. Every plan for that CPU is stored there from then on.
- **`new/` already claimed by another CPU, and no match.** The library creates a folder
  named after this CPU's identity (its values joined by `-`). A save never writes a
  file that another CPU stamped after the store was opened.
- **A twin of one of our CPUs** matches our folder, is served our rows, and saves its
  own races there. `new/` stays empty.
- **Our machines** always match their own folders, so `new/` stays empty in the repo.
- **`VFFT_WISDOM_DIR`** keeps its meaning: the directory it names is the store, with
  no scan, and so is a directory passed to `vfft_wisdom_load`. Tests and probes use it
  with a scratch copy. A named store stamped before the identity carried caches and
  core counts (`host= isa= l1d=` only) with this host and ISA is this CPU's older
  stamp: the first save replaces it. A named store raced on another CPU is served as
  it is and said once.
- When no folder matches, one line names the identity that was looked for and the
  folder taken.
- `vfft_wisdom_folder()` returns the folder in use and `vfft_wisdom_identity()` this
  CPU's identity (`vfft.h`). Tools ask them (`gauntlet/wisdom_folder.py`, through
  `recal_1d_probe --where`): the gauntlet seeds a run's store from this CPU's folder
  and merges into it.

## 3. The CPU identity

| field | example (14900KF) | example (Zen 4 8640HS) |
| --- | --- | --- |
| vendor and model tag | `intel-f6m183` | `amd-f25m117` |
| build ISA | `avx2` | `avx2` |
| P-core L1d | 48 KiB | 32 KiB |
| P-core L2 | 2 MB | 1 MB |
| L3 | 36 MB | 16 MB |
| P-core count | 8 | 6 |
| E-core count | 16 | 0 |

- Two machines share rows only when every field is equal: a 14900K, a 14900KF and a
  13900K match; an i7-14700K (12 E-cores, 33 MB) and an i5-14600K (6 P-cores, 24 MB) do
  not. A threaded verdict depends on the core counts and the four-step's admission on
  L3, so both are part of the identity and no row carries a per-row condition.
- The cache sizes are read from the CPU on a P-core, never taken from a build-time
  constant: the first read moves the calling thread to a P-core it is allowed and
  gives it back (`common/support/cpu_cache.h`). The stamp is the `@meta` line of every
  shard: `host= isa= l1d= l2= l3= pcores= ecores=` (`common/support/cpu_identity.h`).
- The core counts come from the machine's topology, not from the process's allowed set,
  so a restricted affinity mask does not change the identity.
- Every field has an AMD path where the concept applies (`cpu_cache.h` already reads
  AMD's cache leaves); the Zen 4 laptop is the test host.

## 4. A create

1. **Look up** the cell's row in this CPU's folder.
2. **A row is served only if it builds exactly as written**: route, chain, tile and
   every kernel form resolve in this build. Otherwise the row is empty (rule 3) and one
   line names it.
3. **Hit:** build the plan from the row. No race, no lock, no pin.
4. **Miss or recalibrate:** enter the race scope (§5), race the cell's full candidate
   pool, bank the winner in the loaded table, save it (§6), leave the scope.
5. **Build the plan from the banked row**, the same path a hit takes, so a plan built
   after a race is the plan a later hit builds.

`vfft_execute` is unchanged: no allocation, no measurement, no store access.

## 5. The race scope

One scope surrounds every race in the process (`common/support/race_scope.h`). It is
entered at the first clock read inside a create and left when the outermost
`vfft_create` returns, so no race site can miss it and no exit path can leak it.

| part | rule |
| --- | --- |
| **measurement lock** | One lock for the machine, so two processes (or two threads) never race at once: a named mutex on Windows, a locked file in `/tmp` on Linux, both released by the OS when the holder dies. A hit takes none. The wait is bounded (60 s by default); after it the race runs anyway and its winner is served, not saved. |
| **pin** | The racing thread is pinned to the caller's own core in the library's layout: the second P-core (logical CPU 2 on a hyperthreaded part) when the process has no workers, logical CPU 0 when it has (worker 1 spins on the second P-core, `common/support/threads.h:176`). Only a P-core the process is allowed is taken; the others are walked in order. P-cores, E-cores and siblings are read from the OS (`common/support/cpu_topology.h`), on Intel and AMD alike. A thread that ends on no P-core races where it is, and the winner is served, not saved. |
| **sibling guard** | A guard thread holds the pinned core's hyperthread sibling for the race, so the OS cannot park another thread there (1.1-1.5x slower without it). It waits with TPAUSE (Intel) or MWAITX (AMD), which cost the racing thread nothing; a part with neither races unguarded. No guard goes to a CPU outside the set the process is confined to. |
| **priority** | The racing thread's priority is raised for the race. Best effort: where it cannot be raised (Linux without `CAP_SYS_NICE`), the winner is still saved. |
| **restore** | The guard ends, and affinity and priority return to what they were, except that a caller pinned to logical CPU 0 by a pool created during this create stays there, as today. |

- The same scope is public in `vfft.h`: `vfft_measure_configure()` sets it up (pin,
  guard, priority, the lock wait), `vfft_measure_begin()` and `vfft_measure_end()` put
  it around a caller's own timing, `vfft_measure_confine()` confines a process to one
  logical CPU per P-core for a threaded comparison, and `vfft_measure_describe()`
  reports it. A create inside a caller's scope adds nothing to it. The gauntlet holds no
  pin or guard code: `gauntlet/bench_scope.h` maps its switches onto these calls.
- Not covered: the application's own threads on the race core, and other software's
  load. That is the user's responsibility.
- Threaded plans keep today's layout: the caller on core 0, workers on cores 2, 4, ...

## 6. Saving

- **Always.** The read-only default is gone: the bank always enters the loaded table
  (the guard leaves `vw2_bank`, `common/wisdom/wisdom2.h:1292`) and the save runs
  before create returns. The 3D tier banks like the others (`il/rank3/fftnd_il.h:1688,
  1786` gate it on the write flag today).
- **Under a store lock.** A save re-reads the file, merges only the rows this process
  banked, writes a temporary file and renames it over the old one. Two processes saving
  at once lose no rows; a killed holder's lock is released by the OS.
- **Not saved** (rule 6): a race whose pin failed, or that timed out on the measurement
  lock. Both are served from memory for the process.
- **Unwritable store** (a read-only directory): the winner is served from memory and one
  line says why it was not saved.
- **The off switch.** `config.wisdom_write` stays in the struct and is ignored. One
  environment variable turns saving off for a process. It is for rare cases; tests run
  with saving on against a scratch copy, because the save and read-back path is where
  wisdom defects show.

## 7. Rows

- **The build stamp.** Every banked row carries `bld=<version>-<commit>`, for example
  `bld=0.1.0-358acb83`: the library version and the commit of the build that raced it.
  The commit is the last one that touched the library's sources (`src/core`,
  `include`), so the id moves only when the library does. A build without git carries
  the version only; a build from changed sources is marked `-dirty`. The store stamps
  the row at the bank (`vw2_set_build`, `common/wisdom/wisdom2.h`); the build supplies
  the commit (`gauntlet/build.py` generates `vfft_build_id.h`, CMake defines
  `VFFT_BUILD_COMMIT` at configure time). Rows are served whatever their stamp.
  `vfft_wisdom_report()` reports the folder in use, its identity, this build's id
  (`vfft_wisdom_build()`) and the rows counted by build, and lists the rows of one
  build, so older rows can be found and re-raced (`recal_1d_probe --report [BUILD]`).
  Rows banked before this change carry no stamp and count as older than any build.
- **Child recipes** (`il/wisdom/wisdom2_child.h`). A 2D or 3D row carries the recipe
  of each child plan it runs, raced inside the 2D/3D create in the child's own role, on
  a private store (in memory, never saved). A 2D/3D create reads and writes no 1D row;
  a child's clones are created against the same store. Every row the child's store
  holds rides on the parent row under a prefix: row i as `<pre>k<i>=<its key, spaces as
  commas>` and `<pre><i>_<token>=<value>` for each of its payload tokens, so a route
  row, a backward twin, a prime method's row or a four-step child's `fs_` tokens all
  come back whole, and a 2D row that carries its own children nests under a 3D row's
  prefix. The parent row is re-banked (and saved) when a child raced in the create or
  the row carried no recipe yet; a hit stamps nothing.

  | prefix | child | built at |
  | --- | --- | --- |
  | `rp_` | the row plan at N2; the 3D row plan at N3 (`row_` would collide with the four-step's `fs_row_`) | `il/rank2/fft2d_create_il.h`, `il/rank3/fftnd_il.h` `_ilnd_build_flat` |
  | `turn_` | the turn plan at N1 | `fft2d_create_il.h` |
  | `csk_` | the skewed pass's row plan | `fft2d_create_il.h` |
  | `tpc_` | the turned prime column plan | `il/rank2/il2d_tier.h` `_il2d_tpc_build` |
  | `plane_` | the 3D tier's 2D child (with its own `rp_` and friends inside) | `fftnd_il.h` `_ilnd_build_child` |
  | `rp_` / `rp_c2r_` | the real 2D door's row child: the c2c plan at an odd N2, else the real batch at N2 x N1 (by the plan's direction: the two differ) | `fft2d_create_il.h` (the real branch) |
  | `rx_` / `rx_c2r_` | the real 2D door's per-row engines, by the plan's direction: a zr2c engine's child recipe (the row engines' own store); the odd engines (zrf, zrb) carry their recipe in the `rx=` value itself | `il/rank2/il2d_real_plan.h` `_il2d_real_rowplan` / `_c2r` |

- The four-step's child codec follows: the 2D child row under `fs_` carries its row
  plan itself (`fs_rp_*`), so `fs_row_` / `fs_row_bwd_` are no longer written and are
  read as optional. A row from before (twelve in the 14900KF store) completes its child
  once, at its next create, and is re-banked.
- Rows from before this change carry no child recipe: the first create of such a 2D or
  3D cell races its children in role once and re-banks the row. The format change is
  additive: a binary that predates it skips the tokens it does not know.

## 8. Door changes

| door | today | becomes |
| --- | --- | --- |
| every tier | a read-only store refuses the bank; the 1D c2c door then serves its structural pair (827 ns at N=1024 against the winner's 718), the real and 2D doors race again on every create | the winner is banked, saved and served (§6) |
| 1D c2c, both placements | a row that will not build falls to a default chain, an unraced prime cell or a refusal; a form the build lacks leaves the default kernel in place (`il/rank1/k1_commit.h:60-105`); the two-order pick banks an unmeasured pair (`k1_commit.h:940-960`) | the row is empty: full race, winner saved; the order pick banks nothing |
| zr2c | the route race keeps the structural route unless the other is 3% faster (`il/real/zr2c_build.h:756-757`) | the faster route wins |
| 2D, 3D c2c | children are 1D creates on the caller's store | children raced in role, recipes on the parent row (§7) |
| 2D real | the row child (odd N2: a 1D c2c create; else a 1D real batch create at N2 x N1) and the per-row engines are built on the caller's store | children raced in role, recipes on the real row (`rp_` / `rp_c2r_`, `rx_`; §7) |

The real doors already treat a row that no longer builds as a miss
(`il/real/zrp_build.h:952`, `zr2c_build.h:636`, `odd_build.h:24`).

## 9. What does not change

- The split library, including its single-transform (K=1) doors and the zeroed-config
  default. Split needs 4 transforms on AVX2 and 8 on AVX-512 to vectorize; its K=1
  machinery stays as it is.
- The pause between races, the candidate pools, the sample definitions and every
  engine.
- `vfft_execute`, and the caller pin of threaded plans.
- No borrowing, no profiles or packs, no command-line tool, no per-user state folder.

## 10. Cache sizes

The planner acts on the measured sizes: discovery (`cpu_cache.h`) is the default. The
read is taken on a P-core (the calling thread is moved there for it and given back), so
an E-core's caches never size anything. A read that cannot be taken on a P-core, or
whose geometry contradicts its label, sizes from the fallbacks (32 KiB, 1 MB): an
undershoot degrades gracefully, an overshoot does not. `-DVFFT_L1D_DISCOVER=0` pins the
old constants (48 KiB, 2 MB). On the 14900KF the measured sizes equal them.

## 11. Build order

Each step ends with: a build on Windows gcc and Linux gcc, an ICX compile, the
existing probes, a step-specific check, and the gauntlet binaries rebuilt. Checks run
with saving on, against a scratch copy of the store. Nothing is committed without the
owner's review.

| step | contents | check |
| --- | --- | --- |
| 1 (built) | keep the bank, save by default, the store lock and merge-own-rows save, the off switch, `vfft.h` text | per family: a cold cell is raced once, is on disk, and a second process replays it with 0 races, bitwise; a read-only directory serves from memory; two processes saving at once lose no row; a killed lock holder does not block |
| 2 (built) | a row that does not build is empty (1D c2c doors); the order pick stops banking; zr2c's bias removed | rows naming a missing form, chain and tile each re-race and are replaced on disk |
| 3 (built) | the race scope in the library (lock, pin, guard, priority, restore), public in `vfft.h`; the gauntlet calls it and `sibling_guard.h` goes | affinity and priority equal before and after create at every door, races forced; a clock read during create outside the scope fails the check; two racing processes take turns; an unpinnable process serves and does not save |
| 4 (built) | the CPU identity, per-CPU folders, `new/`, the 14900KF move, measured cache sizes; gauntlet and tool paths follow | this machine selects its folder; a forged identity selects `new/` and stamps it; a claimed `new/` leads to a created folder; the 14900KF's picks are unchanged by the measured sizes |
| 5 (built) | the build stamp and the report call | every new row carries the stamp; the report groups rows by build |
| 6 (built) | 2D and 3D children raced in role, the real 2D door's included | a cold 2D and 3D create leaves the 1D files byte-identical; replay is bitwise with 0 races; a four-step row from before completes its child once and replays |

## 12. Open

- **The version number.** `vfft_version()` returns 0.1.0 (`common/support/env.h:118`),
  `CMakeLists.txt:28` says 2.0.0 and the results document is titled v1.0. The stamp
  uses `vfft_version()`.
- **Names, proposed (the owner may change them):** folders `new/` and `14900KF/`; the
  token `bld=`; the prefixes of §7; `VFFT_WISDOM_WRITE=0` (saving off);
  `vfft_measure_configure()`, `vfft_measure_begin()`, `vfft_measure_end()`,
  `vfft_measure_confine()` and `vfft_measure_describe()` for the race scope (the lock
  wait is a field of its configuration, not an environment variable);
  `vfft_wisdom_folder()` and `vfft_wisdom_identity()` for the folder in use and the
  identity; `vfft_wisdom_report()` for the rows-by-build report.
- **The Zen 4 folder's stamp is written from the documented readings** of the laptop
  (32 KiB, 1 MB, 16 MB, 6 cores: `docs/performance/v1_0_Zen4_results.md`), not read on
  it. If the laptop reads another value it takes `new/` and says which identity it
  looked for; its rows were raced with the 48 KiB constant and are re-raced there.
- **A virtual machine is its own CPU.** WSL on the 14900KF presents 16 P-cores and no
  E-cores, so the library gives it a folder of its own.
- **Not ruled:** threaded plans leave the caller pinned to core 0 permanently
  (`vfft.c:238-244`), documented only at `vfft_set_num_threads`.
