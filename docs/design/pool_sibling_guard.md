# The hyperthread siblings of the cores the library runs on

**Scope.** Why a threaded plan can run, and be raced, about twice as slow as it should on an ordinary desktop,
and how the library prevents it by holding the hyperthread sibling of every core it uses, in planning (the
races) and in execution (the pool). The guards are `src/core/common/support/sibling_guard.h`; the pool is
`src/core/common/support/threads.h`; the races run inside the measurement scope,
`src/core/common/support/race_scope.h`; the sibling of a logical CPU comes from `vfft_topo_sibling`
(`src/core/common/support/cpu_topology.h`). Companion: how an execute pins its calling thread
(`thread_pool_caller_arm`, in `threads.h`).

---

## 1. The problem

The pool pins each worker to one logical CPU, one per performance core (the pin stride is the SMT width: on an
i9-14900KF the workers sit on logical CPUs 0, 2, 4, ..., 14), and an execute pins its calling thread to the core
the pool left for it. The other logical CPU of each of those cores (1, 3, ..., 15) stays open to every thread on
the machine.

When another program's thread runs on the sibling of one of our cores, our thread on that core shares its
execution units and runs at about half speed. A threaded execute is a sequence of fork-joins over an equal,
fixed split, so every phase waits for that one thread and the whole transform runs about 2.4 times slower. A
hard-pinned worker cannot move away: the slow state lasts until the other thread leaves the sibling, from a few
milliseconds to the whole life of the process.

Measured on the i9-14900KF at 8 threads, the engine alone in fresh processes (no other library loaded):

| Plane | Most processes | The rest |
|---|---|---|
| 15x4375 | 73 us | 31 us |
| 12x750 | 13 us | 8 us |
| 625x125 | 88 us | 61 us |

Within one process the executes run slow from the first one and switch to fast at a random point (after 0 to
more than 2000 executes), or never. With every sibling held (section 3) all six processes ran fast from the first
execute (27-33 us at 15x4375); without, two of three ran slow. MKL's threaded team, which Windows may move between
cores, was steady in the same conditions. In the 485-plane comparison against MKL at 8 threads (2026-10-30),
23 to 36 planes ran about twice as slow in one of the two engine orders for this reason.

Ruled out: the engine order and MKL's spinning threads (`KMP_BLOCKTIME=0` leaves it), the pool's parking (workers
spin for 200 ms between dispatches), the plan (identical fields in slow and fast processes), and the time since
the process started (a 1 s pause before the first execute leaves it).

## 2. The options measured

Fresh processes not confined to any CPU set (a user's process), the plan's workers changed after create, six
processes per cell, three planes, the machine as it is and with an added background load of 8 or 24 threads that
are not pinned and alternate 5 ms busy and 5 ms asleep. Median process / worst process, in microseconds:

| Plane | Load | Current pool | Siblings held | Workers free over the 16 P-core CPUs | Free over one CPU per P-core | Free over every CPU |
|---|---|---|---|---|---|---|
| 15x4375 | none | 35 / 74 | **28 / 31** | 34 / 34 | 34 / 35 | 35 / 60 |
| 625x125 | none | 46 / 47 | **40 / 40** | 48 / 50 | 45 / 46 | 51 / 77 |
| 1500x750 | none | 693 / 879 | **590 / 613** | 1064 / 1088 | 714 / 870 | 763 / 1569 |
| 15x4375 | 24 threads | 36 / 37 | **33 / 33** | 38 / 41 | 35 / 38 | 38 / 45 |
| 625x125 | 24 threads | 49 / 54 | **40 / 41** | 47 / 50 | 46 / 49 | 46 / 55 |
| 1500x750 | 24 threads | 718 / 749 | **574 / 608** | 945 / 1047 | 722 / 760 | 706 / 1040 |

- **Siblings held**: the fastest in every cell and never slow (54 of 54 processes). It is also 10-20% faster than
  the current pool in its fast state, because other threads visit the siblings briefly even outside the long
  slow spells.
- **Free over the 16 P-core CPUs**: Windows places two workers on the two CPUs of one core; the large plane runs
  at 945-1064 us.
- **Free over one CPU per P-core**: the same as the current pool, slow state included.
- **Free over every CPU**: workers land on efficiency cores; the worst.
- **Holding the siblings only inside the benchmark** gives the "siblings held" numbers in the benchmark and leaves
  users, and the library's own races, with the current pool's behaviour. It is not the solution.

The races suffer the same way. The measurement scope holds the sibling of the racing thread's core only; a race at
T > 1 times its arms on the pool's workers, whose siblings are open, so a threaded verdict can be banked from a
contended arm. Re-racing 74 planes at 8 threads with every sibling held (2026-10-30) moved 10 of them from a banked
one-thread plan to the strips, and fresh guarded races on 58 other planes picked chains up to 1.7 times faster
than the banked ones (8192x256 at one thread: 16.8.8.8, 20.2 ms, against the banked 32.32.8, 35.6 ms).

## 3. The solution: the library holds the siblings of every core it uses

The same guards serve both places, so a plan is chosen under the conditions it is served under. They are one
table (`src/core/common/support/sibling_guard.h`): at most one guard thread per logical CPU, kept by whoever
holds it.

**In planning.** The measurement scope holds the sibling of the racing thread's core, as before. Once the
scope uses the pool -- a race in it forks onto the pool or grows it, or a caller's own scope
(`vfft_measure_begin`) begins while the pool lives -- it holds the siblings of every core the pool uses (the
caller's reserved core and each worker's core) until the scope ends: no guard parks between two arms, and
none is missing while another engine is timed in the same scope. It goes on only once every guard is up
(bounded at some tens of milliseconds), so the first threaded arm runs guarded too. A one-thread race that
neither forks nor grows the pool keeps the single guard.

**In execution.** For every core the pool runs on (each worker's core and the core it leaves for the caller),
the pool keeps one guard on that core's sibling (`vfft_topo_sibling`). A guard follows its worker (worker 0 for
the caller's core): it holds the sibling while the worker is awake -- working, or spinning for the block time
between dispatches -- and parks within one wait (about 35 us) after the worker parks. A dispatch that wakes a
parked worker wakes the worker only; the worker wakes its guards on its own thread, so a dispatch pays nothing
for them. An idle program holds nothing. The guards end with the pool's workers.

**Tools.** The gauntlet and every other tool time the library through the same scope and get the same guards;
nothing exists only in a tool. The library's confinement helper (`vfft_measure_confine`, used by the gauntlet's
bench) confines a process to every logical CPU of the performance cores (mask 0xFFFF on the i9-14900KF): the
siblings stay in the process's set so the guards can hold them, and the pool still takes one logical CPU per
core. A scope with the guard turned off (`vfft_measure_configure`, `VFFT_MEASURE_GUARD_OFF`) parks the pool's
guards for the scope: the control that measures the pool without them.

**The guard.** A loop of timed waits, `TPAUSE` on Intel (`_vfs_tpause_slice`: about 35 us in the light C0.2
state per slice) and `MONITORX`/`MWAITX` on AMD (`_vfs_mwaitx_slice`), until it is told to park or to stop. To
the operating system the sibling is busy, so the scheduler places other threads elsewhere; to the core the
waiting sibling takes almost no execution resources, so our thread on that core runs at full speed.

No guard starts:

- inside a virtual machine (section 4): CPUID leaf 1 ECX bit 31 set with another hypervisor's vendor (leaf
  0x40000000), or Hyper-V without the right to create partitions (leaf 0x40000003, EBX bit 0). A Windows host
  runs as Hyper-V's root partition, holds that right, and keeps its guards;
- where the core has no sibling (SMT off, a core without hyperthreading);
- where the sibling is outside the process's affinity mask (an application that confined the process keeps its
  choice);
- on a CPU a pool worker uses (the layout of a pin-stride override);
- where the CPU has neither timed wait instruction. A PAUSE spinner, which the measurement settings can ask
  for, holds the racing thread's sibling alone.

**Cost.** One extra thread per guarded core (8 on the i9-14900KF at 8 threads). While they hold them -- while
the workers are awake, and in a measurement scope that uses the pool -- the siblings are unavailable to other
programs (on the i9-14900KF they keep the 16 efficiency cores and the cores the pool does not use), and the
guarded cores stay in the light C0.2 state instead of a deep sleep state.

## 4. Open points

- **The first executes after a park.** A parked guard comes back when its worker does; when another program's
  thread runs on its CPU, it waits for the scheduler. On a busy desktop a woken guard took up to 6.8 ms to get
  its CPU back (2026-10-30, a functional check, not a timing), and the first executes after a park run
  partly unguarded until it does. The alternative holds the guards for the whole life of a threaded plan: no
  such window, at the cost of power and the siblings while the program is idle. To choose, measure the first
  executes after a park with and without a background load. The measurements in section 2 ran with the
  guards held throughout.
- **Virtual machines.** Under WSL2 (Linux 6.6 in a Hyper-V virtual machine on the same i9-14900KF, 32 virtual
  CPUs reported as 16 cores x 2) the Linux form works: the guards pin to the siblings and wait in `TPAUSE`. But
  there it makes the pool slower: 15x4375 at 8 threads 62-64 us in 6 of 6 guarded processes against 30-33 us
  (4 of 6) or about 60 (2 of 6) unguarded; 625x125 73-112 against 65-69. A guest's virtual CPUs are host threads
  that the host moves between cores, so a guest's sibling is not a core's sibling, and a waiting guard still
  occupies a host CPU. So no guard starts inside a virtual machine (section 3); under WSL2 the library detects
  it and starts none.
- **Other machines.** The guards are on for bare-metal Linux and for AMD (`MWAITX`), where the hardware cause
  is the same, but neither has been measured; `src/core/common/support/README.md` carries the warning that they
  must be measured there. Intel parts without `WAITPKG` (before Alder Lake and Sapphire Rapids) have neither
  wait instruction: the pool has no guards there.
- **No switch outside a measurement scope.** An application cannot turn the pool's guards off; on a platform
  where they do not help, turning them off is a change in the library.

## 5. Evidence

`docs/research/estimate_threaded/evidence/nd/mt2d/`: `pool_opts.c` (the five pool variants on one plan),
`bgload.c` (the background load), `run_pool_opts.sh`, `pool_opts_report.txt` and `pool_opts_{none,light,heavy}.txt`
(section 2), `ours_timeline.c` (the per-process timelines and the first sibling test of section 1).
`docs/research/estimate_threaded/evidence/guard/`: `guard_check.c` (the built guards on Windows: where they
sit, parking and waking with the workers, the scope's hold, the guard turned off, bitwise output, the end with
the pool, the confinement masks) and `guard_linux.c` (under WSL2: no guard in the virtual machine; the Linux
park, wake, hold and end driven directly).
