# Gate harness debt — what the full sweep cannot certify

Note, 2026-09-16, from the sweep run to certify planning-policy step 1
(`docs/design/planning_policy_design.md`). 27 pass, 1 fail, 2 unbuildable.
None of the three is a behavior defect; each is the harness or a gate's
include strategy. Recorded so a future sweep is not read as "green except
some noise".

## 1. Two gates cannot compile: the planner is no longer self-contained

`form_slot_gate.c` and `il_dp_overflow_gate.c` include
`planning/dp_planner_il.h` directly with only `<stdio.h>` before it. Since
the four-step (2026-09-15) that header references `vfft_k1fs_plan_t` and
`_k1fs_ctx`, which live in `oop/k1_fourstep.h`, which in turn needs
`include/vfft.h` (`vfft_dir_t`, `vfft_config_t`, `vfft_plan`),
`vfft_internal.h` (`struct vfft_plan_s`) and the 2D tier's statics.

Proven pre-existing: both fail identically in a tree with the policy step
reverted (nine errors). Adding `k1_fourstep.h` to the planner header's own
prerequisite list — the shape its siblings (`il2p.h`, `ztt.h`,
`il_flatdit.h`) already use — CASCADES into the public and internal
headers, so it is not the fix.

The fix is one of:
  a. the two gates adopt the TEXTUAL strategy (`#include "vfft.c"`, what
     `sp_ccol_decode_gate` does and `run_gates.py`'s `TEXTUAL` set exists
     for), or
  b. one `planning/prereq.h` that replays vfft.c's include order for any
     TU that wants a planner header, and both gates include it.
(b) is the better shape if more gates ever want the planner; (a) is two
lines. Either way it is a gate change, not a library change.

## 2. One gate the runner cannot invoke

`wisdom_cold_cell_gate` wants `--wisdir <scratch> --N <N> --phase 1|2
--out <dir>` and must run TWICE (phase 1 then phase 2) to mean anything.
`run_gates.py` has three argument styles (flag, bare, none) and no
two-run style. It has failed with its usage line for as long as the
styles have existed.

## 3. Fixed in passing (2026-09-16)

`k1_fourstep_gate` was added 2026-09-15 and never registered, so the
runner invoked it with no arguments and it printed usage. Registered
`("bare", True)` with a 900 s budget; passes.

## 4. Three gates have no source in the tree

`il2d_m1_gate`, `wisdom2_2d_gate`, `il2d_proto_gate` exist only as
prebuilt `.exe` files that predate the current `vfft.c`. The runner
rebuilds every gate first, so it reports them unbuildable and skips them.
Either the sources come back or the executables and their `ARGSTYLE` rows
go (the clean-library law says the latter unless someone wants them).

## 5. Found in passing (2026-09-16): an include swallowed by a comment

In `build_tuned/benches/bench_1d_vs_mkl.c` the `vfft.h` include opens a
three-line comment, and `#include "real_dispatch_config.h"` sits INSIDE it:

```
#include "vfft.h"               /* K=1 kind-4 cascade cells: public front door
#include "real_dispatch_config.h"
                                 * (vfft_create serves the banked route+chain
                                 * verdict). Requires build.py --vfft. */
```

So that header has never been included by the canonical bench. It is at
HEAD, not something this week introduced. NOT fixed here: including it may
change the bench's real-transform dispatch configuration, which is the
owner's call, and a bench that has been measured in this state should not
change quietly. Reported, not touched.
