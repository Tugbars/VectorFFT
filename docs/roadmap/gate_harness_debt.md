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

## 4. The sweep is the CERTIFICATION gate, not the iteration gate (2026-09-17)

Owner: "the gate blindly runs things that the edited code pieces did not
touch." True. Thirty-one gates take 20-30 minutes on an idle machine, most
of it on tiers a given change never reaches, and running the sweep after
every step also means nothing else may touch the machine meanwhile.

The protocol from here: after a step, run the TARGETED set for what it
touched (`run_gates.py --only SUBSTR`, one substring per call, seconds to a
few minutes); run the FULL sweep once per batch, before records. A targeted
green is iteration evidence; only the full sweep certifies, because the
tree is one translation unit and a header edit can move inlining anywhere.

The map, from what a change touches to what exercises it:

| touched | targeted set (`--only`) | what it does NOT cover |
| --- | --- | --- |
| `planning/policy.h` | `policy` (83.9M-check equality against the frozen `_ref_` twins) + the sets of every consumer below | -- |
| `oop/k1_commit.h`, `oop/c2c_*_create.h`, `oop/il2p.h` | `k1` (`k1_fourstep`, `k1_pow2`, `vfft_k1scr`), `il_solo`, `ilp_front`, `ztt` (three gates) | the 2D/3D rows that recurse into the 1D door -- covered by the 2D set |
| `transforms/fft2d/il2d_tier.h`, `il2d_cols.h`, `fft2d_create.h` | `il2d` (`il2d_onechain`, `il2d_real`), `natorder_scratch`, `mt_c2c` | **the 3D tier** |
| `transforms/fft2d/plane_queue.h`, `vfft.c`'s batch cell | `tcbatch`, `mt_c2c` | a 2D IL BATCH cell with T > 1 -- the plane queue has no gate of its own; the 2026-09-17 B1 fix was proved by probe (`il2d_mt_probe.exe ... K=4`), not by a gate |
| `transforms/fftnd/fftnd_il.h` | **nothing** | the whole 3D interleaved tier. See 5. |
| `wisdom2/*` (readers, writers, serializers) | `sp_ccol_decode`, `wisdom_cold_cell` (two-run, by hand), plus the set of whichever tier's rows changed | -- |
| `oop/il_flatdit*.h`, `oop/ztt*.h` | `flatdit`, `ztt`, `ztt_mt`, `ztt_odd`, `odd_ct`, `odd_partner_cells` | -- |
| `support/*` (race body, clocks, alloc, cpu_cache) | everything -- a full sweep | -- |

## 5. There WAS no 3D interleaved gate -- `ilnd_gate` since 2026-09-17

`fftnd_il.h` (1709 lines: the 3D c2c IL tier, axis 0 cycle/strip forms, the
flat child, plane threading) has NO gate in `benches/`. Its coverage in the
sweep is indirect: `il2d_*` through the shared column builder, and the K=1
gates through the recursive row create. On 2026-09-16/17 this tier received
the `nat_req`/`key->ord` fix, the Bluestein provider install, the flat
child's recalibrate copy, and steps R1/R3/R7 of the rank >= 2 policy -- each
proved by a probe (`recal_nd_probe.exe il3d`), none held by a gate. The
first 3D IL gate is worth more than most of the fixes it would have caught:
cold create at a spread of (N1, N2, N3) covering prime and composite axes
and both order classes; forward vs a naive 3D DFT at small cells; warm
replay bitwise; the axis-0 race log reading `(scr)` for a natural cell.

Written 2026-09-17 (`benches/ilnd_gate.c`, registered cold): six cells
(pow2, mixed, a single-radix axis, a Bluestein axis at 23, a replay-only
32^3), both order classes; axis 0 must race once and log `(scr)` whatever
the request's class; the small cells match a naive 3D DFT; the warm create
must not race and must be bitwise. It failed on its first run and found a
real defect (the design doc, "What the first 3D gate found").

## 6. The real tier's census (2026-09-17)

`benches/il2d_real_census.c` creates one 2D IL r2c cell cold with the race
log on; `_il2d_real_rowrace` now logs its LADDER (`[il2d-real] wl ladder
N1xN2: ...`) and not only its winner, so a change to what the ladder admits
can be diffed arm by arm (R2 was). `il2d_real_probe` has an .exe and no
source; this is its replacement for that purpose.

## 7. The targeted-run map, 2026-09-18 additions

| edited | run |
| --- | --- |
| `oop/il_prime.h`, the prime cell in `oop/k1_commit.h` | `ilprime_inner_gate` |
| `transforms/fft2d/il2d_tier.h` (the Bluestein provider, the forms serve), `wisdom2/wisdom2_2d_reader.h` | `il2d_blu_row_gate`, `il2d_onechain_gate`, `ilnd_gate`, `il2d_real_gate` |
| `transforms/fft2d/transpose.h` | `blocked_tail_gate`, `zr2c_fd_gate`, `odd_ct_gate`, `api_matrix_gate` (the split 2D/3D/4D doors) |
| `vfft.c`'s fingerprint | `k1_fourstep_gate`, and RE-CAPTURE `capture_baseline.py` |
| `oop/c2c_oop_create.h`'s MONO validation | `il_solo_gate`, `k1_pow2_gate`, `vfft_k1scr_gate` |

## 8. `form_slot_gate` does not build (found 2026-09-21)

`benches/form_slot_gate.c` -- the resolver invariant, the gate that proves
every kernel a form resolver returns for a slot is correct there -- has not
compiled since the four-step landed (2026-09-15): its TU includes
`planning/dp_planner_il.h` without `oop/k1_fourstep.h` first, so the
planner's `vfft_k1fs_plan_t` and `_k1fs_ctx` are undeclared. `run_gates.py`
has no entry for it either, so the sweep reports it "unbuildable" and moves
on; the `.exe` in benches/ is from 2026-09-15 and passes against the OLD
library, which proves nothing about today's resolvers. Both il2p resolvers
(`n1c`, `t2c`) changed on 2026-09-21 (registry-derived radix sets); the
change was gated by il_solo_gate, flatdit_gate, ilprime_inner_gate and the
forward-reference probe (`benches/k1_fwd_ref_probe.c`) instead.

Resolved 2026-09-21, and ruled: the planner is not a self-contained header
(its four-step arm names the plan type and `_k1fs_ctx`, which need vfft.c's
whole include order), so every TU that drives the planner directly is a
LIBRARY TU: it `#include "vfft.c"` and is built WITHOUT `--vfft` -- the shape
`sp_ccol_decode_gate` already had. `form_slot_gate`, `bwd_forms_race`,
`calibrate_k1_il` and `il_dp_odd_probe` build that way now; the gate ran
against today's resolvers (238 slots correct, 690 absent, 0 wrong). The
owner ruled the gate NOT NECESSARY: it is not registered in `run_gates.py`
and the sweep does not carry its 85 s build.
