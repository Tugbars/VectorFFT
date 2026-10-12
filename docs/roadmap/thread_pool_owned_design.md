# The owned thread pool (design of record)

**Date:** 2026-10-12 · **Status:** DECIDED by the owner (field trial Q13/Q14), BUILT 2026-10-12 (the
field trial's repros: two creates on two threads 12 rounds clean, two threaded executes 0 bad of 200 x 6
pairings, a create under an execute 0 bad and no hang, the caller's affinity and MXCSR unchanged after a
create, 0.00 CPU-s per idle second once no plan lives; gates il_solo, vfft_tcbatch, il2d_real_ip, k1_pow2
pass); the FTZ\|DAZ choice of §4 open ·
**Closes:** field trial V04, V06, L4 (concurrency), V08, V09 (pool side effects), V07 (FTZ|DAZ) ·
**Scope:** `common/support/threads.h` (the pool, already the one owner of dispatch since 2026-09-01),
the front door's pool calls in `vfft.c`, the K=1 planner's context in `il/rank1/k1_commit.h`.

## 1. The promises this pool keeps

- `vfft_execute` is "safe to call concurrently on different plans" (vfft.h) with no exception for
  threaded plans, and it is pure: no thread is created, pinned or resized inside it.
- `vfft_create` may be called from any thread at any time, beside executes and beside other creates;
  it never destroys or resizes workers a live plan uses; it gives the calling thread back as it was
  (affinity, priority, MXCSR).
- A live plan always has the workers it was built with (the thread count is a wisdom key: a plan
  created at T runs as the plan that won its race at T).
- Idle workers cost nothing: they park after the last threaded plan is destroyed and after a block
  time inside a burst; inside a burst dispatch stays the ~10 ns spin.
- `vfft_set_num_threads(n)` sets the count for FUTURE creates and parks what no live plan needs; it
  never takes a worker away from a live plan. `vfft_get_num_threads` returns that setting.

## 2. The mechanisms

1. **Worker slots never move.** `THREAD_POOL_MAX_DISPATCH` worker structs in static storage; a
   worker thread holds a pointer to its slot for its whole life. Growth starts threads for further
   slots; nothing is reallocated, so an execute in flight on slots 0..k never sees a moved struct.
2. **One pool lock** (recursive: creates nest, and a slot-0 body may dispatch again). `thread_pool_run`
   holds it from the first dispatch to the last wait, so two threaded executes on different
   application threads take turns on the workers instead of sharing `done` flags (V04). Growth,
   parking and the setting change take the same lock, so no resize happens under an execute (V06).
3. **Plan retention.** `thread_pool_plan_retain(T)` at create (where `_vfft_pool_arm` stood) and
   `thread_pool_plan_release(T)` at destroy keep a live count per thread count. The workers needed
   are `max{T : live[T] > 0} - 1`; growth happens at retain, never at execute; a release that drops
   the need lets the spare workers park. `thread_pool_workers_for(plan_T)` clamps by the plan's
   snapshot and the alive workers, no longer by the setting, so a shrink of the setting cannot
   strand a live plan.
4. **The serial scope.** The route races that timed their arms serially by resizing the pool to 1
   and back (`_r2c_route_decide`, `_c2r_route_decide`) enter a thread-local serial scope instead:
   `thread_pool_workers_for` answers 1 inside it, on that thread only, and no worker is touched.
5. **Parking.** Each worker spins on its `done` flag for the block time, then parks on its own event;
   a dispatch to a parked worker signals the event (one wake per execute after a pause, ~5 us). When
   no threaded plan is live the block time is zero: the workers park at once. The block time is a
   process setting (`vfft_set_thread_block_ms`, default 200 ms, `VFFT_THREAD_BLOCK_MS` read once when
   the pool starts) and a first-class condition of every threaded measurement.
6. **The caller's thread is returned as it was.** The pool no longer pins the caller to CPU 0; the
   measurement scope's own P-core pin during races is restored at the scope's end as before. The
   create saves the caller's MXCSR at entry and restores it at exit. Workers keep FTZ|DAZ for life.
7. **FTZ|DAZ per execute.** The calling thread's slice of a threaded execute runs in the workers'
   mode: execute reads MXCSR, sets FTZ|DAZ for the call when the caller's mode lacks it, and restores
   it afterwards. The read is ~1 ns; the set and restore cost only callers whose mode lacks FTZ|DAZ
   (see §4 for the cost and the open question).
8. **One create at a time.** A recursive create lock around `_vfft_create_outer`: two creates on two
   threads serialize (the K=1 planner's context, the prime arm, the four-step map and the 2D
   Bluestein context are process statics, "one create at a time" by their own comment; and two
   races at once would corrupt each other's timings anyway). Execute never takes this lock. Lock
   order: create lock, then the pool lock; a worker never takes either.

## 3. What changes where

| file | change |
|---|---|
| `common/support/threads.h` | static slots; the pool lock; retain/release and the needed count; the serial scope; park/unpark with the block time; `thread_pool_size` = the setting; `thread_pool_workers_for` by snapshot and alive workers |
| `vfft.c` | `_vfft_pool_arm` -> retain (no caller pin); plan release in `vfft_destroy`; the route races use the serial scope; `vfft_set_num_threads` = setting + park; the create lock and the MXCSR save/restore in `vfft_create`; the block-time setter |
| `vfft_execute.h`, `il/il_execute.h`, engines | every `_vfft_pool_arm` on an execute path removed (the workers exist for a live plan); FTZ\|DAZ around the call at the door |
| `include/vfft.h` | the promises of §1 stated at `vfft_execute`, `vfft_create`, `vfft_set_num_threads`; the new setter |
| `include/vfft_diagnostics.h` | the block time and the parked state readable |

## 4. Costs and open questions

- The pool lock adds one uncontended lock/unlock (~20 ns) to every threaded execute. Serial plans
  never take it. Measured against the threaded cells before and after (same run, pinned, paced).
- Workers parked by the block time pay one event wake on the first execute after a pause. That is
  the regime the measurement memory names: report the block time with every threaded number.
- **FTZ\|DAZ per execute (owner's ruling: restore after create, set per execute).** Measured on this
  machine (i9-14900KF, pinned, `mxcsr_probe`, 7 x 2e7 iterations, best): reading MXCSR and testing the
  two bits costs nothing measurable (+0.0 ns) when they are set; setting and restoring them around a
  call costs +3.2 ns (two `ldmxcsr`). A small-N execute is 4.6-6 ns through the lean door, so a
  caller whose mode lacks the bits pays about +60% per call. The door therefore applies the
  conditional form (set and restore only when the mode lacks the bits). Open for the owner: whether
  a create should keep leaving FTZ\|DAZ set on the caller (today's behaviour, to be stated in the
  header), so the common caller never pays, with the per-execute check as the safety net for one who
  changed the mode; or restore it after the create, as ruled, and accept +3.2 ns on every execute of
  a caller that does not set the bits itself.
- The caller unpinned: the workers stay pinned to cores 1..T-1 and the caller's slot 0 runs where
  the OS puts it. If a threaded cell measures slower than with the old CPU-0 pin, the owner decides
  between the header's promise and the pin.

## 5. Verification

The field trial's repros, rebuilt at the new tree: `g2v04_s2` / `g2v04_conc_mt --pair 0..5` (two
threaded plans at once: 0 bad outputs), `g2v06_s12x ladder|split|splitm` (a create under an execute: 0
bad, no hang), `conc_create --pair 0..3` (two creates at once: no crash), `g2v05_threads idle`
(parked workers: 0 CPU-s per idle second), `g3v_thr_pin` / `g3v_thr_regrow` (the caller's affinity
unchanged after a create; no regrow in execute), the V07 MXCSR repro (the caller's mode restored, every
slab in one mode). Gates: `il_solo`, `vfft_tcbatch`, `il2d_real_ip`, `k1_pow2` (one at a time).
