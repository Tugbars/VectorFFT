# The descriptor front door: how `vfft.c` turns a config into a running plan

**Scope.** `src/core/vfft.c` — the public `vfft_create` / `vfft_execute` door and everything it
decides on the way to a handle. Which sub-plan a request resolves to, which axes are contracts
and which are searched, how a verdict is resolved and banked, and what the handle owns.

**Audience.** An engineer who knows FFTs and has to decide where a change belongs, or who needs
to know why a given request took the route it did.

**Reading rules used throughout.**

* Every non-obvious claim carries a `file:line`. Where a shipped comment disagrees with the
  code, the code wins and the comment is listed in
  [Appendix A](#appendix-a--comments-that-have-outlived-their-code).
* This document is a **declaration of how the system works**, present tense. It carries no
  change history, no dated decisions, and no "we used to". That narrative is not repo material.
* Measured figures are not reproduced here. Performance numbers live in
  [`docs/performance/v1_0_results.md`](../performance/v1_0_results.md); what each engine
  *searches* lives in [`docs/search_space/`](../search_space/).
* `vfft.c` keeps the comments that state an **invariant at the point where violating it would
  break something**. Everything cross-cutting is here instead. Section 15 lists what stays.

---

## 1. The front door

`vfft_create` dispatches by transform and, for each one, follows the same shape: resolve
wisdom, calibrate on a miss at the requested rigor, build the plan, stamp the executor onto
the handle. It serves C2C (in-place and out-of-place), R2C, C2R, DCT-I/II/III/IV, DST-I/II/III,
DHT, and ranks 1 through 4.

The shipping split engine is `plan_executors.h`, `twiddle.h`, `executor_generic.h` and
`executor.h`. (A previous executor generation, `engine/stride_executor.h`, redefined the same
symbols and was deliberately never included by this translation unit; it was removed on
2026-09-01 after having zero includers since the 2026-07-22 dag integration.)

## 2. Wisdom is a bundle

One `vfft_wisdom` holds every feature's table: the c2c spike table, the OOP two-axis table, the
rfft factorization table, dedicated 2D c2c / 2D r2c / 2D c2r tables, the 3D table, the Bluestein
`(M,B)` table, the 1D c2r path table, and the wisdom2 store
([vfft.c:174-210](../../src/core/vfft.c#L174-L210)).

`config.wisdom == NULL` selects a library-managed bundle rooted at `$VFFT_WISDOM_DIR`, or `.`
when unset. **Banking is always in memory first**, so a verdict is coherent for the rest of the
process. Writing it to disk is a separate decision gated by `config.wisdom_write`; a plan
created without that flag reports on stderr that its verdict was held in memory only
([vfft.c:2348-2362](../../src/core/vfft.c#L2348-L2362)).

Key axes, storage format and the merge law are `wisdom2`'s own subject — see
[`src/core/wisdom2/README.md`](../../src/core/wisdom2/README.md). Two properties matter at this
door: **direction is a key axis and kernel form is not**, and a lookup must compare keys through
`vw2_key_serves` rather than by hand.

## 3. Layout, order and placement are create-time commitments

All three are stamped on the handle at create and never re-inferred at execute.

**Layout** (`INTERLEAVED` / `SPLIT`) picks the dispatch axis. **Placement** decides which
executor and which buffer spelling is legal — and it is a commitment, not a hint: a distinct
destination on an in-place real plan is refused rather than served
([vfft.c:6634](../../src/core/vfft.c#L6634), and its c2r twin at
[:6693](../../src/core/vfft.c#L6693)).

**Order** is a contract. `SCRAMBLED` promises *some* self-consistent permutation — a route's
backward consumes its own forward's comb — not any particular one, so callers must never assume
which permutation scrambled output carries, and the identity permutation qualifies
([vfft.c:3231-3243](../../src/core/vfft.c#L3231-L3243)). Non-`DEFAULT` order on r2c/c2r/trig and
on padded batches is rejected up front rather than served wrong.

The rank test is `dims <= 1`, not `dims == 1`: `0` is the documented spelling of 1D, and a
zeroed config is the header's own quick-start shape
([vfft.c:3474-3481](../../src/core/vfft.c#L3474-L3481)).

## 4. The precedence ladder

**Every** route decision in the library resolves through one ladder:

```
environment racing hook   →  banked verdict  →  live race that banks its winner  →  structural default
   (beats wisdom,                                                                  (never a guess)
    never banks)
```

Four properties are load-bearing:

* **The hook never banks.** A forced arm cannot contaminate the store.
* **A hook set to anything wins, including "off".** Otherwise an off-vs-on A/B silently compares
  on with on ([vfft.c:2947-2956](../../src/core/vfft.c#L2947-L2956)).
* **A banked verdict is honoured at every rigor tier.** That is the point of banking it. Only
  the *race* that produces one is confined to a window; cells outside the window fall to the
  structural default.
* **Environment overrides are applied last in code**, which is what makes them beat wisdom.

Hooks are declared as data in `wisdom2.h`'s env-law table, not applied by that module.

## 5. Calibration, and the house race protocol

Rigor selects a **planner entry point**, not a cost model — the library has no cost-model tier.

| rigor | entry point | search |
|---|---|---|
| `MEASURE` | `vfft_proto_dp_plan_measure` | coarse beam over factorizations, then a per-stage variant refine |
| `PATIENT` | same, with `vfft_proto_dp_set_patient` | wider beam; top-K re-measured rather than believed from the sub-plan cache |
| `EXHAUSTIVE` | `vfft_proto_exhaustive_search` | every valid multiset of N, every unique permutation, full per-stage variant cartesian behind a pre-screen; DIT-only, degrades to patient DP when it cannot cover N |

First create pays the search; the banked entry is replayed by every later create.

**Every in-process race shares one shape**, and it is stated here once rather than at each site
(the executable form of the ARMS x PROTOCOL half is `src/core/support/race.h`; each site keeps its
own protocol constants, verdict rule, key and bank):

* arms alternate order across rounds, and the result is a **median** over rounds — an
  A-then-B race puts the two arms in different thermal windows, which is tolerable for a
  decision that dies with the process and not for one frozen into a record;
* **~3% hysteresis toward the compiled or structural incumbent**, so noise cannot flip a tie;
* private scratch — user buffers are never touched;
* **one arena with a small skew between arms** (separately page-aligned buffers produce 4KB
  aliasing and bimodal timings);
* **planning costs sit outside the timed region** — allocation and plan-build amortize over
  every execute — but *footprint* differences are a real execution effect and must be preserved,
  so a tight arm gets a genuinely tight region, never a strided view of a padded one;
* both arms get an equally good executor: giving one a baked kernel and the other the generic
  path silently favours the first;
* the winner is gated before it is trusted.

Two races deliberately do **not** bank: the pair-ordering race (§7) is memoized per process only.

### Tight versus padded

For a misaligned `K`, the batch allocator asks which buffer *shape* to hand back. **Tight**
builds at stride `K` and sends the leftover lanes through the narrow tail path. **Padded**
builds at stride `Kp`, so every row is aligned and there are no leftovers, at the cost of
computing `Kp - K` wasted lanes at every stage. The verdict lives in the existing `(N,K)`
entry's `exec_me` field — there is no separate padded wisdom file, and the pad plan *is* the
aligned `(N,Kp)` entry.

## 6. K=1 route selection

K=1 resolves to exactly one of: mono, the `il2p` / `il3p` pair-and-chain engines, `il_prime`, or
the cascade above the tier gate.

**IL runs its own pair search and must not inherit the split pair.** The `il2p` registries stop
at `R=64`, so two-pass IL tops out at `N=4096`; above that the honest answer is `IL_NONE`, and
inheriting split's pair produces a route naming an engine that cannot be built
([vfft.c:4938-4965](../../src/core/vfft.c#L4938-L4965)).

Three rules keep the route truthful:

* **The selector is a whitelist**, not a growing negative chain: a route names `2P_PURE` only
  if the plan exists, so execute never dereferences a NULL engine
  ([vfft.c:5028-5040](../../src/core/vfft.c#L5028-L5040)).
* **Availability degrade runs before the JIT key is captured**, or the JIT keys on a route this
  build cannot compile and retries on every create
  ([vfft.c:5082-5092](../../src/core/vfft.c#L5082-L5092)).
* **`il2p` belongs in the handle-exists guard.** Cells like `50 = 5×10` have an IL pair but no
  split route ([vfft.c:5132-5141](../../src/core/vfft.c#L5132-L5141)).

`MONO` is deliberately absent from the **in-place** IL candidate set: its kernels are
`__restrict__` and refuse aliasing ([vfft.c:2603-2612](../../src/core/vfft.c#L2603-L2612)).

There is **no parity constraint** on the pair search — the inline odd-count tail makes odd
factors legal, and the registry probes are the only availability filter
([vfft.c:4979-4986](../../src/core/vfft.c#L4979-L4986)).

### Pair ordering

`(R1,R2)` and `(R2,R1)` install different mid kernels, so for a *heuristic* (unbanked) pair the
ordering is measured at create. The incumbent keeps ties. A **wisdom-banked pair is used exactly
as banked** — the calibrator owns that axis. The pick is memoized for the process, so a cell is
raced once and every later handle for it builds the same plan, which is what preserves the
bitwise-identity contract between handles ([vfft.c:2668-2675](../../src/core/vfft.c#L2668-L2675)).

### Kernel form

Form selection for an `il2p` plan reads three sources in increasing priority: the structural
default installed by `vfft_il2p_create`, a banked kind-3 verdict, and the environment. Forward
and backward hooks both take the packed nibble form `mid | leaf<<4` and parse base 0, so hex and
decimal both work. The hook is also the only way to reach a form no wisdom line can name, since
a kind-3 record is refused unless it carries a split route.

## 7. The scrambled cascade

Above the tier gate, scrambled K=1 is served by the block-split cascade. `zroute` is the single
field both directions dispatch on, and **cutover atomicity is structural**: create keeps exactly
one plan and destroys the loser before the handle exists, so a mixed forward/backward pairing is
inexpressible ([vfft.c:247-259](../../src/core/vfft.c#L247-L259),
[:6468-6475](../../src/core/vfft.c#L6468-L6475)).

A banked kind-4 line is replayed as written: route, chain, terminator pick and tile width. A
line with no route tokens is a legacy verdict and is served as legacy. **A banked chain that
fails the fence falls back to the calibrated default rather than being force-fit**
([vfft.c:2908-2918](../../src/core/vfft.c#L2908-L2918)). A kind-4 row below the cascade tier is a
wrong-slot verdict and is made inert ([vfft.c:2827-2836](../../src/core/vfft.c#L2827-L2836)).

Two kind-4 rows can exist at one N. The problem VERDICT (`role` absent) is banked only by the
OOP create's own race, and a hit attaches the cascade for OOP. The COMPONENT recipe
(`role=comp`) is banked by an in-place or odd-mid race, which must never attach an OOP route by
fiat; it carries the same fields (chain, terminator pick, tile width, fence). An in-place caller
replays comp first, then the verdict. An OOP caller replays its verdict, and with no verdict may
replay a comp recipe only for an odd chain, because an odd candidate races the finished handle at
the commit and is never attached by fiat. A cascade mode row (`mode=zcasc`, `place=ip`) never
carries a chain of its own; it signposts the recipe it served with `ref=` (wisdom2 README §3.3).
On a cold mode row the in-place create builds its cascade candidate from the banked recipe
(the kind-4 race on a miss), so the plan that races the cell's K=1 IL engine is the plan the
signpost will serve (2026-09-03: the race is IL vs IL; there is no convert incumbent). The recipe row also carries the
cascade's per-thread-count MT verdict, one token pair per placement (`zt_mt_t`/`zt_mt` for the
OOP exit, `zt_mt_ip_t`/`zt_mt_ip` for the in-place exit).

**Only the terminator schedule is raced at create.** `stf` against `stf2` for zturn, `sterm`
against `sterm2` for legacy zsplit. Both pairs are bit-identical schedules, so the difference is
code placement and must be measured on the installed binary, never hand-set. A `last==4` chain
has no `stf2` twin, so the only legal pick is pinned and no verdict is returned — timing a
kernel against itself is not a race ([vfft.c:908-915](../../src/core/vfft.c#L908-L915)).
Everything wider — engine, chain, tile width — is searched offline by `dp_planner_il.h`.

A banked tile width is valid **only on the cache it was tuned against**; a mismatch means
untiled plus a loud line, never "use it anyway"
([vfft.c:2937-2946](../../src/core/vfft.c#L2937-L2946)).

### Natural order from the cascade

A natural-order create at or above the gate builds a cascade candidate whose `stfn` terminator
writes natural order directly, with no reorder pass. It **replays the scrambled chain** — that
plan data is order-agnostic — with `recalibrate` cleared on the copy, because the natural and
scrambled verdicts are separate regimes. Only the zturn route carries a natural mode. The
candidate is a *racer against the reorder-tape incumbent*, never a parallel execution path.

Natural verdicts are keyed per placement: in-place rows and out-of-place rows have different
incumbents, and keying them apart is what stops either bank from overwriting the other. The
`nf == 1, factors[0] == N` shape is the no-deployed-chain placeholder, stored as a `ref=`
signpost naming the scrambled cell it replays.

## 8. Real transforms: the zr2c composite

A 1D interleaved-CCE real transform of even `N` at `K=1` is served as a **composite, not a
dedicated engine**: the real input `x[N]` is reinterpreted as complex `z[N/2]`, a child
`c2c(N/2)` in natural order runs it, and the fold recovers the CCE spectrum. C2R is the exact
mirror with the fold leading. This composite is also **the library's only in-place real path**.

Two child shapes exist, named by the store: an out-of-place interleaved child and an in-place
cascade child. The pick is per `(transform, placement)` and resolves by the ladder in §4.

Four contracts hold here:

* The zr2c branch runs **before** the split-path calibrate-on-miss blocks, so a zr2c-served cell
  never pays for or banks rows it never reads
  ([vfft.c:5513-5522](../../src/core/vfft.c#L5513-L5522)).
* **No silent degrade to out-of-place.** If zr2c cannot be built there is no in-place plan to
  give, so create refuses loudly ([vfft.c:5528-5537](../../src/core/vfft.c#L5528-L5537) and its
  c2r twin at [:5658-5667](../../src/core/vfft.c#L5658-L5667)).
* The child must receive `recalibrate` and `wisdom_write`; dropping them narrows two documented
  public contracts ([vfft.c:2211-2223](../../src/core/vfft.c#L2211-L2223)).
* The route-1 input copy is gated on **pointer inequality, not placement** — otherwise route 1
  never reads its source ([vfft.c:2315-2322](../../src/core/vfft.c#L2315-L2322)).

The composite is **pool-free**, and must stay that way: that is what lets a zr2c plan serve as a
transform-contiguous worker clone ([vfft.c:7157-7170](../../src/core/vfft.c#L7157-L7170)).

Scratch planes are 64-byte aligned rather than plain `malloc`
([vfft.c:2233-2238](../../src/core/vfft.c#L2233-L2238)).

## 9. Trig transforms

Every DCT/DST/DHT is a `stride_plan_t` wrapping an inner plan — an r2c plan, or a half-N complex
FFT for DCT-IV. The inner c2c cell is **keyed under the owning transform at the outer size**, not
as a plain c2c cell, so a helper row and a genuine c2c row at the same `(N,K)` cannot collide.
The inner size per family is derived, not stored: DCT-I uses `N-1`, DST-I uses `N+1`, everything
else `N/2`.

## 10. Batch geometry

A **transform-contiguous** batch is K independent K=1 transforms laid end to end, and create
serves it as exactly that: one K=1 handle built through the same front door — inheriting every
K=1 route, verdict and race — executed K times at per-transform block strides. No batched plan,
no layout conversion, no batch-specific kernel, so every K=1 gain reaches batched callers
unchanged.

Block strides are derived at create from the committed `(transform, placement)`:

| shape | source stride | destination stride |
|---|---|---|
| C2C | `2*N` | `2*N` |
| R2C out-of-place | `N` reals | `2*(N/2+1)` |
| C2R out-of-place | `2*(N/2+1)` | `N` reals |
| real in-place | `2*(N/2+1)` | `2*(N/2+1)` |

`N/2+1` is the CCE bin count for either parity ([vfft.c:3343-3355](../../src/core/vfft.c#L3343-L3355)).

Scope is 1D, `INTERLEAVED`, `K>1`. At `K==1` both geometries are the same addressing, so the
request builds its ordinary K=1 plan. SPLIT batches are untouched — their geometry is the split
engines' own contract.

**For R2C and C2R this wrapper is the only route to the zr2c child.** zr2c reinterprets a
transform's `N` contiguous reals as `N/2` complex points, which requires those reals to be
contiguous; lane-major places the two halves of a complex sample `K` apart, so the reinterpret
is not expressible there at any price. That is a structural property of the route, not an
implementation gap.

Real transforms enter the wrapper on an **explicit** `batch_geom` only. `INTERLEAVED` DEFAULT
means transform-contiguous for C2C; for R2C and C2R, DEFAULT means lane-major — the geometry
`r2c_dispatch.h` addresses and the front-door gates assert. Changing that is a public contract
change, decided by a measured race
([vfft.c:3298-3307](../../src/core/vfft.c#L3298-L3307)).

A padded batch descriptor combined with `layout=INTERLEAVED` is rejected at create: the padded
planes are split, so the combination has no meaning.

## 11. Threading

**Transform-contiguous batches** split into contiguous slabs of `ceil(K/T)` transforms; worker
`t` runs its own clone handle, the caller takes slab 0, and there is one wait. Because the unit
of work is a whole transform rather than a set of SIMD lanes, **the slab size needs no
rounding** — a ragged `K` simply gives the last worker fewer complete transforms, so there is no
tail and no partial-lane count ([vfft.c:6872-6882](../../src/core/vfft.c#L6872-L6882)). This is
the opposite of the lane-major slab immediately below it, where floor-rounding silently drops
lanes. Multithreaded and serial runs are bitwise identical.

**The interleaved fast path** splits as lane slabs of `ceil(K/T)` rounded **up to 8** — a slab is
a set of SIMD lanes, so it must stay a whole multiple of the vector width. Fold resolvability is
pre-flighted once per execute, before any dispatch, because it is a property of the plan rather
than of a slab; dispatch is therefore all-or-nothing
([vfft.c:5838-5845](../../src/core/vfft.c#L5838-L5845)).

Whether the slabs run at all is the wrapper's **threading verdict** (`h->tc_mt`): serial loop
versus slabs raced at create on the batch's own cell and banked T-free as `eng=tcb tcmt=` on its
`q=K` row (`_tc_mt_decide`, measurement_arms B5). The 2048-complex-point scalar floor is retired.

The K=1 IL engines are **not reentrant**, so a worker runs its own clone
([vfft.c:289-300](../../src/core/vfft.c#L289-L300)). A clone is accepted only if everything
determining output bits matches the primary — attach pattern, chain plus natord, exact kernel
pointers; terminator pick and tiling are deliberately not compared
([vfft.c:3085-3094](../../src/core/vfft.c#L3085-L3094)). A clone's whole execute path must be
pool-free, and the predicate is conservative per direction
([vfft.c:3036-3047](../../src/core/vfft.c#L3036-L3047)).

## 12. The interleaved execute path

A 1D in-place C2C plan committed to `layout=INTERLEAVED` takes `2*N*K` doubles in lane-major
order, element `e` of lane `t` at `[2*(e*K+t)]`. The destination may alias the source.

Three routes, in order:

1. the cascade (`h->zturn`, both orders; natord under NATURAL);
2. the K=1 IL engines (`k1il2p` / `k1il3p` / `k1ilpr`);
3. nothing else: a handle with neither is a create bug and the execute warns and computes
   nothing. The padded arm, the folded `z->z` adapters and the convert fallback were DELETED
   on 2026-09-03 (owner: no split baseline for IL).

The converts are hand-written intrinsics rather than auto-vectorized scalar loops: AVX-512 moves
8 complex per iteration, AVX2 moves 4, and a plain-C loop is the floor. Both end in a scalar
epilogue and use no masked stores. They perform **no arithmetic**, so their output is
bit-identical to the scalar loops by construction.

## 13. Buffer ownership

`vfft_create` is the only public door into plan creation. With `owned_buffers = 0` it passes
through and allocates nothing beyond the plan. With `owned_buffers = 1` it builds the batch from
**the same config** through `_own_batch_for`, so `vfft_destroy` frees the planes with the plan —
and the batch-versus-handle cross-checks inside the inner create become invariants rather than
reachable failure modes, because no public API hands a foreign batch to create.

Owned batches are 1D and split-layout only; every other combination is rejected loudly at that
door. Callers read the planes back with `vfft_plan_planes` — already in `vfft_execute`'s argument
roles — and the stride with `vfft_plan_stride`, never by computing it.

Stride selection differs by kind. **Only 1D C2C in-place takes the measured tight-vs-padded
verdict**; a lane-aligned `K` and a prime `N` return tight without racing. Every other owned
batch pads unconditionally — C2C out-of-place at `roundup(K,8)` because the OOP kinds and the OOP
wisdom reader both gate on `K % 8`, and the real and trig batches at `roundup(K,4)` because
padding is their only full-SIMD path for a misaligned `K`. Allocation can pause **once** per
`(N,K)` while a pad verdict is raced. Only a nonzero verdict is banked, so a cell whose winner
fails the roundtrip gate races again on its next allocation.

## 14. Refusals

The door refuses rather than degrading, in every one of these cases:

| request | outcome |
|---|---|
| non-DEFAULT order on r2c/c2r/trig or a padded batch | rejected at create |
| padded batch descriptor + `layout=INTERLEAVED` | rejected at create |
| in-place real that cannot build zr2c | refused loudly — no out-of-place substitute |
| distinct destination on an in-place real plan | refused at execute |
| owned batch that is not 1D split-layout | rejected at the public door |
| K=1 cell with no split route and no IL pair | no handle |

## 15. What stays in `vfft.c`

A comment stays in the source when it states an invariant **at the point where violating it
would break something**, and when moving it would separate the rule from the line it governs.
Forty-eight blocks qualify. They fall into five groups:

* **Struct-field contracts** — what a non-NULL slot means and what it implies about the rest of
  the handle (e.g. a wrapper handle: nothing else on the struct is live).
* **Guards whose condition is subtle** — pointer-not-placement gates, `dims <= 1`,
  degrade-before-JIT-key-capture, whitelist-not-blacklist.
* **Paired edits** — sites that must be changed together, each naming its twin. The in-place
  refusals and the execute-time placement checks each exist twice, verbatim, by design.
* **Rounding and width rules** — where `ceil` versus `floor` is load-bearing, and where a slab
  is lanes rather than transforms.
* **Static asserts** — the chain-cap coherence check lives in this TU because of what it
  includes, and the assert must travel with its reason.

---

## Appendix A — comments that have outlived their code

Every entry below was independently verified against the code, then adversarially re-checked;
entries that survived only the first pass are not listed. Each is currently wrong.

| location | says | reality |
|---|---|---|
| `vfft.c:4-6` | *"WIRED: c2c in-place + c2c out-of-place. Other transforms land incrementally"* | R2C, C2R, trig and 2D/3D/4D all ship and dispatch from `_vfft_create_inner` |
| `vfft.c:1-15` | wisdom holds *"c2c spike + OOP 2-axis today; rfft/c2r/bluestein as features land"* | the struct already holds all of them |
| `vfft.c:1-15` | wisdom is *"auto-saved on calibrate"* | `_vw2_persist` writes only under `config.wisdom_write`; the default path warns that the verdict is memory-only |
| `vfft.c:231-238`, `:4726` | cites `row_major_engine.md §13` | no such file exists anywhere in the tree |
| `vfft.c:237` | *"Kill-switch: env VFFT_NO_K1 at create"* | `getenv("VFFT_NO_K1")` appears nowhere in the tree |
| `vfft.c:113-120` | *"This is the only translation unit that sees both headers"* | `dp_planner_il.h` includes both `oop_plan.h` and `zsplit.h` |
| `vfft.c:279-287`, `:3277`, `:6825` | transform-contiguous *"runs it K times at 2\*N-double strides"* | true for C2C only; the loop reads `tcb_sn`/`tcb_dn`, which differ for all three real shapes |
| `vfft.c:279-287` | *"nothing else on the struct is live"* on a wrapper handle | `tcb_sn`/`tcb_dn` are also live |
| `vfft.c:322-330` | *"until the calibrator lands, create uses the structural default"* | the calibrator is in this same file (`_zr2c_build` races both routes) |
| (retired) | `il_me` / `_il_me_decide` | deleted 2026-09-03 with the convert machinery; the row below is kept for history only: it rides its own `il_me` member, and `_il_me_decide` reads and banks exclusively through its `W` parameter |
| `vfft.c:1355-1356` | describes behaviour under `VFFT_WISDOM2_OFF=stride` | the env is retired and ignored; `vw2_off_stride` is **read at 19 sites and written at zero** |
| `vfft.c:2517-2534` | il_kv measured by `bench_1d_vs_mkl.c` building a handle per variant | that bench has no such path |
| `vfft.c:2578-2586` | the chain codec *"stores log2 per factor, so only 4/8/16/32/64 exist"* | `vfft_k1_cc_digit_of` also maps 3, 5, 7 and 11 |
| `vfft.c:2897-2904` | first entry-less N is *"32768+"* | `zsplit.h` seeds `case 32768:`; the first entry-less N is 65536 |
| `vfft.c:3908-3915` | *"padded runs single-thread here"* | `_exec_c2c_inplace` passes `me = h->exec_me` into `_c2c_mt` |
| `vfft.c:4683-4684` | *"Mono/Bailey IL tiers stay OOP-only"* | true for MONO only; the next block attaches Bailey in-place |
| `vfft.c:4185` | *"SCR/LEAF-IP still degrade to PURE until their executors land"* | SCR's executor landed; LEAF-IP was removed |
| `vfft.c:4869-4870` | the no-cascade guard keeps *"≥2048"* scrambled on the cascade dispatch | the tier gate is a knob, not a constant: `VFFT_NAT_ZCASC_MINN` defaults to 2048 (`zsplit.h:134`) and `vfft_zsplit_default_chain` seeds `case 1024` for exactly that purpose. The sentence is right at the default and wrong the moment the gate is lowered — which is the configuration the seed exists to serve |
| `vfft.c:5132-5141` | *"every IL attempt above is layout-gated for the spr < 0 case"* | false for `il2p`, the one route the sentence defends |
| `vfft.c:5830-5832` | folded path taken *"when the pool is single-threaded"*; MT falls back | MT is served natively by lane slabs |
| `vfft.c:6012-6019` | the IL A/B runs *"at the first-execute decision point"* | its only caller chain reaches it from `vfft_create` |
| `vfft.c:7632-7643` | `owned_buffers` allocates *"at a stride chosen by the measured pad-vs-tight verdict"* | true for 1D C2C in-place only; the other three kinds pad unconditionally |

Two references point into `docs/research/`, which is git-ignored and therefore absent from a
fresh clone.

**Duplication that is deliberate, not accidental.** `vfft.c:5528-5537` / `:5658-5667` and
`:6634-6644` / `:6693-6703` are verbatim twins (r2c and c2r). Both pairs must be edited together;
neither should be collapsed without leaving a cross-reference.
