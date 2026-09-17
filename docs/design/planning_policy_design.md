# The planning policy module — one place a law is written

Design, 2026-09-16. Basis: the owner's diagnosis (2026-09-09, "all our
racing, banking and search space heuristics are scattered around lots of
files", `feedback_planning_logic_is_scattered`) and the week that followed
it, in which four of the five defects found by gates were a law written in
N places and changed in N-1.

This design moves no line. It enumerates the laws, names every file each
one is written in today, and fixes the shape and the order of the
migration. The inventory is the deliverable; the module follows it.

## What a "law" is here

A law answers a question about a REQUEST, never about a plan's internals.
For a request (N, layout, order, placement, T, rigor) the questions are:

1. which CONTRACT CELL is this (the tuple wisdom keys, the class the
   `design_contracts.md` tables rule on);
2. which ENGINE FAMILIES may serve it (admission), and which of those RACE
   in this cell (the pool);
3. which WISDOM ROW serves it, and under which token names;
4. RACE or REPLAY: is the row a verdict for this cell at this thread
   count, or must the pool race;
5. what a REFUSAL is: when no engine serves, say so and stop (no
   fallback).

Everything else — how an engine is built, how a race is timed, what a
kernel does — is not policy and does not move.

## The inventory — every law and where it is written today

Counts are call sites, not lines. `src/core/` is implied.

### L1. Band admission (which route may serve which N)

| written in | what it says |
| --- | --- |
| `oop/ztt.h:541,552` | `vfft_ztt_band`, `vfft_ztt_odd_band` |
| `oop/k1_fourstep_band.h:11` | `vfft_k1fs_band` |
| `oop/k1_fourstep.h:554` | `_k1fs_sb_admit` (the super-band's L3 gate) |
| `oop/k1_commit.h:316,330,442` | the race gate's `pow2`/`oddband` split, the four-step's row pre-warm, the scrambled-refusal band |
| `planning/dp_planner_il.h:1359,1443,1513,1522,1802,1809,1814` | the same predicates again, per pool |
| `oop/k1_fourstep.h:162` | the engine re-checks its own band at create |
| `build_tuned/benches/bench_1d_vs_mkl.c:5643` | the bench's own copy of the admission |

Plus the unnamed inline bands: `(N & (N-1)) == 0 && N >= 2048` and
`N < 2048 || (N & 3)` in both pools, and the odd-with-factor-of-4 fence in
the race gate. **Cost this week**: the four-step's scrambled arms reached
262144 because L1 is written once per pool and the scrambled pool's copy
said something else; the ZTURN-T gate caught it, one law later.

### L2. Pool membership (which families race in a cell, per class)

| written in | what it says |
| --- | --- |
| `planning/dp_planner_il.h:1497` | the NATURAL pool (`_il_dp_enumerate_natural_engines` + the band branches) |
| `planning/dp_planner_il.h:1768` | the SCRAMBLED pool (`_il_dp_enumerate`, its own band branches) |
| `planning/dp_planner_il.h:1307..1437` | six family enumerators, each with its own admission (`_il_dp_enumerate_flat_ord`, `_fs`, `_ztt_ord`, `_ztt`, `_ztt_odd`) |
| `oop/il2p.h` | the pair tier's own arm pools (sides, forms) |
| `planning/dp_planner_split_oop.h` | the split library's pools (separate library, same shape) |

The two pools do not share a table: each re-derives "this cell is
ZTURN-T's / the four-step's / the flat DIT's" from L1's predicates.

### L3. Race-or-replay and per-T validity

Twenty-one sites spell `!cfg->recalibrate && <the row carries a verdict>`;
five of them add "and it was raced at THIS T":

| written in | verdict token |
| --- | --- |
| `oop/k1_commit.h:252,433` | the kind-3 row's `il_kv_raced` |
| `oop/k1_commit.h:744` | the flat DIT's `il_mt_t` |
| `oop/k1_commit.h:824` | ZTURN-T's `il_mt_t` / `il_mt_ip_t` |
| `oop/k1_commit.h:931` | the four-step's `il_mt_t` (+ `il_mtsb`) |
| `oop/c2c_ip_create.h:194` | the in-place mode row |
| `oop/c2c_oop_create.h:146,635` | the K=1 row, the natural row |
| `transforms/fft2d/fft2d_create.h:546,677,733` | the 2D chain/axis/Bluestein rows |
| `transforms/fft2d/il2d_tier.h:1401,1559,1785,1975,2026` | the real tier's and c2c tier's rows, `cmtt` |

The per-T rule ("a T=4 verdict never serves a T=8 request") is written
five times in five spellings.

### L4. Order classification (which class a request is, and its row)

| written in | what it says |
| --- | --- |
| `oop/c2c_ip_create.h:121` | `_ip_order_is_nat` (DEFAULT = natural at pow2) |
| `oop/k1_commit.h:410` | `scr_req` |
| `transforms/fft2d/fft2d_create.h:232` | `il2d_ord` |
| `transforms/fft2d/il2d_tier.h` | 8 inline `cfg->order == VFFT_ORDER_NATURAL ? VW2_ORD_NAT : VW2_ORD_SCR` |
| `transforms/fft2d/plane_queue.h:150`, `fftnd/fftnd_il.h` | the rank-N copies |

The 2026-09-07 defect (a natural cell banking on the scrambled row) was
one of these written from the pass's flag instead of the request's order.

### L5. Row-key selection (which wisdom row serves this cell)

Eighteen sites choose among `vw2_oop_lookup_k1`, `vw2_oop_lookup_k1_scr`,
`vw2__oop_k1_scan_ord`, `vw2_stride_lookup_nat`, `vw2_stride_lookup_scrmode`
— `oop/k1_commit.h` (9), `oop/c2c_oop_create.h` (4), `oop/c2c_ip_create.h`
(2), `planning/dp_planner_il.h` (1), and the 2D/real readers' twins. The
matcher law itself (`vw2_key_serves`) is correct and central; WHICH key to
ask with is not.

### L6. Engine presence ("a handle exists")

| written in | what it says |
| --- | --- |
| `oop/c2c_oop_create.h:526` | `spr >= 0 \|\| il2p \|\| il3p \|\| ilpr \|\| ilfd \|\| ztt \|\| fs \|\| mono` |
| `oop/c2c_ip_create.h:230` | `have_k1 = (il2 \|\| il3 \|\| ifd \|\| ztt \|\| fs \|\| mono_f \|\| ilp)` |
| `oop/c2c_ip_create.h:266` | the same list again, as the refusal check |

Three lists that must learn every new engine. **Cost this week**: the
four-step was built, banked and then refused at the door twice (once per
list) before the gate said so.

### L7. Route truthfulness and refusal

`oop/c2c_oop_create.h:162,401,426,444,509` — five `ilr = VFFT_K1_IL_NONE`
("the route names a plan that exists"), one per engine; the refusal text
at `:619` and `oop/c2c_ip_create.h:266`, each with its own reason string.

### L8. Hardware-gated ladders

| written in | gate |
| --- | --- |
| `planning/dp_planner_il.h:993` | L1d below 2048, else L2 |
| `transforms/fft2d/il2d_tier.h:1144,1272,2260` | L2 for the band widths, the strip widths, the cascade widths |
| `transforms/fftnd/fftnd_il.h:781,1070` | L2 for the 3D strip widths |
| `oop/k1_fourstep.h:556` | L3 for the super-band |

Five ladders, one shape: "a candidate is admitted when its working set
fits cache level X", each written by hand.

### L9. Ceilings

`VFFT_ZTT_MAX_N` (`oop/ztt.h:117`), `VFFT_K1FS_MIN_N`/`MAX_N`
(`oop/k1_fourstep_band.h:8,9`), `VFFT_K1_IL_PLAN_MAX_N` /
`_ODD_MAX_N` (`oop/k1_commit.h:292,295`) — combined by hand into one
expression at `oop/k1_commit.h:318`.

### L10. Token and serializer rules

`wisdom2/wisdom2_oop_reader.h`: the route-name table (`:37`), its
`_name_idx` bound (`:233`), two range checks (`:526`, `:906`), and the
per-route payload serializers written TWICE (the kind-3 path and the
lay=il path, `:533..560` and `:914..940`). **Cost this week**: route 10
banked as `il_route=?` because the table had 11 names and a `[10]` bound.

### L11. The bench and calibrator copies

`build_tuned/benches/bench_1d_vs_mkl.c` (the direct-cell admission),
`calibrate_k1_il.c`, and the census/probe benches each re-derive "which
cells exist" from L1's predicates.

## Contract

One module answers L1-L9 for a request. Every door, planner, calibrator
and bench asks it and holds no copy of an answer. The module is
DECLARATIVE: tables and predicates over (N, layout, order, placement, T),
returning names and small structs — never function pointers, never an
engine header's type — so it sits above every engine in the one-TU order
and below every door.

Three things it is NOT. It is not a heuristic: it says which pool RACES,
never which arm wins (wisdom decides that, the owner's law). It is not a
plan builder: no create, no execute, no allocation. It is not a wisdom
reader: it says which ROW KEY serves a cell; `vw2_key_serves` still
matches.

## The module

`src/core/planning/policy.h`, included in `vfft.c` after the band headers
reach it (`oop/ztt_mt.h` pulls `ztt.h` at line 36, `oop/k1_fourstep.h`
pulls `k1_fourstep_band.h` at 241) and before `planning/dp_planner_il.h`
(1078), `oop/k1_commit.h` (1079) and the two doors (1174, 1175). The
wisdom types it names are in scope from line 24.

```
typedef struct {                  /* the request, normalized once */
    int N, K, rank, T;
    int layout;                   /* VW2_LAY_IL / VW2_LAY_SPLIT */
    int ord;                      /* VW2_ORD_NAT / VW2_ORD_SCR — L4 lives here */
    int inplace, rigor, recalibrate;
} vfft_cell_t;

vfft_cell_t vfft_policy_cell(const vfft_config_t *cfg, int N, int K);

/* L1 + L2: the families admitted and raced in this cell, as NAMES */
typedef enum { VFFT_FAM_MONO, VFFT_FAM_PAIR, VFFT_FAM_CHAIN3, VFFT_FAM_FLAT,
               VFFT_FAM_ZTT, VFFT_FAM_ZTT_ODD, VFFT_FAM_FS, VFFT_FAM_PRIME } vfft_fam_t;
int  vfft_policy_pool(const vfft_cell_t *c, vfft_fam_t *out, int max);
int  vfft_policy_admits(const vfft_cell_t *c, vfft_fam_t f);   /* L1, one family */

/* L3 + L5: the row that serves, and whether it is a verdict here */
vw2_key_t vfft_policy_row(const vfft_cell_t *c);
int  vfft_policy_replays(const vfft_cell_t *c, const vw2_rec_t *r);  /* per-T validity */

/* L8: a candidate's working set against the hardware */
int  vfft_policy_fits(long bytes, int level);        /* 1 = L1d, 2 = L2, 3 = L3 */

/* L7: the one refusal, with the cell's own reason */
const char *vfft_policy_refusal(const vfft_cell_t *c);
```

L6 (engine presence) stops being a list: a door asks
`vfft_policy_pool` which families it must try, tries them, and refuses
when none built — the guard becomes a count, and a new engine is added by
adding a family to the pool table, in one place.

L9 becomes the pool table's own column: the band per family, one row per
family, read by `vfft_policy_admits`.

L10 is the wisdom layer's, not this module's, but the same audit applies:
the route-name table, its bound, the range checks and the two serializers
become one table with one writer. That is a separate, smaller change and
is listed in the checklist because the same defect class produced it.

## Migration — behavior-preserving, one law at a time

Each step: move the law, make every old site call the module, build, run
the full gate sweep, and confirm the shipped store's verdicts still
replay bit-identically (`capture_baseline.py --repeat 5`, the refactor
harness). No step changes what any cell decides. A step that cannot keep
a verdict identical is a DEFECT FOUND, and stops the migration until it
is ruled on.

1. `policy.h` with `vfft_cell_t`, L4 (order) and L9 (ceilings) only; all
   L4 sites call it. Smallest possible first cut: it proves the include
   order and the harness.
2. L1: the band table (family x N x class x placement), `vfft_policy_admits`;
   the three band predicates become its readers, the 13 call sites ask the
   module. The bench and calibrators too (L11). This step also absorbs the
   race gate's admission line, which removes step 1's measured transient:
   `vfft_ztt_odd_band` inlined twice in `_k1_il_plan_race`.
3. L2: the pool table; both IL pools become `vfft_policy_pool` + a
   family→enumerator switch that holds no admission of its own.
4. L5 + L3: `vfft_policy_row` and `vfft_policy_replays`; the 21 race-or-replay
   sites and the 18 row-key sites collapse onto them.
5. L6 + L7: the doors' three presence lists and five truthfulness sites
   become the pool count and one refusal.
6. L8: `vfft_policy_fits`; the five ladders read it.
7. L10: the wisdom route table and its serializers unified (separate,
   small, same defect class).

The split library (`dp_planner_split_oop.h`) is NOT in this migration:
SPLIT and IL are two libraries by the owner's law. The module is shaped so
the split door could adopt it later with its own tables; nothing here
assumes IL.

## Gates

The existing sweep is the whole gate: `run_gates.py` (every gate), the
four-step gate, `ztt_gate`, `k1_pow2_gate`, `il2d_real_gate`, the natural
and scrambled front gates, plus the fingerprint baselines. The migration
adds ONE gate of its own, `policy_gate`: for a sample of cells across every
band and both classes and placements, the module's answers must equal what
the old sites computed — built by keeping the old predicates alive as
`_ref` twins for the duration of the migration and asserting equality,
then deleting them with step 7.

## What this fixes, stated as a test

After the migration, adding an engine touches: the family enum, one pool
table row, one band row, one enumerator, one create, one execute, one
destroy. Not three presence lists, two pools, two band call sites and a
name table. Changing a law — "the scrambled pool at a pow2 cell is
ZTURN-T's alone" — touches one table row, and the gate that encodes the
law still proves it.

## Step 1 — what was moved, and the evidence

Moved (2026-09-16): **L4**, the order classification, from `_ip_order_is_nat`
(the in-place door), `scr_req` (`k1_commit.h`), `il2d_ord`
(`fft2d_create.h`), the 2D tier's eight inline copies and `plane_queue.h`'s
— all now call `vfft_policy_ord*`; and **L9**, the two ceiling constants and
the race gate's combined expression, now `vfft_policy_race_max_n`.

The three order laws are STATED, not averaged — they genuinely differ and
the module keeps all three: rank >= 2 sends DEFAULT to the scrambled row;
rank 1 out of place sends DEFAULT to the natural row; rank 1 in place sends
DEFAULT to the natural row only at a power of two. The K=1 ENGINE row is
the place=oop row whatever the request's placement, which is why
`k1_commit.h` asks with `inplace = 0` from both doors.

Evidence:

| check | result |
| --- | --- |
| `policy_gate`, sampled at every band boundary x 3 order classes | 1064 checks, ALL PASS |
| `policy_gate`, EXHAUSTIVE over N = 2..2^23 x 3 order classes | 33,555,492 checks, all four laws **equal** |
| `obj_equiv.py` on `vfft.o`, pre-policy vs policy | 1097 -> 1097 symbols, 0 gone, 0 new, **2 bodies changed** |
| latency A/B, one cell, A-B-A with a rebuild per arm | no regression (below) |
| the full gate sweep (`run_gates.py`) | 27 pass, 1 fail, 2 unbuildable — every failure PROVEN pre-existing (below) |

The two changed bodies are `_k1_il_plan_race` and
`_c2c_ip_create_il.constprop.0`. Disassembled and diffed instruction by
instruction: the same operations in different registers, plus 45 extra
instructions in `_k1_il_plan_race`. The cause is named, not guessed — the
race gate keeps its own `oddband` for the admission line ABOVE the ceiling
(an L1 law, step 2's), so `vfft_ztt_odd_band` — a shift loop and five
integer divisions — is now inlined twice. **This duplicate is a transient
of the L1/L9 split and step 2 removes it** when the band table arrives and
the whole gate becomes one question. Cost measured below: nothing.

The sweep's three non-passes are harness debt, not behavior, and each was
proven so rather than assumed:

- `form_slot_gate`, `il_dp_overflow_gate` — BUILD_FAIL. Both include
  `dp_planner_il.h` directly, which since the four-step (2026-09-15) needs
  `vfft_k1fs_plan_t` and `_k1fs_ctx`. Built in the PRE-POLICY tree they fail
  with the identical nine errors, so the step did not cause them. The
  minimal fix (the planner header listing `k1_fourstep.h` as its own
  prerequisite, as its siblings do) CASCADES — the four-step needs the
  public header and `struct vfft_plan_s` — so these two gates need the
  textual strategy (`#include "vfft.c"`, as `sp_ccol_decode_gate` uses) or
  a shared planner-prerequisite header. Recorded as debt; not step 1's.
- `k1_fourstep_gate` — FAIL with its usage line: added 2026-09-15 and never
  registered in `run_gates.py`, so the runner called it with no arguments.
  Registered 2026-09-16 as `("bare", True)` with a 900 s budget (seeded, so
  the upper-band cells replay instead of racing cold) and it now passes
  through the runner.
- `wisdom_cold_cell_gate` — FAIL with its usage line: it wants
  `--wisdir --N --phase 1|2 --out`, a two-run shape the runner cannot
  express. Older debt, untouched.

Latency, the cell `N = 1048576` natural interleaved K=1 T=1 — chosen
because it exercises every law step 1 moved (its out-of-place order rule
picks its K=1 row, its in-place variant the other rule, its four-step
child is a rank-2 cell using the rank-N rule and the 2D tier's eight
sites, and the pow2 ceiling gates it). Core 2, HIGH, 15 reps, a full
rebuild per arm, run A-B-A so drift is visible (`benches/policy_cell_probe.c`):

```
 arm            create oop (med)   create ip (med)   exec oop (med)   exec ip (med)
 policy          1.9797 ms          1.9866 ms         3.2736 ms        4.3090 ms
 pre-policy      1.9739             1.9805            3.1376           4.1457
 policy again    1.9819             1.9912            3.0965           4.1658
```

CREATE is the path the policy code is on: the two identical policy arms
differ from each other by as much as either differs from the pre-policy arm
(0.1-0.5%), so no regression is resolvable. EXECUTE is the control and must
not move at all: it drifts 5.7% MONOTONICALLY over the session, and the
last arm — with the module — is the fastest of the three, which identifies
the wander as drift rather than the change. The 45 instructions are ~20 ns
against a 1.98 ms create: 0.001%, three orders of magnitude under the
measurement's own floor.

## Step 2 — the band map, and what it is worth

Four laws moved, and the gate caught one of them being wrong on the way in:
`vfft_policy_k1_direct_cell` (what the bench means by "this N is the K=1 IL
tier's") is NOT `vfft_policy_races` (what the planner will race). They
differ at odd N above the race ceiling — N = 262145 is the first — because
such a cell is still SERVED, by the prime engine at the door, which is a
route and not a raced arm. They are two named laws now; conflating them
would have silently narrowed the bench.

The map is the thing the owner expected to find in this file: which engine
families compete at which N, in one table, per order class. Written from
the code (not from `design_contracts.md`, which the code is allowed to
drift from) and then PROVEN against the code:

| check | result |
| --- | --- |
| the map vs BOTH pools, as family sets, every N = 2..2^23 x both classes | equal |
| ZTURN-T's band vs its registry (223 cells) — the equivalence that lets the map gate the family by the band | equal at every N |
| `vfft_policy_races` vs the race gate's two inline refusals, every N | equal |
| `vfft_policy_scr_writer_band` vs k1_commit's no-fallback line, every N | equal |
| `vfft_policy_k1_direct_cell` vs the bench's copy, every N | equal |
| the gate overall | 83,887,134 checks, ALL PASS |

The competition interval the owner named is real and now stated once: at a
power of two, mono + pair + chain3 + ZTURN-T all race from 16 to 1024;
from 2048 ZTURN-T is alone to 262144; at 262144 the natural cell races
ZTURN-T against the four-step while the scrambled cell stays ZTURN-T's;
above it the four-step is alone to 2^22.

Step 1's measured transient is closed. `_k1_il_plan_race` was 1041
instructions before the migration, 1086 after step 1 (the duplicated
`vfft_ztt_odd_band`), and **1037 now** — the two refusals share one
computation, so the function is smaller than it started.

## Step 3 — the pools stop deciding

`_il_dp_enumerate_natural_engines` and `_il_dp_enumerate` re-derived the
band map from the predicates, once each. They are now one function: ask
`vfft_policy_pool` which families race, switch on the answer. Order is part
of the contract and the map returns it (ZTURN-T's odd chains first where
they apply, so a candidate-cap truncation can only eat the arms below
them). Each family's enumerator answers only "what do I have for this N".

Verification, beyond the gate's exhaustive family-set equality: a CANDIDATE
CENSUS — the planner's own verbose arm list, timings stripped, from a COLD
store so every cell races — over nine cells covering every band, both
order classes, captured from the pre-step-3 planner (HEAD, which steps 1-2
never touched) and from the step-3 planner, each freshly built:

| | |
| --- | --- |
| arm COUNT per cell | identical at 9 of 9 |
| arms byte-identical | 7 of 9 |
| the other 2 (1536, 15625) | differ only in the FLAT DIT's raced tile width (`w=`), a measured sub-race outcome — and these are exactly the two cells a same-binary control run had already shown to be run-to-run noisy |
| `policy_gate` | 83,887,134 checks, ALL PASS |
| the full sweep | 27 pass (the three known pre-existing gaps unchanged) |

A discarded first attempt is worth recording: the first baseline census used
a probe binary that predated the same day's super-band ruling, so it
compared two different engines and showed phantom differences. A census is
only a baseline if its binary is built from the tree it claims to measure.

## Steps 4-6 — the narrowed laws, and the differences that stayed

Six parallel surveys, each with an independent adversarial check that was
told to refute it. Every survey came back INCOMPLETE (the checkers found 9
to 49 further sites apiece), and each found differences that look like
duplication and are not. The conclusion was not "these cannot be done" but
"these are not one law each": what shipped is the shared part of each, with
every load-bearing difference left spelled at its call site.

**L3 → `vfft_policy_replays_at_T(banked_T, T)`, seven sites.** A threading
verdict is a measurement AT a thread count; a T=4 verdict must never serve a
T=8 request. That comparison — five token spellings across the flat DIT,
ZTURN-T, the four-step, the 2D c2c and real column-MT verdicts, the 3D
tier and the plane queue — is the whole of what these sites share. What
stayed at the sites: `recalibrate`, which is written at the site in
`k1_commit.h` and `fftnd_il.h` but sits UPSTREAM in the lookup for the 2D
tiers and is order-scoped (`scr_recalib`) elsewhere; "the row carries a
verdict", which is three different tests; and the plane queue's second
fence on the plane count. Deliberately excluded: the T-FREE `tcmt`/`tcmtt`
(the batch verdict RECORDS T and never compares it — the fence would refuse
every replay at another T), the execute-side clamps, and
`fftnd_wisdom.h`'s filter, the only site that NORMALIZES the requested T.

**L5 → the existing L4 laws, six sites.** There is no new helper: the
genuinely shared sub-question is "which order class is this cell", which
`vfft_policy_ord_k1` / `vfft_policy_ord_rankn` already answer. Six sites
that spelled it by hand now ask (`c2c_oop_create.h:114`'s `scr_req`,
`c2c_ip_create.h:142`, three copies inside the single function
`_vfft_create_2d`, and `fftnd_il.h`'s rank-N flag). The rest is
documentation, because it is not one law: `scr_req` also ANDs `layout ==
INTERLEAVED && !vw2_off_oop`, so the policy spelling alone would change what
a SPLIT-layout scrambled request reads and what a kill-switched store does;
and the dir=bwd rows have NO order axis at all (`vw2_oop_rec_k1_bwd`
hard-codes `ORD_NAT`), so "which row serves this cell" has a different
answer there.

**L6 → `vfft_policy_k1_engine_present(...)`, three doors.** Seven engines,
passed as seven SEPARATE PARAMETERS rather than a struct or a pointer list,
and that is the entire mechanism: adding an engine is a compile error at
every call site. A struct with designated initializers would let a new
engine default to 0 at a site nobody updated — which is the defect itself —
and a pointer list would drop every out-of-place MONO cell, because MONO has
no plan object there: it is admitted by ROUTE and its pointers are resolved
thirty lines below the guard. What stayed at the sites: the SPLIT axis's
route (`spr >= 0`, not an engine) and each door's layout gate.

**L7 → documentation.** `-1` and `IL_NONE` are different: -1 means
"unraced, run the heuristic", `IL_NONE` means "raced, the answer is none —
do NOT". A demotion macro was proposed and is safe, but it would live in
`oop_plan.h`, not in the module, so it would not move a law into the shared
file; it is not worth a new symbol in the route namespace. Two findings the
survey offered as live defects were REFUTED by its own checker and are
recorded here so they are not re-derived: the "missing PRIME demotion" is
refuted by the law shipped in `policy.h` this week (PRIME is a door route,
never a raced arm), and the "missing MONO demotion" is unreachable because
the planner refuses to bank the form that would reach it.

**L8 → `vfft_policy_fits_l2` + `vfft_policy_exceeds_l3`, six sites.** Two
helpers because the two laws differ in BOTH polarity and unknown-size rule:
the five L2 ladders admit what fits and treat an unknown cache as refuse;
the super-band admits what does NOT fit and treats an unknown L3 as admit —
which is live, not theoretical, since `l3_seen` has no fallback and is 0 on
an L3-less part. The single `fits()` is now impossible to reintroduce
quietly: a gate arm asserts that a working set BETWEEN L2 and L3 fails both
predicates, which is exactly what a negation would break. One site the
inventory filed under L8 is NOT a fits() at all — `dp_planner_il.h:993`
selects a byte BUDGET by N and hands it across an API boundary, and the
predicate behind it (`il_flatdit.h:809`) treats a zero budget as NO GATE, a
third unknown-policy opposite to both helpers. It stays as it is.

### A correction to this document

The claim that `fft2d_create.h:1162` (c2c) and `:1220` (real) carry
OPPOSITE recalibrate semantics described the tree BEFORE the fix recorded
below. Since the chain lookup at `il2d_tier.h` learned `!cfg->recalibrate`,
both tiers are guarded upstream and the pair is symmetric — which is why
both could migrate to the same fence with the same terms around it.

### What the survey found that is not the migration

Recorded so they are not re-derived, in the order they are worth acting on.

| where | what | state |
| --- | --- | --- |
| `c2c_oop_create.h` fall-through | the cleanup destroyed six engines and not the four-step — a FOURTH hand-maintained per-engine list that never learned it | FIXED 2026-09-16 (`vfft_k1fs_destroy(fs)`); the path is the calloc-failure fall-through, so it leaked rather than answered wrongly |
| `vfft.c`'s fingerprint | no `FP__P(k1fs)` in the subplan presence bitmap, and no detail line — the four-step is invisible to the fingerprint | open |
| `c2c_oop_create.h:508` vs `:560` | the MONO route is validated with `vfft_k1_mono_il_fn(N, 0)` (FORM 0) while the handle resolves the row's BANKED form; a row carrying `il_route=MONO il_kv=1` at N != 64 would build NULL pointers, and `vfft_execute.h:1092` calls them unguarded | latent — the planner refuses to bank that form |
| `transpose.h:73-77` | bakes `TP_L1_BYTES` / `TP_L2_BYTES` and picks its recursion body from them, while the CPU query reports 1.5-2x more | open: two disagreeing cache authorities |
| `il2d_tier.h:1265`, `:2237`, `fftnd_il.h:1061` | four of the five L2 ladders gate only the chain's own stage spans; the static width pool beside them is pushed ungated | open, by design or not — needs a ruling |
| `cpu_cache.h:518,524` | `vfft_cpu_l2_matches` / `_l1d_matches` have zero callers: a banked-width replay rule nothing replays | open |

## Step 7 — the route set stops being five numbers

`VW2_OOP_IL_ROUTE_MAX` is defined from the route enum, the name table is
declared UNSIZED so its length comes from the names, and a compile-time
assertion compares the two. The lookup bound and both range checks read
the constant; each range check keeps its own LOWER bound, because they
differ deliberately (`< 0` admits IL_NONE for the split library's
route-less rows; `<= IL_NONE` refuses it on the lay=il path).

Verified, including the guard itself:

| check | result |
| --- | --- |
| the tree as it stands | builds clean |
| NEGATIVE TEST: enum grows, names do not | build FAILS (`size of array 'vw2__il_name_table_is_complete' is negative`) |
| route 10 banks and replays | `k1_fourstep_gate` ALL PASS; the store carries `il_route=fs`, `ztt`, `2p` — no `?` |

The first version of this assertion was CIRCULAR — the array was declared
`[MAX + 1]`, so `sizeof` measured the macro against itself and passed with
a NULL hole, which is exactly the bug it exists to catch. The negative test
is what exposed it. An assertion nobody has watched fail is not a guard.

## Steps 4-6 — the evidence

| check | result |
| --- | --- |
| `policy_gate`, exhaustive | 83,892,082 checks, ALL PASS (the new arms add L3 over every (banked, requested) pair, L6 over all 2^7 presence combinations against all three doors' spellings, L8 over every power of two to 1 GiB plus both cache capacities +/- 1) |
| NEGATIVE TEST: drop `fs` from the presence helper | gate FAILS, `L6 engine presence: 3 of 384 door checks differ` |
| NEGATIVE TEST: write `exceeds_l3` as `!fits_l2` — the refuted shape | gate FAILS twice, including `L8 polarity: a working set between L2 and L3 must fail BOTH` |
| the full sweep | 27 pass, the three known pre-existing gaps unchanged |

`obj_equiv` is NOT the gate for this step and cannot be: the step adds inline
functions and moves an include, so GCC's inlining budget shifts and the tool's
EQUIVALENT verdict is unreachable by construction (its own header says so).
Measured, like-for-like under one flag set: 1111 -> 1112 symbols, ten changed
bodies, TU instructions +4917 of 11.1M. Every changed body is a function that
contains a migrated site or inlines one.

## Where the migration stops, and why

The eight steps are done. Two further unifications were costed on 2026-09-16
— a 12-agent survey of the whole tree, each scope attacked by an independent
checker — and the owner DECLINED both. They are not pending work; they are
closed with a reason.

**Unifying the `recalibrate` check on one shape: declined.** The flag is
honored in three shapes (at the branch, at the lookup, order-scoped) and all
three are correct today. Converting the tree to one shape is 31 edits across
44 sites, three of which need new function signatures, and the naive form
breaks two contracts: at `k1_commit.h:737/:818/:926` the same fetched row is
both the replay source AND the bank key, so hiding it makes a recalibrate
create re-measure and then bank nothing — the opposite of what `vfft.h:325`
promises; and at `c2c_oop_create.h:144` a change aimed at the IL library
costs the SPLIT library its banked route with no race to compensate. What
survived is a defect list: the paths that do not honor the flag AT ALL
(`docs/roadmap/policy_survey_defects.md`, section A).

**Moving the banking side onto the order policy: declined, and largely
moot.** The rank>=2 tiers (2D c2c, 2D real, 3D IL) already derive the banked
order class from `vfft_policy_ord_rankn`, and the K=1 writer
(`k1_commit.h:657`) takes it from `scr_req`, which is
`vfft_policy_ord_k1`. The order axis is therefore already on the module at
every live writer. What remains hardcoded is either deliberate (the
planner's scrambled arm banks `ord_scr = 1` because it IS the scrambled
bank) or a defect, not migration debt — see section B of the same file, which
is where the audit's real return went.

**L11, the bench and calibrator copies, needs no further work.** The only
genuine copy was `bench_1d_vs_mkl.c`'s direct-cell admission, migrated in
step 2. The three ZTURN-T gates reference `VFFT_ZTT_MAX_N` and
`vfft_ztt_odd_band` — the engine's own band, which is the shared source the
module itself calls — and `calibrate_k1_il.c` takes its cells from argv.
Neither re-derives a law.

## The band map, validated from the outside (2026-09-18)

The owner's check: recalibrate one cell per region on a SCRATCH copy of the
shipped store (`cfg.recalibrate = 1`: re-race, overwrite), read the route the
store banked, and bench the cell with the canonical bench (`--k1noop`,
natural, out of place, T = 1, the recorded protocol) beside the recorded
ratio in `v1_0_results.md`. A wrong law in the module would show twice: a
route by the wrong name, or the right route at the wrong cost.
`build_tuned/band_recal_check.sh`, `benches/recal_1d_probe.c`.

**Every region banked the method the map assigns** -- 24 cells: the mono
band (ZTURN-T at 16, pairs at 32/64), pairs through 512, ZTURN-T 1024 to
262144, the four-step above, ZTURN-T's odd band, chain3, the flat DIT. 17 of
24 ratios within a few percent of the record.

The other seven were run again, twice, and none is a policy or a plumbing
defect. Replaying the SHIPPED verdicts on today's binary reproduces the
record (2097152: 1.387 vs 1.40-1.43; 1048576: 1.347 vs 1.26-1.34). What the
check found is one level down:

| cell | what happened |
| --- | --- |
| 256 | same shipped verdict every time; the bench reads 0.99 or 0.68 depending on which engine runs first -- a 140 ns transform is order-sensitive |
| 15625 | two cold races banked two forms: `t.m.t.t.o @ 625` (0.71) and the recorded `t.t.t.t.o @ 3125` (1.06) |
| 3125 | chain3 wins over the flat DIT in today's races (both admitted: pure odd); one race picked a worse chain3 split (625.5, 0.80) than the next (125.25, 1.00) |
| 512x4096, the four-step's child at 2097152 | the recalibration re-races the CHILD; it banked `chain=64.8 wl=0 sw=64` (the strip form) where the shipped row is `chain=8.8.8 wl=8` -- 14.4 M ns against 8.6 M. A single cold race at a 32 MB plane picked the form the DRAM-roof work refuted |

So: a cold race is ONE sample, and at two cells one sample banked a verdict
20-70% worse than the recorded one. Whether the race body should take more
than one cold sample before banking a verdict that will serve for weeks is
the owner's ruling; the child race's strip arm at the four-step's cells is
the first thing to look at. The shipped store stays: the check proved it is
the better one.

## Checklist

- [x] 1. This design (the inventory above is the survey of 2026-09-16).
- [x] 2. Step 1: `policy.h`, `vfft_cell_t`, L4 + L9; `policy_gate` with the
      `_ref` twins (2026-09-16, evidence below).
- [x] 3. Step 2: L1/L2, the BAND MAP (2026-09-16). The map
      (`vfft_fam_t`, `vfft_policy_pool`, `vfft_policy_admits`),
      `vfft_policy_races` (ownership + budget as one question, which removed
      step 1's measured duplicate), `vfft_policy_scr_writer_band` (the
      scrambled no-fallback law) and `vfft_policy_k1_direct_cell` (the
      bench's copy, L11, now asked not kept). Every one proven equal to the
      site it replaced over N = 2..2^23. The two pools CONSUMING the map is
      step 3.
- [x] 4. Step 3: L2, the pools CONSUME the map (2026-09-16). The two
      enumerators became ONE `_il_dp_enumerate` driven by
      `vfft_policy_pool` + a family switch; MONO, PAIR and CHAIN3 moved out
      of the old 269-line function into one enumerator each, body for body.
      Not one admission rule is left in the planner.
- [x] 5. Steps 4-5 (2026-09-16), in their NARROWED form — the owner ruled
      the narrowed proposals available. L3's per-thread-count fence
      (`vfft_policy_replays_at_T`, 7 sites) and L6's engine presence
      (`vfft_policy_k1_engine_present`, 3 doors) are in the module; L5
      migrated its 6 order-class row selectors onto the EXISTING L4 laws and
      is otherwise documentation; L7 is documentation, its two headline
      findings having been refuted by the check.
- [x] 6. Step 6 (2026-09-16): L8 is TWO helpers, not one —
      `vfft_policy_fits_l2` and `vfft_policy_exceeds_l3`, each carrying its
      own polarity AND its own unknown-size rule in its name and body. Six
      sites migrated. The refuted single `fits()` is now a gate arm: the
      polarity check fails the build's own gate if anyone writes one as the
      negation of the other.
- [x] 7. Step 7: L10, the IL ROUTE SET is one declaration
      (`VW2_OOP_IL_ROUTE_MAX`, from the enum) and the name table is checked
      against it AT COMPILE TIME. The five hardcoded bounds are gone. The
      two serializers are NOT unified — the survey proves they diverge at
      route 10 — and the bwd builder's lower bound is documented as
      deliberate, not stale.
- [x] 8. Records (2026-09-16): `design_contracts.md` section 4 declares the
      band table's one implementation and names the three laws beside it;
      `planning_model.md` section 2 points the `N band` branch at the module
      and says which of the two wins where they disagree; memory.
