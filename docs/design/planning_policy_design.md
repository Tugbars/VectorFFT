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

## Steps 4-6 — what the survey found, and why they stop here

Six parallel surveys (one per law) each with an independent completeness
check. Every survey came back INCOMPLETE — the checkers found 9 to 49
further sites apiece — and, more importantly, each found differences that
look like duplication and are not. The migration's own rule applies: a step
that cannot keep a verdict identical stops for a ruling.

**L3, race-or-replay (~75 sites).** `fft2d_create.h:1162` (c2c) and `:1220`
(real) are textually the same line with OPPOSITE recalibrate semantics.
CONFIRMED IN THE CODE, not merely surveyed: the real tier's `il2d_bcmtt`
comes from `fft2d_create.h:545`, `else if (!cfg->recalibrate &&
vw2_2d_rl_lookup(...))`, so under recalibrate it stays -1 and the branch is
dead. The c2c tier's comes from `_il2d_col_build` →
`vw2_ilcol_chain_lookup` (`il2d_tier.h:1840`), which carries NO recalibrate
term — although `_il2d_col_build` RECEIVES `cfg`, and the same file guards
five other lookups with `cfg->recalibrate` (:1401, :1559, :1785, :1975,
:2026). So a 2D c2c create with `recalibrate = 1` replays the banked chain,
band width, row route AND column-MT verdict instead of re-racing them. See
"An open defect" below. There is also a deliberately T-FREE
verdict (`tcmt`/`tcmtt`) that records the thread count and never compares
it. And `recalibrate` is sometimes a parameter, sometimes order-scoped
(`scr_recalib`). One `policy_replays()` would silently pick one of these.

**L5, row keys (~124 sites).** `scr_req` is not one predicate:
`c2c_oop_create.h:114` also ANDs `layout == INTERLEAVED && !vw2_off_oop`,
so replacing it with the policy spelling changes what a SPLIT-layout
scrambled request reads and what a kill-switched store does. The dir=bwd
rows have NO order axis at all (`vw2_oop_rec_k1_bwd` hard-codes
`ORD_NAT`), so "which row serves this cell" has a different answer there.

**L6, engine presence (~34 sites).** MONO has no plan object at the
out-of-place door — it is admitted by ROUTE, and its function pointer is
assigned sixteen lines after the guard — so a pointer-list helper drops
every MONO cell out of place, and the fall-through serves a SCRAMBLED
spectrum to a natural caller. That is the recorded 2026-07-29 regression.
`spr >= 0` in the same guard is not an engine at all. The execute-side
switches and the fingerprint bitmap are two more lists that are NOT the
same list.

**L7, refusal (~82 sites).** `-1` and `IL_NONE` are different: -1 means
"unraced, run the heuristic", IL_NONE means "raced, the answer is none —
do NOT". Collapsing them re-runs a heuristic the planner already refuted.

**L8, ladders (~27 sites).** Polarity, above.

### An open defect, found by the survey and confirmed (2026-09-16)

`cfg->recalibrate` does not reach the 2D c2c tier's own lookup. A caller
asking for a recalibration of a 2D c2c cell gets its banked chain, axes and
threading verdict replayed; only the real tier re-races. `cfg` is in scope
at the site and its five siblings in the same file are guarded, so this
reads as an omission rather than a policy.

NOT FIXED HERE. Adding the guard changes behavior: every recalibrate 2D c2c
create would begin racing a chain pool (seconds per cell) and would rewrite
banked verdicts. That is the owner's call, and it is exactly the class of
change this migration refuses to make silently.

Today's 2D re-races are unaffected: they DROPPED the rows from the store
rather than passing recalibrate, so they raced for the right reason.

Narrowed proposals, if the owner wants them later: for L3, a named
`policy_replays_at_T()` for ONLY the five per-thread-count fences, leaving
the recalibrate term at each site; for L6, one shared
`vfft_policy_k1_engine_present()` taking the terms explicitly rather than a
pointer list; for L5 and L7, documentation rather than unification.

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

## Checklist

- [ ] 1. This design (the inventory above is the survey of 2026-09-16).
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
- [!] 5. Steps 4-5: L5, L3, L6, L7 — REFUTED AS SCOPED (2026-09-16, the
      survey below). These are not four laws written many times; they are
      four FAMILIES of similar-looking predicates with load-bearing
      differences. Unifying them as written changes behavior at named
      cells. Narrowed proposals below; the owner's ruling, not a
      continuation.
- [!] 6. Step 6: L8, the ladders — REFUTED AS SCOPED. The L3 gate's
      polarity is the OPPOSITE of the seven L2 gates (it admits what does
      NOT fit) and treats "cache size unknown" as admit, where the L2 ones
      must refuse. One shared `fits()` flips the super-band exactly
      backwards. Leave them separate; the finding is the deliverable.
- [x] 7. Step 7: L10, the IL ROUTE SET is one declaration
      (`VW2_OOP_IL_ROUTE_MAX`, from the enum) and the name table is checked
      against it AT COMPILE TIME. The five hardcoded bounds are gone. The
      two serializers are NOT unified — the survey proves they diverge at
      route 10 — and the bwd builder's lower bound is documented as
      deliberate, not stale.
- [ ] 8. Records: `design_contracts.md` (its tables become the module's
      tables, one reference each), `docs/design/planning_model.md`, memory.
