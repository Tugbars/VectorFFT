# Parity plan — forward/backward, and r2c/c2r against IL C2C

Two parity targets, in dependency order:

1. **Directional parity** — the backward search space should equal the forward
   one, within IL C2C.
2. **Transform parity** — r2c/c2r should reach the same search depth as IL
   C2C, which holds the fullest space in the library.

Status of every claim below: **measured** unless marked ASSUMED. Where a phase
depends on an unmeasured hypothesis, the measurement is its own phase and
gates the work after it.

---

## 0 · What is already true

Recorded first, so no phase re-litigates it.

- **r2c/c2r already inherits the NATURAL half of the IL C2C space.** The zr2c
  child is a full `vfft_create` (`vfft.c:2196-2211`) with `rigor` and `wisdom`
  passed through, `order = NATURAL`, `layout = INTERLEAVED`. Not the scrambled
  half — correctly, because the fold requires natural order.
- **The route IS raced.** `vfft.c:2400-2444` races both routes in-context
  through the real execute path: 9 rounds, alternating arm order, median, 3%
  hysteresis, then banks. What is stale is the STORE (all 37 zr2c cells are
  `src=migrated`, no `ns=`), not the machinery.
- **r2c consumes the forward verdict; c2r consumes the `dir=bwd` sibling.**
  One c2c cell serves both real transforms.

### Reach of any kernel-form work

Natural IL candidates collapse with N, so form parity only reaches r2c/c2r
at small real N:

| real N | child N/2 | child natural IL arms |
|---|---|---|
| 256-2048 | 128-1024 | full Bailey space (pair x kv) |
| 4096 | 2048 | 8 |
| 8192 | 4096 | 1 |
| >=16384 | >=8192 | 0 - natural cascade instead |

**Weight the phases accordingly.** Phase 2 buys nothing above real N=8192.

---

## 1 · Directional parity — the free corrections

No new codelets. Each is a defect or an asymmetry already identified.

| # | change | why |
|---|---|---|
| 1.1 | **DONE 2026-08-22.** Canonicalize the backward pool: `0` plus every variant that is not the structural default; MONO dropped. | Two defects, one fix. (a) At R >= 32 create installs variant 2, so `bkv=0` and `PACK(2,2)` were the SAME plan — the blind grid timed one kernel twice under two labels (`0x00` 914.2 vs `0x22` 876.4, a 4% win over itself) and 7 of 16 arms were duplicates. That explains most of the winner-flapping previously charged to thermals. (b) MONO is odd-count COVERAGE, automatic via `count_ok`; where monolithic genuinely competes is R <= 16 (fits 16 ymm registers) and there it IS variant 0, since create only overrides at R >= 32. Result at 32x32: 16 arms -> **4 distinct**, monotone (939.8 / 960.4 / 984.1 / 1067.4), default wins. |
| 1.2 | `vfft_il2p_apply_kv_forms` returns resolve status | The FORWARD applier still does `if (m) p->mid_f = m;` — a miss silently keeps the default, so two distinct `il_kv` values build the SAME plan and the race banks a verdict for a kernel that never ran. Latent today only because the pools and the registry agree BY HAND. The backward applier was fixed 2026-08-21; this is the same defect class. |
| 1.3 | Backward race alternates arm order, median of 9 | Matches the zr2c route race and the `eng=route` §W2 rule: order bias is tolerable while a verdict dies with the process, but must not be FROZEN INTO A RECORD. The backward pass walks the grid once, in order, and banks. |

Exit: re-run `il_dp_cand_census.exe`, update [il_c2c.md](il_c2c.md) §4/§6,
re-race one cell to confirm the arm count. 1.1 is done and [il_c2c.md](il_c2c.md)
carries the corrected pool; 1.2 and 1.3 remain.

## 2 · Directional parity — tangent backward

The real gap. Every tangent codelet in the tree is forward-only, and tangent
is not a losing arm: it WON the forward at 128 (1.04x), 256 (1.01x) and 512
(0.96x, wing32). The backward cannot currently enter its own best construction
in the race.

**Emission verified by probe** — `--cil-tangent --cil-bwd` composes and emits:

| slot | R=8 | R=16 | R=32 |
|---|---|---|---|
| mid (`--cil-t2 --cil-turnst`) | `t2ttan` | `t2ttan` | `t2btw32` |
| leaf (`--cil-n1`) | `n1tan` | `n1tan` | `n1bw32` |

Six codelets, **variant 3 only**. Two findings constrain this:

- **R=32 needs `VFFT_CX_W32TG` DROPPED.** That knob is guarded forward-only by
  design ("the radix-32 FWD wing combine (hand w32tg pass-B); other
  radices/directions keep butterfly_pair"). Without it the generic
  `butterfly_pair` tangent path emits fine — but the result is a COUSIN of the
  forward variant 3, not a twin.
- **`form_tag_of` will name those files `w32`** (tangent + blocked -> `"w32"`)
  while the body is NOT the wing32 construction. A name that asserts a
  construction the body lacks must not ship: either take a distinct tag or
  state it in the provenance header. **Decide before emitting.**

Steps: emit 6 -> compile all pair combos -> correctness gate -> wire into
`vfft_il2p_t2t_bwd_v_fn` / `vfft_il2p_n1_bwd_v_fn` -> `recipes.tsv` rows ->
corpus cells -> re-race.

## 3 · Directional parity — what is NOT reachable

Recorded so it is not rediscovered as a bug.

- **Variant 4 has no backward expression.** `VFFT_CX_STORE128` is INERT
  backward at R=16 — probe produced a byte-identical file to variant 3 (same
  md5). Shipping both would recreate the "two codes, one plan" hazard 1.2
  exists to prevent. The R=32 edge forms (M-128, T256) ride on the
  forward-only w32tg pass. Full v4 parity needs a backward store-edge
  mechanism in the emitter — separate work, not a flag.
- **Forward blocked at R=64 does not exist.** The asymmetry runs BOTH ways:
  backward has `t2bt416`/`t2bt88`/`n1b416`/`n1b88`, forward has only
  monolithic `radix64_z_{t2,n1t}`. Four codelets. **Low priority** by the
  project's own reasoning — t2t was chosen over t2p because "IL plans favour
  many small stages, so R1=64 is rare".

## 4 · Transform parity — the falsifier that gates it

**The open question: is inheriting the standalone c2c verdict LOSSY?**

The `n=N/2` cell is raced as an isolated transform. In the composite the child
runs adjacent to the fold, with different cache residency, its output feeding
the fold's access pattern. ASSUMED, not measured: the best standalone plan may
not be the best child.

**Experiment** (no new code — the hooks exist): fix a real N in the band where
it can matter (N=1024 or 2048, so the child is in the Bailey tier), force child
forms with `VFFT_IL_KV` / `VFFT_IL_BKV`, time the COMPOSITE through the front
door, and compare that ranking against the standalone c2c ranking at N/2.

- **Rankings agree** -> inheritance is correct. Transform parity is already
  achieved by construction. Record it in [il_c2c.md](il_c2c.md) §7 and CLOSE
  the question.
- **Rankings disagree** -> a real context gap, and Phase 5 becomes live.

Run this BEFORE building anything in Phase 5. Otherwise the architecture is
built on an assumption.

## 5 · Transform parity — only if Phase 4 says so

If context matters, the recipe cannot simply be copied into an r2c cell: the
signpost rule forbids copies, because a copy is stale the moment its source
re-races. Options, to be decided with the measurement in hand:

- **A context KEY on the c2c cell** — the same argument that made direction a
  key: if two callers genuinely want different answers, the discriminator
  belongs in the key. Costs a re-race per context.
- **An r2c-owned override field** — narrower, but reintroduces two places that
  can disagree.

Cost either way: re-running the c2c search per real N, and a second recipe for
the same computation. That is why Phase 4 gates it.

## 6 · Transform parity — the independent items

Neither depends on Phase 4.

- **Re-race the 37 stale route cells.** The machinery is correct and already
  uses the good methodology; the store holds pre-race `migrated` values with
  no `ns=`, including two out-of-pattern c2r entries (n=1024, n=65536) that
  nobody can re-check because no number was recorded.
- **The fold has no axis.** One fixed implementation, so there is nothing to
  race — an axis over a set of size one is not an axis. Making it one means
  WRITING alternatives first: an L1-blocked fold, and a fold fused into the
  child's terminator (the "recombine = structural wall" roadmap item).

---

## Order of work

```
1 (free corrections)  ->  2 (tangent backward)  ->  re-race, measure
                                                          |
4 (falsifier)  -------------------------------------------+
   |                                    
   +-- agree    -> close transform parity, record
   +-- disagree -> 5 (context axis, architecture decided by the data)

6 (route re-race, fold alternatives) — independent, any time
3 — recorded as unreachable / deferred, not scheduled
```

**Do not run phase 2's race and phase 4's falsifier on a thermally noisy
machine.** Both compare arms separated by single-digit percentages; a fixed
control arm has been measured spanning 12% on this box. Every banked verdict
from either phase carries that caveat until re-raced quiet.
