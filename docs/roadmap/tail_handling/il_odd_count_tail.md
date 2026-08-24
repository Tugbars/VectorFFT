# IL odd-COUNT tail — design decision and emitter plan

**Status:** ✅ **IMPLEMENTED 2026-07-29** exactly per §3/§6 (inline narrow arm at
`Isa.sse2`, `?tw_vw`/`?msuf` render plumbing, per-entry const types, log3/mask
twins, turned + leg-strided stores degenerate to 128-bit). Monolithic kernels
only; blocked keeps the even contract (§4d still open). 158 cil kernels
regenerated + the 9 zil-provenance pow2 fwd n1t/t2 swapped to cil (tranche 2
of the pipeline port). Gated by `benches/il_tail_gate.c` (counts 1..8,
canary-separated WRONG / NEVER-WRITTEN / OOB) and end-to-end by the public
gate: all-odd pairs (45 = 9x5, 225, 675), 2·odd pairs (18, 50, 150), and
Rader upgrades (19/29/43) all natural, both directions.
🔴 Wiring lesson the public gate caught: an IL-only PAIR handle must be
CREATED (the guard initially omitted il2p), else such cells fall to the
classic path whose default-order kind can be SCRAMBLED — plan-level gates
could never see that.
**Scope:** the full-IL family (`codelet_cil.ml`). The split family already solves this — see §5.

---

## 1. THE PROBLEM

Full-IL codelets pack `per = vw/2` complex per vector (2 on AVX2) and loop:

```c
for (size_t k = 0; k + per <= count; k += per) { ... }
```

`count` is the number of independent columns processed side by side. When `count` is **not a
multiple of `per`**, the loop stops early and the trailing 1..per−1 columns are **never
written** — no error, no diagnostic, just stale buffer contents.

This is not one kernel's bug: **no** emitted cil kernel has a tail, powers of two included.
For pow2 the case is *impossible*, not solved — every factor comes from {4,8,16,32,64}.

It matters for the two-pass route, where each pass's `count` is the *other* factor:

```c
leaf_f(zin, 0, mid,  0, 0,  0, R1, 0, R2, 0, R1);   /* count = R1 */
mid_f (mid, 0, zout, 0, tw, 0, R2, 0, R2, 0, R2);   /* count = R2 */
```

Any odd factor makes one pass's count odd.

**API-parity note:** a behavioural probe (`mkl_oddcount.c`) confirms MKL accepts an arbitrary
`DFTI_NUMBER_OF_TRANSFORMS` — odd counts included — and is correct on the last transform at
N = 4096, 3072, 96, 15, 45. Whatever we ship must do the same; an even-count contract is not a
defensible public API.

---

## 2. WHY IT IS NOT URGENT — THE TAIL IS CURRENTLY UNREACHABLE

Two gates make an odd `count` impossible to deliver to a cil kernel today:

| gate | file:line |
|---|---|
| `if ((R1 & 1) \|\| (R2 & 1)) return 0;` | `src/core/oop/il2p.h:137` |
| `if (R1c < 4 \|\| R1c > 64 \|\| (R1c % 4) \|\| (R2c % 4)) continue;` | `src/core/vfft.c:3115` |

Neither is wired open. A repo-wide search by **exported symbol** finds zero call sites that can
produce an odd count. **Building the tail before one of those gates opens is groundwork for a
path that does not exist**, and any A/B of it measures nothing (this was confirmed the hard
way — a full BASE-vs-NEW bench of the tail produced only harness noise, because the code under
test could not execute).

🔴 **A design choice upstream may remove the need entirely.** For a mixed size N = odd·pow2,
the odd factor can live in either of two dimensions:

- **batched** — the odd factor becomes `count` (e.g. `leaf(1024)` with `count = 3`) ⇒ odd count
  ⇒ tail required;
- **looped** — the odd factor becomes a trip count over whole-transform calls
  (`for (i=0;i<3;i++) cascade(1024)`), each with `count = 1`, and the combining pass then runs
  at `count = 1024` ⇒ **no odd count anywhere, no tail needed**.

Vector-occupancy favours batching only weakly (3 columns in a 2-complex vector is 1.5 vectors
either way); the looped form's real cost is 3 passes with separate setup and worse locality.
**Decide the odd-radix composition first — it determines whether this work is needed at all.**

---

## 3. THE CHOSEN DESIGN — INLINE NARROW ARM

```c
size_t k = 0;
for (; k + per <= count; k += per) { ...wide body, unchanged... }
for (; k < count; ++k)             { ...same DAG rendered at Isa.sse2, one complex... }
```

- Same op-graph at half width. **No scratch, no call, no duplicated column, no store-forwarding
  hazard, no cold i-fetch.**
- Twiddles are compile-time VLITs in the cil family, so the narrow arm's constants are free.
- Written as `for (; k < count; ++k)` rather than `if`, so it generalises when `per > 2`
  (AVX-512). At `per = 2` the branch runs once and predicts perfectly.
- **Add a low-trip bypass** (`if (count < per) goto narrow;`): a `count == 1` call should never
  enter the wide loop at all.

`Isa.sse2` (`isa.ml:134`) is purpose-built for this and documents the intent:
> *"used ONLY as the arbitrary-K remainder pass on the AVX2 path … emitted INLINE inside the
> enclosing avx2,fma codelet (VEX-128, no AVX↔SSE transition penalty); this record is never
> emitted standalone."*

---

## 4. REJECTED ALTERNATIVES

### 4a. Staged re-entry — BUILT, GATED GREEN, THEN REVERTED
An out-of-line `noinline cold` helper gathered the leftover columns into a stack scratch,
**re-entered the kernel** with `count = per`, then scattered back. It was *correct* — a 672-case
gate (14 radices × 3 kinds × 2 directions × counts 1..8, worst 9.4e-14, guard bands separating
WRONG / NEVER-WRITTEN / OOB) — and it left the hot loop's instruction count **identical** in all
7 sampled kernels.

It was still wrong, for a reason that is architectural rather than incidental:

- **Store-to-load forwarding fails on both sides, by construction.** The gather writes 16-byte
  stores into the scratch; the re-entered kernel then issues **32-byte loads that each span two
  separate 16-byte stores**. A load can forward from at most one store, so it must wait for both
  to reach L1 and re-read — and it gates the whole dependency chain. The output side has the
  mirror problem.
- Cold call into `.text.unlikely` on essentially every use, none of it overlapping the caller.
- The full-width body computes 2 columns and **discards one**.
- **`count == 1` is pathological**: the wide loop never runs, so the entire transform for that
  call goes through staging + calls + scatter.

Estimated ≈150–300 cycles, of which ~100–200 is pure overhead, versus ≈30–90 for the inline
narrow arm. A reference copy of the reverted emitter is kept in
`docs/research/il_tail_handling/artifacts/` (that folder is gitignored).

### 4b. Masking — ALREADY MEASURED AND LOST ON THIS CPU
AVX2 cannot predicate the ALU: `vmaskmovpd` masks loads and stores only, so a masked tail runs
the **full-width body anyway** plus mask setup. And the leftover here is exactly one complex =
exactly one xmm, so there is nothing to mask. `emit_c.ml` records the measurement:

> *"Robustly beats masked vmaskmov at BOTH rem=2 (~−35%) and rem=3 (~−12%, K=7 ~95% win-rate,
> tight-interleaved) — even the 2-pass SSE2+scalar is faster than one vmaskmov pass on Raptor
> Lake, so avx2 carries no masked tail at all (no `_vfft_masklo` table)."*

AVX-512 keeps a masked tail because `vmaskz` is full-rate; AVX2's `vmaskmov` is not.

### 4c. Pad the count to a multiple — REJECTED
Needs buffer slack the caller may not have, and writes outside the caller's logical batch — the
same class of out-of-bounds write the padded-batch work already hit.

### 4d. Keep the even-count contract for `blocked` — ❌ REJECTED, RACED AND LOST (2026-08-23)
Earlier reasoning: `blocked` cil kernels (`n1b`, `t2b`, `t2b_log3`) carry their own per-pass
load/store edges, so there is "no single body to re-render at narrow width."

🔴 **That is likely wrong.** The split family covers composites with `~force_mono:true`
(`emit_c.ml:4022,4053,4059`), which renders the DAG **monolithically** at the narrow width
precisely so *"composite codelets don't reference the `__m256d` spill at width 2"* — and its
contract comment states the tail *"holds for EVERY codelet, monolithic AND composite."*
A single narrow lane has no register pressure, so the CT spill scratch is simply not referenced
and there is no `__m256d`-vs-`double` clash.

---

#### SETTLED. Blocked has the tail; the demotion lost.

The earlier reasoning was indeed wrong, and the alternative was raced rather than argued.

**What the two options actually were.** Keeping the even-count contract did not mean
"refuse odd counts" — it meant **demote to the monolithic twin** whenever the partner
factor was odd, which is what `il2p.h`'s `count_ok` gates did. That silently cost roughly
20 cells of the form `N = 32·odd` / `64·odd` their blocked kernel. The alternative was to
give the blocked forms their own inline narrow arm.

**Why this reaches past c2c.** The interleaved REAL route is `x[N]` reinterpreted as
`z[N/2]` → child `c2c(N/2)` → the `zr2c.h` fold (`vfft.c:2185`, route 0 = OOP-IL child).
When `N/2` lands on `32·odd` or `64·odd`, that child hands its `il2p` codelets an **odd
partner count**. So R2C/C2R is a first-class producer of exactly the case the demotion hit
— this is not a c2c-only question.

**Measured — blocked+tail vs the monolithic demotion**, one process, arms alternated,
correctness re-checked before timing:

| radix | monolithic spill | counts | blocked+tail wins by |
|---|---|---|---|
| 32 | 26.5% | 7 … 32 | **+6 … +26%** (`blocked_vs_mono_race.c`) |
| 64 | **44.9%** | 8 / 16 / 32 | **+13 … +155%** (`r64_blocked_race.c`) |

The spread tracks spill, which is the same law that governs blocking itself: at radix 64
the monolithic form burns 44.9% of its bulk loop on stack traffic, so the demotion was
giving away up to ~2.5× at the largest counts. At radix 32 it is a steady but modest win.
The monolithic arm is also the noisy one — its floor moves run to run while the blocked
arm's is stable — which is what heavy spill traffic looks like.

**Correctness of the shipped form** is gated by `benches/blocked_tail_gate.c`: 81 arms over
r32, r64 and the wing32 tangent forms at counts 1..9, each with a canary prefill (an
unwritten column cannot hold the canary and be correct) and guard bands on both sides.

🔴 The `count_ok` parameter still exists in `il2p.h`'s resolvers, deliberately — kept so a
future tail-less form has somewhere to be refused. It is `(void)`-ed today.
⇒ Blocked cil kernels should be re-examined with `force_mono` before being excluded. Treat the
even-count contract on blocked as an open question, not a decision.

---

## 5. THE PATTERN ALREADY EXISTS IN THE SPLIT FAMILY

`generator/lib/emit_c.ml` §3995-4062 emits exactly this shape, inline, after the rounded-down
bulk loop:

```c
if (k < me) {
    const size_t rem = me - k;
    if (rem == 1) { /* DAG rendered at Isa.scalar, force_mono */ }
    else {
        for (; k + 2 <= me; k += 2) { /* DAG rendered at Isa.sse2, force_mono */ }
        if (k < me)               { /* DAG rendered at Isa.scalar, force_mono */ }
    }
}
```

**This is a port of a proven in-tree pattern, not new design work.** Note §4052/4058: it resets
`hoisted_const_tags` before emitting the narrow body so that pass derives its own constants —
which is also the mitigation for the register-allocation risk in §6.

---

## 6. EMITTER PLUMBING — WHAT ACTUALLY HAS TO CHANGE

The DAG is width-independent; the **rendering** is not.

| coupling | what breaks at narrow width | fix | risk |
|---|---|---|---|
| **`render`'s `CTwL`** (`codelet_cil.ml:603`) | `off = (leg-1)*2*isa.vec_width`, sin at `off + vec_width`. At `Isa.sse2` this computes stride 4 / sin+2 against a table laid out stride 8 / sin+4 — **silently wrong addresses** | add an optional `?tw_vw` (the width the TABLE was built for), defaulting to `isa.vec_width` | **LOW** — `render` is local with **3 call sites** (`:855` blocked, `:1033` monolithic, `:1323` emit_k1); with a default, existing emissions are byte-identical **by construction** |
| `_M_IM` / `_M_RE` masks | file-scope `__m256d`; `render` hardcodes the names | emit `__m128d` twins, make the name width-aware | low |
| `emit_const_decls` | emits one `isa.vec_type` for every entry | keys already differ by lane count (`const_name tbl (vw/2)`), so one table serves both — choose the C type **per entry** from `Array.length w` | low |
| LOG3 prologue | binds `_wc%d`/`_ws%d` at wide width | narrow body needs its own prologue, reading the same **wide-geometry** table | low |
| N1T corner-turn store | `permute2f128` leg-pairing is meaningless at `per = 1` | degenerates to a plain per-leg 128-bit store | low |
| `cflip_pd` | — | already width-correct (0x55 / 0x5 / 0x1) | **none** |

🔴 **The key structural fact — the VTW2 record is ALREADY narrow-readable.** The record is
`[c c c c][-s +s -s +s]`. A 128-bit load at `twp[off+0]` yields `[c,c]` — exactly the cos
broadcast one complex needs — and at `twp[off+4]` yields `[-s,+s]` — exactly the sign-folded sin
pair. **Only the address arithmetic assumes the render width.** So this needs no second table.

**Do NOT build a second, narrow twiddle table.** The table is a *runtime* object (`il2p.h` builds
`tw` and the conjugated `twb`). A second layout is an ABI change across every t2 kernel, doubles
setup cost and memory, for a path taken at most once per invocation.

**Register-allocation risk.** Extra live values after the loop can perturb the hot loop's
allocation (`regalloc.ml` §1/§4; radix-4 grew 3.3× when gcc inlined the staged self-call).
Mitigation: emit the narrow arm in **its own block** after the loop and derive its constants
inside that block, exactly as `emit_c.ml` §4052/4058 already does.

---

## 7. VERIFICATION PLAN

1. **`tail_gate.c`** — 14 radices × 3 kinds × 2 directions × **counts 1..8**, separating
   WRONG / NEVER-WRITTEN / OOB with guard bands. Powers of two are included **on purpose**: they
   never had a tail, so they are the regression surface.
2. **`loopcmp.py`** — extract the **hot loop** by its back-edge (the largest jump whose target
   address is lower than its own) and diff BASE vs NEW. Whole-function diffs are noise: prologue
   and ABI shuffling run once per call. **The loop instruction stream is the number that
   matters** — it must be unchanged, or the RA mitigation above has failed.
3. **Byte-identity regeneration** — regenerate all `codelet_cil.ml`-provenance files and confirm
   the only diffs are the added narrow arm.
4. **End-to-end** — `bench_k1_public` all-green with accuracy figures unchanged.

Tooling for 1–3 is written and lives in `docs/research/il_tail_handling/artifacts/`.

---

## 8. SEQUENCING

1. Decide the **odd-radix composition** (batched vs looped). If looped, this document may never
   need implementing.
2. If batched: open `il2p.h:137` / `vfft.c:3115`, which makes the tail reachable.
3. Then implement §3 with the §6 plumbing and the §7 gates.

Related: `arbitrary_k_tail_strategy.md`, `padding_design_decision.md`,
`../../research/il_tail_handling/` (full record, gitignored).
