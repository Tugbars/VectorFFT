# BOUNDARY-IL / SPLIT-INTERIOR — the cascade family (N ≥ 2048)

**These 48 files are NOT pure IL, and that is deliberate. Do not "fix" them.**

Everything in the parent directory (`../`, 201 files) is **pure IL** — packed complex, re/im
adjacent in the register, through all the arithmetic. Everything *here* is the opposite in the
middle: **interleaved at the buffer boundary, separate re/im planes in the interior.**

They are separated into this folder so that "I saw a codelet in the IL tree and used it" cannot
silently pick up a split-interior kernel. Full context: [`../../README.md`](../../README.md).

## What's here

| role | files | what it does |
|---|---|---|
| **ingest / the turn** | `s0t_r4`, `s0s` | reads **packed** input, writes **split planes** into the scratch. `s0s` = plain deinterleaving leaf; `s0t_r4` = fuses the corner-turn into its *stores* with 4 rate-matched cursors |
| **mids** | `msg` | runs entirely on split planes. Complex multiply = 4 real multiplies, **zero lane operations** |
| **IL-edged mids** | `msz` (3/5/7/9/15, fwd) | the `msg` body with **interleaved z on both edges** (deinterleave on load, reinterleave on store, lanes left unordered: unpack only, no `permute4x64`) — MKL's Fact form on our contract, for the flat odd-N DIT engine (`src/core/oop/il_flatdit.h`). Raced 2026-09-05: 2.1–2.5× on the run-4 stage, parity on long runs |
| **terminators** | `sterm`, `stf_r4` | final butterfly, then **re-interleave on the stores**. `sterm` pays TR4 loads; `stf_r4` needs no load shuffles (ZTURN deleted the load-side TR4 by making the ingest store in the geometry the terminator wants to read) |
| **twins** | `sterm2`, `stf2_r4` | 2-quad unroll-and-jam twins, **bit-identical** to their partners. Winner turns on ±5% placement luck ⇒ the `t2q` pick is measured per cell at create time, never hand-set |

Scratch layout: 64-byte `[re ×4][im ×4]` blocks (blocked SoA), `z` addressing `+4` for the
imaginary half, one stream per leg row. **That layout is why the interior is split.**
Contract: **`count % 4 == 0`** (4 columns per iteration) — note this differs from the pure-IL
family's `% 2`.

## Why split, and why only here

A packed complex multiply must swap re/im within each complex (a port-5 shuffle) on **every**
multiply. A split complex multiply is 4 real multiplies with **no shuffle at all**. Below
N≈2048 the packed layout's locality wins; above it, the freed shuffle port and lower op count
win.

Measured, not assumed: a full-IL cascade interior was built and raced against this one under
**identical chains**, and lost — **6157 vs 5609 ns @4096 (−8.9%)**, **−12.6% @16384**.
Benchmarking the front door against MKL agrees: the crossover between the two layouts falls
between **N=2048 and N=4096**, the same place the internal race flipped sign.

🔴 **A full-IL cascade interior has been refuted twice by independent measurement. Do not
propose it again.**

## The distinction that matters

This is **not** the forbidden `il_in`/`il_out` hybrid shape (which lives in
`codelets/oop/avx2/`). The difference is *how often you convert*:

- **this family converts twice for the entire transform** — once at ingest, once at the
  terminator — so the cost amortises across every stage in between;
- **`il_in`/`il_out` convert at every pass boundary** of a 2–3 pass route.

Same conversion, a fraction of the work to spread it over. That is why these are fine and those
measure 0.51–0.89× against pure IL.

Every file here still takes a single interleaved `zin`/`zout` — none of them expose
`in_re`+`in_im`, which is the test for the forbidden shape.
