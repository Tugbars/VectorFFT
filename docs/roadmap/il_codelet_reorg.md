# Interleaved codelet inventory and folder proposal

**Status: PROPOSAL. Nothing has been moved, renamed or deleted.** This document is for
verification. The edit list in §6 is what would be executed on approval.

**Scope.** Interleaved layout only, AVX2 only. Split is out of scope by instruction.

---

## 1. What exists

603 files in four directories, 66 kind-tag families. Counts are mechanical, from the
filename grammar `radix<R>_z_<TAG>[_bwd]_avx2.c`:

| directory | files | tag families |
|---|---|---|
| `codelets/zil/avx2/pure_il/` | 499 | 45 |
| `codelets/zil/avx2/pure_il/tangent/` | 20 | 10 |
| `codelets/zil/avx2/boundary_split/` | 54 | 9 |
| `generator/generated/fused_codelets/` | 30 | 2 (`ztt_drivers` 15, `zttp_drivers` 15) |

Largest families: `n1` 42, `n1c` 42, `n1t` 40, `t2` 40, `t2c` 34, `t2cp` 30, `t2csg` 30,
`t2csgn` 30, `t2_log3` 24, `t2t` 20, `t2tg` 20, `t2b` 16.

---

## 2. Who owns what

Ownership = the engine header that **declares the resolver returning the function pointer**.
Calling someone else's resolver is clienthood, not ownership.

| route | method | kind tags it declares | notes |
|---|---|---|---|
| 3 | **MONO** | `n1` (fwd), `mono64_8x8_il` | `n1` backward belongs to the pair; `n1c` is shared |
| 5 | **2P_PURE** (pair) | `n1t`, `t2`, `t2t`, `_ct` twins, all `blocked`, all `tangent`, `log3` | `il2p.h` declares the whole slot family |
| 6 | **CHAIN3** | `t2tg` **only** (chain3-exclusive) | otherwise reuses the pair's slots |
| 7 | **PRIME** | **none** | composes via the inner provider; owns zero kernels |
| 8 | **FLAT** | `t2cp`, `t2cs`, `t2csg`, `t2csgn`, `t2csgt`, `t2csgnt` | plus `msz`/`mszt`, declared in `il2p.h` but consumed **only** by `il_flatdit.h` |
| 9 | **ZTURN-T** | `t0tp`, `tmg`, `tlf`, `tlfi` (natural); `t0d`, `tmgd`, `tld` (scrambled) | one stage-table binder, `ztt.h:386-396` |
| 10 | **FS** (four-step) | **none** | a 2D IL child + a hand-written AVX2 transpose (`_k1fs_transpose_range`), not a generated codelet |
| — | **2D IL tier** | `n1c`, `t2c` + `b48/b84/b88/b416` forms | rank-2, not a K=1 route |
| — | **3D IL tier** | **none** | pure composition over the 2D column pass + a 1D row plan |

### Three corrections to what we assumed going in

1. **CHAIN3 does not use `n1c`/`t2c`.** Its four slots are `n1t` (leaf fwd), `t2` (both
   mids), `t2tg` (bwd mid B, exclusive) and `n1` (bwd leaf). The `n1c`/`t2c` families belong
   to the 2D column tier, FLAT and MONO's in-place solo.
2. **2D and 3D have no r2c/c2r codelets of their own.** There is no real-layout kind in
   `zil` at all — `ls … | grep -iE 'r2c|c2r|real|herm'` is empty. The 2D real tier builds
   its column pass from the same `n1c`/`t2c` kernels. That settles the open question.
3. **`tldb` does not exist on disk.** The design doc and `ztt.h` prose name four scrambled
   kinds, but the emitter writes `radix{4,8}_z_tld_bwd_avx2.c` — under the filename grammar
   that is tag `tld`, direction bwd.

---

## 3. Reachability — almost nothing is dead

A first pass flagged ~40 tags as unreachable. An adversarial pass then **overturned nearly
all of them**, and the reasons matter more than the count:

- **The backward form race walks variants 1..5 blind per slot** (`dp_planner_il.h:1187-1191`),
  deliberately, so a newly emitted backward codelet is raceable with no edit. Any backward
  variant is therefore reachable by construction.
- **Arm pools** (`vfft_il2p_mid_arm_pool`, `leaf_arm_pool`) enumerate variants that the
  enumerator crosses into `il_kv` candidates.
- **`vfft_il2p_apply_blocked_default`** installs variant 2 structurally at every R≥32 slot —
  those kernels execute unraced, by construction, including inside the PRIME engine's inner
  pair which applies no `il_kv` at all.
- **An env pin is a reacher.** A pinned kernel is still reachable.
- **"No shipped wisdom row" is not the test.** That is the documented *a banked rule blocks
  its own racer* trap: on a cold store or `--recalibrate` the arm runs. MONO is the clean
  example — zero `il_route=mono` rows exist, yet MONO is the **uncalibrated create-time
  default** at `c2c_oop_create.h:248-252`, so it serves every uncalibrated cell with a solo.

### What survived as genuinely unreachable

| item | files | status |
|---|---|---|
| **`t2_log3` + `t2b_log3`** | **38** | Registry macros exist; **zero consumers anywhere in `src/`**. Confirmed independently twice. The strongest delete candidate. |
| **odd-radix `t2b` / `n1b`** (radices 9,15,21,25,27,45,49) | ~14 | `VFFT_IL_{T2B,N1B}_*_RADICES` expanded by no resolver |
| **`t2tan`/`n1ttan` at radices 6, 10, 12** | 6 | No resolver arm (variant 3 stops at 8/16/32) — but `benches/tangent_gate.c` links and numerically gates all six |
| **`t2cs`** | 15 | Reachable **only** via the `VFFT_ILFD_NO_GEN2` env pin; dead in the default build |
| **`t2p` (`--cil-pretw`)** | 0 | Emitter hard-fails; retired 2026-07-29 |
| **`--cil-k1` fused kernel** | 0 | Emitter kind with no cells |
| **route 4 `VFFT_K1_IL_CASCADE`** | — | No writer, no reader. Engine deleted 09-15; the enum value remains |

**Nothing above is proposed for deletion in this document.** You asked for a report.

### Corpus health — 26 files cannot be regenerated

- Corpus cells: **547** (`zil_pure_cells` 493 + `zil_boundary_cells` 54).
- On disk outside the corpus: **26** — 6 stale `pure_il` strays
  (`radix16_z_n1tb44`, `radix16_z_t2b`, `radix32_z_n1tb48`, `radix32_z_n1tb`,
  `radix32_z_t2b48`, `radix32_z_t2b`) and **all 20 of `pure_il/tangent/`**.
- `boundary_split` is clean: 54 on disk, 54 in the corpus, zero drift.

🔴 **`dir_of_quadrant` has no arm for `tangent/` at all** — the whole 20-file subtree is
invisible to the corpus, to `gen_set` and to `emit_il_registry`, yet CMake globs it into the
library and 14 of the 20 are named by live resolver arms in `il2p.h` with 15 shipped wisdom
rows banking them. **Live, load-bearing, and unreproducible.** A `gen_set` regen would not
recreate them. That is the most serious structural finding in this study.

### Unbanked tiers (cold-store state, not dead code)

`wisdom2_3d.txt` contains **zero `@cell` rows** — the entire 3D IL tier is unbanked on this
host, so every 3D cell races its full pool cold. `il_flat=` / `il_forms=` likewise have zero
rows, so route 8 is unbanked. Of 66 `il_route` rows shipped: `2p` 34, `ztt` 16, `chain3` 8,
`fs` 8 — and never `mono`, `flat` or `prime`.

---

## 4. Proposed structure

Three structures were designed independently and judged. The winner organises **by method at
the top**, with form variants nested under their owner and a `shared/` tier for the two
families the headers themselves declare twice.

```
codelets/zil/avx2/                    575 .c   (573 today + 2 moved in)
  README.md                           the IL route table, routes 0..10
  mono/                                 2      <- moved in from codelets/oop/avx2
  pair2p/                             126      n1t 40 · t2 40 · t2t 20 · t2_ct 9 ·
                                               n1t_ct 9 · t2t_ct 4 · n1_ct 4
    blocked/                           46
    tangent/                           20      <- was pure_il/tangent/
    log3/                              38      t2_log3 24 · t2b_log3 14
  chain3/                              20      t2tg
  flat/                               135      t2cp · t2csg · t2csgn · t2cs · t2csgt · t2csgnt
    odd_mid/                           15      <- was boundary_split/; msz 10 · mszt 5
  ztt/                                 39      <- was boundary_split/; tmg · tmgd · t0tp ·
                                               tlf · tlfi · tld · t0d
  shared/                              42      n1 (fwd = MONO, bwd = pair/chain3)
    col/                               76      n1c 42 · t2c 34
      blocked/                         16      n1cb{48,84,88,416} · t2cb{48,84,88,416}

generated/fused_codelets/              30      UNMOVED — derived output with its own dune
                                               promote rule; ztt/README.md points at it
```

No directory for routes 4, 7 or 10 — they own no kernels. They get rows in the top README.

**Why this one.** It is the only proposal whose ownership rule is *read out of the code*
rather than imposed on it. `il2p.h:398-405` literally disclaims ownership of `n1c` in its own
comment; `n1` really is declared twice, in `oop_leaf_registry.h:386` and `il2p.h:388`. So
`shared/` contains exactly the families the headers declare twice — a reading, not a
judgment call.

It also **fixes the `boundary_split` lesson rather than repeating it**: `msz`/`mszt` move to
`flat/odd_mid/` because their sole consumer is `il_flatdit.h` and `ztt.h` never names them,
which is exactly why today's folder name is 28% wrong. And the 38 orphan `log3` files get
their own directory — physically separated as an honest delete candidate instead of hidden
inside a 499-file bucket.

**Accepted weaknesses.** `pair2p/` swallows 230 of 575 files, because routes 5, 6 and 7
genuinely are one engine header. The ownership rule needs two named exceptions (`t2tg` and
`msz`/`mszt` are declared in `il2p.h` but filed under their sole consumer). And +9 quadrants
is a permanent tax — see the required graft below.

---

## 5. Two grafts the judge made preconditions

> 🔴 **Graft 1 is largely dissolved by §11.3.** The registry is derived from the argv cell
> tables, not from disk, so it is invariant under the move; keeping the corpus at 2 quadrants
> removes the forgotten-quadrant hazard rather than generalising around it. Graft 2 (the TSV
> invariant) stands, and §11.7 strengthens it with an `nm` symbol-set check.

1. **One `Corpus.il_quadrants` definition**, consumed by `corpus.ml`,
   `emit_il_registry.ml` and both build files. `emit_il_registry.ml:77-81` currently
   hard-codes `[ "zil-boundary"; "zil-pure" ]`; a forgotten quadrant drops that kind's
   X-macro, which makes a guarded resolver arm compile to `default: return 0` and fail **at
   runtime as a missing kernel**. Going from 2 quadrants to 11 multiplies that surface
   fivefold. Generalising `dir_of_quadrant`'s fallback also *removes* the loud `failwith` a
   three-segment name hits today — so the cheap diff trades a compile-time failure for a
   silent one unless this graft lands with it.
2. **The gate TSV invariant.** `generator/gates/recipes.tsv` (270 zil rows),
   `baseline_manifest.tsv` (573) and `baseline_verdicts.tsv` (297) all carry paths. Invariant
   to assert across the move: *the multiset of (sha256_lf, basename) pairs is identical
   before and after*.

---

## 6. The exact edit list

> 🔴 **Items 1, 2, 4 and 5 are superseded by §11** — read that first. In short: the corpus
> stays at **2 quadrants** with a per-file classifier (so `emit_il_registry.ml` needs no
> change at all), `dir_of_quadrant` keeps its `failwith`, and the build globs must **not**
> become recursive.

1. **`corpus.ml:2437-2449`** — delete the `"zil-boundary"` and `"zil-pure"` arms; generalise
   the fallback from `[fam; isa] -> fam ^ "/" ^ isa` to a join of all `-`-separated segments,
   so quadrant names *are* their paths. Verified safe: `gen_set.ml:16-22`'s `mkdir_p` is
   recursive and `gen_set.ml:63` concatenates root with `dir_of_quadrant q`. Net −2 lines.
2. **`corpus.ml`** — add 11 quadrant names; add a `zil_dir_of_file` classifier and make the
   `matrix_files` arms one-liners. **Do not split `zil_pure_cells`** — all 547 recorded argv
   rows stay byte-for-byte where they are, which is what preserves reproducibility.
3. **`emit_il_registry.ml:77-81`** — consume the shared quadrant list instead of its
   hard-coded pair.
4. **`build.py`** — switch the zil include walk to `rglob` (precedent at `build.py:363-364`).
5. **`CMakeLists.txt`** — `file(GLOB_RECURSE)` for the zil tree, and update
   `CMakeLists.txt:201-202`'s `VFFT_ABSENT_avx512` list, which names all three zil dirs.
6. **`git mv`** the 573 files into the tree above, plus the 2 mono files in from
   `codelets/oop/avx2`.
7. **12 `README.md` files**, one per directory (§7).
8. **Re-path** the 26 corpus-invisible rows in the gate TSVs by hash, then assert the
   invariant from §5.

---

## 7. README plan

Every directory README states the same five things, so they are comparable at a glance:

1. **Which route(s)** resolve these kernels, with the enum value and the resolver `file:line`.
2. **What the kind tag means** — one line per tag: its role in the algorithm, its radix set,
   its directions.
3. **Which wisdom token** selects it, and what the token's values mean.
4. **Who else consumes it** (for `shared/`), or "exclusive".
5. **Reachability status**, naming any tag in that directory currently unreachable and how
   it is reached if at all (env pin, gate, race arm).
6. **The optimization angle** — which of the eight problems in §9.1 this family attacks, the
   mechanism in two or three sentences, and the measured payoff with its scope condition.
   §9.2 and §9.3 hold the verified text for each; §10 says how often the family is actually
   selected in the shipped store.

The one sentence that differs per folder is the first: *"These are the kernels of
`<method>`, resolved by `<header>:<line>`, selected by `<token>`."*

`shared/README.md` additionally carries the ownership rule itself, since it is the directory
that exists because of it. `pair2p/log3/README.md` states plainly that its 38 files have no
consumer in `src/` and are a delete candidate pending your ruling.

---

## 8. Rulings needed

1. **Route 4 `VFFT_K1_IL_CASCADE`** — deliberate wisdom-compat like routes 1 and 2, or a
   leftover to retire?
2. **The 38 `log3` files** — delete, or keep pending a re-race? They have no consumer at all,
   so the sunset policy's "delete after the re-race" has nothing to race.
3. **The 26 unreproducible files** — the 20 tangent files are live and load-bearing, and §10.2
   now puts a number on it: the tangent family holds **52% of all banked pair kernel slots**,
   more than the monolithic default. The library's most-selected non-default kernel family is
   the one a `gen_set` regen would not recreate. Do you want them brought *into* the corpus
   (a `tangent` quadrant + recorded argv), or left outside it deliberately?
   🔴 **This one is a blocker, not just a preference** — §11.4: the per-file classifier must
   return a directory for every file on disk, so phase 1 cannot be specified until this is
   answered either way.
4. **`t2cs`** — keep the `VFFT_ILFD_NO_GEN2` escape hatch, or retire the kind with the pin?
5. **`b416`** — the r64 column form has **never won a banked cell** (§10.3: `b88` 12, `b48` 4,
   `b84` 1, `b416` 0). Under the pool sunset policy that is a re-race-then-retire candidate,
   unlike `log3` it *does* have a racer. Retire, or keep racing?
6. **`il_codelet_design.md` §3** — ~130 lines describing the deleted cascade in the present
   tense (§9.5). Mark historical, or excise?
7. **Execute?** On your word I run §6 as one mechanical change, with the §5 invariant asserted
   before and after and a build + gate sweep to confirm.

---

## 9. Optimization angles — what each family actually solves

Established by a 19-agent study (11 mechanism clusters, each claim re-checked by an
adversarial verifier against the emitted `.c` and the design records). Every claim below
survived that check; the corrections the verifiers forced are recorded in §9.4, because
several of them are load-bearing.

### 9.1 Eight problems, not eight speed ideas

| # | problem | attacked by |
|---|---|---|
| **P1** | **16 ymm against R live complex values.** The emitter measures its own stack traffic: r8 12-14, r16 35-53, r32 158-197, r64 537-554 (`c2c_il.ml:493-494`). As a share of the bulk loop: r3/4/5/7/8/9 **0.0%**, r16 7.6%, r21 27.3%, r25 37.6%, r27 39.3%, r32 26.5%, r64 34.8% (`il_codelet_design.md:141-155`). | `blocked` (`n1b`, `n1tb*`, `t2b*`, `t2bt*`, `n1cb*`, `t2cb*`), `_ct` |
| **P2** | **The four-step transpose pass.** A Bailey pair needs a 2N-double read + 2N-double write between stages, over a buffer that must exist. Two passes cannot pay a layout conversion back (`il2p.h:29-31`). | the **turned store** (`n1t`, `t2t`, `t2tg`) |
| **P3** | **Cross-lane shuffles in the packed complex multiply.** Interleaved `[re,im,re,im]` costs a lane swap per complex multiply; split `[re x4][im x4]` costs none. | the **boundary-split interior** (`tmg`, `tmgd`, `msz`, `msgb`) |
| **P4** | **Twiddle-load bandwidth and geometry.** | `log3` (refuted in IL), the 2D column hoist (`t2c`) |
| **P5** | **Per-call overhead once the run gets short.** Always a symptom, never a root cause — fixed the same way every time: put the loop inside the kernel. | `msz` block loop, `t2csgn` group loop, the fused drivers |
| **P6** | **Execution-port balance.** Naked butterfly adds congest ports 0/1/5. | `tangent` (`e^-iθ = cosθ·(1 − i·tanθ)`), and the store edge `t256`/`m128` it creates |
| **P7** | **Cache-set collapse on a power-of-two stride.** *Not in the original brief, and the largest single measured factor in the tree.* At 2048×2048 R=32 the natural leaf's output stride is 2 MB; 32 output streams contend for 12 and 16 ways — natural **52.3 ms vs scrambled 15.6 ms** serial at 64 MB, a 3.4× penalty that was pure addressing (`il2d_natural_leaf_design.md:20-32`). | the staged natural leaf (`nls=`) |
| **P8** | **Alias tolerance as a type qualifier.** Not performance — legality. `f(z,NULL,z,NULL)` puts one pointer into two `__restrict__` parameters. Cost of the fix: **exactly zero instructions.** | `n1c` / `t2c` |

Two candidate problems were dropped. *Per-call overhead* is P5 — always a symptom.
*Ordering/permutation cost* turns out to **be** cache behaviour, and the tree states the
rule as such: digit-reversed **read** 0.96–1.12× = free; digit-reversed **write** +29–50% =
never (`il_codelet_design.md:296-302`). That is a read-for-ownership argument, not a
permutation-arithmetic one.

### 9.2 Method → problem

- **mono** (`n1`, `n1c`) — answers **the fixed cost of decomposing at all**. A pair plan must
  allocate a 2N-double `mid` plus two VTW2 tables and cos/sin-fill them at create
  (`il2p.h:1150-1170`), and the leaf must store all N points before the mid loads them back.
  Mono pays none of it: `Ls = OLs = 1, count = 1`, no table argument, twiddles as
  compile-time `set1_pd`. It dies exactly where peak live (= N) outgrows 16 registers — r64
  spills 34.8%. Second, unadvertised price: `count = 1` means the wide loop never runs and
  the transform executes in the VEX-128 tail at one complex per register
  (`oop_leaf_registry.h:355`) — **a solo gets DAG ILP and zero lane parallelism.**
- **pair (il2p)** — answers **P2**, and its whole existence is a store-addressing trick:
  `n1t` writes `(leg p, col k) → zout[2*(k*OLs + p)]` so stage 2 reads columns contiguously.
  Measured price of the corner turn: **exactly R `vperm2f128` and zero memory operations**
  (r8: `n1` has 5 shuffles / 0 `vperm2f128`; `n1t` has 13 / 8 — loads, stores and FMA
  identical). Bounded by L1: the crossover lands at 48 KB = in+mid+out at N=1024 = this
  machine's L1d.
- **chain3 (il3p)** — answers a **parity wall**, not a speed problem. Every cil kernel packs
  2 complex per ymm and needs `count % 2 == 0`; in a 2-stage pair the leaf runs at
  `count = R1` and the mid at `count = R2`, so **a 2-stage plan can never host an odd
  factor** (`il2p.h:1375-1384`). Three stages pin the SIMD axis to the leaf's q columns, so
  every count is even and odd factors appear only as kernel *radices*. `t2tg` exists purely
  to make that chain's backward addressable — it wires the otherwise-`(void)`'d `OGs` as a
  leg stride, buying the group transpose as caller address arithmetic instead of a separate
  scatter pass over 2N doubles.
- **flat DIT** — answers **odd N with no efficient ingest**: ZTURN-T's ingest takes a
  power-of-two count of items, odd N needs a sweep-per-factor ladder, and that ladder's
  pathology is P5. (This is the owner's own framing, and it is confirmed.)
- **ZTURN-T natural** — **ordering with no reorder pass.** `t0tp` parks column c's run at
  `rb[c]` so later stages merge adjacent runs; `tlf` does a REINT packed store straight into
  natural order. Ingest verified twiddle-free: `radix8_z_t0tp_avx2.c` has **0 FMA** and 4 mul
  (only the √½ folds). Plus P3 throughout — `tmg` at zero shuffles per column.
- **ZTURN-T scrambled** — **deletes one whole sweep of the array**, under a stated condition.
  The natural terminator's legs are N/R apart, so it sweeps the plane after the tiles are
  done: **17–28% of runtime at every tiled cell**. Scrambled's last stage has `b = 0` —
  twiddle-free, legs adjacent, in place (`radix8_z_tld` 0 mul / 8 FMA against `tlf`'s 14 mul
  / 22 FMA). **But see §9.4(c): it pays a transpose network for that.**
- **four-step** — answers **N above ZTURN-T's 262144 ceiling**. No new kernel family: steps 1
  and 3 are the existing column chain and row children, step 2 fuses into the one seam every
  row already passes through, `_il2d_row_exec`.
- **2D column (`n1c`/`t2c`)** — answers **the convert-wrapper tax**, 1.33–1.50× across the
  cell map, twice reproduced. Its own lever is P4: a 2D stage's twiddle never depends on the
  vector axis, so records hoist out of the column loop entirely and table traffic collapses
  against Bailey's per-lane VTW2 stream.

### 9.3 The cross-cutting axes, with verdicts

| axis | attacks | verdict |
|---|---|---|
| `blocked` | P1 | **Vindicated conditionally.** `t2b48` −18…−20% at kernel level, −5…−14% through `execute_fwd`, 3/3. Lost at odd radices (`n1b` +13.5%). The split digits are a **raced parameter, not a rule** — `b48` and `b84` both give peak live 8 at R=32, so register arithmetic cannot choose between them. |
| `tangent` | P6 | **Vindicated at R8/R16, refuted as a naive port to R32.** −25% R16 mid, −19.8% R16 leaf; R8 is bit-identical to the classic kernel (the fused multiply is by exactly ±1). R32 returns only −3.2% despite a *better* port mix, because the extra constants cost +9 spill stores, +9 spill loads and +24 register moves. Scope is L1-resident. |
| store edge (`t256` / `m128`) | P6's consequence | **Split.** `t256` was dp-promoted at both raceable cells (N=128: 63.6 ns; N=512: 301.1 ns). `m128` **lost at every cell it can serve** — the split stores it deletes are cosmetic (`bound-on-stores` ≤ 0.037). The R32 paired-permute leaf edge lost by **+32.4%**: *the tax was the store edge, not the interior.* |
| `log3` | P4 | **Refuted in IL, live in `split_oop`.** The mechanism is real and large (r27: 52 loads → 10) but IL's flat twiddle is already a zero-register memory operand, so derivation buys loads with registers on a machine with none to sell: **+11% ops at r27, +21% at r9**, visible in the files themselves. |
| `_ct` | P1 for odd composites | **Tracks spill almost linearly.** r25 (37.6% spill) 2.5×, r27 (39.3%) 2.2×, r21 (27.3%) 1.40–1.60×, r15 (16.5%) 1.32–1.35×, **r9 (0.0% spill) loses 0.89–0.94×** — and the forward resolver excludes R=9 for exactly that reason. |
| pre/post twiddle | — | **A legality axis, not an optimization axis.** Placement travels with (direction, sign); a control sweep perturbing one argument at a time gave O(1) error for *every* perturbation. `--cil-pretw` is now a `failwith`. |
| turned store | P2 | **Vindicated unconditionally**, and it is what made the canonical backward `t2t` expressible at all, by decoupling store form from kind. |

**The one-line rule the whole corpus reduces to** — measured, not assumed: *blocking and
factoring pay exactly when the monolithic form spills, and the payoff scales with how much;
where there is no spill there is nothing to recover and the extra passes are pure loss*
(`il_codelet_design.md:157-160`). Hence R≤8 never blocked, R=16 a raced coin flip, R≥32
blocked structurally.

### 9.4 Corrections the verification pass forced

Each of these was a plausible story that the code does not support.

- **(a) `t2cs`'s "180 µs → 104 µs" does not exist.** Those figures appear nowhere in the
  repository. The only recorded count-1-last-stage measurement is 1.7 → 0.9 ns per point at
  10⁵ points (`odd_n_engine.md:463`), and it belongs to `t2csgn`'s `o` form. Also `t2cs` is
  the **fallback**, not the tail default, and its branch sets `bwd_ok = 0` and `scr_ok = 0`.
- **(b) `n1b`'s spill story is refuted by its own cited source.** `CODELET_TAXONOMY.md:57`:
  *n1b has the lowest spill of any blocked form — 7.8% — and lost anyway. Whatever costs it
  the 13.5% is not register pressure and remains unexplained. Do not cite this row as
  evidence that blocking fails at odd radices; cite it as an open question.*
- **(c) The scrambled saving is not a kernel-level shuffle saving.** Counted at matched
  radix: `tld` vs `tlf` is 24 vs 16 shuffles at r4 and 48 vs 32 at r8 — the scrambled kernel
  has **more** shuffles, and fewer total intrinsics (48 vs 52). The design says so directly:
  *tld carries 1.5 shuffles per complex where tlf carries 0.75*, and below the tile band the
  class is ~11% slower than natural in both modes. The win is the **absent pass at pipeline
  level**, which is exactly why the shipped verdict is in-place-only.
- **(d) `__restrict__` no longer discriminates anything.** No cil codelet in the `t2c`/`t2b`
  family carries the qualifier at all now. What makes the 1D kernel unusable in a 2D column
  stage is the **stage contract** — `Ls = D*N2`, `Gs = N2` row pitch, `OGs = D`, and a
  driver-built d-major broadcast table hoisted out of the column loop — a different geometry
  and a different twiddle-sourcing contract, not a pointer-qualifier difference.
- **(e) `b88` / `b416` at r64 are different factorizations** (8×8 vs 4×16, peak live 8 vs 16,
  a 4-instruction arithmetic difference). Only the r32 pair `b48` / `b84` is the
  same-factorization-two-orderings case where geometry alone decides.
- **(f) The fused drivers do not carry a twiddle cursor in a register.** They pass
  compile-time displacements off one base (`tw + 0`, `tw + 112`, `tw + 1008`); `grep 'tw +='`
  over all 30 driver files returns zero hits. The carried cursor is design intent; the
  contiguous stage-order stream is what was actually built.
- **(g) Open contradiction, left open.** The cluster study says `n1` issues all N leg loads up
  front (citing `c2c_il.ml:1226-1238`); the taxonomy agent measured **0 loads before the first
  arithmetic op** in `radix8/16/32/64_z_n1_avx2.c` and concluded the SR scheduler interleaves
  them. Both cannot be right about the emitted file. The register-pressure story does not
  depend on it — that rests on the compiled-assembly spill table — but **the source-ordering
  claim should not be repeated until someone settles it.**

### 9.5 Stale-documentation finding

`docs/design/il_codelet_design.md` §3 describes the **deleted cascade** in the present tense
across ~130 lines. It is the largest stale artifact in the tree and it reads as live —
`boundary_split/` now holds the ZTURN-T stage kernels, not cascade kernels. Flagged, not
touched, under the report ruling.

---

## 10. Cross-check: which codelets are in the winning cells

Decoded from the shipped store, `generated/wisdom2_*.txt` — **734 banked rows**.

### 10.1 Route census

| route | rows |
|---|---|
| `2p` (Bailey pair) | 34 |
| `ztt` (ZTURN-T) | 16 |
| `chain3` | 8 |
| `fs` (four-step) | 8 |

Of the 34 `2p` rows, **23 carry an `il_kv`** → 46 kernel slots (one mid and one leaf each).
That is the population below.

### 10.2 The headline: tangent is the most-banked non-default kernel in the library

Decoding `il_kv` through the variant tables (`il2p.h:673-721`, `:743-788`):

| slot family | slots | share |
|---|---|---|
| **tangent (v3) + tangent M-128/T256 edge (v4)** | **24** | **52%** |
| monolithic default (v0) | 17 | 37% |
| `_ct` odd-composite (v5) | 4 | 9% |
| blocked (v1/v2) | 1 | 2% |

By slot: mids are v0 13 / v1 1 / v3 9; leaves are v3 11 / v0 4 / v4 4 / v5 4. The tangent
family is selected in **more than half of all banked pair slots**, and it dominates the leaf
slot outright.

**This is the finding that should drive the reorganisation.** The tangent folder is precisely
the one that sits entirely outside the corpus: no `dir_of_quadrant` arm, no registry macro,
absent from both `zil_pure_cells` and `zil_boundary_cells` — **a `gen_set` regen would not
recreate it.** The most-selected non-default kernel family in the shipped store is the one the
generator has forgotten how to produce.

### 10.3 Other axes

- **2D column `forms=`**: `b88` 12, `b48` 4, `b84` 1, **`b416` zero**. The r64 pool is raced
  and one side has never won a cell. Nine rows carry an empty `forms=`.
- **`il_ztt=` chains** (16 rows) are diverse — `4.8.8.4`, `4.4`, `8.4.4.3.8`, `8.8.8.4`,
  `8.8.8.8`, `8.8.4.4.8` — so the ZTURN-T stage kernels are exercised across radices 3/4/8,
  not one frozen chain. `il_tw=` banks 1024 (×3), 2048 (×3), 3072 (×1).
- **`il_mt=`** appears on 10 rows (2048 ×5, 512 ×2, 256, 2, 1).
- **`il_c3=` and `il_flat=` bank on zero rows.** CHAIN3 and flat DIT have banked *route* rows
  but no banked per-slot form — their form axes are cold-store, consistent with §3.5's
  unbanked-tier finding. This is **not** evidence their codelets are dead; it means those
  tiers have not been calibrated on this host.

### 10.4 What the cross-check does *not* say

A codelet absent from the banked store is **not** thereby unused. The store is one host's
calibration; the four documented reachers from §3 (the blind backward 1..5 sweep, arm pools,
`vfft_il2p_apply_blocked_default`, env pins) all bypass it; and a banked rule blocks its own
racer, so a form can be unbanked precisely *because* something else is banked. The
cross-check is evidence **for** what is load-bearing, never evidence against what is not
listed.

---

## 11. Code-change plan for the move

Written after reading every path-bearing consumer. **It revises §5 and §6 in two places** —
both revisions make the change smaller, and both are noted inline.

### 11.1 What actually consumes a codelet path

Four systems independently re-derive "which directories hold zil codelets". Today each
names **3** directories. The proposed tree has **11**. That multiplication, not the `git mv`,
is the whole risk.

| # | consumer | how it names dirs | what a mistake does |
|---|---|---|---|
| 1 | `corpus.ml` — `quadrants` (`:2414-2433`) + `dir_of_quadrant` (`:2437-2450`) | two separate literals that can disagree | `gen_set` writes to the wrong dir, or `failwith` |
| 2 | `emit_il_registry.ml:77-81` | hard-codes `[ "zil-boundary"; "zil-pure" ]` | a dropped tag ⇒ `VFFT_IL_<TAG>_*_RADICES` undefined ⇒ **`il2p.h` fails to compile** (108 X-macros are consumed there) |
| 3 | `build_tuned/build.py:57-73` | explicit `dirs` list; missing dir ⇒ **`[warn]` on stderr only** | a family silently vanishes from the archive |
| 4 | `CMakeLists.txt:200-272` | explicit `_zdir` loop + `VFFT_ABSENT_avx512` | guarded — see 11.2 |

Plus three path-keyed TSVs — `recipes.tsv` (1437 rows), `baseline_manifest.tsv` (1740),
`baseline_verdicts.tsv` (1463) — all keyed on the **relative path** as column 1.

### 11.2 🔴 Reject §6 items 4 and 5: do not make the build globs recursive

§6 proposed `rglob` in `build.py` and `file(GLOB_RECURSE)` in CMake. **Reading
`CMakeLists.txt:186-195` shows why that is exactly backwards.** The per-directory loop and
its zero-file `FATAL_ERROR` exist because of a real incident, documented in the file:

> *this file globbed 7 families and printed a single plausible-looking "codelets: 598",
> while the three zil dirs (265 files) were absent for weeks — one aggregate number cannot
> show you a family that is missing.*

A recursive glob restores precisely that failure mode, and it would do so at the moment the
directory count goes 3 → 11. `build.py` is worse still: a missing directory there is a
`[warn]` to stderr, not an error.

**Keep the enumerate-and-count-each-directory shape.** The list grows from 3 entries to 11 in
both files; the guard then protects 11 families instead of 3. That is the change being
bought.

### 11.3 🔴 Revision to §5/§6: keep the corpus at 2 quadrants, classify per file

§6 item 2 wanted 11 new quadrant names *and* an unsplit `zil_pure_cells`. Those are in
tension: `gen_set.ml:63` places files with `Filename.concat !root (dir_of_quadrant q)`, so
one quadrant is one directory — 11 directories would force the 547-row cell table to be
partitioned, which is the thing that preserves reproducibility.

The resolution is the classifier §6 already gestured at, taken further: **`gen_set` places
per file, not per quadrant.**

```ocaml
(* corpus.ml — ONE literal; quadrants and dir_of_quadrant both derive from it *)
let quadrant_dirs : (string * string) list =
  [ "zil-pure",     "zil/avx2/pure_il"       (* base dir; subdir per file below *)
  ; "zil-boundary", "zil/avx2/boundary_split"
  ; ... ]

(* NEW: the only function that knows the 11-way split *)
val zil_subdir_of_file : string -> string   (* "radix32_z_t2b48_avx2.c" -> "pair2p/blocked" *)
```

Consequences, and they are all reductions in scope:

- **`emit_il_registry.ml` needs no change at all.** It consumes `Corpus.files q` — the
  *argv cell tables*, never the disk — so its output is invariant under the move. §5's graft-1
  hazard ("a forgotten quadrant drops a kind's X-macro") **evaporates**, because the quadrant
  count stays at 2. Verified: the generated header's provenance line reads *"from
  `Corpus.files "zil-boundary" + "zil-pure"`"*, and `parse_stem`'s `dir` is the *direction*
  (`` `Fwd``/`` `Bwd``), not a directory.
- **`zil_pure_cells` stays one table, byte-for-byte.** All 547 argv rows keep their recorded
  form, which is what the 97.97% reproducibility result rests on.
- **`dir_of_quadrant`'s `failwith` stays.** §6 wanted to generalise the fallback to a
  segment-join; that trades a loud compile-time failure for a silent wrong path. Instead
  derive both `quadrants` and `dir_of_quadrant` from the one association list above and keep
  `failwith` for anything not in it.
- Only **`gen_set.ml:63`** changes shape: compute the directory per file, `mkdir_p` it once
  per distinct value. `mkdir_p` is already recursive (`gen_set.ml:16-22`), so nested paths
  need nothing else.

### 11.4 🔴 Blocker: ruling 3 gates this phase

The classifier must return a directory for **every** file on disk. 26 files have no corpus
row — including **all 20 tangent files**, which §10.2 shows hold 52% of banked pair slots.
So:

- If ruling 3 is **"bring them into the corpus"**: the argv rows must be recovered first, and
  the classifier is total. Clean.
- If ruling 3 is **"leave them outside"**: `tangent/` is a directory the corpus does not know
  about, and any expected-count check must carry an explicit *"unreproducible, expected 20"*
  declaration — the same "an absence nobody wrote down is a hard error" posture `CMakeLists.txt`
  already takes with `VFFT_ABSENT_avx512`.

**Either answer is workable; the phase cannot be specified until one is given.**

### 11.5 Two hazards I checked and can rule out

- **Stale object reuse is self-healing.** `git mv` preserves mtime, objects are named flat by
  stem (`build.py:252`), and the cache prunes by object name — so a moved file looks like a
  cache hit. But `_is_stale` iterates `[src] + deps` (`build.py:162-167`), the depfile lists
  the *old* source path, that path is now gone, `d.stat()` raises `OSError`, and the handler
  returns `True`. Every moved file rebuilds. **No `.obj/` wipe is required.**
- **No basename collision.** `build.py` documents relying on unique basenames across families.
  Measured: **0** duplicate basenames within the avx2 compiled set. The reorg moves and never
  renames, and the 2 `mono` files coming in from `oop/avx2` were already being compiled, so
  the compiled set is unchanged.

One hazard that is real and cheap: `file(GLOB)` has no `CONFIGURE_DEPENDS`, so **an existing
build tree must be re-configured** after the move or it will link the old file list.

### 11.6 Phase order

The ordering point is that **phase 2 lands before the files move**, so the refactor is
provably a no-op against an unchanged tree. If the binary changes, it was the refactor; if it
changes after phase 3, it was the move. They never confound.

| phase | work | exit check |
|---|---|---|
| **0 — freeze** | Record the multiset of `(sha256_lf, basename)` over all 573 zil files, and `nm --defined-only` the codelet archive. | Baseline stored outside the tree. |
| **1 — collapse the duplication** | `corpus.ml`: one `quadrant_dirs` literal; `quadrants` and `dir_of_quadrant` derive from it; add `zil_subdir_of_file`. `gen_set.ml:63`: place per file. | `gen_set.exe --root <scratch> all` reproduces the tree **in its current shape** — classifier returns today's 3 dirs. |
| **2 — teach the build the 11 names** | `build.py` `dirs` list 3 → 11 entries; CMake `_zdir` loop 3 → 11 and `VFFT_ABSENT_avx512` updated. Directories do not exist yet, so each is declared absent. | Build succeeds; **`nm` symbol set identical to phase 0**; both totals unchanged and equal. |
| **3 — move** | `git mv` 573 files + 2 `mono` files in. Remove the temporary absent-declarations. | CMake per-family counts sum to the same total; `nm` identical again. |
| **4 — re-path the harness** | Rewrite column 1 of the three TSVs by matching `sha256_lf`, not by string-editing paths. | The §5 invariant: multiset of `(sha256_lf, basename)` identical to phase 0. Full gate sweep. |
| **5 — prose** | 12 `README.md` files (§7); update the 4 in-tree comments that name a zil path (`il2p.h` ×2, `ztt.h`, `dp_planner_il.h`), 2 `dune` comments, `tangent_gate.c` ×2, `build_tuned/README.md`, `CODELET_SET.md` ×4, `CODELET_TAXONOMY.md` ×8, and ~31 mentions across `docs/design`, `docs/performance`, `docs/roadmap`. | `grep -r 'pure_il\|boundary_split'` returns only intended hits. |

### 11.7 The verification that actually catches a dropped file

Counts can coincide; symbol sets cannot. The load-bearing check at phases 2, 3 and 4 is
**`nm --defined-only` over `libdagcodelets.a`, diffed against the phase-0 baseline**. A
codelet that silently stops compiling is a missing symbol, and that is invisible to any file
count. Per the standing rule: verify by content, `nm` the binary.

Secondary, in order of strength: the `(sha256_lf, basename)` multiset; `gen_set` into a
scratch root diffed against the real tree; CMake's per-family counts against `build.py`'s
total; the gate sweep last.

### 11.8 Net edit size

Six source files change: `corpus.ml` (one literal replaces two, plus a classifier),
`gen_set.ml` (one line), `build.py` (a list), `CMakeLists.txt` (a list and a declaration),
plus the three TSVs rewritten mechanically by hash. **`emit_il_registry.ml` does not change**,
which is the main thing 11.3 buys. Everything else is `git mv` and prose.

---

## Method

25 agents: 13 parallel tracers (one per method, plus the OCaml generator side and a shipped
wisdom census), an adversarial pass over every unreachable verdict, three independent
structure proposals, and a judge. The file/tag census and the corpus cross-check in §1 and §3
were computed mechanically and independently of the agents; where they disagreed, the
mechanical count wins. No file was modified, no benchmark was run.
