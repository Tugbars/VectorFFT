# The prime engine's inner is raced, not borrowed (design, 2026-09-17)

Owner's ruling, 2026-09-17: **"prime cells should have their own inner race."**

## What a prime cell does today

A prime N cannot be split into stages, so `il_prime.h` turns it into a
convolution and does the convolution with an FFT of a friendlier length M:
Rader's M = N - 1, Bluestein's M = the next power of two >= max(16, 2N - 1).
The METHOD (Rader vs Bluestein) is already raced and banked
(`_ilprime_create_banked`: `vw2_prime_method_lookup` replays, else
`vfft_ilprime_create_method(N, 0)` runs the house race body over both arms,
min-of-3, and `vw2_prime_method_bank` writes `eng= ran=1 src=race` plus a
signpost to the inner's row).

The INNER is not raced. It is BORROWED, in three layers:

| layer | where | what it does |
| --- | --- | --- |
| 1 | `k1_commit.h:210` (`_ilprime_inner_from_wisdom`) | reads the length-M cell's **scrambled K=1 row**, UNGUARDED by recalibrate; if it names ZTURN-T, that chain and tile serve |
| 2 | `k1_commit.h:234` | `if (M > 4096) return 0;` -- above that, layer 1 is the ONLY source; below it, `_k1_il_candidate(W, cfg, M, ...)` asks the K=1 tier for a pair or chain3 at M (this one does honor recalibrate) |
| 3 | `il_prime.h:122` (`_ilprime_inner_make`, when the provider returns 0) | the engine's "structural" inner: the **most balanced pair** by `abs(R1 - R2)`, else `vfft_il3p_default_chain` -- two heuristics, never measured |

So the inner is chosen by proxy (M standalone is not M inside a convolution:
forward, pointwise multiply, backward, in place, any order acceptable), the
caller's `recalibrate` cannot reach layer 1 without racing M's cell nested
inside N's create -- which `_k1_il_dp_busy` forbids -- and layer 3 is
exactly the class of rule the owner's law excludes.

## The contract

A prime cell owns its inner verdict the way every other cell owns its plan:
raced at create on a miss or under `recalibrate`, banked on **the prime cell's
own row**, replayed thereafter. The K=1 tier's row at M is never consulted
for the inner again. Nothing structural remains: the balanced-pair rule and
the default chain become CANDIDATES in the race, not answers.

## The race

- **When**: inside `_ilprime_create_banked`, after the method is settled (a
  replayed hint, or the method race's winner). One race per prime cell per
  method, not a 2D race: the method is chosen with the current inner, then
  the inner is raced for that method. The second-order case -- a better
  inner flipping the method -- is accepted, as it is for every other nested
  axis in the tree.
- **Candidates** (built DIRECTLY, no planner call, so the re-entrancy lock is
  never touched -- the four-step's `_k1fs_sb_chains` is the model):
  every legal `il2p` pair at M (the pool `_ilprime_inner_make` already
  enumerates to pick the balanced one), the `il3p` default chain at M, and
  the ZTURN-T chains at M from the ZTURN-T ladder when M is in its band.
  The pool is a pool; nothing in it is a default.
- **What is timed**: the whole convolution -- `vfft_ilprime_execute` forward
  on the assembled prime plan with that inner installed -- not the inner
  standalone. That is the workload the verdict serves.
- **How**: `support/race.h`, the one race body; alternating order; the
  house hysteresis. A cold prime cell takes seconds; the planner logs on
  entry as the K=1 race does.
- **What banks**: on the prime-method row, beside `eng=`: the inner's route
  and shape -- `inner=ztt chain=... tw=...` / `inner=2p R1.R2` /
  `inner=3p ...`. The signpost `ref_M/ref_lay` that pointed at M's row is
  retired with the borrowing it served.

## What moves, and what does not

- `_ilprime_inner_from_wisdom` becomes: replay the prime cell's own inner
  tokens (guarded by `!cfg->recalibrate`); on a miss, race and bank. Layers 1
  and 2 are deleted -- the read of M's row, the `M > 4096` cliff, the call
  into `_k1_il_candidate`. Layer 3's enumeration survives as the candidate
  pool builder; its "pick the most balanced" line does not.
- `vw2_prime_method_bank/lookup` grow the inner tokens (L10 territory: the
  reader, the writer, the serializer -- one row, no new row family).
- `il_prime.h`'s standalone entry (`vfft_ilprime_create`, no wisdom) keeps
  building SOMETHING for callers with no store, but that is the split
  library's / the bare API's concern and is out of this design's scope.

## Gate

`benches/ilprime_inner_gate.c`: for a spread of primes whose M spans the
bands (Rader and Bluestein, M below and above 4096), cold store:
- the cold create logs the inner race and banks `inner=` on the prime row;
- forward matches the naive DFT;
- the warm create logs no race and is BITWISE the cold forward;
- `recalibrate=1` on the warm store re-races the inner (a poisoned `inner=`
  token is overwritten -- the poison test, because latency alone was a false
  positive once this week).
Watched to FAIL with the race disabled before it counts.

## Cost, honestly

A race per prime cell, once, of a few seconds. The wisdom row format changes
(additive tokens; old rows without `inner=` are a miss and race once). The
K=1 tier's rows at M lose one consumer and nothing else. The risk is in the
serializer, which is why the negative test and the full sweep are not
optional.

## Checklist

- [x] 1. This design.
- [ ] 2. The candidate pool builder (from layer 3's enumeration + ZTURN-T's ladder at M).
- [ ] 3. The race + bank inside `_ilprime_create_banked`; the row tokens in the reader/writer/serializer.
- [ ] 4. Delete layers 1-2; the provider replays or races.
- [ ] 5. The gate, watched to fail.
- [ ] 6. Full sweep; records (`design_contracts.md` section 6, `docs/roadmap/policy_survey_defects.md`, memory).
