# Pure-IL 3-stage chain — odd/prime radices in the K=1 Bailey band

Status: DERIVATION (validated by gate before any runtime wiring).
Scope: K=1, NATURAL order, INTERLEAVED z→z, N ≤ 2048 (the cascade owns ≥2048).
Split-layout K=1 is untouched (standing instruction).

## Why two stages cannot do it

Every cil kernel (n1t, t2) vectorizes 2 complex per ymm and carries the
contract `count % 2 == 0`. In the two-stage il2p plan:

    leaf n1t(R2) runs at count = R1
    mid  t2(R1)  runs at count = R2

so BOTH factors must be even. An odd factor always lands as the other
stage's count. Two-stage genuinely cannot host an odd radix — this is a
parity property of the staging, not a missing kernel.

## The 3-stage shape: N = R2 · A · B  (R1 = A·B), SIMD axis pinned to q

Factor the MID, not the leaf. The leaf's output q ∈ [0,R2) stays the SIMD
column axis of every subsequent pass, so every count is R2 (stage 1: R1).
Constraints: **R1 even and R2 even** — the odd factor sits in A or B as a
kernel RADIX only. Covers all N = odd·2^k with N/odd ≥ 8 (both R1,R2 even).

Derivation (four-step applied twice; il2p's gated 2-stage identity is the
base case):

    X[l·R2 + q] = Σ_{l'} W_N^{l'q} ω_{R1}^{l l'} · M[l'·R2 + q]
    M[c·R2 + p] = Σ_j x[j·R1 + c] ω_{R2}^{jp}          (stage 1, turned store)

Split l' = j·A + c (j ∈ [0,B), c ∈ [0,A)), l = a·B + b:

    X[(aB+b)R2 + q]
      = Σ_c ω_A^{ac} · W_N^{cq} · [ Σ_j M[(jA+c)R2+q] · W_{N/A}^{jq} · ω_B^{jb} ]
        \______stage 2b (radix A)______/  \__________stage 2a (radix B)_________/

using W_N^{(jA+c)q} = W_{N/A}^{jq} · W_N^{cq}. Both bracketed passes are
EXACTLY t2-form: legs pre-twiddled per (leg, column q), then a DFT across
legs — and both tables are INDEPENDENT of the loop that repeats the call
(stage 2a's of c, stage 2b's of b), so one VTW2 table per stage.

## Call sequence (existing kernels only — NO new kernel kind needed)

The inter-stage "transpose" of the group axes (c, b) is pure ADDRESS
ARITHMETIC in the caller (per-call base offsets + leg strides); the
kernels' own turned/straight stores are untouched.

    stage 1   n1t(R2), 1 call:
              in  zin,               Ls = R1,     count = R1  (EVEN)
              out mid1,              OLs = R2     → mid1[(c)·R2·?]: (leg p, col c) → mid1[2(c·R2+p)]
    stage 2a  t2(B), A calls, c = 0..A-1:
              in  mid1 + 2·c·R2,     Ls = A·R2,   count = R2  (EVEN)
              out mid2 + 2·c·R2,     OLs = A·R2
              tw  VTW2(B legs, R2 cols, modulus N/A = B·R2)
              → h at mid2[2((b·A + c)·R2 + q)]
    stage 2b  t2(A), B calls, b = 0..B-1:
              in  mid2 + 2·b·A·R2,   Ls = R2,     count = R2  (EVEN)
              out zout + 2·b·R2,     OLs = B·R2
              tw  VTW2(A legs, R2 cols, modulus N)
              → X[(aB+b)·R2 + q]  — NATURAL order

VTW2 tables use the il2p record builder verbatim with (legs, modulus)
parameters: record (pair pp, leg l) at offset (pp·(legs−1)+(l−1))·8,
[c,c,c,c][−s,+s,−s,+s], angle −2π·l·k/modulus, k = 2pp{+1}.

Kernel-call cost: 1 + A + B calls (≤ ~30 at any covered N). Scratch: two
z-planes (mid1, mid2) of 2N doubles.

## Coverage this unlocks (Bailey band)

    3·2^k : 48=4·(3·4) 96 192 384 768 1536
    5·2^k : 80 160 320 640 1280        7·2^k : 112 ... 1792
    9, 11, 13, 15, 25, 27 · 2^k likewise (all have t2 kernels)

NOT covered (v1): N = 4·odd with a single 4 (e.g. 100 = 4·25 — R1 odd);
needs even-composite radices (6, 10, 12 …) the emitter can produce but has
not been asked for. All-odd N (45, 63, 225): every count odd — Rader/
Bluestein territory, out of scope.

## Order of work (the il2p-bwd discipline: gate the composition BEFORE wiring)

1. Gate harness on the REAL kernels (this doc's call sequence, naive-DFT
   oracle): odd cells 48/96/192/320/1536 + pow2 control 256 = 4·(8·8).
2. il2p.h chain plan (create/execute; chain as PLAN INPUT, default picked
   by a legal-balance rule until the calibrator owns it).
3. Backward: derive separately (do NOT guess from the forward; the 2-stage
   bwd needed its own composition, t2t + n1_bwd).
4. vfft.c: extend the K=1 IL search past its both-factors gate for
   non-pow2 N; route stays 2P_PURE-family; split path untouched.
