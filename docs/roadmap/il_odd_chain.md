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
      = Σ_c ω_A^{ac} · W_N^{c(q+b·R2)} · [ Σ_j M[(jA+c)R2+q] · W_{N/A}^{jq} · ω_B^{jb} ]
        \_______stage 2b (radix A)________/  \__________stage 2a (radix B)_________/

using ω_{R1}^{(jA+c)(aB+b)} = ω_A^{ac}·ω_{R1}^{cb}·ω_B^{jb} and
W_N^{(jA+c)q} = W_{N/A}^{jq}·W_N^{cq}, then W_N^{cq}·ω_{R1}^{cb} =
W_N^{c(q+b·R2)}. Both passes are EXACTLY t2-form (legs twiddled per
(leg, column), DFT across legs).
🔴 Stage 2b's twiddle depends on the COMBINED index q + b·R2 — the first
draft dropped the ω_{R1}^{cb} factor and failed the gate O(1) at every cell
including the pow2 control. So: stage 2a gets ONE table (independent of its
repeat index c); stage 2b gets ONE BIG table over all B·R2 columns at
modulus N, and call b reads its own region starting at column b·R2 (even,
so regions start on VTW2 pair boundaries and the kernel cursor lines up).

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
              tw  VTW2(A legs, B·R2 cols, modulus N) + (b·R2/2)·(A−1)·8
              → X[(aB+b)·R2 + q]  — NATURAL order

GATED 2026-07-29 (build_tuned/benches/il_odd_chain_gate.c, real kernels vs
naive DFT): 12/12 — 48/96/192/320/768/1536/1728 with odd radices 3/5/27 in
either mid position + pow2 control 256; rel err 1.2e-14 .. 5.1e-13.

## Backward: SOLVED + GATED on t2t semantics (13/13, 2026-07-29)

(History: a conj-of-forward draft using pre-twiddle stages was dropped with
the tree-wide retirement of the t2p kind — one canonical bwd semantics,
`--cil-pretw` refuses. This reversed-stage composition replaced it.)

The forward's stages inverted in REVERSE order; every count stays even:

    B1  t2_bwd(A), B calls, b = 0..B-1:   IDFT_A across the a legs
        in  X + 2·b·R2,      Ls = B·R2,   count = R2
        out mid2 + 2·b·A·R2, OLs = R2     (straight store)
        tw  conj VTW2(A legs, B·R2 cols, modulus N) + region b·R2
        → mid2[(bA+c)R2 + q], post-twiddled by conj W_N^{c(q+b·R2)}
    B2  t2tg_bwd(B), A calls, c = 0..A-1: IDFT_B across the b legs
        in  mid2 + 2·c·R2,   Ls = A·R2,   count = R2
        out mid1 + 2·c,      OLs = R1, OGs = A   (LEG-STRIDED turn)
        tw  conj VTW2(B legs, R2 cols, modulus B·R2)
        → mid1[q·R1 + j·A + c], post-twiddled by conj W_{B·R2}^{jq}
    B3  n1_bwd(R2), 1 call:               IDFT_R2 across q
        in  mid1, Ls = R1;  out zout, OLs = R1;  count = R1  → NATURAL

**t2tg** (`--cil-turnst-gs`, symbol `radixR_z_t2tg_bwd_avx2`) is t2t with
the `(void)`'d OGs slot wired as the turned store's LEG STRIDE — required
because the l′ = e + A·f split interleaves leg groups from different c
calls at stride A. Strided legs are not contiguous, so every leg scatters
as two 128-bit halves (2R narrow stores vs t2t's R wide — a plan-level
measured cost, not assumed away). Existing t2t kernels stay byte-identical
(separate symbol). 17 t2tg_bwd kernels emitted (odd 3..27 + pow2 4..64).

GATED (same gate, bwd section, naive-IDFT oracle): 13/13 — all fwd cells'
mirrors incl. 1728 with radix-27 in BOTH positions; 1.0e-14 .. 5.1e-13.

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
