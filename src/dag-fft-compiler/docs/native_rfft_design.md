# Native real mixed-radix FFT (r2hc/hc2r): design

Section 60 companion. Motivating measurements: sections 59c-59f.
The half-complex r2c wrapper pays ~41us of pure memory traffic at
N=256 K=256 (pack ~21us + postprocess ~28us, 78% traffic) on top of
a ~36us inner. MKL pays ~19us for the same job with complex kernels
that are TIED with ours. A native real path makes neither extra
pass. This document fixes the architecture before code.

## Target and acceptance (pre-registered at design time)

r2hc-256 K=256, this container, same-run methodology:
  - >= 1.25x faster than today's in-place r2c (77us -> <= 62us)
  - within 1.10x of MKL r2c measured in the SAME run (MKL swings
    +-25% across runs; only same-run ratios count)
Stretch: at or under the plain c2c-128 time + 15% (i.e. <= ~49us),
which is the "no extra passes" ideal.

## Architecture: FFTW ct-hc2hc mirrored onto the stride stack

Problem type r2hc_N over batched-K split-real input (N rows x K).
DIT recursion identical in shape to the existing complex plan:

  stage 0: r2cf LEAF, radix r0 — real-input DFT codelets applied to
    the r0-decimated row groups. In batch-major layout decimation is
    ROW-STRIDE access: every load is a full contiguous K-row stream.
    This is why the measured 21us pack pass disappears structurally
    rather than by fusion: there is nothing to pack.
  stages 1..s: HC2HC twiddle stages — the ported gen_hc2hc codelets,
    operating on Hermitian-packed data, runtime twiddle arrays,
    exactly mirroring today's t1 cascade.
  last stage: HC2HC_LAST variant that stores natural-order SPLIT
    bins (out_re, out_im rows 0..N/2) directly — the section-59f
    lesson applied at birth: no separate unpack pass, no perm pass.

Factorizer, wisdom, exhaustive search all reuse with a problem-type
key ("r2hc:N:K"). Backward (hc2r: r2cb leaves + hc2hc bwd) is the
symmetric phase 2.

## Decision points

D1. INTER-STAGE FORMAT.
  (A) FFTW packed halfcomplex in a single N x K plane: in-place,
      constant footprint across stages, and it is the contract the
      ported gen_hc2hc already assumes.
  (B) Split hc planes with H = m/2+1 rows per sub-spectrum: matches
      our consumer format but sum r*(m/2+1) > N/2+1 breaks the
      constant-footprint in-place property.
  RECOMMEND (A) internally; the consumer never sees it because
  HC2HC_LAST writes split natural bins.

D2. OUTPUT ORDERING. Digit-reversed + unpermute pass would re-create
  the postprocess we are killing. RECOMMEND the HC2HC_LAST fused
  store (one extra emitted variant per radix; mechanical generator
  work). Fallback if the variant proves hard: DIF-last plan shape.

D3. LEAF ABI. Monolithic rdft codelets exist (generic 7-arg OOP).
  Need an n1-style leaf variant: row-strided legs in, packed hc out.
  RECOMMEND new emit variant r2cf_n1 reusing dft_rdft math with the
  existing n1 leg-stride ABI plumbing.

D4. TWIDDLES. The hc2hc builder takes runtime tw_re/tw_im functions,
  same as t1. RECOMMEND keep runtime arrays (wisdom-compatible,
  matches the whole stack).

D5. RUNTIME SHAPE. RECOMMEND generalizing stride_stage_t with a
  stage-kind enum {N1, T1, R2CF, HC2HC, HC2HC_LAST} + a switch in
  the stage loop, rather than a parallel plan type. The struct
  already has per-kind fn-pointer slots; churn is minimal and the
  complex path is untouched when kind in {N1, T1}. New public shell
  core/rfft.h; core/r2c.h stays for compat and odd N.

D6. SCOPE ORDER. pow2 radices first (the losing cells: 256, 1024),
  odd radices (3/5/7) second — they matter for the dct/dst shells
  and 2D, but the MKL gap lives at pow2.

## Phases

P0 (half day, DE-RISK FIRST): generate one hc2hc (r=4, DIT) and one
  rdft leaf; numerically gate the hc2hc DAG against a reference
  combine. The port has never been generated or gated (absent from
  coverage). Apply the section-56 lesson: verify what policy keys
  (aggressive/reassoc/fma) an hc2hc flag receives — its internal
  DFT differs from the flag n. Extend dbg_eval to hc2hc if anything
  smells.

P1 (1-2 days, generator): r2cf_n1 leaf ABI variant; HC2HC_LAST
  fused-store variant; coverage quadrant rfft-{avx2,avx512};
  per-codelet gates; regen.

P2 (2-3 days, runtime): stage-kind enum + executor switch; rfft
  plan builder on the existing factorizer; wisdom problem-type key;
  correctness gates vs brute at N in {16,32,64,128,256,512,1024}
  all-K-classes; per-stage numeric instrument for the hc format.

P3 (1 day): 3-way bench vs MKL/FFTW same-run; switch dct/dst shell
  inners to the native path (downstream wins ride free: dct2
  Makhoul sits directly on r2c); odd-N r2c keeps the section-57
  shell; docs + ledger scoring of the pre-registered targets.

## Risks, ranked

1. hc2hc port unvalidated end to end (P0 exists to kill this first).
2. Packed-hc indexing (the classic FFTW r/i interleave off-by-ones).
   Mitigation: per-stage property gates + the dbg_eval pattern
   extended to hc evaluation; we own that instrument now.
3. Policy keying redux (section 56): hc2hc/r2cf flags must key
   simplification policy on their INTERNAL DFT structure.
4. In-place aliasing in the HC2HC_LAST natural store (write rows
   overlap read rows): may force the last stage out-of-place into
   the caller's output buffers — which is what we want anyway for
   the 3-pointer API.

## D2 (final design, section 69): hc2c natural-split terminator

OUTPUT ABI: out_re / out_im planes of (N/2+1) x K doubles, natural
order (row f = frequency f), im[0] = 0 and im[N/2] = 0 (N even).

COLUMN COVERAGE THEOREM (replaces the packed d=0 stage): one hc2c
call at interior column k in [1, ceil(m/2)-1] covers ALL natural
rows of residues k and m-k (mod m). Proof sketch: slot s carries
G[s] = X[k+sm]; rows f_s = k+sm <= N/2 store directly; f_s > N/2
store conj(G[s]) at N-f_s = (m-k)+(r-1-s)m — a bijection onto the
residue-(m-k) rows <= N/2. The sym pass (conjugate upper) already
applies the needed im negation.

CONSTANT-BOUNDARY LEMMA (what makes the ABI bakeable): the regime
boundary s* = floor((N/2-k)/m) = floor(r/2 - k/m) is CONSTANT over
the whole interior range 0 < k < m/2:
  r even: s* = r/2 - 1        r odd: s* = (r-1)/2
independent of k and m. Edge f_s = N/2 cannot occur at interior k
(requires k ≡ 0 or k = m/2). Therefore the slot -> pointer map is a
pure function of r and can be baked into the codelet at generation
time.

CODELET ABI v2.1 (hc2c natural; FFTW khc2c-shaped):
  void radixR_hc2c_nat_fwd_ISA(
      const double *in_re, const double *in_im,   /* as hc2hc */
      double *Rp, double *Ip,    /* low rows:  + s*osp, s = 0..s* */
      double *Rm, double *Im,    /* mirror:    + u*osm, u = r-1-s */
      const double *tw_re, const double *tw_im,   /* slots j-1 */
      ptrdiff_t is, ptrdiff_t osp, ptrdiff_t osm, size_t vl);
Executor passes Rp = out_re + k*K, Ip = out_im + k*K, osp = m*K,
Rm = out_re + (m-k)*K, Im = out_im + (m-k)*K, osm = m*K.

EDGES (executor-side, no new codelets):
  k = 0: r2cf into an r-row scratch column, then scatter rows to
    natural homes via the packed slot map (cost: r rows of K once
    per plan — negligible; sign conventions resolved by gate).
  k = m/2 (m even): existing vectorized mid already computes
    (Re G[s], Im G[s]); natural store keeps BOTH at rows
    m/2 + sm <= N/2 and skips uppers (self-paired). Simpler than
    the packed mid.
  nf = 1 (single leaf): same r2cf-then-scatter as k = 0 with r = N.

GATING: codelet formula gate (4-pointer variant) per radix, then
end-to-end plan gate against packed-output -> natural unpack
reference (bit-exact re; im sign per convention), all 14 cells.
