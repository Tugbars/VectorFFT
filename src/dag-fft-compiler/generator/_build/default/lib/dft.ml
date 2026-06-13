(* dft.ml — c2c DFT algorithm selection + recursive Cooley-Tukey decomposition.
 *
 * Public entry: `dft ~sign n input_re input_im` returns (out_re, out_im)
 * expr arrays. `dft_expand n` and `dft_expand_twiddled ~policy ~direction
 * ~sign n` produce assignment lists for codelet emission.
 *
 * pick_algorithm dispatches by N's number-theoretic properties:
 *   - Direct          n = 2, or odd primes (uses conjugate-pair construction)
 *   - Cooley_Tukey    composites with hardcoded (n1, n2) factorizations
 *                     matching production codelet shapes (R=16 → 4×4,
 *                     R=32 → 4×8, R=64 → 8×8, R=128 → 8×16, etc.)
 *   - Split_radix     pow2 ≥ 8, opt-in via VFFT_SPLIT_RADIX env var
 *                     (delegated to split_radix.ml)
 *
 * dft_expand_twiddled covers t1_dit / t1_dif / t1s variants with the
 * cmul-pattern detector preserved (Algsimp.of_expr lifts Sub(Mul,Mul) /
 * Add(Mul,Mul) pairs to Cmul opaque atoms before reassoc shreds them). *)

open Expr

(* === ALGORITHM SELECTION === *)

type algorithm =
  | Direct                        (* prime n, or n=2: use dft_kernel *)
  | Cooley_Tukey of int * int     (* (n1, n2): split DFT-n into n1 columns of n2 *)
  | Split_radix                   (* pow2 n ≥ 8: SR decomposition (N/2 + N/4 + N/4) *)

(* Opt-in routing of pow2 sizes through split-radix instead of CT.
 *
 * Controlled by the VFFT_SPLIT_RADIX environment variable: when set to
 * "1" (or any non-empty value), pick_algorithm routes N ∈ {8, 16, 32, 64}
 * through Split_radix. Otherwise these N continue to use their existing
 * CT factorizations.
 *
 * Rationale for the env var approach (rather than a CLI flag or a default
 * change): the existing recipe machinery in dft_expand_twiddled_spill is
 * calibrated against the CT decomposition's PASS 1 / PASS 2 boundary. SR
 * doesn't have that same boundary structure, so the recipe needs separate
 * adaptation work (next PR). Until that's done, leaving SR opt-in lets us
 * validate correctness without disturbing the existing R=16/32/64 path.
 *)
let newsplit_enabled () : bool =
  match Sys.getenv_opt "VFFT_NEWSPLIT" with
  | None | Some "" -> false
  | Some _ -> true

let split_radix_enabled () : bool =
  newsplit_enabled () ||
  (match Sys.getenv_opt "VFFT_SPLIT_RADIX" with
   | None | Some "" -> false
   | Some _ -> true)

(* Pick algorithm based on n's number-theoretic properties.
 *
 * Match user's hand-coded factorizations:
 *   R=4  → CT(2, 2) — radix-2 inside radix-2
 *   R=8  → CT(2, 4) — radix-2 outer, radix-4 inner
 *   R=16 → CT(4, 4) — two passes of radix-4 (NOT 2×8)
 *
 * For other even n, fall back to radix-2 CT.
 * Odd n (and primes) use direct DFT.
 *
 * Why R=16 = 4×4 instead of 2×8: hand-coded uses two clean radix-4
 * passes with explicit spill between them. The 4×4 structure has a
 * lower-depth dependency tree than the 4-level recursion 2×8 would
 * produce, and matches the user's existing benchmark layout. *)

(* ISA-aware factorization hint.
 *
 * Set from bin/gen_radix.ml at startup based on the --isa flag:
 *   AVX-512 (32 zmm)  → 32
 *   AVX2     (16 ymm) → 16
 *   SSE2/scalar       → 16 (treated as narrow)
 *
 * Read by pick_algorithm to choose factorizations that match the
 * register budget. Default 32 keeps prior AVX-512-tuned behaviour for
 * any caller that doesn't set this.
 *
 * Empirical sweep on AVX2 (doc 60 notes):
 *   R=64 (8,8)  -> 473 spills    (8 complex × 16 ymm = exact budget)
 *   R=64 (4,16) -> 431 spills    (recurses to (4,(4,4)), deepest pass 8 ymm)
 *
 * Reason for ref rather than threading a parameter through:
 * pick_algorithm has 7 call sites including recursive descent through
 * the DFT-construction code; threading would require visible churn
 * across multiple modules. The ref is set once at program startup. *)
let target_vec_regs : int ref = ref 32

(* === Factorization override (for measurement-driven exploration) ===
 *
 * VFFT_CT_FACTOR="N:a,b" forces pick_algorithm(N) = Cooley_Tukey(a,b);
 * "N:direct" forces Direct; "N:sr" forces Split_radix. Scoped to the
 * EXACT radix N only — sub-radices in the recursive descent are strictly
 * smaller than the top-level radix, so this overrides only the codelet
 * being generated, not its internal sub-FFTs. Used by the factorization
 * explorer to A/B candidate decompositions against the emitted metric.
 * Unset => no effect (default hand-tuned table). *)
let factor_override (n : int) : algorithm option =
  match Sys.getenv_opt "VFFT_CT_FACTOR" with
  | None -> None
  | Some s ->
    (try
       let colon = String.index s ':' in
       let nn = int_of_string (String.trim (String.sub s 0 colon)) in
       if nn <> n then None
       else begin
         let rest = String.trim
             (String.sub s (colon + 1) (String.length s - colon - 1)) in
         if rest = "direct" then Some Direct
         else if rest = "sr" then Some Split_radix
         else
           let comma = String.index rest ',' in
           let a = int_of_string (String.trim (String.sub rest 0 comma)) in
           let b = int_of_string (String.trim
                     (String.sub rest (comma + 1) (String.length rest - comma - 1))) in
           Some (Cooley_Tukey (a, b))
       end
     with _ -> None)

let pick_algorithm (n : int) : algorithm =
  match factor_override n with
  | Some a -> a
  | None ->
  if n <= 2 then Direct
  else if split_radix_enabled () && n >= 8 && n land (n - 1) = 0 then
    (* Route pow2 sizes ≥ 8 through Split_radix when VFFT_SPLIT_RADIX is set.
     * N=4 stays as CT(2,2) regardless — SR's recursion bottoms there.
     * N=2 stays as Direct (also an SR base case).
     * Non-pow2 sizes are unaffected: SR requires power-of-two N. *)
    Split_radix
  else match n with
    | 4 -> Cooley_Tukey (2, 2)
    | 8 -> Cooley_Tukey (2, 4)
    | 16 -> Cooley_Tukey (4, 4)
    | 32 -> Cooley_Tukey (4, 8)
    | 64 when !target_vec_regs <= 16 -> Cooley_Tukey (4, 16)
        (* AVX2 (16-ymm budget): (4,16) recurses to (4,(4,4)). Deepest
         * pass has 4 complex = 8 ymm live, well under the 16-ymm budget.
         * Empirically saves 42 spills vs (8,8) on AVX2 (−9%) and also
         * −8 fp ops (920 vs 928). On AVX-512 the savings reverse —
         * (8,8) is symmetric and wins on schedule pressure. *)
    | 64 -> Cooley_Tukey (8, 8)
        (* Hand-coded R=64 uses CT(8, 8) — symmetric factorization that
         * splits the 64-point DFT into 8 sub-FFT-8s in PASS 1 and 8
         * sub-FFT-8s in PASS 2. Same convention as gen_radix64.py. *)
    | 128 -> Cooley_Tukey (8, 16)
        (* R=128 EXPERIMENTAL stress test for spill controller and IR
         * construction at sizes beyond R=64. CT(8, 16) splits N=128 into
         * 16 sub-DFT-8s in PASS 1 and 8 sub-DFT-16s in PASS 2. Inner
         * DFT-16 = CT(4, 4) which is well-tested. The outer DFT-8 is
         * CT(2, 4). All sub-codelets exist and work cleanly.
         *
         * Default fallback CT(2, 64) doesn't terminate in 60s — the
         * recursion through 64 = CT(8, 8) builds an enormous nested IR.
         * CT(8, 16) directly avoids this depth.
         *
         * Peak live count exceeds 32 ZMM significantly; recipe path is
         * mandatory. Probably exposes spill controller weaknesses we
         * want to characterize. See docs/31_R128_spill_stress.md. *)
    | 256 -> Cooley_Tukey (16, 16)
        (* R=256 — balanced factorization. 16 sub-DFT-16s in each pass.
         * Following the precedent set by R=128 (8×16) and R=64 (8×8),
         * this is the natural "balanced" choice for monolithic generation.
         * Generates a ~7800-vec-instr codelet (vs R=128's 3318). *)
    | 512 -> Cooley_Tukey (16, 32)
        (* R=512 — 16×32 chosen empirically. Multi-stage bench at R=512
         * showed 16×32 / 32×16 multi-stage dominated 8×64 / 64×8 across
         * all batch sizes (typically 10-20% faster). The same factorization
         * principle should apply to the monolithic codelet's INTERNAL
         * structure: 16×32 splits the work more evenly across moderate-
         * sized sub-FFTs than 8×64's heavy R=64 + light R=8.
         *
         * Picker entry was initially CT(8, 64) on AVX-512-lane-alignment
         * reasoning, but the data didn't support that intuition — see
         * docs/34_real_shuffle_and_isa_split.md for the empirical comparison.
         *
         * CT(32, 16) was tested in doc 36's Phase 2 prototype and gave
         * WORSE stack op counts (8752 AVX-512, 14846 AVX2) than CT(16, 32)
         * — the smaller Pass 1 clusters helped but the larger Pass 2
         * clusters (sub-DFT-32) hurt more. *)
    | 1024 -> Cooley_Tukey (32, 32)
        (* R=1024 EXPERIMENTAL (doc 41) — symmetric factorization. Both
         * passes have 32 sub-DFT-32 clusters. Inner DFT-32 = CT(4, 8)
         * is well-tested. Asymmetric alternatives (CT(16,64), CT(64,16))
         * would put one pass's clusters at sub-DFT-64 size which is
         * beyond what fits comfortably in 32 ZMM registers. CT(32,32)
         * is the natural extension of the R=512=CT(16,32) line.
         *
         * Test goal: compare monolithic R=1024 stack ops and runtime
         * against multi-stage cascades (R=64×R=16, etc.) under
         * gcc-11 + -flive-range-shrinkage. *)
        (* User's gen_radix32.py uses CT(8, 4) in their (N1, N2) convention.
         * Their convention has input mapping n = N2*n1 + n2 (high digit n1),
         * ours has n = n1 + n2*N1 (low digit n1). The labels swap.
         *
         * Their PASS 1: 4 sub-FFTs of size 8 (radix-8 butterflies).
         * Their PASS 2: 8 sub-FFTs of size 4.
         *
         * In OUR convention, PASS 1 has N1 sub-FFTs of size N2, so we want
         * N1=4, N2=8. The math is equivalent; just the labeling swaps.
         *
         * Why this factorization: 4×8 keeps R=32's register pressure
         * manageable. Each PASS 1 sub-FFT has 8 live values (fits AVX-512's
         * 32 ZMM with room for twiddles); going wider (e.g. 2×16) would
         * blow the register budget. Going narrower (e.g. 8×4) is also valid
         * but produces deeper inner DFTs in PASS 2. *)
    (* Mixed-radix composites — first batch (R=6 only for wiring validation;
     * R=10/12/20/25 follow once R=6 path proves clean). *)
    | 6 -> Cooley_Tukey (3, 2)
        (* R=6 = CT(N1=3, N2=2). PASS 1: 2 sub-DFT-3s (prime, conjugate-pair
         * construction). PASS 2: 3 sub-DFT-2s (trivial add/sub). Internal
         * twiddles W6: 2 unique exponents (W6^1, W6^2) require cmul; W6^3=−1
         * is free. Matches gen_radix6.py's (N1=3, N2=2) factorization in our
         * convention. Peak live ~12 ZMM AVX-512 (fits), ~12 YMM AVX2 (also
         * fits but tight). *)
    | 25 -> Cooley_Tukey (5, 5)
        (* R=25 = CT(N1=5, N2=5). Symmetric prime-prime split. PASS 1: 5
         * sub-DFT-5s. PASS 2: 5 sub-DFT-5s. All 9 unique internal W25
         * exponents require cmul (no free rotations like W^N/2 = -1 or
         * W^N/4 = ±j; none of those exponents arise in W25's grid).
         * Matches gen_radix25.py's (N1=5, N2=5). Peak live ~50 (raw) so
         * spill is mandatory on both ISAs. Highest retiring rate codelet
         * per profiling — recipe path is essential here. *)
    | 10 -> Cooley_Tukey (5, 2)
        (* R=10 = CT(N1=5, N2=2). PASS 1: 2 sub-DFT-5s. PASS 2: 5 sub-DFT-2s
         * (trivial add/sub). Same shape as R=6 (CT with prime N1, trivial
         * N2) but with DFT-5 inner. Internal twiddles W10: only 2 unique
         * non-trivial exponents (W10^1, W10^2 — others are 1, -1, ±j and
         * fold via const_cmul). Matches gen_radix10.py's (N1=5, N2=2). *)
    | 12 -> Cooley_Tukey (4, 3)
        (* R=12 = CT(N1=4, N2=3). First non-prime, non-pow2 split — both
         * factors > 2. PASS 1: 3 sub-DFT-4s (radix-4 butterflies, FMA-poor
         * but very compact). PASS 2: 4 sub-DFT-3s (prime conjugate-pair).
         * Free internal twiddles: W12^3 = -j, W12^6 = -1, W12^9 = j; only
         * 4 unique non-trivial cmul exponents (W12^1, W12^2, W12^4, W12^5
         * ... folded by symmetry). Matches gen_radix12.py's (N1=4, N2=3). *)
    | 20 -> Cooley_Tukey (5, 4)
        (* R=20 = CT(N1=5, N2=4). PASS 1: 4 sub-DFT-5s. PASS 2: 5 sub-DFT-4s.
         * Mix of prime DFT-5 (8 ops/output via conjugate-pair) and pow2
         * DFT-4 (very compact). 12 unique non-trivial W20 exponents; some
         * fold (W20^5 = -j, W20^10 = -1, W20^15 = j). Matches
         * gen_radix20.py's (N1=5, N2=4). Peak live ~40 — spill threshold,
         * recipe path likely essential at AVX-512 (32 ZMM). *)
    | 15 -> Cooley_Tukey (3, 5)
        (* R=15 = CT(N1=3, N2=5). PASS 1: 5 sub-DFT-3s. PASS 2: 3 sub-DFT-5s.
         * Without this entry R=15 falls to Direct DFT-15 (308 ops vs FFTW's
         * 156). With CT(3,5) the costly O(N²) direct path is avoided and
         * Winograd-5 further trims the three DFT-5 instances (181 ops
         * vs FFTW's 156, +16% — bounded by the same n-ary-IR limit that
         * leaves the R=25 gap open). *)
    | 21 -> Cooley_Tukey (3, 7)
        (* R=21 = CT(N1=3, N2=7). PASS 1: 7 sub-DFT-3s. PASS 2: 3 sub-DFT-7s.
         * Without this entry R=21 falls to Direct DFT-21 (O(N²)). With
         * CT(3,7), Winograd-7 trims the three DFT-7 instances substantially
         * relative to conjugate-pair Direct. Same cascade pattern as R=15. *)
    | _ when n mod 2 = 0 -> Cooley_Tukey (2, n / 2)
    | _ -> Direct

(* Whether algsimp's reassoc pass is appropriate for the given n.
 *
 * Reassoc helps when the input is a flat sum-of-products from naive
 * direct DFT expansion — it discovers butterfly subsums by reassociating
 * binary Add chains. But:
 *   - When CT decomposition has structured the input correctly, reassoc
 *     actively destroys that structure by flattening across CT boundaries.
 *   - When dft_direct_conjugate_pair has constructed explicit shared
 *     sub-sums (p_re_m, q_im_m, etc.), reassoc would FLATTEN them into
 *     each output's overall sum and the sharing would be lost. The
 *     conjugate-pair construction is already optimal w.r.t. CSE.
 *
 * So:
 *   - Naive Direct DFT (n=2 only):     reassoc HELPS
 *   - Conjugate-pair Direct (odd n≥3): reassoc HURTS
 *   - CT-decomposed:                   reassoc HURTS
 *)
let needs_reassoc (n : int) : bool =
  match pick_algorithm n with
  | Direct -> n < 3 || n mod 2 = 0  (* only n=2 hits this in practice *)
  | Cooley_Tukey _ -> false
  | Split_radix -> false  (* SR construction is structured like CT — already
                           * has butterfly form; reassoc would flatten it. *)

(* === COOLEY-TUKEY DIT DECOMPOSITION ===
 *
 * Radix-2 DIT decomposition of DFT-N (N even):
 *
 *   X[k]       = E[k] + ω_N^k · O[k]      for k in 0..N/2-1
 *   X[k+N/2]   = E[k] - ω_N^k · O[k]      for k in 0..N/2-1
 *
 * where E = DFT-(N/2) on even-indexed inputs, O = DFT-(N/2) on
 * odd-indexed inputs, and ω_N = exp(-2πi/N) (forward DFT).
 *
 * The complex multiply ω_N^k · O[k] uses constant twiddles (known at
 * code-gen time). For k=0: ω = 1, no multiplication. For k=N/4:
 * ω = -i, just a swap. For k=N/2: ω = -1, just a sign flip.
 * For other k: ω = (cos θ, -sin θ) with θ = 2πk/N — these need
 * actual multiplications by cos/sin constants.
 *
 * We use plain Mul/Add/Sub (NOT Cmul opaque atoms) for these constant
 * twiddle multiplies — algsimp's constant folding handles the trivial
 * k=0/N/4/N/2 cases, and hash-consing shares the √2/2 constant across
 * all uses. Cmul is reserved for runtime-loaded twiddles (the t1_dit
 * premultiplication stage), where we want to PROTECT the cmul structure
 * from reassoc.
 *)

(* Helper: compute the symbolic complex multiply by a constant twiddle.
 * (a + ib) * (c + id) = (ac - bd) + i(ad + bc).
 *
 * Path B optimization (FFTW-style): when |cr| = |ci| = K (the W^k twiddles
 * with k ∈ {1,3,5,7,...} for radix-{2^n}), every output reduces to one
 * of ±K*(xr+xi), ±K*(xr-xi). We emit the FACTORED form so the K-multiply
 * happens AFTER the sum/diff, not before. Two effects:
 *
 *   1. Operation count drops from 6 (4 mul + 1 add + 1 sub) to 4 (2 mul +
 *      1 add + 1 sub). Saves 2 ops per |cr|=|ci| twiddle.
 *
 *   2. The remaining Mul(K, sum) and Mul(K, diff) each have ≤2 downstream
 *      consumers (one + and one -) that look like K*S ± value. These are
 *      perfect FMA absorption targets — multi_use_fma_lift folds each
 *      Mul into its consumers. Net effect from absorbing the Mul: -1 op
 *      AND +1 FMA per absorption, closing the FMA-count gap vs FFTW.
 *
 * For the general |cr| ≠ |ci| case, the original 4-mul form is optimal
 * (no shared factor to extract; algsimp's existing fma_lift handles it).
 *
 * Sign analysis for |cr|=|ci|=K case:
 *   (+K, +K):  out_re = K*(xr-xi),  out_im = K*(xr+xi)
 *   (+K, -K):  out_re = K*(xr+xi),  out_im = K*(xi-xr) = -K*(xr-xi)
 *   (-K, +K):  out_re = -K*(xr+xi), out_im = K*(xr-xi)
 *   (-K, -K):  out_re = K*(xi-xr) = -K*(xr-xi),  out_im = -K*(xr+xi)
 *
 * In all 4 cases the building blocks are K*(xr+xi) and K*(xr-xi); only
 * the signs of the assignments differ. Hash-consing then SHARES the two
 * K-multiplied sums across all uses of this twiddle. *)
let const_cmul (xr : expr) (xi : expr) (cr : float) (ci : float) : expr * expr =
  let abs_eq = abs_float cr = abs_float ci in
  if abs_eq && cr <> 0.0 then begin
    let k = abs_float cr in
    let k_e = Const k in
    let s = Add (xr, xi) in       (* xr + xi *)
    let d = Sub (xr, xi) in       (* xr - xi *)
    let ks = Mul (k_e, s) in      (* K*(xr+xi) *)
    let kd = Mul (k_e, d) in      (* K*(xr-xi) *)
    let sr = if cr > 0.0 then 1 else -1 in
    let si = if ci > 0.0 then 1 else -1 in
    let with_sign sgn e = if sgn > 0 then e else Neg e in
    match sr, si with
    | 1, 1   -> (kd, ks)                           (* out_re=K*D, out_im=K*S *)
    | 1, -1  -> (ks, with_sign (-1) kd)            (* out_re=K*S, out_im=-K*D *)
    | -1, 1  -> (with_sign (-1) ks, kd)            (* out_re=-K*S, out_im=K*D *)
    | -1, -1 -> (with_sign (-1) kd, with_sign (-1) ks) (* out_re=-K*D, out_im=-K*S *)
    | _ -> assert false
  end else begin
    (* General case |cr| ≠ |ci|: emit in TAN-FACTORED form (FFTW genfft
     * with -fma flag). Pick the larger of |cr|, |ci| as the OUTER factor
     * (the "cos"), and the ratio as the INNER factor (the "tan"):
     *
     *   if |cr| ≥ |ci|:
     *     y_re = cr · (xr − (ci/cr)·xi)
     *     y_im = cr · (xi + (ci/cr)·xr)
     *
     *   if |ci| > |cr|:
     *     swap roles; ci is the outer factor, cr/ci is the inner ratio.
     *
     * IMPORTANT — canonical inputs for cross-rotation Const sharing:
     * For symmetric angles, the inner ratio is mathematically the same
     * (e.g. tan(π/8) for both ω^4 and ω^12 at N=64) but the FP inputs
     * differ by 1 ulp because of how sin/cos round. To unify, we round
     * cr and ci to 13 significant digits BEFORE computing the ratio.
     * This is the same precision the smart constructor mk_const uses
     * downstream when the inner constant is hash-consed, but applying
     * it here matters because the rounding-to-13e happens on the
     * INPUTS to the division, not the OUTPUT — and the division
     * amplifies any sub-ulp difference into a 1-ulp difference in
     * the ratio.
     *
     * The unification only fires where there's actually a matching
     * symmetric angle. Non-symmetric rotations keep their distinct
     * (cr, ci) bit patterns up to the 13-digit rounding mk_const
     * would apply anyway. *)
    let round_13 x = float_of_string (Printf.sprintf "%.13e" x) in
    let cr_r = round_13 cr in
    let ci_r = round_13 ci in
    let acr = abs_float cr_r in
    let aci = abs_float ci_r in
    let r_abs = (min acr aci) /. (max acr aci) in
    if acr >= aci then begin
      let tn = if (ci_r >= 0.0) = (cr_r >= 0.0) then r_abs else -. r_abs in
      let cr_e = Const cr_r in
      let tn_e = Const tn in
      let inner_re = Sub (xr, Mul (tn_e, xi)) in  (* xr - tan*xi *)
      let inner_im = Add (xi, Mul (tn_e, xr)) in  (* xi + tan*xr *)
      let out_re = Mul (cr_e, inner_re) in
      let out_im = Mul (cr_e, inner_im) in
      (out_re, out_im)
    end else begin
      let ct = if (cr_r >= 0.0) = (ci_r >= 0.0) then r_abs else -. r_abs in
      let ci_e = Const ci_r in
      let ct_e = Const ct in
      let inner_re = Sub (Mul (ct_e, xr), xi) in   (* ct*xr - xi *)
      let inner_im = Add (xr, Mul (ct_e, xi)) in   (* xr + ct*xi *)
      let out_re = Mul (ci_e, inner_re) in
      let out_im = Mul (ci_e, inner_im) in
      (out_re, out_im)
    end
  end

(* The recursive DFT computation.
 *
 * Inputs:
 *   n          — transform size
 *   input_re k — Expr tree for the k-th input's real component
 *   input_im k — Expr tree for the k-th input's imag component
 *
 * Returns: (re_outputs, im_outputs) as arrays of Expr trees indexed by k.
 *
 * The algorithm dispatch and recursion happens here. Algsimp is NOT
 * called inside — it runs once at the top level, on the fully-expanded
 * tree. This way hash-consing catches sharing across CT recursion levels.
 *)
(* The recursive DFT computation.
 *
 * Inputs:
 *   n          — transform size
 *   input_re k — Expr tree for the k-th input's real component
 *   input_im k — Expr tree for the k-th input's imag component
 *   ?sign      — Fwd uses θ = -2πk/N (DFT); Bwd uses θ = +2πk/N (IDFT, no /N).
 *)
let rec dft ?(sign = `Fwd) (n : int) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  (* Hand-derived Winograd-25 codelet, gated for A/B comparison.
   * See dft_winograd25 (below) for the algebra. *)
  if n = 25 && Sys.getenv_opt "VFFT_WINOGRAD25" = Some "1" then
    dft_winograd25 ~sign input_re input_im
  else
  match pick_algorithm n with
  | Direct ->
    (* For odd N >= 3, the conjugate-pair construction produces a much
     * better-structured DAG than naive direct DFT: pair sums/diffs are
     * shared, per-pair-output intermediates (p_re_m, q_im_m, etc.) are
     * shared between X[m] and X[N-m], and inner sums use linear FMA
     * chains. For N=2 there's nothing to factor; for even N we never
     * reach Direct anyway (CT-decomposed).
     *
     * Special case for N=5: Winograd-5 (dft_winograd5) exploits algebraic
     * identities of 5th roots of unity to reduce 36 ops to 32 and matches
     * FFTW's gen_notw -fma R=5 codelet exactly. Propagates through any
     * radix that decomposes to DFT-5 (R=15, R=20, R=25, R=50, R=100, ...).
     *
     * Empirical trade-off measured for the R=25 cascade
     * (sandbox Xeon 2.80 GHz, 31 trials × 5000 reps × 10 runs):
     *   AVX2  : Winograd ~2% faster (spill traffic dominates chain depth)
     *   AVX-512: Winograd ~4% SLOWER (port parallelism > chain depth)
     *
     * The AVX-512 regression is real and reproducible, but the underlying
     * gap to FFTW (+31 ops at R=25, all butterfly-pair-shared Muls in
     * inter-pass twiddles) is structurally tied to our binary IR. The
     * n-ary `Plus` rewrite that FFTW's genfft uses would unblock both
     * the AVX-512 regression and the R=25/R=64 gap to FFTW — see doc 59
     * addendum for sizing. Until that lands, the choice is between:
     *   (a) Pure default win on AVX2, small loss on AVX-512 R=25
     *   (b) Flag-based dispatch (one more flag, more cognitive load)
     * We pick (a). Code simplicity > marginal AVX-512 R=25 perf. *)
    if n = 5 then begin
      if Sys.getenv_opt "VFFT_CNUM_W5" = Some "1" then begin
        let input = Cnum.signal_of_re_im input_re input_im in
        let out = dft_winograd5_cnum ~sign input in
        Cnum.split_re_im out
      end else
        dft_winograd5 ~sign input_re input_im
    end
    else if n = 7 then
      dft_winograd7 ~sign input_re input_im
    else if n >= 3 && n mod 2 = 1 then
      dft_direct_conjugate_pair ~sign n input_re input_im
    else
      dft_direct ~sign n input_re input_im
  | Cooley_Tukey (n1, n2) -> dft_ct ~sign n1 n2 input_re input_im
  | Split_radix ->
    (* Split-radix lives in its own module (lib/split_radix.ml). Cross-
     * module mutual recursion uses the callback pattern: we pass `dft`
     * itself in as `dft_rec` so SR can recurse on its sub-DFT inputs
     * (size N/2 and N/4) which dispatch back through the picker. *)
    if newsplit_enabled () then
      Split_radix.dft_newsplit ~sign n input_re input_im
    else
      Split_radix.dft_split_radix
        ~dft_rec:(fun ~sign:s n' f g -> dft ~sign:s n' f g)
        ~sign n input_re input_im

(* Direct DFT: matrix-vector form.
 *   X[k].re = Σ_n  a[n] * cos(±2πnk/n) - b[n] * sin(±2πnk/n)
 *   X[k].im = Σ_n  a[n] * sin(±2πnk/n) + b[n] * cos(±2πnk/n)
 * The sign of θ is - for Fwd, + for Bwd. *)
and dft_direct ?(sign = `Fwd) (n : int) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k = 0 to n - 1 do
    let re_sum = ref (Const 0.0) in
    let im_sum = ref (Const 0.0) in
    for nn = 0 to n - 1 do
      let theta = sgn *. 2.0 *. pi *. float_of_int (nn * k) /. float_of_int n in
      let c = cos theta in
      let s = sin theta in
      let a_nn = input_re nn in
      let b_nn = input_im nn in
      re_sum := Add (!re_sum, Sub (Mul (a_nn, Const c), Mul (b_nn, Const s)));
      im_sum := Add (!im_sum, Add (Mul (a_nn, Const s), Mul (b_nn, Const c)))
    done;
    out_re.(k) <- !re_sum;
    out_im.(k) <- !im_sum
  done;
  (out_re, out_im)

(* === DIRECT DFT WITH EXPLICIT CONJUGATE-PAIR FACTORING ===
 *
 * For a real input pair (x[j], x[N-j]) with j in 1..(N-1)/2, the
 * twiddle factors satisfy:
 *
 *   cos(2π·j·m/N)     =  cos(2π·(N-j)·m/N)     [cos is even]
 *   sin(2π·j·m/N)     = -sin(2π·(N-j)·m/N)     [sin is odd]
 *
 * So for forward DFT (using exp(-2πi·j·m/N) = cos(2πjm/N) - i·sin(2πjm/N)):
 *
 *   X[m].re = x[0].re + Σ_{j=1..H} (cos(jm)·s_re_j + sin(jm)·d_im_j)
 *   X[N-m].re = x[0].re + Σ_{j=1..H} (cos(jm)·s_re_j - sin(jm)·d_im_j)
 *
 *   X[m].im = x[0].im + Σ_{j=1..H} (cos(jm)·s_im_j - sin(jm)·d_re_j)
 *   X[N-m].im = x[0].im + Σ_{j=1..H} (cos(jm)·s_im_j + sin(jm)·d_re_j)
 *
 * where H = (N-1)/2 and:
 *   s_re_j = x[j].re + x[N-j].re      d_re_j = x[j].re - x[N-j].re
 *   s_im_j = x[j].im + x[N-j].im      d_im_j = x[j].im - x[N-j].im
 *
 * Per pair (m, N-m) we compute four shared intermediates ONCE:
 *   p_re_m = Σ cos(jm)·s_re_j      (shared between X[m].re and X[N-m].re)
 *   p_im_m = Σ cos(jm)·s_im_j      (shared between X[m].im and X[N-m].im)
 *   q_re_m = Σ sin(jm)·d_re_j      (shared between im outputs, opposite signs)
 *   q_im_m = Σ sin(jm)·d_im_j      (shared between re outputs, opposite signs)
 *
 * The OCaml `expr` is value-shared for the four intermediates; Algsimp's
 * hash-cons preserves the sharing through `of_expr`. The outer outputs
 * use BINARY structure (no flattening) so the shared sub-trees remain
 * intact — caller must use `reassoc:false`.
 *
 * For backward DFT (sign=`Bwd), exp(+2πi·j·m/N), so sin signs flip:
 *
 *   X[m].re   = x[0].re + Σ (cos(jm)·s_re_j - sin(jm)·d_im_j)
 *   X[N-m].re = x[0].re + Σ (cos(jm)·s_re_j + sin(jm)·d_im_j)
 *   X[m].im   = x[0].im + Σ (cos(jm)·s_im_j + sin(jm)·d_re_j)
 *   X[N-m].im = x[0].im + Σ (cos(jm)·s_im_j - sin(jm)·d_re_j)
 *)
and dft_direct_conjugate_pair ?(sign = `Fwd)
    (n : int) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in
  let half = (n - 1) / 2 in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in

  (* === STAGE 1: pair sums and diffs (the s_jk and d_jk subterms) ===
   * Computed once each, shared everywhere via OCaml value reuse → hash-cons. *)
  let s_re = Array.init (half + 1) (fun j ->
    if j = 0 then input_re 0 else Add (input_re j, input_re (n - j))) in
  let s_im = Array.init (half + 1) (fun j ->
    if j = 0 then input_im 0 else Add (input_im j, input_im (n - j))) in
  let d_re = Array.init (half + 1) (fun j ->
    if j = 0 then Const 0.0 else Sub (input_re j, input_re (n - j))) in
  let d_im = Array.init (half + 1) (fun j ->
    if j = 0 then Const 0.0 else Sub (input_im j, input_im (n - j))) in

  (* === STAGE 2: linear-chain weighted sums ===
   * Two variants:
   *
   * `make_sum_with_init initial coeffs terms`:
   *   Builds initial + sign(c1)|c1|·t1 + sign(c2)|c2|·t2 + ... as a left-fold
   *   chain. For positive c, emits `Add(acc, Mul(t, |c|))`; for negative c,
   *   emits `Sub(acc, Mul(t, |c|))`. After fma_lift:
   *     - Add(acc, Mul) lifts to fmadd  →  a*b + acc
   *     - Sub(acc, Mul) lifts to fnmadd →  -a*b + acc
   *   This produces a single FMA chain with mixed +/- coefficients encoded
   *   in the FMA opcode (matching FFTW codelet style). The deepest addend
   *   is `initial`, free at the asm level (it's the FMA's `c` operand).
   *
   * `make_sum coeffs terms`:
   *   Same but no initial accumulator — first term starts as a Mul.
   *   Used for q chains (sine sums) which don't have x[0] to absorb.
   *
   * Why sign-aware: when coeffs have mixed signs (typical for prime DFTs
   * where cos(2πjm/N) > 0 for some j and < 0 for others), our pipeline's
   * factor pass otherwise splits the sum into "positive coefficient" and
   * "negative coefficient" sub-chains, costing ~4 extra ops per pair output
   * block (start-mul on each sub-chain + extra structural sub for combining). *)
  let make_sum_with_init initial coeffs terms =
    let acc = ref initial in
    for j = 1 to half do
      let c = coeffs.(j) in
      let abs_c = Float.abs c in
      let term = Mul (terms.(j), Const abs_c) in
      acc := if c < 0.0 then Sub (!acc, term)
             else Add (!acc, term)
    done;
    !acc
  in
  let make_sum coeffs terms =
    let acc = ref None in
    for j = 1 to half do
      let c = coeffs.(j) in
      let abs_c = Float.abs c in
      let term = Mul (terms.(j), Const abs_c) in
      acc := match !acc with
        | None ->
          (* First term: if positive, start with the Mul as-is.
           * If negative, start with Neg(Mul). fma_lift catches the Neg(Mul)
           * pattern when it's then Add'd to something later, producing fnmadd. *)
          Some (if c < 0.0 then Neg term else term)
        | Some a ->
          Some (if c < 0.0 then Sub (a, term) else Add (a, term))
    done;
    match !acc with Some a -> a | None -> Const 0.0
  in

  (* === STAGE 3: X[0] — sum of all real/imag inputs ===
   * Using the s_re/s_im pair sums, X[0].re = x[0].re + Σ s_re_j
   * (each pair sum is reused from STAGE 1, no duplicate adds.) *)
  let x0_re = ref (input_re 0) in
  let x0_im = ref (input_im 0) in
  for j = 1 to half do
    x0_re := Add (!x0_re, s_re.(j));
    x0_im := Add (!x0_im, s_im.(j))
  done;
  out_re.(0) <- !x0_re;
  out_im.(0) <- !x0_im;

  (* === STAGE 4: pair outputs X[m] and X[N-m] for m = 1..half ===
   * Compute four intermediates once per pair, then combine. *)
  for m = 1 to half do
    let cos_arr = Array.init (half + 1) (fun j ->
      cos (2.0 *. pi *. float_of_int (j * m) /. float_of_int n)) in
    (* Sin coefficient for the q_im_m = Σ sin(jm)·d_im_j intermediate.
     * The output combinations below use:
     *   out_re.(m)     = x[0].re + p_re + q_im   ← needs +sin for forward
     *   out_re.(n-m)   = x[0].re + p_re - q_im
     *
     * For forward (exp(-iθ) = cos-i·sin), X[m].re = Σcos·s_re + Σsin·d_im,
     * so coefficient inside q_im needs to be +sin → factor = +1 → -sgn.
     *
     * For backward (exp(+iθ) = cos+i·sin), X[m].re = Σcos·s_re - Σsin·d_im,
     * so coefficient inside q_im is -sin → factor = -1 → -sgn.
     *
     * Both cases: sin_arr coefficient = -sgn · sin(2πjm/N). *)
    let sin_arr = Array.init (half + 1) (fun j ->
      (-. sgn) *. sin (2.0 *. pi *. float_of_int (j * m) /. float_of_int n)) in

    (* p_re_m / p_im_m: cosine sums WITH x[0] absorbed as the deepest addend.
     * The chain shape is:
     *   ((((x[0] +/- |c1|·s1) +/- |c2|·s2) +/- |c3|·s3) +/- |c4|·s4) +/- |c5|·s5
     * After fma_lift, this becomes 5 nested FMAs (mix of fmadd/fnmadd based on
     * coefficient signs). The deepest fma uses x[0] as its `c` addend (free).
     *
     * q_re_m / q_im_m: sine sums WITHOUT x[0] (first term starts as Mul or
     * Neg(Mul) depending on sign). After fma_lift: 1 mul + 4 fmas chain.
     * Output combines absorb sin chain via Add/Sub at top level. *)
    let p_re_m = make_sum_with_init (input_re 0) cos_arr s_re in
    let p_im_m = make_sum_with_init (input_im 0) cos_arr s_im in
    let q_re_m = make_sum sin_arr d_re in
    let q_im_m = make_sum sin_arr d_im in

    (* Output combinations. p_re_m / p_im_m already include x[0]; we just
     * add or subtract the q chain. Each output requires exactly 1 op
     * (1 add or 1 sub) at this combining level — matching hand-coded
     * FFTW-style codelet structure. *)
    out_re.(m)         <- Add (p_re_m, q_im_m);
    out_re.(n - m)     <- Sub (p_re_m, q_im_m);
    out_im.(m)         <- Sub (p_im_m, q_re_m);
    out_im.(n - m)     <- Add (p_im_m, q_re_m)
  done;
  (out_re, out_im)

(* General Cooley-Tukey DIT decomposition: N = N1 · N2.
 *
 * Standard DIT convention (matches user's gen_radix*.py):
 *   - Input mapping:  n = n1 + n2 · N1  (low digit n1, high digit n2)
 *   - Output mapping: k = k1 · N2 + k2  (high digit k1, low digit k2)
 *
 * Decomposition:
 *   X[k1·N2 + k2] = Σ_{n1} ω_{N1}^{n1·k1} ·
 *                   ω_N^{n1·k2} ·
 *                   Σ_{n2} x[n1 + n2·N1] · ω_{N2}^{n2·k2}
 *
 * Read as three nested operations:
 *   PASS 1: For each n1 (offset), compute DFT-N2 on the strided slice
 *           x[n1], x[n1 + N1], x[n1 + 2·N1], ..., x[n1 + (N2-1)·N1].
 *           Output: pass1[n1][k2].
 *   TWIDDLE: Multiply pass1[n1][k2] by the inter-stage twiddle ω_N^{n1·k2}.
 *   PASS 2: For each k2 (output sub-index), compute DFT-N1 on the
 *           column twiddled[0..N1-1][k2]. Output goes to X[k1·N2 + k2].
 *
 * Concrete examples:
 *   R=4 = CT(2, 2): PASS 1 splits inputs by parity (even/odd).
 *   R=8 = CT(2, 4): PASS 1 splits by parity, two DFT-4s (even/odd indices).
 *   R=16 = CT(4, 4): PASS 1 splits by mod-4 residue, four DFT-4s.
 *)
and dft_winograd5 ?(sign = `Fwd) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  (* Winograd 5-point DFT — exploits algebraic identities of 5th roots of
   * unity to reduce arithmetic vs. the naive direct DFT.
   *
   * Identities used:
   *   cos(2π/5) + cos(4π/5) = -1/2
   *   cos(2π/5) - cos(4π/5) =  √5/2
   *   sin(4π/5)/sin(2π/5)   =  1/φ   (= 2cos(2π/5) = (√5-1)/2)
   *
   * Four hard-coded constants:
   *   k_quarter  = 0.25
   *   k_root5_4  = √5/4
   *   k_sin_2pi5 = sin(2π/5)        ≈ 0.951
   *   k_inv_phi  = 1/φ              ≈ 0.618
   *
   * Op count: 14 add/sub + 18 fma = 32 total (matches FFTW gen_notw -fma).
   *
   * Compared with dft_direct_conjugate_pair's 36 ops at R=5: saves 4 ops
   * via the cos(2π/5) ± cos(4π/5) identity (Winograd cos channel uses 2
   * muls instead of 4) and the sin(4π/5) = sin(2π/5)/φ identity (Winograd
   * sin channel shares one s1 factor across both sin terms).
   *
   * Scheduling for register pressure: emit in the same order as FFTW's
   * gen_notw output — real pre-adds, then output 0, then real outputs
   * (which only need real pre-adds and the imag pair-diffs), then imag
   * outputs. The two channels share no live intermediates after their
   * own outputs are emitted, so peak live ≈ 14 IR nodes (well under the
   * 16-ymm AVX2 budget; trivially fits AVX-512).
   *
   * Sign convention: forward (`Fwd) uses ω = exp(-2πi/5), backward (`Bwd)
   * uses ω = exp(+2πi/5). The cos channel is sign-agnostic; only the
   * imag-channel contribution to real outputs (and vice versa) flips.
   * We handle this by flipping the sign of `k_inv_phi` and `k_sin_2pi5`
   * for Bwd in the appropriate slots. *)
  let two_pi = 8.0 *. atan 1.0 in
  let k_quarter  = Const 0.25 in
  let k_root5_4  = Const (sqrt 5.0 /. 4.0) in
  let k_inv_phi  = Const ((sqrt 5.0 -. 1.0) /. 2.0) in
  (* Sign convention for the sin-channel coupling. For Fwd DFT (θ = -2πnk/N),
   * working through the algebra of (x_n.re + i x_n.im) · (cos(θ) - i sin(θ))
   * gives  Re(X_1) = ... + sin(2π/5)·(x_1.im - x_4.im) + sin(4π/5)·(x_2.im - x_3.im)
   * with POSITIVE sin coupling. For Bwd (θ = +2πnk/N) the sign flips.
   * Absorb that into the constant so the structural signs in the output
   * assignments below stay identical for both directions. *)
  let s_sign = match sign with `Fwd -> 1.0 | `Bwd -> -1.0 in
  let k_sin_2pi5_s = Const (s_sign *. sin (two_pi /. 5.0)) in

  (* Real-channel pre-additions
   *   t4 = x_1.re + x_4.re      ta = t4 - t7
   *   t7 = x_2.re + x_3.re      tt = x_2.re - x_3.re
   *   t8 = t4 + t7              ts = x_1.re - x_4.re *)
  let x0r = input_re 0 in
  let x1r = input_re 1 in
  let x4r = input_re 4 in
  let t4  = Add (x1r, x4r) in
  let x2r = input_re 2 in
  let x3r = input_re 3 in
  let t7  = Add (x2r, x3r) in
  let t8  = Add (t4, t7) in
  let tt  = Sub (x2r, x3r) in
  let ta  = Sub (t4, t7) in
  let ts  = Sub (x1r, x4r) in

  (* Imag-channel pre-additions — same structure. *)
  let x0i = input_im 0 in
  let x1i = input_im 1 in
  let x4i = input_im 4 in
  let tm  = Add (x1i, x4i) in
  let x2i = input_im 2 in
  let x3i = input_im 3 in
  let tn  = Add (x2i, x3i) in
  let te  = Sub (x1i, x4i) in
  let tq  = Sub (tm, tn) in
  let th  = Sub (x2i, x3i) in
  let to_ = Add (tm, tn) in

  let out_re = Array.make 5 (Const 0.0) in
  let out_im = Array.make 5 (Const 0.0) in

  (* Output 0 — trivial sums, no multiplications. *)
  out_re.(0) <- Add (x0r, t8);
  out_im.(0) <- Add (x0i, to_);

  (* Real outputs X_1.re .. X_4.re.
   *   ti = te + (1/φ)·th       — shares sin1·(C.im + D.im·1/φ) = s1·C.im+s2·D.im
   *   tk = th - (1/φ)·te       — pairs for X_2/X_3
   *   t9 = x_0.re - 0.25·t8    — common real anchor
   *   tb = t9 + (√5/4)·ta      — cos-channel for X_1/X_4
   *   tj = t9 - (√5/4)·ta      — cos-channel for X_2/X_3
   * Then:
   *   X_1.re = tb + s1·ti      (FMA)
   *   X_4.re = tb - s1·ti      (FNMS)
   *   X_2.re = tj - s1·tk      (FNMS)
   *   X_3.re = tj + s1·tk      (FMA) *)
  let ti = Add (te, Mul (k_inv_phi, th)) in
  let tk = Sub (th, Mul (k_inv_phi, te)) in
  let t9 = Sub (x0r, Mul (k_quarter, t8)) in
  let tb = Add (t9, Mul (k_root5_4, ta)) in
  let tj = Sub (t9, Mul (k_root5_4, ta)) in
  out_re.(1) <- Add (tb, Mul (k_sin_2pi5_s, ti));
  out_re.(4) <- Sub (tb, Mul (k_sin_2pi5_s, ti));
  out_re.(2) <- Sub (tj, Mul (k_sin_2pi5_s, tk));
  out_re.(3) <- Add (tj, Mul (k_sin_2pi5_s, tk));

  (* Imag outputs X_1.im .. X_4.im. Mirror of real, with (ts, tt) playing
   * the role of (te, th), (tq, to_) playing (ta, t8), and the cross-
   * coupling sign flipped relative to the real channel — that flip is
   * absorbed into k_sin_2pi5_s above for the real channel; here the
   * direct s1·X form is used (sin contribution to imag is positive in
   * Fwd, negative in Bwd, which is what sgn captures via the same
   * k_sin_2pi5_s constant). *)
  let tu = Add (ts, Mul (k_inv_phi, tt)) in
  let tw = Sub (tt, Mul (k_inv_phi, ts)) in
  let tp = Sub (x0i, Mul (k_quarter, to_)) in
  let tr = Add (tp, Mul (k_root5_4, tq)) in
  let tv = Sub (tp, Mul (k_root5_4, tq)) in
  (* Note the sign flip relative to real channel: imag outputs use the
   * OPPOSITE sign on the s1·tu and s1·tw FMAs. For Fwd, X_1.im uses
   * `tr - s1·tu`; for Bwd, `tr + s1·tu`. The k_sin_2pi5_s constant
   * already carries `sgn`, so we keep the structural sign as +real and
   * −imag (which becomes −real, +imag for Bwd). *)
  out_im.(1) <- Sub (tr, Mul (k_sin_2pi5_s, tu));
  out_im.(4) <- Add (tr, Mul (k_sin_2pi5_s, tu));
  out_im.(2) <- Add (tv, Mul (k_sin_2pi5_s, tw));
  out_im.(3) <- Sub (tv, Mul (k_sin_2pi5_s, tw));

  (out_re, out_im)

(* ============================================================
 * dft_winograd5_cnum — same algorithm as dft_winograd5, written
 * via the Cnum combinator layer.
 *
 * This is a TEST CASE for the Cnum infrastructure: it should produce
 * the same (or strictly fewer) ops as dft_winograd5. If it doesn't,
 * something is wrong with Cnum or the Expr smart constructors.
 *
 * The algebra is identical to dft_winograd5. What changes is the
 * SHAPE of the constructed Expr tree — specifically, the `Mul`
 * nodes are placed at leaves of `Sub`/`Add` chains via mk_sub/mk_add,
 * letting the Expr.mk_mul rotation rule fire when downstream code
 * multiplies this by a constant. (For W5 used standalone, there is
 * no downstream constant multiplication; the rotation only matters
 * when W5 is composed in a larger codelet via cscale or cmul.)
 * ============================================================ *)
and dft_winograd5_cnum ?(sign = `Fwd) (input : int -> Cnum.cnum)
    : Cnum.cnum array =
  let two_pi = 8.0 *. atan 1.0 in
  let k_quarter  = Const 0.25 in
  let k_root5_4  = Const (sqrt 5.0 /. 4.0) in
  let k_inv_phi  = Const ((sqrt 5.0 -. 1.0) /. 2.0) in
  let s_sign = match sign with `Fwd -> 1.0 | `Bwd -> -1.0 in
  let k_sin_2pi5_s = Const (s_sign *. sin (two_pi /. 5.0)) in

  let open Expr in
  let open Cnum in

  let x0 = input 0 in
  let x1 = input 1 in
  let x2 = input 2 in
  let x3 = input 3 in
  let x4 = input 4 in

  (* Real-channel pre-additions. *)
  let t4 = mk_add x1.re x4.re in
  let t7 = mk_add x2.re x3.re in
  let t8 = mk_add t4 t7 in
  let tt = mk_sub x2.re x3.re in
  let ta = mk_sub t4 t7 in
  let ts = mk_sub x1.re x4.re in

  (* Imag-channel pre-additions. *)
  let tm = mk_add x1.im x4.im in
  let tn = mk_add x2.im x3.im in
  let te = mk_sub x1.im x4.im in
  let tq = mk_sub tm tn in
  let th = mk_sub x2.im x3.im in
  let to_ = mk_add tm tn in

  let out = Array.make 5 czero in

  (* Output 0 — trivial sums. *)
  out.(0) <- cnum (mk_add x0.re t8) (mk_add x0.im to_);

  (* Real outputs X_1.re .. X_4.re *)
  let ti = mk_add te (mk_mul k_inv_phi th) in
  let tk = mk_sub th (mk_mul k_inv_phi te) in
  let t9 = mk_sub x0.re (mk_mul k_quarter t8) in
  let tb = mk_add t9 (mk_mul k_root5_4 ta) in
  let tj = mk_sub t9 (mk_mul k_root5_4 ta) in
  let x1_re = mk_add tb (mk_mul k_sin_2pi5_s ti) in
  let x4_re = mk_sub tb (mk_mul k_sin_2pi5_s ti) in
  let x2_re = mk_sub tj (mk_mul k_sin_2pi5_s tk) in
  let x3_re = mk_add tj (mk_mul k_sin_2pi5_s tk) in

  (* Imag outputs X_1.im .. X_4.im *)
  let tu = mk_add ts (mk_mul k_inv_phi tt) in
  let tw = mk_sub tt (mk_mul k_inv_phi ts) in
  let tp = mk_sub x0.im (mk_mul k_quarter to_) in
  let tr = mk_add tp (mk_mul k_root5_4 tq) in
  let tv = mk_sub tp (mk_mul k_root5_4 tq) in
  let x1_im = mk_sub tr (mk_mul k_sin_2pi5_s tu) in
  let x4_im = mk_add tr (mk_mul k_sin_2pi5_s tu) in
  let x2_im = mk_add tv (mk_mul k_sin_2pi5_s tw) in
  let x3_im = mk_sub tv (mk_mul k_sin_2pi5_s tw) in

  out.(1) <- cnum x1_re x1_im;
  out.(2) <- cnum x2_re x2_im;
  out.(3) <- cnum x3_re x3_im;
  out.(4) <- cnum x4_re x4_im;
  out

(* ============================================================
 * dft_winograd25 — hand-derived N=25 Winograd codelet.
 *
 * CT(5,5) decomposition with PLUS-OF-TIMES twiddles (instead of the
 * tan-factored form const_cmul uses).
 *
 * Plus-of-times: z = T * Y emitted as
 *   z.re = Sub(Mul(cr, Y.re), Mul(ci, Y.im))
 *   z.im = Add(Mul(cr, Y.im), Mul(ci, Y.re))
 *
 * Tan-factored (current dft_ct + const_cmul):
 *   z.re = Mul(cr, Sub(Y.re, Mul(tan, Y.im)))      [outer Mul opaque]
 *   z.im = Mul(cr, Add(Y.im, Mul(tan, Y.re)))
 *
 * The two forms compute the same value but have different IR shape.
 * Plus-of-times places all multiplications at leaf level, which lets
 * fma_lift fuse them into surrounding Add/Sub chains. Tan-factored
 * wraps the result in an outer Mul that has no FMA target.
 *
 * Trade-off: plus-of-times has 4 muls per cmul (both forms have 4
 * abstract muls, but plus-of-times keeps them leaf-level vs tan's outer).
 * Whether this nets a savings depends on how pass-2 W5 consumes the
 * twiddled values — specifically whether the W5 internal sums benefit
 * from leaf-level Mul shapes available for FMA fusion.
 *)
and dft_winograd25 ?(sign = `Fwd)
    (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  let two_pi = 8.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in
  let n = 25 in

  (* Pass 1: 5 W5s on input columns. *)
  let p1_re = Array.make_matrix 5 5 (Const 0.0) in
  let p1_im = Array.make_matrix 5 5 (Const 0.0) in
  for j_0 = 0 to 4 do
    let col_re k = input_re (j_0 + 5 * k) in
    let col_im k = input_im (j_0 + 5 * k) in
    let (r, i) = dft_winograd5 ~sign col_re col_im in
    for k = 0 to 4 do
      p1_re.(j_0).(k) <- r.(k);
      p1_im.(j_0).(k) <- i.(k)
    done
  done;

  (* Twiddle: plus-of-times form (NOT tan-factored).
   * For (j_0, k_0) with j_0=0 or k_0=0, twiddle is 1 — pass through. *)
  let z_re = Array.make_matrix 5 5 (Const 0.0) in
  let z_im = Array.make_matrix 5 5 (Const 0.0) in
  for j_0 = 0 to 4 do
    for k_0 = 0 to 4 do
      let theta = sgn *. two_pi *. float_of_int (j_0 * k_0) /. float_of_int n in
      let cr = cos theta in
      let ci = sin theta in
      let yr = p1_re.(j_0).(k_0) in
      let yi = p1_im.(j_0).(k_0) in
      if j_0 = 0 || k_0 = 0 then begin
        (* Identity twiddle. *)
        z_re.(j_0).(k_0) <- yr;
        z_im.(j_0).(k_0) <- yi
      end else begin
        (* Plus-of-times: all 4 muls at leaf level. *)
        let cr_e = Const cr in
        let ci_e = Const ci in
        z_re.(j_0).(k_0) <- Sub (Mul (cr_e, yr), Mul (ci_e, yi));
        z_im.(j_0).(k_0) <- Add (Mul (cr_e, yi), Mul (ci_e, yr))
      end
    done
  done;

  (* Pass 2: 5 W5s on rows of z. *)
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k_0 = 0 to 4 do
    let row_re j_0 = z_re.(j_0).(k_0) in
    let row_im j_0 = z_im.(j_0).(k_0) in
    let (r, i) = dft_winograd5 ~sign row_re row_im in
    for k_1 = 0 to 4 do
      out_re.(k_0 + 5 * k_1) <- r.(k_1);
      out_im.(k_0 + 5 * k_1) <- i.(k_1)
    done
  done;

  (out_re, out_im)

and dft_winograd7 ?(sign = `Fwd) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  (* Winograd 7-point DFT — Rader-style decomposition exploiting the
   * multiplicative-group structure of (Z/7Z)*. The cyclic-convolution
   * subproblem factors via Winograd's small-convolution algorithms,
   * yielding 18 add/sub + 42 fma = 60 ops total. Matches FFTW's
   * gen_notw -fma -n 7 codelet exactly (vs 66 ops from our generic
   * conjugate-pair Direct path).
   *
   * Six derived constants come out of the Winograd derivation. Of these,
   * KP_974927912 = sin(4π/7) is the only one carrying cross-channel sin
   * coupling — it flips sign for Bwd. The other five are sign-invariant
   * (rational combinations of the cos values).
   *
   * Algorithm mirrors FFTW's emitted ordering (the algebra is otherwise
   * the same): real and imag pre-additions, output 0, then three output
   * pairs (1,6), (2,5), (3,4). Within each pair, lower-indexed output is
   * FMA, higher-indexed is FNMS — the sign convention encoded structurally
   * for Fwd, flipped by KP_974927912's sign for Bwd.
   *
   * Note the imag-channel pair-diffs use LOW-HIGH (Tj = x_1.im - x_6.im,
   * etc.) while real-channel uses HIGH-LOW (TI = x_6.re - x_1.re). This
   * asymmetry handles the sign of sin in the DFT formula and is required
   * for correctness — flipping it would invert the imag outputs. *)
  let kp_356895867 = Const 0.356895867892209443894399510021300583399127187 in
  let kp_554958132 = Const 0.554958132087371191422194871006410481067288862 in
  let kp_801937735 = Const 0.801937735804838252472204639014890102331838324 in
  let kp_692021471 = Const 0.692021471630095869627814897002069140197260599 in
  let kp_900968867 = Const 0.900968867902419126236102319507445051165919162 in
  let s_sign = match sign with `Fwd -> 1.0 | `Bwd -> -1.0 in
  let kp_974927912_s =
    Const (s_sign *. 0.974927912181823607018131682993931217232785801)
  in

  (* Real-channel pre-additions *)
  let x0r = input_re 0 in
  let x1r = input_re 1 in
  let x2r = input_re 2 in
  let x3r = input_re 3 in
  let x4r = input_re 4 in
  let x5r = input_re 5 in
  let x6r = input_re 6 in
  let t_4  = Add (x1r, x6r) in    (* T4: pair sum  (x1+x6) *)
  let t_i  = Sub (x6r, x1r) in    (* TI: pair diff (x6-x1)  [HIGH-LOW] *)
  let t_7  = Add (x2r, x5r) in
  let t_h  = Sub (x5r, x2r) in
  let t_a  = Add (x3r, x4r) in
  let t_g  = Sub (x4r, x3r) in
  let t_b  = Sub (t_4, Mul (kp_356895867, t_7)) in  (* Tb = T4 - K356·T7 *)
  let t_p  = Sub (t_a, Mul (kp_356895867, t_4)) in
  let t_u  = Sub (t_7, Mul (kp_356895867, t_a)) in
  let t_tt = Add (t_i, Mul (kp_554958132, t_g)) in  (* TT = TI + K555·TG *)
  let t_o2 = Add (t_g, Mul (kp_554958132, t_h)) in  (* TO = TG + K555·TH *)
  let t_jj = Sub (t_h, Mul (kp_554958132, t_i)) in  (* TJ = TH - K555·TI *)

  (* Imag-channel pre-additions  (pair diffs are LOW-HIGH here) *)
  let x0i = input_im 0 in
  let x1i = input_im 1 in
  let x2i = input_im 2 in
  let x3i = input_im 3 in
  let x4i = input_im 4 in
  let x5i = input_im 5 in
  let x6i = input_im 6 in
  let t_aA = Add (x1i, x6i) in    (* TA: pair sum *)
  let t_jI = Sub (x1i, x6i) in    (* Tj: pair diff (LOW-HIGH) *)
  let t_bB = Add (x2i, x5i) in
  let t_gI = Sub (x2i, x5i) in
  let t_cC = Add (x3i, x4i) in
  let t_mM = Sub (x3i, x4i) in
  let t_qQ = Sub (t_aA, Mul (kp_356895867, t_bB)) in   (* TQ = TA - K356·TB *)
  let t_lL = Sub (t_cC, Mul (kp_356895867, t_aA)) in
  let t_dD = Sub (t_bB, Mul (kp_356895867, t_cC)) in
  let t_nN = Add (t_jI, Mul (kp_554958132, t_mM)) in   (* Tn = Tj + K555·Tm *)
  let t_sS = Add (t_mM, Mul (kp_554958132, t_gI)) in   (* Ts = Tm + K555·Tg *)
  let t_xX = Sub (t_gI, Mul (kp_554958132, t_jI)) in   (* Tx = Tg - K555·Tj *)

  let out_re = Array.make 7 (Const 0.0) in
  let out_im = Array.make 7 (Const 0.0) in

  (* Output 0 — chained sums, 3 adds each channel *)
  out_re.(0) <- Add (Add (Add (x0r, t_4), t_7), t_a);
  out_im.(0) <- Add (Add (Add (x0i, t_aA), t_bB), t_cC);

  (* Pair (1, 6) *)
  let to_ = Add (t_gI, Mul (kp_801937735, t_nN)) in     (* To = Tg + K801·Tn *)
  let t_c = Sub (t_a, Mul (kp_692021471, t_b)) in       (* Tc = Ta - K692·Tb *)
  let t_d = Sub (x0r, Mul (kp_900968867, t_c)) in       (* Td = x_0 - K900·Tc *)
  out_re.(1) <- Add (t_d, Mul (kp_974927912_s, to_));
  out_re.(6) <- Sub (t_d, Mul (kp_974927912_s, to_));
  let t_u2 = Add (t_h, Mul (kp_801937735, t_tt)) in
  let t_r2 = Sub (t_cC, Mul (kp_692021471, t_qQ)) in
  let t_s2 = Sub (x0i, Mul (kp_900968867, t_r2)) in
  out_im.(1) <- Add (t_s2, Mul (kp_974927912_s, t_u2));
  out_im.(6) <- Sub (t_s2, Mul (kp_974927912_s, t_u2));

  (* Pair (2, 5) *)
  let t_tT = Sub (t_jI, Mul (kp_801937735, t_sS)) in    (* Tt = Tj - K801·Ts *)
  let t_q  = Sub (t_7, Mul (kp_692021471, t_p)) in
  let t_r  = Sub (x0r, Mul (kp_900968867, t_q)) in
  out_re.(2) <- Add (t_r, Mul (kp_974927912_s, t_tT));
  out_re.(5) <- Sub (t_r, Mul (kp_974927912_s, t_tT));
  let t_pP = Sub (t_i, Mul (kp_801937735, t_o2)) in     (* TP = TI - K801·TO *)
  let t_mM2 = Sub (t_bB, Mul (kp_692021471, t_lL)) in
  let t_nN2 = Sub (x0i, Mul (kp_900968867, t_mM2)) in
  out_im.(2) <- Add (t_nN2, Mul (kp_974927912_s, t_pP));
  out_im.(5) <- Sub (t_nN2, Mul (kp_974927912_s, t_pP));

  (* Pair (3, 4) *)
  let t_y  = Sub (t_mM, Mul (kp_801937735, t_xX)) in
  let t_v  = Sub (t_4, Mul (kp_692021471, t_u)) in
  let t_w  = Sub (x0r, Mul (kp_900968867, t_v)) in
  out_re.(3) <- Add (t_w, Mul (kp_974927912_s, t_y));
  out_re.(4) <- Sub (t_w, Mul (kp_974927912_s, t_y));
  let t_kK = Sub (t_g, Mul (kp_801937735, t_jj)) in
  let t_eE = Sub (t_aA, Mul (kp_692021471, t_dD)) in
  let t_fF = Sub (x0i, Mul (kp_900968867, t_eE)) in
  out_im.(3) <- Add (t_fF, Mul (kp_974927912_s, t_kK));
  out_im.(4) <- Sub (t_fF, Mul (kp_974927912_s, t_kK));

  (out_re, out_im)

and dft_ct ?(sign = `Fwd) (n1 : int) (n2 : int)
           (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  let n = n1 * n2 in
  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in

  (* PASS 1: N1 sub-FFTs of size N2.
   * For each n1_idx in [0, N1), compute DFT-N2 on inputs at
   *   x[n1_idx], x[n1_idx + N1], x[n1_idx + 2·N1], ...
   * pass1[n1_idx][k2] = DFT-N2 result at output bin k2. *)
  let pass1_re = Array.make_matrix n1 n2 (Const 0.0) in
  let pass1_im = Array.make_matrix n1 n2 (Const 0.0) in
  for n1_idx = 0 to n1 - 1 do
    let inner_input_re k2 = input_re (n1_idx + k2 * n1) in
    let inner_input_im k2 = input_im (n1_idx + k2 * n1) in
    let r, i = dft ~sign n2 inner_input_re inner_input_im in
    for k2 = 0 to n2 - 1 do
      pass1_re.(n1_idx).(k2) <- r.(k2);
      pass1_im.(n1_idx).(k2) <- i.(k2)
    done
  done;

  (* INTERNAL TWIDDLES: multiply pass1[n1_idx][k2] by ω_N^{n1_idx·k2}.
   * For n1_idx=0 or k2=0, the twiddle is 1 and const_cmul folds away.
   * Other (n1_idx, k2) pairs introduce non-trivial cmul nodes. *)
  let twiddled_re = Array.make_matrix n1 n2 (Const 0.0) in
  let twiddled_im = Array.make_matrix n1 n2 (Const 0.0) in
  for n1_idx = 0 to n1 - 1 do
    for k2 = 0 to n2 - 1 do
      let theta = sgn *. 2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
      let cr = cos theta in
      let ci = sin theta in
      let (tr, ti) = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
      twiddled_re.(n1_idx).(k2) <- tr;
      twiddled_im.(n1_idx).(k2) <- ti
    done
  done;

  (* PASS 2: N2 sub-FFTs of size N1.
   * For each k2 in [0, N2), compute DFT-N1 on the column
   *   twiddled[0..N1-1][k2]
   * Output: X[k1·N2 + k2] = pass2_result[k1]. *)
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k2 = 0 to n2 - 1 do
    let outer_input_re n1_idx = twiddled_re.(n1_idx).(k2) in
    let outer_input_im n1_idx = twiddled_im.(n1_idx).(k2) in
    let r, i = dft ~sign n1 outer_input_re outer_input_im in
    for k1 = 0 to n1 - 1 do
      out_re.(k1 * n2 + k2) <- r.(k1);
      out_im.(k1 * n2 + k2) <- i.(k1)
    done
  done;

  (out_re, out_im)

(* === TWIDDLE POLICY ===
 *
 * Different codelet variants source their twiddles differently:
 *   TP_Flat — load all (R-1) twiddles directly from a flat buffer.
 *             Maximum loads, minimum FMA. Wins when FMA ports are
 *             saturated and twiddle bandwidth is plentiful.
 *
 *   TP_Log3 — load only base twiddles (R=4: just W^1; R=8: W^1, W^2, W^4)
 *             and derive the rest by complex multiplication.
 *             Saves twiddle loads, costs extra FMAs. Wins when L1
 *             twiddle pressure dominates (high K, deep transforms).
 *
 * The math-layer payoff: log3 is a *substitution* — wherever t1_dit
 * emits Load(Twiddle(j, ...)), log3 substitutes the derivation tree.
 * Algsimp's Cmul handling propagates; nothing else changes. *)

type twiddle_policy =
  | TP_Flat
  | TP_Log3

(* DIT vs DIF — duals of each other:
 *   DIT (Decimation-In-Time):   y = DFT(W ⋅ x)   — twiddle on INPUT, pre-butterfly
 *   DIF (Decimation-In-Frequency): y = W ⋅ DFT(x) — twiddle on OUTPUT, post-butterfly
 *
 * In a CT recursion, you typically pair DIT codelets at one level with
 * DIF codelets at the next so that the twiddle layer flips. FFTW emits
 * both styles for this reason. *)
type direction =
  | DIT
  | DIF

(* Build a complex multiplication as (out_re, out_im) using the cmul pattern
 * that Algsimp.of_expr will lift to Cmul nodes.
 *
 *   ~conj:false (default)  →  (a + ib) · (c + id)        (Fwd external twiddle)
 *   ~conj:true             →  (a + ib) · conj(c + id)   (Bwd external twiddle)
 *
 * Bwd codelets receive the SAME W array as fwd (caller stores forward
 * twiddles); the codelet conjugates internally via this flag.
 * Internal log3 derivations multiply two stored W values that share the
 * same convention (whatever it is) — those calls use ~conj:false.
 *)
let cmul_pattern ?(conj = false) (ar : expr) (ai : expr) (br : expr) (bi : expr)
    : expr * expr =
  if conj then
    (* (a + ib) · (c - id) = (ac + bd) + i(bc - ad) *)
    let out_re = Add (Mul (ar, br), Mul (ai, bi)) in
    let out_im = Sub (Mul (ai, br), Mul (ar, bi)) in
    (out_re, out_im)
  else
    let out_re = Sub (Mul (ar, br), Mul (ai, bi)) in
    let out_im = Add (Mul (ar, bi), Mul (ai, br)) in
    (out_re, out_im)

(* Compute the (re, im) Expr trees for the j-th twiddle (1-indexed)
 * under the given policy and radix. Memoization is critical: derived
 * twiddles like W^7 = W^3·W^4 reference W^3, which must be the SAME
 * expression tree (pointer-equal at the Expr level, hash-cons-equal
 * after lifting) so that the underlying cmul gets shared. *)
let twiddle_expr (policy : twiddle_policy) (n : int) (j : int)
    : expr * expr =
  let cache : (int, expr * expr) Hashtbl.t = Hashtbl.create 16 in
  let rec lookup j =
    match Hashtbl.find_opt cache j with
    | Some result -> result
    | None ->
      let result = compute j in
      Hashtbl.add cache j result;
      result
  and compute j =
    match policy with
    | TP_Flat ->
      (* Direct load from slot (j-1). *)
      (Load (Twiddle (j - 1, true)), Load (Twiddle (j - 1, false)))

    | TP_Log3 ->
      (* Generalized log3: load only the power-of-2 twiddles W^(2^k),
       * derive everything else by binary decomposition.
       *
       * Slot indexing matches TP_Flat (slot = j - 1) so the bench
       * harness fills the same twiddle array regardless of policy —
       * TP_Log3 simply consults a sparse subset of slots:
       *   W^1 → slot 0, W^2 → slot 1, W^4 → slot 3,
       *   W^8 → slot 7, W^16 → slot 15, W^32 → slot 31.
       *
       * Total slot reads per kstep:
       *   R=16: 4 (vs 15 flat)
       *   R=32: 5 (vs 31 flat)
       *   R=64: 6 (vs 63 flat)
       *
       * Decomposition: split j = p + q where p is the highest power
       * of 2 ≤ j. With memoization, each W^k computed once; hash-cons
       * dedupes across legs.
       *
       * Cmul cost per derivation: 4 muls + 2 adds (6 flops). Total:
       *   R=16: 4 loads + 11 cmuls
       *   R=32: 5 loads + 26 cmuls
       *   R=64: 6 loads + 57 cmuls
       *
       * Tradeoff: log3 saves twiddle bandwidth at the cost of arith.
       * Whether it wins depends on which is the bottleneck. *)
      let is_pow2 x = x > 0 && (x land (x - 1)) = 0 in
      let highest_pow2_le j =
        let rec loop p = if p * 2 > j then p else loop (p * 2) in
        loop 1
      in
      if j < 1 || j >= n then
        failwith (Printf.sprintf "TP_Log3: j=%d out of range for n=%d" j n)
      else if is_pow2 j then
        (* Direct load — slot = j - 1, matching TP_Flat layout. *)
        (Load (Twiddle (j - 1, true)), Load (Twiddle (j - 1, false)))
      else
        (* Split j = p + q, derive W^j = W^p · W^q. *)
        let p = highest_pow2_le j in
        let q = j - p in
        let (wpr, wpi) = lookup p in
        let (wqr, wqi) = lookup q in
        cmul_pattern wpr wpi wqr wqi
  in
  lookup j

(* === ASSIGNMENT-LIST WRAPPERS ===
 *
 * Replace the old dft_expand / dft_expand_twiddled with versions that
 * call into this module's recursive dft. The output is the same
 * Expr.assignment list shape as before. *)

let dft_expand ?(sign = `Fwd) (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let out_re, out_im = dft ~sign n input_re input_im in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  (* Reorder so each output's re comes immediately before its im. *)
  let pairs = List.rev !acc in
  (* `acc` was built (re, im, re, im, ...) in reverse, so List.rev
   * gives the correct order. *)
  let _ = pairs in
  List.rev !acc

(* Twiddled (t1_dit) form: pre-multiply inputs by runtime twiddles
 * (Cmul nodes), then run the (possibly Cooley-Tukey-decomposed) DFT
 * on the twiddled inputs.
 *
 * The cmul outputs use the SUB(MUL,MUL)/ADD(MUL,MUL) pattern that
 * Algsimp.of_expr's pattern detector recognizes and lifts to Cmul
 * opaque atoms. The CT decomposition then sees the cmul outputs as
 * leaf-like values in its own Add/Sub structure — algsimp won't
 * shred them because they're inside Cmul nodes after lifting. *)
let dft_expand_twiddled ?(policy = TP_Flat) ?(direction = DIT) ?(sign = `Fwd)
    (n : int) : Expr.assignment list =
  (* Structurally, twiddled codelets have two options:
   *   - PRE-twiddle: multiply inputs by twiddles, then run DFT
   *   - POST-twiddle: run DFT, then multiply outputs by twiddles
   *
   * For forward direction the choice mirrors the algorithm name:
   *   DIT fwd: PRE-twiddle  (decimation-in-time)
   *   DIF fwd: POST-twiddle (decimation-in-frequency)
   *
   * For backward (sign=Bwd), the codelet must be the inverse of the
   * corresponding forward codelet. Since (T·B)⁻¹ = B⁻¹·T⁻¹ and twiddle
   * doesn't commute with butterfly, the inverse has the OPPOSITE order:
   *   DIT bwd = inverse of (T then B) = B⁻¹ then T⁻¹  → POST-twiddle structure
   *   DIF bwd = inverse of (B then T) = T⁻¹ then B⁻¹ → PRE-twiddle structure
   *
   * The inverse butterfly B⁻¹ is the +θ DFT kernel (handled by `dft ~sign`)
   * and T⁻¹ for unit-magnitude twiddles is conj(T) (handled by ~conj here).
   *
   * Twiddle Exprs may be cmul derivation patterns (TP_Log3) or simple
   * Loads (TP_Flat). The per-leg Sub(Mul,Mul)/Add(Mul,Mul) cmul pattern
   * is preserved so Algsimp.of_expr can lift it to Cmul opaque atoms. *)
  let conj = (sign = `Bwd) in
  let pre_twiddle =
    match direction, sign with
    | DIT, `Fwd -> true   (* DIT fwd: T then B *)
    | DIT, `Bwd -> false  (* inverse of DIT fwd: B⁻¹ then T_conj  → POST *)
    | DIF, `Fwd -> false  (* DIF fwd: B then T *)
    | DIF, `Bwd -> true   (* inverse of DIF fwd: T_conj then B⁻¹  → PRE *)
  in
  if pre_twiddle then begin
    (* PRE-twiddle: multiply inputs by twiddles (conj if Bwd), then DFT. *)
    let twiddled_re = Array.make n (Const 0.0) in
    let twiddled_im = Array.make n (Const 0.0) in
    twiddled_re.(0) <- Load (Input (0, true));
    twiddled_im.(0) <- Load (Input (0, false));
    for k = 1 to n - 1 do
      let xr = Load (Input (k, true)) in
      let xi = Load (Input (k, false)) in
      let (wr, wi) = twiddle_expr policy n k in
      let (out_re, out_im) = cmul_pattern ~conj xr xi wr wi in
      twiddled_re.(k) <- out_re;
      twiddled_im.(k) <- out_im
    done;
    let input_re k = twiddled_re.(k) in
    let input_im k = twiddled_im.(k) in
    let out_re, out_im = dft ~sign n input_re input_im in
    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true),  out_re.(k))  :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    List.rev !acc
  end else begin
    (* POST-twiddle: run DFT on raw inputs, then twiddle the outputs
     * (conj if Bwd). *)
    let input_re k = Load (Input (k, true)) in
    let input_im k = Load (Input (k, false)) in
    let raw_re, raw_im = dft ~sign n input_re input_im in
    let acc = ref [] in
    (* Output 0: trivial twiddle, store as-is. *)
    acc := (Output (0, true),  raw_re.(0)) :: !acc;
    acc := (Output (0, false), raw_im.(0)) :: !acc;
    for k = 1 to n - 1 do
      let (wr, wi) = twiddle_expr policy n k in
      let (out_re, out_im) = cmul_pattern ~conj raw_re.(k) raw_im.(k) wr wi in
      acc := (Output (k, true),  out_re) :: !acc;
      acc := (Output (k, false), out_im) :: !acc
    done;
    let sorted = List.sort (fun (a, _) (b, _) ->
      match a, b with
      | Output (ka, ra), Output (kb, rb) ->
        if ka <> kb then compare ka kb
        else compare (not ra) (not rb)
      | _ -> 0
    ) !acc in
    sorted
  end

(* === OUT-OF-PLACE TWIDSQ EXPANSION (FFTW-style intermediate codelet) ===
 *
 * Builds the assignment DAG for an n×n "twiddle square" codelet:
 *   - Input:  row-major n×n block of complex values (n² elements)
 *   - Operation: apply inter-stage twiddle W^{i*k} to each (i, k), then
 *                run an n-point DFT along the k dimension of each row,
 *                producing row-result Y[i, j] for i, j ∈ [0, n).
 *   - Output: transposed layout — physical slot j*n + i gets Y[i, j].
 *
 * This corresponds to FFTW's gen_twidsq.ml codelet. Used at intermediate
 * stages of a multi-stage cascade where the layout transformation between
 * stages would otherwise require a separate transpose pass.
 *
 * Indexing conventions:
 *   - Input(i*n + k, _)            ← element at row i, position k
 *   - Twiddle((i-1)*(n-1) + (k-1), _) ← W^{i*k} for i ∈ [1, n), k ∈ [1, n)
 *     (Row 0 and column 0 have trivial W^0 = 1; no twiddle slots needed.)
 *   - Output(j*n + i, _)           ← row i's j-th DFT output, transposed store
 *
 * Number of twiddle slots: (n-1)² distinct values.
 *
 * NOTE: For the initial prototype we support DIT, TP_Flat only. DIF and
 * TP_Log3 can be added by parallel construction once the DAG semantics
 * are validated. Spill markers are not yet generated — large twidsq sizes
 * will need a parallel _spill variant similar to dft_expand_twiddled_spill.
 *)
let dft_expand_twidsq ?(direction = DIT) ?(sign = `Fwd)
    (n : int) : Expr.assignment list =
  let conj = (sign = `Bwd) in
  match direction with
  | DIT ->
    let acc = ref [] in
    for i = 0 to n - 1 do
      (* Step 1: Apply inter-stage twiddle to row i.
       * For position k > 0 of row i > 0: multiply by W^{i*k}.
       * For row 0 or position 0: pass-through (W^0 = 1). *)
      let twiddled_re = Array.make n (Const 0.0) in
      let twiddled_im = Array.make n (Const 0.0) in
      for k = 0 to n - 1 do
        let xr = Load (Input (i * n + k, true)) in
        let xi = Load (Input (i * n + k, false)) in
        if i = 0 || k = 0 then begin
          twiddled_re.(k) <- xr;
          twiddled_im.(k) <- xi
        end else begin
          let twiddle_slot = (i - 1) * (n - 1) + (k - 1) in
          let wr = Load (Twiddle (twiddle_slot, true)) in
          let wi = Load (Twiddle (twiddle_slot, false)) in
          let (out_re, out_im) = cmul_pattern ~conj xr xi wr wi in
          twiddled_re.(k) <- out_re;
          twiddled_im.(k) <- out_im
        end
      done;
      (* Step 2: Compute DFT-n on the twiddled row.
       * Reuses the existing dft machinery — math layer is buffer-agnostic. *)
      let input_re k = twiddled_re.(k) in
      let input_im k = twiddled_im.(k) in
      let out_re, out_im = dft ~sign n input_re input_im in
      (* Step 3: Store row i's outputs TRANSPOSED.
       * Y[i, j] goes to physical slot j*n + i (transposed layout). *)
      for j = n - 1 downto 0 do
        acc := (Output (j * n + i, true),  out_re.(j))  :: !acc;
        acc := (Output (j * n + i, false), out_im.(j)) :: !acc
      done
    done;
    List.rev !acc

  | DIF ->
    failwith "dft_expand_twidsq: DIF direction not yet implemented"


(* === SPILL-AWARE EXPANSION ===
 *
 * For codelets large enough to overflow the vector register budget
 * (R=32 with ~38 live values vs 32 ZMM), explicit spill management
 * beats GCC's automatic spilling. Hand-coded codelets at this size
 * declare a stack-resident spill_re/spill_im buffer, store all PASS 1
 * outputs to it, then reload at the start of PASS 2. The result is
 * organized memory traffic with predictable stride, vs GCC's scattered
 * spill choices.
 *
 * To support spill emission, the math layer needs to identify which
 * Expr.expr trees correspond to PASS 1 outputs of the outermost CT
 * decomposition. We do this by manually expanding the outermost CT
 * step instead of recursing through `dft`, capturing pass1_re/pass1_im
 * before they get fed into INTERNAL TWIDDLES + PASS 2. Inner sub-DFTs
 * (recursive calls below the outermost level) still use plain `dft`. *)

(* Per-output-bin spill marker. For pass1 output at (n1_idx, k2):
 *   slot     = n1_idx * N2 + k2
 *   re_expr  = the Expr.expr to materialize at spill_re[slot]
 *   im_expr  = the Expr.expr to materialize at spill_im[slot] *)
type spill_marker = {
  slot: int;
  re_expr: expr;
  im_expr: expr;
}

(* Spill-aware version of dft_expand_twiddled.
 *
 * Returns:
 *   - assignments: same shape as dft_expand_twiddled (output stores)
 *   - markers:     spill markers for PASS 1 outputs, empty if n has
 *                  no CT decomposition (Direct DFT case)
 *)
(* IL2 (section 72): TWO independent column instances concatenated at
 * the math layer. Instance 1 shifts Input/Output slots by +n and
 * Twiddle slots by +(n-1). The SU scheduler braids the two
 * independent DAGs by readiness — generator-level column
 * interleaving, the probe for the latency-bound diagnosis. *)
let dft_expand_twiddled_il2 ?(policy = TP_Flat) ?(direction = DIT)
    ?(sign = `Fwd) (n : int) : Expr.assignment list =
  let base = dft_expand_twiddled ~policy ~direction ~sign n in
  let shift_ref = function
    | Expr.Input (k, p)   -> Expr.Input (k + n, p)
    | Expr.Output (k, p)  -> Expr.Output (k + n, p)
    | Expr.Twiddle (t, p) -> Expr.Twiddle (t + (n - 1), p)
  in
  let rec shift = function
    | Const c    -> Const c
    | Load r     -> Load (shift_ref r)
    | Neg e      -> Neg (shift e)
    | Add (a, b) -> Add (shift a, shift b)
    | Sub (a, b) -> Sub (shift a, shift b)
    | Mul (a, b) -> Mul (shift a, shift b)
  in
  let inst2 = List.map (fun (lhs, e) -> (shift_ref lhs, shift e)) base in
  base @ inst2

let dft_expand_twiddled_spill ?(policy = TP_Flat) ?(direction = DIT) ?(sign = `Fwd) (n : int)
    : Expr.assignment list * spill_marker list * (int * int) option =
  let conj = (sign = `Bwd) in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in
  match pick_algorithm n with
  | Direct ->
    (* No CT structure → no spill boundary. Fall back to plain expansion. *)
    (dft_expand_twiddled ~policy ~direction ~sign n, [], None)
  | Split_radix ->
    (* Split-radix doesn't have the same PASS 1 / PASS 2 cluster boundary
     * that the recipe machinery is calibrated against. SR's structure is
     * three recursive sub-DFTs (E of size N/2, O1 and O3 of size N/4) whose
     * outputs combine in a different topology than CT's symmetric two-pass.
     *
     * For this first PR we fall back to plain (non-recipe) expansion: the
     * codelet still generates correctly, just without spill markers. A
     * follow-up PR will design a recipe topology for SR that gives R=32/64
     * comparable spill management to what they get under CT today.
     *
     * This is the same fallback path Direct takes — both algorithms simply
     * lack a clean cluster boundary for the current recipe shape. *)
    (dft_expand_twiddled ~policy ~direction ~sign n, [], None)
  | Cooley_Tukey (n1, n2) ->
    (* Pre vs post twiddle, decided by (direction, sign) — see comment in
     * dft_expand_twiddled. Forward DIT and backward DIF use PRE-twiddle;
     * forward DIF and backward DIT use POST-twiddle. *)
    let pre_twiddle =
      match direction, sign with
      | DIT, `Fwd -> true
      | DIT, `Bwd -> false
      | DIF, `Fwd -> false
      | DIF, `Bwd -> true
    in
    let input_re, input_im =
      if pre_twiddle then begin
        let twiddled_re = Array.make n (Const 0.0) in
        let twiddled_im = Array.make n (Const 0.0) in
        twiddled_re.(0) <- Load (Input (0, true));
        twiddled_im.(0) <- Load (Input (0, false));
        for k = 1 to n - 1 do
          let xr = Load (Input (k, true)) in
          let xi = Load (Input (k, false)) in
          let (wr, wi) = twiddle_expr policy n k in
          let (out_re, out_im) = cmul_pattern ~conj xr xi wr wi in
          twiddled_re.(k) <- out_re;
          twiddled_im.(k) <- out_im
        done;
        ((fun k -> twiddled_re.(k)), (fun k -> twiddled_im.(k)))
      end else
        ((fun k -> Load (Input (k, true))),
         (fun k -> Load (Input (k, false))))
    in

    (* MANUALLY drive the outermost CT step (instead of `dft n input_re input_im`)
     * so we can capture pass1_re/pass1_im as spill markers. The implementation
     * below is a copy of dft_ct's body — kept in sync with dft_ct. *)
    let pi = 4.0 *. atan 1.0 in

    (* PASS 1: N1 sub-FFTs of size N2 — same as dft_ct *)
    let pass1_re = Array.make_matrix n1 n2 (Const 0.0) in
    let pass1_im = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      let inner_input_re k2 = input_re (n1_idx + k2 * n1) in
      let inner_input_im k2 = input_im (n1_idx + k2 * n1) in
      let r, i = dft ~sign n2 inner_input_re inner_input_im in
      for k2 = 0 to n2 - 1 do
        pass1_re.(n1_idx).(k2) <- r.(k2);
        pass1_im.(n1_idx).(k2) <- i.(k2)
      done
    done;

    (* CAPTURE SPILL MARKERS: one per (n1_idx, k2) PASS 1 output bin. *)
    let markers = ref [] in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let slot = n1_idx * n2 + k2 in
        markers := { slot;
                     re_expr = pass1_re.(n1_idx).(k2);
                     im_expr = pass1_im.(n1_idx).(k2);
                   } :: !markers
      done
    done;

    (* INTERNAL TWIDDLES — same as dft_ct, sign-flipped for Bwd *)
    let twiddled_re_inner = Array.make_matrix n1 n2 (Const 0.0) in
    let twiddled_im_inner = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let theta = sgn *. 2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
        let cr = cos theta in
        let ci = sin theta in
        let (tr, ti) = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
        twiddled_re_inner.(n1_idx).(k2) <- tr;
        twiddled_im_inner.(n1_idx).(k2) <- ti
      done
    done;

    (* PASS 2 — same as dft_ct *)
    let raw_re = Array.make n (Const 0.0) in
    let raw_im = Array.make n (Const 0.0) in
    for k2 = 0 to n2 - 1 do
      let outer_input_re n1_idx = twiddled_re_inner.(n1_idx).(k2) in
      let outer_input_im n1_idx = twiddled_im_inner.(n1_idx).(k2) in
      let r, i = dft ~sign n1 outer_input_re outer_input_im in
      for k1 = 0 to n1 - 1 do
        raw_re.(k1 * n2 + k2) <- r.(k1);
        raw_im.(k1 * n2 + k2) <- i.(k1)
      done
    done;

    (* Post-twiddle (if POST structure) — applied to legs 1..n-1. *)
    let out_re = Array.make n (Const 0.0) in
    let out_im = Array.make n (Const 0.0) in
    if pre_twiddle then begin
      Array.blit raw_re 0 out_re 0 n;
      Array.blit raw_im 0 out_im 0 n
    end else begin
      out_re.(0) <- raw_re.(0);
      out_im.(0) <- raw_im.(0);
      for k = 1 to n - 1 do
        let (wr, wi) = twiddle_expr policy n k in
        let (or_, oi) = cmul_pattern ~conj raw_re.(k) raw_im.(k) wr wi in
        out_re.(k) <- or_;
        out_im.(k) <- oi
      done
    end;

    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true),  out_re.(k))  :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    (List.rev !acc, List.rev !markers, Some (n1, n2))

(* No-twiddle (n1) variant with spill markers between PASS 1 and PASS 2.
 *
 * Doc 58 motivation: the monolithic n1 path (dft_expand on line ~684)
 * calls `dft ~sign n` which recursively expands to dft_ct, but builds
 * the result as one big DAG with no pass boundary visible to the
 * scheduler. At R=32+ on AVX-512 this produces peak_live 260-994
 * (8-31× over the 32-zmm budget) and vmovapd counts 2.8-3.4× hand's
 * NFUSE-blocked implementations.
 *
 * The fix is structural: mirror what dft_expand_twiddled_spill does for
 * t1 — manually drive the outermost CT step and emit spill markers
 * between PASS 1 and PASS 2 — but without any outer external-twiddle
 * Cmul layer. The internal twiddles between passes are constants
 * (const_cmul), known at codegen time.
 *
 * After this change, n1 codelets inherit the full SU+spill recipe
 * machinery that t1 already benefits from: bounded per-pass working
 * set, M-project pinning that actually has room to operate, and
 * selective pinning's auto-fusion freedom. Hand-NFUSE-level vmovapd
 * counts become achievable.
 *
 * Returns (assigns, markers, Some (n1, n2)). For sizes that don't
 * benefit from blocking (Direct primes, sizes ≤ threshold where
 * monolithic already fits or wins), falls back to plain dft_expand
 * with empty markers. *)
let dft_expand_n1_blocked ?(sign = `Fwd) (n : int)
    : Expr.assignment list * spill_marker list * (int * int) option =
  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in
  match pick_algorithm n with
  | Direct ->
    (* Primes: no CT structure → no pass boundary → no blocking benefit.
     * Fall back to plain expansion with empty markers. *)
    (dft_expand ~sign n, [], None)
  | Split_radix ->
    (* SR has a different topology than CT's symmetric two-pass; the
     * spill recipe machinery isn't calibrated for it (same caveat as
     * dft_expand_twiddled_spill). Fall back. A follow-up could add an
     * SR-specific blocked path. *)
    (dft_expand ~sign n, [], None)
  | Cooley_Tukey (n1, n2) ->
    let input_re k = Load (Input (k, true)) in
    let input_im k = Load (Input (k, false)) in

    (* PASS 1: N1 sub-FFTs of size N2 — identical to dft_ct.
     * For each n1_idx in [0, N1), compute DFT-N2 on the strided slice
     *   x[n1_idx], x[n1_idx + N1], x[n1_idx + 2·N1], ...
     * Inner DFTs recurse via `dft ~sign n2` — uses CT or Direct as
     * pick_algorithm decides for n2. *)
    let pass1_re = Array.make_matrix n1 n2 (Const 0.0) in
    let pass1_im = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      let inner_input_re k2 = input_re (n1_idx + k2 * n1) in
      let inner_input_im k2 = input_im (n1_idx + k2 * n1) in
      let r, i = dft ~sign n2 inner_input_re inner_input_im in
      for k2 = 0 to n2 - 1 do
        pass1_re.(n1_idx).(k2) <- r.(k2);
        pass1_im.(n1_idx).(k2) <- i.(k2)
      done
    done;

    (* CAPTURE SPILL MARKERS: one per (n1_idx, k2) PASS 1 output bin.
     * Slot indexing matches dft_expand_twiddled_spill so emit_c's spill
     * info construction sees an identical-shape marker list. *)
    let markers = ref [] in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let slot = n1_idx * n2 + k2 in
        markers := { slot;
                     re_expr = pass1_re.(n1_idx).(k2);
                     im_expr = pass1_im.(n1_idx).(k2);
                   } :: !markers
      done
    done;

    (* INTERNAL TWIDDLES: multiply pass1[n1_idx][k2] by Ï_N^{n1_idxÂ·k2}.
     * Identical to dft_ct's twiddle stage (these are codegen-time
     * constants, not runtime loads). For (n1_idx=0 â¨ k2=0) the twiddle
     * is 1 and const_cmul folds away. *)
    let twiddled_re_inner = Array.make_matrix n1 n2 (Const 0.0) in
    let twiddled_im_inner = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let theta = sgn *. 2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
        let cr = cos theta in
        let ci = sin theta in
        let (tr, ti) = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
        twiddled_re_inner.(n1_idx).(k2) <- tr;
        twiddled_im_inner.(n1_idx).(k2) <- ti
      done
    done;

    (* PASS 2: N2 sub-FFTs of size N1 — identical to dft_ct. *)
    let out_re = Array.make n (Const 0.0) in
    let out_im = Array.make n (Const 0.0) in
    for k2 = 0 to n2 - 1 do
      let outer_input_re n1_idx = twiddled_re_inner.(n1_idx).(k2) in
      let outer_input_im n1_idx = twiddled_im_inner.(n1_idx).(k2) in
      let r, i = dft ~sign n1 outer_input_re outer_input_im in
      for k1 = 0 to n1 - 1 do
        out_re.(k1 * n2 + k2) <- r.(k1);
        out_im.(k1 * n2 + k2) <- i.(k1)
      done
    done;

    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true),  out_re.(k))  :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    (List.rev !acc, List.rev !markers, Some (n1, n2))

(* Doc-58-style blocked expansion for the NEWSPLIT construction (scaled
 * conjugate-pair split radix). Same marker contract as
 * dft_expand_n1_blocked, but the seam is split-radix's natural E/O1/O3
 * boundary instead of CT's symmetric two-pass. The fake ct = (4, n/4)
 * exists purely to drive the emitter's cluster arithmetic — see
 * Split_radix.dft_newsplit_blocked for the slot layout. *)
let dft_expand_newsplit_blocked ?(sign = `Fwd) (n : int)
    : Expr.assignment list * spill_marker list * (int * int) option =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let (out_re, out_im, raw_markers) =
    Split_radix.dft_newsplit_blocked ~sign n input_re input_im in
  let markers =
    List.map (fun (slot, re_expr, im_expr) -> { slot; re_expr; im_expr })
      raw_markers in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  (List.rev !acc, markers, Some (4, n / 4))

(* Cost-model rule: should this codelet use the full spill+SU recipe?
 *
 * The rule is empirically derived from benchmarks across (R, ISA, K)
 * combinations. See docs/12_avx2_finding.md for the supporting data.
 *
 * Two clauses:
 *   1. n + 6 > vec_regs:
 *      Peak live exceeds register budget. GCC will re-spill aggressively
 *      without our help. Spill+SU is necessary.
 *
 *   2. vec_regs >= 32 (AVX-512):
 *      Even when peak live fits, AVX-512's wider vectors mean spill
 *      overhead is amortized over more useful work, AND GCC's scheduling
 *      on our DAG produces sub-optimal code that explicit spill+SU fixes.
 *      The recipe wins or ties at every R≥4 we've measured on AVX-512.
 *
 * Empirical bench summary (median SU+Spill / Topo, lower = better):
 *
 *   AVX-512 (vec_regs=32)
 *     R=4:  ~tied (noise)
 *     R=8:  0.90-0.98  ← clause (2) catches this
 *     R=16: substantial wins
 *     R=32: 0.61-0.83
 *     R=64: 0.60-0.92
 *
 *   AVX2 (vec_regs=16)
 *     R=8:  1.00-1.08  ← regression — clause (1) excludes this
 *     R=16: 0.80-0.88  ← clause (1) catches this (22 > 16)
 *     R=32: 0.56-0.81  ← clause (1) catches this (38 > 16)
 *
 *   AVX2 R=5/R=7 update (post-doc-28 fma_lift fix):
 *     The R=8 1.00-1.08 regression noted above was measured WITH fma_lift
 *     unconditional — that's the same regression class doc 28 traced for
 *     composites (extra register pressure from explicit NK_Fma). With
 *     fma_lift gated to primes only, the recipe path is healthy on AVX2 for
 *     small primes too. R=5 and R=7 with the recipe win 5-18% on every
 *     variant; without it they lose 4-37%. So the threshold extends to n≥5
 *     unconditionally. (Clause (3) below.)
 *
 * The unified rule: use the recipe iff CT-decomposed AND any clause holds. *)
let should_spill (n : int) (vec_regs : int) : bool =
  (n + 6 > vec_regs) || vec_regs >= 32 || n >= 5

(* Compatibility: callers may want just clause (1) for register-pressure
 * predictions independent of ISA-specific GCC behavior. *)
let exceeds_register_budget (n : int) (vec_regs : int) : bool =
  n + 6 > vec_regs

(* Doc 58: should the n1 (no-twiddle) codelet use blocked construction
 * (with spill markers between PASS 1 and PASS 2) instead of monolithic?
 *
 * Empirical map of monolithic n1 peak_live vs register budget:
 *   AVX-512 (32 zmm):
 *     R=16   peak_live ~40   mild overflow, monolithic wins (68 vmovapd
 *                            vs hand-blocked 85) — keep monolithic
 *     R=32   peak_live 260   8× over budget, vmovapd 403 (2.8× hand)
 *     R=64   peak_live 267   8× over budget, vmovapd 1363 (3.4× hand)
 *     R=128  peak_live 498   15× over budget, vmovapd explosion
 *     R=256  peak_live 994   31× over budget
 *   AVX2 (16 ymm):
 *     R=8    peak_live ~20   mild overflow, monolithic still wins
 *     R=16   peak_live ~40   borderline, blocking starts to help
 *
 * Crossover: monolithic wins while peak_live stays within ~1.5× budget;
 * above that, the spill cascade dominates and blocked wins. The
 * threshold maps to n ≥ 32 on AVX-512 and n ≥ 32 on AVX2 (where the
 * smaller budget pushes the crossover earlier in absolute n but later
 * relative to budget because the per-codelet sizes are smaller).
 *
 * Empirical threshold: n ≥ 25. The original threshold was n ≥ 32, on
 * the assumption that smaller sizes "already beat hand because the whole
 * DFT fits in registers". Verified for R≤16 (R=16 ties FFTW at 144 ops
 * with 0 muls; whole codelet fits in 32 ZMM registers). But R=25 has
 * 384 ops and a natural 5×5 CT split — its inter-pass live set (25
 * complex values) does NOT fit, so the monolithic emit thrashes
 * registers and produces 1128 vector instructions (450 reg-to-reg
 * moves, 434 stack spills) for an 8.50 ns/transform AVX-512 result.
 * The blocked emit gives 67.98 ns/call (4.7 ns/transform) at AVX-512
 * — a 47% speedup at AVX-512, 39% at AVX2, beating Tugbars's hand-
 * generated 5×5 codelet by 12%.
 *
 * Verified: R=25 AVX-512 default (monolithic) vs blocked numerically
 * identical to 8.88e-16 (machine epsilon); both match FFTW within
 * 1e-13. See /tmp/bench_r25/bench_4way.c.
 *
 * Threshold remains a lower bound. Sizes < 25 (which means R≤16 for
 * CT-decomposed radices given our pick_algorithm) keep monolithic. *)
let should_block_n1 (n : int) (vec_regs : int) : bool =
  (* ISA-aware threshold (section 34). Doc-58's n>=25 was calibrated
   * in the avx512 era; on 16-register targets the R=16 monolithic
   * peak is 35 live values, and blocking it measured: vec movs 83 to
   * 62 with ~zero overflow, arith unchanged, bit-exact, +16.5 percent
   * runtime (+33.6 stacked with the regalloc widening). Default is
   * now 16 when vec_regs <= 16, 25 otherwise; VFFT_N1_BLOCK_MIN
   * remains as an experiment/back-out override in both directions. *)
  let default_min = if vec_regs <= 16 then 16 else 25 in
  let block_min =
    match Sys.getenv_opt "VFFT_N1_BLOCK_MIN" with
    | Some s -> (try int_of_string s with _ -> default_min)
    | None -> default_min in
  match pick_algorithm n with
  | Direct -> false              (* primes: no CT structure to block *)
  | Split_radix -> false          (* SR topology not calibrated for recipe *)
  | Cooley_Tukey _ -> n >= block_min
