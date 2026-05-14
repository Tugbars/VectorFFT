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
let split_radix_enabled () : bool =
  match Sys.getenv_opt "VFFT_SPLIT_RADIX" with
  | None | Some "" -> false
  | Some _ -> true

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
let pick_algorithm (n : int) : algorithm =
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
 * Returns (out_re, out_im) as Expr trees. *)
let const_cmul (xr : expr) (xi : expr) (cr : float) (ci : float) : expr * expr =
  let cr_e = Const cr in
  let ci_e = Const ci in
  (* (xr*cr - xi*ci, xr*ci + xi*cr) *)
  let out_re = Sub (Mul (xr, cr_e), Mul (xi, ci_e)) in
  let out_im = Add (Mul (xr, ci_e), Mul (xi, cr_e)) in
  (out_re, out_im)

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
  match pick_algorithm n with
  | Direct ->
    (* For odd N >= 3, the conjugate-pair construction produces a much
     * better-structured DAG than naive direct DFT: pair sums/diffs are
     * shared, per-pair-output intermediates (p_re_m, q_im_m, etc.) are
     * shared between X[m] and X[N-m], and inner sums use linear FMA
     * chains. For N=2 there's nothing to factor; for even N we never
     * reach Direct anyway (CT-decomposed). *)
    if n >= 3 && n mod 2 = 1 then
      dft_direct_conjugate_pair ~sign n input_re input_im
    else
      dft_direct ~sign n input_re input_im
  | Cooley_Tukey (n1, n2) -> dft_ct ~sign n1 n2 input_re input_im
  | Split_radix ->
    (* Split-radix lives in its own module (lib/split_radix.ml). Cross-
     * module mutual recursion uses the callback pattern: we pass `dft`
     * itself in as `dft_rec` so SR can recurse on its sub-DFT inputs
     * (size N/2 and N/4) which dispatch back through the picker. *)
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
  (* DIT: pre-multiply inputs by twiddles, then run DFT.
   * DIF: run DFT, then post-multiply outputs by twiddles.
   *
   * Leg 0 has trivial twiddle W^0 = 1 in both cases.
   *
   * For sign=Bwd, internal CT twiddles use +θ (handled by `dft ~sign`),
   * and external twiddles use conjugate cmul (handled by ~conj here).
   *
   * Twiddle Exprs may be cmul derivation patterns (TP_Log3) or simple
   * Loads (TP_Flat). The per-leg Sub(Mul,Mul)/Add(Mul,Mul) cmul pattern
   * is preserved so Algsimp.of_expr can lift it to Cmul opaque atoms. *)
  let conj = (sign = `Bwd) in
  match direction with
  | DIT ->
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

  | DIF ->
    (* Run DFT on raw inputs, then twiddle the outputs. *)
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
    (* Reverse so order matches DIT: ascending k, re before im for each k. *)
    let sorted = List.sort (fun (a, _) (b, _) ->
      match a, b with
      | Output (ka, ra), Output (kb, rb) ->
        if ka <> kb then compare ka kb
        else compare (not ra) (not rb)
      | _ -> 0
    ) !acc in
    sorted

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
    (* DIT pre-multiplies inputs by twiddles; DIF leaves inputs raw and
     * post-multiplies outputs at the end. The internal CT decomposition
     * (with spill markers between PASS 1 and PASS 2) is identical. *)
    let input_re, input_im = match direction with
      | DIT ->
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
      | DIF ->
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

    (* For DIF, post-multiply outputs by external twiddles. *)
    let out_re = Array.make n (Const 0.0) in
    let out_im = Array.make n (Const 0.0) in
    (match direction with
     | DIT ->
       Array.blit raw_re 0 out_re 0 n;
       Array.blit raw_im 0 out_im 0 n
     | DIF ->
       out_re.(0) <- raw_re.(0);
       out_im.(0) <- raw_im.(0);
       for k = 1 to n - 1 do
         let (wr, wi) = twiddle_expr policy n k in
         let (or_, oi) = cmul_pattern ~conj raw_re.(k) raw_im.(k) wr wi in
         out_re.(k) <- or_;
         out_im.(k) <- oi
       done);

    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true),  out_re.(k))  :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    (List.rev !acc, List.rev !markers, Some (n1, n2))

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
