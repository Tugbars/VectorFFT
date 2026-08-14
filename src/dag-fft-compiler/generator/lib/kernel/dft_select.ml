(* dft_select.ml — c2c DFT algorithm selection.
 *
 * Bottom layer of the Dft chain (Dft_select < Dft_recurse < Dft; the
 * facade re-exports everything, so call sites say Dft_select.pick_algorithm).
 *
 * pick_algorithm dispatches by N's number-theoretic properties:
 *   - Direct          n = 2, or odd primes (uses conjugate-pair construction)
 *   - Cooley_Tukey    composites with hardcoded (n1, n2) factorizations
 *                     matching production codelet shapes (R=16 → 4×4,
 *                     R=32 → 4×8, R=64 → 8×8, R=128 → 8×16, etc.)
 *   - Split_radix     pow2 ≥ 8, opt-in via VFFT_SPLIT_RADIX env var
 *                     (delegated to split_radix.ml)
 *
 * Also here: the decisions that depend only on N and the register
 * budget — needs_reassoc (whether hash-cons lifting may reassociate),
 * factor_override (VFFT_CT_FACTOR escape hatch), target_vec_regs (the
 * ISA register count the picker plans against; gen_main sets it from
 * the selected ISA before any expansion runs).
 * ------------------------------------------------------------------
 * MODULE CARD (dft_select.ml — grep "MODULE CARD" for the full set)
 * ROLE: algorithm type + pick_algorithm + needs_reassoc +
 * factor_override + the split-radix / newsplit enable predicates.
 * PIPELINE: consulted by Dft_recurse at every recursion level and by
 * drivers (gen_main, pipeline, codelet_oop) for policy decisions.
 * PUBLIC SURFACE (measured): zero direct Dft_select.X references —
 * all callers go through the Dft facade.
 * DEPS: none beyond stdlib (pure N-and-env logic).
 * STATE: target_vec_regs is a process-global ref, default 32; set it
 * before expansion or pow2 factorizations assume AVX-512 budgets.
 * ENV: VFFT_SPLIT_RADIX, VFFT_NEWSPLIT, VFFT_CT_FACTOR.
 * ------------------------------------------------------------------
 *)

(* === ALGORITHM SELECTION === *)

type algorithm =
  | Direct (* prime n, or n=2: use dft_kernel *)
  | Cooley_Tukey of int * int (* (n1, n2): split DFT-n into n1 columns of n2 *)
  | Split_radix (* pow2 n ≥ 8: SR decomposition (N/2 + N/4 + N/4) *)

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
;;

let split_radix_enabled () : bool =
  newsplit_enabled ()
  ||
  match Sys.getenv_opt "VFFT_SPLIT_RADIX" with
  | None | Some "" -> false
  | Some _ -> true
;;

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
       if nn <> n
       then None
       else (
         let rest =
           String.trim (String.sub s (colon + 1) (String.length s - colon - 1))
         in
         if rest = "direct"
         then Some Direct
         else if rest = "sr"
         then Some Split_radix
         else (
           let comma = String.index rest ',' in
           let a = int_of_string (String.trim (String.sub rest 0 comma)) in
           let b =
             int_of_string
               (String.trim
                  (String.sub rest (comma + 1) (String.length rest - comma - 1)))
           in
           Some (Cooley_Tukey (a, b))))
     with
     | _ -> None)
;;

let pick_algorithm (n : int) : algorithm =
  match factor_override n with
  | Some a -> a
  | None ->
    if n <= 2
    then Direct
    else if split_radix_enabled () && n >= 8 && n land (n - 1) = 0
    then
      (* Route pow2 sizes ≥ 8 through Split_radix when VFFT_SPLIT_RADIX is set.
       * N=4 stays as CT(2,2) regardless — SR's recursion bottoms there.
       * N=2 stays as Direct (also an SR base case).
       * Non-pow2 sizes are unaffected: SR requires power-of-two N. *)
      Split_radix
    else (
      match n with
      | 4 -> Cooley_Tukey (2, 2)
      | 8 -> Cooley_Tukey (2, 4)
      | 16 -> Cooley_Tukey (4, 4)
      | 32 -> Cooley_Tukey (4, 8)
      | 64 when !target_vec_regs <= 16 ->
        Cooley_Tukey (4, 16)
        (* AVX2 (16-ymm budget): (4,16) recurses to (4,(4,4)). Deepest
         * pass has 4 complex = 8 ymm live, well under the 16-ymm budget.
         * Empirically saves 42 spills vs (8,8) on AVX2 (−9%) and also
         * −8 fp ops (920 vs 928). On AVX-512 the savings reverse —
         * (8,8) is symmetric and wins on schedule pressure. *)
      | 64 ->
        Cooley_Tukey (8, 8)
        (* Hand-coded R=64 uses CT(8, 8) — symmetric factorization that
         * splits the 64-point DFT into 8 sub-FFT-8s in PASS 1 and 8
         * sub-FFT-8s in PASS 2. Same convention as gen_radix64.py. *)
      | 128 ->
        Cooley_Tukey (8, 16)
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
      | 256 ->
        Cooley_Tukey (16, 16)
        (* R=256 — balanced factorization. 16 sub-DFT-16s in each pass.
         * Following the precedent set by R=128 (8×16) and R=64 (8×8),
         * this is the natural "balanced" choice for monolithic generation.
         * Generates a ~7800-vec-instr codelet (vs R=128's 3318). *)
      | 512 ->
        Cooley_Tukey (16, 32)
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
      | 6 ->
        Cooley_Tukey (3, 2)
        (* R=6 = CT(N1=3, N2=2). PASS 1: 2 sub-DFT-3s (prime, conjugate-pair
         * construction). PASS 2: 3 sub-DFT-2s (trivial add/sub). Internal
         * twiddles W6: 2 unique exponents (W6^1, W6^2) require cmul; W6^3=−1
         * is free. Matches gen_radix6.py's (N1=3, N2=2) factorization in our
         * convention. Peak live ~12 ZMM AVX-512 (fits), ~12 YMM AVX2 (also
         * fits but tight). *)
      | 25 ->
        Cooley_Tukey (5, 5)
        (* R=25 = CT(N1=5, N2=5). Symmetric prime-prime split. PASS 1: 5
         * sub-DFT-5s. PASS 2: 5 sub-DFT-5s. All 9 unique internal W25
         * exponents require cmul (no free rotations like W^N/2 = -1 or
         * W^N/4 = ±j; none of those exponents arise in W25's grid).
         * Matches gen_radix25.py's (N1=5, N2=5). Peak live ~50 (raw) so
         * spill is mandatory on both ISAs. Highest retiring rate codelet
         * per profiling — recipe path is essential here. *)
      | 10 ->
        Cooley_Tukey (5, 2)
        (* R=10 = CT(N1=5, N2=2). PASS 1: 2 sub-DFT-5s. PASS 2: 5 sub-DFT-2s
         * (trivial add/sub). Same shape as R=6 (CT with prime N1, trivial
         * N2) but with DFT-5 inner. Internal twiddles W10: only 2 unique
         * non-trivial exponents (W10^1, W10^2 — others are 1, -1, ±j and
         * fold via const_cmul). Matches gen_radix10.py's (N1=5, N2=2). *)
      | 12 ->
        Cooley_Tukey (4, 3)
        (* R=12 = CT(N1=4, N2=3). First non-prime, non-pow2 split — both
         * factors > 2. PASS 1: 3 sub-DFT-4s (radix-4 butterflies, FMA-poor
         * but very compact). PASS 2: 4 sub-DFT-3s (prime conjugate-pair).
         * Free internal twiddles: W12^3 = -j, W12^6 = -1, W12^9 = j; only
         * 4 unique non-trivial cmul exponents (W12^1, W12^2, W12^4, W12^5
         * ... folded by symmetry). Matches gen_radix12.py's (N1=4, N2=3). *)
      | 20 ->
        Cooley_Tukey (5, 4)
        (* R=20 = CT(N1=5, N2=4). PASS 1: 4 sub-DFT-5s. PASS 2: 5 sub-DFT-4s.
         * Mix of prime DFT-5 (8 ops/output via conjugate-pair) and pow2
         * DFT-4 (very compact). 12 unique non-trivial W20 exponents; some
         * fold (W20^5 = -j, W20^10 = -1, W20^15 = j). Matches
         * gen_radix20.py's (N1=5, N2=4). Peak live ~40 — spill threshold,
         * recipe path likely essential at AVX-512 (32 ZMM). *)
      | 15 ->
        Cooley_Tukey (3, 5)
        (* R=15 = CT(N1=3, N2=5). PASS 1: 5 sub-DFT-3s. PASS 2: 3 sub-DFT-5s.
         * Without this entry R=15 falls to Direct DFT-15 (308 ops vs FFTW's
         * 156). With CT(3,5) the costly O(N²) direct path is avoided and
         * Winograd-5 further trims the three DFT-5 instances (181 ops
         * vs FFTW's 156, +16% — bounded by the same n-ary-IR limit that
         * leaves the R=25 gap open). *)
      | 21 ->
        Cooley_Tukey (3, 7)
        (* R=21 = CT(N1=3, N2=7). PASS 1: 7 sub-DFT-3s. PASS 2: 3 sub-DFT-7s.
         * Without this entry R=21 falls to Direct DFT-21 (O(N²)). With
         * CT(3,7), Winograd-7 trims the three DFT-7 instances substantially
         * relative to conjugate-pair Direct. Same cascade pattern as R=15. *)
      | _ when n mod 2 = 0 -> Cooley_Tukey (2, n / 2)
      | _ -> Direct)
;;

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
  | Direct -> n < 3 || n mod 2 = 0 (* only n=2 hits this in practice *)
  | Cooley_Tukey _ -> false
  | Split_radix -> false
;;
(* SR construction is structured like CT — already
 * has butterfly form; reassoc would flatten it. *)
