(* cx_math.ml — math-layer DAG builders of the full-IL (cil) family.
 * Split out of codelet_cil.ml (Phase 0, 2026-08-09, byte-identity gated).
 * MODULE CARD
 * ROLE: butterfly_pair (the ONE shared twiddle-class selector) + dft_cx
 * (pow2 DIT) + dft_cx_odd (conjugate-pair) + dft_small (dispatcher) +
 * dft_chain (mixed-radix four-step) + cscale_chain.
 * DEPS: Cx_ir only. Pure DAG construction — no scheduling, no emission. *)

open Cx_ir

(* ═══════════════════════════════════════════════════════════════
 *  MATH LAYER — DIT-2 recursion over packed complex
 *
 * Forward sign e^{-2πik/n}, natural order in and out. The twiddle class is
 * chosen per k so the common rotations never cost a general complex
 * multiply (this is the arithmetic the hand emitter established and the
 * race oracle verified):
 *     k=0        -> plain butterfly
 *     4k=n       -> ×(-i)          : CRotNI (shuffle+xor, no multiply)
 *     8k=n       -> (1-i)/√2       : fold √½ into the butterfly via FMA
 *     8k=3n      -> -(1+i)/√2      : same fold, mirrored
 *     otherwise  -> general constant twiddle (BYTW2 with VLIT constants)
 * ═══════════════════════════════════════════════════════════════ *)

let sqh = 0.70710678118654752440

(* ~sign: `Fwd = e^{-2πik/n} (the analysis transform), `Bwd = e^{+2πik/n}
   (the UNNORMALIZED inverse — no 1/N, matching the rest of the library:
   bwd(fwd(x)) = N·x). Every twiddle class flips with the sign:
     w_k = e^{sgn·2πik/n}
     4k=n  -> w = sgn·i        : CRotNI (fwd) / CRotPI (bwd)
     8k=n  -> w = (1 + sgn·i)/√2 : x = o + rot(o), then ±√½·x + e
     8k=3n -> w = (-1 + sgn·i)/√2: x = rot(o) - o, same fold
     else  -> general constant twiddle, s = sgn·sin
   so the ONLY structural difference is which quarter-turn node is used;
   the butterfly shape and op counts are identical in both directions. *)
(* ONE radix-2 butterfly of an n-point DIT stage: combine the k-th outputs of
   the even half (ek) and odd half (ok) into outputs k and k+n/2. The twiddle
   CLASS selection lives here so the monolithic recursion and the BLOCKED
   construction below share exactly one copy of it — they must agree, or the
   two forms would not be numerically interchangeable. *)
let butterfly_pair ~(sign : [ `Fwd | `Bwd ]) ~(n : int) ~(k : int) (ek : t) (ok : t)
  : t * t
  =
  let pi = 4.0 *. atan 1.0 in
  let rot x = if sign = `Fwd then crot x else crotp x in
  let sgn = if sign = `Fwd then -1.0 else 1.0 in
  if k = 0
  then cadd ek ok, csub ek ok
  else if 4 * k = n
  then (
    let t = rot ok in
    cadd ek t, csub ek t)
  else if 8 * k = n
  then (
    (* w = (1 + sgn·i)/√2 : x = o + rot(o), then ±√½·x + e *)
    let x = cadd ok (rot ok) in
    cfma sqh x ek, cfnma sqh x ek)
  else if 8 * k = 3 * n
  then (
    (* w = (-1 + sgn·i)/√2 = √½·(rot(o) - o) *)
    let x = csub (rot ok) ok in
    cfma sqh x ek, cfnma sqh x ek)
  else (
    let c = cos (2.0 *. pi *. float_of_int k /. float_of_int n)
    and s = sgn *. sin (2.0 *. pi *. float_of_int k /. float_of_int n) in
    let t = ctw c s ok in
    cadd ek t, csub ek t)
;;

(* Unwrap a fully-assigned output array. The Option is not decoration: the
   ORIGINAL `Array.make n xs.(0)` pre-filled every slot with INPUT LEG 0, so a
   construction that failed to write a slot emitted plausible-looking code that
   silently returned an input as an output (the r3 signature). With Option the
   same mistake is a generation-time failure instead. Pow2 output is unchanged —
   the unwrapped values are the same nodes. *)
let unwrap_legs (who : string) (out : t option array) : t array =
  Array.mapi
    (fun i o ->
       match o with
       | Some e -> e
       | None -> failwith (Printf.sprintf "%s: output leg %d never assigned" who i))
    out
;;

let rec dft_cx ?(sign = `Fwd) (n : int) (xs : t array) : t array =
  if n = 1
  then xs
  else (
    let h = n / 2 in
    let e = dft_cx ~sign h (Array.init h (fun i -> xs.(2 * i)))
    and o = dft_cx ~sign h (Array.init h (fun i -> xs.((2 * i) + 1))) in
    let out = Array.make n None in
    for k = 0 to h - 1 do
      let a, b = butterfly_pair ~sign ~n ~k e.(k) o.(k) in
      out.(k) <- Some a;
      out.(k + h) <- Some b
    done;
    unwrap_legs "dft_cx" out)
;;

(* ═══════════════════════════════════════════════════════════════
 *  ODD / PRIME RADICES — the CONJUGATE-PAIR construction
 *
 * dft_cx above is a DIT radix-2 recursion and is valid ONLY for powers of two.
 * Odd radices use the conjugate-pair symmetry instead. With H = (n-1)/2:
 *
 *     S_j = x_j + x_{n-j}          D_j = x_j - x_{n-j}        j = 1..H
 *     X[0]   = x0 + SUM_j S_j
 *     P_m    = x0 + SUM_j cos(2pi*j*m/n) * S_j       (REAL scalar weights)
 *     Q_m    =      SUM_j sin(2pi*j*m/n) * D_j       (REAL scalar weights)
 *     X[m]   = P_m + sigma*i*Q_m,   X[n-m] = P_m - sigma*i*Q_m
 *                                    sigma = -1 (Fwd), +1 (Bwd)
 *
 * This is the SAME construction the real/split side hand-builds in
 * dft_recurse.ml (:327-468) — it is NOT discovered by algsimp there (which
 * this module never runs anyway), so quality here is delivered BY
 * CONSTRUCTION, exactly as it is on the split side.
 *
 * THREE MOVES make it fit the existing packed-complex primitives, with NO new
 * node kind:
 *
 *  1. PRE-ROTATE the differences. sigma*i is linear over real scalars, so push
 *     it inside the sum: R_j = rot(D_j) once per j (H rotations for the whole
 *     codelet, hash-consed and shared by every output), instead of one rotation
 *     per output. A Fwd kernel then contains only CRotNI and a Bwd kernel only
 *     CRotPI, which keeps the one-mask-per-direction preamble correct.
 *
 *  2. MAX-MAGNITUDE NORMALIZATION. Seeding the Q chain on the term with the
 *     LARGEST |sin| lets the leading coefficient be absorbed into the output
 *     FMA, so no bare "real scalar times x" node is needed (there is none in
 *     cx_kind). It is also better conditioned: every remaining ratio is <= 1 in
 *     magnitude, whereas seeding at j=1 would divide by the smallest sine.
 *
 *  3. DEGENERATE-COEFFICIENT LADDER (cscale_chain): weights of 0 / +1 / -1 —
 *     which occur for odd COMPOSITES, e.g. n=9 at j=3,m=3 gives cos=1, sin=0 —
 *     collapse to nothing / cadd / csub instead of emitting a multiply.
 *
 * P_m and Q_m are BIT-IDENTICAL between Fwd and Bwd: every weight is the same
 * and no sign flips anywhere. The only difference between the two kernels is
 * which quarter-turn node is used, so BOTH DIRECTIONS come out at the same op
 * count and the same DAG shape.
 * ═══════════════════════════════════════════════════════════════ *)

let cx_eps = 1e-14

(* Weighted sum over packed complex with REAL scalar weights, folded into an
   FMA chain. The SIGN rides in the OPCODE (cfma vs cfnma) and only the
   magnitude becomes a constant — the same convention dft_recurse.ml uses, and
   the reason no fma-lift pass is needed afterwards. *)
let cscale_chain ~(seed : t) (terms : (float * t) list) : t =
  List.fold_left
    (fun acc (c, x) ->
       if abs_float c < cx_eps
       then acc
       else if abs_float (c -. 1.0) < cx_eps
       then cadd acc x
       else if abs_float (c +. 1.0) < cx_eps
       then csub acc x
       else if c > 0.0
       then cfma c x acc
       else cfnma (-.c) x acc)
    seed
    terms
;;

let dft_cx_odd ?(sign = `Fwd) (n : int) (xs : t array) : t array =
  if n < 3 || n mod 2 = 0
  then failwith (Printf.sprintf "dft_cx_odd: needs an ODD n >= 3, got %d" n);
  let pi = 4.0 *. atan 1.0 in
  let rot x = if sign = `Fwd then crot x else crotp x in
  let h = (n - 1) / 2 in
  (* conjugate pairs, shared by every output leg *)
  let s = Array.init h (fun i -> cadd xs.(i + 1) xs.(n - i - 1)) in
  let r = Array.init h (fun i -> rot (csub xs.(i + 1) xs.(n - i - 1))) in
  let out = Array.make n None in
  let acc = ref xs.(0) in
  for i = 0 to h - 1 do
    acc := cadd !acc s.(i)
  done;
  out.(0) <- Some !acc;
  for m = 1 to h do
    let cf j = cos (2.0 *. pi *. float_of_int (j * m) /. float_of_int n) in
    let sf j = sin (2.0 *. pi *. float_of_int (j * m) /. float_of_int n) in
    let p = cscale_chain ~seed:xs.(0) (List.init h (fun i -> cf (i + 1), s.(i))) in
    (* seed the Q chain on the largest |sin| (move 2) *)
    let jstar = ref 0 in
    for i = 1 to h - 1 do
      if abs_float (sf (i + 1)) > abs_float (sf (!jstar + 1)) then jstar := i
    done;
    let cstar = sf (!jstar + 1) in
    (* |sin(2pi*m/n)| > 0 for 1 <= m <= h, and cstar is the max, so this cannot
       fire; keep it as a loud guard rather than dividing by zero silently. *)
    if abs_float cstar < cx_eps
    then failwith (Printf.sprintf "dft_cx_odd: degenerate sine row at n=%d m=%d" n m);
    let qh =
      cscale_chain
        ~seed:r.(!jstar)
        (List.filteri
           (fun i _ -> i <> !jstar)
           (List.init h (fun i -> sf (i + 1) /. cstar, r.(i))))
    in
    let a, b =
      if cstar > 0.0
      then cfma cstar qh p, cfnma cstar qh p
      else cfnma (-.cstar) qh p, cfma (-.cstar) qh p
    in
    out.(m) <- Some a;
    out.(n - m) <- Some b
  done;
  unwrap_legs "dft_cx_odd" out
;;

(* Leaf dispatcher: pow2 -> radix-2 DIT, odd -> conjugate pair, EVEN
   COMPOSITE (6, 10, 12, ...) -> radix-2 DIT splits whose halves
   RE-DISPATCH here, so the odd part bottoms out in the conjugate-pair
   builder instead of dropping legs (dft_cx recursing into an odd half was
   the hazard the old refusal guarded). butterfly_pair is the ONE shared
   copy, general-twiddle arm included, so the mixed recursion is the same
   numeric family — pow2 and odd radices take their old paths untouched
   (byte-identity preserved). Unlocks the 2-stage IL pairs at 4·odd² N
   (100 = 10x10, 36 = 6x6) and even-composite chain mids (200 = 4·(5·10)),
   per Tugbars 2026-07-29. *)
let rec dft_small ?(sign = `Fwd) (n : int) (xs : t array) : t array =
  if n = 1
  then xs
  else if n land (n - 1) = 0
  then dft_cx ~sign n xs
  else if n mod 2 = 1
  then dft_cx_odd ~sign n xs
  else (
    let h = n / 2 in
    let e = dft_small ~sign h (Array.init h (fun i -> xs.(2 * i)))
    and o = dft_small ~sign h (Array.init h (fun i -> xs.((2 * i) + 1))) in
    let out = Array.make n None in
    for k = 0 to h - 1 do
      let a, b = butterfly_pair ~sign ~n ~k e.(k) o.(k) in
      out.(k) <- Some a;
      out.(k + h) <- Some b
    done;
    unwrap_legs "dft_small" out)
;;

(* Mixed-radix four-step over the complex IR: `chain` says how the transform is
   FACTORED, and dft_cx (pure radix-2) is only the leaf.

   Why this exists. dft_cx hardwires 2.2.2.2.2.2 for a 64-point stage — a plan
   decision baked into an emitter, one level below the one already removed from
   emit_k1. The IL chain race measured what it costs: with the stage COUNT
   pinned at two, stage radices grow as sqrt(N), so by N=1024 both stages are
   fully-unrolled DFT_32/DFT_64 that spill, and no choice of two-stage split
   recovers more than 5%. Letting the chain reach inside a stage hands that
   factorization back to the planner.

   Index algebra (standard four-step). With j = j1*n2 + j2 and k = k2*r0 + k1,
   the cross term j1*n2*k2*r0 is a multiple of N and drops, leaving
     X[k2*r0 + k1] = DFT_n2 over j2 of ( DFT_r0(column j2)[k1] * w_N^{j2*k1} )
   so the output lands transposed at k2*r0 + k1, which is why the caller's
   store index is (k2 * n1 + k1) and no output permutation is ever needed. *)
let rec dft_chain ~(sign : [ `Fwd | `Bwd ]) ~(chain : int list) (xs : t array)
  : t array
  =
  let n = Array.length xs in
  let prod = List.fold_left ( * ) 1 chain in
  if prod <> n
  then
    failwith
      (Printf.sprintf
         "codelet_cil: dft_chain got %d inputs but the chain multiplies to %d"
         n
         prod);
  match chain with
  | [] | [ _ ] -> dft_cx ~sign n xs
  | r0 :: rest ->
    let n2 = n / r0 in
    let sgn = if sign = `Fwd then -1.0 else 1.0 in
    let pi = 4.0 *. atan 1.0 in
    let cols =
      Array.init n2 (fun j2 ->
        dft_cx ~sign r0 (Array.init r0 (fun j1 -> xs.((j1 * n2) + j2))))
    in
    let out = Array.make n xs.(0) in
    for k1 = 0 to r0 - 1 do
      let row =
        Array.init n2 (fun j2 ->
          let v = cols.(j2).(k1) in
          if k1 = 0 || j2 = 0
          then v
          else (
            let a = sgn *. 2.0 *. pi *. float_of_int (k1 * j2) /. float_of_int n in
            ctw (cos a) (sin a) v))
      in
      let r = dft_chain ~sign ~chain:rest row in
      for k2 = 0 to n2 - 1 do
        out.((k2 * r0) + k1) <- r.(k2)
      done
    done;
    out
;;
