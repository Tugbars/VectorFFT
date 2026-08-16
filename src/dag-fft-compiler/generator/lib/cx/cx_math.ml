(* cx_math.ml - math-layer DAG builders of the full-IL (cil) family.
 * Split out of codelet_cil.ml (Phase 0, 2026-08-09, byte-identity gated).
 * MODULE CARD
 * ROLE: butterfly_pair (the ONE shared twiddle-class selector) + dft_cx
 * (pow2 DIT) + dft_cx_odd (conjugate-pair) + dft_small (dispatcher) +
 * dft_chain (mixed-radix four-step) + cscale_chain.
 * DEPS: Cx_ir only. Pure DAG construction - no scheduling, no emission. *)

open Cx_ir

(* ═══════════════════════════════════════════════════════════════
 *  MATH LAYER - DIT-2 recursion over packed complex
 *
 * Forward sign e^{-2πik/n}, natural order in and out. The twiddle class is
 * chosen per k so the common rotations never cost a general complex
 * multiply (this is the arithmetic the hand emitter established and the
 * race oracle verified):
 *     k=0        -> plain butterfly
 *     4k=n       -> �-(-i)          : CRotNI (shuffle+xor, no multiply)
 *     8k=n       -> (1-i)/√2       : fold √½ into the butterfly via FMA
 *     8k=3n      -> -(1+i)/√2      : same fold, mirrored
 *     otherwise  -> general constant twiddle (BYTW2 with VLIT constants)
 * ═══════════════════════════════════════════════════════════════ *)

let sqh = 0.70710678118654752440

(* �-��-� TANGENT-INTERIOR VARIANT �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
   docs/roadmap/tangent_emitter_plan.md + docs/performance/
   tangent_scaled_butterflies.md. Default OFF: with the ref false, every
   arm below is byte-identical to the shipped emitter, so the 183-case
   matrix is preserved BY CONSTRUCTION. Enable per process with
   --cil-tangent (gen_main) or VFFT_CX_TANGENT=1.

   What it changes: ONLY the general-twiddle arm of butterfly_pair. The
   classic arm materializes the rotation (ctw) and then does NAKED
   cadd/csub butterflies - adds with no multiply attached, condemned to
   the FADD ports. The tangent arm factors w = cosθ·(1 + sgn·i·tanθ):
   the shear (ok + tanθ·rot ok) runs un-normalized in ONE fma, and the
   cosθ normalization is deferred INTO the butterfly fma pair - the adds
   are promoted to the FMA ports and the standalone rotation multiply
   disappears. The k=0 / ±i / √½ arms are already in this form and are
   shared verbatim by both variants.
   Measured (hand proof kernels, 2026-08-11, paired ±0.3%): R16 mid �'25%,
   R16 leaf �'17%, pure-tangent N=256 �'20…�'24%, N=512 �'14…�'17%.
   ⚠ L1-conditional: at N=1024 the HAND kernels invert (+16%) - the win is
   scoped ≤512-class until blocked-tangent forms are raced. *)
(* M12a: tangent/w32_combine/wing_enabled live in Cx_ir.ctx now. *)

(* ~sign: `Fwd = e^{-2πik/n} (the analysis transform), `Bwd = e^{+2πik/n}
   (the UNNORMALIZED inverse - no 1/N, matching the rest of the library:
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
   construction below share exactly one copy of it - they must agree, or the
   two forms would not be numerically interchangeable. *)
let butterfly_pair
      ~(sign : [ `Fwd | `Bwd ])
      ~(n : int)
      ~(k : int)
      ~(ctx : ctx)
      (ek : t)
      (ok : t)
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
  then
    if ctx.tangent
    then (
      (* tangent form of the same fold: the shear x·(1 + sgn·i) as a
         unit-cosine CTwC (2 uops via the c=1 render peephole - no xor)
         instead of cadd(o, rot o) (3 uops). fma(±1·a + b) rounds once,
         value-identical to the add path. *)
      let x = ctw 1.0 sgn ok in
      cfma sqh x ek, cfnma sqh x ek)
    else (
      (* w = (1 + sgn·i)/√2 : x = o + rot(o), then ±√½·x + e *)
      let x = cadd ok (rot ok) in
      cfma sqh x ek, cfnma sqh x ek)
  else if 8 * k = 3 * n
  then
    if ctx.tangent
    then (
      (* w = (-1 + sgn·i)/√2 = -√½·(o·(1 - sgn·i)): CONJUGATE shear (2
         uops), minus sign absorbed by swapping the butterfly opcodes. *)
      let x = ctw 1.0 (-.sgn) ok in
      cfnma sqh x ek, cfma sqh x ek)
    else (
      (* w = (-1 + sgn·i)/√2 = √½·(rot(o) - o) *)
      let x = csub (rot ok) ok in
      cfma sqh x ek, cfnma sqh x ek)
  else if ctx.tangent
  then (
    (* w = e^{sgn·iθ} = cosθ·(1 + sgn·i·tanθ), θ = 2πk/n.
       shear = ok + tanθ·rot(ok)   - one fma; |shear| = |ok|/|cosθ|, a pure
                                     rotation magnitude, no cancellation
       out   = ek ± cosθ·shear     - normalization fused into the butterfly
       Signs of tan/cos ride in the OPCODE (cfma vs cfnma), magnitudes in
       the constants - the cscale_chain convention. θ ∈ (0,π) here (the
       free/√½ classes are caught above), so t and c go negative past
       π/2; |tan| ≤ tan(7π/16) ≈ 5.03 through R32, gated 3.4e-13 on the
       hand kernels. Loud guard at 8: the first offender is R64's 15π/32
       site - implement quarter-turn composition before lifting it. *)
    let th = 2.0 *. pi *. float_of_int k /. float_of_int n in
    let t = tan th
    and c = cos th in
    if abs_float t > 8.0
    then
      failwith
        (Printf.sprintf
           "butterfly_pair: tangent variant |tan(2pi*%d/%d)| = %.3f > 8 - R>=64 site; \
            needs the quarter-turn composition (tangent_emitter_plan.md)"
           k
           n
           (abs_float t));
    (* shear as a unit-cosine CTwC: x·(1 + i·(sgn·t)) - with the c=1 render
       peephole this is 2 uops (cflip + sign-folded fmadd), no xor, the
       exact hand-kernel idiom. The tan sign rides in the folded constant;
       only cos still splits the butterfly opcodes. *)
    let sh = ctw 1.0 (sgn *. t) ok in
    if c >= 0.0 then cfma c sh ek, cfnma c sh ek else cfnma (-.c) sh ek, cfma (-.c) sh ek)
  else (
    let c = cos (2.0 *. pi *. float_of_int k /. float_of_int n)
    and s = sgn *. sin (2.0 *. pi *. float_of_int k /. float_of_int n) in
    let t = ctw c s ok in
    cadd ek t, csub ek t)
;;

(* ── W32 WING COMBINE (VFFT_CX_W32TG=1, FWD only) ──────────────────────
   Pass-B site of the HAND w32tg/w32tgL kernels, per the A-0 verdict
   (docs/roadmap/r32_tangent_parity_plan.md): the blocked m=2 combine at
   radix 32 with the angle CANONICALIZED to oe = k mod 8, so mirror sites
   (k > 8) share the k-8 constants EXACTLY (kills the ulp-twin table
   entries) and the -i relating them is a composed rotation node that the
   ROTFMA render fold turns into a sign-folded fmadd ([c,-c]·cflip — the
   hand kernel's idiom, zero xors in pass B). oe = 4 uses exact (1, √½).
   Default OFF: without the knob the blocked combine keeps butterfly_pair
   and every existing emission is byte-identical. *)

let butterfly_pair_w32 ~(k : int) (ek : t) (ok : t) : t * t =
  let pi = 4.0 *. atan 1.0 in
  if k = 0
  then cadd ek ok, csub ek ok
  else if k = 8
  then (
    (* W32^8 = -i : rotation composed into the butterfly (ROTFMA folds
       both consumers to fmadd/fnmadd([1,-1]·cflip) — 3 uops, no xor). *)
    let r = crot ok in
    cadd ek r, csub ek r)
  else (
    let oe = k land 7 in
    let tn, c =
      if oe = 4
      then 1.0, sqh
      else (
        let th = pi *. float_of_int oe /. 16.0 in
        tan th, cos th)
    in
    (* fwd shear at the CANONICAL angle: e^{-iθ} = cosθ·(1 - i·tanθ) *)
    let sh = ctw 1.0 (-.tn) ok in
    if k < 8
    then cfma c sh ek, cfnma c sh ek
    else (
      (* W32^k = -i·W32^{k-8}: same shear, rotation composed; ROTFMA
         renders the pair as fmadd/fnmadd([c,-c]·cflip sh) + e. *)
      let r = crot sh in
      cfma c r ek, cfnma c r ek))
;;

(* Unwrap a fully-assigned output array. The Option is not decoration: the
   ORIGINAL `Array.make n xs.(0)` pre-filled every slot with INPUT LEG 0, so a
   construction that failed to write a slot emitted plausible-looking code that
   silently returned an input as an output (the r3 signature). With Option the
   same mistake is a generation-time failure instead. Pow2 output is unchanged -
   the unwrapped values are the same nodes. *)
let unwrap_legs (who : string) (out : t option array) : t array =
  Array.mapi
    (fun i o ->
       match o with
       | Some e -> e
       | None -> failwith (Printf.sprintf "%s: output leg %d never assigned" who i))
    out
;;

(* Wing knob: VFFT_CX_WING=1 swaps the n=16 FWD tangent interior for the
   machine-translated origin construction below (dft_cx16_wing). *)

(* dft_cx16_wing - the origin W-16 interior construction, machine-
 * translated from the validated dataflow (w16_to_cxml.py; self-gated
 * against DFT-16 during generation). Rotations act on combination
 * values (wing pairs) - 3 scalar constants, flips concentrated at +i
 * sites (crotp, shared by hash-consing). FWD only; inputs = post-
 * diagonal legs natural order; outputs natural. *)
let dft_cx16_wing (xs : t array) : t array =
  if Array.length xs <> 16 then failwith "dft_cx16_wing: needs 16 legs";
  let w0 = cadd xs.(0) xs.(8) in
  let w1 = csub xs.(0) xs.(8) in
  let w2 = cadd xs.(4) xs.(12) in
  let w3 = csub xs.(4) xs.(12) in
  let w4 = cadd xs.(6) xs.(14) in
  let w5 = csub xs.(14) xs.(6) in
  let w6 = csub xs.(2) xs.(10) in
  let w7 = cadd xs.(2) xs.(10) in
  let w8 = cadd w5 w6 in
  let w9 = csub w5 w6 in
  let w10 = cadd xs.(1) xs.(9) in
  let w11 = csub xs.(1) xs.(9) in
  let w12 = csub xs.(5) xs.(13) in
  let w13 = cadd xs.(13) xs.(5) in
  let w14 = csub w10 w13 in
  let w15 = cadd w10 w13 in
  let w16 = cfma 0.41421356237309503 w11 w12 in
  let w17 = cfnma 0.41421356237309503 w12 w11 in
  let w18 = csub xs.(15) xs.(7) in
  let w19 = cadd xs.(15) xs.(7) in
  let w20 = cadd xs.(11) xs.(3) in
  let w21 = csub xs.(11) xs.(3) in
  let w22 = cadd w20 w19 in
  let w23 = csub w19 w20 in
  let w24 = cfnma 0.41421356237309503 w21 w18 in
  let w25 = cfma 0.41421356237309503 w18 w21 in
  let w26 = csub w23 w14 in
  let w27 = cadd w14 w23 in
  let w28 = csub w0 w2 in
  let w29 = cfnma 0.70710678118654757 w27 w28 in
  let w30 = cfma 0.70710678118654757 w27 w28 in
  let w31 = csub w4 w7 in
  let w32 = cfnma 0.70710678118654757 w26 w31 in
  let w33 = cfma 0.70710678118654757 w26 w31 in
  let w34 = crot w32 in
  let w35 = cadd w29 w34 in
  let w36 = crotadd w30 w33 in
  let w37 = cadd w2 w0 in
  let w38 = crotadd w29 w32 in
  let w39 = crot w33 in
  let w40 = cadd w30 w39 in
  let w41 = cadd w7 w4 in
  let w42 = cfnma 0.70710678118654757 w8 w1 in
  let w43 = cfma 0.70710678118654757 w8 w1 in
  let w44 = cadd w25 w16 in
  let w45 = cfnma 0.92387953251128674 w44 w42 in
  let w46 = cfma 0.92387953251128674 w44 w42 in
  let w47 = cfma 0.70710678118654757 w9 w3 in
  let w48 = csub w24 w17 in
  let w49 = cfnma 0.92387953251128674 w48 w47 in
  let w50 = cfma 0.92387953251128674 w48 w47 in
  let w51 = crot w49 in
  let w52 = cadd w45 w51 in
  let w53 = crot w50 in
  let w54 = cadd w46 w53 in
  let w55 = csub w16 w25 in
  let w56 = crotadd w45 w49 in
  let w57 = cfnma 0.70710678118654757 w9 w3 in
  let w58 = crotadd w46 w50 in
  let w59 = cadd w24 w17 in
  let w60 = cadd w37 w41 in
  let w61 = csub w37 w41 in
  let w62 = cadd w15 w22 in
  let w63 = csub w22 w15 in
  let w64 = csub w60 w62 in
  let w65 = crotadd w61 w63 in
  let w66 = cadd w60 w62 in
  let w67 = crot w63 in
  let w68 = cadd w61 w67 in
  let w69 = cfnma 0.92387953251128674 w59 w43 in
  let w70 = cfma 0.92387953251128674 w59 w43 in
  let w71 = cfnma 0.92387953251128674 w55 w57 in
  let w72 = cfma 0.92387953251128674 w55 w57 in
  let w73 = crot w71 in
  let w74 = cadd w69 w73 in
  let w75 = crotadd w70 w72 in
  let w76 = crotadd w69 w71 in
  let w77 = crot w72 in
  let w78 = cadd w70 w77 in
  [| w66; w78; w36; w58; w65; w52; w35; w76; w64; w74; w38; w56; w68; w54; w40; w75 |]
;;

(* WING-T2 FULL KERNEL: loads + BYTW2 ingest + wing interior in ORIGIN
   listing order (machine-translated, self-gated). Used ONLY by the T2
   emitter under VFFT_CX_WING when tangent+R16+Fwd, so asis reproduces the
   hand kernel's interleave (peak 15). *)
(* dft_cx16_wing - the origin W-16 interior construction, machine-
 * translated from the validated dataflow (w16_to_cxml.py; self-gated
 * against DFT-16 during generation). Rotations act on combination
 * values (wing pairs) - 3 scalar constants, flips concentrated at +i
 * sites (crotp, shared by hash-consing). FWD only; inputs = post-
 * diagonal legs natural order; outputs natural. *)
let dft_cx16_wing_t2 () : t array =
  let w0 = cload (AZinLeg 8) in
  let w1 = ctwl 8 w0 in
  let w2 = cload (AZinLeg 0) in
  let w3 = cadd w2 w1 in
  let w4 = cload (AZinLeg 4) in
  let w5 = ctwl 4 w4 in
  let w6 = cload (AZinLeg 12) in
  let w7 = csub w2 w1 in
  let w8 = ctwl 12 w6 in
  let w9 = cadd w5 w8 in
  let w10 = cload (AZinLeg 14) in
  let w11 = ctwl 14 w10 in
  let w12 = cload (AZinLeg 10) in
  let w13 = ctwl 10 w12 in
  let w14 = cload (AZinLeg 6) in
  let w15 = ctwl 6 w14 in
  let w16 = cload (AZinLeg 2) in
  let w17 = ctwl 2 w16 in
  let w18 = csub w5 w8 in
  let w19 = cadd w15 w11 in
  let w20 = csub w11 w15 in
  let w21 = csub w17 w13 in
  let w22 = cadd w17 w13 in
  let w23 = cadd w20 w21 in
  let w24 = cload (AZinLeg 1) in
  let w25 = ctwl 1 w24 in
  let w26 = cload (AZinLeg 13) in
  let w27 = ctwl 13 w26 in
  let w28 = cload (AZinLeg 9) in
  let w29 = ctwl 9 w28 in
  let w30 = cload (AZinLeg 5) in
  let w31 = csub w20 w21 in
  let w32 = ctwl 5 w30 in
  let w33 = cadd w25 w29 in
  let w34 = csub w25 w29 in
  let w35 = csub w32 w27 in
  let w36 = cload (AZinLeg 15) in
  let w37 = ctwl 15 w36 in
  let w38 = cload (AZinLeg 11) in
  let w39 = cadd w27 w32 in
  let w40 = ctwl 11 w38 in
  let w41 = cload (AZinLeg 7) in
  let w42 = csub w33 w39 in
  let w43 = cadd w33 w39 in
  let w44 = ctwl 7 w41 in
  let w45 = cfma 0.41421356237309503 w34 w35 in
  let w46 = cload (AZinLeg 3) in
  let w47 = cfnma 0.41421356237309503 w35 w34 in
  let w48 = ctwl 3 w46 in
  let w49 = csub w37 w44 in
  let w50 = cadd w37 w44 in
  let w51 = cadd w40 w48 in
  let w52 = csub w40 w48 in
  let w53 = cadd w51 w50 in
  let w54 = csub w50 w51 in
  let w55 = cfnma 0.41421356237309503 w52 w49 in
  let w56 = cfma 0.41421356237309503 w49 w52 in
  let w57 = csub w54 w42 in
  let w58 = cadd w42 w54 in
  let w59 = csub w3 w9 in
  let w60 = cfnma 0.70710678118654757 w58 w59 in
  let w61 = cfma 0.70710678118654757 w58 w59 in
  let w62 = csub w19 w22 in
  let w63 = cfnma 0.70710678118654757 w57 w62 in
  let w64 = cfma 0.70710678118654757 w57 w62 in
  let w65 = crot w63 in
  let w66 = cadd w60 w65 in
  let w67 = crotadd w61 w64 in
  let w68 = cadd w9 w3 in
  let w69 = crotadd w60 w63 in
  let w70 = crot w64 in
  let w71 = cadd w61 w70 in
  let w72 = cadd w22 w19 in
  let w73 = cfnma 0.70710678118654757 w23 w7 in
  let w74 = cfma 0.70710678118654757 w23 w7 in
  let w75 = cadd w56 w45 in
  let w76 = cfnma 0.92387953251128674 w75 w73 in
  let w77 = cfma 0.92387953251128674 w75 w73 in
  let w78 = cfma 0.70710678118654757 w31 w18 in
  let w79 = csub w55 w47 in
  let w80 = cfnma 0.92387953251128674 w79 w78 in
  let w81 = cfma 0.92387953251128674 w79 w78 in
  let w82 = crot w80 in
  let w83 = cadd w76 w82 in
  let w84 = crot w81 in
  let w85 = cadd w77 w84 in
  let w86 = csub w45 w56 in
  let w87 = crotadd w76 w80 in
  let w88 = cfnma 0.70710678118654757 w31 w18 in
  let w89 = crotadd w77 w81 in
  let w90 = cadd w55 w47 in
  let w91 = cadd w68 w72 in
  let w92 = csub w68 w72 in
  let w93 = cadd w43 w53 in
  let w94 = csub w53 w43 in
  let w95 = csub w91 w93 in
  let w96 = crotadd w92 w94 in
  let w97 = cadd w91 w93 in
  let w98 = crot w94 in
  let w99 = cadd w92 w98 in
  let w100 = cfnma 0.92387953251128674 w90 w74 in
  let w101 = cfma 0.92387953251128674 w90 w74 in
  let w102 = cfnma 0.92387953251128674 w86 w88 in
  let w103 = cfma 0.92387953251128674 w86 w88 in
  let w104 = crot w102 in
  let w105 = cadd w100 w104 in
  let w106 = crotadd w101 w103 in
  let w107 = crotadd w100 w102 in
  let w108 = crot w103 in
  let w109 = cadd w101 w108 in
  [| w97; w109; w67; w89; w96; w83; w66; w107; w95; w105; w69; w87; w99; w85; w71; w106 |]
;;

let rec dft_cx ?(sign = `Fwd) ~(ctx : ctx) (n : int) (xs : t array) : t array =
  if n = 1
  then xs
  else if n = 16 && sign = `Fwd && ctx.tangent && ctx.wing_enabled
  then dft_cx16_wing xs
  else (
    let h = n / 2 in
    let e = dft_cx ~sign ~ctx h (Array.init h (fun i -> xs.(2 * i)))
    and o = dft_cx ~sign ~ctx h (Array.init h (fun i -> xs.((2 * i) + 1))) in
    let out = Array.make n None in
    for k = 0 to h - 1 do
      let a, b = butterfly_pair ~sign ~n ~k ~ctx e.(k) o.(k) in
      out.(k) <- Some a;
      out.(k + h) <- Some b
    done;
    unwrap_legs "dft_cx" out)
;;

(* ═══════════════════════════════════════════════════════════════
 *  ODD / PRIME RADICES - the CONJUGATE-PAIR construction
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
 * dft_recurse.ml (:327-468) - it is NOT discovered by algsimp there (which
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
 *  3. DEGENERATE-COEFFICIENT LADDER (cscale_chain): weights of 0 / +1 / -1 -
 *     which occur for odd COMPOSITES, e.g. n=9 at j=3,m=3 gives cos=1, sin=0 -
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
   magnitude becomes a constant - the same convention dft_recurse.ml uses, and
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

(* Winograd-5 on packed complex — port of dft_recurse.ml's
   dft_winograd5_cnum (owner pointer 2026-08-16: "see how odds/primes were
   wired for split layout"). Same conventions as dft_cx_odd: magnitudes ride
   in FMA constants, sigma*i is PRE-rotated so a Fwd kernel carries only
   CRotNI and a Bwd kernel only CRotPI, and both directions share one DAG
   shape. Constant set: { 1/4, sqrt5/4, (sqrt5-1)/2, sin(2pi/5) }.
   Gated VFFT_CX_WINOGRAD, default OFF = byte-identical trees. *)
let winograd_enabled = ref (Sys.getenv_opt "VFFT_CX_WINOGRAD" = Some "1")

let dft_cx_winograd5 ?(sign = `Fwd) (xs : t array) : t array =
  let rot x = if sign = `Fwd then crot x else crotp x in
  let s5 = sqrt 5.0 in
  let k_sin = sin (2.0 *. (4.0 *. atan 1.0) /. 5.0) in
  let s1 = cadd xs.(1) xs.(4)
  and d1 = csub xs.(1) xs.(4)
  and s2 = cadd xs.(2) xs.(3)
  and d2 = csub xs.(2) xs.(3) in
  let sg = cadd s1 s2 in
  let a = cfnma 0.25 sg xs.(0) in
  let bd = csub s1 s2 in
  let tb = cfma (s5 /. 4.0) bd a
  and tj = cfnma (s5 /. 4.0) bd a in
  let phi = (s5 -. 1.0) /. 2.0 in
  let u = cfma phi d2 d1
  and v = cfnma phi d1 d2 in
  let ru = rot u
  and rv = rot v in
  [| cadd xs.(0) sg
   ; cfma k_sin ru tb
   ; cfnma k_sin rv tj
   ; cfma k_sin rv tj
   ; cfnma k_sin ru tb
  |]
;;

let dft_cx_odd ?(sign = `Fwd) (n : int) (xs : t array) : t array =
  if n < 3 || n mod 2 = 0
  then failwith (Printf.sprintf "dft_cx_odd: needs an ODD n >= 3, got %d" n);
  if n = 5 && !winograd_enabled
  then dft_cx_winograd5 ~sign xs
  else begin
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
  end
;;

(* Leaf dispatcher: pow2 -> radix-2 DIT, odd -> conjugate pair, EVEN
   COMPOSITE (6, 10, 12, ...) -> radix-2 DIT splits whose halves
   RE-DISPATCH here, so the odd part bottoms out in the conjugate-pair
   builder instead of dropping legs (dft_cx recursing into an odd half was
   the hazard the old refusal guarded). butterfly_pair is the ONE shared
   copy, general-twiddle arm included, so the mixed recursion is the same
   numeric family - pow2 and odd radices take their old paths untouched
   (byte-identity preserved). Unlocks the 2-stage IL pairs at 4·odd² N
   (100 = 10x10, 36 = 6x6) and even-composite chain mids (200 = 4·(5·10)),
   per Tugbars 2026-07-29. *)
let rec dft_small ?(sign = `Fwd) ~(ctx : ctx) (n : int) (xs : t array) : t array =
  if n = 1
  then xs
  else if n land (n - 1) = 0
  then dft_cx ~sign ~ctx n xs
  else if n mod 2 = 1
  then dft_cx_odd ~sign n xs
  else (
    let h = n / 2 in
    let e = dft_small ~sign ~ctx h (Array.init h (fun i -> xs.(2 * i)))
    and o = dft_small ~sign ~ctx h (Array.init h (fun i -> xs.((2 * i) + 1))) in
    let out = Array.make n None in
    for k = 0 to h - 1 do
      let a, b = butterfly_pair ~sign ~n ~k ~ctx e.(k) o.(k) in
      out.(k) <- Some a;
      out.(k + h) <- Some b
    done;
    unwrap_legs "dft_small" out)
;;

(* Mixed-radix four-step over the complex IR: `chain` says how the transform is
   FACTORED, and dft_cx (pure radix-2) is only the leaf.

   Why this exists. dft_cx hardwires 2.2.2.2.2.2 for a 64-point stage - a plan
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
let rec dft_chain
          ~(sign : [ `Fwd | `Bwd ])
          ~(ctx : ctx)
          ~(chain : int list)
          (xs : t array)
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
  | [] | [ _ ] -> dft_cx ~sign ~ctx n xs
  | r0 :: rest ->
    let n2 = n / r0 in
    let sgn = if sign = `Fwd then -1.0 else 1.0 in
    let pi = 4.0 *. atan 1.0 in
    let cols =
      Array.init n2 (fun j2 ->
        dft_cx ~sign ~ctx r0 (Array.init r0 (fun j1 -> xs.((j1 * n2) + j2))))
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
      let r = dft_chain ~sign ~ctx ~chain:rest row in
      for k2 = 0 to n2 - 1 do
        out.((k2 * r0) + k1) <- r.(k2)
      done
    done;
    out
;;
