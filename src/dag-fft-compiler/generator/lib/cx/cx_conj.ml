(* cx_conj.ml -- THE COSINE AND SINE PARTS (2026-11-11, the --cil-conj form; design:
 * docs/design/il_avx512_kernel_operations.md).
 *
 * The R-point DFT across registers (every lane an independent transform, the same constants on every
 * lane: the column kernels' body) built so that no value is swapped until an output is made:
 *
 *     X_k     = A_k + r B_k        A_k = sum_m x_m cos(2 pi k m / R)
 *     X_(R-k) = A_k - r B_k        B_k = sum_m x_m sin(2 pi k m / R)        r = -i (forward), +i (backward)
 *
 * A and B are REAL-coefficient combinations of the packed-complex inputs, so building them needs adds,
 * subtracts and FMAs with real constants only; the one swap of B_k serves both outputs of the pair
 * (X_0 and X_(R/2) need none): R/2 - 1 swaps per transform, against one per quarter turn and per
 * constant twiddle in the eager construction (Cx_math.dft_small). The data stays interleaved end to end.
 *
 * A and B are built for k = 0 .. R/2 by the real split-radix recursion (X = E + W^k O1 + W^3k O3 on a
 * real-linear input, Hermitian symmetry for the indices past the half), with a PENDING REAL SCALE on
 * every part (Cx_lazy's idea): a constant scale is free, an add folds the ratio of its two operands'
 * scales into one FMA, and a twiddle on a pair (A, B) factors the larger of |cos|, |sin| out (the
 * tangent form) so it costs two FMAs. The recursion is the same in both directions; only r differs.
 *
 * Nodes: CAdd, CSub, CFmaC, CFnmaC and CRotNI / CRotPI (read through the quarter-turn FMA fold of the
 * renderer). No lazy (CLz) node and no zero vector: every node here renders at every width a column
 * kernel's tails use.
 *
 * MODULE CARD
 * ROLE: dft ~sign n xs -- the outputs X_0 .. X_(n-1) of the n-point DFT of xs, n a power of two.
 * DEPS: Cx_ir (nodes), Cx_lazy.cs (the exact constants). *)

open Cx_ir

(* a part: the value s * z, s a nonzero real; None = zero *)
type part = (t * float) option

let near (a : float) (b : float) : bool = abs_float (a -. b) < 1e-12
let unitlike (s : float) : bool = near (abs_float s) 1.0

let scale (c : float) (x : part) : part =
  match x with
  | None -> None
  | Some _ when abs_float c < 1e-15 -> None
  | Some (z, s) -> Some (z, c *. s)
;;

let neg (x : part) : part = scale (-1.0) x

(* su u + sv v. The result keeps the frame of the operand whose scale is +-1 when only one is; the
   other's ratio rides in one FMA. *)
let add (x : part) (y : part) : part =
  match x, y with
  | None, p | p, None -> p
  | Some (u, su), Some (v, sv) ->
    let (u, su), (v, sv) = if unitlike sv && not (unitlike su) then (v, sv), (u, su) else (u, su), (v, sv) in
    let k = sv /. su in
    if near k 1.0
    then Some (cadd u v, su)
    else if near k (-1.0)
    then Some (csub u v, su)
    else if k > 0.0
    then Some (cfma k v u, su)
    else Some (cfnma (-.k) v u, su)
;;

(* (c - r' s)(A + r B), written for the forward (r = -i, W = c - i s): (cA - sB) + r (sA + cB); the
   backward (r = +i, W = c + i s) gives the same real formulas. A lone part takes the scales for free;
   a pair factors the larger of |c|, |s| out. *)
let tw ((c : float), (s : float)) ((a : part), (b : part)) : part * part =
  match a, b with
  | _, None -> scale c a, scale s a
  | None, _ -> scale (-.s) b, scale c b
  | _ ->
    if abs_float s < 1e-15
    then scale c a, scale c b
    else if abs_float c < 1e-15
    then scale (-.s) b, scale s a
    else if abs_float c >= abs_float s
    then (
      let t = s /. c in
      scale c (add a (scale (-.t) b)), scale c (add (scale t a) b))
    else (
      let t = c /. s in
      scale s (add (scale t a) (neg b)), scale s (add a (scale t b)))
;;

(* W_n^e of the forward as (c, s) with W = c - i s, exact at the multiples of pi/4 (Cx_lazy.cs reduces
   the angle in integers) *)
let cs (e : int) (n : int) : float * float =
  let c, s' = Cx_lazy.cs ~bwd:false e n in
  c, -.s'
;;

(* A_k, B_k for k = 0 .. n/2 of the real-linear input xs (B_0 = B_(n/2) = None) *)
let rec halves (xs : part array) : part array * part array =
  let n = Array.length xs in
  if n = 1
  then [| xs.(0) |], [| None |]
  else if n = 2
  then [| add xs.(0) xs.(1); add xs.(0) (neg xs.(1)) |], [| None; None |]
  else (
    let h = n / 2
    and q = n / 4 in
    let ea, eb = halves (Array.init h (fun i -> xs.(2 * i))) in
    let a1, b1 = halves (Array.init q (fun i -> xs.((4 * i) + 1))) in
    let a3, b3 = halves (Array.init q (fun i -> xs.((4 * i) + 3))) in
    (* a length-m transform's parts at any index (conjugate past m/2) *)
    let at (a : part array) (b : part array) (m : int) (k : int) : part * part =
      let k = k mod m in
      if 2 * k <= m then a.(k), b.(k) else a.(m - k), neg b.(m - k)
    in
    let ra = Array.make (h + 1) None
    and rb = Array.make (h + 1) None in
    for k = 0 to q - 1 do
      let o1a, o1b = tw (cs k n) (at a1 b1 q k) in
      let o3a, o3b = tw (cs (3 * k) n) (at a3 b3 q k) in
      let ta = add o1a o3a
      and tb = add o1b o3b in
      let da = add o1a (neg o3a)
      and db = add o1b (neg o3b) in
      let e0a, e0b = at ea eb h k in
      (* X_k = E_k + T *)
      ra.(k) <- add e0a ta;
      rb.(k) <- add e0b tb;
      (* X_(n/2) = E_0 - T *)
      if k = 0
      then (
        ra.(h) <- add e0a (neg ta);
        rb.(h) <- add e0b (neg tb));
      (* X_(k+n/4) = E_(k+n/4) + r D, and r (Da + r Db) = -Db + r Da *)
      let eqa, eqb = at ea eb h (k + q) in
      ra.(k + q) <- add eqa (neg db);
      rb.(k + q) <- add eqb da
    done;
    ra, rb)
;;

(* an output's real part as a node. The frames of the recursion leave every A_k with the scale +1 at
   radix 4 .. 64 (counted 2026-11-11), so nothing is spent here; a radix that ever left another scale
   fails loudly at generation instead of emitting a bare multiply. *)
let plain ~(n : int) ~(k : int) (x : part) : t =
  match x with
  | None -> failwith (Printf.sprintf "Cx_conj: radix %d, output %d has no real part" n k)
  | Some (z, s) ->
    if near s 1.0
    then z
    else
      failwith
        (Printf.sprintf "Cx_conj: radix %d, output %d keeps the scale %.17g on its real part" n k s)
;;

(* the output pair (A + r B, A - r B): one swap of B (the renderer's quarter-turn FMA fold reads the
   rotation's operand through the swap, and both outputs read the same swapped value) *)
let outputs ~(sign : [ `Fwd | `Bwd ]) ~(n : int) ~(k : int) (a : part) (b : part) : t * t =
  let rot z = if sign = `Fwd then crot z else crotp z in
  let p = plain ~n ~k a in
  match b with
  | None -> p, p
  | Some (v, sv) -> cfma sv (rot v) p, cfnma sv (rot v) p
;;

let dft ~(sign : [ `Fwd | `Bwd ]) (n : int) (xs : t array) : t array =
  if n < 2 || n land (n - 1) <> 0
  then failwith (Printf.sprintf "Cx_conj.dft: radix %d is not a power of two >= 2" n);
  if Array.length xs <> n then failwith "Cx_conj.dft: wrong number of inputs";
  let a, b = halves (Array.map (fun z -> Some (z, 1.0)) xs) in
  let out = Array.make n xs.(0) in
  for k = 0 to n / 2 do
    let x, y = outputs ~sign ~n ~k a.(k) b.(k) in
    out.(k) <- x;
    if k > 0 && k < n / 2 then out.(n - k) <- y
  done;
  out
;;
