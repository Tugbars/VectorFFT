(* cx_real.ml — REAL-input / real-output DFTs on the cx IR, for the
   interleaved real pair's leaf (real_il.ml, kind r2z; 2026-09-29).

   A vector holds vec_width REAL lanes (four real columns at 256 bits), and
   every node built here is a real vector op — add, sub, negate, the
   real-constant fma and scale — so the packed-complex renderer emits them
   unchanged. A HALF SPECTRUM is an array of (re, im) node options over
   k = 0..n/2; None is an exact zero (the DC and Nyquist imaginary parts,
   the folded products), never emitted.

   rdft n x: the n-point real DFT of n real vectors, half-stored. Even n is
   the radix-2 DIT recursion on the even and odd samples (each half-stored,
   Hermitian): for k = 0..n/4, t = W^k O[k], X[k] = E[k] + t and X[n/2 - k] =
   conj(E[k] - t), which covers k = 0..n/2 exactly once. Odd n is the direct
   form on the pair sums s_j = x_j + x_{n-j} and differences d_j.
   irdft n X: the unnormalized inverse (n times x). Even n undoes the
   recursion: E'[k] = X[k] + conj X[n/2 - k] (= 2E[k]) and O'[k] = W^{-k}
   (X[k] - conj X[n/2 - k]) (= 2O[k]) feed the half-size inverses, whose
   n/2 * 2x outputs interleave to n x. Odd n: x_j = X_0 + A_j - B_j and
   x_{n-j} = X_0 + A_j + B_j with A_j = 2 sum Xr_k cos, B_j = 2 sum Xi_k sin.
   Angles are reduced mod n before cos/sin (the Cx_math.odd_angle law). *)

open Cx_ir

type hs = (t option * t option) array

let pi = 4.0 *. atan 1.0

let cs (n : int) (m : int) : float * float =
  let m = ((m mod n) + n) mod n in
  let a = 2.0 *. pi *. float_of_int m /. float_of_int n in
  cos a, sin a
;;

let eps = 1e-14
let near (x : float) (y : float) = abs_float (x -. y) < eps

(* c * x, folding 0 and +-1 *)
let scale (c : float) (x : t) : t option =
  if near c 0.0
  then None
  else if near c 1.0
  then Some x
  else if near c (-1.0)
  then Some (cneg x)
  else Some (ctw c 0.0 x)
;;

let omul (c : float) (x : t option) : t option =
  match x with
  | None -> None
  | Some x -> scale c x
;;

let oadd (a : t option) (b : t option) : t option =
  match a, b with
  | None, x | x, None -> x
  | Some a, Some b -> Some (cadd a b)
;;

let osub (a : t option) (b : t option) : t option =
  match a, b with
  | x, None -> x
  | None, Some b -> Some (cneg b)
  | Some a, Some b -> Some (csub a b)
;;

(* c * x + acc, folding: a 0 or None term vanishes, +-1 is an add/sub *)
let ofma (c : float) (x : t option) (acc : t option) : t option =
  match x with
  | None -> acc
  | Some x ->
    if near c 0.0
    then acc
    else (
      match acc with
      | None -> scale c x
      | Some e ->
        if near c 1.0
        then Some (cadd e x)
        else if near c (-1.0)
        then Some (csub e x)
        else Some (cfma c x e))
;;

let get (x : t option) (who : string) : t =
  match x with
  | Some v -> v
  | None -> failwith ("cx_real: an exact zero where a value was needed: " ^ who)
;;

let rec rdft (n : int) (x : t array) : hs =
  if Array.length x <> n then failwith "cx_real.rdft: sample count";
  if n = 1
  then [| Some x.(0), None |]
  else if n = 2
  then [| Some (cadd x.(0) x.(1)), None; Some (csub x.(0) x.(1)), None |]
  else if n mod 2 = 1
  then rdft_odd n x
  else (
    let m = n / 2 in
    let e = rdft m (Array.init m (fun j -> x.(2 * j))) in
    let o = rdft m (Array.init m (fun j -> x.((2 * j) + 1))) in
    let out = Array.make ((n / 2) + 1) (None, None) in
    for k = 0 to m / 2 do
      let er, ei = e.(k)
      and orr, oi = o.(k) in
      let c, s = cs n k in
      (* W^k = c - i s:  t = (c or + s oi, c oi - s or) *)
      let tr = ofma c orr (omul s oi) in
      let ti = ofma (-.s) orr (omul c oi) in
      out.(k) <- oadd er tr, oadd ei ti;
      if k <> m - k then out.(m - k) <- osub er tr, osub ti ei
    done;
    out)

and rdft_odd (n : int) (x : t array) : hs =
  let h = (n - 1) / 2 in
  let s = Array.init h (fun i -> cadd x.(i + 1) x.(n - 1 - i)) in
  let d = Array.init h (fun i -> csub x.(i + 1) x.(n - 1 - i)) in
  let out = Array.make (h + 1) (None, None) in
  out.(0) <- Array.fold_left (fun acc v -> oadd acc (Some v)) (Some x.(0)) s, None;
  for k = 1 to h do
    let re = ref (Some x.(0))
    and im = ref None in
    for j = 1 to h do
      let c, sn = cs n (j * k) in
      re := ofma c (Some s.(j - 1)) !re;
      im := ofma (-.sn) (Some d.(j - 1)) !im
    done;
    out.(k) <- !re, !im
  done;
  out
;;

let rec irdft (n : int) (xs : hs) : t array =
  if Array.length xs <> (n / 2) + 1 then failwith "cx_real.irdft: bin count";
  if n = 1
  then [| get (fst xs.(0)) "irdft n=1" |]
  else if n = 2
  then (
    let a = get (fst xs.(0)) "irdft X0"
    and b = get (fst xs.(1)) "irdft X1" in
    [| cadd a b; csub a b |])
  else if n mod 2 = 1
  then irdft_odd n xs
  else (
    let m = n / 2 in
    let e = Array.make ((m / 2) + 1) (None, None)
    and o = Array.make ((m / 2) + 1) (None, None) in
    for k = 0 to m / 2 do
      let c, s = cs n k in
      if k = m - k
      then (
        (* the self-paired bin: E' = 2 Xr, O' = W^{-k} (2 i Xi) with W^{-k} = +i *)
        let xr, xi = xs.(k) in
        let di = omul 2.0 xi in
        e.(k) <- omul 2.0 xr, None;
        o.(k) <- omul (-.s) di, omul c di)
      else (
        let xr, xi = xs.(k)
        and yr, yi = xs.(m - k) in
        let er = oadd xr yr
        and ei = osub xi yi
        and dr = osub xr yr
        and di = oadd xi yi in
        (* W^{-k} = c + i s:  (c + is)(dr + i di) = (c dr - s di, c di + s dr) *)
        e.(k) <- er, ei;
        o.(k) <- ofma (-.s) di (omul c dr), ofma s dr (omul c di))
    done;
    let ev = irdft m e
    and ov = irdft m o in
    Array.init n (fun j -> if j mod 2 = 0 then ev.(j / 2) else ov.(j / 2)))

and irdft_odd (n : int) (xs : hs) : t array =
  let h = (n - 1) / 2 in
  let x0 = get (fst xs.(0)) "irdft_odd X0" in
  let out = Array.make n x0 in
  out.(0) <- get (Array.fold_left (fun acc k -> ofma 2.0 (fst xs.(k)) acc) (Some x0)
                    (Array.init h (fun i -> i + 1))) "irdft_odd x0";
  for j = 1 to h do
    let a = ref (Some x0)
    and b = ref None in
    for k = 1 to h do
      let c, s = cs n (j * k) in
      a := ofma (2.0 *. c) (fst xs.(k)) !a;
      b := ofma (2.0 *. s) (snd xs.(k)) !b
    done;
    out.(j) <- get (osub !a !b) "irdft_odd x_j";
    out.(n - j) <- get (oadd !a !b) "irdft_odd x_{n-j}"
  done;
  out
;;
