(* cx_lazy.ml -- LAZY TERMS over the packed-complex IR (2026-11-11).
 *
 * A term is a value with a pending real scale per complex lane and a pending quarter turn:
 *
 *     value = s_lane * i^r * z          (r = 0 or 1, z a Cx_ir node)
 *
 * A DFT built on terms never multiplies by a twiddle on its own. A twiddle w = c + i s leaves a
 * scale on the value -- s (k z + i z) with k = c / s, or c (z + (s / c) i z) -- for one swap and
 * one FMA, and the next butterfly's add absorbs the scale as the constant of ITS FMA:
 *
 *     p +- s z   = fmadd / fnmadd (z, [s s], p)
 *     p +- s i z = fmadd / fnmadd (cflip z, [-s s], p)
 *
 * so a twiddled radix-2 butterfly is a swap and three FMA-class operations, and one with a
 * quarter turn a swap and two, where the eager form (multiply, then add and subtract) is a swap,
 * a multiply, an FMA and two adds. The scales are per lane, so the lane-varying twiddles between
 * the two stages of a fused transform (lane = output index of stage 1) carry through the corner
 * turn into stage 2 the same way. Multiplications by +-1 and +-i cost nothing: they change the
 * term's sign or turn.
 *
 * WHY A BUILDER AND NOT A REWRITE PASS: the cx IR has no bare multiply -- a twiddle is one node
 * (CTwC / CTwV) with its constants inside -- so no pass can move a scale out of a twiddle and
 * into the add that follows (cx_pipeline.ml, "why the cascade is short"). The scale has to be
 * kept apart while the DAG is built. The nodes are Cx_ir.CLz (the lazy operations).
 *
 * The algebra and the DFT are those of the FP32 mono study (2026-11-08; one-call kernels that
 * beat the two-pass plans at N = 16 .. 1024): decimation in time, radix 2, radix 4 or split
 * radix per size, whichever builds the fewest operations (strategies, a dry run per candidate).
 *
 * MODULE CARD
 * ROLE: term, the term algebra (mat, mulw, comb, twiddle), dft, strategies.
 * DEPS: Cx_ir (nodes), Cx_pipeline (the reachable walk of the dry runs).
 * GOTCHA: strategies builds and discards DAGs: it calls Cx_ir.reset, so call it BEFORE the
 * emission's own reset and builds. *)

open Cx_ir

type term =
  { z : t
  ; s : float array (* one real factor per complex lane *)
  ; r : int (* 0, or 1 = a pending multiplication by i *)
  }

let ones (per : int) : float array = Array.make per 1.0
let plain_of ~(per : int) (z : t) : term = { z; s = ones per; r = 0 }
let all_eq (v : float) (s : float array) : bool = Array.for_all (fun x -> x = v) s
let is_plain (t : term) : bool = t.r = 0 && all_eq 1.0 t.s
let neg (s : float array) : float array = Array.map (fun x -> -.x) s

(* p + i z. 256 bits: addsub through the swap (Cx_ir.CRotAdd: no constant). 512 bits has no addsub:
   the same value as one FMA with the constant [-1 +1] (the emitter sets addsub before it builds). *)
let addsub = ref true

let rotadd ~(per : int) (p : t) (z : t) : t =
  if !addsub then crotadd p z else clz LFmaR (Array.make per 1.0) z p
;;

(* W_n^e = exp(-2 pi i e / n) (backward: the conjugate) as (c, s), the value c + i s. Exact at the
   multiples of pi/4; elsewhere the angle is reduced to [0, pi/4] in integers before cos / sin. *)
let cs ~(bwd : bool) (e : int) (n : int) : float * float =
  let e = ((e mod n) + n) mod n in
  let e = if bwd then (n - e) mod n else e in
  let q4 = 4 * e / n
  and rem = 4 * e mod n in
  let pi = 4.0 *. atan 1.0 in
  let cp, sp =
    if rem = 0
    then 1.0, 0.0
    else if 2 * rem = n
    then (
      let h = sqrt 0.5 in
      h, h)
    else if 2 * rem < n
    then (
      let a = 0.5 *. pi *. float_of_int rem /. float_of_int n in
      cos a, sin a)
    else (
      let a = 0.5 *. pi *. float_of_int (n - rem) /. float_of_int n in
      sin a, cos a)
  in
  let c, sn =
    match q4 with
    | 0 -> cp, sp
    | 1 -> -.sp, cp
    | 2 -> -.cp, -.sp
    | _ -> sp, -.cp
  in
  c +. 0.0, -.sn +. 0.0
;;

type unit_kind =
  | One
  | MOne
  | MI (* -i *)
  | PI (* +i *)
  | Gen of float * float

(* W_n^e by its integers: the exact units, else the general (c, s) *)
let unitk ~(bwd : bool) (e : int) (n : int) : unit_kind =
  let e = ((e mod n) + n) mod n in
  let e = if bwd then (n - e) mod n else e in
  if e = 0
  then One
  else if 2 * e = n
  then MOne
  else if 4 * e = n
  then MI
  else if 4 * e = 3 * n
  then PI
  else (
    let c, s = cs ~bwd:false e n in
    Gen (c, s))
;;

(* the term as a plain value: no instruction for a plain term, one multiply otherwise *)
let mat (t : term) : t =
  if t.r = 0 then if all_eq 1.0 t.s then t.z else clz LMul t.s t.z t.z else clz LMulR t.s t.z t.z
;;

(* t * W_n^e, the same twiddle on every lane: +-1 and +-i move into the term; a general twiddle is
   one FMA (through the swap): z' = k z + i z, k = c / s, the value s z' *)
let mulw ~(bwd : bool) (t : term) (e : int) (n : int) : term =
  match unitk ~bwd e n with
  | One -> t
  | MOne -> { t with s = neg t.s }
  | MI -> if t.r = 0 then { t with s = neg t.s; r = 1 } else { t with r = 0 }
  | PI -> if t.r = 0 then { t with r = 1 } else { t with s = neg t.s; r = 0 }
  | Gen (c, s) ->
    let c, s = if t.r <> 0 then -.s, c else c, s in
    let k = c /. s in
    let zz =
      if k = 1.0
      then rotadd ~per:(Array.length t.s) t.z t.z
      else clz LKI (Array.make (Array.length t.s) k) t.z t.z
    in
    { z = zz; s = Array.map (fun x -> x *. s) t.s; r = 0 }
;;

(* p a plain value, x a term. rev = false: (p + x, p - x); rev = true: (x + p, x - p). Plain. *)
let pm (p : t) (x : term) (rev : bool) : t * t =
  let z = x.z
  and s = x.s in
  let one = all_eq 1.0 s
  and mone = all_eq (-1.0) s in
  if x.r = 0
  then
    if one
    then cadd p z, if rev then csub z p else csub p z
    else if mone && not rev
    then csub p z, cadd p z
    else clz LFma s z p, if rev then clz LFms s z p else clz LFnma s z p
  else if (not rev) && one
  then rotadd ~per:(Array.length s) p z, clz LFnmaR s z p
  else if (not rev) && mone
  then clz LFnmaR (neg s) z p, rotadd ~per:(Array.length s) p z
  else clz LFmaR s z p, if rev then clz LFmsR s z p else clz LFnmaR s z p
;;

(* (u + v, u - v). The result keeps the frame (scale and turn) of the plainer operand; the other is
   folded in by one FMA with the per-lane ratio of the two scales. *)
let comb (u : term) (v : term) : term * term =
  let per = Array.length u.s in
  if is_plain u && is_plain v
  then plain_of ~per (cadd u.z v.z), plain_of ~per (csub u.z v.z)
  else (
    let score (t : term) =
      ( (if is_plain t then 0 else 1)
      , t.r
      , if Array.for_all (fun x -> abs_float x = 1.0) t.s then 0 else 1 )
    in
    let fr, ot, rev = if compare (score v) (score u) < 0 then v, u, true else u, v, false in
    let rho = Array.mapi (fun i o -> o /. fr.s.(i)) ot.s in
    let rr = ot.r - fr.r in
    let rr, rho = if rr < 0 then 1, neg rho else rr, rho in
    let a, b = pm fr.z { z = ot.z; s = rho; r = rr } rev in
    { z = a; s = fr.s; r = fr.r }, { z = b; s = fr.s; r = fr.r })
;;

(* the smallest carried scale a lazy lane-varying twiddle may leave: below it the ratio s / c (or
   c / s) of the form grows past 4.9 and the worst-case error with it (the study's setting) *)
let tw_min_scale = 0.2

(* t * W_n^(exps.(lane)): lane-varying twiddles, the term's own scale and turn folded into the
   constants. A term with r = 0:
     every lane's cos usable:  z' = z + (s / c) i z   (one FMA through the swap), the value c z'
     every lane's sin usable:  z' = (c / s) z + i z   (one FMA through the swap), the value s z'
       (of the two, the one whose smallest carried scale is larger)
     otherwise the plain multiply (CTwV: swap, multiply, FMA), a plain term. *)
let twiddle ~(bwd : bool) (t : term) (exps : int array) (n : int) : term =
  let per = Array.length t.s in
  let ws =
    Array.mapi
      (fun i e ->
         let c, s = cs ~bwd e n in
         let c, s = c *. t.s.(i), s *. t.s.(i) in
         if t.r <> 0 then -.s, c else c, s)
      exps
  in
  let usable f = Array.for_all (fun w -> f w <> 0.0 && abs_float (f w) >= tw_min_scale) ws in
  let smallest f = Array.fold_left (fun m w -> min m (abs_float (f w))) infinity ws in
  let form =
    match usable fst, usable snd with
    | true, true -> if smallest fst >= smallest snd then `C else `K
    | true, false -> `C
    | false, true -> `K
    | false, false -> `Plain
  in
  match form with
  | `C ->
    { z = clz LFmaR (Array.map (fun (c, s) -> s /. c) ws) t.z t.z; s = Array.map fst ws; r = 0 }
  | `K -> { z = clz LKI (Array.map (fun (c, s) -> c /. s) ws) t.z t.z; s = Array.map snd ws; r = 0 }
  | `Plain -> plain_of ~per (ctwv ws t.z)
;;

(* ── the DFT across registers ─────────────────────────────────────────── *)

type strat =
  | R2
  | R4
  | SR

(* DFT_R of the terms xs (every lane an independent R-point DFT), natural order out. st r = the
   construction at size r >= 4. out k term is called as soon as output k of THIS call exists (the
   sub-DFTs are not reported): a caller that consumes an output there -- stores it, or finishes a
   block of outputs -- builds its nodes next to the output's, so in construction order the
   output dies early. *)
let rec dft ?(out : (int -> term -> unit) option) ~(bwd : bool) ~(st : int -> strat) (xs : term array)
  : term array
  =
  let r = Array.length xs in
  let tell k t =
    match out with
    | Some f -> f k t
    | None -> ()
  in
  if r = 1
  then (
    tell 0 xs.(0);
    [| xs.(0) |])
  else if r = 2
  then (
    let p, m = comb xs.(0) xs.(1) in
    tell 0 p;
    tell 1 m;
    [| p; m |])
  else (
    let res = Array.make r xs.(0) in
    let set k t =
      res.(k) <- t;
      tell k t
    in
    let sub step off = Array.init (r / step) (fun i -> xs.(off + (i * step))) in
    let go xs = dft ~bwd ~st xs in
    (match st r with
     | R2 ->
       let e = go (sub 2 0) in
       let o = go (sub 2 1) in
       for k = 0 to (r / 2) - 1 do
         let p, m = comb e.(k) (mulw ~bwd o.(k) k r) in
         set k p;
         set (k + (r / 2)) m
       done
     | R4 ->
       let e0 = go (sub 4 0) in
       let e1 = go (sub 4 1) in
       let e2 = go (sub 4 2) in
       let e3 = go (sub 4 3) in
       for k = 0 to (r / 4) - 1 do
         let t1 = mulw ~bwd e1.(k) k r in
         let t2 = mulw ~bwd e2.(k) (2 * k) r in
         let t3 = mulw ~bwd e3.(k) (3 * k) r in
         let a, b = comb e0.(k) t2 in
         let c, d = comb t1 t3 in
         let x0, x2 = comb a c in
         set k x0;
         set (k + (r / 2)) x2;
         let x1, x3 = comb b (mulw ~bwd d 1 4) in
         set (k + (r / 4)) x1;
         set (k + (3 * r / 4)) x3
       done
     | SR ->
       let e = go (sub 2 0) in
       let o1 = go (sub 4 1) in
       let o3 = go (sub 4 3) in
       for k = 0 to (r / 4) - 1 do
         let t1 = mulw ~bwd o1.(k) k r in
         let t3 = mulw ~bwd o3.(k) (3 * k) r in
         let sm, df = comb t1 t3 in
         let x0, x2 = comb e.(k) sm in
         set k x0;
         set (k + (r / 2)) x2;
         let x1, x3 = comb e.(k + (r / 4)) (mulw ~bwd df 1 4) in
         set (k + (r / 4)) x1;
         set (k + (3 * r / 4)) x3
       done);
    res)
;;

(* the operations a set of roots costs: one per node that is no load, the plain twiddle's separate
   multiply, and one per distinct swapped value *)
let op_count (roots : t list) : int =
  let n = ref 0 in
  let swapped : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let sw (z : t) = Hashtbl.replace swapped z.tag () in
  Cx_pipeline.iter_reachable roots (fun e ->
    match e.node with
    | CIn _ | CLoad _ -> ()
    | CLz ((LMulR | LFmaR | LFnmaR | LFmsR | LKI), _, z, _) ->
      incr n;
      sw z
    | CRotAdd (_, y) ->
      incr n;
      sw y
    | CTwV (_, x) | CTwC (_, _, x) | CTwA (_, _, x) ->
      n := !n + 2;
      sw x
    | _ -> incr n);
  !n + Hashtbl.length swapped
;;

(* The construction per size 4, 8, .. maxr: the cheapest of radix 4, split radix and radix 2 by
   op_count of a dry run on plain inputs with plain outputs (ties in that order). Resets the IR. *)
let strategies ~(bwd : bool) ~(per : int) (maxr : int) : int -> strat =
  let tbl : (int, strat) Hashtbl.t = Hashtbl.create 8 in
  let st r =
    match Hashtbl.find_opt tbl r with
    | Some s -> s
    | None -> failwith (Printf.sprintf "Cx_lazy: no construction chosen for DFT_%d" r)
  in
  let cost r =
    reset ();
    let xs = Array.init r (fun i -> plain_of ~per (cin i)) in
    let ys = dft ~bwd ~st xs in
    op_count (Array.to_list (Array.map mat ys))
  in
  let r = ref 4 in
  while !r <= maxr do
    let best = ref None in
    List.iter
      (fun s ->
         if not (s = R4 && !r mod 4 <> 0)
         then (
           Hashtbl.replace tbl !r s;
           let c = cost !r in
           match !best with
           | Some (bc, _) when bc <= c -> ()
           | _ -> best := Some (c, s)))
      [ R4; SR; R2 ];
    (match !best with
     | Some (_, s) -> Hashtbl.replace tbl !r s
     | None -> failwith "Cx_lazy.strategies: no candidate");
    r := 2 * !r
  done;
  reset ();
  st
;;

let strat_name = function
  | R2 -> "radix 2"
  | R4 -> "radix 4"
  | SR -> "split radix"
;;
