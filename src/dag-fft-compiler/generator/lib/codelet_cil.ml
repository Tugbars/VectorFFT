(* codelet_cil.ml — pipeline-hosted INTERLEAVED-COMPLEX (full-IL) codelets.
 *
 * Full IL is THE mainstream layout, not an alternative: MKL and FFTW both
 * default to interleaved, it is the natural order + the natural K=1 handling
 * standard, it pays no R2C/C2R packing tax, and roughly 80% of users are on
 * it. Split-complex is the specialization for high-throughput batched work.
 * The goal is IL as COMPLETE as FFTW's/MKL's. This module generates the
 * interleaved family the way the rest of the generator works, replacing
 * codelet_zil.ml's hand-scheduled raw-string emission.
 *
 * KINDS: n1 (solo monolithic), n1t (bailey2 stage-1 leaf, four-step
 * transpose fused into the stores), t2 (bailey2 stage-2 mid, streamed VTW2
 * twiddles) — each in BOTH directions.
 *
 * WHAT IS SHARED WITH THE REST OF THE PIPELINE
 * --------------------------------------------
 *   Schedule.Make  — the SR (Starve-Retire) list scheduler, instantiated on
 *                    THIS module's complex IR via the SCHED_NODE signature.
 *                    Replaces codelet_zil's hand A/B interleaving; the same
 *                    scheduler that drives every split-real codelet. MEASURED
 *                    WIN at large radix: -39% reg-reg moves at r32, -49% at
 *                    r64, -34% stack traffic at r16 vs the hand order.
 *   Isa.*          — every intrinsic is built through the ISA layer, so the
 *                    emitted code is width-parametric (AVX2 today, AVX-512
 *                    by passing a different Isa.t) instead of 486 literal
 *                    _mm256_ calls.
 *
 * WHAT IS *NOT* SHARED — Ir / Algsimp / Dft, and the honest reasons
 * -----------------------------------------------------------------
 * Those are REAL-VALUED by construction: the DAG has already split complex
 * into re/im subtrees, and `of_expr` hard-matches the split-cmul shape, so a
 * packed-complex DAG cannot flow through them. Merging packed-complex kinds
 * into Ir.node_kind was measured at ~150-180 exhaustive-match arms across 7
 * modules (dune's dev profile makes non-exhaustive matches fatal).
 *
 * Do NOT read that as "algsimp is unnecessary" — it is NOT (see
 * regalloc.ml §1/§4 and the measurements in zil_pipeline_port.md §11):
 * fma_lift's real job is to leave gcc no mul+add to fuse, so gcc stops
 * re-scheduling off the SU+GH order and stops churning the register
 * allocator. Disabling it costs +8..22% ops on pow2 and +33..71% on
 * odd/prime.
 *
 * This module does not NEED those passes for two specific reasons:
 *   1. pow2 IL is FMA-shaped BY CONSTRUCTION — the twiddle classes fold
 *      sqrt(1/2) into FMAs and BYTW2 is exactly 1 mul + 1 fma (optimal for a
 *      complex multiply). Verified: mul/fma counts identical to the hand
 *      kernels, reg-reg moves stuck at 2-7. There is no FMA-lift headroom
 *      left to recover.
 *   2. For odd/prime radices algsimp would NOT help anyway: what it
 *      contributes there is Winograd STRUCTURE, which it finds in the REAL
 *      expansion (c*x_a + c*x_b -> c*(x_a+x_b)); packed-complex nodes hide
 *      that structure inside opaque complex atoms. Odd/prime IL therefore
 *      needs a COMPLEX Winograd/Rader math layer — an open item, and the
 *      reason `emit` currently hard-fails on non-pow2 radices.
 *
 * Acknowledged duplication: `dft_cx` below re-expresses the DIT recursion
 * that dft.ml already has for the real-valued side. ~40 lines, no cleverness,
 * but it IS a second copy — keep the twiddle-class selection in sync if the
 * real-side one ever changes. CSE is not lost: hash-consing here gives the
 * same sharing Ir gets (verified by bit-identical output vs the hand kernels).
 *
 * IL PRIMITIVES (the only ops with no real-lane equivalent)
 * ---------------------------------------------------------
 * A vector holds vec_width/2 complex as [re,im,re,im,...]:
 *   complex add/sub  = plain vector add/sub          (Isa.add_pd / sub_pd)
 *   c*x + e (real c) = plain FMA                     (Isa.fmadd_pd)
 *   x * (-i)         = xor(cflip x, _M_IM)   fwd quarter-turn
 *   x * (+i)         = xor(cflip x, _M_RE)   bwd quarter-turn
 *   x * (c + i·s)    = fmadd(cvec, x, mul(svec, cflip x))   -- BYTW2 shape,
 *                      cvec/svec either emit-time VLIT constants (n1) or
 *                      loaded from the streamed VTW2 table (t2).
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_cil.ml — grep "MODULE CARD" for the full set)
 * ROLE: interleaved-complex (full-IL) codelet emitter — solo + bailey2,
 * fwd + bwd, pow2 radices.
 * PIPELINE: cx math -> hash-cons -> Schedule.Make(Node).su_schedule ->
 * Isa-parametric emission.
 * PUBLIC SURFACE: emit ~kind ~dir ~radix ~isa ~uarch
 * (gen_main --cil-n1 / --cil-n1t / --cil-t2 [--cil-bwd]).
 * DEPS: Schedule (functor), Isa, Uarch, Expr (elem_ref labels only).
 * GOTCHA 1: `reset ()` MUST run before each codelet — the hash-cons table
 * and tag counter are module-global, exactly like Algsimp.reset.
 * GOTCHA 2: bwd conjugation for t2 is TABLE-SIDE (caller passes a conjugated
 * table); the kernel's BYTW2 is bit-for-bit the forward one, only its
 * POSITION moves (pre-butterfly fwd, post-butterfly bwd).
 * ------------------------------------------------------------------ *)

(* ═══════════════════════════════════════════════════════════════
 *  THE COMPLEX IR
 * ═══════════════════════════════════════════════════════════════ *)

type cx_kind =
  | CIn of int (* input leg i (a packed-complex load) *)
  | CAdd of t * t
  | CSub of t * t
  | CRotNI of t (* x * (-i) : cflip then negate the IM lane *)
  | CRotPI of t
  (* x * (+i) : cflip then negate the RE lane. The backward twin of
     CRotNI — an inverse transform's quarter-turn goes the other way.
     (a+bi)*(+i) = -b + ai, so from cflip = [b,a] we negate lane 0. *)
  | CFmaC of float * t * t (* c*x + e,  real scalar c *)
  | CFnmaC of float * t * t (* -c*x + e, real scalar c *)
  | CTwC of float * float * t (* x * (c + i*s), emit-time constants *)
  | CTwV of (float * float) array * t
  (* x * w, with a DIFFERENT emit-time constant per complex lane. The K=1
     fused kernel needs this: one vector holds two DIFFERENT output indices
     k1 of the same transform, so their twiddles w_N^{k1*j2} differ. Still
     zero runtime cost — the whole [c,c,c',c'][-s,+s,-s',+s'] pair is a
     file-scope VLIT, exactly like CTwC's. Array length = vec_width/2. *)
  | CTwL of int * t
  (* x * w[leg], w LOADED from the streamed VTW2 table — the bailey2 t2
     mid. Same BYTW2 shape as CTwC, but cvec/svec come from the runtime
     cursor `twp` instead of file-scope VLIT constants. The int is the
     LEG index; the record offset is (leg-1)*2*VW because leg 0 is
     untwiddled and each record is [c×VW][s×VW] (cos-first, sign-folded
     — one data-side shuffle, zero table-side work). *)

and t =
  { tag : int
  ; node : cx_kind
  }

(* Hash-consing: structural equality becomes tag equality, which is what
 * gives us CSE for free (the shared ±i rotations and repeated subsums in a
 * radix-8 body dedup automatically). Mirrors Ir.hashcons. *)
let hcons : (cx_kind, t) Hashtbl.t = Hashtbl.create 256
let next_tag = ref 0

let reset () =
  Hashtbl.reset hcons;
  next_tag := 0
;;

let mk (nk : cx_kind) : t =
  match Hashtbl.find_opt hcons nk with
  | Some e -> e
  | None ->
    let e = { tag = !next_tag; node = nk } in
    incr next_tag;
    Hashtbl.add hcons nk e;
    e
;;

let cin i = mk (CIn i)
let cadd a b = mk (CAdd (a, b))
let csub a b = mk (CSub (a, b))
let crot a = mk (CRotNI a)
let crotp a = mk (CRotPI a)
let cfma c x e = mk (CFmaC (c, x, e))
let cfnma c x e = mk (CFnmaC (c, x, e))
let ctw c s x = mk (CTwC (c, s, x))
let ctwv w x = mk (CTwV (w, x))
let ctwl leg x = mk (CTwL (leg, x))

(* ═══════════════════════════════════════════════════════════════
 *  SCHEDULER INSTANTIATION — the shared SR scheduler over this IR
 * ═══════════════════════════════════════════════════════════════ *)

module Node : Schedule.SCHED_NODE with type payload = cx_kind and type t = t = struct
  type payload = cx_kind

  type nonrec t = t =
    { tag : int
    ; node : payload
    }

  let preds (e : t) : t list =
    match e.node with
    | CIn _ -> []
    | CRotNI a | CRotPI a -> [ a ]
    | CAdd (a, b) | CSub (a, b) -> [ a; b ]
    | CFmaC (_, x, e) | CFnmaC (_, x, e) -> [ x; e ]
    | CTwC (_, _, x) | CTwV (_, x) | CTwL (_, x) -> [ x ]
  ;;

  (* Cycle costs, same convention as schedule.ml's real-valued table.
     CRotNI is a shuffle + xor (both ~1c, dependent) — charged as add
     latency, matching how NK_Neg (a sign-flip xor) is charged. CTwC is a
     mul + fma chain, dominated by fma latency, exactly like NK_Cmul*. *)
  let latency (uarch : Uarch.t) (e : t) : int =
    match e.node with
    | CIn _ -> uarch.load_l1_latency
    | CAdd _ | CSub _ -> uarch.add_latency
    | CRotNI _ | CRotPI _ -> uarch.add_latency
    | CFmaC _ | CFnmaC _ -> uarch.fma_latency
    | CTwC _ | CTwV _ -> uarch.fma_latency
    | CTwL _ ->
      (* table load feeds the same mul+fma chain; the load is off the
         critical path in steady state, so charge the arithmetic. *)
      uarch.fma_latency
  ;;

  let is_load (e : t) =
    match e.node with
    | CIn _ -> true
    | _ -> false
  ;;

  (* No standalone const nodes: real coefficients ride inside CFmaC/CTwC as
     emit-time set1/VLIT operands, so there is nothing for the lookahead
     leaf policy to defer. *)
  let is_const (_ : t) = false

  let kind_char (e : t) =
    match e.node with
    | CIn _ -> 'L'
    | CAdd _ | CSub _ -> 'A'
    | CRotNI _ | CRotPI _ -> 'R'
    | CFmaC _ | CFnmaC _ -> 'F'
    | CTwC _ | CTwV _ -> 'X'
    | CTwL _ -> 'T'
  ;;
end

module Sched = Schedule.Make (Node)

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

let rec dft_cx ?(sign = `Fwd) (n : int) (xs : t array) : t array =
  if n = 1
  then xs
  else (
    let h = n / 2 in
    let e = dft_cx ~sign h (Array.init h (fun i -> xs.(2 * i)))
    and o = dft_cx ~sign h (Array.init h (fun i -> xs.((2 * i) + 1))) in
    let out = Array.make n xs.(0) in
    for k = 0 to h - 1 do
      let a, b = butterfly_pair ~sign ~n ~k e.(k) o.(k) in
      out.(k) <- a;
      out.(k + h) <- b
    done;
    out)
;;

(* ═══════════════════════════════════════════════════════════════
 *  EMISSION
 * ═══════════════════════════════════════════════════════════════ *)

(* Distinct CTwC constants become file-scope VLIT vectors (cos-broadcast +
 * sign-folded sin), so a general twiddle costs no runtime broadcast: the
 * BYTW2 shape is fmadd(_ZWn_c, x, mul(_ZWn_s, cflip x)). *)
(* A VLIT twiddle constant: one (cos, sin) per complex lane. A broadcast
   constant (CTwC) is just the degenerate case where every lane matches. *)
type consts = (string, string * (float * float) array) Hashtbl.t

let const_name_v (tbl : consts) (w : (float * float) array) : string =
  let key =
    String.concat "_" (Array.to_list (Array.map (fun (c, s) -> Printf.sprintf "%.17g:%.17g" c s) w))
  in
  match Hashtbl.find_opt tbl key with
  | Some (n, _) -> n
  | None ->
    let n = Printf.sprintf "_ZW%d" (Hashtbl.length tbl) in
    Hashtbl.add tbl key (n, w);
    n
;;

let const_name (tbl : consts) (lanes : int) (c : float) (s : float) : string =
  const_name_v tbl (Array.make lanes (c, s))
;;

let emit_const_decls (isa : Isa.t) (tbl : consts) : string =
  let b = Buffer.create 256 in
  let items =
    Hashtbl.fold (fun _ v acc -> v :: acc) tbl []
    |> List.sort (fun (a, _) (b, _) -> compare a b)
  in
  List.iter
    (fun (n, w) ->
       (* cos duplicated across each complex; sin sign-folded as [-s, +s] per
          complex so the cflip'd product lands with the right signs for
          (a+bi)(c+is). One (c,s) PER LANE — a broadcast constant is just the
          case where every lane is equal. *)
       let cos_lanes =
         String.concat
           ", "
           (Array.to_list (Array.map (fun (c, _) -> Printf.sprintf "%.17g, %.17g" c c) w))
       in
       let sin_lanes =
         String.concat
           ", "
           (Array.to_list
              (Array.map (fun (_, s) -> Printf.sprintf "%.17g, %.17g" (-.s) s) w))
       in
       Buffer.add_string
         b
         (Printf.sprintf "static const %s %s_c = { %s };\n" isa.Isa.vec_type n cos_lanes);
       Buffer.add_string
         b
         (Printf.sprintf "static const %s %s_s = { %s };\n" isa.Isa.vec_type n sin_lanes))
    items;
  Buffer.contents b
;;

(* Render one scheduled node as a C initializer expression. *)
let render (isa : Isa.t) (tbl : consts) (e : t) : string =
  let v (x : t) = Printf.sprintf "z%d" x.tag in
  match e.node with
  | CIn _ -> failwith "codelet_cil.render: CIn is emitted by the load edge"
  | CAdd (a, b) -> Isa.add_pd isa (v a) (v b)
  | CSub (a, b) -> Isa.sub_pd isa (v a) (v b)
  | CRotNI a -> Isa.xor_mask_pd isa (Isa.cflip_pd isa (v a)) "_M_IM"
  | CRotPI a -> Isa.xor_mask_pd isa (Isa.cflip_pd isa (v a)) "_M_RE"
  | CFmaC (c, x, acc) ->
    Isa.fmadd_pd isa (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)) (v x) (v acc)
  | CFnmaC (c, x, acc) ->
    Isa.fnmadd_pd isa (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)) (v x) (v acc)
  | CTwC (c, s, x) ->
    let w = const_name tbl (isa.Isa.vec_width / 2) c s in
    Isa.fmadd_pd
      isa
      (w ^ "_c")
      (v x)
      (Isa.mul_pd isa (w ^ "_s") (Isa.cflip_pd isa (v x)))
  | CTwV (ws, x) ->
    let w = const_name_v tbl ws in
    Isa.fmadd_pd
      isa
      (w ^ "_c")
      (v x)
      (Isa.mul_pd isa (w ^ "_s") (Isa.cflip_pd isa (v x)))
  | CTwL (leg, x) ->
    (* BYTW2 against the streamed VTW2 record for this leg. *)
    let off = (leg - 1) * 2 * isa.Isa.vec_width in
    let c = Isa.loadu_pd isa (Printf.sprintf "twp[%d]" off)
    and s = Isa.loadu_pd isa (Printf.sprintf "twp[%d]" (off + isa.Isa.vec_width)) in
    Isa.fmadd_pd isa c (v x) (Isa.mul_pd isa s (Isa.cflip_pd isa (v x)))
;;

(* ═══════════════════════════════════════════════════════════════
 *  KINDS
 *
 *   N1   — solo / monolithic leaf: leg-major z in, leg-major z out.
 *   N1T  — bailey2 stage-1 leaf: same butterfly, but the four-step's
 *          TRANSPOSE is fused into the stores (corner-turn), so stage 2
 *          reads columns contiguously and no separate transpose pass is
 *          needed. Output element (leg p, column k) lands at
 *          zout[2*(k*OLs + p)].
 *   T2   — bailey2 stage-2 mid: legs 1..R-1 pre-twiddled from the
 *          streamed VTW2 table (cursor advances one record-set per
 *          column-group), then the same butterfly. Leg-major stores.
 * ═══════════════════════════════════════════════════════════════ *)
type kind =
  | N1
  | N1T
  | T2

let kind_of_string = function
  | "n1" -> N1
  | "n1t" -> N1T
  | "t2" -> T2
  | s -> failwith (Printf.sprintf "codelet_cil: unknown kind %s (n1 | n1t | t2)" s)
;;

let kind_name = function
  | N1 -> "n1"
  | N1T -> "n1t"
  | T2 -> "t2"
;;

(* DIRECTION.
   n1 / n1t bwd are just the inverse butterfly (unnormalized: no 1/N).
   t2 bwd is the exact inverse of t2 fwd. Since fwd computes
     y = DFT(w (.) x)
   the inverse is
     x = conj(w) (.) IDFT(y)
   i.e. the twiddle moves to AFTER the butterfly and is conjugated. We do NOT
   conjugate in-kernel: the caller passes a conjugated table (the same
   table-side convention the split cascade uses for twspb/twqb), so the BYTW2
   apply is bit-for-bit the forward one and only its POSITION changes. *)
type dir =
  | Fwd
  | Bwd

(* Emit a solo (monolithic, twiddle-free) interleaved n1 codelet.
   ABI: the frozen 11-arg z ABI shared with codelet_zil, so emitted files
   are drop-in against the same benches/drivers. *)
let emit ~(kind : kind) ~(dir : dir) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t)
  : string
  =
  let vw = isa.Isa.vec_width in
  if vw mod 2 <> 0 then failwith "codelet_cil: interleaved needs an even vec_width";
  let per = vw / 2 in
  (* complex per vector *)
  if kind = N1T && per <> 2
  then
    (* The corner-turn store pairs two legs with one permute2f128 (a
       2-complex-per-vector shape). A width-8 vector holds 4 complex and
       needs a 4-way lane shuffle instead — not written yet. *)
    failwith "codelet_cil: n1t corner-turn store is written for 2 complex/vector (avx2)";
  (* RADIX GATE — dft_cx is a DIT-RADIX-2 recursion: it splits n into n/2
     even + n/2 odd, so it is only valid for powers of two. For an odd n it
     would silently DROP the last element (n=3 -> h=1 covers legs 0,1 and
     never writes out[2]), emitting plausible-looking but WRONG code. Odd
     and prime radices need a different complex construction (Winograd /
     Rader at the complex level) — see zil_pipeline_port.md §11.5. Until
     then, fail loudly rather than emit garbage. *)
  if radix < 2 || radix land (radix - 1) <> 0
  then
    failwith
      (Printf.sprintf
         "codelet_cil: radix %d unsupported — the complex DIT recursion is \
          radix-2 (powers of two only). Odd/prime radices need a complex \
          Winograd/Rader construction, not yet written."
         radix);
  reset ();
  let sign = if dir = Fwd then `Fwd else `Bwd in
  let pre_tw = kind = T2 && dir = Fwd in
  let post_tw = kind = T2 && dir = Bwd in
  let inputs =
    Array.init radix (fun i ->
      (* T2 fwd PRE-twiddles legs 1..R-1 from the streamed table; leg 0 is
         untwiddled (w^0 = 1), which is why records start at leg 1. *)
      if pre_tw && i > 0 then ctwl i (cin i) else cin i)
  in
  let outs = dft_cx ~sign radix inputs in
  (* T2 bwd POST-twiddles: conj(w) (.) IDFT(y). Same BYTW2 apply, same table
     slots, just after the butterfly — see the `dir` note above. *)
  let outs =
    if post_tw then Array.mapi (fun i e -> if i > 0 then ctwl i e else e) outs else outs
  in
  (* Label outputs with Expr.elem_ref so the shared scheduler can identify
     sinks; only the index is meaningful here (one complex output per leg). *)
  let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs) in
  let scheduled = Sched.su_schedule uarch assigns in
  let tbl : consts = Hashtbl.create 16 in
  (* Render the body first: it populates the constant table that the file
     preamble must declare. *)
  let body = Buffer.create 4096 in
  (* ─── ONE SCHEDULED PASS ──────────────────────────────────────────
     Build a sub-DAG, run it through the SHARED scheduler, and emit
     loads / defs / stores for it. Used by the BLOCKED construction,
     which is several passes joined through a spill array.

     Each pass calls `reset ()`, so tag numbering (and therefore the
     `zN` variable names) RESTARTS — every pass must therefore be
     emitted inside its own C brace scope, and fully emitted before the
     next pass begins. Same discipline as sterm2's two DAGs. *)
  let emit_pass
        ~(label : string)
        ~(nin : int)
        ~(load_of : int -> string)
        ~(build : t array -> t array)
        ~(store : int -> string -> unit)
    : unit
    =
    reset ();
    let ins = Array.init nin cin in
    let outs = build ins in
    let assigns =
      Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs)
    in
    let sch = Sched.su_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    Array.iteri
      (fun i (e : t) ->
         Buffer.add_string
           body
           (Printf.sprintf
              "        %s\n"
              (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (load_of i))))
      ins;
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             Buffer.add_string
               body
               (Printf.sprintf
                  "        %s\n"
                  (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (render isa tbl e)))))
      sch;
    Array.iteri (fun i (e : t) -> store i (Printf.sprintf "z%d" e.tag)) outs;
    Buffer.add_string body "        }\n"
  in
  ignore emit_pass;
  let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun ((_ : Expr.elem_ref option), (e : t)) ->
       match e.node with
       | CIn _ -> () (* materialized by the load edge *)
       | _ ->
         if not (Hashtbl.mem seen e.tag)
         then (
           Hashtbl.replace seen e.tag ();
           Buffer.add_string
             body
             (Printf.sprintf
                "        %s\n"
                (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (render isa tbl e)))))
    scheduled;
  let buf = Buffer.create 8192 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (codelet_cil.ml; docs/roadmap/zil_pipeline_port.md §11).\n\
       \ * radix-%d %s: %d complex per %d-bit vector. Body scheduled by the\n\
       \ * SHARED SR scheduler (Schedule.Make over the complex IR) and rendered\n\
       \ * through the ISA layer, so the same source emits AVX2 / AVX-512.\n\
       \ * %s\n\
       \ * CONTRACT: count %% %d == 0 (%d columns per iteration). */\n"
       radix
       (match kind with
        | N1 -> "solo n1 (natural order in/out, twiddle-free)"
        | N1T -> "bailey2 stage-1 leaf n1t (four-step TRANSPOSE fused into the stores)"
        | T2 -> "bailey2 stage-2 mid t2 (streamed VTW2 twiddles, BYTW2 apply)")
       per
       (vw * 64)
       (match kind with
        | N1 -> "tw_re/tw_im unused."
        | N1T ->
          "Stores are corner-turned: output (leg p, column k) -> zout[2*(k*OLs + p)],\n\
          \ * so stage 2 reads whole columns contiguously and no separate transpose\n\
          \ * pass is needed. tw_re/tw_im unused."
        | T2 ->
          Printf.sprintf
            "tw_re = VTW2 records, cos-first and sign-folded: per column-group, per\n\
            \ * leg 1..R-1, one %d-double record [c x%d][-s,+s ...]. Cursor advances\n\
            \ * %d doubles per group; BYTW2 = fmadd(c, x, mul(s, cflip x)) — ONE\n\
            \ * data-side shuffle, zero table-side work. tw_im unused."
            (2 * vw)
            vw
            ((radix - 1) * 2 * vw))
       per
       per);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  (* Only the quarter-turn mask this direction actually uses — emitting both
     would leave an unused static const (warning noise). *)
  if dir = Fwd
  then (
    Buffer.add_string buf (Isa.im_mask_decl isa "_M_IM");
    Buffer.add_string buf "  /* negate im lanes: x*(-i) */\n")
  else (
    Buffer.add_string buf (Isa.re_mask_decl isa "_M_RE");
    Buffer.add_string buf "  /* negate re lanes: x*(+i) */\n");
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string
    buf
    (Printf.sprintf
       "\n\
        __attribute__((target(\"%s\")))\n\
        void radix%d_z_%s_%s_%s(\n\
       \    const double * __restrict__ zin,\n\
       \    const double * __restrict__ zin_unused,\n\
       \    double       * __restrict__ zout,\n\
       \    double       * __restrict__ zout_unused,\n\
       \    const double * tw_re, const double * tw_im,\n\
       \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
        {\n\
       \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OGs;%s\n"
       isa.Isa.target_attr
       radix
       (kind_name kind)
       (if dir = Fwd then "fwd" else "bwd")
       isa.Isa.name
       (if kind = T2 then "" else " (void)tw_re;"));
  Buffer.add_string
    buf
    (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" per per);
  (* T2's streamed cursor: one record-set per column-group. *)
  if kind = T2
  then
    Buffer.add_string
      buf
      (Printf.sprintf
         "        const double *twp = tw_re + (k / %d) * (size_t)%d;\n"
         per
         ((radix - 1) * 2 * vw));
  (* load edge: one packed-complex vector per leg *)
  for l = 0 to radix - 1 do
    Buffer.add_string
      buf
      (Printf.sprintf
         "        %s\n"
         (Isa.const_decl
            isa
            (Printf.sprintf "z%d" (cin l).tag)
            (Isa.loadu_pd isa (Printf.sprintf "zin[2*((size_t)%d*Ls + k)]" l))))
  done;
  Buffer.add_buffer buf body;
  (* store edge *)
  (match kind with
   | N1 | T2 ->
     (* leg-major: leg l's `per` columns stay contiguous *)
     Array.iteri
       (fun l (e : t) ->
          Buffer.add_string
            buf
            (Printf.sprintf
               "        %s;\n"
               (Isa.storeu_pd
                  isa
                  (Printf.sprintf "zout[2*((size_t)%d*OLs + k)]" l)
                  (Printf.sprintf "z%d" e.tag))))
       outs
   | N1T ->
     (* CORNER-TURN (the four-step transpose, fused into the stores).
        Each output vector holds one leg's 2 columns: out_p = [c_k, c_{k+1}].
        Pairing legs p,p+1 and swapping 128-bit lanes regroups them into
        [leg p, leg p+1] of ONE column — so column k's legs land
        contiguously at zout[2*(k*OLs + p)]. Two stores per leg-pair, both
        full-width: no scalar tail, no separate transpose pass. *)
     let p2f = Isa.intr isa "permute2f128_pd" in
     let n = Array.length outs in
     let l = ref 0 in
     while !l < n do
       let a = Printf.sprintf "z%d" outs.(!l).tag
       and b = Printf.sprintf "z%d" outs.(!l + 1).tag in
       Buffer.add_string
         buf
         (Printf.sprintf
            "        %s;\n        %s;\n"
            (Isa.storeu_pd
               isa
               (Printf.sprintf "zout[2*((size_t)k*OLs + %d)]" !l)
               (Printf.sprintf "%s(%s, %s, 0x20)" p2f a b))
            (Isa.storeu_pd
               isa
               (Printf.sprintf "zout[2*(((size_t)k + 1)*OLs + %d)]" !l)
               (Printf.sprintf "%s(%s, %s, 0x31)" p2f a b)));
       l := !l + 2
     done);
  Buffer.add_string buf "    }\n}\n";
  Buffer.contents buf
;;

(* ═══════════════════════════════════════════════════════════════
 *  FUSED K=1 — the whole N-point transform as ONE function
 *
 * This is the IL-NATIVE shape, and it is deliberately NOT built by calling
 * the n1t/t2 codelets in sequence. Those carry the split family's staged
 * vocabulary — an 11-arg batched ABI, a memory plane crossed between two
 * function calls, and a RUNTIME twiddle table. At K=1 with N fixed at
 * generation time none of that is wanted:
 *
 *   - MKL's own K=1 interleaved path is ONE function
 *     (docs/research/mkl_highN_cascade_anatomy.md: "the whole 2^k K=1
 *     cascade is ONE function", in-place on a contiguous plane);
 *   - every twiddle is a COMPILE-TIME constant when N is known, so a
 *     runtime table is pure waste — they become file-scope VLITs (CTwV);
 *   - the stage boundary is a register transpose, not an ABI crossing.
 *
 * STRUCTURE — four-step over N = n1*n2, both stages inside one function.
 * Index input as x[j1*n2 + j2] and output as X[k2*n1 + k1].
 *   stage A : for each PAIR of columns (j2, j2+1) — adjacent complexes, so
 *             one vector load per leg — run DFT_n1 over j1 lane-wise. Park
 *             to a function-scope plane P[k1][c].
 *   turn    : two permute2f128 per (k1-pair, c) regroup lanes from
 *             "two j2 of one k1" to "two k1 of one j2" — the four-step
 *             transpose, in REGISTERS.
 *   stage B : multiply by w_N^{k1*j2} (per-lane VLIT, since the two lanes
 *             carry k1 = 2d and 2d+1) and run DFT_n2 over j2. Stores land
 *             at complex k2*n1 + 2d — adjacent, so full-width, and that
 *             address IS natural order. No output permutation, ever.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_k1 ~(dir : dir) ~(n : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let vw = isa.Isa.vec_width in
  if vw <> 4
  then failwith "codelet_cil: fused K=1 is written for 2 complex/vector (avx2)";
  if n < 4 || n land (n - 1) <> 0
  then
    failwith
      (Printf.sprintf "codelet_cil: fused K=1 needs a power of two >= 4 (got %d)" n);
  (* Squarest split keeps both stages' register pressure down. *)
  let best = ref 0 in
  for c = 2 to 64 do
    if c land (c - 1) = 0 && n mod c = 0
    then (
      let o = n / c in
      if o >= 2 && o <= 64 && o land (o - 1) = 0
      then if !best = 0 || min c o > min !best (n / !best) then best := c)
  done;
  if !best = 0
  then
    failwith
      (Printf.sprintf "codelet_cil: N=%d does not factor into two radices <= 64" n);
  let n1 = !best in
  let n2 = n / n1 in
  let sign = if dir = Fwd then `Fwd else `Bwd in
  let sgn = if dir = Fwd then -1.0 else 1.0 in
  let pi = 4.0 *. atan 1.0 in
  let tbl : consts = Hashtbl.create 64 in
  let body = Buffer.create 16384 in
  (* One scheduled sub-DAG. Tags restart per call, so each gets its own C
     brace scope; `pre` emits glue (the register transpose) inside it. *)
  let pass
        ~(label : string)
        ~(nin : int)
        ~(pre : unit -> unit)
        ~(load_of : int -> string)
        ~(build : t array -> t array)
        ~(store : int -> string -> unit)
    : unit
    =
    reset ();
    let ins = Array.init nin cin in
    let outs = build ins in
    let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs) in
    let sch = Sched.su_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "    { /* %s */\n" label);
    pre ();
    Array.iteri
      (fun i (e : t) ->
         Buffer.add_string
           body
           (Printf.sprintf
              "    %s\n"
              (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (load_of i))))
      ins;
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             Buffer.add_string
               body
               (Printf.sprintf
                  "    %s\n"
                  (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (render isa tbl e)))))
      sch;
    Array.iteri (fun i (e : t) -> store i (Printf.sprintf "z%d" e.tag)) outs;
    Buffer.add_string body "    }\n"
  in
  (* ── stage A: DFT_n1 down each column pair, park to the plane ── *)
  for c = 0 to (n2 / 2) - 1 do
    pass
      ~label:
        (Printf.sprintf "stage A: columns j2=%d,%d -> P[k1][%d]" (2 * c) ((2 * c) + 1) c)
      ~nin:n1
      ~pre:(fun () -> ())
      ~load_of:(fun j1 ->
        Isa.loadu_pd isa (Printf.sprintf "zin[%d]" (2 * ((j1 * n2) + (2 * c)))))
      ~build:(fun ins -> dft_cx ~sign n1 ins)
      ~store:(fun k1 v ->
        Buffer.add_string
          body
          (Printf.sprintf
             "    %s;\n"
             (Isa.storeu_pd isa (Printf.sprintf "P[%d]" (vw * ((k1 * (n2 / 2)) + c))) v)))
  done;
  (* ── stage B: register turn + per-lane constant twiddle + DFT_n2 ── *)
  let p2f = Isa.intr isa "permute2f128_pd" in
  for d = 0 to (n1 / 2) - 1 do
    pass
      ~label:
        (Printf.sprintf
           "stage B: k1=%d,%d -> X[k2*%d + %d]"
           (2 * d)
           ((2 * d) + 1)
           n1
           (2 * d))
      ~nin:n2
      ~pre:(fun () ->
        for c = 0 to (n2 / 2) - 1 do
          Buffer.add_string
            body
            (Printf.sprintf
               "    %s\n    %s\n"
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_a%d" c)
                  (Isa.loadu_pd isa (Printf.sprintf "P[%d]" (vw * ((2 * d * (n2 / 2)) + c)))))
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_b%d" c)
                  (Isa.loadu_pd
                     isa
                     (Printf.sprintf "P[%d]" (vw * ((((2 * d) + 1) * (n2 / 2)) + c))))));
          Buffer.add_string
            body
            (Printf.sprintf
               "    %s\n    %s\n"
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_t%d" (2 * c))
                  (Printf.sprintf "%s(_a%d, _b%d, 0x20)" p2f c c))
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_t%d" ((2 * c) + 1))
                  (Printf.sprintf "%s(_a%d, _b%d, 0x31)" p2f c c)))
        done)
      ~load_of:(fun j2 -> Printf.sprintf "_t%d" j2)
      ~build:(fun ins ->
        let tw =
          Array.mapi
            (fun j2 x ->
               if j2 = 0
               then x
               else
                 ctwv
                   (Array.init
                      (vw / 2)
                      (fun lane ->
                         let k1 = (2 * d) + lane in
                         let a =
                           sgn *. 2.0 *. pi *. float_of_int (k1 * j2) /. float_of_int n
                         in
                         cos a, sin a))
                   x)
            ins
        in
        dft_cx ~sign n2 tw)
      ~store:(fun k2 v ->
        Buffer.add_string
          body
          (Printf.sprintf
             "    %s;\n"
             (Isa.storeu_pd
                isa
                (Printf.sprintf "zout[%d]" (2 * ((k2 * n1) + (2 * d))))
                v)))
  done;
  let buf = Buffer.create 32768 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — FULL-IL FUSED K=1 (codelet_cil.ml).\n\
       \ * N=%d as ONE function: %d x %d four-step, both stages fused, the\n\
       \ * stage boundary is a REGISTER transpose (permute2f128), and every\n\
       \ * twiddle is a compile-time constant — no runtime table, no staged\n\
       \ * codelet ABI, no split conversion anywhere. Natural order in/out.\n\
       \ * Interior is interleaved end to end: %d complex per %d-bit vector. */\n\
        #include <immintrin.h>\n\
        #include <stddef.h>\n\n"
       n
       n1
       n2
       (vw / 2)
       (vw * 64));
  Buffer.add_string
    buf
    (if dir = Fwd
     then Isa.im_mask_decl isa "_M_IM" ^ "  /* x*(-i) */\n"
     else Isa.re_mask_decl isa "_M_RE" ^ "  /* x*(+i) */\n");
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string
    buf
    (Printf.sprintf
       "\n\
        __attribute__((target(\"%s\")))\n\
        void vfft_cil_%d_%s_%s(const double * __restrict__ zin,\n\
       \                              double * __restrict__ zout)\n\
        {\n\
       \    double P[%d];  /* stage-A results; L1-resident, never escapes.\n\
       \                       Flat doubles so &P[i] is the double* that the\n\
       \                       load/store intrinsics take. */\n"
       isa.Isa.target_attr
       n
       (if dir = Fwd then "fwd" else "bwd")
       isa.Isa.name
       (2 * n));
  Buffer.add_buffer buf body;
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;
