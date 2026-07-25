(* codelet_cil.ml — pipeline-hosted INTERLEAVED-COMPLEX (full-IL) codelets.
 *
 * The full-IL layout is a first-class product surface: MKL and FFTW default
 * to interleaved, IL pays no R2C/C2R conversion tax, and it makes K=1
 * (single-transform) natural. This module generates the interleaved family
 * — solo (monolithic n1) now, bailey2 (four-step leaf + t2 mid) next — the
 * way the rest of the generator works, instead of codelet_zil.ml's
 * hand-scheduled raw-string emission.
 *
 * WHAT IS SHARED WITH THE REST OF THE PIPELINE
 * --------------------------------------------
 *   Schedule.Make  — the SR (Starve-Retire) list scheduler, instantiated on
 *                    THIS module's complex IR via the SCHED_NODE signature.
 *                    Replaces codelet_zil's hand A/B interleaving; the same
 *                    scheduler that drives every split-real codelet.
 *   Isa.*          — every intrinsic is built through the ISA layer, so the
 *                    emitted code is width-parametric (AVX2 today, AVX-512
 *                    by passing a different Isa.t) instead of 486 literal
 *                    _mm256_ calls.
 *
 * WHAT IS DELIBERATELY *NOT* SHARED
 * ---------------------------------
 *   Ir / Algsimp / Dft. Those are REAL-VALUED by construction (the DAG has
 *   already split complex into re/im subtrees; `of_expr` hard-matches the
 *   split-cmul shape). Merging packed-complex kinds into Ir.node_kind was
 *   measured at ~150-180 exhaustive-match arms across 7 modules — most of
 *   them inside simplify.ml / fma_passes.ml, real-valued passes an IL kernel
 *   never runs. So this module keeps its own small complex IR. CSE is not
 *   lost: hash-consing here gives the same sharing Ir gets.
 *   See docs/roadmap/zil_pipeline_port.md §11.
 *
 * IL PRIMITIVES (the only ops with no real-lane equivalent)
 * ---------------------------------------------------------
 * A vector holds vec_width/2 complex as [re,im,re,im,...]:
 *   complex add/sub  = plain vector add/sub          (Isa.add_pd / sub_pd)
 *   c*x + e (real c) = plain FMA                     (Isa.fmadd_pd)
 *   x * (-i)         = xor(cflip x, _M_IM)           (Isa.cflip_pd + xor_mask)
 *   x * (c + i·s)    = fmadd(cvec, x, mul(svec, cflip x))   -- BYTW2 shape,
 *                      with cvec/svec emit-time constants for the n1 family.
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_cil.ml — grep "MODULE CARD" for the full set)
 * ROLE: interleaved-complex codelet emitter (solo n1; bailey2 next).
 * PIPELINE: cx math -> hash-cons -> Schedule.Make(Node).su_schedule ->
 * Isa-parametric emission.
 * PUBLIC SURFACE: emit_n1 (gen_main --cil-n1).
 * DEPS: Schedule (functor), Isa, Uarch, Expr (elem_ref labels only).
 * GOTCHA: `reset ()` MUST run before each codelet — the hash-cons table
 * and tag counter are module-global, exactly like Algsimp.reset.
 * ------------------------------------------------------------------ *)

(* ═══════════════════════════════════════════════════════════════
 *  THE COMPLEX IR
 * ═══════════════════════════════════════════════════════════════ *)

type cx_kind =
  | CIn of int (* input leg i (a packed-complex load) *)
  | CAdd of t * t
  | CSub of t * t
  | CRotNI of t (* x * (-i) *)
  | CFmaC of float * t * t (* c*x + e,  real scalar c *)
  | CFnmaC of float * t * t (* -c*x + e, real scalar c *)
  | CTwC of float * float * t (* x * (c + i*s), emit-time constants *)

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
let cfma c x e = mk (CFmaC (c, x, e))
let cfnma c x e = mk (CFnmaC (c, x, e))
let ctw c s x = mk (CTwC (c, s, x))

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
    | CRotNI a -> [ a ]
    | CAdd (a, b) | CSub (a, b) -> [ a; b ]
    | CFmaC (_, x, e) | CFnmaC (_, x, e) -> [ x; e ]
    | CTwC (_, _, x) -> [ x ]
  ;;

  (* Cycle costs, same convention as schedule.ml's real-valued table.
     CRotNI is a shuffle + xor (both ~1c, dependent) — charged as add
     latency, matching how NK_Neg (a sign-flip xor) is charged. CTwC is a
     mul + fma chain, dominated by fma latency, exactly like NK_Cmul*. *)
  let latency (uarch : Uarch.t) (e : t) : int =
    match e.node with
    | CIn _ -> uarch.load_l1_latency
    | CAdd _ | CSub _ -> uarch.add_latency
    | CRotNI _ -> uarch.add_latency
    | CFmaC _ | CFnmaC _ -> uarch.fma_latency
    | CTwC _ -> uarch.fma_latency
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
    | CRotNI _ -> 'R'
    | CFmaC _ | CFnmaC _ -> 'F'
    | CTwC _ -> 'X'
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

let rec dft_cx (n : int) (xs : t array) : t array =
  if n = 1
  then xs
  else (
    let h = n / 2 in
    let e = dft_cx h (Array.init h (fun i -> xs.(2 * i)))
    and o = dft_cx h (Array.init h (fun i -> xs.((2 * i) + 1))) in
    let out = Array.make n xs.(0) in
    let pi = 4.0 *. atan 1.0 in
    for k = 0 to h - 1 do
      let c = cos (2.0 *. pi *. float_of_int k /. float_of_int n)
      and s = -.sin (2.0 *. pi *. float_of_int k /. float_of_int n) in
      if k = 0
      then (
        out.(k) <- cadd e.(k) o.(k);
        out.(k + h) <- csub e.(k) o.(k))
      else if 4 * k = n
      then (
        let t = crot o.(k) in
        out.(k) <- cadd e.(k) t;
        out.(k + h) <- csub e.(k) t)
      else if 8 * k = n
      then (
        (* w = (1-i)/√2 : x = o + o·(-i), then ±√½·x + e *)
        let x = cadd o.(k) (crot o.(k)) in
        out.(k) <- cfma sqh x e.(k);
        out.(k + h) <- cfnma sqh x e.(k))
      else if 8 * k = 3 * n
      then (
        (* w = -(1+i)/√2 = √½·(rot(x) - x) *)
        let x = csub (crot o.(k)) o.(k) in
        out.(k) <- cfma sqh x e.(k);
        out.(k + h) <- cfnma sqh x e.(k))
      else (
        let t = ctw c s o.(k) in
        out.(k) <- cadd e.(k) t;
        out.(k + h) <- csub e.(k) t)
    done;
    out)
;;

(* ═══════════════════════════════════════════════════════════════
 *  EMISSION
 * ═══════════════════════════════════════════════════════════════ *)

(* Distinct CTwC constants become file-scope VLIT vectors (cos-broadcast +
 * sign-folded sin), so a general twiddle costs no runtime broadcast: the
 * BYTW2 shape is fmadd(_ZWn_c, x, mul(_ZWn_s, cflip x)). *)
type consts = (string, string * float * float) Hashtbl.t

let const_name (tbl : consts) (c : float) (s : float) : string =
  let key = Printf.sprintf "%.17g_%.17g" c s in
  match Hashtbl.find_opt tbl key with
  | Some (n, _, _) -> n
  | None ->
    let n = Printf.sprintf "_ZW%d" (Hashtbl.length tbl) in
    Hashtbl.add tbl key (n, c, s);
    n
;;

let emit_const_decls (isa : Isa.t) (tbl : consts) : string =
  let b = Buffer.create 256 in
  let lanes = isa.Isa.vec_width / 2 in
  let items =
    Hashtbl.fold (fun _ v acc -> v :: acc) tbl []
    |> List.sort (fun (a, _, _) (b, _, _) -> compare a b)
  in
  List.iter
    (fun (n, c, s) ->
       let rep x = String.concat ", " (List.init (lanes * 2) (fun _ -> x)) in
       (* cos broadcast; sin sign-folded as [-s, +s] per complex so the
          cflip'd product lands with the right signs for (a+bi)(c+is). *)
       let sin_lane =
         String.concat
           ", "
           (List.init lanes (fun _ -> Printf.sprintf "%.17g, %.17g" (-.s) s))
       in
       Buffer.add_string
         b
         (Printf.sprintf
            "static const %s %s_c = { %s };\n"
            isa.Isa.vec_type
            n
            (rep (Printf.sprintf "%.17g" c)));
       Buffer.add_string
         b
         (Printf.sprintf "static const %s %s_s = { %s };\n" isa.Isa.vec_type n sin_lane))
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
  | CFmaC (c, x, acc) ->
    Isa.fmadd_pd isa (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)) (v x) (v acc)
  | CFnmaC (c, x, acc) ->
    Isa.fnmadd_pd isa (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)) (v x) (v acc)
  | CTwC (c, s, x) ->
    let w = const_name tbl c s in
    Isa.fmadd_pd
      isa
      (w ^ "_c")
      (v x)
      (Isa.mul_pd isa (w ^ "_s") (Isa.cflip_pd isa (v x)))
;;

(* Emit a solo (monolithic, twiddle-free) interleaved n1 codelet.
   ABI: the frozen 11-arg z ABI shared with codelet_zil, so emitted files
   are drop-in against the same benches/drivers. *)
let emit_n1 ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let vw = isa.Isa.vec_width in
  if vw mod 2 <> 0 then failwith "codelet_cil: interleaved needs an even vec_width";
  let per = vw / 2 in
  (* complex per vector *)
  reset ();
  let outs = dft_cx radix (Array.init radix cin) in
  (* Label outputs with Expr.elem_ref so the shared scheduler can identify
     sinks; only the index is meaningful here (one complex output per leg). *)
  let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs) in
  let scheduled = Sched.su_schedule uarch assigns in
  let tbl : consts = Hashtbl.create 16 in
  (* Render the body first: it populates the constant table that the file
     preamble must declare. *)
  let body = Buffer.create 4096 in
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
       \ * radix-%d solo n1: %d complex per %d-bit vector, natural order in/out,\n\
       \ * twiddle-free. Body scheduled by the SHARED SR scheduler\n\
       \ * (Schedule.Make over the complex IR) and rendered through the ISA\n\
       \ * layer, so the same source emits AVX2 / AVX-512.\n\
       \ * CONTRACT: count %% %d == 0 (%d columns per iteration). */\n"
       radix
       per
       (vw * 64)
       per
       per);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string buf (Isa.im_mask_decl isa "_M_IM");
  Buffer.add_string buf "  /* negate im lanes: x*(-i) */\n";
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string
    buf
    (Printf.sprintf
       "\n\
        __attribute__((target(\"%s\")))\n\
        void radix%d_z_n1_fwd_%s(\n\
       \    const double * __restrict__ zin,\n\
       \    const double * __restrict__ zin_unused,\n\
       \    double       * __restrict__ zout,\n\
       \    double       * __restrict__ zout_unused,\n\
       \    const double * tw_re, const double * tw_im,\n\
       \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
        {\n\
       \    (void)zin_unused; (void)zout_unused; (void)tw_re; (void)tw_im; (void)Gs; \
        (void)OGs;\n"
       isa.Isa.target_attr
       radix
       isa.Isa.name);
  Buffer.add_string
    buf
    (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" per per);
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
    outs;
  Buffer.add_string buf "    }\n}\n";
  Buffer.contents buf
;;
