(* cx_render.ml — C-text rendering of scheduled cx nodes.
 * Split out of codelet_cil.ml (Phase 0, 2026-08-09, byte-identity gated).
 * MODULE CARD
 * ROLE: consts interning (VLIT twiddles) + emit_const_decls + render (one
 * node -> one C initializer, via the Isa string layer) + log3_plan +
 * emit_log3_prologue.
 * DEPS: Cx_ir (nodes + tw_log3 ref), Isa. Reads state refs at render time. *)

open Cx_ir

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
          case where every lane is equal. The C TYPE is chosen PER ENTRY from
          the lane count: the odd-count tail renders the same DAG at
          Isa.sse2, whose 1-lane constants land in this same table (distinct
          keys) and must declare as __m128d. Wide-only files are unchanged. *)
       let ty =
         if Array.length w * 2 = isa.Isa.vec_width
         then isa.Isa.vec_type
         else Isa.sse2.Isa.vec_type
       in
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
         (Printf.sprintf "static const %s %s_c = { %s };\n" ty n cos_lanes);
       Buffer.add_string
         b
         (Printf.sprintf "static const %s %s_s = { %s };\n" ty n sin_lanes))
    items;
  Buffer.contents b
;;

(* Render one scheduled node as a C initializer expression.
   ?tw_vw — the vec_width the runtime VTW2 TABLE was built for. Defaults to
   the render ISA's own width (byte-identical for every existing call). The
   odd-count tail renders at Isa.sse2 against a table laid out for the WIDE
   width, so it passes the wide width here: the record is already
   narrow-readable ([c,c] at off, [-s,+s] at off+tw_vw) — ONLY the address
   arithmetic must not shrink with the render width.
   ?msuf — suffix for the quarter-turn mask / log3 prologue names, so the
   narrow arm references its own __m128d twins (_M_IM_n / _wc%d_n). *)
let render ?(tw_vw = 0) ?(msuf = "") (isa : Isa.t) (tbl : consts) (e : t) : string =
  let twv = if tw_vw = 0 then isa.Isa.vec_width else tw_vw in
  let v (x : t) = Printf.sprintf "z%d" x.tag in
  match e.node with
  | CIn _ -> failwith "codelet_cil.render: CIn is emitted by the load edge"
  | CAdd (a, b) -> Isa.add_pd isa (v a) (v b)
  | CSub (a, b) -> Isa.sub_pd isa (v a) (v b)
  (* complex negation: flip both lanes' signs — same rendering the real side
     uses for NK_Neg (xor with a -0.0 broadcast; no named mask needed). *)
  | CNeg a -> Isa.xor_mask_pd isa (v a) (Isa.set1_pd_str isa "-0.0")
  | CRotNI a -> Isa.xor_mask_pd isa (Isa.cflip_pd isa (v a)) ("_M_IM" ^ msuf)
  | CRotPI a -> Isa.xor_mask_pd isa (Isa.cflip_pd isa (v a)) ("_M_RE" ^ msuf)
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
    (* BYTW2 against the VTW2 record for this leg. Under FLAT the record is
       loaded inline from the streamed cursor; under LOG3 it is a name bound
       by the prologue (loaded for power-of-two legs, DERIVED otherwise), so
       the flat path emits byte-identically to before. *)
    let c, s =
      if !tw_log3
      then Printf.sprintf "_wc%d%s" leg msuf, Printf.sprintf "_ws%d%s" leg msuf
      else (
        let off = (leg - 1) * 2 * twv in
        ( Isa.loadu_pd isa (Printf.sprintf "twp[%d]" off)
        , Isa.loadu_pd isa (Printf.sprintf "twp[%d]" (off + twv)) ))
    in
    Isa.fmadd_pd isa c (v x) (Isa.mul_pd isa s (Isa.cflip_pd isa (v x)))
;;

(* ── LOG3 twiddle sourcing for the streamed VTW2 table ───────────────────
 *
 * Mirrors dft.ml's TP_Log3 as a SUBSTITUTION: read only the power-of-two
 * legs from the table and derive the rest by complex multiplication, with
 * the SAME slot layout as flat (slot = leg-1), so one table serves both
 * policies and the kernels stay interchangeable. R=8 reads 3 records
 * instead of 7; R=64 reads 6 instead of 63.
 *
 * Derive-then-apply, never chain-apply: chaining x*W^p then *W^q would put
 * both multiplies on the DATA critical path. Here the derivation is
 * loop-invariant per column-group and sits off it entirely.
 *
 * THE FOLDED FORMAT DERIVES ITSELF. A VTW2 record is cos-broadcast
 * cp = [c,c,c,c] and SIGN-FOLDED sin sp = [-s,+s,-s,+s]. Then
 *   sp*sq = [(-sp)(-sq), (+sp)(+sq), ...] = [sp.sq x4]   -- signs cancel
 * so with cj = cp.cq - sp.sq and sj = cp.sq + sp.cq:
 *   _wc_j = cp*cq  - sp*sq     -> [cj x4]              (mul + fnmadd)
 *   _ws_j = cp*sq  + sp*cq     -> [-sj,+sj,-sj,+sj]    (mul + fmadd)
 * Four vector ops, NO shuffles, and the fold is preserved automatically —
 * no unpack/repack of the record is needed. *)
let log3_plan (radix : int) : (int * (int * int) option) list =
  let is_pow2 x = x > 0 && x land (x - 1) = 0 in
  let highest_pow2_le j =
    let rec go p = if p * 2 > j then p else go (p * 2) in
    go 1
  in
  let out = ref [] in
  (* Ascending j guarantees dependency order: p and q are both < j. *)
  for j = 1 to radix - 1 do
    if is_pow2 j
    then out := (j, None) :: !out
    else (
      let p = highest_pow2_le j in
      out := (j, Some (p, j - p)) :: !out)
  done;
  List.rev !out
;;

let emit_log3_prologue
      ?(tw_vw = 0) ?(msuf = "")
      (buf : Buffer.t) (isa : Isa.t) (radix : int) : unit =
  (* ?tw_vw / ?msuf as in `render`: the narrow tail re-binds its own
     _wc%d_n/_ws%d_n names at Isa.sse2 against the WIDE-geometry table. *)
  let vw = if tw_vw = 0 then isa.Isa.vec_width else tw_vw in
  let nload = ref 0 in
  List.iter
    (fun (j, src) ->
       let cj = Printf.sprintf "_wc%d%s" j msuf
       and sj = Printf.sprintf "_ws%d%s" j msuf in
       match src with
       | None ->
         incr nload;
         let off = (j - 1) * 2 * vw in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl isa cj (Isa.loadu_pd isa (Printf.sprintf "twp[%d]" off)))
              (Isa.const_decl
                 isa
                 sj
                 (Isa.loadu_pd isa (Printf.sprintf "twp[%d]" (off + vw)))))
       | Some (p, q) ->
         let cp = Printf.sprintf "_wc%d%s" p msuf
         and sp = Printf.sprintf "_ws%d%s" p msuf
         and cq = Printf.sprintf "_wc%d%s" q msuf
         and sq = Printf.sprintf "_ws%d%s" q msuf in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 cj
                 (Isa.fnmadd_pd isa sp sq (Isa.mul_pd isa cp cq)))
              (Isa.const_decl
                 isa
                 sj
                 (Isa.fmadd_pd isa cp sq (Isa.mul_pd isa sp cq)))))
    (log3_plan radix);
  Buffer.add_string
    buf
    (Printf.sprintf
       "        /* log3: %d of %d VTW2 records loaded, %d derived */\n"
       !nload
       (radix - 1)
       (radix - 1 - !nload))
;;
