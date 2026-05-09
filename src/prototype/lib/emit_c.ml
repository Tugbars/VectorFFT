(* emit_c.ml — naive AVX-512 C emitter for radix-N t1_dit codelets.
 *
 * Walks the hash-consed DAG in topological order, emits one __m512d
 * variable per node. No scheduling, no register allocation — GCC handles
 * those when compiling the resulting C.
 *
 * The output signature matches user's gen_radix4.py t1_dit form:
 *
 *   void radix<N>_t1_dit_fwd_avx512(
 *       const double *in_re,  const double *in_im,
 *       double       *out_re, double       *out_im,
 *       const double *tw_re,  const double *tw_im,
 *       size_t K)
 *
 * Inputs/outputs are split-complex with K-stride layout: element j's real
 * component is in_re[j*K + k], imag in_im[j*K + k]. Twiddles same layout.
 * Loop iterates k by 8 (AVX-512 vector width for double).
 *)

open Algsimp

(* === Topological sort of the DAG nodes ===
 *
 * We need to emit definitions in dependency order: a node's definition
 * must come AFTER the definitions of all its operands. Since hash-consing
 * assigns tags in construction order (bottom-up), sorting by tag gives
 * a valid topological order automatically. *)
let topo_sort_reachable (roots : t list) : t list =
  let seen = Hashtbl.create 256 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag e;
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg e1 -> visit e1
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
        visit a; visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        visit a; visit b; visit c; visit d
      | NK_Fma (a, b, c, _, _) ->
        visit a; visit b; visit c
    end
  in
  List.iter visit roots;
  let nodes = Hashtbl.fold (fun _ e acc -> e :: acc) seen [] in
  List.sort (fun a b -> compare a.tag b.tag) nodes

(* === Render a Load operation as C ===
 *
 * Inputs and twiddles use K-strided layout: the j-th element's vector
 * is at &arr[j*K + k]. We use _mm512_loadu_pd because we don't enforce
 * alignment in the emitted code (matches FFTW's safer default).
 *
 * The `in_place` flag changes the input buffer name from `in_re/in_im`
 * to `rio_re/rio_im` (matches user's t1_dit signature) and the stride
 * variable name from `K` to `ios` (matches user's signature).
 *
 * The `t1s` flag (scalar-broadcast twiddles) changes Twiddle loads from
 * vector strided loads (`_mm512_loadu_pd(&tw_re[j*me + k])`) to scalar
 * broadcasts (`_mm512_set1_pd(tw_re[j])`). t1s is for inner CT codelets
 * where all k iterations share the same twiddle set; the bench harness
 * passes a smaller twiddle array (n-1 scalars instead of (n-1)*me). *)
let render_load ~(isa : Isa.t) ~(in_place : bool) ~(t1s : bool)
    (r : Expr.elem_ref) : string =
  let in_buf is_re = match in_place, is_re with
    | true,  true  -> "rio_re"
    | true,  false -> "rio_im"
    | false, true  -> "in_re"
    | false, false -> "in_im"
  in
  let stride = if in_place then "ios" else "K" in
  let tw_stride = if in_place then "me" else "K" in
  match r with
  | Expr.Input (j, true)   -> Isa.loadu_pd isa (Printf.sprintf "%s[%d*%s + k]" (in_buf true) j stride)
  | Expr.Input (j, false)  -> Isa.loadu_pd isa (Printf.sprintf "%s[%d*%s + k]" (in_buf false) j stride)
  | Expr.Twiddle (j, true) ->
    if t1s then Isa.set1_pd_str isa (Printf.sprintf "tw_re[%d]" j)
    else Isa.loadu_pd isa (Printf.sprintf "tw_re[%d*%s + k]" j tw_stride)
  | Expr.Twiddle (j, false)->
    if t1s then Isa.set1_pd_str isa (Printf.sprintf "tw_im[%d]" j)
    else Isa.loadu_pd isa (Printf.sprintf "tw_im[%d*%s + k]" j tw_stride)
  | Expr.Output _ ->
    failwith "render_load: Output ref shouldn't appear as a Load source"

(* === Render a single node's definition as C ===
 *
 * Each node becomes a `const VECTYPE t<tag> = <expr>;` declaration.
 * For Cmul nodes we expand to the underlying FMA arithmetic at emit time
 * (the math layer kept Cmul opaque to protect it from reassoc, but the
 * emitter is the right place to lower it back to vector instructions).
 *
 * Cmul.re(xr, xi, wr, wi) = xr*wr - xi*wi
 *   FMA form: vfnmadd(xi, wi, mul(xr, wr))   -- one mul + one fnmadd
 *
 * Cmul.im(xr, xi, wr, wi) = xr*wi + xi*wr
 *   FMA form: vfmadd(xr, wi, mul(xi, wr))    -- one mul + one fmadd
 *
 * Both AVX-512 and AVX2 have FMA (target attr "avx2,fma"), so the
 * pattern is the same; only the intrinsic prefix differs.
 *)
(* Maximum recursion depth for single-use inlining.
 * Each level inlines one node into its consumer's expression. Single-use
 * values form a chain only as long as their predecessor chain (each node
 * is single-use to one consumer), so depth = N bounds the inlined chain
 * length to N nodes. We pick a value high enough to handle prime DFT
 * codelets (R=17 has 6-deep FMA chains + sums); the practical concern
 * is C source readability and compiler handling of long expressions,
 * not correctness. Multi-use nodes act as natural "stop" points. *)
let inline_max_depth = 32

let render_node_def
    ?(fused = false)
    ?(fused_muls : (int, unit) Hashtbl.t option = None)
    ?(inline_set : (int, unit) Hashtbl.t option = None)
    ~(isa : Isa.t) ~(in_place : bool) ~(t1s : bool) (e : t) : string =
  let v t = Printf.sprintf "t%d" t.tag in
  (* Helper: try to extract operands of a fused-Mul predecessor.
   * Returns Some (x, y) if `n` is a Mul tagged for FMA fusion;
   * the standalone definition of `n` will have been suppressed elsewhere. *)
  let as_fused_mul n =
    match fused_muls with
    | None -> None
    | Some tbl ->
      if not (Hashtbl.mem tbl n.tag) then None
      else
        match n.node with
        | NK_Mul (x, y) -> Some (x, y)
        | _ -> None
  in
  (* Should this node be inlined into its consumer's expression? *)
  let should_inline n =
    match inline_set with
    | None -> false
    | Some tbl -> Hashtbl.mem tbl n.tag
  in
  (* Render an operand. If single-use, inline its expression recursively
   * (up to depth limit). Otherwise, just emit `t<tag>` and rely on the
   * standalone declaration (which will be emitted elsewhere). *)
  let rec render_operand depth n =
    if depth >= inline_max_depth || not (should_inline n) then v n
    else render_inlined depth n
  and render_inlined depth n =
    (* Recursive case: inline this node's expression. Don't inline Loads
     * (their memory operand is fine, but inlining them duplicates loads)
     * or Cmul nodes (they'd require complex parenthesization for the
     * pseudo-FMA pair semantics). *)
    match n.node with
    | NK_Const c ->
      Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)
    | NK_Load _ -> v n     (* don't inline loads — keep named *)
    | NK_Neg inner ->
      (match inner.node with
       | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" (-. c))
       | _ -> Isa.xor_pd isa (render_operand (depth+1) inner)
                            (Isa.set1_pd_str isa "-0.0"))
    | NK_Add (a, b) ->
      (match as_fused_mul a with
       | Some (x, y) ->
         Isa.fmadd_pd isa (render_operand (depth+1) x)
                          (render_operand (depth+1) y)
                          (render_operand (depth+1) b)
       | None ->
         match as_fused_mul b with
         | Some (x, y) ->
           Isa.fmadd_pd isa (render_operand (depth+1) x)
                            (render_operand (depth+1) y)
                            (render_operand (depth+1) a)
         | None ->
           Isa.add_pd isa (render_operand (depth+1) a)
                          (render_operand (depth+1) b))
    | NK_Sub (a, b) ->
      (match as_fused_mul b with
       | Some (x, y) ->
         Isa.fnmadd_pd isa (render_operand (depth+1) x)
                           (render_operand (depth+1) y)
                           (render_operand (depth+1) a)
       | None ->
         match as_fused_mul a with
         | Some (x, y) ->
           Isa.fmsub_pd isa (render_operand (depth+1) x)
                            (render_operand (depth+1) y)
                            (render_operand (depth+1) b)
         | None ->
           Isa.sub_pd isa (render_operand (depth+1) a)
                          (render_operand (depth+1) b))
    | NK_Mul (a, b) ->
      Isa.mul_pd isa (render_operand (depth+1) a) (render_operand (depth+1) b)
    | NK_CmulRe _ | NK_CmulIm _ -> v n  (* don't inline cmul *)
    | NK_Fma (a, b, c, neg_mul, neg_add) ->
      let ra = render_operand (depth+1) a in
      let rb = render_operand (depth+1) b in
      let rc = render_operand (depth+1) c in
      (match neg_mul, neg_add with
       | false, false -> Isa.fmadd_pd  isa ra rb rc
       | false, true  -> Isa.fmsub_pd  isa ra rb rc
       | true,  false -> Isa.fnmadd_pd isa ra rb rc
       | true,  true  -> Isa.fnmsub_pd isa ra rb rc)
  in
  (* Operand renderer for THIS node's body — depth=0 meaning we're already
   * inside the body of `e`, so its operands start at depth=0 (and inline up
   * to inline_max_depth from there). *)
  let op = render_operand 0 in
  let body = match e.node with
    | NK_Const c ->
      Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)
    | NK_Load r -> render_load ~isa ~in_place ~t1s r
    | NK_Neg inner ->
      (* Neg(Const c) is a compile-time constant — emit as a single
       * broadcast of -c rather than a runtime XOR. *)
      (match inner.node with
       | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" (-. c))
       | _ ->
         Isa.xor_pd isa (op inner) (Isa.set1_pd_str isa "-0.0"))
    | NK_Add (a, b) ->
      (* Try to fuse: Add(Mul(x,y), b) → fmadd(x, y, b)
       *              Add(a, Mul(x,y)) → fmadd(x, y, a)
       * Prefer fusing the LEFT operand if both are fusable Muls
       * (arbitrary tie-break; only one Mul can fuse per Add). *)
      (match as_fused_mul a with
       | Some (x, y) -> Isa.fmadd_pd isa (op x) (op y) (op b)
       | None ->
         match as_fused_mul b with
         | Some (x, y) -> Isa.fmadd_pd isa (op x) (op y) (op a)
         | None -> Isa.add_pd isa (op a) (op b))
    | NK_Sub (a, b) ->
      (* Sub(a, b) = a - b. Two FMA fusion opportunities:
       *   Sub(Mul(x,y), b) → fmsub(x, y, b)   (x*y - b)
       *   Sub(a, Mul(x,y)) → fnmadd(x, y, a)  (a - x*y = -x*y + a)
       * Prefer fusing the second operand (subtracted Mul) since fnmadd
       * tends to produce slightly tighter scheduling on x86. *)
      (match as_fused_mul b with
       | Some (x, y) -> Isa.fnmadd_pd isa (op x) (op y) (op a)
       | None ->
         match as_fused_mul a with
         | Some (x, y) -> Isa.fmsub_pd isa (op x) (op y) (op b)
         | None -> Isa.sub_pd isa (op a) (op b))
    | NK_Mul (a, b) -> Isa.mul_pd isa (op a) (op b)
    | NK_CmulRe (xr, xi, wr, wi) ->
      Isa.fnmadd_pd isa (op xi) (op wi) (Isa.mul_pd isa (op xr) (op wr))
    | NK_CmulIm (xr, xi, wr, wi) ->
      Isa.fmadd_pd isa (op xr) (op wi) (Isa.mul_pd isa (op xi) (op wr))
    | NK_Fma (a, b, c, neg_mul, neg_add) ->
      (* (neg_mul ? -a*b : a*b) + (neg_add ? -c : c)
       *
       *   neg_mul=F, neg_add=F:  a*b + c       → fmadd
       *   neg_mul=F, neg_add=T:  a*b - c       → fmsub
       *   neg_mul=T, neg_add=F:  -a*b + c      → fnmadd
       *   neg_mul=T, neg_add=T:  -a*b - c      → fnmsub *)
      (match neg_mul, neg_add with
       | false, false -> Isa.fmadd_pd  isa (op a) (op b) (op c)
       | false, true  -> Isa.fmsub_pd  isa (op a) (op b) (op c)
       | true,  false -> Isa.fnmadd_pd isa (op a) (op b) (op c)
       | true,  true  -> Isa.fnmsub_pd isa (op a) (op b) (op c))
  in
  if fused then
    (* Plain assignment to outer-scope variable — no declarator. *)
    Printf.sprintf "        %s = %s;" (v e) body
  else
    Printf.sprintf "        %s" (Isa.const_decl isa (v e) body)

(* === Emit a complete codelet ===
 *
 * Two signatures supported:
 *
 * Out-of-place (in_place=false, the default):
 *   void NAME(in_re, in_im, out_re, out_im, tw_re, tw_im, K)
 *
 * In-place (in_place=true, matches user's hand-coded t1_dit):
 *   void NAME(rio_re, rio_im, tw_re, tw_im, ios, me)
 *   — rio_* serves as both input and output (same buffer)
 *   — ios is the stride between legs in the rio buffer
 *   — me is the batch size (= K), used as twiddle stride
 *
 * In-place is safe here because the topological sort places all input
 * loads at the top of the function body and all output stores at the
 * bottom — by the time any store fires, all loads have completed and
 * their results are in registers.
 *)
type scheduler =
  | Topological              (* sort reachable nodes by tag, flat emit *)
  | Bisection                (* Frigo's recursive bisection, flat emit *)
  | Annotated_topological    (* topological order + nested-block scopes (annotate.ml) *)
  | Annotated_bisection      (* bisection schedule + nested-block scopes *)
  | SU of Uarch.t            (* Sethi-Ullman list scheduler with µarch profile *)
  | Annotated_SU of Uarch.t  (* SU + nested blocks *)

(* === SINGLE-USE INLINING SET ===
 *
 * Compute the set of node tags that should be inlined at their consumer
 * rather than emitted as separate `const __m512d t<tag> = ...;`
 * declarations. Inlining matches FFTW hand-coded codelet style:
 *
 *   const __m512d t1 = _mm512_sub_pd(a, b);
 *   const __m512d t2 = _mm512_mul_pd(K, t1);
 *
 * vs the inlined form:
 *
 *   const __m512d t2 = _mm512_mul_pd(K, _mm512_sub_pd(a, b));
 *
 * Both compute the same value, but the second gives GCC a tighter SSA
 * form: t1 has no name, no scope, and its lifetime is implicit in the
 * outer expression. Empirically, hand-coded R=13 t1_dif uses ~120 nested
 * intrinsic call patterns; our linearized output uses ~24. The gap is
 * register pressure: every named intermediate is one more SSA value
 * GCC's allocator has to track. Inlining single-use values closes the
 * gap to hand parity on R=11/13/17 t1_dif (and helps DIT too).
 *
 * Criteria for inlining:
 *   - Use count is exactly 1 (the value flows to one consumer)
 *   - Not a Load (loads have memory operands; inlining duplicates them)
 *   - Not a Cmul (Cmul.re/Cmul.im share state via 2-instruction sequence)
 *   - Not a sink (sinks are output assignments)
 *   - Not in fused_muls (already suppressed by FMA fusion path)
 *
 * "Use count" = number of distinct nodes that reference this tag as a
 * predecessor PLUS 1 if the tag also appears as an output assignment
 * (the store counts as a use).
 *)
let compute_inline_set
    ?(fused_muls : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) : (int, unit) Hashtbl.t =
  let roots = List.map snd assigns in
  let nodes = topo_sort_reachable roots in
  (* Use count = how many other nodes reference this tag. *)
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let bump tag =
    let cur = try Hashtbl.find use_count tag with Not_found -> 0 in
    Hashtbl.replace use_count tag (cur + 1)
  in
  List.iter (fun n ->
    match n.node with
    | NK_Const _ | NK_Load _ -> ()
    | NK_Neg a -> bump a.tag
    | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
      bump a.tag; bump b.tag
    | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
      bump a.tag; bump b.tag; bump c.tag; bump d.tag
    | NK_Fma (a, b, c, _, _) ->
      bump a.tag; bump b.tag; bump c.tag
  ) nodes;
  (* Each output assignment also counts as a use. *)
  List.iter (fun (_, e) -> bump e.tag) assigns;
  (* Sinks: tags that are direct output assignments. Don't inline these
   * — they need a named t<tag> for the store to reference. *)
  let sink_tags : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  List.iter (fun (_, e) -> Hashtbl.replace sink_tags e.tag ()) assigns;
  let result = Hashtbl.create 256 in
  List.iter (fun n ->
    let count = try Hashtbl.find use_count n.tag with Not_found -> 0 in
    let is_sink = Hashtbl.mem sink_tags n.tag in
    let is_fused = match fused_muls with
      | None -> false
      | Some tbl -> Hashtbl.mem tbl n.tag
    in
    let kind_inlinable = match n.node with
      | NK_Load _ -> false       (* don't duplicate loads *)
      | NK_CmulRe _ | NK_CmulIm _ -> false  (* paired emit *)
      | NK_Const _ -> false      (* already inlined as set1 broadcast *)
      | _ -> true
    in
    if count = 1 && not is_sink && not is_fused && kind_inlinable then
      Hashtbl.add result n.tag ()
  ) nodes;
  result

(* === SPILL CONFIGURATION ===
 *
 * When ?spill is provided, emission produces a PASS-1/PASS-2 split:
 *   - PASS 1: compute everything up to and including the spill targets.
 *             Emit explicit stack-array stores for spill targets.
 *   - PASS 2: reload spill targets into a fresh nested scope; compute
 *             dependents; emit final output stores.
 *
 * The spill_table maps Algsimp tag → slot index. Two parallel arrays
 * are declared at function entry: spill_re[N] and spill_im[N] of vector
 * type. The re/im distinction comes from the tag being marked as either
 * a real or imaginary PASS 1 output.
 *
 * `num_slots` is the size of each array (max slot index + 1).
 *
 * `fused_slots` (if non-empty) marks spill slots whose values are kept
 * in registers across the PASS 1/PASS 2 boundary instead of being stored
 * to and reloaded from spill_re[] / spill_im[]. Set by make_spill_info
 * when ?ct=(n1,n2) and ?fuse=M are provided: fuses the M PASS 2 sub-DFTs
 * whose inputs are emitted LAST in PASS 1 (giving short lifetime extension).
 *
 * For CT(n1, n2): PASS 2 sub-DFT-n1 #k2 consumes slots {n1_idx*n2 + k2 :
 * n1_idx in 0..n1-1}. We fuse k2 in {n2-fuse..n2-1} since these correspond
 * to the LAST sub-DFT-n2 output positions in each PASS 1 sub-FFT — the
 * latest-emitted (and thus latest-stored) values, which are also the
 * first-consumed in PASS 2 emission order. *)
type spill_info = {
  re_slot: (int, int) Hashtbl.t;  (* re tag → slot *)
  im_slot: (int, int) Hashtbl.t;  (* im tag → slot *)
  num_slots: int;
  fused_slots: (int, unit) Hashtbl.t;  (* slots NOT spilled — kept in regs *)
  ct_n1: int;  (* PASS 1 sub-FFT count, 0 if not CT-decomposed *)
  ct_n2: int;  (* PASS 1 sub-FFT size, 0 if not CT-decomposed *)
}

let make_spill_info ?ct ?(fuse = 0) (markers : Algsimp.spill_tag_marker list) : spill_info =
  let re_slot = Hashtbl.create 64 in
  let im_slot = Hashtbl.create 64 in
  let max_slot = ref (-1) in
  List.iter (fun m ->
    Hashtbl.replace re_slot m.Algsimp.re_tag m.slot;
    Hashtbl.replace im_slot m.Algsimp.im_tag m.slot;
    if m.slot > !max_slot then max_slot := m.slot
  ) markers;
  let fused_slots = Hashtbl.create 16 in
  let ct_n1, ct_n2 = match ct with
    | Some (n1, n2) -> (n1, n2)
    | None -> (0, 0)
  in
  (match ct with
   | Some (n1, n2) when fuse > 0 ->
     let m = min fuse n2 in
     for k2 = n2 - m to n2 - 1 do
       for n1_idx = 0 to n1 - 1 do
         Hashtbl.replace fused_slots (n1_idx * n2 + k2) ()
       done
     done
   | _ -> ());
  { re_slot; im_slot; num_slots = !max_slot + 1; fused_slots; ct_n1; ct_n2 }

let is_spilled (sp : spill_info) (tag : int) : bool =
  Hashtbl.mem sp.re_slot tag || Hashtbl.mem sp.im_slot tag

let is_fused_slot (sp : spill_info) (slot : int) : bool =
  Hashtbl.mem sp.fused_slots slot

(* Is this tag's spill slot fused (kept in register, not stored)? *)
let is_fused_tag (sp : spill_info) (tag : int) : bool =
  match Hashtbl.find_opt sp.re_slot tag, Hashtbl.find_opt sp.im_slot tag with
  | Some s, _ | _, Some s -> is_fused_slot sp s
  | None, None -> false


(* Split a topologically-ordered list of nodes into PASS 1 and PASS 2.
 *
 * A node is PASS 2 iff it transitively depends on a spilled tag (i.e.,
 * some pred or pred-of-pred etc. is in the spill_table). A node is
 * PASS 1 if it doesn't, INCLUDING the spilled tags themselves (they
 * are the boundary, computed in PASS 1, then spilled before PASS 2).
 *
 * Walk in topological order so each node's preds have already been
 * classified by the time we reach it. *)
let classify_passes (sp : spill_info) (nodes : t list)
    : (int, [`Pass1 | `Pass2]) Hashtbl.t =
  let cls = Hashtbl.create 256 in
  let preds_of e = match e.node with
    | NK_Const _ | NK_Load _ -> []
    | NK_Neg a -> [a]
    | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> [a; b]
    | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) -> [a; b; c; d]
    | NK_Fma (a, b, c, _, _) -> [a; b; c]
  in
  List.iter (fun e ->
    if is_spilled sp e.tag then
      Hashtbl.add cls e.tag `Pass1
    else
      let pred_in_pass2 = List.exists (fun p ->
        match Hashtbl.find_opt cls p.tag with
        | Some `Pass2 -> true
        | _ -> false
      ) (preds_of e) in
      let pred_is_spilled = List.exists (fun p ->
        is_spilled sp p.tag
      ) (preds_of e) in
      if pred_in_pass2 || pred_is_spilled then
        Hashtbl.add cls e.tag `Pass2
      else
        Hashtbl.add cls e.tag `Pass1
  ) nodes;

  (* DIF post-multiply: Twiddle Loads (and log3 cmul derivations of them) have
   * no spill-slot ancestors, so the forward pass classifies them as Pass1.
   * But their CONSUMERS may be in Pass2 (cmul on PASS 2 outputs). C block
   * scoping means Pass1-emitted variables go out of scope before Pass2;
   * references would fail to compile.
   *
   * Backward pass: reclassify any Pass1 node whose consumers are exclusively
   * in Pass2 → push to Pass2. This handles DIF cleanly (Twiddle Loads, log3
   * cmul derivations) without changing DIT behavior (where consumers of
   * Loads/cmul derivations are pre-multiply ops in Pass1). *)
  let consumers : (int, t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun e ->
    List.iter (fun p ->
      let prev = try Hashtbl.find consumers p.tag with Not_found -> [] in
      Hashtbl.replace consumers p.tag (e :: prev)
    ) (preds_of e)
  ) nodes;
  (* Iterate to fixpoint: a node X may need reclassification once a node it
   * feeds (Y) gets reclassified, in case Y was the reason X stayed Pass1. *)
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter (fun e ->
      match Hashtbl.find_opt cls e.tag with
      | Some `Pass1 when not (is_spilled sp e.tag) ->
        let cs = try Hashtbl.find consumers e.tag with Not_found -> [] in
        if cs <> [] && List.for_all (fun c ->
          Hashtbl.find_opt cls c.tag = Some `Pass2
        ) cs then begin
          Hashtbl.replace cls e.tag `Pass2;
          changed := true
        end
      | _ -> ()
    ) nodes
  done;

  cls

let emit_codelet
    ?(in_place = false)
    ?(t1s = false)
    ?(scheduler = Topological)
    ?(isa = Isa.avx512)
    ?(gh = false)
    ?(bb_budget : float option = None)
    ?(spill : spill_info option = None)
    (assigns : (Expr.elem_ref * t) list)
    ~(name : string) : string =
  let buf = Buffer.create 4096 in
  Buffer.add_string buf "/* Auto-generated by vfft_v2 codelet generator. */\n";
  Buffer.add_string buf "#include <immintrin.h>\n";
  Buffer.add_string buf "#include <stddef.h>\n\n";
  Buffer.add_string buf (Printf.sprintf "__attribute__((target(\"%s\")))\n" isa.target_attr);
  Buffer.add_string buf (Printf.sprintf "void %s(\n" name);

  if in_place then begin
    Buffer.add_string buf "    double       * __restrict__ rio_re,\n";
    Buffer.add_string buf "    double       * __restrict__ rio_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    size_t ios,\n";
    Buffer.add_string buf "    size_t me)\n";
    Buffer.add_string buf "{\n";
    (* Spill array decl, OUTSIDE the for loop so it's allocated once *)
    (match spill with
     | None -> ()
     | Some sp ->
       Buffer.add_string buf (Printf.sprintf
         "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
       Buffer.add_string buf (Printf.sprintf
         "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf (Printf.sprintf "    for (size_t k = 0; k < me; k += %d) {\n" isa.vec_width)
  end else begin
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    size_t K)\n";
    Buffer.add_string buf "{\n";
    (match spill with
     | None -> ()
     | Some sp ->
       Buffer.add_string buf (Printf.sprintf
         "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
       Buffer.add_string buf (Printf.sprintf
         "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf (Printf.sprintf "    for (size_t k = 0; k < K; k += %d) {\n" isa.vec_width)
  end;

  let out_buf is_re = match in_place, is_re with
    | true,  true  -> "rio_re"
    | true,  false -> "rio_im"
    | false, true  -> "out_re"
    | false, false -> "out_im"
  in
  let stride = if in_place then "ios" else "K" in

  let emit_store buf oref e =
    match oref with
    | Expr.Output (k, true) ->
      Buffer.add_string buf "        ";
      Buffer.add_string buf
        (Isa.storeu_pd isa
           (Printf.sprintf "%s[%d*%s + k]" (out_buf true) k stride)
           (Printf.sprintf "t%d" e.tag));
      Buffer.add_string buf ";\n"
    | Expr.Output (k, false) ->
      Buffer.add_string buf "        ";
      Buffer.add_string buf
        (Isa.storeu_pd isa
           (Printf.sprintf "%s[%d*%s + k]" (out_buf false) k stride)
           (Printf.sprintf "t%d" e.tag));
      Buffer.add_string buf ";\n"
    | _ -> failwith "emit_codelet: assignment LHS must be an Output"
  in

  (* Spill-aware emission path. When ?spill is provided, take this path
   * regardless of scheduler choice — the spill structure imposes a strict
   * pass boundary that supersedes whatever ordering the scheduler would
   * pick across passes. Within each pass, we still emit in topological
   * order (matching Topological scheduler behavior).
   *
   * Currently only Topological scheduling within passes is supported.
   * SU + spill is straightforward to add later: run SU per-pass on the
   * filtered node lists. For first validation, Topo+spill is the priority. *)
  (match spill with
   | Some sp ->
     let roots = List.map snd assigns in
     let nodes = topo_sort_reachable roots in
     let cls = classify_passes sp nodes in

     (* === FMA FUSION ANALYSIS ===
      *
      * A NK_Mul node with exactly one consumer (an Add or Sub) can be
      * fused into that consumer as an FMA. This eliminates one mul
      * instruction and replaces an add/sub with an fmadd/fmsub/fnmadd.
      *
      * Hand-coded R=32 has 64 fmadd / 49 fmsub / 15 fnmadd vs our
      * 47 / 26 / 45 — Hand has ~10 more FMAs and ~10 fewer add/subs.
      * That's the gap we're closing here. *)
     let preds_of_general (e : t) : t list = match e.node with
       | NK_Const _ | NK_Load _ -> []
       | NK_Neg a -> [a]
       | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> [a; b]
       | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) -> [a; b; c; d]
       | NK_Fma (a, b, c, _, _) -> [a; b; c]
     in
     let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
     let bump_use tag =
       let cur = try Hashtbl.find use_count tag with Not_found -> 0 in
       Hashtbl.replace use_count tag (cur + 1)
     in
     List.iter (fun e ->
       List.iter (fun p -> bump_use p.tag) (preds_of_general e)
     ) nodes;
     (* Output stores also consume their root expressions. *)
     List.iter (fun (_, e) -> bump_use e.tag) assigns;

     (* A Mul is fusable if:
      *  - use_count = 1 (exactly one consumer)
      *  - that consumer is an Add or Sub (where FMA fits)
      *  - it is NOT a spill target (must be materializable as a value)
      *  - it is NOT a fused tag (cross-boundary survivor; needs a real def)
      *)
     let is_spill_target tag =
       Hashtbl.mem sp.re_slot tag || Hashtbl.mem sp.im_slot tag
     in
     let is_fused_pass1_tag tag =
       match Hashtbl.find_opt sp.re_slot tag, Hashtbl.find_opt sp.im_slot tag with
       | Some s, _ | _, Some s -> is_fused_slot sp s
       | None, None -> false
     in
     let fused_muls : (int, unit) Hashtbl.t = Hashtbl.create 64 in
     let _is_fusable_mul n =
       match n.node with
       | NK_Mul _ ->
         let count = try Hashtbl.find use_count n.tag with Not_found -> 0 in
         count = 1
         && not (is_spill_target n.tag)
         && not (is_fused_pass1_tag n.tag)
       | _ -> false
     in
     (* DISABLED: source-level FMA fusion was a wash — GCC already fuses
      * mul+add into FMA via -mfma at -O3 on contracted intrinsic patterns,
      * and our explicit fmadd emission turned out to slightly hurt in some
      * regimes by constraining GCC's instruction selection. Keeping the
      * machinery (used_count, is_fusable_mul) for future experiments,
      * but no Muls go into fused_muls — emission falls back to standalone
      * mul + add/sub everywhere, and GCC fuses these on its own.
      *
      * TODO: revisit at the assembly level — if we want to force specific
      * FMA variants (213 vs 231), we'd need to do that via inline asm or
      * by understanding GCC's variant-selection heuristic. *)
     ignore _is_fusable_mul;

     let fused_muls_opt = Some fused_muls in
     (* Constants are leaves (no predecessors) shared across passes via
      * hash-consing — a single NK_Const node may be referenced by both
      * PASS 1 and PASS 2 nodes (e.g. 1/√2 used in radix-8 internal
      * twiddles in PASS 1 and again in radix-4 internal twiddles in
      * PASS 2). To keep them in scope across both pass scopes, hoist
      * NK_Const declarations to the for-loop body top, BEFORE either
      * pass scope opens.
      *
      * Loads stay in their classified pass — they depend on the loop
      * variable `k` and are used only by their direct consumers. *)
     let is_const e = match e.node with NK_Const _ -> true | _ -> false in
     let const_nodes = List.filter is_const nodes in
     let pass1_nodes = List.filter (fun e ->
       (not (is_const e)) &&
       (match Hashtbl.find_opt cls e.tag with
        | Some `Pass1 -> true | _ -> false)
     ) nodes in
     let pass2_nodes = List.filter (fun e ->
       (not (is_const e)) &&
       (match Hashtbl.find_opt cls e.tag with
        | Some `Pass2 -> true | _ -> false)
     ) nodes in

     (* Split output assigns by where their value is computed.
      *
      * Output stores must be emitted in the same C scope where the value
      * is in scope. Pass 1 outputs (value computed in PASS 1, no spilled
      * dependencies) get their stores at the end of PASS 1's `{ ... }`
      * block; Pass 2 outputs at the end of PASS 2's block. The original
      * design assumed all outputs were Pass 2 (everything depended on
      * spilled intermediates), but composite codelets like R=32 t1_dit
      * have outputs whose entire dep chain is twiddled-input cmuls →
      * inner-DFT chains that DON'T cross the spill boundary. Without
      * splitting, those outputs were emitted as `_mm512_storeu_pd(..., t<N>)`
      * inside PASS 2's scope, but t<N> was declared in PASS 1's scope
      * which had already closed — undefined-reference compile errors. *)
     let pass1_assigns = List.filter (fun (_, e) ->
       Hashtbl.find_opt cls e.tag = Some `Pass1
     ) assigns in
     let pass2_assigns = List.filter (fun (_, e) ->
       Hashtbl.find_opt cls e.tag = Some `Pass2
     ) assigns in

     (* Helper: list (slot, tag) pairs sorted by slot for deterministic output.
      * Currently unused — deferred-reload path emits reloads on demand
      * rather than in slot order, but keep helper available. *)
     let _sorted_by_slot (h : (int, int) Hashtbl.t) : (int * int) list =
       Hashtbl.fold (fun tag slot acc -> (slot, tag) :: acc) h []
       |> List.sort (fun (s1, _) (s2, _) -> compare s1 s2)
     in

     (* Hoisted constants — emitted at for-loop body top, in scope everywhere. *)
     List.iter (fun e ->
       Buffer.add_string buf (render_node_def ~isa ~in_place ~t1s e);
       Buffer.add_char buf '\n'
     ) const_nodes;
     Buffer.add_char buf '\n';

     (* Tag → slot lookups (separate hashtables for re/im so we can tell
      * which spill array a tag belongs to). *)
     let lookup_re_slot tag = Hashtbl.find_opt sp.re_slot tag in
     let lookup_im_slot tag = Hashtbl.find_opt sp.im_slot tag in

     (* Fused tags: those whose slot is in fused_slots. These keep their
      * SSA values alive across the PASS 1 / PASS 2 boundary instead of
      * round-tripping through spill_re/spill_im. They need:
      *   1. Forward declaration at outer scope (before PASS 1 `{`)
      *   2. PASS 1 emission as assignment (no `__m512d` declarator)
      *   3. No spill store at end of PASS 1 emission for that tag
      *   4. No reload at start of PASS 2 for that slot
      * They remain accessible in PASS 2 by their original t<tag> name. *)
     let is_fused_tag tag =
       match lookup_re_slot tag, lookup_im_slot tag with
       | Some s, _ | _, Some s -> is_fused_slot sp s
       | None, None -> false
     in

     (* Forward-declare fused tags at outer scope. *)
     let fused_pass1_tags = List.filter_map (fun e ->
       if is_fused_tag e.tag then Some e.tag else None
     ) pass1_nodes in
     if fused_pass1_tags <> [] then begin
       let names = List.map (fun t -> Printf.sprintf "t%d" t) fused_pass1_tags in
       Buffer.add_string buf "        ";
       Buffer.add_string buf (Isa.forward_decl isa names);
       Buffer.add_char buf '\n';
       Buffer.add_char buf '\n'
     end;

     (* Block-sequential PASS 1 ordering.
      *
      * Plain topological (tag) order interleaves sub-FFT computations,
      * causing peak live-set in PASS 1 to be the sum of all in-flight
      * sub-FFTs' state. With CT(N1, N2), all N1 sub-FFTs are
      * independent, so they CAN be ordered block-sequentially
      * (sub-FFT 0 fully complete, then sub-FFT 1, ...). This is what
      * hand-coded does, and it keeps peak live = O(N2) instead of
      * O(N1*N2).
      *
      * For each PASS 1 node, we compute min_descendant_slot — the
      * smallest spill slot reachable from this node. Spill targets
      * have their own slot. Intermediates inherit from their
      * (forward) successors. Sorting nodes by (min_descendant_slot,
      * tag) clusters sub-FFTs:
      *   - Sub-FFT 0 owns slots 0..N2-1
      *   - Sub-FFT 1 owns slots N2..2N2-1
      *   - etc.
      * so all sub-FFT-0 nodes sort before sub-FFT-1 nodes.
      *
      * Within a sub-FFT, tag order is preserved so dependencies are
      * respected. Across sub-FFTs, there are no dependencies (CT
      * independence) — except for shared constants, which we already
      * hoisted outside both pass scopes. *)
     let preds_of e = match e.node with
       | NK_Const _ | NK_Load _ -> []
       | NK_Neg a -> [a]
       | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> [a; b]
       | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) -> [a; b; c; d]
       | NK_Fma (a, b, c, _, _) -> [a; b; c]
     in
     (* Build forward succs map for PASS 1 nodes only. *)
     let pass1_set = Hashtbl.create 256 in
     List.iter (fun e -> Hashtbl.add pass1_set e.tag ()) pass1_nodes;
     let succs : (int, int list) Hashtbl.t = Hashtbl.create 256 in
     List.iter (fun e ->
       List.iter (fun p ->
         if Hashtbl.mem pass1_set p.tag then begin
           let cur = try Hashtbl.find succs p.tag with Not_found -> [] in
           Hashtbl.replace succs p.tag (e.tag :: cur)
         end
       ) (preds_of e)
     ) pass1_nodes;
     (* Compute min_slot bottom-up by walking high-tag-first
      * (reverse topological — successors before predecessors). *)
     let min_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
     List.iter (fun e ->
       let my_slot =
         match lookup_re_slot e.tag, lookup_im_slot e.tag with
         | Some s, _ | _, Some s -> Some s     (* spill target *)
         | None,   None ->
           let s_tags = try Hashtbl.find succs e.tag with Not_found -> [] in
           let s_slots = List.filter_map
             (fun t -> Hashtbl.find_opt min_slot t) s_tags in
           (match s_slots with
            | [] -> None
            | _ -> Some (List.fold_left min max_int s_slots))
       in
       match my_slot with
       | Some s -> Hashtbl.add min_slot e.tag s
       | None -> ()
     ) (List.rev pass1_nodes);
     let pass1_blocked_topo = List.sort (fun a b ->
       let sa = try Hashtbl.find min_slot a.tag with Not_found -> max_int in
       let sb = try Hashtbl.find min_slot b.tag with Not_found -> max_int in
       if sa <> sb then compare sa sb
       else compare a.tag b.tag
     ) pass1_nodes in

     (* If scheduler is SU, replace tag-order WITHIN each sub-FFT cluster
      * with SU ordering. Cluster boundary = min_slot range corresponding
      * to one PASS 1 sub-FFT. CT(N1, N2): cluster k owns slots [k*N2, (k+1)*N2 - 1].
      * Sub-FFTs are mutually independent (CT property), so SU within a
      * cluster is safe — it cannot reorder across cluster boundaries.
      *
      * For non-CT cases (ct_n2 = 0), fall back to global tag order. *)
     let pass1_blocked = match scheduler with
       | SU uarch when sp.ct_n2 > 0 ->
         (* Group pass1_blocked_topo by cluster (sub-FFT index = min_slot / N2). *)
         let cluster_of_node e =
           match Hashtbl.find_opt min_slot e.tag with
           | Some s -> s / sp.ct_n2
           | None -> sp.ct_n1  (* unreachable nodes go to a fake cluster *)
         in
         (* Walk pass1_blocked_topo, splitting into runs of same cluster.
          * Within each run, run su_schedule_subset. *)
         let groups : (int * t list) list =
           let rec go acc current_cluster current_acc = function
             | [] ->
               (match current_acc with
                | [] -> List.rev acc
                | _ -> List.rev ((current_cluster, List.rev current_acc) :: acc))
             | n :: rest ->
               let c = cluster_of_node n in
               if c = current_cluster then
                 go acc current_cluster (n :: current_acc) rest
               else begin
                 let acc' = match current_acc with
                   | [] -> acc
                   | _ -> (current_cluster, List.rev current_acc) :: acc
                 in
                 go acc' c [n] rest
               end
           in
           match pass1_blocked_topo with
           | [] -> []
           | n :: rest -> go [] (cluster_of_node n) [n] rest
         in
         List.concat_map (fun (_cluster, group_nodes) ->
           (* Sinks for this cluster: spill targets within the cluster. *)
           let cluster_sinks = List.filter (fun e ->
             Hashtbl.mem sp.re_slot e.tag || Hashtbl.mem sp.im_slot e.tag
           ) group_nodes in
           if cluster_sinks = [] then group_nodes  (* no sinks → keep as-is *)
           else
             (match bb_budget with
              | None -> Schedule.su_schedule_subset uarch ~gh
                          ~subset:group_nodes ~sinks:cluster_sinks
              | Some t -> Bb.bb_schedule_subset uarch ~time_budget_sec:t
                            ~subset:group_nodes ~sinks:cluster_sinks)
         ) groups
       | _ -> pass1_blocked_topo
     in

     (* PASS 1 nested scope: emit block-sequentially with immediate spill.
      * For fused tags: emit as assignment (no declarator) to outer-scope
      * variable, and skip the spill store. *)
     Buffer.add_string buf "        {\n";
     List.iter (fun e ->
       (* Skip standalone emission of FMA-fused Mul nodes. *)
       if Hashtbl.mem fused_muls e.tag then ()
       else begin
         let fused = is_fused_tag e.tag in
         Buffer.add_string buf
           (render_node_def ~fused ~fused_muls:fused_muls_opt ~t1s
              ~isa ~in_place e);
         Buffer.add_char buf '\n';
         if not fused then begin
           (match lookup_re_slot e.tag with
            | Some slot ->
              Buffer.add_string buf (Printf.sprintf
                "            %s(&spill_re[%d], t%d);\n"
              isa.storeu_pd slot e.tag)
          | None -> ());
         (match lookup_im_slot e.tag with
          | Some slot ->
            Buffer.add_string buf (Printf.sprintf
              "            %s(&spill_im[%d], t%d);\n"
              isa.storeu_pd slot e.tag)
          | None -> ())
       end
       end
     ) pass1_blocked;
     (* Emit stores for Pass 1 outputs at end of PASS 1 — values are still
      * in scope here. Pass 2 outputs are stored later, inside PASS 2's
      * scope (per-cluster flush + safety net). *)
     List.iter (fun (lhs, e) -> emit_store buf lhs e) pass1_assigns;
     Buffer.add_string buf "        }\n";

     (* PASS 2 nested scope: deferred-reload emission.
      *
      * KEY INSIGHT: bulk-loading all 32 spilled values at PASS 2 top
      * forces 32 live values plus PASS 2's working set, often exceeding
      * the 32 ZMM register budget and causing GCC to re-spill internally
      * (~148 extra stack ops measured). Hand avoids this by loading each
      * spilled value just-in-time at first use.
      *
      * We replicate that here: walk PASS 2 in scheduled order, and for
      * each node, emit reloads of any not-yet-reloaded spilled predecessors
      * immediately before emitting the node itself. Each reload still
      * fires exactly once. This keeps peak live at PASS 2 manageable. *)
     Buffer.add_string buf "        {\n";
     (* PASS 2 emission: cluster by sub-DFT (when CT-decomposed), then SU
      * within each cluster. This matches Hand's structure: process sub-DFT
      * #0 fully (load 4 inputs, compute, store 4 outputs), then sub-DFT #1,
      * etc. Keeps peak live within PASS 2 around N1+working-set instead
      * of N1*N2+working-set. Removes most of the GCC re-spilling.
      *
      * Each PASS 2 sub-DFT #k2 (for k2 in 0..N2-1) consumes spill slots
      * {n1*N2 + k2 : n1 in 0..N1-1} and produces N1 outputs. We assign
      * each PASS 2 node to a cluster by computing the minimum spill slot
      * it transitively reads, then taking that mod N2.
      *
      * For non-CT (or no spill targets visible), fall back to flat SU
      * over the whole PASS 2. *)
     let cluster_of_pass2_node : (int, int) Hashtbl.t = Hashtbl.create 256 in
     if sp.ct_n2 > 0 then begin
       let min_input_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
       List.iter (fun e ->
         let direct_slot =
           match Hashtbl.find_opt sp.re_slot e.tag, Hashtbl.find_opt sp.im_slot e.tag with
           | Some s, _ | _, Some s -> Some s
           | None, None -> None
         in
         let pred_min =
           List.fold_left (fun acc p ->
             match Hashtbl.find_opt min_input_slot p.tag with
             | Some s -> (match acc with None -> Some s | Some a -> Some (min a s))
             | None -> acc
           ) None (preds_of_general e)
         in
         let my_min = match direct_slot, pred_min with
           | Some a, Some b -> Some (min a b)
           | Some a, None | None, Some a -> Some a
           | None, None -> None
         in
         (match my_min with
          | Some s -> Hashtbl.add min_input_slot e.tag s
          | None -> ())
       ) nodes;
       List.iter (fun e ->
         match Hashtbl.find_opt min_input_slot e.tag with
         | Some s -> Hashtbl.add cluster_of_pass2_node e.tag (s mod sp.ct_n2)
         | None -> ()
       ) pass2_nodes;
       (* DIF post-multiply Twiddle Loads have no spill-slot ancestors —
        * they're consumed by Cmuls on PASS 2 outputs. Assign each
        * unclustered Pass2 Load to the cluster of its (first) consumer. *)
       let consumers_p2 : (int, t list) Hashtbl.t = Hashtbl.create 256 in
       List.iter (fun e ->
         List.iter (fun p ->
           let prev = try Hashtbl.find consumers_p2 p.tag with Not_found -> [] in
           Hashtbl.replace consumers_p2 p.tag (e :: prev)
         ) (preds_of_general e)
       ) pass2_nodes;
       List.iter (fun e ->
         if not (Hashtbl.mem cluster_of_pass2_node e.tag) then begin
           let cs = try Hashtbl.find consumers_p2 e.tag with Not_found -> [] in
           let consumer_cluster = List.fold_left (fun acc c ->
             match acc, Hashtbl.find_opt cluster_of_pass2_node c.tag with
             | None, Some k -> Some k
             | _ -> acc
           ) None cs in
           match consumer_cluster with
           | Some k -> Hashtbl.add cluster_of_pass2_node e.tag k
           | None -> ()
         end
       ) pass2_nodes
     end;
     let pass2_ordered = match scheduler with
       | SU uarch when pass2_nodes <> [] && sp.ct_n2 > 0 ->
         (* Group by cluster (k2), then SU within. *)
         let groups = Array.make sp.ct_n2 [] in
         List.iter (fun e ->
           match Hashtbl.find_opt cluster_of_pass2_node e.tag with
           | Some k2 -> groups.(k2) <- e :: groups.(k2)
           | None -> ()  (* unreachable nodes — shouldn't happen *)
         ) pass2_nodes;
         (* Reverse each group to restore topo order, then SU per group. *)
         let assign_tags = List.fold_left (fun acc (_, e) ->
           Hashtbl.replace acc e.tag (); acc
         ) (Hashtbl.create 32) assigns in
         let result = ref [] in
         (* Emit sub-DFTs in increasing k2 order. *)
         for k2 = 0 to sp.ct_n2 - 1 do
           let group_nodes = List.rev groups.(k2) in
           let group_sinks = List.filter (fun e ->
             Hashtbl.mem assign_tags e.tag
           ) group_nodes in
           let scheduled =
             if group_nodes = [] then []
             else if group_sinks = [] then group_nodes
             else
               (match bb_budget with
                | None -> Schedule.su_schedule_subset uarch ~gh
                            ~subset:group_nodes ~sinks:group_sinks
                | Some t -> Bb.bb_schedule_subset uarch ~time_budget_sec:t
                              ~subset:group_nodes ~sinks:group_sinks)
           in
           result := scheduled :: !result
         done;
         List.concat (List.rev !result)
       | SU uarch when pass2_nodes <> [] ->
         let assign_tags = List.fold_left (fun acc (_, e) ->
           Hashtbl.replace acc e.tag (); acc
         ) (Hashtbl.create 32) assigns in
         let pass2_sinks = List.filter (fun e ->
           Hashtbl.mem assign_tags e.tag
         ) pass2_nodes in
         if pass2_sinks = [] then pass2_nodes
         else
           (match bb_budget with
            | None -> Schedule.su_schedule_subset uarch ~gh
                        ~subset:pass2_nodes ~sinks:pass2_sinks
            | Some t -> Bb.bb_schedule_subset uarch ~time_budget_sec:t
                          ~subset:pass2_nodes ~sinks:pass2_sinks)
       | _ -> pass2_nodes
     in

     (* Track which spilled tags have been reloaded. Walk pass2_ordered
      * and for each node, emit any pending reloads of its predecessors
      * before emitting the node. *)
     let reloaded : (int, unit) Hashtbl.t = Hashtbl.create 32 in
     (* Look through fused Muls when computing predecessors for reload
      * tracking. When Add(Mul(x,y), c) gets emitted as fmadd(x,y,c),
      * the rendered code references x, y, c directly — so x/y must be
      * reloaded if they're spilled, even though the Mul node itself
      * isn't directly referenced. *)
     let rec preds_of (e : t) : t list =
       match e.node with
       | NK_Const _ | NK_Load _ -> []
       | NK_Neg a -> through_fused_mul a
       | NK_Add (a, b) | NK_Sub (a, b) ->
         through_fused_mul a @ through_fused_mul b
       | NK_Mul (a, b) -> [a; b]
       | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) -> [a; b; c; d]
       | NK_Fma (a, b, c, _, _) -> [a; b; c]
     and through_fused_mul (n : t) : t list =
       if Hashtbl.mem fused_muls n.tag then
         match n.node with
         | NK_Mul (x, y) -> [x; y]
         | _ -> [n]  (* shouldn't happen — only Muls get fused *)
       else [n]
     in
     let emit_reload_if_needed (p : t) =
       if Hashtbl.mem reloaded p.tag then ()
       else begin
         let do_reload arr_name slot =
           Buffer.add_string buf (Printf.sprintf
             "            const %s t%d = %s(&%s[%d]);\n"
             isa.vec_type p.tag isa.loadu_pd arr_name slot);
           Hashtbl.add reloaded p.tag ()
         in
         match Hashtbl.find_opt sp.re_slot p.tag with
         | Some slot when not (is_fused_slot sp slot) ->
           do_reload "spill_re" slot
         | _ ->
           match Hashtbl.find_opt sp.im_slot p.tag with
           | Some slot when not (is_fused_slot sp slot) ->
             do_reload "spill_im" slot
           | _ -> ()
       end
     in

     (* Group assigns by their PASS 2 cluster so each sub-DFT's outputs
      * can be stored immediately after its computation. This frees the
      * registers holding the outputs and reduces peak live for the rest
      * of PASS 2. *)
     let assigns_by_cluster : (int, (Expr.elem_ref * t) list) Hashtbl.t =
       Hashtbl.create 16
     in
     (* Only Pass 2 assigns can have a cluster (cluster_of_pass2_node is
      * populated from pass2_nodes), so iterating pass2_assigns is exact;
      * Pass 1 assigns were stored at the end of PASS 1 and are skipped. *)
     List.iter (fun ((_, e) as a) ->
       match Hashtbl.find_opt cluster_of_pass2_node e.tag with
       | Some k2 ->
         let cur = try Hashtbl.find assigns_by_cluster k2 with Not_found -> [] in
         Hashtbl.replace assigns_by_cluster k2 (a :: cur)
       | None -> ()
     ) pass2_assigns;
     let last_pass2_cluster : int option ref = ref None in
     let flush_cluster_stores k2 =
       match Hashtbl.find_opt assigns_by_cluster k2 with
       | Some clist ->
         List.iter (fun (lhs, e) ->
           emit_reload_if_needed e;
           emit_store buf lhs e
         ) (List.rev clist)
       | None -> ()
     in
     List.iter (fun e ->
       (* Skip standalone emission of FMA-fused Mul nodes. *)
       if Hashtbl.mem fused_muls e.tag then ()
       else begin
         (* Emit reloads of any spilled predecessors not yet reloaded.
          * For fused-Muls we'd never reach here, but we still need to
          * reload predecessors of THE consuming Add/Sub. *)
         List.iter emit_reload_if_needed (preds_of e);
         Buffer.add_string buf
           (render_node_def ~fused_muls:fused_muls_opt ~isa ~in_place ~t1s e);
         Buffer.add_char buf '\n'
       end;
       (* Cluster-boundary detection: when this node finishes a cluster
        * (all of its cluster's nodes have been emitted), flush that
        * cluster's stores immediately. We check by tracking the cluster
        * of the LAST emitted node and detecting the transition. *)
       (match Hashtbl.find_opt cluster_of_pass2_node e.tag,
              !last_pass2_cluster with
        | Some c, Some prev when c <> prev ->
          (* Emit prev cluster's stores, since we just transitioned away. *)
          flush_cluster_stores prev;
          last_pass2_cluster := Some c
        | Some c, None ->
          last_pass2_cluster := Some c
        | _ -> ())
     ) pass2_ordered;
     (* Flush the final cluster's stores. *)
     (match !last_pass2_cluster with
      | Some c -> flush_cluster_stores c
      | None -> ());
     (* Safety net: emit stores for Pass 2 outputs not associated with
      * any cluster. Pass 1 outputs were already stored at the end of
      * PASS 1 above; we exclusively iterate pass2_assigns here. *)
     List.iter (fun ((_, e) as a) ->
       match Hashtbl.find_opt cluster_of_pass2_node e.tag with
       | Some _ -> ()  (* already emitted via cluster *)
       | None ->
         emit_reload_if_needed e;
         let (lhs, e) = a in emit_store buf lhs e
     ) pass2_assigns;
     Buffer.add_string buf "        }\n"

   | None ->
  (match scheduler with
   | Topological ->
     (* Existing path: emit all definitions in topo order, then stores. *)
     let roots = List.map snd assigns in
     let nodes = topo_sort_reachable roots in
     List.iter (fun e ->
       Buffer.add_string buf (render_node_def ~isa ~in_place ~t1s e);
       Buffer.add_char buf '\n'
     ) nodes;
     Buffer.add_char buf '\n';
     List.iter (fun (lhs, e) -> emit_store buf lhs e) assigns

   | Bisection ->
     (* Frigo's bisection schedule: each entry is either an intermediate
      * (None, e) → emit definition, or an output (Some oref, e) → emit store.
      *
      * One subtlety: leaves (NK_Const, NK_Load) appear in the schedule
      * but render as a "load" definition. This is fine — they're still
      * t<tag> = _mm512_loadu_pd(...) lines.
      *
      * Another subtlety: an intermediate node and its output sibling share
      * the same alg_node (same tag). The intermediate's definition emits
      * once; when we hit the output node afterwards, we just store the
      * existing t<tag>. We track which tags have been defined to avoid
      * duplicate definitions. *)
     let scheduled = Schedule.bisection_schedule assigns in
     let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
     List.iter (fun (oref_opt, e) ->
       match oref_opt with
       | None ->
         (* Intermediate computation: emit definition if not already. *)
         if not (Hashtbl.mem defined e.tag) then begin
           Hashtbl.add defined e.tag ();
           Buffer.add_string buf (render_node_def ~isa ~in_place ~t1s e);
           Buffer.add_char buf '\n'
         end
       | Some oref ->
         (* Output: ensure the value is defined, then emit a store. *)
         if not (Hashtbl.mem defined e.tag) then begin
           Hashtbl.add defined e.tag ();
           Buffer.add_string buf (render_node_def ~isa ~in_place ~t1s e);
           Buffer.add_char buf '\n'
         end;
         emit_store buf oref e
     ) scheduled

   | Annotated_topological ->
     (* Topological order, but emitted with nested-block scopes via annotate.ml.
      * Same instructions, same order — just nested `{ ... }` to communicate
      * variable lifetimes to GCC. *)
     let roots = List.map snd assigns in
     let nodes = topo_sort_reachable roots in
     (* Build the entry list: intermediates first (in topo order), then stores. *)
     let entries =
       List.map (fun e -> (None, e)) nodes
       @ List.map (fun (lhs, e) -> (Some lhs, e)) assigns
     in
     let render_intermediate e = render_node_def ~isa ~in_place ~t1s e in
     let render_store oref e =
       let buf2 = Buffer.create 128 in
       emit_store buf2 oref e;
       (* emit_store added its own \n; strip the trailing \n and indent. *)
       let s = Buffer.contents buf2 in
       String.trim s
     in
     let scope = Annotate.annotate entries in
     Annotate.emit_scope isa buf render_intermediate render_store scope

   | Annotated_bisection ->
     (* Bisection schedule, emitted with nested-block scopes via annotate.ml. *)
     let scheduled = Schedule.bisection_schedule assigns in
     (* Bisection's output may have an alg_node appearing twice (once as
      * intermediate, once for store). Dedupe so annotate sees a clean list. *)
     let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
     let entries = List.filter_map (fun (oref_opt, e) ->
       match oref_opt with
       | None ->
         if Hashtbl.mem defined e.tag then None
         else begin
           Hashtbl.add defined e.tag ();
           Some (None, e)
         end
       | Some oref ->
         if not (Hashtbl.mem defined e.tag) then begin
           (* Need to emit the definition before the store *)
           Hashtbl.add defined e.tag ();
           (* Note: this case shouldn't happen in well-formed bisection output. *)
           Some (Some oref, e)
         end else
           Some (Some oref, e)
     ) scheduled in
     let render_intermediate e = render_node_def ~isa ~in_place ~t1s e in
     let render_store oref e =
       let buf2 = Buffer.create 128 in
       emit_store buf2 oref e;
       String.trim (Buffer.contents buf2)
     in
     let scope = Annotate.annotate entries in
     Annotate.emit_scope isa buf render_intermediate render_store scope

   | SU uarch ->
     (* SU list scheduler: priority = (cp_dist DESC, su_num ASC).
      * Output shape mirrors Bisection: list of (oref_opt, alg_node)
      * where None = intermediate, Some oref = store.
      *
      * Single-use inlining: any intermediate with exactly one consumer
      * (in the DAG OR via output assignment) is inlined at the consumer
      * rather than emitted as a standalone declaration. This matches
      * hand-coded FFTW codelet style and significantly reduces register
      * pressure for DIF prime codelets. *)
     let scheduled = Schedule.su_schedule uarch assigns in
     let inline_set = compute_inline_set assigns in
     let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
     let is_inlined e = Hashtbl.mem inline_set e.tag in
     List.iter (fun (oref_opt, e) ->
       match oref_opt with
       | None ->
         (* Skip emission for inlined values — their consumer will inline. *)
         if not (is_inlined e) && not (Hashtbl.mem defined e.tag) then begin
           Hashtbl.add defined e.tag ();
           Buffer.add_string buf
             (render_node_def ~isa ~in_place ~t1s ~inline_set:(Some inline_set) e);
           Buffer.add_char buf '\n'
         end
       | Some oref ->
         (* Stores reference the value by t<tag>, so the value MUST be
          * named. Sinks are excluded from inline_set, so this is safe. *)
         if not (Hashtbl.mem defined e.tag) then begin
           Hashtbl.add defined e.tag ();
           Buffer.add_string buf
             (render_node_def ~isa ~in_place ~t1s ~inline_set:(Some inline_set) e);
           Buffer.add_char buf '\n'
         end;
         emit_store buf oref e
     ) scheduled

   | Annotated_SU uarch ->
     let scheduled = Schedule.su_schedule uarch assigns in
     let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
     let entries = List.filter_map (fun (oref_opt, e) ->
       match oref_opt with
       | None ->
         if Hashtbl.mem defined e.tag then None
         else begin
           Hashtbl.add defined e.tag ();
           Some (None, e)
         end
       | Some oref -> Some (Some oref, e)
     ) scheduled in
     let render_intermediate e = render_node_def ~isa ~in_place ~t1s e in
     let render_store oref e =
       let buf2 = Buffer.create 128 in
       emit_store buf2 oref e;
       String.trim (Buffer.contents buf2)
     in
     let scope = Annotate.annotate entries in
     Annotate.emit_scope isa buf render_intermediate render_store scope));

  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n";
  Buffer.contents buf
