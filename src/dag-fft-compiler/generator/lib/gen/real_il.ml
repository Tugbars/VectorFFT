(* real_il.ml — the interleaved REAL pair's kinds (2026-09-29; docs/roadmap/
   il_real_engine_research.md, method 1). Pipeline-hosted like c2c_il: the cx
   IR, the shared scheduler, the ISA-parametric renderer; only the loads, the
   untangle / the real DFT and the store edges are these kinds' own.

   THE PAIR. A real length N = R1 * R2 (both even, R2 >= 4). Two forms:

   FORM A (the packed leaf): x read as z[N/2] = x[2m] + i x[2m+1] is R1/2
   packed columns of R2 complex legs; the stock n1t(R2) leaf (count = R1/2,
   Ls = R1/2, OLs = R2) leaves packed column c's spectrum Z_c[p] at
   z[c*R2 + p], the two-for-one spectrum of the real columns 2c and 2c+1:
     Y_{2c}[p] = (Z_c[p] + conj Z_c[R2-p]) / 2,  Y_{2c+1}[p] = (Z_c[p] - conj
     Z_c[R2-p]) / (2i).
   t2h untangles in the top stage: for the column pair (p, p+1), p odd in
   1..R2/2-1, every packed leg is loaded directly and MIRRORED (columns
   R2-p-1, R2-p reversed and conjugated: one permute4x64 + one xor), the two
   real legs are formed (add, sub, the -i turn), pre-twiddled from the record
   set (the records carry the 1/2; leg 0 takes it as a real scale), the
   radix-R1 butterfly runs, and output leg q stores at X[q R2 + p] (q < R1/2,
   direct) or its mirror X[(R1-q) R2 - p] (reversed and conjugated).

   FORM B (the real leaf): r2z reads R1 real columns, four per vector, at
   the leg stride Ls = R1, computes the real R2-point DFT with real
   arithmetic (cx_real.ml) and stores column c's half spectrum packed —
   slot p = X_c[p] for p = 1..R2/2-1, slot 0 = (X_c[0], X_c[R2/2]), both
   real — at zout[2*(c*OLs + p)] with OLs = R2: half of every R2-wide row,
   the layout that makes the top stage in place. The store edge interleaves
   (re, im) per column: unpacklo/hi over the four columns, then the
   permute2f128 pairing of two slots (one shuffle per complex output); a
   two-column remainder runs at VEX-128. t2m is t2h without the untangle:
   the R1 legs are the half-stored real sub-spectra (direct loads at Ls =
   R2), twiddled unhalved, the same butterfly and mirror stores.

   Both forms: when R2/2 is even the last regular column runs at VEX-128,
   and the self-mirrored columns 0 and R2/2 run as ONE 2-lane pass with
   their own record set (lane 0 the plain leg, lane 1 the w_N^{j R2/2}
   twiddles) and per-lane 128-bit stores: lane 0 = X[q R2] for q <= R1/2
   (q = R1/2 IS the Nyquist bin), lane 1 = X[q R2 + R2/2] for q < R1/2.
   Form A gathers the pass's legs from the packed columns' bins 0 and
   R2/2 (self-mirrored: conj only); form B from the packed slot (a, b) as
   [a 0 | b 0]. X[0] and X[N/2] come out with exactly zero imaginary parts.

   IN PLACE (both tops): a column pair reads exactly the slots it writes
   (form A: its direct and mirror columns of every packed leg; form B: its
   columns of every row, the mirrors landing in the rows' unused halves), so
   the top runs on the leaf's plane itself; zin == zout is the shipped call.

   THE BACKWARDS (c2r) are the transposes. t2h bwd: the column pair's legs
   are the CCE bins X[q R2 + p] (q < R1/2 direct; q >= R1/2 from the mirror
   bin), the inverse butterfly, the conjugated records POST-butterfly, the
   re-tangle Z_c[p] = E + iO, Z_c[R2-p] = conj(E - iO), stored corner-turned
   (column p, packed leg c) -> zout[2*(p*OLs + c)], OLs = R1/2, for the stock
   n1 bwd(R2). t2m bwd: the same loads and butterfly, the R1 real legs
   stored corner-turned (column p, leg j) -> zout[2*(p*OLs + j)], OLs = R1,
   the special pass writing each leg's (Y[0], Y[R2/2]) packed slot; r2z bwd
   reads that plane four columns at a time (two loads, two permute2f128,
   two unpacks per slot), runs the unnormalized inverse real DFT and stores
   R2 real rows at zout[j*OLs + k], OLs = R1: the real output in natural
   order, N times x.

   ABI — the frozen 11-arg z signature; tw_re = the record sets, one per
   column pair (2s+1, 2s+2) in order, then the special set at index
   count/2; a set is (R1-1) records of [c c c' c'][-s +s -s' +s'], 2*VW
   doubles each. Ls / OLs = the leg pitch in / out, count = R2/2 (the tops)
   or R1 (the real leaf). Gs, OGs, tw_im unused. 256-bit ISA only. *)

open Cx_ir
open Cx_render

type dir =
  | Fwd
  | Bwd

type kind =
  | T2h (* form A top: the untangling Hermitian mid over the packed leaf *)
  | T2m (* form B top: the Hermitian mid over the real leaf's half spectra *)
  | R2z (* form B leaf: real columns -> packed half spectra (bwd: the inverse) *)
  | R1c (* the real FLAT leaf: real legs over contiguous columns -> the digit runs *)
  | R2zr (* the real ROWS: the real leaf over row-major rows -> each row's CCE bins *)

let kind_name = function
  | T2h -> "t2h"
  | T2m -> "t2m"
  | R2z -> "r2z"
  | R1c -> "r1c"
  | R2zr -> "r2zr"
;;

let kind_of_string = function
  | "t2h" -> T2h
  | "t2m" -> T2m
  | "r2z" -> R2z
  | "r1c" -> R1c
  | "r2zr" -> R2zr
  | s -> failwith ("real_il: unknown kind " ^ s ^ " (t2h | t2m | r2z | r1c | r2zr)")
;;

(* ═══════════════════════════════════════════════════════════════
   THE TOPS: t2h (untangle) and t2m (real legs)
   ═══════════════════════════════════════════════════════════════ *)
let emit_top ~(untangle : bool) ~(dir : dir) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t)
  : string
  =
  let vw = isa.Isa.vec_width in
  if radix < 4 || radix mod 2 <> 0
  then failwith "real_il: the top stage needs an even radix >= 4";
  let kname = if untangle then "t2h" else "t2m" in
  let h = radix / 2 in
  let rec_d = (radix - 1) * 2 * vw in
  let ctx =
    make_ctx
      ~tw_group:false
      ~tw_log3:false
      ~tw_pre:false
      ~tw_gen2:false
      ~colstride:false
      ~st_turn:false
      ~st_turn_gs:false
      ~tangent:false
  in
  let tbl : consts = Hashtbl.create 16 in
  let sign = if dir = Fwd then `Fwd else `Bwd in
  let dname = if dir = Fwd then "fwd" else "bwd" in
  let emit_group
        ~(body : Buffer.t)
        ~(nisa : Isa.t)
        ~(msuf : string)
        ~(tw_vw : int)
        ~(special : bool)
        ~(label : string)
    : unit
    =
    reset ();
    let per = nisa.Isa.vec_width / 2 in
    let loads = ref [] in
    let ld (a : caddr) : t =
      let e = cload a in
      loads := (a, e) :: !loads;
      e
    in
    let name (e : t) = Printf.sprintf "z%d" e.tag in
    let line (s : string) = Buffer.add_string body (Printf.sprintf "        %s\n" s) in
    let assigns, stores =
      match dir with
      | Fwd ->
        let legs =
          if untangle
          then (
            let eo =
              Array.init h (fun c ->
                let a = ld (if special then AZinSpec c else AZinLeg c) in
                let b = ld (if special then AZinSpecC c else AZinMir (c + 1)) in
                cadd a b, crot (csub a b))
            in
            Array.init radix (fun j ->
              let e, o = eo.(j / 2) in
              if j mod 2 = 0 then if j = 0 then ctw 0.5 0.0 e else ctwl j e else ctwl j o))
          else
            Array.init radix (fun j ->
              let x = ld (if special then AZinSpecP j else AZinLeg j) in
              if j = 0 then x else ctwl j x)
        in
        let outs = Cx_math.dft_small ~sign ~ctx radix legs in
        let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs) in
        let stores () =
          if special
          then
            Array.iteri
              (fun q (e : t) ->
                 if q <= h
                 then
                   line
                     (render_store Isa.sse2 (AZoutSpec (q, 0)) (Isa.cx_part_pd nisa (name e) 0)
                      ^ ";");
                 if q < h
                 then
                   line
                     (render_store Isa.sse2 (AZoutSpec (q, 1)) (Isa.cx_part_pd nisa (name e) 1)
                      ^ ";"))
              outs
          else
            Array.iteri
              (fun q (e : t) ->
                 let a = if q < h then AZoutLeg q else AZoutMir (radix - q) in
                 line (render_store nisa a (name e) ^ ";"))
              outs
        in
        assigns, stores
      | Bwd ->
        let xs =
          Array.init radix (fun q ->
            if special
            then ld (AZinSpecB (radix, q))
            else if q < h
            then ld (AZinLeg q)
            else ld (AZinMir (radix - q)))
        in
        let outs = Cx_math.dft_small ~sign ~ctx radix xs in
        let ys = Array.mapi (fun j e -> if j > 0 then ctwl j e else e) outs in
        if special
        then (
          let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) ys) in
          let stores () =
            if untangle
            then
              for c = 0 to h - 1 do
                (* Z_c[0] = (E, O) of lane 0, Z_c[R2/2] = (E, O) of lane 1 *)
                line
                  (Printf.sprintf
                     "{ const __m256d _u = _mm256_unpacklo_pd(%s, %s);"
                     (name ys.(2 * c))
                     (name ys.((2 * c) + 1)));
                line
                  ("  "
                   ^ render_store Isa.sse2 (AZoutSpecT (c, 0)) (Isa.cx_part_pd nisa "_u" 0)
                   ^ ";");
                line
                  ("  "
                   ^ render_store Isa.sse2 (AZoutSpecT (c, 1)) (Isa.cx_part_pd nisa "_u" 1)
                   ^ "; }")
              done
            else
              for j = 0 to radix - 1 do
                (* leg j's packed slot (Y[0], Y[R2/2]) = the real parts of its two lanes *)
                line
                  (render_store
                     Isa.sse2
                     (AZoutSpecT (j, 0))
                     (Printf.sprintf
                        "_mm256_castpd256_pd128(_mm256_permute4x64_pd(%s, 0x08))"
                        (name ys.(j)))
                   ^ ";")
              done
          in
          assigns, stores)
        else (
          (* the corner-turned store of the output legs: form A re-tangles
             each packed leg (V = E + iO, and W = E - iO whose conjugate is
             the mirror pair); form B stores the real legs as they are *)
          let legs_v, legs_w =
            if untangle
            then
              ( Array.init h (fun c -> crotadd ys.(2 * c) ys.((2 * c) + 1))
              , Some (Array.init h (fun c -> cadd ys.(2 * c) (crot ys.((2 * c) + 1))) ) )
            else ys, None
          in
          let nl = Array.length legs_v in
          let groups =
            let rec go l0 acc =
              if l0 >= nl then List.rev acc else go (l0 + per) ((l0, min per (nl - l0)) :: acc)
            in
            go 0 []
          in
          let turned (legs : t array) (l0 : int) (r : int) =
            if r = per
            then (
              let cols, inter = Cx_math.turn_transpose ~per (Array.sub legs l0 r) in
              `Full (cols, inter))
            else `Quarter legs.(l0)
          in
          let tv = List.map (fun (l0, r) -> l0, r, turned legs_v l0 r) groups in
          let tw =
            match legs_w with
            | Some w -> List.map (fun (l0, r) -> l0, r, turned w l0 r) groups
            | None -> []
          in
          let roots =
            List.concat_map
              (fun (_, _, g) ->
                 match g with
                 | `Full (cols, _) -> Array.to_list cols
                 | `Quarter x -> [ x ])
              (tv @ tw)
          in
          let assigns = List.mapi (fun i e -> Expr.Output (i, true), e) roots in
          let stores () =
            let emit_g (mir : bool) (l0, _, g) =
              let addr c = if mir then AZoutTurnMir (l0, c) else AZoutTurn (l0, c) in
              match g with
              | `Full (cols, _) ->
                Array.iteri (fun c (col : t) -> line (render_store nisa (addr c) (name col) ^ ";")) cols
              | `Quarter x ->
                for c = 0 to per - 1 do
                  line (render_store Isa.sse2 (addr c) (Isa.cx_part_pd nisa (name x) c) ^ ";")
                done
            in
            List.iter (emit_g false) tv;
            List.iter (emit_g true) tw
          in
          assigns, stores)
    in
    let assigns =
      Cx_pipeline.prepare_codelet
        ~who:(Printf.sprintf "r%d_%s_%s%s" radix kname dname (if special then "_spec" else ""))
        ~uarch
        assigns
    in
    let sch = C2c_il.cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    List.iter
      (fun ((a : caddr), (e : t)) -> line (Isa.const_decl nisa (name e) (render_load nisa a)))
      (List.rev !loads);
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ | CLoad _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             line (Isa.const_decl nisa (name e) (render ~ctx ~tw_vw ~msuf nisa tbl e))))
      sch;
    stores ();
    Buffer.add_string body "        }\n"
  in
  let body_w = Buffer.create 8192 in
  let body_n = Buffer.create 4096 in
  let body_s = Buffer.create 4096 in
  emit_group ~body:body_w ~nisa:isa ~msuf:"" ~tw_vw:0 ~special:false ~label:"the column pair (k, k+1)";
  emit_group
    ~body:body_n
    ~nisa:Isa.sse2
    ~msuf:"_n"
    ~tw_vw:vw
    ~special:false
    ~label:"the last regular column at VEX-128";
  emit_group
    ~body:body_s
    ~nisa:isa
    ~msuf:""
    ~tw_vw:0
    ~special:true
    ~label:"the self-mirrored columns 0 and count";
  let buf = Buffer.create 16384 in
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:(Printf.sprintf "full-IL (interleaved-complex) %s, radix-%d %s" kname radix dname)
       [ Printf.sprintf "ISA: %s; %d complex per vector" isa.Isa.name (vw / 2)
       ; Printf.sprintf "Uarch: %s" uarch.Uarch.name
       ; Printf.sprintf
           "Form: the real pair's Hermitian top stage, %s (real_il.ml)"
           (if untangle then "untangling the packed leaf" else "over the real leaf's half spectra")
       ]);
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (real_il.ml). radix-%d %s %s: %s\n\
       \ * tw_re = the record sets: per column pair (2s+1, 2s+2), then the special\n\
       \ * set at index count/2; each (R-1) records of [c c c' c'][-s +s -s' +s'],\n\
       \ * %s. Ls = the leg pitch in, OLs = the leg pitch out,\n\
       \ * count = R2/2 (the half-stored columns). Gs, OGs, tw_im unused.\n\
       \ * Columns 1..count-1 in pairs (the last one at VEX-128 when count is even),\n\
       \ * then the self-mirrored columns 0 and count as one 2-lane pass. */\n"
       radix
       kname
       dname
       (match dir, untangle with
        | Fwd, true ->
          "the packed leaf plane (R2 legs per packed column) -> the CCE\n\
          \ * half spectrum X[0..N/2], in place (zin == zout is the shipped call)."
        | Fwd, false ->
          "the real leaf's half spectra (R1 rows of R2/2 packed slots at\n\
          \ * pitch R2) -> the CCE half spectrum X[0..N/2], in place (zin == zout)."
        | Bwd, true ->
          "the CCE half spectrum -> the packed leaf plane, corner-turned\n\
          \ * (column p, packed leg c) -> zout[2*(p*OLs + c)] for the stock n1 bwd."
        | Bwd, false ->
          "the CCE half spectrum -> the real legs' half spectra, corner-turned\n\
          \ * (column p, leg j) -> zout[2*(p*OLs + j)] for the r2z bwd leaf.")
       (match dir, untangle with
        | Fwd, true -> "halved (the untangle's 1/2; leg 0 takes it as a real scale)"
        | Fwd, false -> "unhalved"
        | Bwd, _ -> "conjugated, applied POST-butterfly, unhalved"));
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string buf (Isa.im_mask_decl isa "_M_IM");
  Buffer.add_string buf "  /* negate im lanes: conj, x*(-i) */\n";
  Buffer.add_string buf (Isa.im_mask_decl Isa.sse2 "_M_IM_n");
  Buffer.add_string buf "  /* tail twin */\n";
  if dir = Bwd
  then (
    Buffer.add_string buf (Isa.re_mask_decl isa "_M_RE");
    Buffer.add_string buf "  /* negate re lanes: x*(+i) */\n";
    Buffer.add_string buf (Isa.re_mask_decl Isa.sse2 "_M_RE_n");
    Buffer.add_string buf "  /* tail twin */\n";
    Buffer.add_string
      buf
      "static const __m256d _M_IMHI = { 0.0, 0.0, 0.0, -0.0 };  /* conj lane 1 only (the special gather) */\n");
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string buf "\n";
  Buffer.add_string
    buf
    (Abi.z11_signature
       ~alias_tolerant:(dir = Fwd)
       ~symbol:(Printf.sprintf "radix%d_z_%s_%s_%s" radix kname dname isa.Isa.name)
       ~target_attr:(Isa.cx_target_attr isa)
       ());
  Buffer.add_string buf "    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OGs;\n";
  Buffer.add_string buf "    size_t k = 1;\n";
  Buffer.add_string buf "    for (; k + 2 <= count; k += 2) {\n";
  Buffer.add_string
    buf
    (Printf.sprintf "        const double *twp = tw_re + ((k - 1) / 2) * (size_t)%d;\n" rec_d);
  Buffer.add_buffer buf body_w;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    if (k < count) {  /* the last regular column: the same DAG at VEX-128 */\n";
  Buffer.add_string
    buf
    (Printf.sprintf "        const double *twp = tw_re + ((k - 1) / 2) * (size_t)%d;\n" rec_d);
  Buffer.add_buffer buf body_n;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    {  /* the self-mirrored columns 0 and count: one 2-lane pass */\n";
  Buffer.add_string
    buf
    (Printf.sprintf "        const double *twp = tw_re + (count / 2) * (size_t)%d;\n" rec_d);
  Buffer.add_buffer buf body_s;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

(* ═══════════════════════════════════════════════════════════════
   THE REAL LEAF: r2z (fwd: real columns -> packed half spectra;
   bwd: the corner-turned half spectra -> real columns, N times x)
   ═══════════════════════════════════════════════════════════════ *)
let emit_leaf ~(dir : dir) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let vw = isa.Isa.vec_width in
  if vw <> 4 then failwith "real_il: r2z is emitted for the 256-bit ISA only";
  if radix < 4 || radix mod 2 <> 0 then failwith "real_il: r2z needs an even radix >= 4";
  let h = radix / 2 in
  (* h slots per column: slot 0 = (X[0], X[h]) packed, slots 1..h-1 = X[p] *)
  let ctx =
    make_ctx
      ~tw_group:false
      ~tw_log3:false
      ~tw_pre:false
      ~tw_gen2:false
      ~colstride:false
      ~st_turn:false
      ~st_turn_gs:false
      ~tangent:false
  in
  let tbl : consts = Hashtbl.create 16 in
  let dname = if dir = Fwd then "fwd" else "bwd" in
  (* one column group: `lanes` real columns (4 wide, 2 at VEX-128) *)
  let emit_group ~(body : Buffer.t) ~(nisa : Isa.t) ~(label : string) : unit =
    reset ();
    let lanes = nisa.Isa.vec_width in
    let loads = ref [] in
    let ld (a : caddr) : t =
      let e = cload a in
      loads := (a, e) :: !loads;
      e
    in
    let name (e : t) = Printf.sprintf "z%d" e.tag in
    let line (s : string) = Buffer.add_string body (Printf.sprintf "        %s\n" s) in
    let zero () = csub (cin 0) (cin 0) in
    let force (x : t option) : t =
      match x with
      | Some v -> v
      | None -> zero ()
    in
    let assigns, stores =
      match dir with
      | Fwd ->
        let x = Array.init radix (fun l -> ld (AXinLeg l)) in
        let xs = Cx_real.rdft radix x in
        (* slot p's (A, B) vectors: A = re, B = im (slot 0: B = X[h].re) *)
        let ab p =
          if p = 0
          then force (fst xs.(0)), force (fst xs.(h))
          else force (fst xs.(p)), force (snd xs.(p))
        in
        if lanes = 4
        then (
          (* pairs of slots (p, p+1): four unpacks, four turns, four stores *)
          let roots = ref []
          and st = ref [] in
          let p = ref 0 in
          while !p < h do
            if !p + 1 < h
            then (
              let a0, b0 = ab !p
              and a1, b1 = ab (!p + 1) in
              let ulo = cunpack a0 b0 false
              and uhi = cunpack a0 b0 true
              and vlo = cunpack a1 b1 false
              and vhi = cunpack a1 b1 true in
              let t0 = cturn ulo vlo false
              and t2 = cturn ulo vlo true
              and t1 = cturn uhi vhi false
              and t3 = cturn uhi vhi true in
              List.iter
                (fun (c, tnode) ->
                   roots := tnode :: !roots;
                   st := (AZoutTurn (!p, c), tnode, None) :: !st)
                [ 0, t0; 1, t1; 2, t2; 3, t3 ];
              p := !p + 2)
            else (
              (* the lone last slot (h odd): its two unpacks scatter as quarters *)
              let a0, b0 = ab !p in
              let ulo = cunpack a0 b0 false
              and uhi = cunpack a0 b0 true in
              roots := uhi :: ulo :: !roots;
              st := (AZoutTurn (!p, 0), ulo, Some 0) :: (AZoutTurn (!p, 2), ulo, Some 1)
                    :: (AZoutTurn (!p, 1), uhi, Some 0) :: (AZoutTurn (!p, 3), uhi, Some 1) :: !st;
              p := !p + 1)
          done;
          let roots = List.rev !roots
          and st = List.rev !st in
          ( List.mapi (fun i e -> Expr.Output (i, true), e) roots
          , fun () ->
              List.iter
                (fun (a, e, part) ->
                   match part with
                   | None -> line (render_store nisa a (name e) ^ ";")
                   | Some c ->
                     line (render_store Isa.sse2 a (Isa.cx_part_pd nisa (name e) c) ^ ";"))
                st ))
        else (
          (* VEX-128: two columns; unpacklo = column k's slot, unpackhi = column k+1's *)
          let roots = ref []
          and st = ref [] in
          for p = 0 to h - 1 do
            let a, b = ab p in
            let ulo = cunpack a b false
            and uhi = cunpack a b true in
            roots := uhi :: ulo :: !roots;
            st := (AZoutTurn (p, 1), uhi) :: (AZoutTurn (p, 0), ulo) :: !st
          done;
          let roots = List.rev !roots
          and st = List.rev !st in
          ( List.mapi (fun i e -> Expr.Output (i, true), e) roots
          , fun () -> List.iter (fun (a, e) -> line (render_store nisa a (name e) ^ ";")) st ))
      | Bwd ->
        (* slot p of the column group -> (re, im) vectors over the columns *)
        let re_im p =
          if lanes = 4
          then (
            let l0 = ld (AZinLeg p)
            and l1 = ld (AZinLegOff (p, 2)) in
            let z0 = cturn l0 l1 false
            and z1 = cturn l0 l1 true in
            cunpack z0 z1 false, cunpack z0 z1 true)
          else (
            let l0 = ld (AZinLeg p)
            and l1 = ld (AZinLegOff (p, 1)) in
            cunpack l0 l1 false, cunpack l0 l1 true)
        in
        let xs = Array.make (h + 1) (None, None) in
        for p = 0 to h - 1 do
          let a, b = re_im p in
          if p = 0
          then (
            xs.(0) <- Some a, None;
            xs.(h) <- Some b, None)
          else xs.(p) <- Some a, Some b
        done;
        let out = Cx_real.irdft radix xs in
        let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) out) in
        ( assigns
        , fun () -> Array.iteri (fun j (e : t) -> line (render_store nisa (AXoutLeg j) (name e) ^ ";")) out )
    in
    let assigns =
      Cx_pipeline.prepare_codelet
        ~who:(Printf.sprintf "r%d_r2z_%s_%d" radix dname lanes)
        ~uarch
        assigns
    in
    let sch = C2c_il.cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    List.iter
      (fun ((a : caddr), (e : t)) -> line (Isa.const_decl nisa (name e) (render_load nisa a)))
      (List.rev !loads);
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ | CLoad _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             line (Isa.const_decl nisa (name e) (render ~ctx nisa tbl e))))
      sch;
    stores ();
    Buffer.add_string body "        }\n"
  in
  let body_w = Buffer.create 8192 in
  let body_n = Buffer.create 4096 in
  emit_group ~body:body_w ~nisa:isa ~label:"four real columns";
  emit_group ~body:body_n ~nisa:Isa.sse2 ~label:"the two-column remainder at VEX-128";
  let buf = Buffer.create 16384 in
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:(Printf.sprintf "full-IL (interleaved-complex) r2z, radix-%d %s" radix dname)
       [ Printf.sprintf "ISA: %s; %d real columns per vector" isa.Isa.name vw
       ; Printf.sprintf "Uarch: %s" uarch.Uarch.name
       ; "Form: the real pair's real leaf (real_il.ml, cx_real.ml)"
       ]);
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (real_il.ml). radix-%d r2z %s: %s\n\
       \ * count = the real columns (a multiple of 2; four per wide iteration,\n\
       \ * a two-column remainder at VEX-128). tw_re, tw_im, Gs, OGs unused. */\n"
       radix
       dname
       (if dir = Fwd
        then
          "real columns (leg l at zin[l*Ls + k]) -> each column's packed half\n\
          \ * spectrum, slot p at zout[2*(c*OLs + p)]: slot 0 = (X[0], X[R/2]),\n\
          \ * slots 1..R/2-1 = X[p]; real arithmetic throughout (cx_real.ml)."
        else
          "the corner-turned half spectra (slot p of column c at\n\
          \ * zin[2*(p*Ls + c)], slot 0 packed) -> R real rows zout[l*OLs + k],\n\
          \ * the unnormalized inverse (R times x)."));
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string buf "\n";
  Buffer.add_string
    buf
    (Abi.z11_signature
       ~alias_tolerant:false
       ~symbol:(Printf.sprintf "radix%d_z_r2z_%s_%s" radix dname isa.Isa.name)
       ~target_attr:(Isa.cx_target_attr isa)
       ());
  Buffer.add_string buf "    (void)zin_unused; (void)zout_unused; (void)tw_re; (void)tw_im; (void)Gs; (void)OGs;\n";
  Buffer.add_string buf "    size_t k = 0;\n";
  Buffer.add_string buf (Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw);
  Buffer.add_buffer buf body_w;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    if (k < count) {  /* two columns at VEX-128 */\n";
  Buffer.add_buffer buf body_n;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

(* ═══════════════════════════════════════════════════════════════
   THE REAL ROWS: r2zr (2026-10-01; the row pass of a real plane)

   The whole real R-point DFT of every ROW of a row-major plane, a lane a
   row: row k's samples at zin[k*Ls + j] (Ls = the row pitch in doubles),
   its CCE bins 0..R/2 at zout[2*(k*OLs + p)] (OLs = the row pitch in
   complex, >= R/2 + 1). The arithmetic is the real leaf's (cx_real.ml,
   real throughout, four rows per vector); the edges are the row-major ones:

   - the load edge takes four rows' samples 4b..4b+3 as four vectors and
     transposes the block (two unpacks per row pair, one turn per sample),
     so sample j across the four rows is one vector; R = 4m + 2 takes its
     last two samples from an overlapping block at R - 4;
   - the store edge is r2z's per-row interleave over the slots 0..R/2, with
     the DC and Nyquist slots as (x, 0): slots pair up (p, p+1) into one
     256-bit store per row, a lone last slot leaves as per-row halves.

   count rows, at least two: four per wide iteration, then two at VEX-128;
   a lone last row runs with the row before it (out of place, the same
   values written again). The backward twin follows it (emit_rows_bwd).
   ═══════════════════════════════════════════════════════════════ *)
let emit_rows_fwd ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let vw = isa.Isa.vec_width in
  if vw <> 4 then failwith "real_il: r2zr is emitted for the 256-bit ISA only";
  if radix < 4 || radix mod 2 <> 0 then failwith "real_il: r2zr needs an even radix >= 4";
  let h = radix / 2 in
  let ctx =
    make_ctx
      ~tw_group:false
      ~tw_log3:false
      ~tw_pre:false
      ~tw_gen2:false
      ~colstride:false
      ~st_turn:false
      ~st_turn_gs:false
      ~tangent:false
  in
  let tbl : consts = Hashtbl.create 16 in
  let emit_group ~(body : Buffer.t) ~(nisa : Isa.t) ~(label : string) : unit =
    reset ();
    let lanes = nisa.Isa.vec_width in
    let loads = ref [] in
    let ld (a : caddr) : t =
      let e = cload a in
      loads := (a, e) :: !loads;
      e
    in
    let name (e : t) = Printf.sprintf "z%d" e.tag in
    let line (s : string) = Buffer.add_string body (Printf.sprintf "        %s\n" s) in
    (* the load edge: a block = `lanes` rows x `lanes` samples, transposed;
       `keep` = the samples of the block this call supplies *)
    let x : t option array = Array.make radix None in
    let block (off : int) (keep : int list) : unit =
      let s =
        if lanes = 4
        then (
          let a = Array.init 4 (fun r -> ld (AXinRow (r, off))) in
          let u0 = cunpack a.(0) a.(1) false
          and u1 = cunpack a.(0) a.(1) true
          and u2 = cunpack a.(2) a.(3) false
          and u3 = cunpack a.(2) a.(3) true in
          [| cturn u0 u2 false; cturn u1 u3 false; cturn u0 u2 true; cturn u1 u3 true |])
        else (
          let a0 = ld (AXinRow (0, off))
          and a1 = ld (AXinRow (1, off)) in
          [| cunpack a0 a1 false; cunpack a0 a1 true |])
      in
      List.iter (fun j -> x.(off + j) <- Some s.(j)) keep
    in
    for b = 0 to (radix / lanes) - 1 do
      block (b * lanes) (List.init lanes Fun.id)
    done;
    if radix mod lanes <> 0 then block (radix - lanes) [ lanes - 2; lanes - 1 ];
    let x =
      Array.mapi
        (fun j v ->
           match v with
           | Some e -> e
           | None -> failwith (Printf.sprintf "real_il.r2zr: sample %d not loaded" j))
        x
    in
    let xs = Cx_real.rdft radix x in
    let zero = ld AZeroV in
    let force (v : t option) : t =
      match v with
      | Some e -> e
      | None -> zero
    in
    (* slot p's (A, B) vectors over the rows: A = re, B = im; the DC and the
       Nyquist slots are real *)
    let ab p =
      if p = 0 || p = h then force (fst xs.(p)), zero else force (fst xs.(p)), force (snd xs.(p))
    in
    let roots = ref []
    and st = ref [] in
    if lanes = 4
    then (
      let p = ref 0 in
      while !p <= h do
        if !p + 1 <= h
        then (
          let a0, b0 = ab !p
          and a1, b1 = ab (!p + 1) in
          let ulo = cunpack a0 b0 false
          and uhi = cunpack a0 b0 true
          and vlo = cunpack a1 b1 false
          and vhi = cunpack a1 b1 true in
          let t0 = cturn ulo vlo false
          and t2 = cturn ulo vlo true
          and t1 = cturn uhi vhi false
          and t3 = cturn uhi vhi true in
          List.iter
            (fun (c, tnode) ->
               roots := tnode :: !roots;
               st := (AZoutTurn (!p, c), tnode, None) :: !st)
            [ 0, t0; 1, t1; 2, t2; 3, t3 ];
          p := !p + 2)
        else (
          (* the lone last slot: its two unpacks leave as per-row halves *)
          let a0, b0 = ab !p in
          let ulo = cunpack a0 b0 false
          and uhi = cunpack a0 b0 true in
          roots := uhi :: ulo :: !roots;
          st := (AZoutTurn (!p, 0), ulo, Some 0) :: (AZoutTurn (!p, 2), ulo, Some 1)
                :: (AZoutTurn (!p, 1), uhi, Some 0) :: (AZoutTurn (!p, 3), uhi, Some 1) :: !st;
          p := !p + 1)
      done)
    else
      (* VEX-128: two rows; unpacklo = row k's slot, unpackhi = row k+1's *)
      for p = 0 to h do
        let a, b = ab p in
        let ulo = cunpack a b false
        and uhi = cunpack a b true in
        roots := uhi :: ulo :: !roots;
        st := (AZoutTurn (p, 1), uhi, None) :: (AZoutTurn (p, 0), ulo, None) :: !st
      done;
    let roots = List.rev !roots
    and st = List.rev !st in
    let assigns = List.mapi (fun i e -> Expr.Output (i, true), e) roots in
    let assigns =
      Cx_pipeline.prepare_codelet ~who:(Printf.sprintf "r%d_r2zr_fwd_%d" radix lanes) ~uarch assigns
    in
    let sch = C2c_il.cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    List.iter
      (fun ((a : caddr), (e : t)) -> line (Isa.const_decl nisa (name e) (render_load nisa a)))
      (List.rev !loads);
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ | CLoad _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             line (Isa.const_decl nisa (name e) (render ~ctx nisa tbl e))))
      sch;
    List.iter
      (fun (a, e, part) ->
         match part with
         | None -> line (render_store nisa a (name e) ^ ";")
         | Some c -> line (render_store Isa.sse2 a (Isa.cx_part_pd nisa (name e) c) ^ ";"))
      st;
    Buffer.add_string body "        }\n"
  in
  let body_w = Buffer.create 8192 in
  let body_n = Buffer.create 4096 in
  emit_group ~body:body_w ~nisa:isa ~label:"four rows";
  emit_group ~body:body_n ~nisa:Isa.sse2 ~label:"two rows at VEX-128";
  let buf = Buffer.create 16384 in
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:(Printf.sprintf "full-IL (interleaved-complex) r2zr, radix-%d fwd" radix)
       [ Printf.sprintf "ISA: %s; %d rows per vector" isa.Isa.name vw
       ; Printf.sprintf "Uarch: %s" uarch.Uarch.name
       ; "Form: the real rows (real_il.ml, cx_real.ml)"
       ]);
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (real_il.ml). radix-%d r2zr fwd: the real %d-point DFT of every\n\
       \ * ROW of a row-major plane (row k's samples at zin[k*Ls + j]) -> its CCE\n\
       \ * bins 0..%d at zout[2*(k*OLs + p)]; real arithmetic throughout (cx_real.ml),\n\
       \ * the DC and Nyquist bins stored with a zero imaginary part.\n\
       \ * count = the rows, at least two: four per wide iteration, then two at\n\
       \ * VEX-128 (a lone last row runs with the row before it). Out of place.\n\
       \ * tw_re, tw_im, Gs, OGs unused. */\n"
       radix
       radix
       h);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string buf "\n";
  Buffer.add_string
    buf
    (Abi.z11_signature
       ~alias_tolerant:false
       ~symbol:(Printf.sprintf "radix%d_z_r2zr_fwd_%s" radix isa.Isa.name)
       ~target_attr:(Isa.cx_target_attr isa)
       ());
  Buffer.add_string buf "    (void)zin_unused; (void)zout_unused; (void)tw_re; (void)tw_im; (void)Gs; (void)OGs;\n";
  Buffer.add_string buf "    size_t k = 0;\n";
  Buffer.add_string buf (Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw);
  Buffer.add_buffer buf body_w;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    while (k < count) {  /* two rows at VEX-128 */\n";
  Buffer.add_string buf "        if (k + 2 > count) k = count - 2;  /* a lone last row: with the row before it */\n";
  Buffer.add_buffer buf body_n;
  Buffer.add_string buf "        k += 2;\n";
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

(* ═══════════════════════════════════════════════════════════════
   THE REAL ROWS, BACKWARD: r2zr bwd (2026-10-05): the CCE bins 0..R/2 of
   every row of a row-major plane -> its R real samples, unnormalized (R
   times x). The arithmetic is cx_real.ml's inverse recursion; the edges
   are the forward's, swapped:
   - the load edge takes four rows' slot pairs (p, p+1) as four vectors
     (row k+r's bins at zin[2*((k+r)*Ls + p)]) and turns them into the
     per-slot (re, im) lane vectors: one 128-bit lane turn per row pair,
     one unpack per component. The DC and Nyquist imaginary parts are
     never read. A lone last slot (R/2 even) loads as the half rows (0, 2)
     and (1, 3), one vector each;
   - the store edge transposes four sample vectors back into four row
     blocks through the forward's load network (its own inverse); R = 4m + 2
     leaves its last two samples as per-row halves.
   count rows, at least two: four per wide iteration, then two at VEX-128;
   a lone last row runs with the row before it. Out of place.
   ═══════════════════════════════════════════════════════════════ *)
let emit_rows_bwd ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let vw = isa.Isa.vec_width in
  if vw <> 4 then failwith "real_il: r2zr is emitted for the 256-bit ISA only";
  if radix < 4 || radix mod 2 <> 0 then failwith "real_il: r2zr needs an even radix >= 4";
  let h = radix / 2 in
  let ctx =
    make_ctx
      ~tw_group:false
      ~tw_log3:false
      ~tw_pre:false
      ~tw_gen2:false
      ~colstride:false
      ~st_turn:false
      ~st_turn_gs:false
      ~tangent:false
  in
  let tbl : consts = Hashtbl.create 16 in
  let emit_group ~(body : Buffer.t) ~(nisa : Isa.t) ~(label : string) : unit =
    reset ();
    let lanes = nisa.Isa.vec_width in
    let loads = ref [] in
    let ld (a : caddr) : t =
      let e = cload a in
      loads := (a, e) :: !loads;
      e
    in
    let name (e : t) = Printf.sprintf "z%d" e.tag in
    let line (s : string) = Buffer.add_string body (Printf.sprintf "        %s\n" s) in
    (* the load edge: slot p's (re, im) vectors over the rows; the DC and
       Nyquist slots real *)
    let xs = Array.make (h + 1) (None, None) in
    let slot (p : int) (lo : t) (hi : t) : unit =
      let re = cunpack lo hi false in
      if p = 0 || p = h then xs.(p) <- (Some re, None) else xs.(p) <- (Some re, Some (cunpack lo hi true))
    in
    if lanes = 4
    then (
      let p = ref 0 in
      while !p <= h do
        if !p + 1 <= h
        then (
          (* rows 0..3, slots (p, p+1): a lane turn per row pair *)
          let l = Array.init 4 (fun r -> ld (AZinTurn (!p, r))) in
          let ulo = cturn l.(0) l.(2) false
          and uhi = cturn l.(1) l.(3) false
          and vlo = cturn l.(0) l.(2) true
          and vhi = cturn l.(1) l.(3) true in
          slot !p ulo uhi;
          slot (!p + 1) vlo vhi;
          p := !p + 2)
        else (
          (* the lone last slot: the half rows (0, 2) and (1, 3) *)
          let ulo = ld (AZinTurnH (!p, 0))
          and uhi = ld (AZinTurnH (!p, 1)) in
          slot !p ulo uhi;
          p := !p + 1)
      done)
    else
      (* VEX-128: two rows; slot p of each row as one load *)
      for p = 0 to h do
        let l0 = ld (AZinTurn (p, 0))
        and l1 = ld (AZinTurn (p, 1)) in
        slot p l0 l1
      done;
    let x = Cx_real.irdft radix xs in
    (* the store edge: the sample vectors back into row blocks *)
    let roots = ref []
    and st = ref [] in
    if lanes = 4
    then (
      let block (off : int) : unit =
        let u0 = cunpack x.(off) x.(off + 1) false
        and u1 = cunpack x.(off) x.(off + 1) true
        and u2 = cunpack x.(off + 2) x.(off + 3) false
        and u3 = cunpack x.(off + 2) x.(off + 3) true in
        List.iter
          (fun (r, e) ->
             roots := e :: !roots;
             st := (AXoutRow (r, off), e, None) :: !st)
          [ 0, cturn u0 u2 false; 1, cturn u1 u3 false; 2, cturn u0 u2 true; 3, cturn u1 u3 true ]
      in
      for b = 0 to (radix / 4) - 1 do
        block (4 * b)
      done;
      if radix mod 4 <> 0
      then (
        (* the last two samples: two unpacks, four per-row halves *)
        let off = radix - 2 in
        let u0 = cunpack x.(off) x.(off + 1) false
        and u1 = cunpack x.(off) x.(off + 1) true in
        roots := u1 :: u0 :: !roots;
        st := (AXoutRow (0, off), u0, Some 0) :: (AXoutRow (2, off), u0, Some 1)
              :: (AXoutRow (1, off), u1, Some 0) :: (AXoutRow (3, off), u1, Some 1) :: !st))
    else
      (* VEX-128: two rows, a sample pair per store; unpacklo = row k's, unpackhi = row k+1's *)
      for b = 0 to h - 1 do
        let off = 2 * b in
        let u0 = cunpack x.(off) x.(off + 1) false
        and u1 = cunpack x.(off) x.(off + 1) true in
        roots := u1 :: u0 :: !roots;
        st := (AXoutRow (1, off), u1, None) :: (AXoutRow (0, off), u0, None) :: !st
      done;
    let roots = List.rev !roots
    and st = List.rev !st in
    let assigns = List.mapi (fun i e -> Expr.Output (i, true), e) roots in
    let assigns =
      Cx_pipeline.prepare_codelet ~who:(Printf.sprintf "r%d_r2zr_bwd_%d" radix lanes) ~uarch assigns
    in
    let sch = C2c_il.cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    List.iter
      (fun ((a : caddr), (e : t)) -> line (Isa.const_decl nisa (name e) (render_load nisa a)))
      (List.rev !loads);
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ | CLoad _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             line (Isa.const_decl nisa (name e) (render ~ctx nisa tbl e))))
      sch;
    List.iter
      (fun (a, e, part) ->
         match part with
         | None -> line (render_store nisa a (name e) ^ ";")
         | Some c -> line (render_store Isa.sse2 a (Isa.cx_part_pd nisa (name e) c) ^ ";"))
      st;
    Buffer.add_string body "        }\n"
  in
  let body_w = Buffer.create 8192 in
  let body_n = Buffer.create 4096 in
  emit_group ~body:body_w ~nisa:isa ~label:"four rows";
  emit_group ~body:body_n ~nisa:Isa.sse2 ~label:"two rows at VEX-128";
  let buf = Buffer.create 16384 in
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:(Printf.sprintf "full-IL (interleaved-complex) r2zr, radix-%d bwd" radix)
       [ Printf.sprintf "ISA: %s; %d rows per vector" isa.Isa.name vw
       ; Printf.sprintf "Uarch: %s" uarch.Uarch.name
       ; "Form: the real rows, backward (real_il.ml, cx_real.ml)"
       ]);
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (real_il.ml). radix-%d r2zr bwd: the CCE bins 0..%d of every\n\
       \ * ROW of a row-major plane (row k's bins at zin[2*(k*Ls + p)]) -> its %d real\n\
       \ * samples at zout[k*OLs + j], unnormalized (%d times x); real arithmetic\n\
       \ * throughout (cx_real.ml), the DC and Nyquist imaginary parts never read.\n\
       \ * count = the rows, at least two: four per wide iteration, then two at\n\
       \ * VEX-128 (a lone last row runs with the row before it). Out of place.\n\
       \ * tw_re, tw_im, Gs, OGs unused. */\n"
       radix
       h
       radix
       radix);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string buf "\n";
  Buffer.add_string
    buf
    (Abi.z11_signature
       ~alias_tolerant:false
       ~symbol:(Printf.sprintf "radix%d_z_r2zr_bwd_%s" radix isa.Isa.name)
       ~target_attr:(Isa.cx_target_attr isa)
       ());
  Buffer.add_string buf "    (void)zin_unused; (void)zout_unused; (void)tw_re; (void)tw_im; (void)Gs; (void)OGs;\n";
  Buffer.add_string buf "    size_t k = 0;\n";
  Buffer.add_string buf (Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw);
  Buffer.add_buffer buf body_w;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    while (k < count) {  /* two rows at VEX-128 */\n";
  Buffer.add_string buf "        if (k + 2 > count) k = count - 2;  /* a lone last row: with the row before it */\n";
  Buffer.add_buffer buf body_n;
  Buffer.add_string buf "        k += 2;\n";
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

let emit_rows ~(dir : dir) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  match dir with
  | Fwd -> emit_rows_fwd ~radix ~isa ~uarch
  | Bwd -> emit_rows_bwd ~radix ~isa ~uarch
;;

(* ═══════════════════════════════════════════════════════════════
   THE REAL FLAT LEAF: r1c (2026-09-30; the real flat DIT's one kind)

   The flat DIT's first stage transforms across the most significant digit:
   R legs at stride D = N/R, count = D contiguous columns. On REAL input
   that stage is the real R-point DFT of each column (cx_real.ml) and its
   output is Hermitian in the digit p, so only p = 0..(R-1)/2 exist: digit 0
   is a REAL run of D (the same problem again, R times smaller) and every
   digit p >= 1 a COMPLEX run of D -- the state the c2c flat DIT is in after
   its own leaf, for that digit's block. R odd.

   fwd: legs zin[l*Ls + k] (real) -> digit 0 at zout[k] (real), digit p at
        zout[2p*OLs + 2k] (interleaved complex) = block p of the c2c flat
        plane at pitch OLs, so the c2c stages run on the digit blocks as
        they stand (the second half of block 0 is unused).
   bwd: the inverse, unnormalized (R times x).
   Four real columns per vector; the digit store interleaves (re, im) per
   column (two unpacks and two lane turns per digit). count is ANY: a
   two-column step at VEX-128, then the last lone column (D is odd for an
   odd N). Out of place only (a digit's run interleaves into columns the
   walk has not read yet). tw_re, tw_im, Gs, OGs unused.
   ═══════════════════════════════════════════════════════════════ *)
let emit_rleaf ~(dir : dir) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let vw = isa.Isa.vec_width in
  if vw <> 4 then failwith "real_il: r1c is emitted for the 256-bit ISA only";
  if radix < 3 || radix mod 2 <> 1 then failwith "real_il: r1c needs an odd radix >= 3";
  let h = radix / 2 in
  let ctx =
    make_ctx
      ~tw_group:false
      ~tw_log3:false
      ~tw_pre:false
      ~tw_gen2:false
      ~colstride:false
      ~st_turn:false
      ~st_turn_gs:false
      ~tangent:false
  in
  let tbl : consts = Hashtbl.create 16 in
  let dname = if dir = Fwd then "fwd" else "bwd" in
  (* one column group: `lanes` real columns (4 wide; 2 or 1 at VEX-128) *)
  let emit_group ~(body : Buffer.t) ~(nisa : Isa.t) ~(lanes : int) ~(label : string) : unit =
    reset ();
    let loads = ref [] in
    let ld (a : caddr) : t =
      let e = cload a in
      loads := (a, e) :: !loads;
      e
    in
    let name (e : t) = Printf.sprintf "z%d" e.tag in
    let line (s : string) = Buffer.add_string body (Printf.sprintf "        %s\n" s) in
    let assigns, stores =
      match dir with
      | Fwd ->
        let x = Array.init radix (fun l -> ld (if lanes = 1 then AZinReal l else AXinLeg l)) in
        let xs = Cx_real.rdft radix x in
        let d0 = Cx_real.get (fst xs.(0)) "r1c digit 0" in
        let roots = ref [ d0 ]
        and st = ref [ (if lanes = 1 then AZoutReal 0 else AXoutLeg 0), d0 ] in
        for d = 1 to h do
          let re = Cx_real.get (fst xs.(d)) "r1c re"
          and im = Cx_real.get (snd xs.(d)) "r1c im" in
          let ulo = cunpack re im false in
          if lanes = 4
          then (
            let uhi = cunpack re im true in
            let t0 = cturn ulo uhi false
            and t1 = cturn ulo uhi true in
            roots := t1 :: t0 :: !roots;
            st := (ADigOut (d, 4), t1) :: (ADigOut (d, 0), t0) :: !st)
          else if lanes = 2
          then (
            let uhi = cunpack re im true in
            roots := uhi :: ulo :: !roots;
            st := (ADigOut (d, 2), uhi) :: (ADigOut (d, 0), ulo) :: !st)
          else (
            roots := ulo :: !roots;
            st := (ADigOut (d, 0), ulo) :: !st)
        done;
        let roots = List.rev !roots
        and st = List.rev !st in
        ( List.mapi (fun i e -> Expr.Output (i, true), e) roots
        , fun () -> List.iter (fun (a, e) -> line (render_store nisa a (name e) ^ ";")) st )
      | Bwd ->
        let xs = Array.make (h + 1) (None, None) in
        xs.(0) <- Some (ld (if lanes = 1 then AZinReal 0 else AXinLeg 0)), None;
        for d = 1 to h do
          let re, im =
            if lanes = 4
            then (
              let l0 = ld (ADigIn (d, 0))
              and l1 = ld (ADigIn (d, 4)) in
              let z0 = cturn l0 l1 false
              and z1 = cturn l0 l1 true in
              cunpack z0 z1 false, cunpack z0 z1 true)
            else if lanes = 2
            then (
              let l0 = ld (ADigIn (d, 0))
              and l1 = ld (ADigIn (d, 2)) in
              cunpack l0 l1 false, cunpack l0 l1 true)
            else (
              let l0 = ld (ADigIn (d, 0)) in
              l0, cunpack l0 l0 true)
          in
          xs.(d) <- Some re, Some im
        done;
        let out = Cx_real.irdft radix xs in
        ( Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) out)
        , fun () ->
            Array.iteri
              (fun j (e : t) ->
                 line (render_store nisa (if lanes = 1 then AZoutReal j else AXoutLeg j) (name e) ^ ";"))
              out )
    in
    let assigns =
      Cx_pipeline.prepare_codelet
        ~who:(Printf.sprintf "r%d_r1c_%s_%d" radix dname lanes)
        ~uarch
        assigns
    in
    let sch = C2c_il.cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    List.iter
      (fun ((a : caddr), (e : t)) -> line (Isa.const_decl nisa (name e) (render_load nisa a)))
      (List.rev !loads);
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ | CLoad _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             line (Isa.const_decl nisa (name e) (render ~ctx nisa tbl e))))
      sch;
    stores ();
    Buffer.add_string body "        }\n"
  in
  let body_w = Buffer.create 8192 in
  let body_n = Buffer.create 4096 in
  let body_1 = Buffer.create 4096 in
  emit_group ~body:body_w ~nisa:isa ~lanes:4 ~label:"four real columns";
  emit_group ~body:body_n ~nisa:Isa.sse2 ~lanes:2 ~label:"two columns at VEX-128";
  emit_group ~body:body_1 ~nisa:Isa.sse2 ~lanes:1 ~label:"the last lone column";
  let buf = Buffer.create 16384 in
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:(Printf.sprintf "full-IL (interleaved-complex) r1c, radix-%d %s" radix dname)
       [ Printf.sprintf "ISA: %s; %d real columns per vector" isa.Isa.name vw
       ; Printf.sprintf "Uarch: %s" uarch.Uarch.name
       ; "Form: the real flat DIT's leaf (real_il.ml, cx_real.ml)"
       ]);
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (real_il.ml). radix-%d r1c %s: %s\n\
       \ * count = the columns (any: four per wide iteration, then two at VEX-128,\n\
       \ * then the lone last one). Out of place. tw_re, tw_im, Gs, OGs unused. */\n"
       radix
       dname
       (if dir = Fwd
        then
          "R real legs (leg l at zin[l*Ls + k]) -> the digit runs of the\n\
          \ * real R-point DFT of each column: digit 0 real at zout[k], digit p in\n\
          \ * 1..(R-1)/2 complex at zout[2p*OLs + 2k]; real arithmetic (cx_real.ml)."
        else
          "the digit runs (digit 0 real at zin[k], digit p complex at\n\
          \ * zin[2p*Ls + 2k]) -> R real legs zout[l*OLs + k], the unnormalized\n\
          \ * inverse (R times x)."));
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string buf "\n";
  Buffer.add_string
    buf
    (Abi.z11_signature
       ~alias_tolerant:false
       ~symbol:(Printf.sprintf "radix%d_z_r1c_%s_%s" radix dname isa.Isa.name)
       ~target_attr:(Isa.cx_target_attr isa)
       ());
  Buffer.add_string buf "    (void)zin_unused; (void)zout_unused; (void)tw_re; (void)tw_im; (void)Gs; (void)OGs;\n";
  Buffer.add_string buf "    size_t k = 0;\n";
  Buffer.add_string buf (Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw);
  Buffer.add_buffer buf body_w;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    if (k + 2 <= count) {  /* two columns at VEX-128 */\n";
  Buffer.add_buffer buf body_n;
  Buffer.add_string buf "        k += 2;\n";
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "    if (k < count) {  /* the last lone column */\n";
  Buffer.add_buffer buf body_1;
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

let emit ~(kind : kind) ~(dir : dir) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  if isa.Isa.vec_width <> 4
  then failwith "real_il: the real pair's kinds are emitted for the 256-bit ISA only";
  match kind with
  | T2h -> emit_top ~untangle:true ~dir ~radix ~isa ~uarch
  | T2m -> emit_top ~untangle:false ~dir ~radix ~isa ~uarch
  | R2z -> emit_leaf ~dir ~radix ~isa ~uarch
  | R1c -> emit_rleaf ~dir ~radix ~isa ~uarch
  | R2zr -> emit_rows ~dir ~radix ~isa ~uarch
;;
