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
    String.concat
      "_"
      (Array.to_list (Array.map (fun (c, s) -> Printf.sprintf "%.17g:%.17g" c s) w))
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

let emit_const_decls (_isa : Isa.t) (tbl : consts) : string =
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
         match Array.length w with
         | 1 -> "__m128d"
         | 2 -> "__m256d"
         | 4 -> "__m512d"
         | n -> failwith (Printf.sprintf "emit_const_decls: %d-lane constant" n)
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

(* ── Symbolic address -> C expression (complete-IR arc, 2026-08-09).
 * Width-independent: the strings are double-array indexing, identical for
 * the wide body and the SSE2 tail. These MUST reproduce the historical
 * hand-edge strings byte-for-byte — they are the byte-identity contract. *)
(* ROTFMA (VFFT_CX_ROTFMA=1): render-level fold of a -i rotation into its
   consuming butterfly — CAdd/CSub/CFmaC/CFnmaC over a CRotNI child emit
   fnmadd/fmadd(w_s, cflip y, acc) where w_s = [-c,+c,...] reuses the
   (1, c) constant-table pair. BIT-EXACT: the sign moves into an exact
   constant, one rounding either way (the √½-fold argument). This is the
   hand w32tg pass-B idiom ([c,-c]·flip, zero xors). Default OFF ⇒ every
   existing emission is byte-identical. Note: a folded CRotNI def with no
   other consumer remains as a dead SSA line — gcc DCEs it; asm is clean. *)
(* M12a: rotfma lives in Cx_ir.ctx now. *)

(* t2cs COLUMN-STRIDE mode (set by C2c_il.emit for the kind, reset for every
   other emission): column k lives at k*Gs (in) / k*OGs (out) instead of k.
   The narrow (VEX-128, single-column) path renders through addr_str as
   usual; the wide path goes through render_load / render_store below,
   which emit two 128-bit halves (columns k and k+1). *)
let colstride = ref false

(* the HERMITIAN edge of the real flat DIT's last stage (t2csgh / t2csght,
   2026-10-01; set by C2c_il.emit for those kinds, 0 otherwise): the stage's
   legs are the bins of one block, leg l at l*OLs (fwd: OLs = the leg stride
   of the half spectrum, OGs = its column pitch); the half spectrum holds bins
   0..N/2 only, so leg l < R/2 goes to its place, leg l > R/2 conjugated to
   the mirror N - bin, and the middle leg to whichever side its bin is on. The
   body knows its column's absolute place through _d0 = (zout - origin)/2 (the
   wrapper passes the origin in zin_unused and the mirror base out + 2N in
   zout_unused); the transposed backward reads the same way (origin in
   zout_unused, mirror in zin_unused, _d0 = (zin - origin)/2). herm = 1 the
   store edge, 2 the load edge; herm_hl = R/2. *)
let herm = ref 0
let herm_hl = ref 0

(* the Hermitian kernel's MAIN width (set by C2c_il.emit): a 256-bit arm of a 512-bit kernel
   (its ladder tail) names the ymm twin of the conjugation mask, _M_IM_y *)
let herm_main_vw = ref 4
let herm_ymask () = if !herm_main_vw > 4 then "_M_IM_y" else "_M_IM"

let addr_str (a : caddr) : string =
  match a with
  | AZinLeg l when !colstride -> Printf.sprintf "zin[2*((size_t)%d*Ls + (size_t)k*Gs)]" l
  | AZoutLeg l when !colstride -> Printf.sprintf "zout[2*((size_t)%d*OLs + (size_t)k*OGs)]" l
  | AZinLeg l -> Printf.sprintf "zin[2*((size_t)%d*Ls + k)]" l
  | AZoutLeg l -> Printf.sprintf "zout[2*((size_t)%d*OLs + k)]" l
  (* col c = column k + c of the wide iteration (0 .. per-1): per = 2 prints
     the historical k / k + 1 strings byte-for-byte. *)
  | AZoutTurn (l, 0) -> Printf.sprintf "zout[2*((size_t)k*OLs + %d)]" l
  | AZoutTurn (l, c) -> Printf.sprintf "zout[2*(((size_t)k + %d)*OLs + %d)]" c l
  | AZoutTurnG (l, 0) -> Printf.sprintf "zout[2*((size_t)k*OLs + (size_t)%d*OGs)]" l
  | AZoutTurnG (l, c) -> Printf.sprintf "zout[2*(((size_t)k + %d)*OLs + (size_t)%d*OGs)]" c l
  | AS i -> Printf.sprintf "S[%d]" i
  | AP i -> Printf.sprintf "P[%d]" i
  | AZinAbs i -> Printf.sprintf "zin[%d]" i
  | AZoutAbs i -> Printf.sprintf "zout[%d]" i
  | ATw i -> Printf.sprintf "twp[%d]" i
  | AZoutSpec (q, 0) -> Printf.sprintf "zout[2*((size_t)%d*OLs)]" q
  | AZoutSpec (q, _) -> Printf.sprintf "zout[2*((size_t)%d*OLs + count)]" q
  | AZoutSpecT (c, 0) -> Printf.sprintf "zout[2*((size_t)%d)]" c
  | AZoutSpecT (c, _) -> Printf.sprintf "zout[2*((size_t)count*OLs + %d)]" c
  | AXinLeg l -> Printf.sprintf "zin[(size_t)%d*Ls + k]" l
  | AXoutLeg l -> Printf.sprintf "zout[(size_t)%d*OLs + k]" l
  | AZinLegOff (l, o) -> Printf.sprintf "zin[2*((size_t)%d*Ls + k + %d)]" l o
  | AZinReal l -> Printf.sprintf "zin[(size_t)%d*Ls + k]" l
  | AZinHerm (l, r) -> Printf.sprintf "zin[2*((size_t)%d*Ls + k)]" (r - l)
  | AZoutReal l -> Printf.sprintf "zout[(size_t)%d*OLs + k]" l
  | ADigOut (d, o) -> Printf.sprintf "zout[(size_t)%d*OLs + 2*k + %d]" (2 * d) o
  | ADigIn (d, o) -> Printf.sprintf "zin[(size_t)%d*Ls + 2*k + %d]" (2 * d) o
  | AXinRow (0, o) -> Printf.sprintf "zin[(size_t)k*Ls + %d]" o
  | AXinRow (r, o) -> Printf.sprintf "zin[((size_t)k + %d)*Ls + %d]" r o
  | AZinTurn (l, 0) -> Printf.sprintf "zin[2*((size_t)k*Ls + %d)]" l
  | AZinTurn (l, r) -> Printf.sprintf "zin[2*(((size_t)k + %d)*Ls + %d)]" r l
  | AXoutRow (0, o) -> Printf.sprintf "zout[(size_t)k*OLs + %d]" o
  | AXoutRow (r, o) -> Printf.sprintf "zout[((size_t)k + %d)*OLs + %d]" r o
  | AZinTurnH _ -> failwith "cx_render.addr_str: the half-row pair renders through render_load"
  | AZeroV -> failwith "cx_render.addr_str: the zero vector has no address (render_load)"
  | AZinMir _ | AZoutMir _ | AZoutTurnMir _ | AZinSpec _ | AZinSpecC _ | AZinSpecB _ | AZinSpecP _ ->
    failwith "cx_render.addr_str: a mirror / special form renders through render_load or render_store"
;;

(* ── the t2h mirror and special forms (real_il.ml) ──────────────────────
   A mirror LOAD is the reversed, conjugated vector of the mirror columns:
   at 256 bits one permute4x64 (swap the 128-bit halves) and one xor; at
   VEX-128 the reversal is the identity and only the xor remains. A mirror
   STORE is the same two ops on the value. The turned mirror store lands one
   column per store, so it conjugates only. The masks are the file's own
   _M_IM (256) / _M_IM_n (128) — both are declared by every t2h file. *)
let mir_mask (isa : Isa.t) : string = if isa.Isa.vec_width = 2 then "_M_IM_n" else "_M_IM"

let mir_value (isa : Isa.t) (v : string) : string =
  match isa.Isa.vec_width with
  | 4 -> Printf.sprintf "_mm256_xor_pd(_mm256_permute4x64_pd(%s, 0x4E), _M_IM)" v
  | 2 -> Printf.sprintf "_mm_xor_pd(%s, _M_IM_n)" v
  | w -> failwith (Printf.sprintf "cx_render.mir_value: no mirror at vec_width %d" w)
;;

let spec_gather (i0 : string) (i1 : string) : string =
  Printf.sprintf
    "_mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(&zin[2*(%s)])), \
     _mm_loadu_pd(&zin[2*(%s)]), 1)"
    i0
    i1
;;

(* VFFT_CX_STORE128 — the M-128 store edge (TURNED-axis, mid slot): each
   leg-major OUTPUT store leaves as two contiguous 128-bit halves instead of
   one 256-bit store. A 16B store at complex granularity can never straddle a
   cache line, so this deletes the strided-odd-offset split stores (11.8%
   measured on the R16 tangent mid, VTune 2026-08-15) at the price of one
   extractf128 + one extra store uop per output. Scoped to AZoutLeg on the
   256-bit ISA only — the S plane, turn stores, and the VEX-128 odd-count
   tail (whose values are __m128d) are untouched. Default OFF ⇒ byte-identical.
   Since 2026-10-01 also a kind flag: --cil-st128 sets it for one emission (the
   n1c column leaf's half-store twin, the "h" form); C2c_il.emit resets it to
   the environment's default for every other emission. *)
let store128_env = Sys.getenv_opt "VFFT_CX_STORE128" = Some "1"
let store128 = ref store128_env

(* A store node as a C statement (no trailing ';'). The value operand is the
 * def name of the sunk node — same convention as `render`'s operands. *)
(* A leg LOAD as a C expression: the wide t2cs form gathers columns k and
   k+1 (two blocks, Gs apart) as two 128-bit halves; every other case is
   the plain wide load. *)
let render_load (isa : Isa.t) (a : caddr) : string =
  match a with
  | AZinMir m ->
    mir_value
      isa
      (Isa.loadu_pd
         isa
         (Printf.sprintf "zin[2*((size_t)%d*Ls - k - %d)]" m ((isa.Isa.vec_width / 2) - 1)))
  | AZinSpecP c ->
    (* the packed (DC, Nyquist) slot of real leg c as the two real legs of
       the self-mirrored columns: (a, b) -> [a 0 | b 0] *)
    Printf.sprintf
      "_mm256_blend_pd(_mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(&zin[2*((size_t)%d*Ls)])), 0x50), _mm256_setzero_pd(), 0xA)"
      c
  | AZinSpec c ->
    spec_gather (Printf.sprintf "(size_t)%d*Ls" c) (Printf.sprintf "(size_t)%d*Ls + count" c)
  | AZinSpecC c ->
    Printf.sprintf
      "_mm256_xor_pd(%s, _M_IM)"
      (spec_gather (Printf.sprintf "(size_t)%d*Ls" c) (Printf.sprintf "(size_t)%d*Ls + count" c))
  | AZinSpecB (r, q) ->
    (* the c2r special leg q: lane 0 = X[q R2] (q > R/2: conj X[(R-q) R2]),
       lane 1 = X[q R2 + R2/2] (q >= R/2: conj X[(R-1-q) R2 + R2/2]) *)
    let h = r / 2 in
    let i0, c0 =
      if q <= h
      then Printf.sprintf "(size_t)%d*Ls" q, false
      else Printf.sprintf "(size_t)%d*Ls" (r - q), true
    in
    let i1, c1 =
      if q < h
      then Printf.sprintf "(size_t)%d*Ls + count" q, false
      else Printf.sprintf "(size_t)%d*Ls + count" (r - 1 - q), true
    in
    let v = spec_gather i0 i1 in
    (match c0, c1 with
     | false, false -> v
     | true, true -> Printf.sprintf "_mm256_xor_pd(%s, _M_IM)" v
     | false, true -> Printf.sprintf "_mm256_xor_pd(%s, _M_IMHI)" v
     | true, false -> failwith "cx_render.render_load: AZinSpecB lane pattern")
  (* the Hermitian load edge: the mirrored legs conjugated from the mirror, the
     middle leg per column by its bin (2*place < Ls: below N/2) *)
  | AZinLeg l when !colstride && !herm = 2 && l > !herm_hl && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm256_xor_pd(_mm256_loadu2_m128d(zin_unused - 2*(_d0 + ((size_t)k + 1)*Gs + (size_t)%d*Ls), \
       zin_unused - 2*(_d0 + (size_t)k*Gs + (size_t)%d*Ls)), %s)"
      l
      l
      (herm_ymask ())
  | AZinLeg l when !colstride && !herm = 2 && l = !herm_hl && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "({ const size_t _dk = _d0 + (size_t)k*Gs, _dk1 = _d0 + ((size_t)k + 1)*Gs; \
       const __m128d _c0 = (2*_dk < Ls) ? _mm_loadu_pd(&zin[2*((size_t)%d*Ls + (size_t)k*Gs)]) \
       : _mm_xor_pd(_mm_loadu_pd(zin_unused - 2*(_dk + (size_t)%d*Ls)), _M_IM_n); \
       const __m128d _c1 = (2*_dk1 < Ls) ? _mm_loadu_pd(&zin[2*((size_t)%d*Ls + ((size_t)k + 1)*Gs)]) \
       : _mm_xor_pd(_mm_loadu_pd(zin_unused - 2*(_dk1 + (size_t)%d*Ls)), _M_IM_n); \
       _mm256_insertf128_pd(_mm256_castpd128_pd256(_c0), _c1, 1); })"
      l l l l
  | AZinLeg l when !colstride && !herm = 2 && l > !herm_hl && isa.Isa.vec_width = 2 ->
    Printf.sprintf "_mm_xor_pd(_mm_loadu_pd(zin_unused - 2*(_d0 + (size_t)k*Gs + (size_t)%d*Ls)), _M_IM_n)" l
  | AZinLeg l when !colstride && !herm = 2 && l = !herm_hl && isa.Isa.vec_width = 2 ->
    Printf.sprintf
      "((2*(_d0 + (size_t)k*Gs) < Ls) ? _mm_loadu_pd(&zin[2*((size_t)%d*Ls + (size_t)k*Gs)]) \
       : _mm_xor_pd(_mm_loadu_pd(zin_unused - 2*(_d0 + (size_t)k*Gs + (size_t)%d*Ls)), _M_IM_n))"
      l l
  (* the Hermitian load edge at 512 bits: four columns k..k+3 at pitch Gs, as the 256-bit
     edge twice (the mirrored legs: two loadu2 pairs joined, conjugated; the middle leg: per
     column by its bin) *)
  | AZinLeg l when !colstride && !herm = 2 && l > !herm_hl && isa.Isa.vec_width = 8 ->
    let m j = Printf.sprintf "zin_unused - 2*(_d0 + ((size_t)k + %d)*Gs + (size_t)%d*Ls)" j l in
    Printf.sprintf
      "_mm512_xor_pd(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu2_m128d(%s, %s)), \
       _mm256_loadu2_m128d(%s, %s), 1), _M_IM)"
      (m 1) (m 0) (m 3) (m 2)
  | AZinLeg l when !colstride && !herm = 2 && l = !herm_hl && isa.Isa.vec_width = 8 ->
    let c j =
      Printf.sprintf
        "const size_t _dk%d = _d0 + ((size_t)k + %d)*Gs; \
         const __m128d _c%d = (2*_dk%d < Ls) ? _mm_loadu_pd(&zin[2*((size_t)%d*Ls + ((size_t)k + %d)*Gs)]) \
         : _mm_xor_pd(_mm_loadu_pd(zin_unused - 2*(_dk%d + (size_t)%d*Ls)), _M_IM_n); "
        j j j j l j j l
    in
    Printf.sprintf
      "({ %s%s%s%s_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_insertf128_pd(_mm256_castpd128_pd256(_c0), _c1, 1)), \
       _mm256_insertf128_pd(_mm256_castpd128_pd256(_c2), _c3, 1), 1); })"
      (c 0) (c 1) (c 2) (c 3)
  | AZinLeg l when !colstride && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm256_loadu2_m128d(&zin[2*((size_t)%d*Ls + ((size_t)k + 1)*Gs)], \
       &zin[2*((size_t)%d*Ls + (size_t)k*Gs)])"
      l
      l
  | AZinLeg l when !colstride && isa.Isa.vec_width = 8 ->
    (* width 8: 4 columns at pitch Gs -> two loadu2 pairs joined *)
    Printf.sprintf
      "_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu2_m128d(&zin[2*((size_t)%d*Ls + ((size_t)k + 1)*Gs)], \
       &zin[2*((size_t)%d*Ls + (size_t)k*Gs)])), _mm256_loadu2_m128d(&zin[2*((size_t)%d*Ls + ((size_t)k + 3)*Gs)], \
       &zin[2*((size_t)%d*Ls + ((size_t)k + 2)*Gs)]), 1)"
      l l l l
  (* the real mono's loads: a real lane per column, imaginary zero; the
     backward's Hermitian half as the conjugate of the mirrored bin *)
  | AZinReal l when isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm256_setr_pd(zin[(size_t)%d*Ls + k], 0.0, zin[(size_t)%d*Ls + k + 1], 0.0)"
      l
      l
  | AZinReal l when isa.Isa.vec_width = 2 -> Printf.sprintf "_mm_load_sd(&zin[(size_t)%d*Ls + k])" l
  | AZinReal l when isa.Isa.vec_width = 8 ->
    (* the 4 columns' reals in one 256-bit load, spread to the real lanes by
       one lane permute; the mask zeroes the imaginary lanes *)
    Printf.sprintf
      "_mm512_maskz_permutexvar_pd(0x55, _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), _mm512_castpd256_pd512(_mm256_loadu_pd(&zin[(size_t)%d*Ls + k])))"
      l
  | AZinReal _ -> failwith "cx_render.render_load: AZinReal renders at 512 / 256 / 128 bits only"
  | AZinHerm (l, r) when isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm256_xor_pd(_mm256_loadu_pd(&zin[2*((size_t)%d*Ls + k)]), _mm256_setr_pd(0.0, -0.0, 0.0, -0.0))"
      (r - l)
  | AZinHerm (l, r) when isa.Isa.vec_width = 2 ->
    Printf.sprintf
      "_mm_xor_pd(_mm_loadu_pd(&zin[2*((size_t)%d*Ls + k)]), _mm_setr_pd(0.0, -0.0))"
      (r - l)
  | AZinHerm (l, r) when isa.Isa.vec_width = 8 ->
    Printf.sprintf
      "_mm512_xor_pd(_mm512_loadu_pd(&zin[2*((size_t)%d*Ls + k)]), _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0))"
      (r - l)
  | AZinHerm _ -> failwith "cx_render.render_load: AZinHerm renders at 512 / 256 / 128 bits only"
  | AZeroV when isa.Isa.vec_width = 4 -> "_mm256_setzero_pd()"
  | AZeroV when isa.Isa.vec_width = 2 -> "_mm_setzero_pd()"
  | AZeroV -> failwith "cx_render.render_load: AZeroV renders at 256 / 128 bits only"
  (* the r2zr backward's lone last slot: rows k+r and k+r+2 as the two halves *)
  | AZinTurnH (l, r) when isa.Isa.vec_width = 4 ->
    Printf.sprintf "_mm256_loadu2_m128d(&%s, &%s)" (addr_str (AZinTurn (l, r + 2))) (addr_str (AZinTurn (l, r)))
  | AZinTurnH _ -> failwith "cx_render.render_load: AZinTurnH renders at 256 bits only"
  | _ -> Isa.loadu_pd isa (addr_str a)
;;

let render_store (isa : Isa.t) (a : caddr) (v : string) : string =
  match a with
  | AZoutMir m ->
    Isa.storeu_pd
      isa
      (Printf.sprintf "zout[2*((size_t)%d*OLs - k - %d)]" m ((isa.Isa.vec_width / 2) - 1))
      (mir_value isa v)
  | AZoutTurnMir (l, c) ->
    Isa.storeu_pd
      isa
      (Printf.sprintf "zout[2*(((size_t)Ls - k - %d)*OLs + %d)]" c l)
      (Isa.xor_mask_pd isa v (mir_mask isa))
  (* the Hermitian store edge (see herm above) *)
  | AZoutLeg l when !colstride && !herm = 1 && l > !herm_hl && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm256_storeu2_m128d(zout_unused - 2*(_d0 + ((size_t)k + 1)*OGs + (size_t)%d*OLs), \
       zout_unused - 2*(_d0 + (size_t)k*OGs + (size_t)%d*OLs), _mm256_xor_pd(%s, %s))"
      l
      l
      v
      (herm_ymask ())
  | AZoutLeg l when !colstride && !herm = 1 && l = !herm_hl && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "{ const __m256d _hv = %s; const size_t _dk = _d0 + (size_t)k*OGs, _dk1 = _d0 + ((size_t)k + 1)*OGs; \
       if (2*_dk < OLs) _mm_storeu_pd(&zout[2*((size_t)%d*OLs + (size_t)k*OGs)], _mm256_castpd256_pd128(_hv)); \
       else _mm_storeu_pd(zout_unused - 2*(_dk + (size_t)%d*OLs), _mm_xor_pd(_mm256_castpd256_pd128(_hv), _M_IM_n)); \
       if (2*_dk1 < OLs) _mm_storeu_pd(&zout[2*((size_t)%d*OLs + ((size_t)k + 1)*OGs)], _mm256_extractf128_pd(_hv, 1)); \
       else _mm_storeu_pd(zout_unused - 2*(_dk1 + (size_t)%d*OLs), _mm_xor_pd(_mm256_extractf128_pd(_hv, 1), _M_IM_n)); }"
      v l l l l
  | AZoutLeg l when !colstride && !herm = 1 && l > !herm_hl && isa.Isa.vec_width = 2 ->
    Printf.sprintf "_mm_storeu_pd(zout_unused - 2*(_d0 + (size_t)k*OGs + (size_t)%d*OLs), _mm_xor_pd(%s, _M_IM_n))" l v
  | AZoutLeg l when !colstride && !herm = 1 && l = !herm_hl && isa.Isa.vec_width = 2 ->
    Printf.sprintf
      "{ const __m128d _hv = %s; if (2*(_d0 + (size_t)k*OGs) < OLs) _mm_storeu_pd(&zout[2*((size_t)%d*OLs + (size_t)k*OGs)], _hv); \
       else _mm_storeu_pd(zout_unused - 2*(_d0 + (size_t)k*OGs + (size_t)%d*OLs), _mm_xor_pd(_hv, _M_IM_n)); }"
      v l l
  (* the Hermitian store edge at 512 bits: four columns k..k+3 at pitch OGs *)
  | AZoutLeg l when !colstride && !herm = 1 && l > !herm_hl && isa.Isa.vec_width = 8 ->
    let m j = Printf.sprintf "zout_unused - 2*(_d0 + ((size_t)k + %d)*OGs + (size_t)%d*OLs)" j l in
    Printf.sprintf
      "{ const __m512d _hv = _mm512_xor_pd(%s, _M_IM); \
       _mm256_storeu2_m128d(%s, %s, _mm512_castpd512_pd256(_hv)); \
       _mm256_storeu2_m128d(%s, %s, _mm512_extractf64x4_pd(_hv, 1)); }"
      v (m 1) (m 0) (m 3) (m 2)
  | AZoutLeg l when !colstride && !herm = 1 && l = !herm_hl && isa.Isa.vec_width = 8 ->
    let c j =
      let part = Isa.cx_part_pd isa "_hv" j in
      Printf.sprintf
        "{ const size_t _dk%d = _d0 + ((size_t)k + %d)*OGs; \
         if (2*_dk%d < OLs) _mm_storeu_pd(&zout[2*((size_t)%d*OLs + ((size_t)k + %d)*OGs)], %s); \
         else _mm_storeu_pd(zout_unused - 2*(_dk%d + (size_t)%d*OLs), _mm_xor_pd(%s, _M_IM_n)); } "
        j j j l j part j l part
    in
    Printf.sprintf "{ const __m512d _hv = %s; %s%s%s%s}" v (c 0) (c 1) (c 2) (c 3)
  | AZoutLeg l when !colstride && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm256_storeu2_m128d(&zout[2*((size_t)%d*OLs + ((size_t)k + 1)*OGs)], \
       &zout[2*((size_t)%d*OLs + (size_t)k*OGs)], %s)"
      l
      l
      v
  | AZoutLeg l when !colstride && isa.Isa.vec_width = 8 ->
    Printf.sprintf
      "{ const __m512d _cv = %s; _mm256_storeu2_m128d(&zout[2*((size_t)%d*OLs + ((size_t)k + 1)*OGs)], \
       &zout[2*((size_t)%d*OLs + (size_t)k*OGs)], _mm512_castpd512_pd256(_cv)); \
       _mm256_storeu2_m128d(&zout[2*((size_t)%d*OLs + ((size_t)k + 3)*OGs)], \
       &zout[2*((size_t)%d*OLs + ((size_t)k + 2)*OGs)], _mm512_extractf64x4_pd(_cv, 1)); }"
      v l l l l
  | AZoutLeg _ when !store128 && isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm_storeu_pd(&%s, _mm256_castpd256_pd128(%s)); _mm_storeu_pd(&%s + 2, \
       _mm256_extractf128_pd(%s, 1))"
      (addr_str a)
      v
      (addr_str a)
      v
  (* the real mono's backward: the real lane of each column *)
  | AZoutReal l when isa.Isa.vec_width = 4 ->
    Printf.sprintf
      "_mm_store_sd(&zout[(size_t)%d*OLs + k], _mm256_castpd256_pd128(%s)); _mm_store_sd(&zout[(size_t)%d*OLs + k + 1], _mm256_extractf128_pd(%s, 1))"
      l
      v
      l
      v
  | AZoutReal l when isa.Isa.vec_width = 2 -> Printf.sprintf "_mm_store_sd(&zout[(size_t)%d*OLs + k], %s)" l v
  | AZoutReal l when isa.Isa.vec_width = 8 ->
    (* the 4 real lanes gathered into the low half by one lane permute, one
       256-bit store *)
    Printf.sprintf
      "_mm256_storeu_pd(&zout[(size_t)%d*OLs + k], _mm512_castpd512_pd256(_mm512_permutexvar_pd(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), %s)))"
      l
      v
  | AZoutReal _ -> failwith "cx_render.render_store: AZoutReal renders at 512 / 256 / 128 bits only"
  | _ -> Isa.storeu_pd isa (addr_str a) v
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
let render
      ?(mode = Isa.LS_vector)
      ?(tw_vw = 0)
      ?(msuf = "")
      ?(name = fun t -> Printf.sprintf "z%d" t)
      ~(ctx : ctx)
      (isa : Isa.t)
      (tbl : consts)
      (e : t)
  : string
  =
  (* ?name resolves an operand tag to its CURRENT C variable name. Default is
     the SSA "z<tag>"; cx_spill overrides it so a reloaded value picks up its
     fresh reload name for all uses after the reload. Byte-identical when the
     default resolver is used (every existing caller). *)
  let twv = if tw_vw = 0 then isa.Isa.vec_width else tw_vw in
  let v (x : t) = name x.tag in
  match e.node with
  | CIn _ -> failwith "codelet_cil.render: CIn is emitted by the load edge"
  | CLoad a -> Isa.loadu_pd ~mode isa (addr_str a)
  | CStore _ -> failwith "codelet_cil.render: CStore is a statement — use render_store"
  | CTurn (a, b, odd) -> Isa.cx_deint_pd isa ~odd (v a) (v b)
  | CUnpack (a, b, hi) ->
    (match isa.Isa.vec_width with
     | 8 -> Printf.sprintf "_mm512_unpack%s_pd(%s, %s)" (if hi then "hi" else "lo") (v a) (v b)
     | 4 -> Printf.sprintf "_mm256_unpack%s_pd(%s, %s)" (if hi then "hi" else "lo") (v a) (v b)
     | 2 -> Printf.sprintf "_mm_unpack%s_pd(%s, %s)" (if hi then "hi" else "lo") (v a) (v b)
     | w -> failwith (Printf.sprintf "cx_render: no unpack at vec_width %d" w))
  | CPart (a, c) -> Isa.cx_part_pd isa (v a) c
  | CAdd (a, { node = CRotNI y; _ }) when ctx.rotfma ->
    (* a + (-i)·y = a - [-1,+1]·cflip y : fnmadd on the (1,1) pair's _s. *)
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 1.0 in
    Isa.fnmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v a)
  | CSub (a, { node = CRotNI y; _ }) when ctx.rotfma ->
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 1.0 in
    Isa.fmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v a)
  (* the +i mirror (2026-09-11, the backward wing combine): (+i)·y =
     [-1,+1]·cflip y, so a + (+i)·y is the fmadd and a - (+i)·y the fnmadd —
     the same exact constant, the opposite opcode. BIT-EXACT as above. *)
  | CAdd (a, { node = CRotPI y; _ }) when ctx.rotfma ->
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 1.0 in
    Isa.fmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v a)
  | CSub (a, { node = CRotPI y; _ }) when ctx.rotfma ->
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 1.0 in
    Isa.fnmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v a)
  | CAdd (a, b) -> Isa.add_pd isa (v a) (v b)
  | CSub (a, b) -> Isa.sub_pd isa (v a) (v b)
  (* complex negation: flip both lanes' signs — same rendering the real side
     uses for NK_Neg (xor with a -0.0 broadcast; no named mask needed). *)
  | CNeg a -> Isa.xor_mask_pd isa (v a) (Isa.set1_pd_str isa "-0.0")
  | CRotNI a -> Isa.xor_mask_pd isa (Isa.cflip_pd isa (v a)) ("_M_IM" ^ msuf)
  | CRotPI a -> Isa.xor_mask_pd isa (Isa.cflip_pd isa (v a)) ("_M_RE" ^ msuf)
  | CRotAdd (a, y) ->
    (* a + i*y fused: addsub(a, cflip y) — 2 uops, no mask. Wider ISAs
       compose the value-identical fwd-legal form a - (-i*y). *)
    if isa.Isa.vec_width <= 4
    then Isa.addsub_pd isa (v a) (Isa.cflip_pd isa (v y))
    else
      Isa.sub_pd isa (v a) (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v y)) ("_M_IM" ^ msuf))
  | CFmaC (c, { node = CRotNI y; _ }, acc) when ctx.rotfma ->
    (* acc + c·(-i·y) = acc - [-c,+c]·cflip y — the [c,-c] hand idiom. *)
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 c in
    Isa.fnmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v acc)
  | CFnmaC (c, { node = CRotNI y; _ }, acc) when ctx.rotfma ->
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 c in
    Isa.fmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v acc)
  | CFmaC (c, { node = CRotPI y; _ }, acc) when ctx.rotfma ->
    (* acc + c·(+i·y) = acc + [-c,+c]·cflip y *)
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 c in
    Isa.fmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v acc)
  | CFnmaC (c, { node = CRotPI y; _ }, acc) when ctx.rotfma ->
    let w = const_name tbl (isa.Isa.vec_width / 2) 1.0 c in
    Isa.fnmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v y)) (v acc)
  | CFmaC (c, x, acc) ->
    Isa.fmadd_pd isa (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)) (v x) (v acc)
  | CFnmaC (c, x, acc) ->
    Isa.fnmadd_pd isa (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)) (v x) (v acc)
  | CTwC (c, s, x) when c = 1.0 ->
    (* Unit-cosine twiddle = a tangent SHEAR x·(1 + i·s). The c-multiply is
       an exact identity, so fold it away: fmadd(_s, cflip x, x) — 2 uops,
       no mask, no standalone mul. fma with an exact ±s product rounds the
       sum once, same as the add it replaces. Serves the tangent variant's
       shears (cx_math butterfly_pair); classic emissions never build a
       c=1 CTwC, so flag-off output is untouched. *)
    let w = const_name tbl (isa.Isa.vec_width / 2) c s in
    Isa.fmadd_pd isa (w ^ "_s") (Isa.cflip_pd isa (v x)) (v x)
  | CTwC (c, s, x) when s = 0.0 ->
    (* A REAL scale c*x (2026-09-23): the seed term of the odd DFT's plain
       sine chain (cx_math dft_cx_odd_pair). One multiply; the entry's _s
       half is a zero vector nothing references. *)
    let w = const_name tbl (isa.Isa.vec_width / 2) c s in
    Isa.mul_pd isa (w ^ "_c") (v x)
  | CTwC (c, s, x) ->
    let w = const_name tbl (isa.Isa.vec_width / 2) c s in
    Isa.fmadd_pd isa (w ^ "_c") (v x) (Isa.mul_pd isa (w ^ "_s") (Isa.cflip_pd isa (v x)))
  | CTwV (ws, x) ->
    let w = const_name_v tbl ws in
    Isa.fmadd_pd isa (w ^ "_c") (v x) (Isa.mul_pd isa (w ^ "_s") (Isa.cflip_pd isa (v x)))
  | CTwL (leg, x) ->
    (* BYTW2 against the VTW2 record for this leg. Under FLAT the record is
       loaded inline from the streamed cursor; under LOG3 it is a name bound
       by the prologue (loaded for power-of-two legs, DERIVED otherwise), so
       the flat path emits byte-identically to before. *)
    let c, s =
      if ctx.tw_log3 || ctx.tw_group || ctx.tw_gen2
      then Printf.sprintf "_wc%d%s" leg msuf, Printf.sprintf "_ws%d%s" leg msuf
      else (
        let off = (leg - 1) * 2 * twv in
        ( Isa.loadu_pd ~mode isa (addr_str (ATw off))
        , Isa.loadu_pd ~mode isa (addr_str (ATw (off + twv))) ))
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
      ?(mode = Isa.LS_vector)
      ?(tw_vw = 0)
      ?(msuf = "")
      (buf : Buffer.t)
      (isa : Isa.t)
      (radix : int)
  : unit
  =
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
              (Isa.const_decl isa cj (Isa.loadu_pd ~mode isa (addr_str (ATw off))))
              (Isa.const_decl isa sj (Isa.loadu_pd ~mode isa (addr_str (ATw (off + vw))))))
       | Some (p, q) ->
         let cp = Printf.sprintf "_wc%d%s" p msuf
         and sp = Printf.sprintf "_ws%d%s" p msuf
         and cq = Printf.sprintf "_wc%d%s" q msuf
         and sq = Printf.sprintf "_ws%d%s" q msuf in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl isa cj (Isa.fnmadd_pd isa sp sq (Isa.mul_pd isa cp cq)))
              (Isa.const_decl isa sj (Isa.fmadd_pd isa cp sq (Isa.mul_pd isa sp cq)))))
    (log3_plan radix);
  Buffer.add_string
    buf
    (Printf.sprintf
       "        /* log3: %d of %d VTW2 records loaded, %d derived */\n"
       !nload
       (radix - 1)
       (radix - 1 - !nload))
;;

(* ── GEN2 twiddle prologue (t2csg) ───────────────────────────────────────
 * The stream is generated, not loaded. Per column pair the W^1 record is
 *   (_wc1, _ws1) = T1[pair] (x) T2            (one record cmul, 4 ops)
 * with T1 the per-pair record at the cursor (tw_re: [c0,c0,c1,c1]
 * [-s0,+s0,-s1,+s1] over the pair's two columns) and T2 the per-call
 * broadcast record hoisted from tw_im into _wgc/_wgs. Every higher leg is
 * derived from W^1 by the PowW1 plan: W^(2m) = (W^m)^2, W^j = W^p . W^(j-p),
 * p the highest pow2 <= j — record cmuls, folded format preserved. *)
let gen2_plan (radix : int) : (int * (int * int)) list =
  let is_pow2 x = x > 0 && x land (x - 1) = 0 in
  let highest_pow2_le j =
    let rec go p = if p * 2 > j then p else go (p * 2) in
    go 1
  in
  let out = ref [] in
  for j = 2 to radix - 1 do
    if is_pow2 j
    then out := (j, (j / 2, j / 2)) :: !out
    else (
      let p = highest_pow2_le j in
      out := (j, (p, j - p)) :: !out)
  done;
  List.rev !out
;;

(* ── gen2 TWIDDLE POLICIES (2026-10-13) ─────────────────────────────────
 * Which legs are BASES — each a once-rounded T1_b x T2_b record pair, b the
 * base power — and how the other legs derive from them. "chain" is the
 * shipped tree above: W^1 alone, so leg j inherits j times the error of
 * W^1 whatever the tree. The others load more bases so a leg is a product
 * of at most a few of them:
 *   log3  bases min(3^i, R-1): leg j = W^x . W^(j-x) or conj(W^(x-j)) . W^x
 *   log2  bases 2^i: leg j = W^x . W^(j-x), x the highest pow2 below j
 *   sqrt  bases 1..k-1 and k, 2k, ... (k*k >= R-1): one product per leg
 *   full  every leg a base, nothing derived
 *   four3 exactly four log3 bases at every radix (raced 2026-10-13: never
 *         better than the per-radix table, 6-16% slower at the large primes)
 * Selected by VFFT_CX_GEN2POL (cx_ir.ml); the symbol carries the tag. *)
let int_pow (m : int) (e : int) : int =
  let rec go r e = if e = 0 then r else go (r * m) (e - 1) in
  go 1 e
;;

let gen2_pol_bases (pol : string) (radix : int) : int list =
  match pol with
  | "chain" -> [ 1 ]
  | "log3" ->
    let rec terms i pi s = if s >= radix - 1 then i else terms (i + 1) (3 * pi) (s + pi) in
    let n = terms 0 1 0 in
    List.init n (fun i -> min (int_pow 3 i) (radix - 1))
  | "log2" ->
    let rec go p acc = if p > radix - 1 then List.rev acc else go (p * 2) (p :: acc) in
    go 1 []
  | "sqrt" ->
    let k =
      let rec go k = if k * k >= radix - 1 then k else go (k + 1) in
      go 1
    in
    let low = List.init (min (k - 1) (radix - 1)) (fun i -> i + 1) in
    let rec high j acc = if j > radix - 1 then List.rev acc else high (j + k) (j :: acc) in
    low @ high k []
  | "full" -> List.init (radix - 1) (fun i -> i + 1)
  | "four3" ->
    (* exactly four log3 bases, the last capped at R-1 (a fixed record count) *)
    let rec go t acc =
      if t >= 4
      then List.rev acc
      else (
        let p = int_pow 3 t in
        if p >= radix - 1 then List.rev ((radix - 1) :: acc) else go (t + 1) (p :: acc))
    in
    go 0 []
  | _ -> failwith ("cx_render: unknown gen2 policy " ^ pol)
;;

let gen2_nbase (pol : string) (radix : int) : int = List.length (gen2_pol_bases pol radix)

(* The policy a radix SHIPS with: raced per radix on 2026-10-13 (every flat
   radix as the last stage of an L2-resident chain, two group sizes, the four
   accurate policies against the chain; docs/design/exact_trig_constants.md).
   log3 won at every radix up to 15 and at 17, 21, 25, 37; log2 at 16 and at
   the larger primes. TWIN of il2p.h vfft_il2p_gen2_bases, which fills the
   records the kernels read: the two tables must agree, base for base. *)
let gen2_default_policy (radix : int) : string =
  match radix with
  | 16 | 19 | 23 | 27 | 29 | 31 | 41 | 43 | 47 -> "log2"
  | _ -> "log3"
;;

(* the derivations in dependency order: (j, (a, b, conj)) reads
   W^j = (if conj then conj (W^a) else W^a) . W^b, a and b already formed. *)
let gen2_pol_plan (pol : string) (radix : int) : (int * (int * int * bool)) list =
  let bases = gen2_pol_bases pol radix in
  let formed = Hashtbl.create 64 in
  List.iter (fun b -> Hashtbl.replace formed b ()) bases;
  let out = ref [] in
  let add j a b cj =
    Hashtbl.replace formed j ();
    out := (j, (a, b, cj)) :: !out
  in
  (match pol with
   | "chain" -> List.iter (fun (j, (p, q)) -> add j p q false) (gen2_plan radix)
   | "log3" ->
     let rec form i =
       if i = 0 || Hashtbl.mem formed i
       then ()
       else (
         let x =
           let rec go x = if x >= i then x else go (x * 3) in
           go 1
         in
         if i + i >= x
         then (
           let x = min x (radix - 1) in
           form (x - i);
           form x;
           add i (x - i) x true)
         else (
           let x =
             let rec go x = if x * 3 <= i then go (x * 3) else x in
             go 1
           in
           form (i - x);
           form x;
           add i (i - x) x false))
     in
     for j = 1 to radix - 1 do
       form j
     done
   | "log2" ->
     let rec form i =
       if Hashtbl.mem formed i
       then ()
       else (
         let x =
           let rec go x = if x * 2 < i then go (x * 2) else x in
           go 1
         in
         form x;
         form (i - x);
         add i x (i - x) false)
     in
     for j = 1 to radix - 1 do
       form j
     done
   | "sqrt" ->
     let k =
       let rec go k = if k * k >= radix - 1 then k else go (k + 1) in
       go 1
     in
     for j = 1 to radix - 1 do
       if not (Hashtbl.mem formed j) then add j (j / k * k) (j mod k) false
     done
   | "full" -> ()
   | "four3" ->
     (* the pow3 split where the split's base exists, else the largest base below the leg *)
     let is_base i = List.mem i bases in
     let rec form i =
       if i = 0 || Hashtbl.mem formed i
       then ()
       else (
         let x =
           let rec go x = if x >= i then x else go (x * 3) in
           go 1
         in
         if i + i >= x && is_base x
         then (
           form (x - i);
           form x;
           add i (x - i) x true)
         else (
           let lb = List.fold_left (fun m b -> if b <= i && b > m then b else m) 0 bases in
           form (i - lb);
           form lb;
           add i (i - lb) lb false))
     in
     for j = 1 to radix - 1 do
       form j
     done
   | _ -> failwith ("cx_render: unknown gen2 policy " ^ pol));
  List.rev !out
;;

(* the prologue of a policy other than "chain": base t loads its T1 record at
   the cursor (slot t) and forms W^b = T1_t x T2_t against the hoisted
   _wgc{t}/_wgs{t} record; then the derived legs, folded format preserved. *)
let emit_gen2_pol_prologue
      ~(mode : Isa.ls_mode)
      ~(vw : int)
      ~(msuf : string)
      (buf : Buffer.t)
      (isa : Isa.t)
      (radix : int)
      (pol : string)
  : unit
  =
  let nm j = Printf.sprintf "_wc%d%s" j msuf, Printf.sprintf "_ws%d%s" j msuf in
  let bases = gen2_pol_bases pol radix in
  List.iteri
    (fun t b ->
       let tc = Printf.sprintf "_t1c%d%s" t msuf
       and ts = Printf.sprintf "_t1s%d%s" t msuf
       and gc = Printf.sprintf "_wgc%d%s" t msuf
       and gs = Printf.sprintf "_wgs%d%s" t msuf in
       let cb, sb = nm b in
       let off = t * 2 * vw in
       Buffer.add_string
         buf
         (Printf.sprintf
            "        %s\n        %s\n        %s\n        %s\n"
            (Isa.const_decl isa tc (Isa.loadu_pd ~mode isa (addr_str (ATw off))))
            (Isa.const_decl isa ts (Isa.loadu_pd ~mode isa (addr_str (ATw (off + vw)))))
            (Isa.const_decl isa cb (Isa.fnmadd_pd isa ts gs (Isa.mul_pd isa tc gc)))
            (Isa.const_decl isa sb (Isa.fmadd_pd isa tc gs (Isa.mul_pd isa ts gc)))))
    bases;
  List.iter
    (fun (j, (a, b, cj)) ->
       let cjn, sjn = nm j
       and ca, sa = nm a
       and cb, sb = nm b in
       Buffer.add_string
         buf
         (Printf.sprintf
            "        %s\n        %s\n"
            (if cj
             then Isa.const_decl isa cjn (Isa.fmadd_pd isa sa sb (Isa.mul_pd isa ca cb))
             else Isa.const_decl isa cjn (Isa.fnmadd_pd isa sa sb (Isa.mul_pd isa ca cb)))
            (if cj
             then Isa.const_decl isa sjn (Isa.fnmadd_pd isa sa cb (Isa.mul_pd isa ca sb))
             else Isa.const_decl isa sjn (Isa.fmadd_pd isa ca sb (Isa.mul_pd isa sa cb)))))
    (gen2_pol_plan pol radix);
  Buffer.add_string
    buf
    (Printf.sprintf
       "        /* gen2/%s: %d bases = T1[pair] x T2, %d legs derived */\n"
       pol
       (List.length bases)
       (radix - 1 - List.length bases))
;;

let emit_gen2_prologue
      ?(mode = Isa.LS_vector)
      ?(tw_vw = 0)
      ?(msuf = "")
      ?(pol = "chain")
      (buf : Buffer.t)
      (isa : Isa.t)
      (radix : int)
  : unit
  =
  let vw = if tw_vw = 0 then isa.Isa.vec_width else tw_vw in
  if pol <> "chain"
  then emit_gen2_pol_prologue ~mode ~vw ~msuf buf isa radix pol
  else (
  let nm j = Printf.sprintf "_wc%d%s" j msuf, Printf.sprintf "_ws%d%s" j msuf in
  let c1, s1 = nm 1 in
  let tc = Printf.sprintf "_t1c%s" msuf
  and ts = Printf.sprintf "_t1s%s" msuf
  and gc = Printf.sprintf "_wgc%s" msuf
  and gs = Printf.sprintf "_wgs%s" msuf in
  Buffer.add_string
    buf
    (Printf.sprintf
       "        %s
        %s
        %s
        %s
"
       (Isa.const_decl isa tc (Isa.loadu_pd ~mode isa (addr_str (ATw 0))))
       (Isa.const_decl isa ts (Isa.loadu_pd ~mode isa (addr_str (ATw vw))))
       (Isa.const_decl isa c1 (Isa.fnmadd_pd isa ts gs (Isa.mul_pd isa tc gc)))
       (Isa.const_decl isa s1 (Isa.fmadd_pd isa tc gs (Isa.mul_pd isa ts gc))));
  List.iter
    (fun (j, (p, q)) ->
       let cj, sj = nm j and cp, sp = nm p and cq, sq = nm q in
       Buffer.add_string
         buf
         (Printf.sprintf
            "        %s
        %s
"
            (Isa.const_decl isa cj (Isa.fnmadd_pd isa sp sq (Isa.mul_pd isa cp cq)))
            (Isa.const_decl isa sj (Isa.fmadd_pd isa cp sq (Isa.mul_pd isa sp cq)))))
    (gen2_plan radix);
  Buffer.add_string
    buf
    (Printf.sprintf "        /* gen2: W^1 = T1[pair] x T2, %d legs derived */
" (radix - 2)))
;;

(* ── GROUP twiddle prologue (t2c) ────────────────────────────────────────
 * Load EVERY leg's record into the _wc/_ws names — per-(d,leg) records
 * hoisted out of the column loop and reused across the whole tile (the
 * z-T1S broadcast sourcing, il_native_design.md §6c, consumed by the 2D
 * column stage). Same record format and slot layout as flat/log3
 * (slot = leg-1, [c x vw][sign-folded s x vw]), no derivation. ?tw_vw /
 * ?msuf as in emit_log3_prologue: the narrow tail re-binds _wc%d_n names
 * at Isa.sse2 against the WIDE-geometry table. *)
let emit_group_prologue
      ?(tw_vw = 0)
      ?(msuf = "")
      (buf : Buffer.t)
      (isa : Isa.t)
      (radix : int)
  : unit
  =
  let vw = if tw_vw = 0 then isa.Isa.vec_width else tw_vw in
  for j = 1 to radix - 1 do
    let cj = Printf.sprintf "_wc%d%s" j msuf
    and sj = Printf.sprintf "_ws%d%s" j msuf in
    let off = (j - 1) * 2 * vw in
    Buffer.add_string
      buf
      (Printf.sprintf
         "        %s\n        %s\n"
         (Isa.const_decl isa cj (Isa.loadu_pd isa (addr_str (ATw off))))
         (Isa.const_decl isa sj (Isa.loadu_pd isa (addr_str (ATw (off + vw))))))
  done
;;
