(* abi.ml — M4.  See abi.mli.  The param lists below are a TRANSCRIPTION of
   the 13-arm ladder (emit_c.ml, historical :1446-1911) — proven equivalent
   by VFFT_ABI_XCHECK over every emission before the ladder is deleted. *)

type shape =
  | Strided of
      { il : [ `None | `In | `Out ]
      ; r2c : [ `No | `Fwd | `Bwd ]
      }
  | In_place of { il : [ `None | `In | `Out ] }
  | Twidsq
  | R2cb
  | R2cf
  | R2c_term_ls
  | R2c_term of { rt : bool }
  | Hc2c_nat of { ranged : bool }
  | Hc2c_nat_bwd of { ranged : bool }
  | Hc_strided of { ranged : bool }
  | N1_oop_strided
  | R2r
  | Oop_generic

type t =
  { symbol : string
  ; target_attr : string
  ; params : Layout.param list
  }

let pt plane ~const ~prefix = Layout.pointers plane ~const ~prefix ~twin:false ()
let sc ctype name = [ Layout.scalar ~ctype ~name ]
let tw = Layout.tw_pair ~restrict_:true
let split_in p = pt Layout.Split ~const:true ~prefix:p
let split_out p = pt Layout.Split ~const:false ~prefix:p
let real_in p = pt Layout.Real ~const:true ~prefix:p
let real_out p = pt Layout.Real ~const:false ~prefix:p
let inter_in p = pt Layout.Inter ~const:true ~prefix:p
let inter_out p = pt Layout.Inter ~const:false ~prefix:p

let params_of_shape = function
  | Strided { il; r2c } ->
    let data =
      match il, r2c with
      | `In, `No -> inter_in "in" @ split_out "rio"
      | `Out, `No -> split_in "rio" @ inter_out "out"
      | `None, `Fwd -> real_in "rio" @ split_out "out"
      | `None, `Bwd -> split_in "in" @ real_out "out"
      | `None, `No -> split_out "rio"
      | (`In | `Out), (`Fwd | `Bwd) ->
        invalid_arg "Abi: strided il + r2c (the banned combination)"
    in
    let strides =
      match r2c with
      | `Bwd -> sc "size_t" "in_stride" @ sc "size_t" "row_stride_in"
      | `Fwd -> sc "size_t" "row_stride_in" @ sc "size_t" "out_stride"
      | `No -> sc "size_t" "row_stride"
    in
    data @ tw @ strides @ sc "size_t" "me"
  | In_place { il } ->
    let data =
      match il with
      | `In -> inter_in "in" @ split_out "rio"
      | `Out -> split_in "rio" @ inter_out "out"
      | `None -> split_out "rio"
    in
    data @ tw @ sc "size_t" "ios" @ sc "size_t" "me"
  | Twidsq ->
    split_in "in"
    @ split_out "out"
    @ tw
    @ sc "size_t" "is"
    @ sc "size_t" "os"
    @ sc "size_t" "V"
  | R2cb ->
    split_in "in"
    @ real_out "out_re"
    @ sc "ptrdiff_t" "is_re"
    @ sc "ptrdiff_t" "is_im"
    @ sc "ptrdiff_t" "os_re"
    @ sc "size_t" "vl"
  | R2cf ->
    real_in "in_re"
    @ real_out "out_re"
    @ real_out "out_im"
    @ sc "ptrdiff_t" "is"
    @ sc "ptrdiff_t" "os_re"
    @ sc "ptrdiff_t" "os_im"
    @ sc "size_t" "vl"
  | R2c_term_ls ->
    split_in "ink"
    @ split_in "inm"
    @ split_out "Xp"
    @ split_out "Xm"
    @ tw
    @ sc "ptrdiff_t" "is_leg"
    @ sc "ptrdiff_t" "osp"
    @ sc "ptrdiff_t" "osm"
    @ sc "size_t" "vl"
  | R2c_term { rt } ->
    split_in "in"
    @ split_out "Xp"
    @ split_out "Xm"
    @ (if rt then tw else [])
    @ sc "ptrdiff_t" "is"
    @ sc "size_t" "vl"
  | Hc2c_nat { ranged } ->
    split_in "in"
    @ real_out "Rp"
    @ real_out "Ip"
    @ real_out "Rm"
    @ real_out "Im"
    @ tw
    @ sc "ptrdiff_t" "is"
    @ sc "ptrdiff_t" "osp"
    @ sc "ptrdiff_t" "osm"
    @ (if ranged
       then sc "ptrdiff_t" "cs_in" @ sc "ptrdiff_t" "cs_out" @ sc "int" "kcount"
       else [])
    @ sc "size_t" "vl"
  | Hc2c_nat_bwd { ranged } ->
    real_in "Rp"
    @ real_in "Ip"
    @ real_in "Rm"
    @ real_in "Im"
    @ split_out "out"
    @ tw
    @ sc "ptrdiff_t" "isp"
    @ sc "ptrdiff_t" "ism"
    @ sc "ptrdiff_t" "os"
    @ (if ranged
       then sc "ptrdiff_t" "cs_in" @ sc "ptrdiff_t" "cs_out" @ sc "int" "kcount"
       else [])
    @ sc "size_t" "vl"
  | Hc_strided { ranged } ->
    split_in "in"
    @ split_out "out"
    @ tw
    @ sc "ptrdiff_t" "is"
    @ sc "ptrdiff_t" "os"
    @ (if ranged
       then sc "ptrdiff_t" "cs_in" @ sc "ptrdiff_t" "cs_out" @ sc "int" "kcount"
       else [])
    @ sc "size_t" "vl"
  | N1_oop_strided ->
    split_in "in"
    @ split_out "out"
    @ sc "size_t" "is"
    @ sc "size_t" "os"
    @ sc "size_t" "vl"
  | R2r -> real_in "in" @ real_out "out" @ sc "size_t" "K"
  | Oop_generic -> split_in "in" @ split_out "out" @ tw @ sc "size_t" "K"
;;

let make ~symbol ~target_attr shape =
  { symbol; target_attr; params = params_of_shape shape }
;;

let signature t =
  let line (p : Layout.param) =
    Printf.sprintf
      "    %s%s%s"
      p.Layout.ctype
      (if p.Layout.restrict_ then "__restrict__ " else "")
      p.Layout.name
  in
  Printf.sprintf
    "__attribute__((target(\"%s\")))\nvoid %s(\n%s)\n{\n"
    t.target_attr
    t.symbol
    (String.concat ",\n" (List.map line t.params))
;;

(* The frozen 11-arg z ABI — see abi.mli.  A LITERAL, deliberately: the
   grouping is part of the frozen bytes, and one source replaces the
   twice-printed divergence risk (zsplit derived its silencers, cil
   hardcoded them — that asymmetry stays caller-side and out of scope). *)
let z11_signature ~symbol ~target_attr =
  String.concat
    ""
    [ Printf.sprintf "__attribute__((target(\"%s\")))\n" target_attr
    ; Printf.sprintf "void %s(\n" symbol
    ; "    const double * __restrict__ zin,\n"
    ; "    const double * __restrict__ zin_unused,\n"
    ; "    double       * __restrict__ zout,\n"
    ; "    double       * __restrict__ zout_unused,\n"
    ; "    const double * tw_re, const double * tw_im,\n"
    ; "    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n"
    ; "{\n"
    ]
;;
