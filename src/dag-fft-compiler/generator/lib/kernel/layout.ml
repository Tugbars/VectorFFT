(* layout.ml — M3.  See layout.mli: the anti-hybrid law lives here.  Rendering
   is byte-exact to the three historical printer sites it replaced
   (codelet_oop signature, emit_c in-place arm, emit_c strided chain) — the
   corpus gate is the proof. *)

type plane = Split | Inter | Inter_sw | Real

type buffers =
  | Rio of plane
  | From_z
  | To_z
  | Oop of { load : plane; store : plane }

type param =
  { ctype : string
  ; name : string
  ; restrict_ : bool
  ; silence : bool
  ; comment : string option
  }

let ct_const = "const double * "
let ct_mut = "double       * "

let scalar_ctypes = [ "size_t"; "int"; "ptrdiff_t"; "uint32_t"; "double" ]

let scalar ~ctype ~name =
  if not (List.mem ctype scalar_ctypes)
  then invalid_arg ("Layout.scalar: not a scalar C type: " ^ ctype);
  { ctype = ctype ^ " "; name; restrict_ = false; silence = false; comment = None }

let pointers plane ~const ~prefix ~twin ?comment () =
  let ct = if const then ct_const else ct_mut in
  let p ?(silence = false) ?comment name =
    { ctype = ct; name; restrict_ = true; silence; comment }
  in
  match plane with
  | Split -> [ p (prefix ^ "_re"); p (prefix ^ "_im") ]
  | Real -> [ p ?comment prefix ]
  | Inter | Inter_sw ->
    p ?comment (prefix ^ "_z")
    :: (if twin then [ p ~silence:true (prefix ^ "_unused") ] else [])

let tw_pair ~restrict_ =
  [ { ctype = ct_const; name = "tw_re"; restrict_; silence = false; comment = None }
  ; { ctype = ct_const; name = "tw_im"; restrict_; silence = false; comment = None }
  ]

(* Byte-exact line: 4-space indent, ctype (already trailing-spaced),
   optional __restrict__, name, comma; a comment is padded to column 48 —
   the alignment the historical printers used (verified: in_z pads 10,
   out_z pads 9, both landing at col 48). *)
let render p =
  let base =
    Printf.sprintf "    %s%s%s," p.ctype (if p.restrict_ then "__restrict__ " else "") p.name
  in
  match p.comment with
  | None -> base ^ "\n"
  | Some c ->
    let pad = max 1 (47 - String.length base) in
    base ^ String.make pad ' ' ^ c ^ "\n"

let silencer p = if p.silence then Some (Printf.sprintf "    (void)%s;\n" p.name) else None

let buffers_of_oop_bools ~il_in ~il_in_sw ~il_out ~il_out_sw =
  if il_in && il_in_sw
  then invalid_arg "Layout: --oop-il-in and --oop-il-in-sw are mutually exclusive";
  if il_out && il_out_sw
  then invalid_arg "Layout: --oop-il-out and --oop-il-out-sw are mutually exclusive";
  let side active sw = if sw then Inter_sw else if active then Inter else Split in
  Oop { load = side il_in il_in_sw; store = side il_out il_out_sw }

let ip_buffers_of_bools ~il_in ~il_out =
  if il_in && il_out
  then
    invalid_arg
      "Layout: --ip-il-in + --ip-il-out is the banned hybrid combination (one \
       in-place plane cannot be interleaved on both sides — use the OOP il \
       forms)";
  if il_in then From_z else if il_out then To_z else Rio Split
