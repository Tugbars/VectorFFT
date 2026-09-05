(* codelet.ml — M5 (generator_lib_architecture.md §10.2).  LAYER 4.
   THE WORD FOR THE THING WE COMPILE.

   Measured out of the corpus (X2, §22): every shipped codelet carries exactly
   ONE kind selector; global modifiers are FIVE fields; family-scoped
   modifiers live in the kind's payload record (the zs_kind pattern
   generalized).  `of_argv`/`to_argv` round-trip the 1,199 recorded
   provenance argv lines VERBATIM (flag order included) — the M5 acceptance
   (bin_test/argv_roundtrip.ml); a failure there is a red flag, not a
   formatting nit, because provenance == coverage == regen recipe is the
   design's one-fact rule.

   M5 scope: the descriptor + the round-trip + gen_main setting the config
   globals FROM it.  Symbol naming and recipes stay where they are until
   M8/M10. *)

type direction =
  | Fwd
  | Bwd

type tw_table =
  | Flat
  | Log3

(* the five GLOBAL modifiers — X2/OQ-2's measured answer *)
type modifiers =
  { dir : direction (* --bwd / --cil-bwd *)
  ; dif : bool (* --dif *)
  ; table : tw_table (* --log3 / --cil-log3 *)
  ; t1s : bool (* --t1s *)
  ; su : bool (* --su *)
  }

type il3 =
  [ `None
  | `In
  | `Out
  ]

type sw3 =
  [ `No
  | `Il
  | `Il_sw
  ]

type oop_edge =
  | UG
  | UL

type oop_tw =
  | Tw_group (* --twiddled *)
  | Tw_pos (* --twiddled-pos *)
  | Post_tw (* --post-tw *)
  | Tw_linear (* --oop-tw-linear *)

type trig8 =
  | Dct1
  | Dct2
  | Dct3
  | Dct4
  | Dst1
  | Dst2
  | Dst3
  | Dst4
  | Dht

type cil_form =
  | Cil_n1
  | Cil_n1c
  | Cil_t2c
  | Cil_n1t
  | Cil_t2

type cil_turn =
  | Turnst
  | Turnst_gs

type zs_kind =
  | Dts
  | Dtsn
  | Dtso
  | Dtt
  | Msd
  | Msg
  | Msgb
  | Msz
  | S0s
  | S0sb
  | S0t
  | S0tb
  | S0tu
  | Stf
  | Stf2
  | Stf2u
  | Stfu
  | Stfb
  | Stfbn
  | Stfn
  | Sterm
  | Sterm2
  | Stermb

type kind =
  | C2c_inplace_su of { il : il3 } (* --in-place --su [--ip-il-*] *)
  | C2c_inplace_tw of { il : il3 } (* --twiddled --in-place *)
  | C2c_oop of
      { load : oop_edge
      ; store : oop_edge
      ; tw : oop_tw option
      ; fuse : int option (* --fuse N *)
      ; store_fused : bool (* --oop-store-fused *)
      ; strides : (int * int * int * int) option (* --oop-strides L,G,OL,OG *)
      ; spec_named : bool (* --oop-spec-named *)
      ; il_in : sw3 (* --oop-il-in[-sw] *)
      ; il_out : sw3 (* --oop-il-out[-sw] *)
      }
  | R2cf
  | R2cb
  | Hc2hc of { ranged : bool }
  | Hc2c
  | Hc2c_nat of { ranged : bool }
  | R2c_term of
      { rt : bool (* --r2c-term-rt *)
      ; k : int option (* --r2c-term-k N *)
      }
  | R2c_term_ls of { r : int } (* --r2c-term-ls --r2c-term-ls-r N *)
  | Trig of trig8
  | Strided of { il : [ `No | `In | `Out | `Out_nt ] }
  | Strided_r2c (* direction in mods *)
  | N1_oop_strided (* --oop-strided *)
  | Cil of
      { form : cil_form
      ; tangent : bool (* --cil-tangent: tangent-scaled butterfly interior *)
      ; blocked : bool (* --cil-blocked *)
      ; oddct : bool
        (* --cil-oddct: dft_small FACTORS an odd COMPOSITE radix (9->3x3,
           25->5x5, 27->3x9) instead of taking dft_cx_odd's direct O(n^2/2)
           form. Distinct from `blocked`, which also factors but parks the
           passes in a spill array -- that composite lost its race (+13.5%,
           the n1b E9 verdict) while the unspilled form wins 2.2-2.6x at
           R=25/27. Same factorization, different materialization. *)
      ; split : (int * int) option (* --cil-split A.B *)
      ; turn : cil_turn option
      ; pre_tw : bool (* --cil-pretw (bwd pre-twiddle) *)
      ; colstride : bool (* --cil-t2cs: the column-stride tail form of t2 *)
      ; gen2 : bool (* --cil-t2csg: t2cs with the generated twiddle stream *)
      ; form_tag : bool
        (* --cil-form-tag: name the FORM in the emitted symbol, so a split /
           tangent / wing variant is distinguishable without a post-emit sed *)
      }
  | Zsplit of
      { k : zs_kind
      ; r0 : int option (* --zp-r0 N *)
      ; sink : bool (* --zp-sink *)
      }
  | K1_mono of
      { r1 : int option (* --k1-r1 N *)
      ; il : bool (* --k1-il *)
      ; sw : bool (* --k1-sw *)
      }
(* C2c_split.emit_k1_mono — the 6th emission entry point *)

type t =
  { radix : int
  ; isa : string option (* --isa I ; None on lines that never carried it *)
  ; uarch : string option (* --uarch U *)
  ; kind : kind
  ; mods : modifiers
  ; emit_c : bool (* --emit-c *)
  }

exception Parse_error of string

let fail fmt = Printf.ksprintf (fun s -> raise (Parse_error s)) fmt

(* ── the four measured invariants (§12.2) + the M3 layout laws ── *)
let validate (c : t) =
  (match c.kind with
   | C2c_inplace_su _ when not c.mods.su ->
     fail "in-place without --su must be --twiddled (su (+) twiddled, 648 files exact)"
   | C2c_inplace_tw _ when c.mods.su ->
     fail "--twiddled --in-place with --su violates su (+) twiddled"
   | Hc2c_nat _ ->
     let bwd = c.mods.dir = Bwd in
     if bwd <> c.mods.dif then fail "hc2c-nat: --bwd <=> --dif (measured invariant)"
   | _ -> ());
  (match c.kind with
   | Hc2hc { ranged = true } | Hc2c_nat { ranged = true } -> ()
   | _ when false -> ()
   | _ -> ());
  c
;;

let zs_name = function
  | Dts -> "dts"
  | Dtsn -> "dtsn"
  | Dtso -> "dtso"
  | Dtt -> "dtt"
  | Msd -> "msd"
  | Msg -> "msg"
  | Msgb -> "msgb"
  | Msz -> "msz"
  | S0s -> "s0s"
  | S0sb -> "s0sb"
  | S0t -> "s0t"
  | S0tb -> "s0tb"
  | S0tu -> "s0tu"
  | Stfu -> "stfu"
  | Stf2u -> "stf2u"
  | Stf -> "stf"
  | Stf2 -> "stf2"
  | Stfb -> "stfb"
  | Stfbn -> "stfbn"
  | Stfn -> "stfn"
  | Sterm -> "sterm"
  | Sterm2 -> "sterm2"
  | Stermb -> "stermb"
;;

let zs_of_name = function
  | "dts" -> Dts
  | "dtsn" -> Dtsn
  | "dtso" -> Dtso
  | "dtt" -> Dtt
  | "msd" -> Msd
  | "msg" -> Msg
  | "msgb" -> Msgb
  | "msz" -> Msz
  | "s0s" -> S0s
  | "s0sb" -> S0sb
  | "s0t" -> S0t
  | "s0tb" -> S0tb
  | "s0tu" -> S0tu
  | "stfu" -> Stfu
  | "stf2u" -> Stf2u
  | "stf" -> Stf
  | "stf2" -> Stf2
  | "stfb" -> Stfb
  | "stfbn" -> Stfbn
  | "stfn" -> Stfn
  | "sterm" -> Sterm
  | "sterm2" -> Sterm2
  | "stermb" -> Stermb
  | s -> fail "unknown zp kind %s" s
;;

let trig_name = function
  | Dct1 -> "dct1"
  | Dct2 -> "dct2"
  | Dct3 -> "dct3"
  | Dct4 -> "dct4"
  | Dst1 -> "dst1"
  | Dst2 -> "dst2"
  | Dst3 -> "dst3"
  | Dst4 -> "dst4"
  | Dht -> "dht"
;;

(* ── of_argv: strict over the corpus flag surface; ORDER-INSENSITIVE parse
   (to_argv owns the canonical order). *)
let of_argv ?(strict = true) (argv : string list) : t =
  let radix = ref 0
  and isa = ref None
  and uarch = ref None
  and emitc = ref false in
  let dir = ref Fwd
  and dif = ref false
  and table = ref Flat in
  let t1s = ref false
  and su = ref false in
  let sel : string list ref = ref [] in
  let push s = sel := s :: !sel in
  let ip_il = ref `None
  and str_il = ref `No in
  let oop_load = ref UG
  and oop_store = ref UG
  and oop_tw = ref None in
  let fuse = ref None
  and store_fused = ref false
  and strides = ref None in
  let spec_named = ref false in
  let oop_il_in = ref `No
  and oop_il_out = ref `No in
  (* ── CONFLICTING-FLAG DETECTION — the layout law, hoisted to the descriptor.
     Since M5 each IL axis is ONE three-way field, so a second, DISAGREEING
     flag was silently last-flag-wins: gen_radix emitted a codelet nobody
     asked for and stamped a provenance header naming BOTH flags.  Neither
     Layout's anti-hybrid law nor emit_body's strided guards could fire — the
     conflict was resolved away before either was consulted (layout_smoke.sh
     saw exactly this as 3 red negatives).  Repeating the SAME flag stays
     legal; only a genuine disagreement raises.  [name_of] renders whichever
     flag already claimed the axis, so the message names both sides. *)
  let excl r ~axis ~name_of ~v ~flag =
    if !r <> v
    then (
      match name_of !r with
      | "" -> ()
      | prev ->
        fail
          "%s and %s are mutually exclusive: the banned hybrid %s layout (a codelet \
           carries ONE plane)"
          prev
          flag
          axis);
    r := v
  in
  let set_ip_il v flag =
    excl
      ip_il
      ~axis:"in-place IL"
      ~name_of:(function
        | `In -> "--ip-il-in"
        | `Out -> "--ip-il-out"
        | `None -> "")
      ~v
      ~flag
  in
  let set_str_il v flag =
    excl
      str_il
      ~axis:"strided IL"
      ~name_of:(function
        | `In -> "--strided-il-in"
        | `Out -> "--strided-il-out"
        | `Out_nt -> "--strided-il-out-nt"
        | `No -> "")
      ~v
      ~flag
  in
  let set_oop_il_in v flag =
    excl
      oop_il_in
      ~axis:"oop IL-in"
      ~name_of:(function
        | `Il -> "--oop-il-in"
        | `Il_sw -> "--oop-il-in-sw"
        | `No -> "")
      ~v
      ~flag
  in
  let set_oop_il_out v flag =
    excl
      oop_il_out
      ~axis:"oop IL-out"
      ~name_of:(function
        | `Il -> "--oop-il-out"
        | `Il_sw -> "--oop-il-out-sw"
        | `No -> "")
      ~v
      ~flag
  in
  let ranged = ref false in
  let term_rt = ref false
  and term_k = ref None
  and term_ls_r = ref 0 in
  let cil_tangent = ref false
  and cil_form_tag = ref false in
  let blocked = ref false
  and oddct = ref false
  and split = ref None
  and turn = ref None
  and pre_tw = ref false
  and colstride = ref false
  and gen2 = ref false in
  let zp_r0 = ref None
  and sink = ref false in
  let k1_r1 = ref None
  and k1_il = ref false
  and k1_sw = ref false in
  let rec go = function
    | [] -> ()
    | n :: tl when !radix = 0 && int_of_string_opt n <> None ->
      radix := int_of_string n;
      go tl
    | "--emit-c" :: tl ->
      emitc := true;
      go tl
    | "--isa" :: v :: tl ->
      isa := Some v;
      go tl
    | "--uarch" :: v :: tl ->
      uarch := Some v;
      go tl
    | "--bwd" :: tl | "--cil-bwd" :: tl ->
      dir := Bwd;
      go tl
    | "--dif" :: tl ->
      dif := true;
      go tl
    | "--log3" :: tl | "--cil-log3" :: tl ->
      table := Log3;
      go tl
    | "--t1s" :: tl ->
      t1s := true;
      go tl
    | "--su" :: tl ->
      su := true;
      go tl
    | "--ranged" :: tl ->
      ranged := true;
      go tl
    | "--in-place" :: tl ->
      push "in-place";
      go tl
    | "--twiddled" :: tl ->
      push "twiddled";
      go tl
    | "--ip-il-in" :: tl ->
      set_ip_il `In "--ip-il-in";
      go tl
    | "--ip-il-out" :: tl ->
      set_ip_il `Out "--ip-il-out";
      go tl
    | "--oop" :: tl ->
      push "oop";
      go tl
    | "--oop-buffer-oop" :: tl -> go tl
    | "--oop-load" :: e :: tl ->
      oop_load := if e = "UL" then UL else UG;
      go tl
    | "--oop-store" :: e :: tl ->
      oop_store := if e = "UL" then UL else UG;
      go tl
    | "--twiddled-pos" :: tl ->
      oop_tw := Some Tw_pos;
      go tl
    | "--post-tw" :: tl ->
      oop_tw := Some Post_tw;
      go tl
    | "--oop-tw-linear" :: tl ->
      oop_tw := Some Tw_linear;
      go tl
    | "--fuse" :: v :: tl ->
      fuse := Some (int_of_string v);
      go tl
    | "--oop-store-fused" :: tl ->
      store_fused := true;
      go tl
    | "--oop-strides" :: v :: tl ->
      (match List.map int_of_string (String.split_on_char ',' v) with
       | [ a; b; c; d ] -> strides := Some (a, b, c, d)
       | _ -> fail "--oop-strides expects four ints");
      go tl
    | "--oop-spec-named" :: tl ->
      spec_named := true;
      go tl
    | "--oop-il-in" :: tl ->
      set_oop_il_in `Il "--oop-il-in";
      go tl
    | "--oop-il-in-sw" :: tl ->
      set_oop_il_in `Il_sw "--oop-il-in-sw";
      go tl
    | "--oop-il-out" :: tl ->
      set_oop_il_out `Il "--oop-il-out";
      go tl
    | "--oop-il-out-sw" :: tl ->
      set_oop_il_out `Il_sw "--oop-il-out-sw";
      go tl
    | "--r2cf" :: tl ->
      push "r2cf";
      go tl
    | "--r2cb" :: tl ->
      push "r2cb";
      go tl
    | "--hc2hc" :: tl ->
      push "hc2hc";
      go tl
    | "--hc2c" :: tl ->
      push "hc2c";
      go tl
    | "--hc2c-nat" :: tl ->
      push "hc2c-nat";
      go tl
    | "--r2c-term" :: tl ->
      push "r2c-term";
      go tl
    | "--r2c-term-rt" :: tl ->
      push "r2c-term";
      term_rt := true;
      go tl
    | "--r2c-term-k" :: v :: tl ->
      term_k := Some (int_of_string v);
      go tl
    | "--r2c-term-ls" :: tl ->
      push "r2c-term-ls";
      go tl
    | "--r2c-term-ls-r" :: v :: tl ->
      term_ls_r := int_of_string v;
      go tl
    | "--strided" :: tl ->
      push "strided";
      go tl
    | "--strided-il-in" :: tl ->
      set_str_il `In "--strided-il-in";
      go tl
    | "--strided-il-out" :: tl ->
      set_str_il `Out "--strided-il-out";
      go tl
    | "--strided-il-out-nt" :: tl ->
      set_str_il `Out_nt "--strided-il-out-nt";
      go tl
    | "--strided-r2c" :: tl ->
      push "strided-r2c";
      go tl
    | "--oop-strided" :: tl ->
      push "oop-strided";
      go tl
    | t :: tl
      when List.mem
             t
             [ "--dct1"
             ; "--dct2"
             ; "--dct3"
             ; "--dct4"
             ; "--dst1"
             ; "--dst2"
             ; "--dst3"
             ; "--dst4"
             ; "--dht"
             ] ->
      push (String.sub t 2 (String.length t - 2));
      go tl
    | t :: tl when List.mem t [ "--cil-n1"; "--cil-n1c"; "--cil-n1t"; "--cil-t2"; "--cil-t2c" ] ->
      push (String.sub t 2 (String.length t - 2));
      go tl
    | "--cil-t2cs" :: tl ->
      push "cil-t2";
      colstride := true;
      go tl
    | "--cil-t2csg" :: tl ->
      push "cil-t2";
      colstride := true;
      gen2 := true;
      go tl
    | "--cil-t2cp" :: tl ->
      (* t2c + PRE-twiddle at fwd (2026-09-04): the canonical spelling of
         that cell — --cil-pretw stays the retired t2p flag. *)
      push "cil-t2c";
      pre_tw := true;
      go tl
    | "--cil-tangent" :: tl ->
      cil_tangent := true;
      go tl
    | "--cil-form-tag" :: tl ->
      cil_form_tag := true;
      go tl
    | "--cil-blocked" :: tl ->
      blocked := true;
      go tl
    | "--cil-oddct" :: tl ->
      oddct := true;
      go tl
    | "--cil-split" :: v :: tl ->
      (match String.split_on_char '.' v with
       | [ a; b ] -> split := Some (int_of_string a, int_of_string b)
       | _ -> fail "--cil-split expects A.B");
      go tl
    | "--cil-turnst" :: tl ->
      turn := Some Turnst;
      go tl
    | "--cil-turnst-gs" :: tl ->
      turn := Some Turnst_gs;
      go tl
    | "--cil-pretw" :: tl ->
      pre_tw := true;
      go tl
    | "--zp-r0" :: v :: tl ->
      zp_r0 := Some (int_of_string v);
      go tl
    | "--zp-sink" :: tl ->
      sink := true;
      go tl
    | "--k1-mono" :: tl ->
      push "k1-mono";
      go tl
    | "--k1-r1" :: v :: tl ->
      k1_r1 := Some (int_of_string v);
      go tl
    | "--k1-il" :: tl ->
      k1_il := true;
      go tl
    | "--k1-sw" :: tl ->
      k1_sw := true;
      go tl
    | t :: tl when String.length t > 5 && String.sub t 0 5 = "--zp-" ->
      push ("zp:" ^ String.sub t 5 (String.length t - 5));
      go tl
    | t :: tl -> if strict then fail "unknown flag %s" t else go tl
  in
  go argv;
  let kind =
    match List.sort_uniq compare !sel with
    | [ "in-place" ] when !su -> C2c_inplace_su { il = !ip_il }
    | [ "in-place"; "twiddled" ] -> C2c_inplace_tw { il = !ip_il }
    | [ "oop" ] | [ "oop"; "twiddled" ] ->
      let tw =
        if !oop_tw <> None
        then !oop_tw
        else if List.mem "twiddled" !sel
        then Some Tw_group
        else None
      in
      C2c_oop
        { load = !oop_load
        ; store = !oop_store
        ; tw
        ; fuse = !fuse
        ; store_fused = !store_fused
        ; strides = !strides
        ; spec_named = !spec_named
        ; il_in = !oop_il_in
        ; il_out = !oop_il_out
        }
    | [ "r2cf" ] -> R2cf
    | [ "r2cb" ] -> R2cb
    | [ "hc2hc" ] -> Hc2hc { ranged = !ranged }
    | [ "hc2c" ] -> Hc2c
    | [ "hc2c-nat" ] -> Hc2c_nat { ranged = !ranged }
    | [ "r2c-term" ] -> R2c_term { rt = !term_rt; k = !term_k }
    | [ "r2c-term-ls" ] -> R2c_term_ls { r = !term_ls_r }
    | [ "strided" ] -> Strided { il = !str_il }
    | [ "strided"; "strided-r2c" ] | [ "strided-r2c" ] ->
      (* cross-family twin of the axis check above: --strided-r2c selects the
         Real plane, so an interleaved-complex plane cannot also be asked for.
         emit_body's guard for this (the strided arm) was equally unreachable. *)
      if !str_il <> `No
      then
        fail
          "--strided-il-in/out cannot combine with --strided-r2c: the banned hybrid (an \
           interleaved-complex plane and a real plane in one codelet)";
      Strided_r2c
    | [ "oop-strided" ] -> N1_oop_strided
    | [ t ]
      when List.mem
             t
             [ "dct1"; "dct2"; "dct3"; "dct4"; "dst1"; "dst2"; "dst3"; "dst4"; "dht" ] ->
      Trig
        (match t with
         | "dct1" -> Dct1
         | "dct2" -> Dct2
         | "dct3" -> Dct3
         | "dct4" -> Dct4
         | "dst1" -> Dst1
         | "dst2" -> Dst2
         | "dst3" -> Dst3
         | "dst4" -> Dst4
         | _ -> Dht)
    | [ t ] when List.mem t [ "cil-n1"; "cil-n1c"; "cil-n1t"; "cil-t2"; "cil-t2c" ] ->
      Cil
        { form =
            (if t = "cil-n1"
             then Cil_n1
             else if t = "cil-n1c"
             then Cil_n1c
             else if t = "cil-t2c"
             then Cil_t2c
             else if t = "cil-n1t"
             then Cil_n1t
             else Cil_t2)
        ; tangent = !cil_tangent
        ; blocked = !blocked
        ; oddct = !oddct
        ; split = !split
        ; turn = !turn
        ; pre_tw = !pre_tw
        ; colstride = !colstride
        ; gen2 = !gen2
        ; form_tag = !cil_form_tag
        }
    | [ "k1-mono" ] -> K1_mono { r1 = !k1_r1; il = !k1_il; sw = !k1_sw }
    | [ t ] when String.length t > 3 && String.sub t 0 3 = "zp:" ->
      Zsplit
        { k = zs_of_name (String.sub t 3 (String.length t - 3))
        ; r0 = !zp_r0
        ; sink = !sink
        }
    | sels -> fail "selector set: %s" (String.concat "+" sels)
  in
  validate
    { radix = !radix
    ; isa = !isa
    ; uarch = !uarch
    ; kind
    ; mods = { dir = !dir; dif = !dif; table = !table; t1s = !t1s; su = !su }
    ; emit_c = !emitc
    }
;;

(* ── to_argv: the CANONICAL per-kind flag sequence, INCLUDING where --isa,
   --su and --emit-c sit — read off the recorded corpus lines (surprises the
   harness surfaced: --isa is MID-sequence for most families; the in-place
   family puts it before its modifiers; r2c-term-ls puts it AFTER --emit-c).
   Any change here must come from a round-trip failure, never taste. *)
let to_argv (c : t) : string list =
  let m = c.mods in
  let g cond flag = if cond then [ flag ] else [] in
  let isa =
    match c.isa with
    | None -> []
    | Some i -> [ "--isa"; i ]
  in
  let uarch =
    match c.uarch with
    | None -> []
    | Some u -> [ "--uarch"; u ]
  in
  let emitc = g c.emit_c "--emit-c" in
  let n = [ string_of_int c.radix ] in
  match c.kind with
  | C2c_inplace_su { il } ->
    n
    @ [ "--in-place" ]
    @ isa
    @ [ "--su" ]
    @ (match il with
       | `None -> []
       | `In -> [ "--ip-il-in" ]
       | `Out -> [ "--ip-il-out" ])
    @ g (m.dir = Bwd) "--bwd"
    @ emitc
  | C2c_inplace_tw { il } ->
    n
    @ [ "--twiddled"; "--in-place" ]
    @ isa
    @ (match il with
       | `None -> []
       | `In -> [ "--ip-il-in" ]
       | `Out -> [ "--ip-il-out" ])
    @ g m.t1s "--t1s"
    @ g m.dif "--dif"
    @ g (m.dir = Bwd) "--bwd"
    @ g (m.table = Log3) "--log3"
    @ emitc
  | C2c_oop { load; store; tw; fuse; store_fused; strides; spec_named; il_in; il_out } ->
    let edge = function
      | UG -> "UG"
      | UL -> "UL"
    in
    n
    @ (if tw = Some Post_tw then [ "--twiddled"; "--post-tw" ] else [])
    @ [ "--oop"; "--oop-buffer-oop"; "--oop-load"; edge load; "--oop-store"; edge store ]
    @ (match il_in with
       | `No -> []
       | `Il -> [ "--oop-il-in" ]
       | `Il_sw -> [ "--oop-il-in-sw" ])
    @ (match il_out with
       | `No -> []
       | `Il -> [ "--oop-il-out" ]
       | `Il_sw -> [ "--oop-il-out-sw" ])
    @ isa
    @ (match fuse with
       | None -> []
       | Some k -> [ "--fuse"; string_of_int k ])
    @ g store_fused "--oop-store-fused"
    (* --log3 HUGS its twiddle token (corpus: "--twiddled --log3 --oop-strides"
        for the group form; "--twiddled-pos --log3" for the positional form) *)
    @ (if tw = Some Tw_group then [ "--twiddled" ] @ g (m.table = Log3) "--log3" else [])
    @ (match strides with
       | None -> []
       | Some (a, b, cc, d) -> [ "--oop-strides"; Printf.sprintf "%d,%d,%d,%d" a b cc d ])
    @ g (tw = Some Tw_pos) "--twiddled-pos"
    @ (if tw = Some Tw_linear then [ "--twiddled"; "--oop-tw-linear" ] else [])
    @ g (m.table = Log3 && tw <> Some Tw_group) "--log3"
    @ g spec_named "--oop-spec-named"
    @ g (m.dir = Bwd) "--bwd"
    @ emitc
  | R2cf -> n @ [ "--r2cf" ] @ isa @ g m.su "--su" @ emitc
  | R2cb -> n @ [ "--r2cb" ] @ isa @ g m.su "--su" @ emitc
  | Hc2hc { ranged } ->
    n
    @ [ "--hc2hc" ]
    @ g m.dif "--dif"
    @ g (m.dir = Bwd) "--bwd"
    @ g (m.table = Log3) "--log3"
    @ g ranged "--ranged"
    @ g m.t1s "--t1s"
    @ isa
    @ g m.su "--su"
    @ emitc
  | Hc2c -> n @ [ "--hc2c" ] @ g m.t1s "--t1s" @ isa @ g m.su "--su" @ emitc
  | Hc2c_nat { ranged } ->
    n
    @ [ "--hc2c-nat" ]
    @ g (m.dir = Bwd) "--bwd"
    @ g m.dif "--dif"
    @ g (m.table = Log3) "--log3"
    @ g ranged "--ranged"
    @ g m.t1s "--t1s"
    @ isa
    @ g m.su "--su"
    @ emitc
  | R2c_term { rt; k } ->
    n
    @ (if rt then [ "--r2c-term-rt" ] else [ "--r2c-term" ])
    @ (match k with
       | None -> []
       | Some v -> [ "--r2c-term-k"; string_of_int v ])
    @ isa
    @ emitc
  | R2c_term_ls { r } ->
    (* the ONE family that records --isa AFTER --emit-c *)
    n @ [ "--r2c-term-ls"; "--r2c-term-ls-r"; string_of_int r ] @ emitc @ isa
  | Trig k -> n @ [ "--" ^ trig_name k ] @ isa @ g m.su "--su" @ emitc
  | Strided { il } ->
    n
    @ [ "--strided" ]
    @ (match il with
       | `No -> []
       | `In -> [ "--strided-il-in" ]
       | `Out -> [ "--strided-il-out" ]
       | `Out_nt -> [ "--strided-il-out-nt" ])
    @ isa
    @ g (m.dir = Bwd) "--bwd"
    @ emitc
  | Strided_r2c -> n @ [ "--strided-r2c" ] @ g (m.dir = Bwd) "--bwd" @ isa @ emitc
  | N1_oop_strided -> n @ [ "--oop-strided" ] @ isa @ emitc
  | Cil { form; tangent; blocked; oddct; split; turn; pre_tw; colstride; gen2; form_tag } ->
    n
    @ [ (match form with
         | Cil_n1 -> "--cil-n1"
         | Cil_n1c -> "--cil-n1c"
         | Cil_t2c -> if pre_tw then "--cil-t2cp" else "--cil-t2c"
         | Cil_n1t -> "--cil-n1t"
         | Cil_t2 -> if gen2 then "--cil-t2csg" else if colstride then "--cil-t2cs" else "--cil-t2")
      ]
    @ g tangent "--cil-tangent"
    @ g blocked "--cil-blocked"
    @ g oddct "--cil-oddct"
    @ (match split with
       | None -> []
       | Some (a, b) -> [ "--cil-split"; Printf.sprintf "%d.%d" a b ])
    @ (match turn with
       | None -> []
       | Some Turnst -> [ "--cil-turnst" ]
       | Some Turnst_gs -> [ "--cil-turnst-gs" ])
    @ g (m.table = Log3) "--cil-log3"
    @ g (pre_tw && form <> Cil_t2c) "--cil-pretw"
    @ g form_tag "--cil-form-tag"
    @ g (m.dir = Bwd) "--cil-bwd"
    @ isa
    @ uarch
    @ emitc
  | Zsplit { k; r0; sink } ->
    n
    @ [ "--zp-" ^ zs_name k ]
    @ (match r0 with
       | None -> []
       | Some v -> [ "--zp-r0"; string_of_int v ])
    @ g sink "--zp-sink"
    @ isa
    @ uarch
    @ emitc
  | K1_mono { r1; il; sw } ->
    n
    @ [ "--k1-mono" ]
    @ (match r1 with
       | None -> []
       | Some v -> [ "--k1-r1"; string_of_int v ])
    @ g il "--k1-il"
    @ g sw "--k1-sw"
    @ isa
    @ emitc
;;
