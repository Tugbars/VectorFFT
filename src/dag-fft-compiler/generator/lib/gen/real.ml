(* real.ml — M8.3: the Real feature module (§9 #33): r2c · c2r ·
 * hc2hc/hc2c · the r2r trig zoo (318 files).  This tranche hosts the
 * family's STRIDED lattice arms + the hc_ranged trailer — the r2c math
 * §5.4 lists as accidental co-location in the emitter (§6a36 fused
 * conjugate split, §6a38/§6a45 merge prologues), moved VERBATIM from
 * emit_body.ml behind Emit_body.family_hooks.
 *
 * The thin preamble-ladder arms (body_preamble + v-loop glue, 3–10 lines
 * each) STAY in Emit_body: they are parameterized calls into shared
 * machinery — moving them drags sc/spill/assigns/anyk_tail/
 * emit_v_loop_header across the seam for zero de-duplication (§9's own
 * rule: a feature that differs only in ABI does not earn the content).
 *
 * gen_main routes every real-family cfg here; Emit_body fails LOUDLY if a
 * family-owned arm is reached without its hook. *)

(* the moved arms say cfg.Cfg.x — the same resolution emit_body gets from
   its `open Emit_render`, taken here as a narrow alias. *)
module Cfg = Emit_render.Cfg

let emit_codelet
      ~sc
      ~(cfg : Emit_render.Cfg.t)
      ~in_place
      ~t1s
      ~twidsq
      ~twidsq_n
      ~strided
      ~radix
      ~scheduler
      ~(isa : Isa.t)
      ~gh
      ~bb_budget
      ~spill
      ~is_log3
      deduped
      ~name
  =
  let hooks =
    { Emit_body.strided_prologue =
        Some
          (fun buf ->
    if cfg.Cfg.strided_r2c_bwd
    then (
      Buffer.add_string
        buf
        "    /* two-for-one c2r (\xc2\xa76a38 bwd emission): the merge prologue builds\n\
        \     * Z = X1 + i*X2 lane vectors from two half-spectra per lane; the c2c\n\
        \     * bwd body then yields z[n] whose Re/Im ARE the even/odd real rows,\n\
        \     * written by the standard store lattice via the pair shadow. me =\n\
        \     * PAIRS; in_stride >= N/2+1; output unnormalized (rows = N*x). */\n";
      Buffer.add_string buf "    double       * __restrict__ rio_re = out;\n";
      Buffer.add_string
        buf
        "    double       * __restrict__ rio_im = out + row_stride_in;\n";
      Buffer.add_string buf "    const size_t row_stride = 2 * row_stride_in;\n")
    else if cfg.Cfg.strided_r2c
    then (
      Buffer.add_string
        buf
        "    /* two-for-one r2c (\xc2\xa76a36 emission): even rows enter as re lanes,\n\
        \     * odd rows as im; row_stride is the PAIR stride; the fused split\n\
        \     * emits both rows' half-spectra. me = PAIRS; out_stride >= N/2+1.\n\
        \     * Reference: r16_r2c_fwd_strided.c (hand-written, -51.7%/-44.9%). */\n";
      Buffer.add_string buf "    const double * __restrict__ rio_re = rio;\n";
      Buffer.add_string
        buf
        "    const double * __restrict__ rio_im = rio + row_stride_in;\n";
      Buffer.add_string buf "    const size_t row_stride = 2 * row_stride_in;\n")
)
    ; strided_locals =
        Some
          (fun buf ->
    if cfg.Cfg.strided_r2c_bwd
    then
      for j = 0 to radix / 2 do
        Buffer.add_string
          buf
          (Printf.sprintf
             "        %s _hx1r_%d, _hx1i_%d, _hx2r_%d, _hx2i_%d;\n"
             isa.vec_type
             j
             j
             j
             j)
      done
)
    ; strided_load =
        Some
          (fun buf ->
            if isa.Isa.vec_width = 4
            then (
        (* \xc2\xa76a38 merge prologue: transposing-load the four half-plane
           bin-blocks (X1 = even rows, X2 = odd; re/im planes), then
           Z[f] = X1[f] + i*X2[f] and the Hermitian mirror
           Z[n-f] = conj(X1[f]) + i*conj(X2[f]). General formula absorbs
           DC/Nyquist via the zero-imag contract. All lane_* covered. *)
        let n = radix in
        let h = n / 2 in
        if n mod 2 <> 0 then failwith "strided-r2c bwd requires even radix";
        let full = h / 4 in
        for c = 0 to full - 1 do
          let j0 = c * 4 in
          List.iter
            (fun (pl, roff, nm) ->
               Buffer.add_string
                 buf
                 (Printf.sprintf
                    "        {  /* merge load group: %s bins %d..%d */\n"
                    nm
                    j0
                    (j0 + 3));
               for r = 0 to 3 do
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m256d _mr%d = \
                       _mm256_loadu_pd(&%s[(2*(b+%d)+%d)*in_stride + %d]);\n"
                      r
                      pl
                      r
                      roff
                      j0)
               done;
               for k = 0 to 3 do
                 let base = k / 2 * 2 in
                 let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m256d _mt%d = _mm256_%s_pd(_mr%d, _mr%d);\n"
                      k
                      op
                      base
                      (base + 1))
               done;
               for i = 0 to 3 do
                 let ta = i mod 2 in
                 let tb = 2 + (i mod 2) in
                 let imm = if i < 2 then "0x20" else "0x31" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            %s_%d = _mm256_permute2f128_pd(_mt%d, _mt%d, %s);\n"
                      nm
                      (j0 + i)
                      ta
                      tb
                      imm)
               done;
               Buffer.add_string buf "        }\n")
            [ "in_re", 0, "_hx1r"
            ; "in_im", 0, "_hx1i"
            ; "in_re", 1, "_hx2r"
            ; "in_im", 1, "_hx2i"
            ]
        done;
        (* declare the half vectors (assigned in the groups above / tails below) *)
        ();
        for f = 4 * full to h do
          List.iter
            (fun (pl, roff, nm) ->
               Buffer.add_string
                 buf
                 (Printf.sprintf
                    "        %s_%d = _mm256_set_pd(%s[(2*(b+3)+%d)*in_stride + %d], \
                     %s[(2*(b+2)+%d)*in_stride + %d], %s[(2*(b+1)+%d)*in_stride + %d], \
                     %s[(2*(b+0)+%d)*in_stride + %d]);\n"
                    nm
                    f
                    pl
                    roff
                    f
                    pl
                    roff
                    f
                    pl
                    roff
                    f
                    pl
                    roff
                    f))
            [ "in_re", 0, "_hx1r"
            ; "in_im", 0, "_hx1i"
            ; "in_re", 1, "_hx2r"
            ; "in_im", 1, "_hx2i"
            ]
        done;
        for f = 0 to h do
          Buffer.add_string
            buf
            (Printf.sprintf
               "        lane_re_%d = _mm256_sub_pd(_hx1r_%d, _hx2i_%d);\n"
               f
               f
               f);
          Buffer.add_string
            buf
            (Printf.sprintf
               "        lane_im_%d = _mm256_add_pd(_hx1i_%d, _hx2r_%d);\n"
               f
               f
               f);
          if f >= 1 && f <= h - 1
          then (
            Buffer.add_string
              buf
              (Printf.sprintf
                 "        lane_re_%d = _mm256_add_pd(_hx1r_%d, _hx2i_%d);\n"
                 (n - f)
                 f
                 f);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "        lane_im_%d = _mm256_sub_pd(_hx2r_%d, _hx1i_%d);\n"
                 (n - f)
                 f
                 f))
        done
)
            else (
        (* \xc2\xa76a45: avx512 merge prologue — 8x8 transposing loads of the
           four half-planes, then Z = X1 + i*X2 with Hermitian mirrors. *)
        let n = radix in
        let h = n / 2 in
        if n mod 2 <> 0 then failwith "strided-r2c bwd requires even radix";
        let full = h / 8 in
        for c = 0 to full - 1 do
          let j0 = c * 8 in
          List.iter
            (fun (pl, roff, nm) ->
               Buffer.add_string
                 buf
                 (Printf.sprintf
                    "        {  /* merge 8x8 load group: %s bins %d..%d */\n"
                    nm
                    j0
                    (j0 + 7));
               for r = 0 to 7 do
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m512d _mr%d = \
                       _mm512_loadu_pd(&%s[(2*(b+%d)+%d)*in_stride + %d]);\n"
                      r
                      pl
                      r
                      roff
                      j0)
               done;
               for k = 0 to 7 do
                 let base = k / 2 * 2 in
                 let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m512d _mt%d = _mm512_%s_pd(_mr%d, _mr%d);\n"
                      k
                      op
                      base
                      (base + 1))
               done;
               for k = 0 to 7 do
                 let ua = (k mod 4 mod 2) + (k / 4 * 4) in
                 let ub = ua + 2 in
                 let idx = if k mod 4 < 2 then "_tp_idx_lo" else "_tp_idx_hi" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m512d _mv%d = _mm512_permutex2var_pd(_mt%d, \
                       %s, _mt%d);\n"
                      k
                      ua
                      idx
                      ub)
               done;
               for i = 0 to 7 do
                 let va = if i < 4 then i else i - 4 in
                 let vb = if i < 4 then i + 4 else i in
                 let imm = if i < 4 then "0x44" else "0xEE" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            %s_%d = _mm512_shuffle_f64x2(_mv%d, _mv%d, %s);\n"
                      nm
                      (j0 + i)
                      va
                      vb
                      imm)
               done;
               Buffer.add_string buf "        }\n")
            [ "in_re", 0, "_hx1r"
            ; "in_im", 0, "_hx1i"
            ; "in_re", 1, "_hx2r"
            ; "in_im", 1, "_hx2i"
            ]
        done;
        for f = 8 * full to h do
          List.iter
            (fun (pl, roff, nm) ->
               Buffer.add_string
                 buf
                 (Printf.sprintf "        %s_%d = _mm512_set_pd(" nm f);
               for w = 7 downto 0 do
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "%s[(2*(b+%d)+%d)*in_stride + %d]%s"
                      pl
                      w
                      roff
                      f
                      (if w = 0 then ");\n" else ", "))
               done)
            [ "in_re", 0, "_hx1r"
            ; "in_im", 0, "_hx1i"
            ; "in_re", 1, "_hx2r"
            ; "in_im", 1, "_hx2i"
            ]
        done;
        for f = 0 to h do
          Buffer.add_string
            buf
            (Printf.sprintf
               "        lane_re_%d = _mm512_sub_pd(_hx1r_%d, _hx2i_%d);\n"
               f
               f
               f);
          Buffer.add_string
            buf
            (Printf.sprintf
               "        lane_im_%d = _mm512_add_pd(_hx1i_%d, _hx2r_%d);\n"
               f
               f
               f);
          if f >= 1 && f <= h - 1
          then (
            Buffer.add_string
              buf
              (Printf.sprintf
                 "        lane_re_%d = _mm512_add_pd(_hx1r_%d, _hx2i_%d);\n"
                 (n - f)
                 f
                 f);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "        lane_im_%d = _mm512_sub_pd(_hx2r_%d, _hx1i_%d);\n"
                 (n - f)
                 f
                 f))
        done
))
    ; strided_store =
        Some
          (fun buf ->
            if isa.Isa.vec_width = 4
            then (
      (* \xc2\xa76a36 emission: fused two-for-one conjugate split. x1 = even
         row's half-spectrum, x2 = odd row's; mirror g=(n-f) mod n makes
         f=0 and Nyquist self-mirroring. Every out_lane_* is consumed. *)
      let n = radix in
      let h = n / 2 in
      if n mod 2 <> 0 then failwith "strided-r2c requires even radix";
      Buffer.add_string buf "        const __m256d _half = _mm256_set1_pd(0.5);\n";
      for f = 0 to h do
        let g = if f = 0 then 0 else n - f in
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m256d x1r_%d = _mm256_mul_pd(_half, \
              _mm256_add_pd(out_lane_re_%d, out_lane_re_%d));\n"
             f
             f
             g);
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m256d x1i_%d = _mm256_mul_pd(_half, \
              _mm256_sub_pd(out_lane_im_%d, out_lane_im_%d));\n"
             f
             f
             g);
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m256d x2r_%d = _mm256_mul_pd(_half, \
              _mm256_add_pd(out_lane_im_%d, out_lane_im_%d));\n"
             f
             f
             g);
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m256d x2i_%d = _mm256_mul_pd(_half, \
              _mm256_sub_pd(out_lane_re_%d, out_lane_re_%d));\n"
             f
             g
             f)
      done;
      let full = h / 4 in
      for c = 0 to full - 1 do
        let j0 = c * 4 in
        List.iter
          (fun (xs, roff) ->
             List.iter
               (fun (pl, sfx) ->
                  Buffer.add_string
                    buf
                    (Printf.sprintf
                       "        {  /* r2c store group: x%s%s, bins %d..%d */\n"
                       xs
                       sfx
                       j0
                       (j0 + 3));
                  for k = 0 to 3 do
                    let base = j0 + (k / 2 * 2) in
                    let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
                    Buffer.add_string
                      buf
                      (Printf.sprintf
                         "            const __m256d _v%d = _mm256_%s_pd(x%s%s_%d, \
                          x%s%s_%d);\n"
                         k
                         op
                         xs
                         sfx
                         base
                         xs
                         sfx
                         (base + 1))
                  done;
                  for i = 0 to 3 do
                    let pa = i mod 2 in
                    let pb = 2 + (i mod 2) in
                    let imm = if i < 2 then "0x20" else "0x31" in
                    Buffer.add_string
                      buf
                      (Printf.sprintf
                         "            _mm256_storeu_pd(&%s[(2*(b+%d)+%d)*out_stride + \
                          %d], _mm256_permute2f128_pd(_v%d, _v%d, %s));\n"
                         pl
                         i
                         roff
                         j0
                         pa
                         pb
                         imm)
                  done;
                  Buffer.add_string buf "        }\n")
               [ "out_re", "r"; "out_im", "i" ])
          [ "1", 0; "2", 1 ]
      done;
      for f = 4 * full to h do
        Buffer.add_string
          buf
          (Printf.sprintf "        {  /* r2c scalar-tail bin %d */\n" f);
        Buffer.add_string buf "            double _q[16] __attribute__((aligned(32)));\n";
        Buffer.add_string
          buf
          (Printf.sprintf
             "            _mm256_store_pd(_q, x1r_%d); _mm256_store_pd(_q + 4, x1i_%d);\n"
             f
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "            _mm256_store_pd(_q + 8, x2r_%d); _mm256_store_pd(_q + 12, \
              x2i_%d);\n"
             f
             f);
        Buffer.add_string buf "            for (int _w = 0; _w < 4; _w++) {\n";
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_re[(2*(b+(size_t)_w))*out_stride + %d] = _q[_w];\n"
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_im[(2*(b+(size_t)_w))*out_stride + %d] = _q[4+_w];\n"
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_re[(2*(b+(size_t)_w)+1)*out_stride + %d] = _q[8+_w];\n"
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_im[(2*(b+(size_t)_w)+1)*out_stride + %d] = _q[12+_w];\n"
             f);
        Buffer.add_string buf "            }\n";
        Buffer.add_string buf "        }\n"
      done
)
            else (
      (* \xc2\xa76a45: avx512 r2c split postamble. Same formulas as the avx2
         edition on __m512d lane vectors (8 lanes = 8 PAIRS = 16 rows per
         block); stores via the 3-stage inverse (unpack -> permutex2var
         tp_idx -> shuffle_f64x2) in 8-bin chunks; scalar-tail bins incl.
         Nyquist via a 32-double spill. *)
      let n = radix in
      let h = n / 2 in
      if n mod 2 <> 0 then failwith "strided-r2c requires even radix";
      Buffer.add_string buf "        const __m512d _half = _mm512_set1_pd(0.5);\n";
      for f = 0 to h do
        let g = if f = 0 then 0 else n - f in
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m512d x1r_%d = _mm512_mul_pd(_half, \
              _mm512_add_pd(out_lane_re_%d, out_lane_re_%d));\n"
             f
             f
             g);
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m512d x1i_%d = _mm512_mul_pd(_half, \
              _mm512_sub_pd(out_lane_im_%d, out_lane_im_%d));\n"
             f
             f
             g);
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m512d x2r_%d = _mm512_mul_pd(_half, \
              _mm512_add_pd(out_lane_im_%d, out_lane_im_%d));\n"
             f
             f
             g);
        Buffer.add_string
          buf
          (Printf.sprintf
             "        const __m512d x2i_%d = _mm512_mul_pd(_half, \
              _mm512_sub_pd(out_lane_re_%d, out_lane_re_%d));\n"
             f
             g
             f)
      done;
      let full = h / 8 in
      for c = 0 to full - 1 do
        let j0 = c * 8 in
        List.iter
          (fun (xs, roff) ->
             List.iter
               (fun (pl, sfx) ->
                  Buffer.add_string
                    buf
                    (Printf.sprintf
                       "        {  /* r2c 8x8 store group: x%s%s, bins %d..%d */\n"
                       xs
                       sfx
                       j0
                       (j0 + 7));
                  for k = 0 to 7 do
                    let base = j0 + (k / 2 * 2) in
                    let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
                    Buffer.add_string
                      buf
                      (Printf.sprintf
                         "            const __m512d _u%d = _mm512_%s_pd(x%s%s_%d, \
                          x%s%s_%d);\n"
                         k
                         op
                         xs
                         sfx
                         base
                         xs
                         sfx
                         (base + 1))
                  done;
                  for k = 0 to 7 do
                    let ua = (k mod 4 mod 2) + (k / 4 * 4) in
                    let ub = ua + 2 in
                    let idx = if k mod 4 < 2 then "_tp_idx_lo" else "_tp_idx_hi" in
                    Buffer.add_string
                      buf
                      (Printf.sprintf
                         "            const __m512d _v%d = _mm512_permutex2var_pd(_u%d, \
                          %s, _u%d);\n"
                         k
                         ua
                         idx
                         ub)
                  done;
                  for i = 0 to 7 do
                    let va = if i < 4 then i else i - 4 in
                    let vb = if i < 4 then i + 4 else i in
                    let imm = if i < 4 then "0x44" else "0xEE" in
                    Buffer.add_string
                      buf
                      (Printf.sprintf
                         "            _mm512_storeu_pd(&%s[(2*(b+%d)+%d)*out_stride + \
                          %d], _mm512_shuffle_f64x2(_v%d, _v%d, %s));\n"
                         pl
                         i
                         roff
                         j0
                         va
                         vb
                         imm)
                  done;
                  Buffer.add_string buf "        }\n")
               [ "out_re", "r"; "out_im", "i" ])
          [ "1", 0; "2", 1 ]
      done;
      for f = 8 * full to h do
        Buffer.add_string
          buf
          (Printf.sprintf "        {  /* r2c scalar-tail bin %d */\n" f);
        Buffer.add_string buf "            double _q[32] __attribute__((aligned(64)));\n";
        Buffer.add_string
          buf
          (Printf.sprintf
             "            _mm512_store_pd(_q, x1r_%d); _mm512_store_pd(_q + 8, x1i_%d);\n"
             f
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "            _mm512_store_pd(_q + 16, x2r_%d); _mm512_store_pd(_q + 24, \
              x2i_%d);\n"
             f
             f);
        Buffer.add_string buf "            for (int _w = 0; _w < 8; _w++) {\n";
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_re[(2*(b+(size_t)_w))*out_stride + %d] = _q[_w];\n"
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_im[(2*(b+(size_t)_w))*out_stride + %d] = _q[8+_w];\n"
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_re[(2*(b+(size_t)_w)+1)*out_stride + %d] = _q[16+_w];\n"
             f);
        Buffer.add_string
          buf
          (Printf.sprintf
             "                out_im[(2*(b+(size_t)_w)+1)*out_stride + %d] = _q[24+_w];\n"
             f);
        Buffer.add_string buf "            }\n";
        Buffer.add_string buf "        }\n"
      done
))
    ; trailer =
        Some
          (fun buf ->
  if cfg.Cfg.hc_ranged
  then (
    let r = cfg.Cfg.hc_ranged_r in
    if cfg.Cfg.hc2c_natural_bwd
    then (
      (* mirror of the forward natural advance: the 4 SPLIT inputs walk like the
       * forward's split outputs (direct +, mirror -), the 2 PACKED outputs walk
       * like the forward's packed inputs (re +, im -). *)
      Buffer.add_string buf "    Rp += cs_in; Ip += cs_in;\n";
      Buffer.add_string buf "    Rm -= cs_in; Im -= cs_in;\n";
      Buffer.add_string buf "    out_re += cs_out; out_im -= cs_out;\n";
      Buffer.add_string buf (Printf.sprintf "    tw_re += %d; tw_im += %d;\n" r r))
    else if cfg.Cfg.hc2c_natural
    then (
      Buffer.add_string buf "    in_re += cs_in; in_im -= cs_in;\n";
      Buffer.add_string buf "    Rp += cs_out; Ip += cs_out;\n";
      Buffer.add_string buf "    Rm -= cs_out; Im -= cs_out;\n";
      Buffer.add_string buf (Printf.sprintf "    tw_re += %d; tw_im += %d;\n" r r))
    else (
      Buffer.add_string buf "    in_re += cs_in; in_im -= cs_in;\n";
      Buffer.add_string buf "    out_re += cs_out; out_im -= cs_out;\n";
      Buffer.add_string buf (Printf.sprintf "    tw_re += %d; tw_im += %d;\n" r r));
    Buffer.add_string buf "    }\n")
)
    }
  in
  Emit_body.emit_codelet
    ~hooks
    ~sc
    ~cfg
    ~in_place
    ~t1s
    ~twidsq
    ~twidsq_n
    ~strided
    ~radix
    ~scheduler
    ~isa
    ~gh
    ~bb_budget
    ~spill
    ~is_log3
    deduped
    ~name
;;
