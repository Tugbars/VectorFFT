(* simd.ml -- M7.  The SIMD transpose-lattice vocabulary (S10.2).
   The four FEATURE-BLIND plain arms of emit_c's strided load/store
   lattices, moved here byte-verbatim (M7's corrected scope: the
   r2c- and il-conditional arms stay in emit_c until M8 relocates
   them into their families).  Each function emits the per-side
   (re/im) transpose groups for one ISA width:
     load_transpose_4x4   AVX2    4x4  rows->lanes   (fwd)
     load_transpose_8x8   AVX-512 8x8  rows->lanes   (fwd, 3-stage)
     store_transpose_4x4  AVX2    4x4  lanes->rows   (inverse)
     store_transpose_8x8  AVX-512 8x8  lanes->rows   (inverse)
   [groups] = radix / vec_width, computed by the caller. *)

let load_transpose_4x4 ~buf ~groups =
  for which_side = 0 to 1 do
    let suf = if which_side = 0 then "re" else "im" in
    for g = 0 to groups - 1 do
      let j0 = g * 4 in
      Buffer.add_string
        buf
        (Printf.sprintf
           "        {  /* 4x4 transpose group: fft_idx %d..%d, %s */\n"
           j0
           (j0 + 3)
           suf);
      for r = 0 to 3 do
        Buffer.add_string
          buf
          (Printf.sprintf
             "            const __m256d _row_%s_%d = \
              _mm256_loadu_pd(&rio_%s[(b+%d)*row_stride + %d]);\n"
             suf
             r
             suf
             r
             j0)
      done;
      (* 4×4 transpose: r0 r1 r2 r3 (each row = 4 cols) → c0 c1 c2 c3 *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _t0_%s = _mm256_unpacklo_pd(_row_%s_0, _row_%s_1);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _t1_%s = _mm256_unpackhi_pd(_row_%s_0, _row_%s_1);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _t2_%s = _mm256_unpacklo_pd(_row_%s_2, _row_%s_3);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _t3_%s = _mm256_unpackhi_pd(_row_%s_2, _row_%s_3);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm256_permute2f128_pd(_t0_%s, _t2_%s, 0x20);\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm256_permute2f128_pd(_t1_%s, _t3_%s, 0x20);\n"
           suf
           (j0 + 1)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm256_permute2f128_pd(_t0_%s, _t2_%s, 0x31);\n"
           suf
           (j0 + 2)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm256_permute2f128_pd(_t1_%s, _t3_%s, 0x31);\n"
           suf
           (j0 + 3)
           suf
           suf);
      Buffer.add_string buf "        }\n"
    done
  done
;;

let load_transpose_8x8 ~buf ~groups =
  for which_side = 0 to 1 do
    let suf = if which_side = 0 then "re" else "im" in
    for g = 0 to groups - 1 do
      let j0 = g * 8 in
      Buffer.add_string
        buf
        (Printf.sprintf
           "        {  /* 8x8 transpose group: fft_idx %d..%d, %s */\n"
           j0
           (j0 + 7)
           suf);
      (* 8 row loads *)
      for r = 0 to 7 do
        Buffer.add_string
          buf
          (Printf.sprintf
             "            const __m512d _row_%s_%d = \
              _mm512_loadu_pd(&rio_%s[(b+%d)*row_stride + %d]);\n"
             suf
             r
             suf
             r
             j0)
      done;
      (* Stage 1: 8 unpacklo/unpackhi_pd. *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t0_%s = _mm512_unpacklo_pd(_row_%s_0, _row_%s_1);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t1_%s = _mm512_unpackhi_pd(_row_%s_0, _row_%s_1);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t2_%s = _mm512_unpacklo_pd(_row_%s_2, _row_%s_3);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t3_%s = _mm512_unpackhi_pd(_row_%s_2, _row_%s_3);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t4_%s = _mm512_unpacklo_pd(_row_%s_4, _row_%s_5);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t5_%s = _mm512_unpackhi_pd(_row_%s_4, _row_%s_5);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t6_%s = _mm512_unpacklo_pd(_row_%s_6, _row_%s_7);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _t7_%s = _mm512_unpackhi_pd(_row_%s_6, _row_%s_7);\n"
           suf
           suf
           suf);
      (* Stage 2: 8 permutex2var_pd. *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x0_%s = _mm512_permutex2var_pd(_t0_%s, \
            _tp_idx_lo, _t2_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x1_%s = _mm512_permutex2var_pd(_t1_%s, \
            _tp_idx_lo, _t3_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x2_%s = _mm512_permutex2var_pd(_t0_%s, \
            _tp_idx_hi, _t2_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x3_%s = _mm512_permutex2var_pd(_t1_%s, \
            _tp_idx_hi, _t3_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x4_%s = _mm512_permutex2var_pd(_t4_%s, \
            _tp_idx_lo, _t6_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x5_%s = _mm512_permutex2var_pd(_t5_%s, \
            _tp_idx_lo, _t7_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x6_%s = _mm512_permutex2var_pd(_t4_%s, \
            _tp_idx_hi, _t6_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _x7_%s = _mm512_permutex2var_pd(_t5_%s, \
            _tp_idx_hi, _t7_%s);\n"
           suf
           suf
           suf);
      (* Stage 3: 8 shuffle_f64x2, assign directly to lane_re_{j0..j0+7}. *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x0_%s, _x4_%s, 0x44);\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x1_%s, _x5_%s, 0x44);\n"
           suf
           (j0 + 1)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x2_%s, _x6_%s, 0x44);\n"
           suf
           (j0 + 2)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x3_%s, _x7_%s, 0x44);\n"
           suf
           (j0 + 3)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x0_%s, _x4_%s, 0xEE);\n"
           suf
           (j0 + 4)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x1_%s, _x5_%s, 0xEE);\n"
           suf
           (j0 + 5)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x2_%s, _x6_%s, 0xEE);\n"
           suf
           (j0 + 6)
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            lane_%s_%d = _mm512_shuffle_f64x2(_x3_%s, _x7_%s, 0xEE);\n"
           suf
           (j0 + 7)
           suf
           suf);
      Buffer.add_string buf "        }\n"
    done
  done
;;

let store_transpose_4x4 ~buf ~groups =
  for which_side = 0 to 1 do
    let suf = if which_side = 0 then "re" else "im" in
    for g = 0 to groups - 1 do
      let j0 = g * 4 in
      Buffer.add_string
        buf
        (Printf.sprintf
           "        {  /* inverse 4x4 transpose group: fft_idx %d..%d, %s */\n"
           j0
           (j0 + 3)
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _u0_%s = _mm256_unpacklo_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           j0
           suf
           (j0 + 1));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _u1_%s = _mm256_unpackhi_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           j0
           suf
           (j0 + 1));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _u2_%s = _mm256_unpacklo_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 2)
           suf
           (j0 + 3));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m256d _u3_%s = _mm256_unpackhi_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 2)
           suf
           (j0 + 3));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm256_storeu_pd(&rio_%s[(b+0)*row_stride + %d], \
            _mm256_permute2f128_pd(_u0_%s, _u2_%s, 0x20));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm256_storeu_pd(&rio_%s[(b+1)*row_stride + %d], \
            _mm256_permute2f128_pd(_u1_%s, _u3_%s, 0x20));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm256_storeu_pd(&rio_%s[(b+2)*row_stride + %d], \
            _mm256_permute2f128_pd(_u0_%s, _u2_%s, 0x31));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm256_storeu_pd(&rio_%s[(b+3)*row_stride + %d], \
            _mm256_permute2f128_pd(_u1_%s, _u3_%s, 0x31));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string buf "        }\n"
    done
  done
;;

let store_transpose_8x8 ~buf ~groups =
  for which_side = 0 to 1 do
    let suf = if which_side = 0 then "re" else "im" in
    for g = 0 to groups - 1 do
      let j0 = g * 8 in
      Buffer.add_string
        buf
        (Printf.sprintf
           "        {  /* inverse 8x8 transpose group: fft_idx %d..%d, %s */\n"
           j0
           (j0 + 7)
           suf);
      (* Stage 1: 8 unpacklo/unpackhi_pd on out_lane pairs. *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u0_%s = _mm512_unpacklo_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           j0
           suf
           (j0 + 1));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u1_%s = _mm512_unpackhi_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           j0
           suf
           (j0 + 1));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u2_%s = _mm512_unpacklo_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 2)
           suf
           (j0 + 3));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u3_%s = _mm512_unpackhi_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 2)
           suf
           (j0 + 3));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u4_%s = _mm512_unpacklo_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 4)
           suf
           (j0 + 5));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u5_%s = _mm512_unpackhi_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 4)
           suf
           (j0 + 5));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u6_%s = _mm512_unpacklo_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 6)
           suf
           (j0 + 7));
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _u7_%s = _mm512_unpackhi_pd(out_lane_%s_%d, \
            out_lane_%s_%d);\n"
           suf
           suf
           (j0 + 6)
           suf
           (j0 + 7));
      (* Stage 2: 8 permutex2var_pd. *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v0_%s = _mm512_permutex2var_pd(_u0_%s, \
            _tp_idx_lo, _u2_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v1_%s = _mm512_permutex2var_pd(_u1_%s, \
            _tp_idx_lo, _u3_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v2_%s = _mm512_permutex2var_pd(_u0_%s, \
            _tp_idx_hi, _u2_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v3_%s = _mm512_permutex2var_pd(_u1_%s, \
            _tp_idx_hi, _u3_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v4_%s = _mm512_permutex2var_pd(_u4_%s, \
            _tp_idx_lo, _u6_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v5_%s = _mm512_permutex2var_pd(_u5_%s, \
            _tp_idx_lo, _u7_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v6_%s = _mm512_permutex2var_pd(_u4_%s, \
            _tp_idx_hi, _u6_%s);\n"
           suf
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            const __m512d _v7_%s = _mm512_permutex2var_pd(_u5_%s, \
            _tp_idx_hi, _u7_%s);\n"
           suf
           suf
           suf);
      (* Stage 3 + store: 8 storeu_pd, each fused with a shuffle_f64x2. *)
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+0)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v0_%s, _v4_%s, 0x44));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+1)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v1_%s, _v5_%s, 0x44));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+2)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v2_%s, _v6_%s, 0x44));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+3)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v3_%s, _v7_%s, 0x44));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+4)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v0_%s, _v4_%s, 0xEE));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+5)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v1_%s, _v5_%s, 0xEE));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+6)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v2_%s, _v6_%s, 0xEE));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string
        buf
        (Printf.sprintf
           "            _mm512_storeu_pd(&rio_%s[(b+7)*row_stride + %d], \
            _mm512_shuffle_f64x2(_v3_%s, _v7_%s, 0xEE));\n"
           suf
           j0
           suf
           suf);
      Buffer.add_string buf "        }\n"
    done
  done
;;
