#!/usr/bin/env python3
"""il_derive.py — derive interleaved-boundary codelets from emitted split ones.

Per-line, scope-proof, width-dispatching transform:
  il_in       : every load  of ptr_re/ptr_im (512/256/128/scalar) -> its half
                extracted from pair loads of z (duplicates are L1-hot / CSE'd)
  il_out      : every store to ptr_re/ptr_im -> expand-permute + masked store
                into even/odd slots of z (k-masks on 512; hoisted masks on AVX2;
                storel/storeh on 128; index arithmetic on scalar)
  il_out_flip : same with (im,re) pair order — bwd-via-swap on fwd math
Arithmetic untouched; signature+name rewritten; provenance prepended.
"""
import re, sys, os

def derive(src_path, dst_path, mode, isa, ptr_re, ptr_im, ptr_z,
           fn_old, fn_new, sig_old, sig_new):
    s = open(src_path).read()
    lines = s.split("\n")
    out = []
    esc_re, esc_im = re.escape(ptr_re), re.escape(ptr_im)

    if mode == "il_in":
        WID = [("_mm512_loadu_pd", "__m512d", 8),
               ("_mm256_loadu_pd", "__m256d", 4),
               ("_mm_loadu_pd",    "__m128d", 2)]
        pats = []
        for ld, vt, w in WID:
            for side, esc in (("re", esc_re), ("im", esc_im)):
                pats.append((re.compile(
                    r"^(\s*)((?:const\s+__m\d+d\s+)?\S+)\s*=\s*" + re.escape(ld) +
                    r"\(&" + esc + r"\[(.+)\]\);\s*$"), side, ld, vt, w))
        cnt = 0
        for ln in lines:
            hit = None
            for pat, side, ld, vt, w in pats:
                m = pat.match(ln)
                if m: hit = (m, side, ld, vt, w); break
            if not hit:
                mk = re.match(r"^(\s*)(\S+)\s*=\s*_mm512_maskz_loadu_pd\((\w+),\s*&(" +
                              esc_re + "|" + esc_im + r")\[(.+)\]\);\s*$", ln)
                if mk:
                    ind, lhs, km, ptr, expr = mk.groups(); expr = expr.strip()
                    side = "re" if ptr == ptr_re else "im"
                    sel = "_vfft_il_idx_e" if side == "re" else "_vfft_il_idx_o"
                    cnt += 1; t = f"_ilt{cnt}"
                    out.append(
                        f"{ind}{{ unsigned {t}d = _pdep_u32((unsigned){km}, 0x5555u) * 3u;\n"
                        f"{ind}  __m512d {t}a = _mm512_maskz_loadu_pd((__mmask8){t}d, &{ptr_z}[2*({expr})]);\n"
                        f"{ind}  __m512d {t}b = _mm512_maskz_loadu_pd((__mmask8)({t}d >> 8), &{ptr_z}[2*({expr}) + 8]);\n"
                        f"{ind}  {lhs} = _mm512_permutex2var_pd({t}a, {sel}, {t}b); }}")
                    continue
                ms = re.match(r"^(\s*)(\S+)\s*=\s*" + esc_re + r"\[(.+)\];\s*$", ln)
                if ms:
                    out.append(f"{ms.group(1)}{ms.group(2)} = {ptr_z}[2*({ms.group(3).strip()})];")
                    cnt += 1; continue
                ms = re.match(r"^(\s*)(\S+)\s*=\s*" + esc_im + r"\[(.+)\];\s*$", ln)
                if ms:
                    out.append(f"{ms.group(1)}{ms.group(2)} = {ptr_z}[2*({ms.group(3).strip()}) + 1];")
                    cnt += 1; continue
                out.append(ln); continue
            m, side, ld, vt, w = hit
            ind, lhs, expr = m.group(1), m.group(2), m.group(3).strip()
            cnt += 1; t = f"_ilt{cnt}"
            var, decl = lhs, ""
            if "__m" in lhs:
                var = lhs.split()[-1]
                base = lhs.replace("const ", "", 1)
                decl = f"{ind}{base.rsplit(None, 1)[0]} {var};\n"
            if w == 8:
                sel = "_vfft_il_idx_e" if side == "re" else "_vfft_il_idx_o"
                body = (f"{ind}{{ {vt} {t}a = {ld}(&{ptr_z}[2*({expr})]);\n"
                        f"{ind}  {vt} {t}b = {ld}(&{ptr_z}[2*({expr}) + 8]);\n"
                        f"{ind}  {var} = _mm512_permutex2var_pd({t}a, {sel}, {t}b); }}")
            elif w == 4:
                op = "_mm256_unpacklo_pd" if side == "re" else "_mm256_unpackhi_pd"
                body = (f"{ind}{{ {vt} {t}a = {ld}(&{ptr_z}[2*({expr})]);\n"
                        f"{ind}  {vt} {t}b = {ld}(&{ptr_z}[2*({expr}) + 4]);\n"
                        f"{ind}  {var} = _mm256_permute4x64_pd({op}({t}a, {t}b), 0xD8); }}")
            else:
                op = "_mm_unpacklo_pd" if side == "re" else "_mm_unpackhi_pd"
                body = (f"{ind}{{ {vt} {t}a = {ld}(&{ptr_z}[2*({expr})]);\n"
                        f"{ind}  {vt} {t}b = {ld}(&{ptr_z}[2*({expr}) + 2]);\n"
                        f"{ind}  {var} = {op}({t}a, {t}b); }}")
            out.append(decl + body)
        assert cnt > 0, "no il_in lines found"

    elif mode in ("il_out", "il_out_flip"):
        WID = [("_mm512_storeu_pd", 8), ("_mm256_storeu_pd", 4), ("_mm_storeu_pd", 2)]
        pats = []
        for st, w in WID:
            for side, esc in (("re", esc_re), ("im", esc_im)):
                pats.append((re.compile(
                    r"^(\s*)" + re.escape(st) + r"\(&" + esc +
                    r"\[(.+?)\],\s*(.+)\);\s*$"), side, st, w))
        cnt = 0
        # -- pair-fusion pre-pass (6a9): the list scheduler sinks stores and
        #    emits (re[E], im[E]) adjacent pairs with identical index
        #    expressions. Fusing a pair yields ONE full store per z vector
        #    (2 permutex2var / unpack+perm2f128) instead of two half-writes
        #    with complementary masks -- halves store uops and removes the
        #    per-line double-write RFO pathology. Unpaired lines fall through
        #    to the per-line lattice below.
        def _pair_match(la, lb):
            for pat, side, st, w in pats:
                ma = pat.match(la)
                if ma and side == "re":
                    for pat2, side2, st2, w2 in pats:
                        if st2 != st: continue
                        mb = pat2.match(lb)
                        if mb and side2 == "im" and mb.group(2).strip() == ma.group(2).strip():
                            return ("plain", w, ma.group(1), ma.group(2).strip(),
                                    ma.group(3), mb.group(3), None)
            mre = re.compile(r"^(\s*)_mm512_mask_storeu_pd\(&" + esc_re +
                             r"\[(.+?)\],\s*(\w+),\s*(.+)\);\s*$")
            mim = re.compile(r"^(\s*)_mm512_mask_storeu_pd\(&" + esc_im +
                             r"\[(.+?)\],\s*(\w+),\s*(.+)\);\s*$")
            ma = mre.match(la)
            if ma:
                mb = mim.match(lb)
                if mb and mb.group(2).strip() == ma.group(2).strip() \
                      and mb.group(3) == ma.group(3):
                    return ("mask", 8, ma.group(1), ma.group(2).strip(),
                            ma.group(4), mb.group(4), ma.group(3))
            return None
        fused = []
        li = 0
        while li < len(lines):
            pm = _pair_match(lines[li], lines[li + 1]) if li + 1 < len(lines) else None
            if pm is None:
                fused.append(lines[li]); li += 1; continue
            kind, w, ind, expr, v_re, v_im, km = pm
            va, vb = (v_re, v_im) if mode == "il_out" else (v_im, v_re)
            cnt += 1; t = f"_ofp{cnt}"
            if kind == "mask":
                fused.append(
                    f"{ind}{{ unsigned {t}d = _pdep_u32((unsigned){km}, 0x5555u) * 3u;\n"
                    f"{ind}  const __m512d {t}a = {va};\n"
                    f"{ind}  const __m512d {t}b = {vb};\n"
                    f"{ind}  _mm512_mask_storeu_pd(&{ptr_z}[2*({expr})], (__mmask8){t}d,"
                    f" _mm512_permutex2var_pd({t}a, _vfft_il_pair_lo, {t}b));\n"
                    f"{ind}  _mm512_mask_storeu_pd(&{ptr_z}[2*({expr}) + 8], (__mmask8)({t}d >> 8),"
                    f" _mm512_permutex2var_pd({t}a, _vfft_il_pair_hi, {t}b)); }}")
            elif w == 8:
                fused.append(
                    f"{ind}{{ const __m512d {t}a = {va};\n"
                    f"{ind}  const __m512d {t}b = {vb};\n"
                    f"{ind}  _mm512_storeu_pd(&{ptr_z}[2*({expr})],"
                    f" _mm512_permutex2var_pd({t}a, _vfft_il_pair_lo, {t}b));\n"
                    f"{ind}  _mm512_storeu_pd(&{ptr_z}[2*({expr}) + 8],"
                    f" _mm512_permutex2var_pd({t}a, _vfft_il_pair_hi, {t}b)); }}")
            elif w == 4:
                fused.append(
                    f"{ind}{{ const __m256d {t}a = {va};\n"
                    f"{ind}  const __m256d {t}b = {vb};\n"
                    f"{ind}  const __m256d {t}p = _mm256_unpacklo_pd({t}a, {t}b);\n"
                    f"{ind}  const __m256d {t}q = _mm256_unpackhi_pd({t}a, {t}b);\n"
                    f"{ind}  _mm256_storeu_pd(&{ptr_z}[2*({expr})],"
                    f" _mm256_permute2f128_pd({t}p, {t}q, 0x20));\n"
                    f"{ind}  _mm256_storeu_pd(&{ptr_z}[2*({expr}) + 4],"
                    f" _mm256_permute2f128_pd({t}p, {t}q, 0x31)); }}")
            else:
                fused.append(
                    f"{ind}{{ const __m128d {t}a = {va};\n"
                    f"{ind}  const __m128d {t}b = {vb};\n"
                    f"{ind}  _mm_storeu_pd(&{ptr_z}[2*({expr})], _mm_unpacklo_pd({t}a, {t}b));\n"
                    f"{ind}  _mm_storeu_pd(&{ptr_z}[2*({expr}) + 2], _mm_unpackhi_pd({t}a, {t}b)); }}")
            li += 2
        lines = fused
        for ln in lines:
            hit = None
            for pat, side, st, w in pats:
                m = pat.match(ln)
                if m: hit = (m, side, st, w); break
            if not hit:
                mk = re.match(r"^(\s*)_mm512_mask_storeu_pd\(&(" + esc_re + "|" + esc_im +
                              r")\[(.+?)\],\s*(\w+),\s*(.+)\);\s*$", ln)
                if mk:
                    ind, ptr, expr, km, val = mk.groups(); expr = expr.strip()
                    side = "re" if ptr == ptr_re else "im"
                    if mode == "il_out_flip":
                        side = "im" if side == "re" else "re"
                    slot = "0x55" if side == "re" else "0xAA"
                    cnt += 1; t = f"_ost{cnt}"
                    out.append(
                        f"{ind}{{ unsigned {t}d = _pdep_u32((unsigned){km}, 0x5555u) * 3u;\n"
                        f"{ind}  __m512d {t}v = {val};\n"
                        f"{ind}  _mm512_mask_storeu_pd(&{ptr_z}[2*({expr})], (__mmask8)({slot} & {t}d),"
                        f" _mm512_permutexvar_pd(_vfft_il_exp_lo, {t}v));\n"
                        f"{ind}  _mm512_mask_storeu_pd(&{ptr_z}[2*({expr}) + 8], (__mmask8)({slot} & ({t}d >> 8)),"
                        f" _mm512_permutexvar_pd(_vfft_il_exp_hi, {t}v)); }}")
                    continue
                ms = re.match(r"^(\s*)" + esc_re + r"\[(.+?)\]\s*=\s*(.+);\s*$", ln)
                sside = "re" if ms else None
                if not ms:
                    ms = re.match(r"^(\s*)" + esc_im + r"\[(.+?)\]\s*=\s*(.+);\s*$", ln)
                    if ms: sside = "im"
                if ms:
                    if mode == "il_out_flip":
                        sside = "im" if sside == "re" else "re"
                    off = "" if sside == "re" else " + 1"
                    out.append(f"{ms.group(1)}{ptr_z}[2*({ms.group(2).strip()}){off}] = {ms.group(3)};")
                    cnt += 1; continue
                out.append(ln); continue
            m, side, st, w = hit
            ind, expr, val = m.group(1), m.group(2).strip(), m.group(3)
            if mode == "il_out_flip":
                side = "im" if side == "re" else "re"
            cnt += 1; t = f"_ost{cnt}"
            if w == 8:
                mask = "0x55" if side == "re" else "0xAA"
                body = (f"{ind}{{ __m512d {t}v = {val};\n"
                        f"{ind}  _mm512_mask_storeu_pd(&{ptr_z}[2*({expr})], {mask},"
                        f" _mm512_permutexvar_pd(_vfft_il_exp_lo, {t}v));\n"
                        f"{ind}  _mm512_mask_storeu_pd(&{ptr_z}[2*({expr}) + 8], {mask},"
                        f" _mm512_permutexvar_pd(_vfft_il_exp_hi, {t}v)); }}")
            elif w == 4:
                mask = "_vfft_il_m_even" if side == "re" else "_vfft_il_m_odd"
                body = (f"{ind}{{ __m256d {t}v = {val};\n"
                        f"{ind}  _mm256_maskstore_pd(&{ptr_z}[2*({expr})], {mask},"
                        f" _mm256_permute4x64_pd({t}v, 0x50));\n"
                        f"{ind}  _mm256_maskstore_pd(&{ptr_z}[2*({expr}) + 4], {mask},"
                        f" _mm256_permute4x64_pd({t}v, 0xFA)); }}")
            else:
                off = "" if side == "re" else " + 1"
                body = (f"{ind}{{ __m128d {t}v = {val};\n"
                        f"{ind}  _mm_storel_pd(&{ptr_z}[2*({expr}){off}], {t}v);\n"
                        f"{ind}  _mm_storeh_pd(&{ptr_z}[2*({expr}) + 2{off}], {t}v); }}")
            out.append(body)
        assert cnt > 0, "no il_out lines found"
    else:
        raise SystemExit("bad mode")

    s = "\n".join(out)
    assert sig_old in s, "signature anchor missing"
    s = s.replace(sig_old, sig_new, 1)
    assert fn_old in s
    s = s.replace(fn_old, fn_new)
    # hoist constants after the opening brace
    i = s.find(fn_new); i = s.find("{", i)
    idx = ""
    if isa == "avx512":
        idx += ("\n    const __m512i _vfft_il_idx_e  = _mm512_setr_epi64(0,2,4,6,8,10,12,14);"
                "\n    const __m512i _vfft_il_idx_o  = _mm512_setr_epi64(1,3,5,7,9,11,13,15);"
                "\n    const __m512i _vfft_il_exp_lo = _mm512_setr_epi64(0,0,1,1,2,2,3,3);"
                "\n    const __m512i _vfft_il_pair_lo = _mm512_setr_epi64(0,8,1,9,2,10,3,11);"
                "\n    const __m512i _vfft_il_pair_hi = _mm512_setr_epi64(4,12,5,13,6,14,7,15);"
                "\n    (void)_vfft_il_pair_lo; (void)_vfft_il_pair_hi;"
                "\n    const __m512i _vfft_il_exp_hi = _mm512_setr_epi64(4,4,5,5,6,6,7,7);")
    idx += ("\n    const __m256i _vfft_il_m_even = _mm256_set_epi64x(0,-1,0,-1);"
            "\n    const __m256i _vfft_il_m_odd  = _mm256_set_epi64x(-1,0,-1,0);\n"
            "    (void)_vfft_il_m_even; (void)_vfft_il_m_odd;\n")
    s = s[:i+1] + idx + s[i+1:]
    prov = (f"/* DERIVED CODELET — mechanical IL transform (il_derive.py)\n"
            f" * source: {os.path.basename(src_path)}  mode: {mode}  isa: {isa}\n"
            f" * Every {ptr_re}/{ptr_im} access replaced per-line by the interleave\n"
            f" * lattice against {ptr_z}; arithmetic untouched; widths 512/256/128/scalar\n"
            f" * all handled. {'PAIR ORDER (im,re) — bwd-via-swap contract.' if mode=='il_out_flip' else ''}\n"
            f" * Regenerate by re-running the tool. */\n")
    open(dst_path, "w").write(prov + s)
    print(f"  {os.path.basename(dst_path)}  ok ({cnt} sites)")

if __name__ == "__main__":
    import json
    for job in json.load(open(sys.argv[1])):
        derive(**job)
