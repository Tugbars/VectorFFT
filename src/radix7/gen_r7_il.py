#!/usr/bin/env python3
"""
gen_r7_il.py — Post-process R=7 split codelets into IL variants.

Replaces:
  _mm256_load_pd(&in_re[N*K+k])  →  IL deinterleave
  _mm256_load_pd(&in_im[N*K+k])  →  (paired with above)
  _mm256_store_pd(&out_re[N*K+k], expr)  →  IL reinterleave
  _mm256_store_pd(&out_im[N*K+k], expr)  →  (paired with above)

Function signatures:  (in_re, in_im, out_re, out_im, ...) → (in, out, ...)
"""
import re, sys

def make_il_deinterleave_avx2(var_re, var_im, idx_expr, has_decl=False):
    """Generate AVX2 IL deinterleave for a load pair."""
    if has_decl:
        # Variables declared here — no scope wrapper, they stay visible
        return [
            f'        __m256d _dil_lo = _mm256_load_pd(&in[2*({idx_expr})]);',
            f'        __m256d _dil_hi = _mm256_load_pd(&in[2*({idx_expr}+2)]);',
            f'        __m256d {var_re} = _mm256_permute4x64_pd(_mm256_shuffle_pd(_dil_lo,_dil_hi,0x0), 0xD8);',
            f'        __m256d {var_im} = _mm256_permute4x64_pd(_mm256_shuffle_pd(_dil_lo,_dil_hi,0xF), 0xD8);',
        ]
    else:
        return [
            f'        {{ __m256d _lo = _mm256_load_pd(&in[2*({idx_expr})]);',
            f'          __m256d _hi = _mm256_load_pd(&in[2*({idx_expr}+2)]);',
            f'          {var_re} = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0x0), 0xD8);',
            f'          {var_im} = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0xF), 0xD8); }}',
        ]

def make_il_reinterleave_avx2(expr_re, expr_im, idx_expr):
    """Generate AVX2 IL reinterleave for a store pair."""
    return [
        f'        {{ __m256d _rp = _mm256_permute4x64_pd({expr_re}, 0xD8);',
        f'          __m256d _ip = _mm256_permute4x64_pd({expr_im}, 0xD8);',
        f'          _mm256_store_pd(&out[2*({idx_expr})], _mm256_shuffle_pd(_rp,_ip,0x0));',
        f'          _mm256_store_pd(&out[2*({idx_expr}+2)], _mm256_shuffle_pd(_rp,_ip,0xF)); }}',
    ]

def make_il_deinterleave_avx512(var_re, var_im, idx_expr, has_decl=False):
    if has_decl:
        return [
            f'        __m512d _dil_lo = _mm512_load_pd(&in[2*({idx_expr})]);',
            f'        __m512d _dil_hi = _mm512_load_pd(&in[2*({idx_expr}+4)]);',
            f'        __m512d {var_re} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpacklo_pd(_dil_lo,_dil_hi));',
            f'        __m512d {var_im} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpackhi_pd(_dil_lo,_dil_hi));',
        ]
    else:
        return [
            f'        {{ __m512d _lo = _mm512_load_pd(&in[2*({idx_expr})]);',
            f'          __m512d _hi = _mm512_load_pd(&in[2*({idx_expr}+4)]);',
            f'          {var_re} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpacklo_pd(_lo,_hi));',
            f'          {var_im} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpackhi_pd(_lo,_hi)); }}',
        ]

def make_il_reinterleave_avx512(expr_re, expr_im, idx_expr):
    return [
        f'        {{ __m512d _rp = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {expr_re});',
        f'          __m512d _ip = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {expr_im});',
        f'          _mm512_store_pd(&out[2*({idx_expr})], _mm512_unpacklo_pd(_rp,_ip));',
        f'          _mm512_store_pd(&out[2*({idx_expr}+4)], _mm512_unpackhi_pd(_rp,_ip)); }}',
    ]

def make_il_deinterleave_scalar(var_re, var_im, idx_expr, has_decl=False):
    decl = 'double ' if has_decl else ''
    return [f'        {decl}{var_re} = in[2*({idx_expr})]; {decl}{var_im} = in[2*({idx_expr})+1];']

def make_il_reinterleave_scalar(expr_re, expr_im, idx_expr):
    return [f'        out[2*({idx_expr})] = {expr_re}; out[2*({idx_expr})+1] = {expr_im};']

def process_file(input_path, isa):
    with open(input_path) as f:
        content = f.read()

    if isa == 'avx2':
        P = '_mm256'
        deinterleave = make_il_deinterleave_avx2
        reinterleave = make_il_reinterleave_avx2
    elif isa == 'avx512':
        P = '_mm512'
        deinterleave = make_il_deinterleave_avx512
        reinterleave = make_il_reinterleave_avx512
    else:
        P = ''
        deinterleave = make_il_deinterleave_scalar
        reinterleave = make_il_reinterleave_scalar

    # Fix function signatures
    content = content.replace('in_re, const double * __restrict__ in_im,', 'in,')
    content = content.replace('out_re, double * __restrict__ out_im,', 'out,')
    # Fix remaining in_re/in_im/out_re/out_im in comments etc
    
    # Rename functions: add _il before _{isa}
    content = re.sub(r'(radix7_\w+?)_(' + isa + r')', r'\1_il_\2', content)
    
    # Fix guard
    ISA = isa.upper()
    content = content.replace(f'FFT_RADIX7_{ISA}_H', f'FFT_RADIX7_{ISA}_IL_H')
    content = content.replace(f'FFT_RADIX7_{ISA}_DIF_TW_H', f'FFT_RADIX7_{ISA}_IL_DIF_TW_H')
    
    # Update file description
    content = content.replace(f'DFT-7 {ISA}', f'DFT-7 {ISA} interleaved')

    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if isa == 'scalar':
            # Scalar: *(&in_re[expr]) pattern — match any variable name
            m_re = re.match(r'(\s+)((?:(?:const\s+)?double\s+)?)(\w+)\s*=\s*\*\(&in_re\[(.+?)\]\);', line)
            m_im = None
            if m_re and i+1 < len(lines):
                m_im = re.match(r'(\s+)((?:(?:const\s+)?double\s+)?)(\w+)\s*=\s*\*\(&in_im\[(.+?)\]\);', lines[i+1])
            if m_re and m_im and m_re.group(4) == m_im.group(4):
                idx = m_re.group(4)
                has_decl = len(m_re.group(2).strip()) > 0
                result.extend(deinterleave(m_re.group(3), m_im.group(3), idx, has_decl=has_decl))
                i += 2
                continue

            # Scalar store: *(&out_re[expr]) = val;
            m_sre = re.match(r'(\s+)\*\(&out_re\[(.+?)\]\)\s*=\s*(.+?);', line)
            m_sim = None
            if m_sre and i+1 < len(lines):
                m_sim = re.match(r'(\s+)\*\(&out_im\[(.+?)\]\)\s*=\s*(.+?);', lines[i+1])
            if m_sre and m_sim and m_sre.group(2) == m_sim.group(2):
                idx = m_sre.group(2)
                result.extend(reinterleave(m_sre.group(3), m_sim.group(3), idx))
                i += 2
                continue
        else:
            # SIMD: load_pd(&in_re[expr]) paired with load_pd(&in_im[expr])
            # Match any variable assigned from in_re/in_im arrays
            load_re_pat = rf'(\s+)((?:(?:const\s+)?(?:__m\d+d)\s+)?)(\w+)\s*=\s*{P}_load_pd\(&in_re\[(.+?)\]\);'
            load_im_pat = rf'(\s+)((?:(?:const\s+)?(?:__m\d+d)\s+)?)(\w+)\s*=\s*{P}_load_pd\(&in_im\[(.+?)\]\);'
            
            m_re = re.match(load_re_pat, line)
            m_im = None
            if m_re and i+1 < len(lines):
                m_im = re.match(load_im_pat, lines[i+1])
            
            if m_re and m_im and m_re.group(4) == m_im.group(4):
                idx = m_re.group(4)
                has_decl = len(m_re.group(2).strip()) > 0
                result.extend(deinterleave(m_re.group(3), m_im.group(3), idx, has_decl=has_decl))
                i += 2
                continue

            # SIMD store paired: store_pd(&out_re[expr], val); store_pd(&out_im[expr], val);
            # Can be on same line or adjacent lines
            store_re_pat = rf'{P}_store_pd\(&out_re\[(.+?)\],(.+?)\);'
            store_im_pat = rf'{P}_store_pd\(&out_im\[(.+?)\],(.+?)\);'
            
            m_sre = re.search(store_re_pat, line)
            m_sim = re.search(store_im_pat, line)
            
            if m_sre and m_sim and m_sre.group(1) == m_sim.group(1):
                # Both on same line
                idx = m_sre.group(1)
                result.extend(reinterleave(m_sre.group(2), m_sim.group(2), idx))
                i += 1
                continue
            
            if m_sre and not m_sim and i+1 < len(lines):
                m_sim = re.search(store_im_pat, lines[i+1])
                if m_sim and m_sre.group(1) == m_sim.group(1):
                    idx = m_sre.group(1)
                    result.extend(reinterleave(m_sre.group(2), m_sim.group(2), idx))
                    i += 2
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r7_il.py <input.h> <scalar|avx2|avx512>", file=sys.stderr)
        sys.exit(1)
    print(process_file(sys.argv[1], sys.argv[2]))
