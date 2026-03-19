#!/usr/bin/env python3
"""
gen_r32_il.py — Post-process R=32 split codelets into IL variants.
Only transforms in_re/in_im/out_re/out_im — twiddle and spill stay split.
Usage: python3 gen_r32_il.py <input.h> <avx2|avx512>
"""
import re, sys

def il_deinterleave(isa, var_re, var_im, idx_expr, has_decl=False):
    if isa == 'avx2':
        decl = '__m256d ' if has_decl else ''
        return [
            f'        {{ __m256d _lo = _mm256_load_pd(&in[2*({idx_expr})]);',
            f'          __m256d _hi = _mm256_load_pd(&in[2*({idx_expr}+2)]);',
            f'          {decl}{var_re} = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0x0), 0xD8);',
            f'          {decl}{var_im} = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0xF), 0xD8); }}',
        ]
    else:
        decl = '__m512d ' if has_decl else ''
        return [
            f'        {{ __m512d _lo = _mm512_load_pd(&in[2*({idx_expr})]);',
            f'          __m512d _hi = _mm512_load_pd(&in[2*({idx_expr}+4)]);',
            f'          {decl}{var_re} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpacklo_pd(_lo,_hi));',
            f'          {decl}{var_im} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpackhi_pd(_lo,_hi)); }}',
        ]

def il_reinterleave(isa, expr_re, expr_im, idx_expr):
    if isa == 'avx2':
        return [
            f'        {{ __m256d _rp = _mm256_permute4x64_pd({expr_re}, 0xD8);',
            f'          __m256d _ip = _mm256_permute4x64_pd({expr_im}, 0xD8);',
            f'          _mm256_store_pd(&out[2*({idx_expr})], _mm256_shuffle_pd(_rp,_ip,0x0));',
            f'          _mm256_store_pd(&out[2*({idx_expr}+2)], _mm256_shuffle_pd(_rp,_ip,0xF)); }}',
        ]
    else:
        return [
            f'        {{ __m512d _rp = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {expr_re});',
            f'          __m512d _ip = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {expr_im});',
            f'          _mm512_store_pd(&out[2*({idx_expr})], _mm512_unpacklo_pd(_rp,_ip));',
            f'          _mm512_store_pd(&out[2*({idx_expr}+4)], _mm512_unpackhi_pd(_rp,_ip)); }}',
        ]

def process_file(input_text, isa):
    content = input_text
    content = content.replace(
        'const double * __restrict__ in_re, const double * __restrict__ in_im,',
        'const double * __restrict__ in,')
    content = content.replace(
        'double * __restrict__ out_re, double * __restrict__ out_im,',
        'double * __restrict__ out,')
    content = content.replace(
        'const double * RESTRICT in_re, const double * RESTRICT in_im,',
        'const double * RESTRICT in,')
    content = content.replace(
        'double * RESTRICT out_re, double * RESTRICT out_im,',
        'double * RESTRICT out,')
    content = re.sub(r'(radix32_tw_\w+?_kernel)_(fwd|bwd)', r'\1_il_\2', content)
    ISA = isa.upper()
    for suffix in ['_TW_LADDER_H', '_TW_H', '_DIF_TW_H', '_TW_V2_H']:
        old_g = f'FFT_RADIX32_{ISA}{suffix}'
        content = content.replace(old_g, old_g.replace('_H', '_IL_H'))
    content = content.replace('DFT-32 ', 'DFT-32 interleaved ')

    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        matched = False

        # LD macro loads
        m_re = re.match(r'(\s+)(\w+) = LD\(&in_re\[(.+?)\]\);', line)
        if m_re and i+1 < len(lines):
            m_im = re.match(r'(\s+)(\w+) = LD\(&in_im\[(.+?)\]\);', lines[i+1])
            if m_im and m_re.group(3) == m_im.group(3):
                result.extend(il_deinterleave(isa, m_re.group(2), m_im.group(2), m_re.group(3)))
                i += 2; matched = True

        # Raw intrinsic loads (avx512)
        if not matched and isa == 'avx512':
            m_re = re.match(r'(\s+)((?:(?:const\s+)?__m512d\s+)?)(\w+) = _mm512_load(?:u)?_pd\(&in_re\[(.+?)\]\);', line)
            if m_re and i+1 < len(lines):
                m_im = re.match(r'(\s+)((?:(?:const\s+)?__m512d\s+)?)(\w+) = _mm512_load(?:u)?_pd\(&in_im\[(.+?)\]\);', lines[i+1])
                if m_im and m_re.group(4) == m_im.group(4):
                    has_decl = len(m_re.group(2).strip()) > 0
                    result.extend(il_deinterleave(isa, m_re.group(3), m_im.group(3), m_re.group(4), has_decl))
                    i += 2; matched = True

        # ST macro stores (same line)
        if not matched:
            m_sre = re.search(r'ST\(&out_re\[(.+?)\],(.+?)\);', line)
            m_sim = re.search(r'ST\(&out_im\[(.+?)\],(.+?)\);', line)
            if m_sre and m_sim and m_sre.group(1) == m_sim.group(1):
                result.extend(il_reinterleave(isa, m_sre.group(2), m_sim.group(2), m_sre.group(1)))
                i += 1; matched = True
            elif m_sre and not m_sim and i+1 < len(lines):
                m_sim = re.search(r'ST\(&out_im\[(.+?)\],(.+?)\);', lines[i+1])
                if m_sim and m_sre.group(1) == m_sim.group(1):
                    result.extend(il_reinterleave(isa, m_sre.group(2), m_sim.group(2), m_sre.group(1)))
                    i += 2; matched = True

        # Raw intrinsic stores (avx512)
        if not matched and isa == 'avx512':
            m_sre = re.search(r'_mm512_store(?:u)?_pd\(&out_re\[(.+?)\]\s*,\s*(.+?)\);', line)
            m_sim = re.search(r'_mm512_store(?:u)?_pd\(&out_im\[(.+?)\]\s*,\s*(.+?)\);', line)
            if m_sre and m_sim and m_sre.group(1) == m_sim.group(1):
                result.extend(il_reinterleave(isa, m_sre.group(2), m_sim.group(2), m_sre.group(1)))
                i += 1; matched = True
            elif m_sre and not m_sim and i+1 < len(lines):
                m_sim = re.search(r'_mm512_store(?:u)?_pd\(&out_im\[(.+?)\]\s*,\s*(.+?)\);', lines[i+1])
                if m_sim and m_sre.group(1) == m_sim.group(1):
                    result.extend(il_reinterleave(isa, m_sre.group(2), m_sim.group(2), m_sre.group(1)))
                    i += 2; matched = True

        if not matched:
            result.append(line)
            i += 1

    return '\n'.join(result)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r32_il.py <input.h> <avx2|avx512>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        text = f.read()
    print(process_file(text, sys.argv[2]))
