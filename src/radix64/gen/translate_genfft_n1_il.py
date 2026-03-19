#!/usr/bin/env python3
"""
translate_genfft_n1_il.py — Translate FFTW genfft n1fv/n1bv DAG to native interleaved.

Native IL: each variable is ONE vector of interleaved [re0,im0,re1,im1,...].
FFTW's genfft DAG maps almost 1:1 — only VFMAI/VFNMSI need special handling.

  VADD(a, b)     → add(a, b)                          [trivial]
  VSUB(a, b)     → sub(a, b)                          [trivial]
  VFMA(c, a, b)  → fmadd(a, broadcast(c), b)          [scalar c]
  VFNMS(c, a, b) → fnmadd(a, broadcast(c), b)         [scalar c]
  VFMS(c, a, b)  → fmsub(a, broadcast(c), b)          [scalar c]
  VMUL(c, a)     → mul(a, broadcast(c))               [scalar c]
  VFMAI(b, a)    → addsub(a, FLIP(b))                 [a + j*b]
  VFNMSI(b, a)   → addsub(a, neg(FLIP(b)))            [a - j*b]
  LD(idx)        → load(&in[2*(idx*K+k)])
  ST(idx, v)     → store(&out[2*(idx*K+k)], v)

AVX2:   k-step=2 (2 parallel DFT-25s), _mm256_addsub_pd
AVX-512: k-step=4 (4 parallel DFT-25s), _mm512_fmaddsub_pd for ×j

Usage: python3 translate_genfft_n1_il.py <fftw_src_dir> <radix> <avx2|avx512>
"""

import re, sys, math


def parse_fma_section(filepath):
    with open(filepath) as f:
        text = f.read()
    m = re.search(r'ARCH_PREFERS_FMA.*?\n(.*?)^#else', text, re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError(f"No FMA section in {filepath}")
    return m.group(1)


def extract_constants(section):
    consts = {}
    for m in re.finditer(r'DVK\((\w+),\s*([+\-]?\d+\.\d+)\)', section):
        consts[m.group(1)] = float(m.group(2))
    return consts


def flatten_nested(section):
    """Flatten ALL nested V-expressions into temp vars.
    Repeatedly finds the innermost V-call that's an argument to another V-call
    and extracts it into a temp variable until no nesting remains."""
    tmp_counter = [0]
    V_OPS = {'VADD','VSUB','VFMA','VFNMS','VFMS','VFMAI','VFNMSI','VMUL'}

    def make_tmp():
        tmp_counter[0] += 1
        return f'_ftmp{tmp_counter[0]}'

    def find_balanced(s, start):
        """From s[start] which should be '(', find matching ')'. Return index of ')'."""
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '(': depth += 1
            elif s[i] == ')':
                depth -= 1
                if depth == 0: return i
        return len(s) - 1

    def find_innermost_nested_vcall(s):
        """Find a V-call whose argument is another V-call.
        Returns (inner_start, inner_end) of the innermost such V-call, or None."""
        # Find all V-op positions
        positions = []
        for op in V_OPS:
            idx = 0
            while True:
                pos = s.find(op + '(', idx)
                if pos < 0: break
                # Make sure it's not part of a longer word (e.g. LDK)
                if pos > 0 and s[pos-1].isalpha():
                    idx = pos + 1
                    continue
                paren_start = pos + len(op)
                paren_end = find_balanced(s, paren_start)
                positions.append((pos, paren_end + 1, op))
                idx = pos + 1

        # Sort by start position
        positions.sort()

        # Find a V-call that is INSIDE another V-call's arguments
        # The innermost one (smallest span containing another V-call)
        for i, (s1, e1, op1) in enumerate(positions):
            for j, (s2, e2, op2) in enumerate(positions):
                if i == j: continue
                # Is (s1,e1) contained within (s2,e2)'s arguments?
                arg_start2 = s2 + len(op2) + 1  # after the '('
                arg_end2 = e2 - 1  # before the ')'
                if s1 >= arg_start2 and e1 <= e2:
                    # s1..e1 is inside s2..e2
                    # Check if s1..e1 itself contains no other V-call (it's innermost)
                    inner_text = s[s1:e1]
                    has_inner = False
                    for op3 in V_OPS:
                        p = inner_text.find(op3 + '(', len(op1) + 1)
                        if p >= 0 and (p == 0 or not inner_text[p-1].isalpha()):
                            has_inner = True
                            break
                    if not has_inner:
                        return (s1, e1)
        return None

    lines = section.split('\n')
    changed = True
    while changed:
        changed = False
        new_lines = []
        for line in lines:
            stripped = line.strip().rstrip(';')
            if not stripped or stripped.startswith('/*') or stripped.startswith('V ') or \
               stripped.startswith('for ') or 'MAKE_VOLATILE' in stripped or \
               'VLEAVE' in stripped or 'VENTER' in stripped or \
               stripped.startswith('{') or stripped.startswith('}') or \
               stripped.startswith('INT') or stripped.startswith('const R') or \
               stripped.startswith('R *') or 'DVK(' in stripped:
                new_lines.append(line)
                continue

            result = find_innermost_nested_vcall(stripped)
            if result:
                s1, e1 = result
                inner_call = stripped[s1:e1]
                tmp = make_tmp()
                indent = line[:len(line) - len(line.lstrip())]
                if not indent: indent = '\t\t    '
                new_stripped = stripped[:s1] + tmp + stripped[e1:]
                new_lines.append(f'{indent}{tmp} = {inner_call};')
                new_lines.append(f'{indent}{new_stripped};')
                changed = True
            else:
                new_lines.append(line)
        lines = new_lines

    return '\n'.join(lines)


def extract_ops(section):
    """Extract all DAG operations in order."""
    ops = []
    lines = section.split('\n')
    for line in lines:
        line = line.strip().rstrip(';')
        if not line or line.startswith('/*') or line.startswith('V ') or \
           line.startswith('for ') or line.startswith('{') or line.startswith('}') or \
           'MAKE_VOLATILE' in line or 'VLEAVE' in line or 'VENTER' in line:
            continue
        
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not m:
            continue
        var = m.group(1)
        expr = m.group(2).strip()
        if expr.startswith('ST('):
            continue
        
        op = parse_expr(expr)
        if op:
            ops.append((var, op))
    return ops


def parse_expr(expr):
    # LD
    m = re.match(r'LD\(&\(xi\[(?:WS\(is,\s*(\d+)\)|0)\]\)', expr)
    if m: return ('LD', int(m.group(1)) if m.group(1) else 0)
    
    # VADD
    m = re.match(r'VADD\((\w+),\s*(\w+)\)', expr)
    if m: return ('VADD', m.group(1), m.group(2))
    
    # VSUB
    m = re.match(r'VSUB\((\w+),\s*(\w+)\)', expr)
    if m: return ('VSUB', m.group(1), m.group(2))
    
    # VFMA(LDK(c), a, b)
    m = re.match(r'VFMA\(LDK\((\w+)\),\s*(\w+),\s*(\w+)\)', expr)
    if m: return ('VFMA', m.group(1), m.group(2), m.group(3))
    
    # VFNMS(LDK(c), a, b)
    m = re.match(r'VFNMS\(LDK\((\w+)\),\s*(\w+),\s*(\w+)\)', expr)
    if m: return ('VFNMS', m.group(1), m.group(2), m.group(3))
    
    # VFMS(LDK(c), a, b)
    m = re.match(r'VFMS\(LDK\((\w+)\),\s*(\w+),\s*(\w+)\)', expr)
    if m: return ('VFMS', m.group(1), m.group(2), m.group(3))
    
    # VMUL(LDK(c), a)
    m = re.match(r'VMUL\(LDK\((\w+)\),\s*(\w+)\)', expr)
    if m: return ('VMUL', m.group(1), m.group(2))
    
    # VFMAI(b, a)
    m = re.match(r'VFMAI\((\w+),\s*(\w+)\)', expr)
    if m: return ('VFMAI', m.group(1), m.group(2))
    
    # VFNMSI(b, a)
    m = re.match(r'VFNMSI\((\w+),\s*(\w+)\)', expr)
    if m: return ('VFNMSI', m.group(1), m.group(2))
    
    return None


def extract_stores(section):
    stores = []
    for m in re.finditer(
        r'ST\(&\(xo\[(?:WS\(os,\s*(\d+)\)|0)\]\),\s*'
        r'(VFMAI|VFNMSI|VADD|VSUB)\((\w+),\s*(\w+)\)', section):
        idx = int(m.group(1)) if m.group(1) else 0
        stores.append((idx, m.group(2), m.group(3), m.group(4)))
    return stores


def emit_native_il(isa, direction, fftw_src_dir, radix):
    fwd = direction == 'fwd'
    suffix = 'fv' if fwd else 'bv'
    filepath = f'{fftw_src_dir}/dft/simd/common/n1{suffix}_{radix}.c'
    
    section = parse_fma_section(filepath)
    section = flatten_nested(section)
    consts = extract_constants(section)
    dag_ops = extract_ops(section)
    stores = extract_stores(section)
    
    if isa == 'avx512':
        T = '__m512d'
        P = '_mm512'
        W = 4  # k-step: 4 parallel DFTs
        attr = '__attribute__((target("avx512f,avx512dq,fma")))'
        flip = lambda x: f'{P}_permute_pd({x}, 0x55)'
        # VFMAI(b, a) = a + j*b: fmaddsub(ones, a, FLIP(b)) = [a-flip, a+flip, ...]
        # Actually fmaddsub = [a*b-c, a*b+c, ...] at even,odd positions
        # fmaddsub(ones, a, FLIP(b)) = [1*a[0]-FLIP[0], 1*a[1]+FLIP[1], ...]
        #   = [a_re - b_im, a_im + b_re, ...] = VFMAI ✓
        vfmai = lambda a, b: f'{P}_fmaddsub_pd(ones, {a}, {flip(b)})'
        # VFNMSI(b, a) = a - j*b: fmsubadd(ones, a, FLIP(b))
        # fmsubadd = [a*b+c, a*b-c, ...] at even,odd
        # = [a_re + b_im, a_im - b_re, ...] = VFNMSI ✓
        vfnmsi = lambda a, b: f'{P}_fmsubadd_pd(ones, {a}, {flip(b)})'
    else:  # avx2
        T = '__m256d'
        P = '_mm256'
        W = 2  # k-step: 2 parallel DFTs
        attr = '__attribute__((target("avx2,fma")))'
        flip = lambda x: f'{P}_permute_pd({x}, 0x5)'
        # addsub(a, FLIP(b)) = [a[0]-FLIP[0], a[1]+FLIP[1], ...] = VFMAI
        vfmai = lambda a, b: f'{P}_addsub_pd({a}, {flip(b)})'
        # VFNMSI: addsub(a, neg(FLIP(b)))
        vfnmsi = lambda a, b: f'{P}_addsub_pd({a}, {neg_vec(flip(b))})'
    
    neg_vec = lambda x: f'{P}_sub_pd({P}_setzero_pd(), {x})'
    add = lambda a,b: f'{P}_add_pd({a},{b})'
    sub = lambda a,b: f'{P}_sub_pd({a},{b})'
    load = lambda addr: f'{P}_load_pd({addr})'
    store_f = lambda addr,v: f'{P}_store_pd({addr},{v});'
    
    lines = []
    ind = 1
    def o(s): lines.append('    '*ind + s)
    def c(s): o(f'/* {s} */')
    def b(): lines.append('')
    
    lines.append(attr)
    lines.append('static void')
    lines.append(f'radix{radix}_n1_dit_kernel_{direction}_il_{isa}(')
    lines.append(f'    const double * __restrict__ in,')
    lines.append(f'    double * __restrict__ out,')
    lines.append(f'    size_t K)')
    lines.append('{')
    
    c(f'Monolithic DFT-{radix} native IL — translated from FFTW genfft n1{suffix}_{radix}')
    c(f'Each variable = one {T} holding {W} interleaved complex values')
    c(f'{len(consts)} constants, {len(dag_ops)} DAG ops, {len(stores)} stores')
    b()
    
    # Constants
    for name, val in sorted(consts.items()):
        o(f'const {T} {name} = {P}_set1_pd({val:.18e});')
    if isa == 'avx512':
        o(f'const {T} ones = {P}_set1_pd(1.0);')
    b()
    
    # Declare temp variables (including flattened temps from nested expressions)
    all_vars = set()
    input_vars = set()
    for var, op in dag_ops:
        all_vars.add(var)
        if op[0] == 'LD':
            input_vars.add(var)
    # Also find _ftmp vars used in the DAG
    for var, op in dag_ops:
        for arg in op[1:]:
            if isinstance(arg, str) and arg.startswith('_ftmp'):
                all_vars.add(arg)
    temp_vars = sorted(all_vars - input_vars)
    
    chunk = 10
    for i in range(0, len(temp_vars), chunk):
        group = temp_vars[i:i+chunk]
        o(f'{T} {", ".join(group)};')
    b()
    
    o(f'for (size_t k = 0; k < K; k += {W}) {{')
    ind += 1
    
    for var, op_parsed in dag_ops:
        opname = op_parsed[0]
        
        if opname == 'LD':
            idx = op_parsed[1]
            o(f'{T} {var} = {load(f"&in[2*({idx}*K+k)]")};')
        
        elif opname == 'VADD':
            a, bv = op_parsed[1], op_parsed[2]
            o(f'{var} = {add(a, bv)};')
        
        elif opname == 'VSUB':
            a, bv = op_parsed[1], op_parsed[2]
            o(f'{var} = {sub(a, bv)};')
        
        elif opname == 'VFMA':
            const_name, a, bv = op_parsed[1], op_parsed[2], op_parsed[3]
            o(f'{var} = {P}_fmadd_pd({a}, {const_name}, {bv});')
        
        elif opname == 'VFNMS':
            const_name, a, bv = op_parsed[1], op_parsed[2], op_parsed[3]
            o(f'{var} = {P}_fnmadd_pd({a}, {const_name}, {bv});')
        
        elif opname == 'VFMS':
            const_name, a, bv = op_parsed[1], op_parsed[2], op_parsed[3]
            o(f'{var} = {P}_fmsub_pd({a}, {const_name}, {bv});')
        
        elif opname == 'VMUL':
            const_name, a = op_parsed[1], op_parsed[2]
            o(f'{var} = {P}_mul_pd({a}, {const_name});')
        
        elif opname == 'VFMAI':
            bv, a = op_parsed[1], op_parsed[2]
            o(f'{var} = {vfmai(a, bv)};')
        
        elif opname == 'VFNMSI':
            bv, a = op_parsed[1], op_parsed[2]
            o(f'{var} = {vfnmsi(a, bv)};')
    
    b()
    c('Stores')
    for idx, op, arg1, arg2 in stores:
        if op == 'VADD':
            o(store_f(f'&out[2*({idx}*K+k)]', add(arg1, arg2)))
        elif op == 'VFMAI':
            o(store_f(f'&out[2*({idx}*K+k)]', vfmai(arg2, arg1)))
        elif op == 'VFNMSI':
            o(store_f(f'&out[2*({idx}*K+k)]', vfnmsi(arg2, arg1)))
        elif op == 'VSUB':
            o(store_f(f'&out[2*({idx}*K+k)]', sub(arg1, arg2)))
    
    ind -= 1
    o('}')
    lines.append('}')
    lines.append('')
    return lines


def gen_file(isa, fftw_src_dir, radix):
    ISA = isa.upper()
    guard = f'FFT_RADIX{radix}_{ISA}_N1_MONO_IL_H'
    
    L = ['/**',
         f' * @file fft_radix{radix}_{isa}_n1_mono_il.h',
         f' * @brief Monolithic DFT-{radix} N1 native IL {ISA}',
         f' * Single flat DAG, zero passes. Each variable = one vector register.',
         f' * Translated from FFTW genfft n1fv/n1bv_{radix}.',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '#include <immintrin.h>', '']
    
    for d in ('fwd', 'bwd'):
        L.extend(emit_native_il(isa, d, fftw_src_dir, radix))
    
    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: translate_genfft_n1_il.py <fftw_src_dir> <radix> <avx2|avx512>", file=sys.stderr)
        sys.exit(1)
    fftw_dir = sys.argv[1]
    radix = int(sys.argv[2])
    isa = sys.argv[3]
    lines = gen_file(isa, fftw_dir, radix)
    print('\n'.join(lines))
