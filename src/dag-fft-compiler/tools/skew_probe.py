#!/usr/bin/env python3
# Sum-skew probe (text-space, dup_probe-style scaffold).
# Parses FMA product-chains  F(a,b, F(a,b, ... mul(a,b)))  and DC
# add-nests, rebuilds them under a reassociation strategy, compiles,
# counts. Strategy 'id' must reproduce the file byte-identically
# (parser validation). Rounding changes => truth-gate class.
import re, subprocess, sys, os

GCC=['gcc','-O3','-mavx2','-mfma','-march=raptorlake','-w','-S']
SPILL=re.compile(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)')

def parse_call(s,i):
    m=re.match(r'_mm256_(fmadd|fnmadd|mul|add)_pd\(',s[i:])
    if not m:
        m2=re.match(r'[A-Za-z_]\w*',s[i:])
        return ('id',s[i:i+m2.end()]),i+m2.end()
    kind=m.group(1); i+=m.end(); args=[]
    while True:
        a,i=parse_call(s,i); args.append(a)
        if s[i]==',': i+=1
        while s[i]==' ': i+=1
        if s[i]==')': return (kind,args),i+1

def chain_of(ast):
    # returns (products [(sign,a,b)] dataflow-first, kindstr) or None
    prods=[]
    def leaf(x): return x[0]=='id'
    n=ast
    while True:
        k,a=n
        if k in ('fmadd','fnmadd') and len(a)==3 and leaf(a[0]) and leaf(a[1]):
            prods.append(('+' if k=='fmadd' else '-',a[0][1],a[1][1])); n=a[2]
        elif k=='mul' and leaf(a[0]) and leaf(a[1]):
            prods.append(('+',a[0][1],a[1][1])); prods.reverse()
            return prods,'fma'
        else: return None
def addchain_of(ast):
    terms=[]; n=ast
    while True:
        k,a=n
        if k=='add' and len(a)==2 and a[0][0]=='id':
            terms.append(a[0][1]); n=a[1]
        elif k=='id':
            terms.append(a); terms.reverse(); return terms,'add'
        else: return None

def emit_fma(prods):
    # prods dataflow-first; prods[0] must be '+'
    s=f'_mm256_mul_pd({prods[0][1]}, {prods[0][2]})'
    for sg,a,b in prods[1:]:
        f='fmadd' if sg=='+' else 'fnmadd'
        s=f'_mm256_{f}_pd({a}, {b}, {s})'
    return s
def emit_add(terms):
    s=terms[0]
    for t in terms[1:]: s=f'_mm256_add_pd({t}, {s})'
    return s
def emit_add_tree(terms):
    if len(terms)==1: return terms[0]
    h=len(terms)//2
    return f'_mm256_add_pd({emit_add_tree(terms[:h])}, {emit_add_tree(terms[h:])})'
def emit_split2c(prods):
    h=(len(prods)+1)//2
    A=fix_pos(prods[:h]); B=fix_pos(prods[h:])
    if not A or not B: return None
    return f'_mm256_add_pd({emit_fma(A)}, {emit_fma(B)})'
def emit_split3(prods):
    P=[fix_pos(prods[i::3]) for i in range(3)]
    if any(p is None for p in P): return None
    return (f'_mm256_add_pd({emit_fma(P[0])}, '
            f'_mm256_add_pd({emit_fma(P[1])}, {emit_fma(P[2])}))')
def fix_pos(prods):
    if prods[0][0]=='+': return prods
    for j,p in enumerate(prods):
        if p[0]=='+':
            prods[0],prods[j]=prods[j],prods[0]; return prods
    return None
def emit_split2(prods):
    A=fix_pos(prods[0::2]); B=fix_pos(prods[1::2])
    if not A or not B: return None
    return f'_mm256_add_pd({emit_fma(A)}, {emit_fma(B)})'

def rewrite(src,strat,dc):
    out=[]; ci=0
    for line in src.splitlines(keepends=True):
        m=re.match(r'(\s*(?:const )?__m256d \w+ = )(_mm256_\w+.*)(;\s*)$',line)
        if not m: out.append(line); continue
        pre,rhs,post=m.groups()
        try: ast,_=parse_call(rhs,0)
        except Exception: out.append(line); continue
        ch=chain_of(ast)
        if ch and len(ch[0])>=3:
            prods=ch[0][:]; n=len(prods)
            if strat=='id': e=emit_fma(prods)
            elif strat=='rev': e=emit_fma(fix_pos(list(reversed(prods))))
            elif strat=='alt':
                e=emit_fma(prods if ci%2==0 else fix_pos(list(reversed(prods))))
            elif strat=='stag':
                r=ci%n; e=emit_fma(fix_pos(prods[r:]+prods[:r]))
            elif strat=='split2':
                e=emit_split2(prods) or emit_fma(prods)
            elif strat=='split2c':
                e=emit_split2c(prods) or emit_fma(prods)
            elif strat=='split3':
                e=(emit_split3(prods) if len(prods)>=6 else None) or emit_fma(prods)
            else: e=emit_fma(prods)
            if e is None: e=emit_fma(prods)
            out.append(pre+e+post); ci+=1; continue
        ac=addchain_of(ast)
        if ac and len(ac[0])>=4 and dc:
            out.append(pre+emit_add_tree(ac[0])+post); continue
        if ac and len(ac[0])>=4 and strat=='id':
            out.append(pre+emit_add(ac[0])+post); continue
        out.append(line)
    return ''.join(out)

def count(path):
    subprocess.run(GCC+[path,'-o','/tmp/sk.s'],check=True)
    ls=[l for l in open('/tmp/sk.s') if l.startswith('\t') and not l.lstrip().startswith('.')]
    sp=sum(1 for l in ls if SPILL.search(l))
    return len(ls),sp

if __name__=='__main__':
    f=sys.argv[1]; strat=sys.argv[2]; dc='--dc' in sys.argv
    src=open(f).read()
    env=dict(os.environ); env.setdefault('VFFT_NO_ANYK_TAIL','1')
    new=rewrite(src,strat,dc)
    if strat=='id' and not dc:
        print('IDENTITY:' , 'BYTE-EXACT' if new==src else 'MISMATCH')
    open('/tmp/skew.c','w').write(new)
    i,s=count('/tmp/skew.c')
    print(f'{strat}{"+dc" if dc else ""} -> {i}/{s}')
