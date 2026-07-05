#!/usr/bin/env python3
"""v4 chained duplication: for a long-span multi-use value with DEEP operands,
re-derive the whole chain at the last use from RE-LOADED leaves; barrier only
the re-loaded leaves (interior recomputation over barriered leaves is
automatically opaque to FRE). Chain restricted to add/sub/mul + leaves.
Usage: dup_probe4.py <codelet.c> <span_S> <cap> [maxchain]"""
import re, sys, subprocess
SRC, S, CAP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
MAXCHAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 8
lines = open(SRC).read().splitlines()
defre = re.compile(r'^(\s*)const __m256d (t\d+) = (.*);\s*$')
defs, rhs_of, ind_of, kind = {}, {}, {}, {}
for i,l in enumerate(lines):
    m = defre.match(l)
    if not m: continue
    n, rhs = m.group(2), m.group(3)
    defs[n]=i; rhs_of[n]=rhs; ind_of[n]=m.group(1)
    if 'set1' in rhs or 'loadu' in rhs or 'load_pd' in rhs: kind[n]='leaf'
    elif re.match(r'_mm256_(add|sub|mul)_pd\(t\d+, ?t\d+\)$', rhs): kind[n]='cheap'
    else: kind[n]='hard'
uses={n:[] for n in defs}
for i,l in enumerate(lines):
    for n in set(re.findall(r'\bt\d+\b', l)):
        if n in defs and i>defs[n]: uses[n].append(i)
def closure(n, acc):
    """topo-ordered chain defs needed to recompute n; None if hard node hit."""
    if n in acc: return acc
    k = kind.get(n)
    if k == 'leaf': acc[n]='leaf'; return acc
    if k != 'cheap': return None
    for o in re.findall(r't\d+', rhs_of[n]):
        if closure(o, acc) is None: return None
    acc[n]='cheap'; return acc
cands=[]
for n in defs:
    if kind.get(n)=='cheap' and len(uses[n])>=2 and uses[n][-1]-defs[n]>=S:
        c = closure(n, {})
        if c is not None and len(c) <= MAXCHAIN:
            cands.append((uses[n][-1]-defs[n], len(c), n, c))
cands.sort(key=lambda x:(-x[0], x[1]))
chosen = cands[:CAP]
ins = {}
for span, clen, n, c in chosen:
    last = uses[n][-1]; ind = ind_of[n]
    seq=[]
    order = sorted(c, key=lambda x: defs[x])           # topo by def position
    for m in order:
        r = rhs_of[m]
        for o in sorted(c, key=len, reverse=True):     # rewrite operands to _d
            r = re.sub(rf'\b{o}\b', o+'_e', r)
        if c[m]=='leaf':
            seq.append(f"{ind}__m256d {m}_e = {rhs_of[m]}; __asm__ volatile(\"\" : \"+x\"({m}_e));")
        else:
            seq.append(f"{ind}const __m256d {m}_e = {r};")
    ins.setdefault(last, []).append((n, seq))
out=[]
for i,l in enumerate(lines):
    if i in ins:
        for n, seq in ins[i]:
            out.extend(seq)
            l = re.sub(rf'\b{n}\b', n+'_e', l)
    out.append(l)
open('/tmp/dup4.c','w').write("\n".join(out)+"\n")
subprocess.run(['gcc','-O3','-mavx2','-mfma','-march=raptorlake','-w','-S','/tmp/dup4.c','-o','/tmp/dup4.s'], check=True)
ls=[l for l in open('/tmp/dup4.s') if l.startswith('\t') and not l.lstrip().startswith('.')]
sp=sum(1 for l in ls if re.search(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)',l))
fm=sum(1 for l in ls if re.search(r'vf?n?m(add|sub)',l))
tot_chain=sum(cl for _,cl,_,_ in chosen)
print(f"chains={len(chosen)} (nodes={tot_chain}) -> insns={len(ls)} spills={sp} fma={fm}")
