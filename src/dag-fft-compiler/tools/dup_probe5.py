#!/usr/bin/env python3
"""v5 'deep v2': duplicate long-span multi-use cheap values at ANY depth,
gated on operand availability at the clone point:
  - operand is a leaf (load/set1)            -> reference original, cost 0
  - operand's last use >= clone point        -> reference original, cost 0
    (it is alive there anyway; zero lifetime extension by construction)
  - operand is cheap and recursively coverable -> re-derive (+1 def+barrier)
  - anything else                             -> reject candidate
Every emitted def carries its own one-value barrier. cost = emitted defs.
Usage: dup_probe5.py <codelet.c> <span_S> <cap> <maxcost>"""
import re, sys, subprocess
SRC, S, CAP, MAXCOST = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
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
def plan(n, p, emitted):
    """resolve operand names of n for a clone at line p; returns (opmap, ok)."""
    opmap={}
    for o in re.findall(r't\d+', rhs_of[n]):
        if o in emitted: opmap[o]=o+'_e'
        elif kind.get(o)=='leaf' or (uses.get(o) and uses[o][-1] >= p):
            opmap[o]=o
        elif kind.get(o)=='cheap':
            sub, ok = plan(o, p, emitted)
            if not ok or len(emitted)+1 > MAXCOST: return None, False
            emitted[o]=sub
            opmap[o]=o+'_e'
        else: return None, False
    return opmap, True
cands=[]
for n in defs:
    if kind.get(n)=='cheap' and len(uses[n])>=2 and uses[n][-1]-defs[n]>=S:
        emitted={}
        opmap, ok = plan(n, uses[n][-1], emitted)
        if ok and len(emitted)+1 <= MAXCOST:
            cands.append((uses[n][-1]-defs[n], len(emitted)+1, n, emitted, opmap))
cands.sort(key=lambda x:(-x[0], x[1]))
chosen=cands[:CAP]
ins={}
for span, cost, n, emitted, opmap in chosen:
    p = uses[n][-1]; ind = ind_of[n]; seq=[]
    for m in sorted(emitted, key=lambda x: defs[x]):     # interiors, topo order
        r = rhs_of[m]
        for o, nm in sorted(emitted[m].items(), key=lambda kv: -len(kv[0])):
            r = re.sub(rf'\b{o}\b', nm, r)
        seq.append(f'{ind}__m256d {m}_e = {r}; __asm__ volatile("" : "+x"({m}_e));')
    r = rhs_of[n]
    for o, nm in sorted(opmap.items(), key=lambda kv: -len(kv[0])):
        r = re.sub(rf'\b{o}\b', nm, r)
    seq.append(f'{ind}__m256d {n}_e = {r}; __asm__ volatile("" : "+x"({n}_e));')
    ins.setdefault(p, []).append((n, seq))
out=[]
for i,l in enumerate(lines):
    if i in ins:
        for n, seq in ins[i]:
            out.extend(seq)
            l = re.sub(rf'\b{n}\b', n+'_e', l)
    out.append(l)
open('/tmp/dup5.c','w').write("\n".join(out)+"\n")
subprocess.run(['gcc','-O3','-mavx2','-mfma','-march=raptorlake','-w','-S','/tmp/dup5.c','-o','/tmp/dup5.s'], check=True)
ls=[l for l in open('/tmp/dup5.s') if l.startswith('\t') and not l.lstrip().startswith('.')]
sp=sum(1 for l in ls if re.search(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)',l))
fm=sum(1 for l in ls if re.search(r'vf?n?m(add|sub)',l))
tc=sum(c for _,c,_,_,_ in chosen)
print(f"clones={len(chosen)} (defs={tc}) -> insns={len(ls)} spills={sp} fma={fm}")
