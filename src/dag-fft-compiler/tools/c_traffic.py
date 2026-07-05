#!/usr/bin/env python3
"""Belady traffic of an EMITTED codelet's order, parsed from the C.
Seam scratch (spill_re/im[k]) is treated the way SRA treats it:
stores bind slot->producer, loads are aliases of the producer (no op).
Consts (set1) are budget-exempt. Usage: c_traffic.py file.c [R]"""
import re, sys
src=sys.argv[1]; R=int(sys.argv[2]) if len(sys.argv)>2 else 16
lines=open(src).read().splitlines()
defre=re.compile(r'^\s*(?:const )?__m256d (t\d+) = (.*);\s*$')
stre =re.compile(r'^\s*(spill_(?:re|im))\[(\d+)\] = (t\d+);')
ldre =re.compile(r'^\s*(?:const )?__m256d (t\d+) = (spill_(?:re|im))\[(\d+)\];')
alias={}; slotval={}
order=[]; kind={}; preds={}
for l in lines:
    m=ldre.match(l)
    if m:
        alias[m.group(1)]=slotval[(m.group(2),m.group(3))]; continue
    m=stre.match(l)
    if m:
        v=m.group(3); slotval[(m.group(1),m.group(2))]=alias.get(v,v); continue
    m=defre.match(l)
    if m:
        n,rhs=m.group(1),m.group(2)
        k='C' if 'set1' in rhs else 'L' if ('loadu' in rhs or 'load_pd' in rhs) else 'A'
        ps=[alias.get(p,p) for p in re.findall(r'\bt\d+\b',rhs)]
        order.append(n); kind[n]=k; preds[n]=ps; continue
    for v in re.findall(r'storeu_pd\([^,]+, *(t\d+)\)',l):
        s=f"sink_{len(order)}"; order.append(s); kind[s]='K'; preds[s]=[alias.get(v,v)]
users={n:[] for n in order}
for n in order:
    for p in preds[n]:
        if p in users: users[p].append(n)
pos={t:i for i,t in enumerate(order)}
nu={t:sorted(pos[u] for u in users[t]) for t in order}
cur={t:0 for t in order}; F=set(); stored=set(); tr=0; ml=0
def nxt(v,s):
    while cur[v]<len(nu[v]) and nu[v][cur[v]]<=s: cur[v]+=1
    return nu[v][cur[v]] if cur[v]<len(nu[v]) else 1<<30
for s,v in enumerate(order):
    for p in preds[v]:
        if p not in kind or kind[p]=='C': continue
        if p not in F:
            if len(F)>=R:
                ev=max((w for w in F if w not in preds[v]),key=lambda w:nxt(w,s))
                if nxt(ev,s)<(1<<30) and ev not in stored: tr+=1; stored.add(ev)
                F.discard(ev)
            F.add(p); tr+=1
    for p in preds[v]:
        if p in kind and kind[p]!='C' and nxt(p,s)>=(1<<30): F.discard(p)
    if kind[v] not in ('C','K') and users[v]:
        if len(F)>=R:
            ev=max(F,key=lambda w:nxt(w,s))
            if nxt(ev,s)<(1<<30) and ev not in stored: tr+=1; stored.add(ev)
            F.discard(ev)
        F.add(v)
    ml=max(ml,len(F))
print(f"{src}: nodes={len(order)} traffic@R{R}={tr} maxlive={ml}")
