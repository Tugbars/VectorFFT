#!/usr/bin/env python3
"""Self-spilling probe: Belady-optimal explicit allocation on an emitted
"""RECORDED NEGATIVE (doc 70): self-spilling falsified on pow2 — the Belady plan itself loses to the blocked construction (76-88 vs 52-59 ops @R=16), and mechanism defense is compiler-specific. Kept for the primes-maybe and for MSVC baselining."""
monolithic codelet. Keeps live slot-takers <= B; evicts furthest-next-use
to a static aligned scratch array; reloads are barrier-laundered
(__asm__ "+x") so the compiler cannot store-forward and resurrect the
original live range. Consts (set1) are budget-exempt (rematerializable).
Usage: spill_inject.py in.c out.c B"""
import re, sys
src,dst,B = sys.argv[1],sys.argv[2],int(sys.argv[3])
lines=open(src).read().splitlines()
defre=re.compile(r'^(\s*)const __m256d (t\d+) = (.*);\s*$')
kind={}; defs={}
for i,l in enumerate(lines):
    m=defre.match(l)
    if m:
        n=m.group(2); defs[n]=i
        kind[n]='C' if 'set1' in m.group(3) else 'S'   # S = slot-taker
uses={n:[] for n in defs}
for i,l in enumerate(lines):
    for n in set(re.findall(r'\bt\d+\b',l)):
        if n in defs and i>defs[n]: uses[n].append(i)
cur={n:n for n in defs}          # current SSA name of each value
inreg=set(); slot={}; stored=set(); free=[]; nslots=0
nxt={n:0 for n in defs}
def nextuse(n,i):
    u=uses[n]
    while nxt[n]<len(u) and u[nxt[n]]<=i: nxt[n]+=1
    return u[nxt[n]] if nxt[n]<len(u) else 1<<30
out=[]; RL=[0]
def evict(i,protect,ind):
    global nslots
    v=max((w for w in inreg if w not in protect),key=lambda w:nextuse(w,i))
    inreg.discard(v)
    if nextuse(v,i)<(1<<30) and v not in stored:
        if v not in slot:
            slot[v]=free.pop() if free else nslots
            if slot[v]==nslots: nslots+=1
        out.append(f"{ind}vfft_scr[{slot[v]}] = {cur[v]};")
        stored.add(v)
for i,l in enumerate(lines):
    m=defre.match(l)
    ops=[n for n in set(re.findall(r'\bt\d+\b',l)) if n in defs and i>defs[n]]
    ind=(m.group(1) if m else re.match(r'^(\s*)',l).group(1))
    # ensure operands resident
    for p in sorted(ops):
        if kind[p]=='C': continue
        if p not in inreg:
            while len(inreg)>=B: evict(i,set(ops)|({m.group(2)} if m else set()),ind)
            RL[0]+=1; nn=f"{p}_r{RL[0]}"
            out.append(f"{ind}__m256d {nn} = vfft_scr[{slot[p]}];")
            cur[p]=nn; inreg.add(p)
            if nextuse(p,i+0)>=(1<<30): pass
    # rewrite uses to current names
    for p in ops:
        if cur[p]!=p: l=re.sub(rf'\b{p}\b',cur[p],l)
    if m:
        n=m.group(2)
        if kind[n]=='S':
            while len(inreg)>=B: evict(i,set(ops)|{n},ind)
            inreg.add(n)
    out.append(l)
    # drop dead values from the file
    for p in ops:
        if nextuse(p,i)>=(1<<30):
            inreg.discard(p)
            if p in slot: free.append(slot.pop(p))
body="\n".join(out)
hdr="#include <immintrin.h>\nstatic __m256d vfft_scr_a[%d] __attribute__((aligned(64)));"%max(nslots,1)
body=body.replace("#include <immintrin.h>",hdr,1)
inj='\n  __m256d * vfft_scr = vfft_scr_a; __asm__ volatile("" : "+r"(vfft_scr));'
m2=re.search(r"\)\s*\{", body)
body=body[:m2.end()]+inj+body[m2.end():]
open(dst,"w").write(body+"\n")
sys.stderr.write(f"slots={nslots} scratch_ops={body.count('vfft_scr[')}\n")
