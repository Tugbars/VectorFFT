#!/usr/bin/env python3
"""Tie-break slot race: pluggable 3rd-key variants in the exact SR
replica. Keys: S sink-first, D cp_dist DESC, then ONE of:
  U classic SU (tree)   W first-owner DAG-SU (shared: one parent pays
  full, others see 1)   X shared-as-1 DAG-SU (all parents see 1 for
  any multi-use operand)   K dynamic kills DESC (operands retired)
  N cone-sinks ASC (distinct reachable sinks: narrow first)
  - none. Tag ASC always last. Load law: production (lazy+src).
Usage: ablate2.py dump kinds <keys e.g. SDW>"""
import sys
dump,kf,KEYS = sys.argv[1],sys.argv[2],sys.argv[3]
kind={}
for l in open(kf): t,k=l.split(); kind[int(t)]=k
preds={}; su=[]
for l in open(dump):
    l=l.strip()
    if not l or l.startswith('#'):continue
    h,_,r=l.partition(':'); t=int(h); su.append(t)
    preds[t]=[int(x) for x in r.split()] if r.strip() else []
users={t:[] for t in su}
for t in su:
    for p in preds[t]: users[p].append(t)
LAT={'C':0,'L':5}
cp={}
for t in reversed(su):
    cp[t]=LAT.get(kind[t],4)+max((cp[u] for u in users[t]),default=0)
def kary(ss):
    ss=sorted(ss,reverse=True)
    return max((s+i for i,s in enumerate(ss)),default=1)
def su_label(shared_mode):
    lab={}; owned=set()
    for t in su:                                   # tag/topo order
        ps=preds[t]; k=kind[t]
        def see(p):
            if len(users[p])<=1: return lab[p]
            if shared_mode=='X': return 1
            if p in owned: return 1
            owned.add(p); return lab[p]            # first owner pays
        if k in ('C','L'): lab[t]=1
        elif k=='N': lab[t]=see(ps[0])
        elif len(ps)==2:
            a,b=see(ps[0]),see(ps[1]); lab[t]=a+1 if a==b else max(a,b)
        else: lab[t]=kary([see(p) for p in ps])
    return lab
sunU=su_label(None)                                 # classic ignores sharing:
# recompute classic exactly (every parent sees full label)
sunU={}
for t in su:
    ps=preds[t]; k=kind[t]
    if k in ('C','L'): sunU[t]=1
    elif k=='N': sunU[t]=sunU[ps[0]]
    elif len(ps)==2:
        a,b=sunU[ps[0]],sunU[ps[1]]; sunU[t]=a+1 if a==b else max(a,b)
    else: sunU[t]=kary([sunU[p] for p in ps])
sunW=su_label('W'); sunX=su_label('X')
csink={}
for t in reversed(su):
    s=set()
    if not users[t]: s={t}
    for u in users[t]: s|=csink[u]
    csink[t]=s
ncone={t:len(csink[t]) for t in su}
last=[None]
unsp={t:len(preds[t]) for t in su}
remu={t:len(users[t]) for t in su}
ready={t for t in su if unsp[t]==0}
sched=set(); out=[]
loads=sorted(t for t in su if kind[t]=='L'); li=0
def key(n):
    k=[]
    if 'S' in KEYS: k.append(0 if not users[n] else 1)
    if 'D' in KEYS: k.append(-cp[n])
    if 'U' in KEYS: k.append(sunU[n])
    if 'W' in KEYS: k.append(sunW[n])
    if 'X' in KEYS: k.append(sunX[n])
    if 'K' in KEYS: k.append(-sum(1 for p in preds[n] if kind[p]!='C' and remu[p]==1))
    if 'A' in KEYS:
        k.append(-len(csink[n] & csink[last[0]]) if last[0] is not None else 0)
    if 'N' in KEYS: k.append(ncone[n])
    k.append(n); return tuple(k)
while len(sched)<len(su):
    pool={n for n in ready if kind[n]!='L'}
    if pool: n=min(pool,key=key)
    else:
        while loads[li] in sched: li+=1
        n=loads[li]
    sched.add(n); ready.discard(n); out.append(n); last[0]=n
    for p in preds[n]: remu[p]-=1
    for u in users[n]:
        unsp[u]-=1
        if unsp[u]==0: ready.add(u)
print("\n".join(map(str,out)))
