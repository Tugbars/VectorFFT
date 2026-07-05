#!/usr/bin/env python3
"""Faithful Python replica of the production picker (monolithic path) +
component ablations. Validates itself by exact order match vs the dump.
Keys: S=sink-first, D=cp_dist DESC, U=su_num ASC (tag always final).
Load laws: src (lazy + source order, production) | any (lazy, tag order
among ready) | arith (loads compete as arithmetic).
Usage: ablate.py dump kinds <keys e.g. SDU> <loadlaw>"""
import sys
dump,kf,KEYS,LL = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
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
sun={}
for t in su:
    ps=preds[t]; k=kind[t]
    if k in ('C','L'): sun[t]=1
    elif k=='N': sun[t]=sun[ps[0]]
    elif len(ps)==2:
        a,b=sun[ps[0]],sun[ps[1]]; sun[t]=a+1 if a==b else max(a,b)
    else:
        ss=sorted((sun[p] for p in ps),reverse=True)
        sun[t]=max(s+i for i,s in enumerate(ss))
unsp={t:len(preds[t]) for t in su}
ready=set(t for t in su if unsp[t]==0)
sched=set(); out=[]
loads=sorted(t for t in su if kind[t]=='L'); li=0
def cmp_key(n):
    k=[]
    if 'S' in KEYS: k.append(0 if not users[n] else 1)
    if 'D' in KEYS: k.append(-cp[n])
    if 'U' in KEYS: k.append(sun[n])
    k.append(n)
    return tuple(k)
while len(sched)<len(su):
    if LL=='arith':
        pool=ready
    else:
        pool={n for n in ready if kind[n]!='L'}
    if pool:
        n=min(pool,key=cmp_key)
    else:
        if LL=='src':
            while loads[li] in sched: li+=1
            n=loads[li]
            if n not in ready: raise SystemExit("stall")
        else:
            n=min((x for x in ready if kind[x]=='L'))
    sched.add(n); ready.discard(n); out.append(n)
    for u in users[n]:
        unsp[u]-=1
        if unsp[u]==0: ready.add(u)
print("\n".join(map(str,out)))
