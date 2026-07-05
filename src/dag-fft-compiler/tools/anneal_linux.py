#!/usr/bin/env python3
"""Linux-native schedule annealer (container port of tools/anneal.py).

Same validated objective: MINIMIZE total insns, gated on (a) FMA count
invariant (arithmetic/port guard) and (b) spills never above the su baseline.
Scoring = inject order -> gen_radix (native) -> gcc -S -> parse .s.
No link, no objdump, no dlopen. Memoized. Adaptive reheat on stall.

Usage: anneal_linux.py R [iters] [seed]
"""
import os, re, sys, math, random, subprocess, hashlib

R     = int(sys.argv[1]) if len(sys.argv) > 1 else 13
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
random.seed(SEED)

GENDIR = "/home/claude/vectorfft/VectorFFT-dev-arbitraryTail/src/dag-fft-compiler/generator"
GEN    = GENDIR + "/_build/default/bin/gen_radix.exe"
WORK   = "/home/claude/sched_search/work"
os.makedirs(WORK, exist_ok=True)
GCC    = ["gcc", "-O3", "-mavx2", "-mfma", "-march=raptorlake", "-w", "-S"]
GENARGS = [str(R), "--in-place", "--isa", "avx2", "--su", "--emit-c"]

SPILL_RE = re.compile(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)')
FMA_RE   = re.compile(r'vf?n?m(add|sub)')

def run_gen(env_extra, out_c):
    env = dict(os.environ, VFFT_NO_ANYK_TAIL="1", **env_extra)
    with open(out_c, "w") as f:
        subprocess.run([GEN] + GENARGS, cwd=GENDIR, env=env,
                       stdout=f, stderr=subprocess.DEVNULL)

def get_dump():
    dump = f"{WORK}/dump.txt"
    run_gen({"VFFT_SCHED_DUMP": dump}, f"{WORK}/su.c")
    order, preds, dagsig = [], {}, None
    for line in open(dump):
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            if line.startswith("#dagsig "): dagsig = line.split()[1]
            continue
        head, _, rest = line.partition(":")
        t = int(head)
        order.append(t)
        preds[t] = [int(x) for x in rest.split()] if rest.strip() else []
    return order, preds, dagsig

memo = {}
DAGSIG = None
def score(order):
    key = hashlib.md5(",".join(map(str, order)).encode()).hexdigest()
    if key in memo:
        return memo[key], True
    of = f"{WORK}/ord.txt"
    with open(of, "w") as f:
        if DAGSIG: f.write(f"#dagsig {DAGSIG}\n")
        f.write("\n".join(map(str, order)) + "\n")
    cf, sf = f"{WORK}/cand.c", f"{WORK}/cand.s"
    if os.path.exists(sf): os.remove(sf)
    run_gen({"VFFT_SCHED_ORDER": of}, cf)
    r = subprocess.run(GCC + [cf, "-o", sf], capture_output=True)
    if r.returncode != 0 or not os.path.exists(sf):
        memo[key] = None
        return None, False
    insns = spills = fma = 0
    for l in open(sf):
        if l.startswith("\t") and not l.lstrip().startswith("."):
            insns += 1
            if SPILL_RE.search(l): spills += 1
            if FMA_RE.search(l):   fma += 1
    memo[key] = (insns, spills, fma)
    return memo[key], False

# --- move operators (identical semantics to tools/anneal.py) ---
def reinsert(order, preds, succs):
    o = order[:]
    i = random.randrange(len(o))
    node = o.pop(i)
    pos = {t: j for j, t in enumerate(o)}
    lo = max((pos[p] for p in preds[node] if p in pos), default=-1) + 1
    hi = min((pos[s] for s in succs[node] if s in pos), default=len(o))
    o.insert(random.randint(lo, hi), node)
    return o

def block_move(order, preds, succs):
    n = len(order)
    i = random.randrange(n)
    j = min(n, i + random.randint(2, 6))
    block = order[i:j]; bset = set(block)
    rest = order[:i] + order[j:]
    pos = {t: k for k, t in enumerate(rest)}
    lo = -1; hi = len(rest)
    for nd in block:
        for p in preds[nd]:
            if p in pos and p not in bset: lo = max(lo, pos[p])
        for s in succs[nd]:
            if s in pos and s not in bset: hi = min(hi, pos[s])
    lo += 1
    if lo > hi: return order
    p = random.randint(lo, hi)
    return rest[:p] + block + rest[p:]

def segment_reverse(order, preds, succs):
    n = len(order)
    i = random.randrange(n)
    j = min(n, i + random.randint(2, 8))
    seg = order[i:j]; sset = set(seg)
    for nd in seg:
        if any(p in sset for p in preds[nd]):
            return order
    return order[:i] + seg[::-1] + order[j:]

def propose(order, preds, succs):
    r = random.random()
    if r < 0.6:    return reinsert(order, preds, succs)
    elif r < 0.85: return block_move(order, preds, succs)
    else:          return segment_reverse(order, preds, succs)

def main():
    global DAGSIG
    su_order, preds, DAGSIG = get_dump()
    succs = {t: [] for t in su_order}
    for t in su_order:
        for p in preds[t]:
            if p in succs: succs[p].append(t)
    (b_ins, b_sp, b_fma), _ = score(su_order)
    print(f"R={R} nodes={len(su_order)} su: insns={b_ins} spills={b_sp} "
          f"fma={b_fma}  iters={ITERS} seed={SEED}", flush=True)

    cur, cur_ins = su_order, b_ins
    best, best_ins, best_sp = su_order, b_ins, b_sp
    T = max(3.0, b_ins * 0.02)
    evals = hits = 0
    stall = 0
    for it in range(ITERS):
        cand = propose(cur, preds, succs)
        r, cached = score(cand)
        if cached: hits += 1
        else: evals += 1
        if r is None:
            continue
        ins, sp, fma = r
        if fma != b_fma or sp > b_sp:
            continue
        if ins < cur_ins or random.random() < math.exp(-(ins - cur_ins) / max(T, 1e-6)):
            cur, cur_ins = cand, ins
            if ins < best_ins or (ins == best_ins and sp < best_sp):
                best, best_ins, best_sp = cand, ins, sp
                stall = 0
                print(f"  it={it:4d} T={T:6.2f} NEW BEST insns={ins} spills={sp}"
                      f" (su {b_ins}/{b_sp})", flush=True)
        stall += 1
        if stall > 120:                     # adaptive reheat from the best
            cur, cur_ins = best, best_ins
            T = max(3.0, b_ins * 0.01)
            stall = 0
        T *= 0.985

    win = best_ins < b_ins and best_sp <= b_sp
    strong = best_ins < b_ins and best_sp < b_sp
    print(f"\nDONE evals={evals} memo_hits={hits}  su({b_ins}/{b_sp})"
          f"  best({best_ins}/{best_sp})"
          f"  d_insns={b_ins - best_ins} d_spills={b_sp - best_sp}"
          f"  {'STRONG WIN' if strong else 'WIN' if win else 'no win'}", flush=True)
    if win:
        out = f"best_r{R}_s{SEED}.txt"
        with open(out, "w") as f:
            if DAGSIG: f.write(f"#dagsig {DAGSIG}\n")
            f.write("\n".join(map(str, best)) + "\n")
        print(f"best order -> {out}", flush=True)

if __name__ == "__main__":
    main()
