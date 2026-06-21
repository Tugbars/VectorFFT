#!/usr/bin/env python3
"""Phase 2.1 — schedule annealer for a single monolithic codelet.

Searches the codelet's internal instruction order to MINIMIZE realized asm
stack-spills (the Phase-0.3-validated proxy for runtime; peak_live is NOT used).
Seeded from su's order; moves are legal reinsertions (every candidate is a valid
topological order). Scoring = inject order -> WSL gen -> native gcc -> objdump.

Usage:  python anneal.py R [iters] [seed]
"""
import os, re, sys, math, random, subprocess

R     = int(sys.argv[1]) if len(sys.argv) > 1 else 13
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
random.seed(SEED)

EXP_WIN = r"C:\Users\Tugbars\Desktop\highSpeedFFT\src\dag-fft-compiler\experiments\sched_search"
EXP_WSL = "/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/experiments/sched_search"
GENDIR  = "/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator"
GENW    = GENDIR + "/_build/default/bin/gen_radix.exe"
GCC     = r"C:\mingw152\mingw64\bin\gcc.exe"
OBJDUMP = r"C:\mingw152\mingw64\bin\objdump.exe"
PROD    = ["-O3","-mavx2","-mfma","-march=native","-fpermissive","-w"]
SPILL_RE = re.compile(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)')
INSN_RE  = re.compile(r'^\s+[0-9a-f]+:')              # an emitted instruction line
FMA_RE   = re.compile(r'vf?n?m(add|sub)')            # FMA-port ops (must be reorder-invariant)

def wsl(envset, redirect_to=None):
    cmd = ["wsl.exe","-e","bash","-lc",
           f"cd {GENDIR} && {envset} {GENW} {R} --in-place --isa avx2 --su --emit-c"]
    if redirect_to:
        with open(redirect_to,"w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    else:
        return subprocess.run(cmd, capture_output=True, text=True).stdout

def get_dump():
    dump_wsl = EXP_WSL + "/anneal_dump.txt"
    c = EXP_WIN + r"\anneal_su.c"
    wsl(f"VFFT_SCHED_DUMP={dump_wsl}", redirect_to=c)
    order, preds = [], {}
    with open(EXP_WIN + r"\anneal_dump.txt") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            head, _, rest = line.partition(":")
            t = int(head)
            order.append(t)
            preds[t] = [int(x) for x in rest.split()] if rest.strip() else []
    return order, preds

def score(order, tag):
    """Inject `order`, compile, return (total_insns, stack_spills, fma_count)
    or None if the order is illegal (compile fails). Objective = total_insns
    (subsumes spills + reg-reg moves); fma_count is the reorder-invariant
    arithmetic/port guard (must equal baseline). Per the noise-robust win rule:
    reduce spills without increasing total instructions or port pressure."""
    of_win = EXP_WIN + rf"\anneal_ord_{tag}.txt"
    of_wsl = EXP_WSL + f"/anneal_ord_{tag}.txt"
    with open(of_win,"w") as f:
        f.write("\n".join(str(t) for t in order) + "\n")
    cfile = EXP_WIN + rf"\anneal_cand_{tag}.c"
    dll   = EXP_WIN + rf"\anneal_cand_{tag}.dll"
    if os.path.exists(dll): os.remove(dll)
    wsl(f"VFFT_SCHED_ORDER={of_wsl}", redirect_to=cfile)
    subprocess.run([GCC]+PROD+["-shared",cfile,"-o",dll], stderr=subprocess.DEVNULL)
    if not os.path.exists(dll): return None
    lines = subprocess.run([OBJDUMP,"-d","--no-show-raw-insn",dll],
                           capture_output=True, text=True).stdout.splitlines()
    insns = sum(1 for l in lines if INSN_RE.match(l))
    spills = sum(1 for l in lines if SPILL_RE.search(l))
    fma = sum(1 for l in lines if FMA_RE.search(l))
    return (insns, spills, fma)

def reinsert(order, preds, succs):
    """Move one node to a random legal position."""
    o = order[:]
    i = random.randrange(len(o))
    node = o.pop(i)
    pos = {t:j for j,t in enumerate(o)}
    lo = max((pos[p] for p in preds[node] if p in pos), default=-1) + 1
    hi = min((pos[s] for s in succs[node] if s in pos), default=len(o))
    o.insert(random.randint(lo, hi), node)
    return o

def block_move(order, preds, succs):
    """Move a contiguous block (internal order preserved) to a legal position."""
    n = len(order)
    i = random.randrange(n)
    j = min(n, i + random.randint(2, 6))
    block = order[i:j]; bset = set(block)
    rest = order[:i] + order[j:]
    pos = {t:k for k,t in enumerate(rest)}
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
    """Reverse a contiguous segment iff it has no internal dependency edges
    (an antichain) — otherwise reversal would violate a pred->succ edge."""
    n = len(order)
    i = random.randrange(n)
    j = min(n, i + random.randint(2, 8))
    seg = order[i:j]; sset = set(seg)
    for nd in seg:
        if any(p in sset for p in preds[nd]):
            return order  # internal edge -> not reversible
    return order[:i] + seg[::-1] + order[j:]

def propose(order, preds, succs):
    r = random.random()
    if r < 0.6:   return reinsert(order, preds, succs)
    elif r < 0.85: return block_move(order, preds, succs)
    else:          return segment_reverse(order, preds, succs)

def main():
    su_order, preds = get_dump()
    succs = {t:[] for t in su_order}
    for t in su_order:
        for p in preds[t]:
            if p in succs: succs[p].append(t)
    b_ins, b_sp, b_fma = score(su_order, "base")
    print(f"R={R} nodes={len(su_order)} su: insns={b_ins} spills={b_sp} fma={b_fma}"
          f"  iters={ITERS} seed={SEED}", flush=True)

    # Objective = total instructions (subsumes spills + reg-reg moves).
    # Gate: a candidate is only valid if it does NOT increase spills and keeps
    # the FMA count invariant (reorder must not change arithmetic / FMA-port
    # pressure). That is the noise-robust win rule: reduce spills without
    # increasing total instructions or saturating a port.
    cur, cur_ins = su_order, b_ins
    best, best_ins, best_sp = su_order, b_ins, b_sp
    T = max(3.0, b_ins * 0.02)
    evals = 0
    for it in range(ITERS):
        cand = propose(cur, preds, succs)
        r = score(cand, "c"); evals += 1
        if r is None:
            continue
        ins, sp, fma = r
        if fma != b_fma or sp > b_sp:      # arithmetic/port guard + spills-not-up gate
            continue
        if ins < cur_ins or random.random() < math.exp(-(ins - cur_ins)/max(T,1e-6)):
            cur, cur_ins = cand, ins
            if ins < best_ins:
                best, best_ins, best_sp = cand, ins, sp
                print(f"  it={it:4d} T={T:5.2f} NEW BEST insns={ins} spills={sp}"
                      f" (su insns={b_ins} spills={b_sp})", flush=True)
        T *= 0.97
    win = best_ins < b_ins and best_sp < b_sp
    print(f"\nDONE evals={evals}  su(insns={b_ins},spills={b_sp})"
          f"  best(insns={best_ins},spills={best_sp})"
          f"  d_insns={b_ins-best_ins} d_spills={b_sp-best_sp}  {'WIN' if win else 'no win'}",
          flush=True)
    if win:
        with open(EXP_WIN + rf"\anneal_best_{R}.txt","w") as f:
            f.write("\n".join(str(t) for t in best) + "\n")
        print(f"best order -> anneal_best_{R}.txt", flush=True)

if __name__ == "__main__":
    main()
