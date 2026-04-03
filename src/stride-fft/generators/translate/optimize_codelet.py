#!/usr/bin/env python3
"""
optimize_codelet.py — Production optimizer for R=32/64 AVX2/AVX-512 codelets.

Three passes:
  Pass 1: Build full dependency graph with Sethi-Ullman register numbers
  Pass 2: Schedule using SU numbers — evaluate heavier subtrees first,
           promote stores as early as possible to free registers
  Pass 3: ILP interleaving — pair independent ops from different dependency
           chains to saturate dual FMA ports

The key insight: genfft's DAG has sharing (it's a DAG, not a tree), so pure
Sethi-Ullman doesn't apply directly. We use the SU heuristic for priority
in a list scheduler: ops with higher SU numbers (needing more registers to
evaluate their subtree) get scheduled first when they're in the ready set.
"""
import sys, re
from collections import defaultdict

sys.path.insert(0, '.')
from translate_genfft_tw import extract_fma_body, parse_body, emit_codelet


def build_graph(ops):
    """Build dependency graph. Returns producers, successors, predecessors."""
    producers = {}  # var -> op_index
    succs = defaultdict(set)  # op_index -> set of successor op_indices
    preds = defaultdict(set)  # op_index -> set of predecessor op_indices

    for i, op in enumerate(ops):
        if op[0] in ('LOAD', 'LOAD_W'):
            producers[op[1]] = i
        elif op[0] != 'STORE' and len(op) > 1:
            producers[op[1]] = i

    def get_inputs(op):
        if op[0] in ('LOAD', 'LOAD_W'):
            return set()
        elif op[0] == 'STORE':
            refs = set()
            for var in producers:
                if re.search(r'\b' + re.escape(var) + r'\b', op[3]):
                    refs.add(var)
            return refs
        elif op[0] in ('FMA', 'FNMS', 'FMS', 'FNMA'):
            return {v for v in (op[2], op[3], op[4]) if v in producers}
        elif op[0] == 'BINOP':
            return {v for v in (op[3], op[4]) if v in producers}
        elif op[0] == 'NEG':
            return {op[2]} & set(producers.keys())
        return set()

    for i, op in enumerate(ops):
        for var in get_inputs(op):
            if var in producers:
                pred_idx = producers[var]
                preds[i].add(pred_idx)
                succs[pred_idx].add(i)

    return producers, succs, preds


def compute_su_numbers(ops, succs, preds):
    """Compute Sethi-Ullman register need estimate per op.
    
    For DAGs (not trees), we use: SU(node) = max over children of SU(child),
    with +1 when we need to hold a result while evaluating siblings.
    This is a heuristic — exact SU only applies to trees.
    """
    n = len(ops)
    su = [1] * n  # default: 1 register needed

    # Topological order (reverse — process leaves first)
    visited = [False] * n
    topo_rev = []

    def dfs(node):
        if visited[node]:
            return
        visited[node] = True
        for s in succs.get(node, set()):
            dfs(s)
        topo_rev.append(node)

    for i in range(n):
        if not visited[i]:
            dfs(i)

    # Process in reverse topological order (leaves first)
    for node in topo_rev:
        children_su = sorted([su[c] for c in preds.get(node, set())], reverse=True)
        if not children_su:
            su[node] = 1  # leaf
        elif len(children_su) == 1:
            su[node] = children_su[0]
        else:
            # Sethi-Ullman: evaluate heaviest subtree first
            # Need max(su[0], su[1]+1, su[2]+2, ...)
            su[node] = max(c + i for i, c in enumerate(children_su))

    return su


def count_consumers(ops, producers, succs):
    """Count how many ops consume each variable. Used for last-use detection."""
    var_consumers = defaultdict(int)
    for i, op in enumerate(ops):
        if op[0] in ('LOAD', 'LOAD_W'):
            continue
        if op[0] == 'STORE':
            for var in producers:
                if re.search(r'\b' + re.escape(var) + r'\b', op[3]):
                    var_consumers[var] += 1
        elif op[0] in ('FMA', 'FNMS', 'FMS', 'FNMA'):
            for v in (op[2], op[3], op[4]):
                if v in producers:
                    var_consumers[v] += 1
        elif op[0] == 'BINOP':
            for v in (op[3], op[4]):
                if v in producers:
                    var_consumers[v] += 1
        elif op[0] == 'NEG':
            if op[2] in producers:
                var_consumers[op[2]] += 1
    return var_consumers


def schedule_su(ops):
    """List scheduler using Sethi-Ullman priorities.
    
    Ready set: ops whose predecessors are all scheduled.
    Priority: (is_store, -SU_number, original_index)
    
    Stores get highest priority (free registers immediately).
    Among non-stores, higher SU = schedule first (evaluate heavy subtrees
    while we have registers, light ones can wait).
    """
    import heapq

    n = len(ops)
    producers, succs, preds = build_graph(ops)
    su = compute_su_numbers(ops, succs, preds)

    # In-degree
    in_deg = [len(preds.get(i, set())) for i in range(n)]

    # Ready queue: (priority, original_index)
    # Lower priority number = scheduled first
    # Stores: priority -2 (highest)
    # Others: priority based on -SU (higher SU = more urgent)
    ready = []
    for i in range(n):
        if in_deg[i] == 0:
            is_store = 1 if ops[i][0] == 'STORE' else 0
            heapq.heappush(ready, (-is_store, -su[i], i))

    scheduled = []
    emitted = set()

    while ready:
        _, _, idx = heapq.heappop(ready)
        if idx in emitted:
            continue
        scheduled.append(ops[idx])
        emitted.add(idx)

        for s in succs.get(idx, set()):
            if s not in emitted:
                in_deg[s] -= 1
                if in_deg[s] <= 0:
                    is_store = 1 if ops[s][0] == 'STORE' else 0
                    heapq.heappush(ready, (-is_store, -su[s], s))

    return scheduled


def ilp_interleave(ops):
    """Post-pass: swap adjacent dependent ops with nearby independent ones.
    
    For each pair (a, b): if b depends on a, find a nearby independent c and swap.
    Uses direct variable-name checks — no index tracking that goes stale after swaps.
    """
    def var_produced(op):
        if op[0] in ('LOAD', 'LOAD_W'):
            return op[1]
        elif op[0] != 'STORE' and len(op) > 1:
            return op[1]
        return None

    def vars_consumed(op):
        if op[0] in ('LOAD', 'LOAD_W'):
            return set()
        elif op[0] in ('FMA', 'FNMS', 'FMS', 'FNMA'):
            return {op[2], op[3], op[4]}
        elif op[0] == 'BINOP':
            return {op[3], op[4]}
        elif op[0] == 'NEG':
            return {op[2]}
        elif op[0] == 'STORE':
            return set(re.findall(r'\b[A-Za-z_]\w*\b', op[3]))
        return set()

    result = list(ops)
    n = len(result)

    for i in range(n - 1):
        a = result[i]
        b = result[i + 1]

        a_prod = var_produced(a)
        b_cons = vars_consumed(b)

        if a_prod and a_prod in b_cons:
            for j in range(i + 2, min(i + 10, n)):
                c = result[j]
                c_cons = vars_consumed(c)
                c_prod = var_produced(c)

                # c must not consume a's output
                if a_prod in c_cons:
                    continue

                # c must not consume b's output
                b_prod = var_produced(b)
                if b_prod and b_prod in c_cons:
                    continue

                # c must not consume anything produced between i+1 and j-1
                # (those ops will come AFTER c if we swap)
                skip = False
                for k in range(i + 1, j):
                    kp = var_produced(result[k])
                    if kp and kp in c_cons:
                        skip = True
                        break
                if skip:
                    continue

                # Nothing between i+1 and j must consume c's output
                if c_prod:
                    for k in range(i + 1, j):
                        if c_prod in vars_consumed(result[k]):
                            skip = True
                            break
                if skip:
                    continue

                # Nothing between i+2 and j-1 must consume b's output
                # (b moves from i+1 to j, so anything between needs b defined before)
                if b_prod:
                    for k in range(i + 2, j):
                        if b_prod in vars_consumed(result[k]):
                            skip = True
                            break
                if skip:
                    continue

                result[i + 1], result[j] = result[j], result[i + 1]
                break

    return result


def analyze_pressure(ops):
    """Compute peak live register count."""
    producers = {}
    for i, op in enumerate(ops):
        if op[0] in ('LOAD', 'LOAD_W'):
            producers[op[1]] = i
        elif op[0] != 'STORE' and len(op) > 1:
            producers[op[1]] = i

    last_use = {}
    for i, op in enumerate(ops):
        refs = set()
        if op[0] in ('FMA', 'FNMS', 'FMS', 'FNMA'):
            refs = {v for v in (op[2], op[3], op[4]) if v in producers}
        elif op[0] == 'BINOP':
            refs = {v for v in (op[3], op[4]) if v in producers}
        elif op[0] == 'NEG':
            refs = {op[2]} & set(producers.keys())
        elif op[0] == 'STORE':
            refs = {v for v in producers if re.search(r'\b' + re.escape(v) + r'\b', op[3])}
        for v in refs:
            last_use[v] = i

    live = set()
    peak = 0
    for i, op in enumerate(ops):
        dead = {v for v in live if last_use.get(v, -1) < i}
        live -= dead
        if op[0] in ('LOAD', 'LOAD_W'):
            live.add(op[1])
        elif op[0] != 'STORE' and len(op) > 1:
            live.add(op[1])
        peak = max(peak, len(live))
    return peak


def optimize_and_emit(fftw_src, radix, isa):
    """Full pipeline: parse → schedule → ILP → emit."""
    body = extract_fma_body(fftw_src, radix)
    ops_orig, constants, var_decls = parse_body(body, radix)

    # Pass 1: Sethi-Ullman scheduling
    ops_su = schedule_su(ops_orig)

    # Pass 2: ILP interleaving
    ops_final = ilp_interleave(ops_su)

    # Analyze
    peak_orig = analyze_pressure(ops_orig)
    peak_su = analyze_pressure(ops_su)
    peak_final = analyze_pressure(ops_final)

    print(f"R={radix} {isa}: peak_live orig={peak_orig} → SU={peak_su} → SU+ILP={peak_final} "
          f"(AVX2 spills: {max(0,peak_orig-14)}→{max(0,peak_final-14)})",
          file=sys.stderr)

    # Emit optimized codelet
    lines = emit_codelet(ops_final, constants, var_decls, radix, isa)
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: optimize_codelet.py <radix> <isa> [fftw_src]", file=sys.stderr)
        sys.exit(1)

    radix = int(sys.argv[1])
    isa = sys.argv[2]
    fftw_src = sys.argv[3] if len(sys.argv) > 3 else '/home/claude/fftw-3.3.10'

    code = optimize_and_emit(fftw_src, radix, isa)
    print(code)
