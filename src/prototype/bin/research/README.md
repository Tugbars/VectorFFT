# bin/research/

Diagnostic and investigation tools used during research arcs.
Not part of the codelet generation pipeline; kept for reference
if a question reopens.

## profile_pipeline.ml

Per-pass timing of the algsimp pipeline at varying R. Locates
which pass dominates end-to-end generation time. Used in
[doc 32](../../docs/32_of_expr_memo.md) to identify that
`of_assignments` was the O(N^4) bottleneck before the memo fix.

Run: `dune exec bin/research/profile_pipeline.exe -- 64`

## sr_diag.ml

Textual vs unique post-hashcons node count diagnostic. Compares
the structural sizes of CT and split-radix DAGs at a given R,
showing the redundancy ratio that motivated investigating SR
in the first place. Used in [doc 31](../../docs/31_split_radix_research_arc.md).

Run: `dune exec bin/research/sr_diag.exe -- 64`

## sr_structural_diff.ml

Canonical S-expression fingerprint diff between CT and SR output
DAGs at a given R. Reports what fraction of output bins are
structurally identical between the two algorithms after algsimp
normalization. Used in doc 31's investigation of whether CT and
SR converge after algsimp (they don't — most bins differ
structurally even when op counts converge).

Run: `dune exec bin/research/sr_structural_diff.exe -- 64`

## sr_union_probe.ml

Builds the union DAG of CT and SR outputs at a given R and
reports how much sharing exists. Probes whether a hypothetical
"cooperative" generator that ran both algorithms and shared
subexpressions could beat either alone. Used in doc 31 — answer
was that the union is 22-42% smaller than naive sum but always
larger than either alone, so true cooperation would require
e-graph saturation (not pursued).

Run: `dune exec bin/research/sr_union_probe.exe -- 64`
