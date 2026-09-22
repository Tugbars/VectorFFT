# The refactor-safety baseline

The tools that make a code restructuring provable: capture byte-diffable
artifacts of the library's behaviour before a change, capture them again after,
diff. No timing is ever part of an artifact. Procedure and stop rules:
[docs/design/refactor_safety_harness.md](../../../docs/design/refactor_safety_harness.md);
the reference artifacts and their rules: [reference/README.md](reference/README.md).

```
python src/tools/baseline/capture_baseline.py --out <scratch> [--repeat 3]   # capture the two artifacts
python src/tools/baseline/capture_baseline.py --out src/tools/baseline/reference --repeat 5   # re-stamp the reference
python src/tools/baseline/slice_ladder.py --parent <fn> --helper <fn> ...    # one migration step's ladder of checks
python src/tools/baseline/obj_equiv.py before.o after.o                      # the object-code equivalence proof
python src/tools/baseline/sym_census.py <obj> --defined|--undefined|--mutable
python src/tools/baseline/race_census.py [src/core/vfft.c ...]
python src/tools/baseline/trig_capture.py --out FILE [--repeat 3]
```

## Files

- `capture_baseline.py` -- `golden_bits.txt` (refusal decisions + output-bit digests) and
  `fp_replay.txt`, one process per cell, a fresh seeded store per cell in scratch, LF always,
  repeated so a nondeterministic cell is recorded as such instead of sampled once.
- `harness_golden.c`, `fp_sweep.c`, `trig_digest_probe.c` -- the harness programs; built by
  `gauntlet/build.py` (`--vfft --compile`), binaries land beside them (ignored).
- `slice_ladder.py` -- drives `obj_equiv.py`, `sym_census.py`, `race_census.py` and the
  capture for one migration step.
- `reference/` -- the committed reference artifacts (`*.txt` ride an explicit `.gitignore`
  negation and an `eol=lf` pin in `.gitattributes`; `vfft_baseline.o` is pinned binary).

Moved here from `build_tuned/` on 2026-09-22; every path inside was repointed.
