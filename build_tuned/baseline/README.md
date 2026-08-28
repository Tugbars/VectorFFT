# Refactor baseline artifacts

The reference state the `vfft.c` migration diffs against. Captured at the SHA in
`baseline_sha.txt`, **before any step of the migration**.

Procedure and stop rules: [docs/design/refactor_safety_harness.md](../../docs/design/refactor_safety_harness.md).
Step list: [docs/design/refactor_migration_plan.md](../../docs/design/refactor_migration_plan.md).

## Rules

- **Compare only within one flags key.** Every artifact here is tied to the compiler,
  flags and host in `toolchain_flags.txt`. `-march=native` makes them host-bound; a
  cross-host or cross-flag comparison is out of contract and means nothing.
- **Never diff a nanosecond.** No artifact in this directory contains a timing. The one
  clock-bearing check (the performance leg) compares against the archived *binary*
  re-run in the same session, not against a number stored here.
- **Truncate, never append.** A re-captured artifact replaces its predecessor; an
  appended one silently stops being a baseline.
- These files are committed on purpose. `.gitignore:85` ignores `*.txt`, so they ride an
  explicit negation, and `.gitattributes` pins `eol=lf` — a CRLF conversion would make
  every line differ.

## Artifacts

| file | what it is | re-derive with |
|---|---|---|
| `baseline_sha.txt` | the commit, capture time, and the working tree state at capture | `git rev-parse HEAD` |
| `toolchain_flags.txt` | compiler version, path, host, identity flags | `gcc --version` |
| `include_flags.txt` | the exact `-I` set used for the identity object | see below |
| `wisdom_store.sha256` | per-file hash of all 11 committed wisdom files + the cell count | `sha256sum` over `generated/*.txt` |
| `vfft_baseline.o` | the identity object, fingerprint **off** | command below |
| `nm_defined.txt` | defined symbols, sorted — catches a function orphaned, duplicated or newly external | `nm --defined-only` |
| `nm_undefined.txt` | undefined symbols, sorted — catches a new external dependency | `nm -u` |
| `mutable_objects.txt` | file-scope mutable objects (`b/B/d/D`), sorted | `nm` filtered |
| `gates_build.txt` | build result for all 32 gates at this SHA | `build_all_gates.sh` |
| `build_all_gates.sh` | rebuilds every gate — **the prebuilt `.exe` files all predate `vfft.c`** | — |

Not yet captured, and why: `golden_bits.txt`, `refusal_matrix.txt` and `accuracy_ref.txt`
need `harness_golden.c` (plan step 3); `fp_replay.txt` and the engagement deltas need the
fingerprint emitter and counters (plan step 4). Those are edits, and an edit is not step 1.

## Re-deriving the identity object

```
gcc -c -O2 -mavx2 -mfma $(cat include_flags.txt) ../../src/core/vfft.c -o vfft_new.o
python ../obj_equiv.py vfft_baseline.o vfft_new.o
```

`obj_equiv.py` proves every emitted symbol body is unchanged. It is **measured blind to
`.rdata`** — a `0.97 → 0.96` hysteresis change passes it as EQUIVALENT — so it covers code
only, and the race protocol census covers the constants. Neither alone is sufficient.

## Note on `mutable_objects.txt`

It lists 42 entries, including compiler-mangled function-local statics (`_ord_n.6`,
`_ord_pick.5`). That is deliberate: a `static` inside a function body in a header is one
copy **per includer** too, and a source-level grep for file-scope statics would miss it.
The `.bss` section entry is stable noise and diffs identically.
