# flat/ — the flat mixed-radix DIT for odd N

198 files. ZTURN-T's first stage needs a power-of-two item count, so an odd N
runs one sweep per factor instead; these kinds keep the short last sweeps fast
by putting the loop inside the kernel. Run by `src/core/oop/il_flatdit.h`
under `il_route=flat` (1,002 shipped 1D rows on 2026-09-17); the odd middle
radices live in [`odd_mid/`](odd_mid/).

| kind | what it is |
|---|---|
| `t2cp` (+ `_bwd`) | the per-column-pair stage of a long run |
| `t2cs` | the per-pair LOADED twiddle stream for a short run; reachable only through the `VFFT_ILFD_NO_GEN2` pin today, kept as the loaded form a highest-accuracy build configuration would take (the owner's ruling of 2026-09-24: no accuracy arms, no accuracy modes; a build switch, maybe, later) |
| `t2csg`, `t2csgt_bwd` | the GENERATED stream: one record per pair, one per group, the other legs derived in-kernel (`gen2`). This derivation is what sets the flat route's worst-length accuracy (README, Accuracy) |
| `t2csgn`, `t2csgnt_bwd` | `t2csg` with the last stage's group loop inside the kernel |

Per-stage forms are raced and banked per cell (`msz` for a middle stage, the
tail kinds above for a short run).

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../README.md`](../README.md).
