# P3 metal kit: native rfft vs MKL vs FFTW

Build: `bash benchmarks/p3_metal/build.sh` (repo root). Run:
`PERF=1 CPU=2 bash benchmarks/p3_metal/run.sh`.

Methodology: pinned core, performance governor, 64MB cache-bust
between trials, median-of-min over 5 trials x REPS, SAME-RUN ratios
only. Every vfft lane self-validates against FFTW output before
timing (err column must read OK).

What the metal session must adjudicate (container verdicts in
docs/context_wall_plan.md and sections 64-70 — several may invert):
1. RANGED (T1, FFTW-structural): base vs ranged variants.
2. PREFETCH (E1): measured NEGATIVE on the dev container.
3. Lane blocking: VFFT_KB env (e.g. 32/64/96); container NEGATIVE.
4. Factorization sweep is built in; the winner is machine-dependent.
5. The MKL gap (~56us container, noisy): real number wanted.
TODO lane: the half-complex wrapper (production r2c API) — link the
main library objects and add an f_wrapper; see core/r2c.h.
T2 (dobatch copy-to-contiguous) is DESIGNED, NOT BUILT — see
docs/t2_dobatch_design_note.md before implementing.
