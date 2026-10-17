# A sample chapter

This chapter is not part of the guide. It uses each construct of `STYLE.md` once, so that you can see the source of each beside what the build makes of it: a listing and what it printed, two equations, a figure, a table, cross-references, a note, steps and an excerpt of an example program.

(sample-sec-pair)=
## Compute a transform and its inverse

For a sequence $x[n]$ of $N$ complex samples, a plan of `VFFT_C2C` executed with `VFFT_FORWARD` computes

$$
%lead: Forward:
X[k] = \sum_{n=0}^{N-1} x[n]\, W_N^{kn}, \qquad 0 \le k \le N-1,
$$ (sample-eq-forward)

where $W_N = e^{-j2\pi/N}$. Executed with `VFFT_BACKWARD`, the same plan computes

$$
%lead: Backward:
\sum_{k=0}^{N-1} X[k]\, W_N^{-kn} = N\, x[n], \qquad 0 \le n \le N-1,
$$ (sample-eq-backward)

so a forward transform followed by a backward one returns the input times $N$. You divide by $N$ yourself, best in a pass your program makes anyway.

The program below creates a plan for $N = 64$, runs Eq. {eq}`sample-eq-forward` and then Eq. {eq}`sample-eq-backward`, and checks that the result is $N$ times the input:

:::{listing} sample_roundtrip.c
:::

It prints one line, {output}`sample_roundtrip`. The same output as a block, the form for anything longer than one line:

:::{output} sample_roundtrip
:::

What the program wrote to its standard error is kept as well, and embeds the same way:

:::{output} sample_roundtrip
:stream: stderr
:::

A section that discusses one step shows that step only. This is the region `create` of the same file:

:::{listing} sample_roundtrip.c
:region: create
:::

(sample-sec-bins)=
## Find a frequency in the half spectrum

The spectrum of a real sequence has $X[k] = X^*[N-k]$, so a `VFFT_R2C` plan stores only the bins $0 \le k \le N/2$. {numref}`sample-fig-bins` shows the frequency each bin holds.

:::{figure} sample_half_spectrum.svg
:name: sample-fig-bins
:alt: A row of eight cells. The first five hold X[0] to X[4] and are stored; the last three, shaded, hold the conjugates of X[3], X[2] and X[1] and are not stored. Under the cells, bins 0 to 7 and their frequencies: 0, 1, 2, 3, plus or minus 4, minus 3, minus 2, minus 1.
The bins of the DFT of a real sequence of $N = 8$ samples. A `VFFT_R2C` plan stores bins $0$ to $N/2$. Bin $k \le N/2$ holds frequency $k$, and bin $k > N/2$ holds frequency $k - N$.
:::

{numref}`sample-tbl-sizes` gives the sizes of the buffers in doubles; `vfft_info()` reports the same numbers as `in_doubles` and `out_doubles`.

:::{table} The buffers of a 1D plan of $N$ points, in doubles.
:name: sample-tbl-sizes
| Transform | Input | Output |
|---|---|---|
| `VFFT_C2C` | $2N$ | $2N$ |
| `VFFT_R2C` | $N$ | $2(N/2 + 1)$ |
| `VFFT_C2R` | $2(N/2 + 1)$ | $N$ |
:::

:::{note}
Read the sizes from `vfft_info()` instead of computing them, as the program of {ref}`sample-sec-pair` does.
:::

## Quote an example program

A worked example quotes its program where the text needs it. This excerpt of `poisson3d.c` runs the R2C plan, divides in the half spectrum and runs the C2R plan; the line `...` stands for the lines left out:

```c excerpt=poisson3d.c
        vfft_execute(r2c, VFFT_FORWARD, f, F);
...
        divide_by_laplacian(F);
        vfft_execute(c2r, VFFT_BACKWARD, F, phi);
```

The harness runs the example as it is, so what it printed embeds the same way as a listing's output:

:::{output} poisson3d
:::

To build and run the listings of the guide yourself:

:::{steps}
1. Install VectorFFT, and note the folder you installed it to, `<prefix>`.
2. Configure the listings against it: `cmake -S listings -B build -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH=<prefix>`.
3. Build them with `cmake --build build`, and run one:
:::

```sh
./build/sample_roundtrip
```
