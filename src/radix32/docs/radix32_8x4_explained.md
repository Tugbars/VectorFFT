# Radix-32 (8×4) FFT Structure Explained

## 1️⃣ What "Pass 1" and "Pass 2" Mean

The radix-32 FFT is decomposed as **32 = 8 × 4**.

-   **Pass 1 (Radix-8)**: First stage, processes 4 groups × K
    butterflies.\
    Performs 8-point FFTs on each group.\
    Access pattern: sequential within group.

-   **Pass 2 (Radix-4)**: Second stage, processes 8 positions × K
    butterflies.\
    Performs 4-point FFTs using partial results from Pass 1.\
    Needs transpose for sequential access.

So:\
- Pass 1 = "Radix-8 stage"\
- Pass 2 = "Radix-4 stage"

------------------------------------------------------------------------

## 2️⃣ What "P0, P1, ..." Mean in Pass 2

Each **P (position)** corresponds to one of the 8 sub-FFTs of size 4
inside the 32-point block after Pass 1.

For each position **P = 0..7**, you run a radix-4 butterfly across the
four elements that belong to that position.

### Examples

    P0 reads: [0,1,2,3]
    P1 reads: [4,5,6,7]
    P2 reads: [8,9,10,11]
    ...

**Twiddles used per position:** - P0: identity (no cmul needed)\
- P1--P7: twiddled (use W₃₂\^{8m}, W₃₂\^{16m}, W₃₂\^{24m})

------------------------------------------------------------------------

## 3️⃣ Visual Layout

    Input
      │
      ▼
    Pass 1 (Radix-8 stage) → 4 groups × K
      │
      ▼
    Transpose (4×K → K×4)
      │
      ▼
    Pass 2 (Radix-4 stage) → 8 positions × K
      │
      ▼
    Output

------------------------------------------------------------------------

## 4️⃣ Summary

-   Pass 1: Radix-8 stage (4 groups × K)\
-   Pass 2: Radix-4 stage (8 positions × K)\
-   P0--P7: Positions within Pass 2 (P0 has identity twiddle)\
-   Transpose: Bridges the two stages for contiguous memory access

**In short:**\
- Pass 1 = Stage 1 of FFT (Radix-8)\
- Pass 2 = Stage 2 of FFT (Radix-4)\
- P0 = Position 0 in Pass 2 (no twiddle multiply)
