"""make_cells.py -- the timing cells of the rank-4 to 7 complex transforms: 4d-cells.txt .. 7d-cells.txt beside this file.

A fixed list, written from arithmetic alone, so every machine times the same shapes. Each rank has cells of four
classes -- every extent a power of two; extents 2^a x odd; odd extents; a prime above 47 on one axis (no kernel of
its own) -- from a few KiB to the largest array a rank's cube reaches under 2^24 points, as cubes and as shapes with
one long or one short axis (a short first axis leaves the threads few blocks; a long last axis is the row pass).

    python tests/rank_n/make_cells.py

The gauntlet reads one file a run (a run is one rank):
    python gauntlet/gauntlet.py run --cells @tests/rank_n/4d-cells.txt --name <run> --bin-dir build/gauntlet --est
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = {
    4: [
        ("power of two", ["4x4x4x4", "8x8x8x8", "16x16x16x16", "32x32x32x32", "64x64x64x64",
                          "16x16x16x32", "32x32x32x64", "64x16x16x16", "16x16x16x64", "256x8x8x8", "8x8x8x256",
                          "2x64x64x64", "4x32x32x32", "64x64x64x4", "128x128x8x8", "8x8x128x128"]),
        ("2^a x odd", ["12x12x12x12", "24x24x24x24", "48x48x48x48", "20x20x20x20", "10x12x14x16", "40x30x20x10",
                       "96x24x12x6", "6x12x24x96"]),
        ("odd", ["9x9x9x9", "15x15x15x15", "25x25x25x25", "27x9x9x9", "5x7x9x11", "35x15x9x5", "45x45x15x15"]),
        ("prime > 47", ["53x16x16x16", "16x53x16x16", "16x16x16x53", "101x8x8x8", "127x12x12x12"]),
    ],
    5: [
        ("power of two", ["4x4x4x4x4", "8x8x8x8x8", "16x16x16x16x16", "32x16x16x16x16", "2x4x8x16x32", "32x16x8x4x2",
                          "16x16x16x8x8", "4x4x4x4x256"]),
        ("2^a x odd", ["6x6x6x6x6", "12x12x12x12x12", "10x10x10x10x10", "20x16x12x8x4", "24x12x6x12x24"]),
        ("odd", ["5x5x5x5x5", "9x9x9x9x9", "15x9x7x5x3", "3x5x7x9x11", "15x15x15x15x15"]),
        ("prime > 47", ["53x8x8x8x8", "8x8x53x8x8", "8x8x8x8x53"]),
    ],
    6: [
        ("power of two", ["4x4x4x4x4x4", "8x8x8x8x8x8", "16x16x16x16x16x16", "2x4x2x4x2x4", "4x4x4x8x8x8",
                          "8x8x8x4x4x4", "16x4x4x4x4x16"]),
        ("2^a x odd", ["6x6x6x6x6x6", "10x10x10x10x10x10", "12x10x8x6x4x2", "2x4x6x8x10x12"]),
        ("odd", ["3x3x3x3x3x3", "5x5x5x5x5x5", "9x9x9x9x9x9", "3x5x7x7x5x3"]),
        ("prime > 47", ["53x4x4x4x4x4", "4x4x4x4x4x53"]),
    ],
    7: [
        ("power of two", ["2x2x2x2x2x2x2", "4x4x4x4x4x4x4", "8x8x8x8x8x8x8", "4x4x4x4x8x8x8", "8x8x8x8x8x8x4",
                          "2x4x8x2x4x8x2"]),
        ("2^a x odd", ["6x6x6x6x6x6x6", "2x3x4x5x6x7x8", "10x8x6x4x6x8x10"]),
        ("odd", ["3x3x3x3x3x3x3", "5x5x5x5x5x5x5", "3x5x7x3x5x7x3"]),
        ("prime > 47", ["53x2x2x2x2x2x2", "2x2x2x2x2x2x53"]),
    ],
}


def points(sh):
    p = 1
    for t in sh.split("x"):
        p *= int(t)
    return p


def main():
    for rank, groups in sorted(CELLS.items()):
        path = os.path.join(HERE, "%dd-cells.txt" % rank)
        n = sum(len(g) for _, g in groups)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# %dd-cells.txt -- %d shapes of rank %d, the timing cells of the rank-4 to 7 complex transforms\n" % (rank, n, rank))
            f.write("# Written by make_cells.py from arithmetic alone; every platform times the same cells.\n")
            f.write("# The gauntlet reads it: python gauntlet/gauntlet.py run --cells @tests/rank_n/%dd-cells.txt --est\n" % rank)
            f.write("# %-28s # %-14s points\n" % ("shape", "class"))
            for cls, shapes in groups:
                for sh in shapes:
                    assert sh.count("x") == rank - 1 and points(sh) <= 1 << 24, sh
                    f.write("%-30s # %-14s %d\n" % (sh, cls, points(sh)))
        print("wrote %s: %d shapes" % (path, n))


if __name__ == "__main__":
    main()
