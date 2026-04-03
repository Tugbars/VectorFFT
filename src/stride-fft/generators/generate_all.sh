#!/bin/bash
set -e

GEN_DIR="$(cd "$(dirname "$0")" && pwd)"
SFFT_DIR="$(dirname "$GEN_DIR")"
AVX2_DIR="$SFFT_DIR/codelets/avx2"
AVX512_DIR="$SFFT_DIR/codelets/avx512"
SCALAR_DIR="$SFFT_DIR/codelets/scalar"

export PYTHONIOENCODING=utf-8

mkdir -p "$AVX2_DIR" "$AVX512_DIR" "$SCALAR_DIR"

generate_isa() {
    local ISA=$1
    local DIR=$2
    echo "Generating $ISA codelets..."

    # R=2, R=4, R=8: legacy generators
    for R in 2 4 8; do
        python3 "$GEN_DIR/gen_radix${R}.py" "$ISA" > "$DIR/fft_radix${R}_${ISA}.h" 2>/dev/null
    done

    # R=3..25: ct_n1 + ct_t1_dit + ct_t1_dit_log3
    for R in 3 5 6 7 10 11 12 13 16 17 19 20 25; do
        python3 "$GEN_DIR/gen_radix${R}.py" --isa "$ISA" --variant ct_n1 > "$DIR/fft_radix${R}_${ISA}_ct_n1.h" 2>/dev/null
        python3 "$GEN_DIR/gen_radix${R}.py" --isa "$ISA" --variant ct_t1_dit > "$DIR/fft_radix${R}_${ISA}_ct_t1_dit.h" 2>/dev/null
        python3 "$GEN_DIR/gen_radix${R}.py" --isa "$ISA" --variant ct_t1_dit_log3 > "$DIR/fft_radix${R}_${ISA}_ct_t1_dit_log3.h" 2>/dev/null
    done

    # R=32: ct_n1 + ct_t1_dit (no log3)
    python3 "$GEN_DIR/gen_radix32.py" --isa "$ISA" --variant ct_n1 > "$DIR/fft_radix32_${ISA}_ct_n1.h" 2>/dev/null
    python3 "$GEN_DIR/gen_radix32.py" --isa "$ISA" --variant ct_t1_dit > "$DIR/fft_radix32_${ISA}_ct_t1_dit.h" 2>/dev/null

    # R=64: ct_n1 + ct_t1_dit + ct_t1_dit_log3
    python3 "$GEN_DIR/gen_radix64.py" --isa "$ISA" --variant ct_n1 > "$DIR/fft_radix64_${ISA}_ct_n1.h" 2>/dev/null
    python3 "$GEN_DIR/gen_radix64.py" --isa "$ISA" --variant ct_t1_dit > "$DIR/fft_radix64_${ISA}_ct_t1_dit.h" 2>/dev/null
    python3 "$GEN_DIR/gen_radix64.py" --isa "$ISA" --variant ct_t1_dit_log3 > "$DIR/fft_radix64_${ISA}_ct_t1_dit_log3.h" 2>/dev/null

    echo "  Done. $DIR"
}

generate_isa avx2   "$AVX2_DIR"
generate_isa avx512 "$AVX512_DIR"
generate_isa scalar "$SCALAR_DIR"

echo ""
echo "All codelets generated."
