#!/bin/bash
# build_avx2_arms.sh ARMS_DIR OUT_EXE -- compile the arm kernels that
# gen_avx2_arms.sh emitted (the codelet library's own flags) and link
# avx2_tail_arms_bench.c against them. Windows, mingw gcc; run from the Bash tool.
set -eu
ARMS=${1:?usage: build_avx2_arms.sh ARMS_DIR OUT_EXE}
EXE=${2:?usage: build_avx2_arms.sh ARMS_DIR OUT_EXE}
H=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$H/../../../../.." && pwd)
CC=${CC:-/c/mingw152/mingw64/bin/gcc.exe}
KFLAGS="-O3 -mavx2 -mfma -march=native -mno-avx512f -fpermissive -w"
OBJ="$ARMS/obj"
mkdir -p "$OBJ"
ls "$ARMS"/*.c | xargs -P 8 -I{} sh -c 'b=$(basename {} .c); "$0" $1 -c {} -o "$2/$b.o"' "$CC" "$KFLAGS" "$OBJ"
ls "$OBJ"/*.o | cygpath -m -f - > "$OBJ/objs.rsp"   # 210 objects overflow a Windows command line
"$CC" -O2 -mavx2 -mfma -march=native -mno-avx512f -I"$ROOT/gauntlet" -I"$ROOT/src/core/common/support" \
      "$H/avx2_tail_arms_bench.c" @"$OBJ/objs.rsp" -o "$EXE"
echo "built $EXE ($(ls "$OBJ"/*.o | wc -l) kernels)"
