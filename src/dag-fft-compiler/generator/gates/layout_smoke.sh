#!/bin/bash
# =============================================================================
# layout_smoke.sh — M3's compensating gate (generator_lib_architecture.md §14.2).
#
# The five IL-layout arms have ZERO corpus representatives, so the byte gate
# cannot see them.  This smoke drives every layout shape through the REAL
# emitter, compiles the output with gcc -Werror, and asserts the declared
# pointer parameters are exactly the referenced ones.  It also asserts the
# NEGATIVE space: the previously-silent illegal combinations must now FAIL
# LOUDLY at emission (Layout raises) instead of emitting uncompilable C.
#
# Weaker than byte-identity — stated, not oversold — but strictly stronger
# than before, where `--ip-il-in --ip-il-out` silently produced broken C.
#
# Usage:  bash layout_smoke.sh          (WSL; needs ~/.opam/5.2.0 + gcc)
# Exit:   0 all arms emit+compile and negatives fail loudly; 1 otherwise.
# =============================================================================
set -u
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-/tmp/vfft_layout_smoke}"
rm -rf "$WORK"; mkdir -p "$WORK"

export PATH="$HOME/.opam/5.2.0/bin:$PATH"
export DUNE_CACHE=disabled
( cd "$GEN" && dune build bin/gen_radix.exe ) || { echo "FATAL: build failed"; exit 1; }
EXE="$GEN/_build/default/bin/gen_radix.exe"

PASS=0; FAIL=0

emit_ok() {  # emit_ok <name> <args...> — must emit, gcc-compile, and use all pointers
  local name="$1"; shift
  local c="$WORK/$name.c"
  if ! "$EXE" "$@" > "$c" 2>"$c.err"; then
    echo "FAIL  $name: emission errored: $(head -1 "$c.err")"; FAIL=$((FAIL+1)); return
  fi
  # -Wno-error=incompatible-pointer-types: the in-place spill path stores via
  # &spill_re[i] (__m256d*) where the intrinsic takes double* — a PRE-EXISTING
  # emitted idiom (648 corpus files, byte-identical, compiles as a warning in
  # the production build).  Everything else stays -Werror; an undeclared
  # identifier (the class this smoke exists for) is fatal regardless.
  if ! gcc -c -mavx2 -mfma -Werror -Wall -Wno-error=incompatible-pointer-types \
       -o "$WORK/$name.o" "$c" 2>"$c.gcc"; then
    echo "FAIL  $name: gcc rejected: $(head -1 "$c.gcc")"; FAIL=$((FAIL+1)); return
  fi
  # declared DATA-PLANE pointers == referenced identifiers (unused => (void)'d).
  # tw_re/tw_im are exempt: the historical uniformity convention declares them
  # on no-twiddle kinds without even a (void) (in-place n1 family, shipped).
  local bad=0
  while IFS= read -r ptr; do
    case "$ptr" in tw_re|tw_im) continue ;; esac
    body_uses=$(grep -c "\b$ptr\b" "$c")
    [ "$body_uses" -lt 2 ] && { echo "FAIL  $name: declared pointer $ptr never referenced"; bad=1; }
  done < <(grep -oE '__restrict__ [a-z_0-9]+' "$c" | awk '{print $2}' | sort -u)
  [ $bad -eq 1 ] && { FAIL=$((FAIL+1)); return; }
  echo "ok    $name"; PASS=$((PASS+1))
}

emit_fails_loudly() {  # emission must exit nonzero WITH a message (never broken C)
  local name="$1"; shift
  local c="$WORK/$name.c"
  if "$EXE" "$@" > "$c" 2>"$c.err"; then
    echo "FAIL  $name: illegal combo EMITTED (silent acceptance regressed)"; FAIL=$((FAIL+1)); return
  fi
  if ! grep -qiE "hybrid|mutually exclusive|cannot combine" "$c.err"; then
    echo "FAIL  $name: failed without a layout-law message: $(head -1 "$c.err")"
    FAIL=$((FAIL+1)); return
  fi
  echo "ok    $name (refused loudly)"; PASS=$((PASS+1))
}

echo "== positive arms (every layout shape, incl. the five zero-representative ones) =="
emit_ok ip_plain        16 --in-place --su --isa avx2 --emit-c
emit_ok ip_il_in        16 --in-place --su --ip-il-in --isa avx2 --emit-c
emit_ok ip_il_out       16 --in-place --su --ip-il-out --isa avx2 --emit-c
emit_ok strided_plain   16 --strided --isa avx2 --emit-c
emit_ok strided_il_in   16 --strided --strided-il-in --isa avx2 --emit-c
emit_ok strided_il_out  16 --strided --strided-il-out --isa avx2 --emit-c
emit_ok strided_ilo_nt  16 --strided --strided-il-out-nt --isa avx2 --emit-c
emit_ok strided_r2c     16 --strided-r2c --isa avx2 --emit-c
emit_ok strided_r2c_bwd 16 --strided-r2c --bwd --isa avx2 --emit-c
emit_ok oop_split       16 --oop --oop-buffer-oop --oop-load UG --oop-store UG --isa avx2 --emit-c
emit_ok oop_il_in       16 --oop --oop-buffer-oop --oop-load UG --oop-store UG --oop-il-in --isa avx2 --emit-c
emit_ok oop_il_out      16 --oop --oop-buffer-oop --oop-load UG --oop-store UG --oop-il-out --isa avx2 --emit-c
emit_ok oop_il_both     16 --oop --oop-buffer-oop --oop-load UG --oop-store UG --oop-il-in --oop-il-out --isa avx2 --emit-c

echo "== negative arms (the law must fire LOUDLY at emission) =="
emit_fails_loudly neg_ip_both       16 --in-place --su --ip-il-in --ip-il-out --isa avx2 --emit-c
emit_fails_loudly neg_strided_both  16 --strided --strided-il-in --strided-il-out --isa avx2 --emit-c
emit_fails_loudly neg_il_plus_r2c   16 --strided --strided-il-in --strided-r2c --isa avx2 --emit-c
emit_fails_loudly neg_oop_sw_clash  16 --oop --oop-buffer-oop --oop-load UG --oop-store UG --oop-il-out --oop-il-out-sw --isa avx2 --emit-c

echo
echo "layout_smoke: $PASS ok, $FAIL failed"
[ $FAIL -eq 0 ] && { echo "LAYOUT SMOKE PASS"; exit 0; } || { echo "LAYOUT SMOKE FAIL"; exit 1; }
