#!/bin/bash
# Regenerate IN PLACE every codelet in the tree that codelet_cil.ml owns, so
# the odd-COUNT tail reaches the shipping binary. Files emitted by the OLD
# codelet_zil.ml are left ALONE (different emitter, not touched by this work).
# The file SET is derived from what is already in the tree, so nothing is
# added or dropped -- only regenerated.
GEN=/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator
TREE=/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/codelets/zil/avx2
cd "$GEN" || exit 1
EXE=./_build/default/bin/gen_radix.exe

# blocked splits for the odd composites (genset.sh's table); pow2 uses the
# emitter's own built-in rule and must NOT be given a split.
split_for () { case "$1" in
  9) echo 3.3;; 15) echo 3.5;; 21) echo 3.7;; 25) echo 5.5;;
  27) echo 3.9;; 45) echo 5.9;; 49) echo 7.7;; *) echo "";; esac; }

ok=0; bad=0; skipped=0
for f in "$TREE"/*.c; do
  base=$(basename "$f" .c)
  head -2 "$f" | grep -q "codelet_cil.ml" || { skipped=$((skipped+1)); continue; }

  # radix{R}_z_{kindtag}[_log3][_bwd]_avx2
  R=${base#radix}; R=${R%%_*}
  rest=${base#radix${R}_z_}; rest=${rest%_avx2}
  dflag=""; case "$rest" in *_bwd) dflag="--cil-bwd"; rest=${rest%_bwd};; esac
  l3="";    case "$rest" in *_log3) l3="--cil-log3";  rest=${rest%_log3};; esac
  blk="";   case "$rest" in n1b|t2b) blk="--cil-blocked"; rest=${rest%b};; esac
  sp=""
  if [ -n "$blk" ]; then s=$(split_for "$R"); [ -n "$s" ] && sp="--cil-split $s"; fi

  if $EXE "$R" --cil-"$rest" --emit-c $dflag $blk $sp $l3 \
        --isa avx2 --uarch raptor_lake_avx2 > "$f.tmp" 2>"$TREE/.e"; then
    mv "$f.tmp" "$f"; ok=$((ok+1))
  else
    bad=$((bad+1)); rm -f "$f.tmp"
    echo "  FAIL $base (R=$R kind=$rest $blk $sp $l3 $dflag): $(head -c 140 "$TREE/.e" | tr '\n' ' ')"
  fi
done
rm -f "$TREE/.e"
echo "regenerated OK=$ok  FAILED=$bad  left-alone(zil)=$skipped"
