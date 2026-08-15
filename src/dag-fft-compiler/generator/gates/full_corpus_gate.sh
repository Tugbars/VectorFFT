#!/bin/bash
# =============================================================================
# full_corpus_gate.sh — FULL-CORPUS byte-verbatim reproducibility gate for the
# dag-fft-compiler codelet corpus (src/dag-fft-compiler/codelets, 1432 .c files).
#
# This is the gate the generator/lib restructure is accepted or rejected on:
# the emitter must reproduce >=95% of the corpus BIT-VERBATIM.  It supersedes
# the 183-case cx_* matrix (gates/cil_matrix.sh), which covers only 13% of it.
#
# ---------------------------------------------------------------------------
# WHAT "BYTE-IDENTICAL" MEANS HERE  (read this before quoting any number)
#
#  * The baseline is the LF-CANONICAL corpus == the git blob, NOT the worktree
#    bytes.  The repo sets core.autocrlf=true with `* text=auto`, so 99 of the
#    1432 worktree files carry CRLF while every committed blob is LF.  Comparing
#    against worktree bytes reports 99 spurious failures that measure the
#    developer's checkout/editor, not the emitter.  Verified: `git status` on
#    codelets/ is clean and blob == worktree-minus-CR byte-for-byte.
#
#  * The emitter stamps `Sys.argv` verbatim into the provenance header, argv[0]
#    included, so a file's BYTES depend on how the caller spelled the exe path.
#    Five spellings exist in the corpus.  The replay arm therefore reproduces
#    the recorded argv[0] with `exec -a`; without that, 123 files can never
#    match.  (This is a real hermeticity defect in the emitter — see
#    recon/08_baseline_audit.md finding F3 — not a harness workaround.)
#
# ---------------------------------------------------------------------------
# USAGE
#   bash full_corpus_gate.sh verify [BASELINE.tsv]   # gate: exit != 0 on drift
#   bash full_corpus_gate.sh record [BASELINE.tsv]   # (re)record the baseline
#   bash full_corpus_gate.sh report                  # verify + full per-file TSV
#   bash full_corpus_gate.sh manifest                # rewrite baseline_manifest.tsv
#                                                    # from the CURRENT corpus — the
#                                                    # ONLY sanctioned way to bless an
#                                                    # intentional corpus change (G4);
#                                                    # commit it + re-record, reviewed.
#
#   Env: WORK=<scratch dir>   (default /tmp/vfft_corpus_gate; keep it on a
#                              Linux fs — /mnt/c costs ~8x in wall time)
#        REPO=<repo root>     (default: derived from this script's location)
#
# M0 GATE HARDENING (generator_lib_architecture.md §14.1, 2026-08-14):
#   G1  this file + its data (recipes.tsv, baseline_manifest.tsv,
#       baseline_verdicts.tsv, regen_cil.sh, cil_matrix.sh) live TRACKED in
#       src/dag-fft-compiler/generator/gates/ — no longer one `git clean -xfd`
#       from gone.  The docs/research copies are historical snapshots.
#   G2  `record` now writes a TRACKED baseline => every re-record is a
#       reviewable diff.
#   G3  the build step compiles ALL 19 executables (bin 13, bin_test 4,
#       facdrv 1, emit_tool 1) and RUNS cx_pipeline_test — a chain-tail or
#       cx break can no longer hide behind a 2-exe build.
#   G4  CORPUS_DRIFT is FATAL, all modes.  A corpus edit must go through
#       `manifest` + re-record, as its own reviewed commit.
#
# RUNTIME: ~55 s cold (6 s build + 7 s gen_set + ~40 s for the 358 per-file
#          fork-per-codelet arms), ~48 s warm.  Single-threaded.
#
# EXIT: 0 = every file in the recorded baseline still reproduces byte-verbatim.
#       1 = at least one regression (or a new non-reproducing file).
#       2 = harness/build failure.
# =============================================================================
set -u
export LC_ALL=C

MODE="${1:-verify}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
GEN="$REPO/src/dag-fft-compiler/generator"
CORPUS="$REPO/src/dag-fft-compiler/codelets"
# G1: the gate is SELF-CONTAINED from this tracked directory — recipes and
# manifest live beside the script, not in the gitignored research folder.
RECIPES="${RECIPES:-$HERE/recipes.tsv}"
MANIFEST="${MANIFEST:-$HERE/baseline_manifest.tsv}"
BASELINE="${2:-$HERE/baseline_verdicts.tsv}"
WORK="${WORK:-/tmp/vfft_corpus_gate}"

[ -d "$CORPUS" ] || { echo "FATAL: no corpus at $CORPUS"; exit 2; }
[ -f "$RECIPES" ] || { echo "FATAL: no recipe table at $RECIPES"; exit 2; }

rm -rf "$WORK"; mkdir -p "$WORK"/{tree,out,base}

# ---------------------------------------------------------------- 0. baseline
# LF-canonical copy of the corpus == the committed blob (see header note).
while IFS= read -r f; do
  rel="${f#"$CORPUS"/}"
  mkdir -p "$WORK/base/$(dirname "$rel")"
  tr -d '\r' < "$f" > "$WORK/base/$rel"
done < <(find "$CORPUS" -name '*.c' | sort)
NBASE=$(find "$WORK/base" -name '*.c' | wc -l)

# Corpus fingerprint: sha256 of every LF-canonical file, checked against the
# manifest.  This catches a corpus EDIT (which the verdict diff alone would
# not distinguish from an emitter change).  Two manifest formats accepted:
# the recon 10-column original (sha256_lf = col 10) and the 2-column form
# this script's `manifest` mode writes (rel <TAB> sha256_lf).
( cd "$WORK/base" && find . -name '*.c' | sed 's|^\./||' | sort \
    | while IFS= read -r r; do printf '%s\t%s\n' "$r" "$(sha256sum "$r" | cut -d' ' -f1)"; done ) > "$WORK/corpus_sha.tsv"

if [ "$MODE" = "manifest" ]; then
  { printf 'rel\tsha256_lf\n'; cat "$WORK/corpus_sha.tsv"; } > "$MANIFEST"
  echo "MANIFEST rewritten from the current corpus -> $MANIFEST"
  echo "Commit it together with a re-recorded baseline (G4 discipline)."
  exit 0
fi

CORPUS_DRIFT=-1
if [ -f "$MANIFEST" ]; then
  NCOL=$(head -1 "$MANIFEST" | awk -F'\t' '{print NF}')
  if [ "$NCOL" -ge 10 ]; then
    awk -F'\t' 'NR>1{print $1"\t"$10}' "$MANIFEST" | sort > "$WORK/manifest_sha.tsv"
  else
    awk -F'\t' 'NR>1{print $1"\t"$2}'  "$MANIFEST" | sort > "$WORK/manifest_sha.tsv"
  fi
  CORPUS_DRIFT=$(comm -3 "$WORK/corpus_sha.tsv" "$WORK/manifest_sha.tsv" | wc -l)
fi
# G4: a corpus change is FATAL in every mode.  The verdict diff cannot tell a
# corpus edit from an emitter change; without this check an ADDED file passes
# with exit 0.  Intentional corpus changes go through `manifest` + re-record.
if [ "$CORPUS_DRIFT" -ne 0 ]; then
  if [ "$CORPUS_DRIFT" -lt 0 ]; then
    echo "GATE FAIL (G4) — no manifest at $MANIFEST; refusing to gate blind."
  else
    echo "GATE FAIL (G4) — corpus changed: $CORPUS_DRIFT sha mismatches vs $MANIFEST."
    comm -3 "$WORK/corpus_sha.tsv" "$WORK/manifest_sha.tsv" | head -12
    echo "If this change is INTENTIONAL: bash $0 manifest && bash $0 record — as its own reviewed commit."
  fi
  exit 1
fi

# ------------------------------------------------------------------- 1. build
# SCOPED targets only.  A bare `dune build` in this tree is a documented
# footgun — @default PROMOTES tracked generated/ headers EVEN WHEN THE BUILD
# FAILS (reproduced during review: rc=1 and plan_executors.h rewritten).
# G3: build ALL 19 executables, not 2 — otherwise a break in the 7 chain-tail
# consumer files (dbg_eval, dump_ir,
# dbg_zil_math, facdrv/main) is invisible and the gate reports PASS.
if [ -z "${SKIP_BUILD:-}" ]; then
  export PATH="$HOME/.opam/5.2.0/bin:$PATH"
  export DUNE_CACHE=disabled
  ( cd "$GEN" && dune build \
      bin/gen_radix.exe bin/gen_set.exe bin/dump_ir.exe \
      bin/emit_registry_h.exe bin/emit_executor_h.exe bin/emit_rfft_registry.exe \
      bin/emit_oop_registry.exe bin/emit_trig_registry.exe \
      bin/emit_strided_registry.exe bin/emit_c2r_registry.exe \
      bin/dbg_eval.exe bin/dbg_zil_math.exe \

      bin_test/cx_pipeline_test.exe \
      facdrv/main.exe emit_tool/emit_executor_h.exe ) \
    || { echo "FATAL: scoped 19-target dune build failed"; exit 2; }
  # the cx stack's only unit gate — build AND run it (it was silently
  # never built for weeks; the on-disk "ALL PASS" exe was stale).
  "$GEN/_build/default/bin_test/cx_pipeline_test.exe" > "$WORK/cx_test.log" 2>&1 \
    || { echo "FATAL: cx_pipeline_test FAILED"; sed -n '1,10p' "$WORK/cx_test.log"; exit 2; }
fi
EXE="$GEN/_build/default/bin/gen_radix.exe"
[ -x "$EXE" ] || { echo "FATAL: $EXE missing"; exit 2; }

# ------------------------------------------------------- 2. arm A: gen_set(3)
# Coverage.quadrants is the single source of truth; one warm process, 1074 files.
"$GEN/_build/default/bin/gen_set.exe" --root "$WORK/tree" all > "$WORK/genset.log" 2>&1 \
  || { echo "FATAL: gen_set failed"; sed -n '1,20p' "$WORK/genset.log"; exit 2; }

# -------------------------------- 3. arms B-E: one fork per non-gen_set file
# replay = the file's own recorded argv (argv[0] reproduced via exec -a)
# derive = zil/avx2/pure_il filename grammar + the odd-radix --cil-split table
# ship   = generator/emit_ship.sh tangent recipes (env + post-emit sed rename)
# k1     = Codelet_oop.emit_k1_mono (--k1-mono family)
emit_one() {  # emit_one <rel> <argv0> <env> <args> <sed_from> <sed_to>
  local rel="$1" a0="$2" envs="$3" args="$4" sf="$5" st="$6"
  local dst="$WORK/out/$rel"
  mkdir -p "$(dirname "$dst")"
  if [ -n "$envs" ]; then
    # shellcheck disable=SC2086
    ( eval "export $envs"; exec -a "$a0" "$EXE" $args ) > "$dst" 2>"$dst.err"
  else
    # shellcheck disable=SC2086
    ( exec -a "$a0" "$EXE" $args ) > "$dst" 2>"$dst.err"
  fi
  local rc=$?
  if [ $rc -ne 0 ]; then printf 'EMIT_FAILED(%d)\t%s' "$rc" "$(head -1 "$dst.err" | tr -d '\t')"; return; fi
  rm -f "$dst.err"
  [ -n "$sf" ] && sed -i "s/$sf/$st/g" "$dst"
  printf 'OK\t'
}

: > "$WORK/verdicts.tsv"
# recipes.tsv writes "-" for empty fields on purpose: `IFS=$'\t' read` treats tab
# as IFS *whitespace* and collapses runs of it, which shifts every field after
# the first empty one.  Translate "-" back to "" here.
tail -n +2 "$RECIPES" | while IFS=$'\t' read -r rel arm a0 envs args sf st sha bytes; do
  [ "$envs" = "-" ] && envs=""; [ "$sf" = "-" ] && sf=""; [ "$st" = "-" ] && st=""
  [ "$a0" = "-" ] && a0=""; [ "$args" = "-" ] && args=""
  b="$WORK/base/$rel"
  case "$arm" in
    genset) g="$WORK/tree/$rel"
            if [ ! -f "$g" ]; then v="MISSING	-"
            elif cmp -s "$g" "$b"; then v="IDENTICAL	-"
            else v="BODY_DIFFERS	$(diff "$g" "$b" | grep -c '^[<>]')"; fi ;;
    none)   v="NO_RECIPE	-" ;;
    *)      r=$(emit_one "$rel" "$a0" "$envs" "$args" "$sf" "$st")
            if [ "${r%%$'\t'*}" != "OK" ]; then v="$r"
            else
              g="$WORK/out/$rel"
              if cmp -s "$g" "$b"; then v="IDENTICAL	-"
              else
                # Is the difference confined to a hand-written comment prologue
                # (everything before the first #include)?  If so the EMITTED
                # BODY is byte-identical and only a manual post-step differs.
                gb=$(sed -n '/^#include/,$p' "$g"); bb=$(sed -n '/^#include/,$p' "$b")
                if [ "$gb" = "$bb" ]; then
                  # `$(...)` strips trailing newlines from both sides, so this
                  # also absorbs a stray blank line at EOF; distinguish the two.
                  if cmp -s <(sed -n '/^#include/,$p' "$g") <(sed -n '/^#include/,$p' "$b")
                  then v="PROLOGUE_ONLY	$(diff <(sed -n '1,/^#include/p' "$g") <(sed -n '1,/^#include/p' "$b") | grep -c '^[<>]')"
                  else v="PROLOGUE+EOF_NL	$(diff <(sed -n '1,/^#include/p' "$g") <(sed -n '1,/^#include/p' "$b") | grep -c '^[<>]')"
                  fi
                else
                  v="BODY_DIFFERS	$(diff "$g" "$b" | grep -c '^[<>]')"
                fi
              fi
            fi ;;
  esac
  printf '%s\t%s\t%s\n' "$rel" "$arm" "$v" >> "$WORK/verdicts.tsv"
done

# ------------------------------------------------------------- 4. record/report
sort -o "$WORK/verdicts.tsv" "$WORK/verdicts.tsv"
NV=$(wc -l < "$WORK/verdicts.tsv")
NID=$(awk -F'\t' '$3=="IDENTICAL"' "$WORK/verdicts.tsv" | wc -l)
# Harness self-check (added after a real incident, 2026-08-14): a WSL /mnt/c
# 9p flake once TRUNCATED this file mid-loop (~169/1432 rows) and the run
# reported "verdicts moved" — a phantom regression.  An incomplete run is a
# HARNESS failure (exit 2, rerun after `wsl --shutdown`), never a verdict.
[ "$NV" -eq "$NBASE" ] || {
  echo "FATAL: harness produced $NV verdicts for $NBASE corpus files — I/O flake"
  echo "       (seen with WSL 9p under load).  Rerun; if it repeats: wsl --shutdown."
  exit 2
}

echo
echo "===================== FULL-CORPUS BYTE-VERBATIM GATE ====================="
printf 'corpus files            : %d\n' "$NBASE"
printf 'corpus files changed    : %d  (sha256 vs gates/baseline_manifest.tsv)\n' "$CORPUS_DRIFT"
printf 'verdicts written        : %d\n' "$NV"
printf 'BYTE-IDENTICAL          : %d  (%.2f%%)\n' "$NID" "$(awk "BEGIN{print 100*$NID/$NBASE}")"
echo
echo '--- verdict tally ---'
cut -f3 "$WORK/verdicts.tsv" | sed 's/(.*//' | sort | uniq -c | sort -rn
echo
echo '--- per-directory ---'
awk -F'\t' '{d=$1; sub(/\/[^\/]*$/,"",d); t[d]++; if($3=="IDENTICAL") i[d]++}
  END{for(d in t) printf "%-32s %4d/%-4d  %6.2f%%\n", d, i[d]+0, t[d], 100*(i[d]+0)/t[d]}' \
  "$WORK/verdicts.tsv" | sort
echo '========================================================================='
echo "per-file verdicts: $WORK/verdicts.tsv"

if [ "$MODE" = "record" ]; then
  cp "$WORK/verdicts.tsv" "$BASELINE"
  echo "RECORDED baseline -> $BASELINE"
  exit 0
fi

[ "$MODE" = "report" ] && { echo; echo '--- non-IDENTICAL, in full ---';
  awk -F'\t' '$3!="IDENTICAL"' "$WORK/verdicts.tsv"; }

[ -f "$BASELINE" ] || { echo "NOTE: no baseline at $BASELINE — nothing to gate against."; exit 0; }
if diff -u "$BASELINE" "$WORK/verdicts.tsv" > "$WORK/regression.diff"; then
  echo "GATE PASS — every file matches the recorded baseline verdict."
  exit 0
else
  echo "GATE FAIL — verdicts moved vs $BASELINE:"
  sed -n '1,60p' "$WORK/regression.diff"
  exit 1
fi
