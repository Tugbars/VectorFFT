#!/bin/bash
# Thin invoker (section 39). ALL coverage and recipes live in OCaml:
# generator/lib/coverage.ml (the single source of truth) walked by
# bin/gen_set.ml, one warm process for the whole tree (~9s vs ~30min
# for the old per-codelet fork loop). This script exists only so the
# documented entry point and arsenal.sh keep working unchanged.
#
#   bash regen_codelets.sh [all | inplace-avx2 | inplace-avx512 |
#                           oop-avx2 | oop-avx512 | strided-avx2 |
#                           strided-avx512]
set -eu
cd "$(dirname "$0")/.."
# 🔴 NEVER fall back to a bare `dune build` here (M0/G5, generator_lib_
# architecture.md §14.1): @default PROMOTES tracked generated/ headers EVEN
# WHEN THE BUILD FAILS (reproduced: rc=1 and plan_executors.h rewritten),
# and the old `2>/dev/null || dune build` fired exactly when someone was
# mid-refactor — corrupting the tree while hiding the real error.
dune build bin/gen_set.exe \
  || { echo "regen_codelets: scoped build of bin/gen_set.exe FAILED — fix the build; do NOT run a bare dune build" >&2; exit 1; }
exec ./_build/default/bin/gen_set.exe --root ../codelets "${@:-all}"
