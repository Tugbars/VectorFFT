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
dune build bin/gen_set.exe 2>/dev/null || dune build
exec ./_build/default/bin/gen_set.exe --root ../codelets "${@:-all}"
