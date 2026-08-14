(* knobs.ml — M6.0.  See knobs.mli. *)

let snap key = lazy (Sys.getenv_opt key)
let get l () = Lazy.force l

let sched_order = get (snap "VFFT_SCHED_ORDER")
let sched_loads = get (snap "VFFT_SCHED_LOADS")
let load_pace = get (snap "VFFT_LOAD_PACE")
let su_tiebreak = get (snap "VFFT_SU_TIEBREAK")
let gh_threshold = get (snap "VFFT_GH_THRESHOLD")
let fma_multiuse = get (snap "VFFT_FMA_MULTIUSE")
let collect_m = get (snap "VFFT_COLLECT_M")
let deep_collect = get (snap "VFFT_DEEP_COLLECT")

module Trace = struct
  let flag key = let l = snap key in fun () -> Lazy.force l <> None
  let factor = flag "FACTOR_TRACE"
  let mulift = flag "MULIFT_TRACE"
  let mulfma = flag "MULFMA_TRACE"
  let fma_addend = flag "FMA_ADDEND_TRACE"
  let flatten_fma_mul = flag "FLATTEN_FMA_MUL_TRACE"
  let flatten_fma_mul_verbose = flag "FLATTEN_FMA_MUL_TRACE_VERBOSE"
  let deep_collect = let l = snap "VFFT_DEEP_COLLECT_TRACE" in fun () -> Lazy.force l = Some "1"
  let sched_dump = get (snap "VFFT_SCHED_DUMP")
end
