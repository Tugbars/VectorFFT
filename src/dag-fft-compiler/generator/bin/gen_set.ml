(* gen_set — THE codelet-tree generator (section 39).
 *
 * Walks Coverage (the single source of truth) and emits every codelet
 * in-process via Gen_main.run, one warm process for the whole tree
 * instead of 753 forks. Per-codelet stdout is captured by dup2-ing a
 * temporary file (the target's name + ".tmp", in the target's folder, or in
 * its nearest existing parent while that folder does not exist yet) over
 * fd 1 around the run; provenance stamps carry the LOGICAL per-codelet
 * command via Emit_render.provenance_argv.
 *
 * A codelet lands under its name only whole: the temporary file is renamed
 * onto the target once the emitter returns with output, and removed when it
 * refuses, so a run never leaves an empty or partial codelet behind (an
 * existing target is kept as it was). A folder is created only when its
 * first codelet lands, so a refusal never leaves an empty folder behind
 * either (CMakeLists.txt refuses a codelet folder that holds no .c file;
 * an existing folder is kept as it was). The first refusal stops the run
 * (FAIL on stderr, exit 1); the files emitted before it are complete. A
 * cell the generator cannot emit at a zil ISA is declared absent in
 * Corpus.zil_isa_gap and never reaches emit_one.
 *
 * Usage:
 *   gen_set.exe [--root DIR] [--port P] [quadrant ...]      (default: all)
 * --root: the codelets/ directory (default "codelets" under cwd).
 * "all" is the FP64 tree (Corpus.quadrants); "f32" the FP32 quadrants (cil kernels and
 * ZTURN-T stage kernels; Corpus.quadrants_f32: zil-f32-avx2, zil-f32-avx512), never part
 * of "all".
 * --port P (FP32 only, 2026-11-10): every cil cell re-emitted with --cil-port P (the
 * FMA-port setting) under its own file and symbol name -- the A/B variants of a
 * corpus, written to their own --root; the FP32 ZTURN-T stage kernels, which have no
 * port setting, are re-emitted unchanged so the root holds the whole FP32 corpus; an
 * FP64 cell refuses it (the first refusal stops the run, as always). The FP32 corpus
 * itself carries the pair family at 50 (Corpus.zil_f32_with_port, the A/B's verdict);
 * --port 0 re-emits every cil cell at 0, the all-off base of an A/B. *)

let rec mkdir_p (dir : string) : unit =
  if dir = "" || dir = "." || dir = "/" || Sys.file_exists dir
  then ()
  else (
    mkdir_p (Filename.dirname dir);
    try Unix.mkdir dir 0o755 with
    | Unix.Unix_error (Unix.EEXIST, _, _) -> ())
;;

(* the deepest folder on the way to [dir] that exists ("." and "/" always do) *)
let rec existing_dir (dir : string) : string =
  if Sys.file_exists dir then dir else existing_dir (Filename.dirname dir)
;;

let emit_one (path : string) (argv_tail : string list) : unit =
  let argv = Array.of_list (("gen_radix.exe" :: argv_tail) @ [ "--emit-c" ]) in
  let dir = Filename.dirname path in
  let tmp = Filename.concat (existing_dir dir) (Filename.basename path ^ ".tmp") in
  flush stdout;
  let fd = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o644 in
  let saved = Unix.dup Unix.stdout in
  Unix.dup2 fd Unix.stdout;
  Unix.close fd;
  (try
     Fun.protect
       ~finally:(fun () ->
         flush stdout;
         Unix.dup2 saved Unix.stdout;
         Unix.close saved)
       (fun () -> Gen_main.run argv);
     if (Unix.stat tmp).Unix.st_size = 0 then failwith "gen_set: the emitter wrote nothing"
   with
   | e ->
     (try Sys.remove tmp with
      | Sys_error _ -> ());
     raise e);
  mkdir_p dir;
  Unix.rename tmp path
;;

(* the cell's argv with --cil-port P (the canonical position, through the descriptor) *)
let with_port (port : int option) (tail : string list) : string list =
  match port with
  | None -> tail
  | Some port -> (
    let open Codelet in
    let c = of_argv (tail @ [ "--emit-c" ]) in
    match c.kind with
    | Cil r ->
      List.filter (fun a -> a <> "--emit-c") (to_argv { c with kind = Cil { r with port } })
    | Zsplit _ when (match c.isa with Some i -> List.mem i Corpus.zil_f32_isas | None -> false) ->
      tail   (* the FP32 ZTURN-T stage kernels: no port setting, emitted unchanged *)
    | _ -> failwith "gen_set: --port applies to the cil kinds only")
;;

let () =
  let root = ref "codelets" in
  let port = ref None in
  let targets = ref [] in
  let rec parse i =
    if i < Array.length Sys.argv
    then (
      match Sys.argv.(i) with
      | "--root" when i + 1 < Array.length Sys.argv ->
        root := Sys.argv.(i + 1);
        parse (i + 2) |> ignore
      | "--port" when i + 1 < Array.length Sys.argv ->
        (port :=
           match int_of_string_opt Sys.argv.(i + 1) with
           | Some p when p >= 0 && p <= 100 -> Some p
           | _ -> failwith "gen_set: --port takes a percentage 0..100");
        parse (i + 2) |> ignore
      | t ->
        targets := t :: !targets;
        parse (i + 1) |> ignore)
  in
  ignore (parse 1);
  let targets =
    match List.rev !targets with
    | [] | [ "all" ] -> Corpus.quadrants
    | [ "f32" ] -> Corpus.quadrants_f32
    | l -> l
  in
  let total = ref 0 in
  List.iter
    (fun q ->
       let files = List.map (fun (name, tail) -> name, with_port !port tail) (Corpus.files q) in
       List.iter
         (fun (name, tail) ->
            (* the folder follows the file's kind (2026-09-24): Corpus.dir_of_file *)
            let dir = Filename.concat !root (Corpus.dir_of_file q name) in
            let path = Filename.concat dir name in
            (try emit_one path tail with
             | e ->
               Printf.eprintf "FAIL %s: %s\n" path (Printexc.to_string e);
               exit 1);
            incr total)
         files;
       Printf.printf "%s: %d files\n%!" q (List.length files))
    targets;
  Printf.printf "gen_set: %d files total\n%!" !total
;;
