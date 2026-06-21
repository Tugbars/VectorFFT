(* gen_set — THE codelet-tree generator (section 39).
 *
 * Walks Coverage (the single source of truth) and emits every codelet
 * in-process via Gen_main.run, one warm process for the whole tree
 * instead of 753 forks. Per-codelet stdout is captured by dup2-ing the
 * target file over fd 1 around the run; provenance stamps carry the
 * LOGICAL per-codelet command via Emit_c.provenance_argv.
 *
 * Usage:
 *   gen_set.exe [--root DIR] [quadrant ...]      (default: all)
 * Quadrants: inplace-avx2 inplace-avx512 oop-avx2 oop-avx512
 *            strided-avx2 strided-avx512
 * --root: the codelets/ directory (default "codelets" under cwd). *)

let rec mkdir_p (dir : string) : unit =
  if dir = "" || dir = "." || dir = "/" || Sys.file_exists dir then ()
  else begin
    mkdir_p (Filename.dirname dir);
    try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  end

let emit_one (path : string) (argv_tail : string list) : unit =
  let argv = Array.of_list (("gen_radix.exe" :: argv_tail) @ [ "--emit-c" ]) in
  flush stdout;
  let fd =
    Unix.openfile path [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o644
  in
  let saved = Unix.dup Unix.stdout in
  Unix.dup2 fd Unix.stdout;
  Unix.close fd;
  Fun.protect
    ~finally:(fun () ->
      flush stdout;
      Unix.dup2 saved Unix.stdout;
      Unix.close saved)
    (fun () -> Vfft_v2.Gen_main.run argv)

let () =
  let root = ref "codelets" in
  let targets = ref [] in
  let rec parse i =
    if i < Array.length Sys.argv then
      begin match Sys.argv.(i) with
      | "--root" when i + 1 < Array.length Sys.argv ->
          root := Sys.argv.(i + 1);
          parse (i + 2) |> ignore
      | t ->
          targets := t :: !targets;
          parse (i + 1) |> ignore
      end
  in
  ignore (parse 1);
  let targets =
    match List.rev !targets with
    | [] | [ "all" ] -> Vfft_v2.Coverage.quadrants
    | l -> l
  in
  let total = ref 0 in
  List.iter
    (fun q ->
      let files = Vfft_v2.Coverage.files q in
      let dir = Filename.concat !root (Vfft_v2.Coverage.dir_of_quadrant q) in
      mkdir_p dir;
      List.iter
        (fun (name, tail) ->
          let path = Filename.concat dir name in
          (try emit_one path tail
           with e ->
             Printf.eprintf "FAIL %s: %s\n" path (Printexc.to_string e);
             exit 1);
          incr total)
        files;
      Printf.printf "%s: %d files\n%!" q (List.length files))
    targets;
  Printf.printf "gen_set: %d files total\n%!" !total
