(* argv_roundtrip.ml — M5's acceptance: Codelet.of_argv / to_argv must
   round-trip every RECORDED argv line VERBATIM (flag order included).
   Input: a file of argv strings, one per line (argv[0] already stripped).
   Output: totals + every failure, worst first. *)

let () =
  let ic = open_in Sys.argv.(1) in
  let total = ref 0 and ok = ref 0 in
  let parse_fail = ref [] and rt_fail = ref [] in
  (try
     while true do
       let line = input_line ic in
       let line =
         if String.length line > 0 && line.[String.length line - 1] = '\r'
         then String.sub line 0 (String.length line - 1)
         else line
       in
       if String.length line > 0
       then (
         incr total;
         let argv =
           String.split_on_char ' ' line |> List.filter (fun s -> s <> "")
         in
         match Codelet.of_argv argv with
         | exception Codelet.Parse_error m ->
           parse_fail := (line, m) :: !parse_fail
         | c ->
           let back = String.concat " " (Codelet.to_argv c) in
           if back = line then incr ok else rt_fail := (line, back) :: !rt_fail)
     done
   with End_of_file -> ());
  Printf.printf "lines: %d  round-trip OK: %d  parse-fail: %d  order/content-fail: %d\n"
    !total !ok (List.length !parse_fail) (List.length !rt_fail);
  List.iteri
    (fun i (l, m) -> if i < 15 then Printf.printf "PARSE  [%s]  %s\n" m l)
    (List.rev !parse_fail);
  List.iteri
    (fun i (l, b) ->
       if i < 15 then Printf.printf "RTFAIL\n  rec: %s\n  got: %s\n" l b)
    (List.rev !rt_fail);
  exit (if !parse_fail = [] && !rt_fail = [] then 0 else 1)
