let () =
  (* both arities: the default keeps __restrict__, alias_tolerant drops it for
     the kinds il2p calls with zin == zout (forward T2, backward N1). *)
  print_string
    (Abi.z11_signature ~symbol:"radix16_z_t2_fwd_avx2" ~target_attr:"avx2,fma" ());
  print_string
    (Abi.z11_signature
       ~alias_tolerant:true
       ~symbol:"radix16_z_t2_fwd_avx2"
       ~target_attr:"avx2,fma"
       ())
;;
