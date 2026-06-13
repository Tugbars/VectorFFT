0000000002609500 <mkl_dft_avx512_mg_colbatch_plain_fwd_16_d>:
 2609500:	f3 0f 1e fa          	endbr64
 2609504:	55                   	push   %rbp
 2609505:	41 57                	push   %r15
 2609507:	41 56                	push   %r14
 2609509:	41 55                	push   %r13
 260950b:	41 54                	push   %r12
 260950d:	53                   	push   %rbx
 260950e:	48 89 d7             	mov    %rdx,%rdi
 2609511:	49 89 f6             	mov    %rsi,%r14
 2609514:	89 c8                	mov    %ecx,%eax
 2609516:	83 e0 03             	and    $0x3,%eax
 2609519:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
 260951e:	48 c1 e9 02          	shr    $0x2,%rcx
 2609522:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
 2609527:	0f 84 5d 04 00 00    	je     260998a <mkl_dft_avx512_mg_colbatch_plain_fwd_16_d+0x48a>
 260952d:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
 2609532:	48 83 f8 02          	cmp    $0x2,%rax
 2609536:	b9 01 00 00 00       	mov    $0x1,%ecx
 260953b:	48 0f 4d c8          	cmovge %rax,%rcx
 260953f:	48 89 4c 24 f8       	mov    %rcx,-0x8(%rsp)
 2609544:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
 2609549:	48 ff c0             	inc    %rax
 260954c:	48 89 44 24 f0       	mov    %rax,-0x10(%rsp)
 2609551:	4c 8b 7c 24 48       	mov    0x48(%rsp),%r15
 2609556:	49 c1 e7 04          	shl    $0x4,%r15
 260955a:	4c 8b 64 24 40       	mov    0x40(%rsp),%r12
 260955f:	49 c1 e4 04          	shl    $0x4,%r12
 2609563:	4d 89 c5             	mov    %r8,%r13
 2609566:	49 c1 e5 04          	shl    $0x4,%r13
 260956a:	4c 89 cd             	mov    %r9,%rbp
 260956d:	48 c1 e5 04          	shl    $0x4,%rbp
 2609571:	45 31 db             	xor    %r11d,%r11d
 2609574:	62 f2 fd 48 19 05 0a 	vbroadcastsd 0x263580a(%rip),%zmm0        # 4c3ed88 <row_factorization_db+0xac58>
 260957b:	58 63 02 
 260957e:	62 f2 fd 48 19 0d 78 	vbroadcastsd 0x262dc78(%rip),%zmm1        # 4c37200 <row_factorization_db+0x30d0>
 2609585:	dc 62 02 
 2609588:	62 f2 fd 48 19 15 ee 	vbroadcastsd 0x26357ee(%rip),%zmm2        # 4c3ed80 <row_factorization_db+0xac50>
 260958f:	57 63 02 
 2609592:	62 f2 fd 48 19 1d 64 	vbroadcastsd 0x263cf64(%rip),%zmm3        # 4c46500 <row_factorization_db+0x123d0>
 2609599:	cf 63 02 
 260959c:	62 f2 fd 48 19 25 62 	vbroadcastsd 0x263cf62(%rip),%zmm4        # 4c46508 <row_factorization_db+0x123d8>
 26095a3:	cf 63 02 
 26095a6:	62 f2 fd 48 19 2d d8 	vbroadcastsd 0x25e2cd8(%rip),%zmm5        # 4bec288 <convolution.id+0x88>
 26095ad:	2c 5e 02 
 26095b0:	4c 89 f6             	mov    %r14,%rsi
 26095b3:	48 89 fa             	mov    %rdi,%rdx
 26095b6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
 26095bd:	00 00 00 
 26095c0:	31 c0                	xor    %eax,%eax
 26095c2:	48 8b 4c 24 f0       	mov    -0x10(%rsp),%rcx
 26095c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
 26095ce:	00 00 
 26095d0:	48 8d 1c 06          	lea    (%rsi,%rax,1),%rbx
 26095d4:	62 d1 fd 48 10 7c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm7
 26095db:	00 
 26095dc:	4c 01 eb             	add    %r13,%rbx
 26095df:	62 51 fd 48 10 44 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm8
 26095e6:	00 
 26095e7:	4c 01 eb             	add    %r13,%rbx
 26095ea:	62 51 fd 48 10 4c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm9
 26095f1:	00 
 26095f2:	4c 01 eb             	add    %r13,%rbx
 26095f5:	62 d1 fd 48 10 74 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm6
 26095fc:	00 
 26095fd:	4c 01 eb             	add    %r13,%rbx
 2609600:	62 51 fd 48 10 54 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm10
 2609607:	00 
 2609608:	4c 01 eb             	add    %r13,%rbx
 260960b:	62 51 fd 48 10 5c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm11
 2609612:	00 
 2609613:	4c 01 eb             	add    %r13,%rbx
 2609616:	62 51 fd 48 10 64 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm12
 260961d:	00 
 260961e:	4c 01 eb             	add    %r13,%rbx
 2609621:	62 51 fd 48 10 6c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm13
 2609628:	00 
 2609629:	4c 01 eb             	add    %r13,%rbx
 260962c:	62 51 fd 48 10 74 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm14
 2609633:	00 
 2609634:	4c 01 eb             	add    %r13,%rbx
 2609637:	62 51 fd 48 10 7c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm15
 260963e:	00 
 260963f:	4c 01 eb             	add    %r13,%rbx
 2609642:	62 c1 fd 48 10 44 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm16
 2609649:	00 
 260964a:	4c 01 eb             	add    %r13,%rbx
 260964d:	62 c1 fd 48 10 4c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm17
 2609654:	00 
 2609655:	4c 01 eb             	add    %r13,%rbx
 2609658:	62 c1 fd 48 10 54 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm18
 260965f:	00 
 2609660:	4c 01 eb             	add    %r13,%rbx
 2609663:	62 c1 fd 48 10 5c 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm19
 260966a:	00 
 260966b:	4c 8d 14 02          	lea    (%rdx,%rax,1),%r10
 260966f:	4c 01 eb             	add    %r13,%rbx
 2609672:	62 c1 fd 48 10 64 1d 	vmovupd 0x0(%r13,%rbx,1),%zmm20
 2609679:	00 
 260967a:	62 e1 fd 48 10 2c 06 	vmovupd (%rsi,%rax,1),%zmm21
 2609681:	62 c1 d5 40 58 f5    	vaddpd %zmm13,%zmm21,%zmm22
 2609687:	62 a1 cd 48 58 f9    	vaddpd %zmm17,%zmm6,%zmm23
 260968d:	62 41 bd 48 58 c7    	vaddpd %zmm15,%zmm8,%zmm24
 2609693:	62 51 bd 48 5c c7    	vsubpd %zmm15,%zmm8,%zmm8
 2609699:	62 51 c5 48 58 fe    	vaddpd %zmm14,%zmm7,%zmm15
 260969f:	62 21 a5 48 58 cb    	vaddpd %zmm19,%zmm11,%zmm25
 26096a5:	62 d1 c5 48 5c fe    	vsubpd %zmm14,%zmm7,%zmm7
 26096ab:	62 31 b5 48 58 f0    	vaddpd %zmm16,%zmm9,%zmm14
 26096b1:	62 31 b5 48 5c c8    	vsubpd %zmm16,%zmm9,%zmm9
 26096b7:	62 a1 ad 48 58 c2    	vaddpd %zmm18,%zmm10,%zmm16
 26096bd:	62 31 ad 48 5c d2    	vsubpd %zmm18,%zmm10,%zmm10
 26096c3:	62 31 a5 48 5c db    	vsubpd %zmm19,%zmm11,%zmm11
 26096c9:	62 a1 9d 48 58 d4    	vaddpd %zmm20,%zmm12,%zmm18
 26096cf:	62 31 9d 48 5c e4    	vsubpd %zmm20,%zmm12,%zmm12
 26096d5:	62 a1 cd 40 58 df    	vaddpd %zmm23,%zmm22,%zmm19
 26096db:	62 a1 85 48 58 e0    	vaddpd %zmm16,%zmm15,%zmm20
 26096e1:	62 31 85 48 5c f8    	vsubpd %zmm16,%zmm15,%zmm15
 26096e7:	62 81 bd 40 58 c1    	vaddpd %zmm25,%zmm24,%zmm16
 26096ed:	62 21 8d 48 58 d2    	vaddpd %zmm18,%zmm14,%zmm26
 26096f3:	62 41 c5 48 58 dc    	vaddpd %zmm12,%zmm7,%zmm27
 26096f9:	62 41 c5 48 5c e4    	vsubpd %zmm12,%zmm7,%zmm28
 26096ff:	62 41 b5 48 58 ea    	vaddpd %zmm10,%zmm9,%zmm29
 2609705:	62 51 ad 48 5c d1    	vsubpd %zmm9,%zmm10,%zmm10
 260970b:	62 b1 8d 48 5c fa    	vsubpd %zmm18,%zmm14,%zmm7
 2609711:	62 31 cd 40 5c cf    	vsubpd %zmm23,%zmm22,%zmm9
 2609717:	62 11 bd 40 5c f1    	vsubpd %zmm25,%zmm24,%zmm14
 260971d:	62 51 d5 40 5c ed    	vsubpd %zmm13,%zmm21,%zmm13
 2609723:	62 a1 cd 48 5c c9    	vsubpd %zmm17,%zmm6,%zmm17
 2609729:	62 f1 85 48 5c f7    	vsubpd %zmm7,%zmm15,%zmm6
 260972f:	62 71 85 48 58 ff    	vaddpd %zmm7,%zmm15,%zmm15
 2609735:	62 81 ad 48 58 d4    	vaddpd %zmm28,%zmm10,%zmm18
 260973b:	62 91 95 40 58 fb    	vaddpd %zmm27,%zmm29,%zmm7
 2609741:	62 c1 bd 48 5c eb    	vsubpd %zmm11,%zmm8,%zmm21
 2609747:	62 c1 bd 48 58 f3    	vaddpd %zmm11,%zmm8,%zmm22
 260974d:	62 31 e5 40 58 d8    	vaddpd %zmm16,%zmm19,%zmm11
 2609753:	62 11 dd 40 58 e2    	vaddpd %zmm26,%zmm20,%zmm12
 2609759:	62 71 c5 48 59 c0    	vmulpd %zmm0,%zmm7,%zmm8
 260975f:	62 b1 e5 40 5c f8    	vsubpd %zmm16,%zmm19,%zmm7
 2609765:	62 e1 fd 48 28 c1    	vmovapd %zmm1,%zmm16
 260976b:	62 c2 d5 40 ac c5    	vfnmadd213pd %zmm13,%zmm21,%zmm16
 2609771:	62 c2 f5 48 a8 ed    	vfmadd213pd %zmm13,%zmm1,%zmm21
 2609777:	62 71 ed 40 59 ea    	vmulpd %zmm2,%zmm18,%zmm13
 260977d:	62 42 e5 48 ac e5    	vfnmadd213pd %zmm13,%zmm3,%zmm28
 2609783:	62 52 dd 48 a8 d5    	vfmadd213pd %zmm13,%zmm4,%zmm10
 2609789:	62 e1 fd 48 28 d1    	vmovapd %zmm1,%zmm18
 260978f:	62 a2 cd 40 ac d1    	vfnmadd213pd %zmm17,%zmm22,%zmm18
 2609795:	62 a2 f5 48 a8 f1    	vfmadd213pd %zmm17,%zmm1,%zmm22
 260979b:	62 42 dd 48 a8 d8    	vfmadd213pd %zmm8,%zmm4,%zmm27
 26097a1:	62 42 e5 48 a8 e8    	vfmadd213pd %zmm8,%zmm3,%zmm29
 26097a7:	62 71 fd 48 28 c1    	vmovapd %zmm1,%zmm8
 26097ad:	62 81 dd 40 5c ca    	vsubpd %zmm26,%zmm20,%zmm17
 26097b3:	62 52 cd 48 ac c1    	vfnmadd213pd %zmm9,%zmm6,%zmm8
 26097b9:	62 d2 f5 48 a8 f1    	vfmadd213pd %zmm9,%zmm1,%zmm6
 26097bf:	62 e1 fd 48 28 d9    	vmovapd %zmm1,%zmm19
 26097c5:	62 c2 85 48 ac de    	vfnmadd213pd %zmm14,%zmm15,%zmm19
 26097cb:	62 52 f5 48 ae fe    	vfnmsub213pd %zmm14,%zmm1,%zmm15
 26097d1:	62 11 fd 40 58 cc    	vaddpd %zmm28,%zmm16,%zmm9
 26097d7:	62 11 fd 40 5c f4    	vsubpd %zmm28,%zmm16,%zmm14
 26097dd:	62 51 d5 40 58 ea    	vaddpd %zmm10,%zmm21,%zmm13
 26097e3:	62 51 d5 40 5c d2    	vsubpd %zmm10,%zmm21,%zmm10
 26097e9:	62 81 ed 40 58 e3    	vaddpd %zmm27,%zmm18,%zmm20
 26097ef:	62 81 cd 40 58 ed    	vaddpd %zmm29,%zmm22,%zmm21
 26097f5:	62 81 cd 40 5c f5    	vsubpd %zmm29,%zmm22,%zmm22
 26097fb:	62 81 ed 40 5c d3    	vsubpd %zmm27,%zmm18,%zmm18
 2609801:	62 a1 e5 40 c6 c3 55 	vshufpd $0x55,%zmm19,%zmm19,%zmm16
 2609808:	62 c1 85 48 c6 df 55 	vshufpd $0x55,%zmm15,%zmm15,%zmm19
 260980f:	62 a1 f5 40 c6 c9 55 	vshufpd $0x55,%zmm17,%zmm17,%zmm17
 2609816:	62 31 dd 40 c6 fc 55 	vshufpd $0x55,%zmm20,%zmm20,%zmm15
 260981d:	62 a1 ed 40 c6 d2 55 	vshufpd $0x55,%zmm18,%zmm18,%zmm18
 2609824:	62 a1 cd 40 c6 e6 55 	vshufpd $0x55,%zmm22,%zmm22,%zmm20
 260982b:	62 c1 a5 48 58 f4    	vaddpd %zmm12,%zmm11,%zmm22
 2609831:	62 c1 fd 48 28 f8    	vmovapd %zmm8,%zmm23
 2609837:	62 a2 d5 48 a7 f8    	vfmsubadd213pd %zmm16,%zmm5,%zmm23
 260983d:	62 61 fd 48 28 c6    	vmovapd %zmm6,%zmm24
 2609843:	62 22 d5 48 a7 c3    	vfmsubadd213pd %zmm19,%zmm5,%zmm24
 2609849:	62 61 fd 48 28 cf    	vmovapd %zmm7,%zmm25
 260984f:	62 22 d5 48 a7 c9    	vfmsubadd213pd %zmm17,%zmm5,%zmm25
 2609855:	62 41 fd 48 28 d1    	vmovapd %zmm9,%zmm26
 260985b:	62 42 d5 48 a7 d7    	vfmsubadd213pd %zmm15,%zmm5,%zmm26
 2609861:	62 a1 d5 40 c6 ed 55 	vshufpd $0x55,%zmm21,%zmm21,%zmm21
 2609868:	62 41 fd 48 28 de    	vmovapd %zmm14,%zmm27
 260986e:	62 22 d5 48 a6 da    	vfmaddsub213pd %zmm18,%zmm5,%zmm27
 2609874:	62 41 fd 48 28 e5    	vmovapd %zmm13,%zmm28
 260987a:	62 22 d5 48 a7 e5    	vfmsubadd213pd %zmm21,%zmm5,%zmm28
 2609880:	62 41 fd 48 28 ea    	vmovapd %zmm10,%zmm29
 2609886:	62 e1 fd 48 11 34 02 	vmovupd %zmm22,(%rdx,%rax,1)
 260988d:	62 21 fd 48 11 54 15 	vmovupd %zmm26,0x0(%rbp,%r10,1)
 2609894:	00 
 2609895:	62 22 d5 48 a6 ec    	vfmaddsub213pd %zmm20,%zmm5,%zmm29
 260989b:	49 01 ea             	add    %rbp,%r10
 260989e:	62 a1 fd 48 11 7c 15 	vmovupd %zmm23,0x0(%rbp,%r10,1)
 26098a5:	00 
 26098a6:	49 01 ea             	add    %rbp,%r10
 26098a9:	62 51 a5 48 5c dc    	vsubpd %zmm12,%zmm11,%zmm11
 26098af:	62 21 fd 48 11 6c 15 	vmovupd %zmm29,0x0(%rbp,%r10,1)
 26098b6:	00 
 26098b7:	49 01 ea             	add    %rbp,%r10
 26098ba:	62 21 fd 48 11 4c 15 	vmovupd %zmm25,0x0(%rbp,%r10,1)
 26098c1:	00 
 26098c2:	62 32 d5 48 a7 f2    	vfmsubadd213pd %zmm18,%zmm5,%zmm14
 26098c8:	62 b2 d5 48 a6 f3    	vfmaddsub213pd %zmm19,%zmm5,%zmm6
 26098ce:	49 01 ea             	add    %rbp,%r10
 26098d1:	62 21 fd 48 11 64 15 	vmovupd %zmm28,0x0(%rbp,%r10,1)
 26098d8:	00 
 26098d9:	49 01 ea             	add    %rbp,%r10
 26098dc:	62 32 d5 48 a6 ed    	vfmaddsub213pd %zmm21,%zmm5,%zmm13
 26098e2:	62 21 fd 48 11 44 15 	vmovupd %zmm24,0x0(%rbp,%r10,1)
 26098e9:	00 
 26098ea:	49 01 ea             	add    %rbp,%r10
 26098ed:	62 21 fd 48 11 5c 15 	vmovupd %zmm27,0x0(%rbp,%r10,1)
 26098f4:	00 
 26098f5:	62 b2 d5 48 a6 f9    	vfmaddsub213pd %zmm17,%zmm5,%zmm7
 26098fb:	62 32 d5 48 a7 d4    	vfmsubadd213pd %zmm20,%zmm5,%zmm10
 2609901:	49 01 ea             	add    %rbp,%r10
 2609904:	62 31 fd 48 11 5c 15 	vmovupd %zmm11,0x0(%rbp,%r10,1)
 260990b:	00 
 260990c:	49 01 ea             	add    %rbp,%r10
 260990f:	62 32 d5 48 a6 c0    	vfmaddsub213pd %zmm16,%zmm5,%zmm8
 2609915:	62 31 fd 48 11 74 15 	vmovupd %zmm14,0x0(%rbp,%r10,1)
 260991c:	00 
 260991d:	49 01 ea             	add    %rbp,%r10
 2609920:	62 b1 fd 48 11 74 15 	vmovupd %zmm6,0x0(%rbp,%r10,1)
 2609927:	00 
 2609928:	62 52 d5 48 a6 cf    	vfmaddsub213pd %zmm15,%zmm5,%zmm9
 260992e:	49 01 ea             	add    %rbp,%r10
 2609931:	62 31 fd 48 11 6c 15 	vmovupd %zmm13,0x0(%rbp,%r10,1)
 2609938:	00 
 2609939:	49 01 ea             	add    %rbp,%r10
 260993c:	62 b1 fd 48 11 7c 15 	vmovupd %zmm7,0x0(%rbp,%r10,1)
 2609943:	00 
 2609944:	49 01 ea             	add    %rbp,%r10
 2609947:	62 31 fd 48 11 54 15 	vmovupd %zmm10,0x0(%rbp,%r10,1)
 260994e:	00 
 260994f:	49 01 ea             	add    %rbp,%r10
 2609952:	62 31 fd 48 11 44 15 	vmovupd %zmm8,0x0(%rbp,%r10,1)
 2609959:	00 
 260995a:	49 01 ea             	add    %rbp,%r10
 260995d:	62 31 fd 48 11 4c 15 	vmovupd %zmm9,0x0(%rbp,%r10,1)
 2609964:	00 
 2609965:	48 ff c9             	dec    %rcx
 2609968:	48 83 c0 40          	add    $0x40,%rax
 260996c:	48 83 f9 01          	cmp    $0x1,%rcx
 2609970:	0f 8f 5a fc ff ff    	jg     26095d0 <mkl_dft_avx512_mg_colbatch_plain_fwd_16_d+0xd0>
 2609976:	49 ff c3             	inc    %r11
 2609979:	4c 01 fa             	add    %r15,%rdx
 260997c:	4c 01 e6             	add    %r12,%rsi
 260997f:	4c 3b 5c 24 f8       	cmp    -0x8(%rsp),%r11
 2609984:	0f 85 36 fc ff ff    	jne    26095c0 <mkl_dft_avx512_mg_colbatch_plain_fwd_16_d+0xc0>
 260998a:	48 8b 4c 24 e8       	mov    -0x18(%rsp),%rcx
 260998f:	48 85 c9             	test   %rcx,%rcx
 2609992:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
 2609997:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
 260999c:	0f 84 1a 04 00 00    	je     2609dbc <mkl_dft_avx512_mg_colbatch_plain_fwd_16_d+0x8bc>
 26099a2:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
 26099a7:	48 c1 e0 06          	shl    $0x6,%rax
 26099ab:	49 01 c6             	add    %rax,%r14
 26099ae:	48 01 c7             	add    %rax,%rdi
 26099b1:	00 c9                	add    %cl,%cl
 26099b3:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 26099b8:	d2 e0                	shl    %cl,%al
 26099ba:	f6 d0                	not    %al
 26099bc:	c5 fb 92 c8          	kmovd  %eax,%k1
 26099c0:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
 26099c5:	48 83 f9 02          	cmp    $0x2,%rcx
 26099c9:	b8 01 00 00 00       	mov    $0x1,%eax
 26099ce:	48 0f 4d c1          	cmovge %rcx,%rax
 26099d2:	48 c1 e6 04          	shl    $0x4,%rsi
 26099d6:	48 c1 e2 04          	shl    $0x4,%rdx
 26099da:	49 69 c8 f0 00 00 00 	imul   $0xf0,%r8,%rcx
 26099e1:	49 c1 e0 04          	shl    $0x4,%r8
 26099e5:	48 29 ce             	sub    %rcx,%rsi
 26099e8:	49 69 c9 f0 00 00 00 	imul   $0xf0,%r9,%rcx
 26099ef:	49 c1 e1 04          	shl    $0x4,%r9
 26099f3:	48 29 ca             	sub    %rcx,%rdx
 26099f6:	62 f2 fd 48 19 05 88 	vbroadcastsd 0x2635388(%rip),%zmm0        # 4c3ed88 <row_factorization_db+0xac58>
 26099fd:	53 63 02 
 2609a00:	62 f2 fd 48 19 0d f6 	vbroadcastsd 0x262d7f6(%rip),%zmm1        # 4c37200 <row_factorization_db+0x30d0>
 2609a07:	d7 62 02 
 2609a0a:	62 f2 fd 48 19 15 6c 	vbroadcastsd 0x263536c(%rip),%zmm2        # 4c3ed80 <row_factorization_db+0xac50>
 2609a11:	53 63 02 
 2609a14:	62 f2 fd 48 19 1d e2 	vbroadcastsd 0x263cae2(%rip),%zmm3        # 4c46500 <row_factorization_db+0x123d0>
 2609a1b:	ca 63 02 
 2609a1e:	62 f2 fd 48 19 25 e0 	vbroadcastsd 0x263cae0(%rip),%zmm4        # 4c46508 <row_factorization_db+0x123d8>
 2609a25:	ca 63 02 
 2609a28:	62 f2 fd 48 19 2d 56 	vbroadcastsd 0x25e2856(%rip),%zmm5        # 4bec288 <convolution.id+0x88>
 2609a2f:	28 5e 02 
 2609a32:	4c 01 c6             	add    %r8,%rsi
 2609a35:	4c 01 ca             	add    %r9,%rdx
 2609a38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
 2609a3f:	00 
 2609a40:	62 d1 fd c9 10 36    	vmovupd (%r14),%zmm6{%k1}{z}
 2609a46:	62 11 fd c9 10 04 06 	vmovupd (%r14,%r8,1),%zmm8{%k1}{z}
 2609a4d:	4d 01 c6             	add    %r8,%r14
 2609a50:	62 11 fd c9 10 0c 30 	vmovupd (%r8,%r14,1),%zmm9{%k1}{z}
 2609a57:	4d 01 c6             	add    %r8,%r14
 2609a5a:	62 11 fd c9 10 14 30 	vmovupd (%r8,%r14,1),%zmm10{%k1}{z}
 2609a61:	4d 01 c6             	add    %r8,%r14
 2609a64:	62 91 fd c9 10 3c 30 	vmovupd (%r8,%r14,1),%zmm7{%k1}{z}
 2609a6b:	4d 01 c6             	add    %r8,%r14
 2609a6e:	62 11 fd c9 10 1c 30 	vmovupd (%r8,%r14,1),%zmm11{%k1}{z}
 2609a75:	4d 01 c6             	add    %r8,%r14
 2609a78:	62 11 fd c9 10 24 30 	vmovupd (%r8,%r14,1),%zmm12{%k1}{z}
 2609a7f:	4d 01 c6             	add    %r8,%r14
 2609a82:	62 11 fd c9 10 2c 30 	vmovupd (%r8,%r14,1),%zmm13{%k1}{z}
 2609a89:	4d 01 c6             	add    %r8,%r14
 2609a8c:	62 11 fd c9 10 34 30 	vmovupd (%r8,%r14,1),%zmm14{%k1}{z}
 2609a93:	4d 01 c6             	add    %r8,%r14
 2609a96:	62 11 fd c9 10 3c 30 	vmovupd (%r8,%r14,1),%zmm15{%k1}{z}
 2609a9d:	4d 01 c6             	add    %r8,%r14
 2609aa0:	62 81 fd c9 10 04 30 	vmovupd (%r8,%r14,1),%zmm16{%k1}{z}
 2609aa7:	4d 01 c6             	add    %r8,%r14
 2609aaa:	62 81 fd c9 10 0c 30 	vmovupd (%r8,%r14,1),%zmm17{%k1}{z}
 2609ab1:	4d 01 c6             	add    %r8,%r14
 2609ab4:	62 81 fd c9 10 14 30 	vmovupd (%r8,%r14,1),%zmm18{%k1}{z}
 2609abb:	4d 01 c6             	add    %r8,%r14
 2609abe:	62 81 fd c9 10 1c 30 	vmovupd (%r8,%r14,1),%zmm19{%k1}{z}
 2609ac5:	4d 01 c6             	add    %r8,%r14
 2609ac8:	62 81 fd c9 10 24 30 	vmovupd (%r8,%r14,1),%zmm20{%k1}{z}
 2609acf:	4d 01 c6             	add    %r8,%r14
 2609ad2:	62 81 fd c9 10 2c 30 	vmovupd (%r8,%r14,1),%zmm21{%k1}{z}
 2609ad9:	62 c1 cd 48 58 f6    	vaddpd %zmm14,%zmm6,%zmm22
 2609adf:	62 a1 c5 48 58 fa    	vaddpd %zmm18,%zmm7,%zmm23
 2609ae5:	62 21 b5 48 58 c0    	vaddpd %zmm16,%zmm9,%zmm24
 2609aeb:	62 31 b5 48 5c c8    	vsubpd %zmm16,%zmm9,%zmm9
 2609af1:	62 c1 bd 48 58 c7    	vaddpd %zmm15,%zmm8,%zmm16
 2609af7:	62 21 9d 48 58 cc    	vaddpd %zmm20,%zmm12,%zmm25
 2609afd:	62 51 bd 48 5c c7    	vsubpd %zmm15,%zmm8,%zmm8
 2609b03:	62 31 ad 48 58 f9    	vaddpd %zmm17,%zmm10,%zmm15
 2609b09:	62 31 ad 48 5c d1    	vsubpd %zmm17,%zmm10,%zmm10
 2609b0f:	62 a1 a5 48 58 cb    	vaddpd %zmm19,%zmm11,%zmm17
 2609b15:	62 31 a5 48 5c db    	vsubpd %zmm19,%zmm11,%zmm11
 2609b1b:	62 31 9d 48 5c e4    	vsubpd %zmm20,%zmm12,%zmm12
 2609b21:	62 a1 95 48 58 dd    	vaddpd %zmm21,%zmm13,%zmm19
 2609b27:	62 31 95 48 5c ed    	vsubpd %zmm21,%zmm13,%zmm13
 2609b2d:	62 a1 cd 40 58 e7    	vaddpd %zmm23,%zmm22,%zmm20
 2609b33:	62 a1 fd 40 58 e9    	vaddpd %zmm17,%zmm16,%zmm21
 2609b39:	62 a1 fd 40 5c c1    	vsubpd %zmm17,%zmm16,%zmm16
 2609b3f:	62 81 bd 40 58 c9    	vaddpd %zmm25,%zmm24,%zmm17
 2609b45:	62 21 85 48 58 d3    	vaddpd %zmm19,%zmm15,%zmm26
 2609b4b:	62 41 bd 48 58 dd    	vaddpd %zmm13,%zmm8,%zmm27
 2609b51:	62 51 bd 48 5c ed    	vsubpd %zmm13,%zmm8,%zmm13
 2609b57:	62 41 ad 48 58 e3    	vaddpd %zmm11,%zmm10,%zmm28
 2609b5d:	62 51 a5 48 5c d2    	vsubpd %zmm10,%zmm11,%zmm10
 2609b63:	62 31 85 48 5c c3    	vsubpd %zmm19,%zmm15,%zmm8
 2609b69:	62 31 cd 40 5c ff    	vsubpd %zmm23,%zmm22,%zmm15
 2609b6f:	62 81 bd 40 5c d9    	vsubpd %zmm25,%zmm24,%zmm19
 2609b75:	62 51 cd 48 5c f6    	vsubpd %zmm14,%zmm6,%zmm14
 2609b7b:	62 a1 c5 48 5c d2    	vsubpd %zmm18,%zmm7,%zmm18
 2609b81:	62 d1 fd 40 5c f0    	vsubpd %zmm8,%zmm16,%zmm6
 2609b87:	62 c1 fd 40 58 f0    	vaddpd %zmm8,%zmm16,%zmm22
 2609b8d:	62 51 ad 48 58 c5    	vaddpd %zmm13,%zmm10,%zmm8
 2609b93:	62 91 9d 40 58 fb    	vaddpd %zmm27,%zmm28,%zmm7
 2609b99:	62 c1 b5 48 5c c4    	vsubpd %zmm12,%zmm9,%zmm16
 2609b9f:	62 c1 b5 48 58 fc    	vaddpd %zmm12,%zmm9,%zmm23
 2609ba5:	62 31 dd 40 58 d9    	vaddpd %zmm17,%zmm20,%zmm11
 2609bab:	62 11 d5 40 58 e2    	vaddpd %zmm26,%zmm21,%zmm12
 2609bb1:	62 71 c5 48 59 c8    	vmulpd %zmm0,%zmm7,%zmm9
 2609bb7:	62 b1 dd 40 5c f9    	vsubpd %zmm17,%zmm20,%zmm7
 2609bbd:	62 e1 fd 48 28 c9    	vmovapd %zmm1,%zmm17
 2609bc3:	62 c2 fd 40 ac ce    	vfnmadd213pd %zmm14,%zmm16,%zmm17
 2609bc9:	62 c2 f5 48 a8 c6    	vfmadd213pd %zmm14,%zmm1,%zmm16
 2609bcf:	62 71 bd 48 59 c2    	vmulpd %zmm2,%zmm8,%zmm8
 2609bd5:	62 52 e5 48 ac e8    	vfnmadd213pd %zmm8,%zmm3,%zmm13
 2609bdb:	62 52 dd 48 a8 d0    	vfmadd213pd %zmm8,%zmm4,%zmm10
 2609be1:	62 e1 fd 48 28 e1    	vmovapd %zmm1,%zmm20
 2609be7:	62 a2 c5 40 ac e2    	vfnmadd213pd %zmm18,%zmm23,%zmm20
 2609bed:	62 a2 f5 48 a8 fa    	vfmadd213pd %zmm18,%zmm1,%zmm23
 2609bf3:	62 42 dd 48 a8 d9    	vfmadd213pd %zmm9,%zmm4,%zmm27
 2609bf9:	62 42 e5 48 a8 e1    	vfmadd213pd %zmm9,%zmm3,%zmm28
 2609bff:	62 71 fd 48 28 c1    	vmovapd %zmm1,%zmm8
 2609c05:	62 81 d5 40 5c d2    	vsubpd %zmm26,%zmm21,%zmm18
 2609c0b:	62 52 cd 48 ac c7    	vfnmadd213pd %zmm15,%zmm6,%zmm8
 2609c11:	62 d2 f5 48 a8 f7    	vfmadd213pd %zmm15,%zmm1,%zmm6
 2609c17:	62 71 fd 48 28 f9    	vmovapd %zmm1,%zmm15
 2609c1d:	62 32 cd 40 ac fb    	vfnmadd213pd %zmm19,%zmm22,%zmm15
 2609c23:	62 a2 f5 48 ae f3    	vfnmsub213pd %zmm19,%zmm1,%zmm22
 2609c29:	62 51 f5 40 58 cd    	vaddpd %zmm13,%zmm17,%zmm9
 2609c2f:	62 51 f5 40 5c f5    	vsubpd %zmm13,%zmm17,%zmm14
 2609c35:	62 51 fd 40 58 ea    	vaddpd %zmm10,%zmm16,%zmm13
 2609c3b:	62 51 fd 40 5c d2    	vsubpd %zmm10,%zmm16,%zmm10
 2609c41:	62 81 dd 40 58 cb    	vaddpd %zmm27,%zmm20,%zmm17
 2609c47:	62 81 c5 40 58 dc    	vaddpd %zmm28,%zmm23,%zmm19
 2609c4d:	62 81 c5 40 5c ec    	vsubpd %zmm28,%zmm23,%zmm21
 2609c53:	62 81 dd 40 5c e3    	vsubpd %zmm27,%zmm20,%zmm20
 2609c59:	62 c1 85 48 c6 c7 55 	vshufpd $0x55,%zmm15,%zmm15,%zmm16
 2609c60:	62 a1 cd 40 c6 f6 55 	vshufpd $0x55,%zmm22,%zmm22,%zmm22
 2609c67:	62 a1 ed 40 c6 d2 55 	vshufpd $0x55,%zmm18,%zmm18,%zmm18
 2609c6e:	62 31 f5 40 c6 f9 55 	vshufpd $0x55,%zmm17,%zmm17,%zmm15
 2609c75:	62 a1 dd 40 c6 cc 55 	vshufpd $0x55,%zmm20,%zmm20,%zmm17
 2609c7c:	62 a1 d5 40 c6 e5 55 	vshufpd $0x55,%zmm21,%zmm21,%zmm20
 2609c83:	62 c1 a5 48 58 ec    	vaddpd %zmm12,%zmm11,%zmm21
 2609c89:	62 c1 fd 48 28 f8    	vmovapd %zmm8,%zmm23
 2609c8f:	62 a2 d5 48 a7 f8    	vfmsubadd213pd %zmm16,%zmm5,%zmm23
 2609c95:	62 61 fd 48 28 c6    	vmovapd %zmm6,%zmm24
 2609c9b:	62 22 d5 48 a7 c6    	vfmsubadd213pd %zmm22,%zmm5,%zmm24
 2609ca1:	62 61 fd 48 28 cf    	vmovapd %zmm7,%zmm25
 2609ca7:	62 22 d5 48 a7 ca    	vfmsubadd213pd %zmm18,%zmm5,%zmm25
 2609cad:	62 41 fd 48 28 d1    	vmovapd %zmm9,%zmm26
 2609cb3:	62 42 d5 48 a7 d7    	vfmsubadd213pd %zmm15,%zmm5,%zmm26
 2609cb9:	62 a1 e5 40 c6 db 55 	vshufpd $0x55,%zmm19,%zmm19,%zmm19
 2609cc0:	62 41 fd 48 28 de    	vmovapd %zmm14,%zmm27
 2609cc6:	62 22 d5 48 a6 d9    	vfmaddsub213pd %zmm17,%zmm5,%zmm27
 2609ccc:	62 41 fd 48 28 e5    	vmovapd %zmm13,%zmm28
 2609cd2:	62 22 d5 48 a7 e3    	vfmsubadd213pd %zmm19,%zmm5,%zmm28
 2609cd8:	62 41 fd 48 28 ea    	vmovapd %zmm10,%zmm29
 2609cde:	62 e1 fd 49 11 2f    	vmovupd %zmm21,(%rdi){%k1}
 2609ce4:	62 21 fd 49 11 14 0f 	vmovupd %zmm26,(%rdi,%r9,1){%k1}
 2609ceb:	62 22 d5 48 a6 ec    	vfmaddsub213pd %zmm20,%zmm5,%zmm29
 2609cf1:	4c 01 cf             	add    %r9,%rdi
 2609cf4:	62 c1 fd 49 11 3c 39 	vmovupd %zmm23,(%r9,%rdi,1){%k1}
 2609cfb:	4c 01 cf             	add    %r9,%rdi
 2609cfe:	62 51 a5 48 5c dc    	vsubpd %zmm12,%zmm11,%zmm11
 2609d04:	62 41 fd 49 11 2c 39 	vmovupd %zmm29,(%r9,%rdi,1){%k1}
 2609d0b:	4c 01 cf             	add    %r9,%rdi
 2609d0e:	62 41 fd 49 11 0c 39 	vmovupd %zmm25,(%r9,%rdi,1){%k1}
 2609d15:	62 32 d5 48 a7 f1    	vfmsubadd213pd %zmm17,%zmm5,%zmm14
 2609d1b:	62 b2 d5 48 a6 f6    	vfmaddsub213pd %zmm22,%zmm5,%zmm6
 2609d21:	4c 01 cf             	add    %r9,%rdi
 2609d24:	62 41 fd 49 11 24 39 	vmovupd %zmm28,(%r9,%rdi,1){%k1}
 2609d2b:	4c 01 cf             	add    %r9,%rdi
 2609d2e:	62 32 d5 48 a6 eb    	vfmaddsub213pd %zmm19,%zmm5,%zmm13
 2609d34:	62 41 fd 49 11 04 39 	vmovupd %zmm24,(%r9,%rdi,1){%k1}
 2609d3b:	4c 01 cf             	add    %r9,%rdi
 2609d3e:	62 41 fd 49 11 1c 39 	vmovupd %zmm27,(%r9,%rdi,1){%k1}
 2609d45:	62 b2 d5 48 a6 fa    	vfmaddsub213pd %zmm18,%zmm5,%zmm7
 2609d4b:	62 32 d5 48 a7 d4    	vfmsubadd213pd %zmm20,%zmm5,%zmm10
 2609d51:	4c 01 cf             	add    %r9,%rdi
 2609d54:	62 51 fd 49 11 1c 39 	vmovupd %zmm11,(%r9,%rdi,1){%k1}
 2609d5b:	4c 01 cf             	add    %r9,%rdi
 2609d5e:	62 32 d5 48 a6 c0    	vfmaddsub213pd %zmm16,%zmm5,%zmm8
 2609d64:	62 51 fd 49 11 34 39 	vmovupd %zmm14,(%r9,%rdi,1){%k1}
 2609d6b:	4c 01 cf             	add    %r9,%rdi
 2609d6e:	62 d1 fd 49 11 34 39 	vmovupd %zmm6,(%r9,%rdi,1){%k1}
 2609d75:	62 52 d5 48 a6 cf    	vfmaddsub213pd %zmm15,%zmm5,%zmm9
 2609d7b:	4c 01 cf             	add    %r9,%rdi
 2609d7e:	62 51 fd 49 11 2c 39 	vmovupd %zmm13,(%r9,%rdi,1){%k1}
 2609d85:	4c 01 cf             	add    %r9,%rdi
 2609d88:	62 d1 fd 49 11 3c 39 	vmovupd %zmm7,(%r9,%rdi,1){%k1}
 2609d8f:	4c 01 cf             	add    %r9,%rdi
 2609d92:	62 51 fd 49 11 14 39 	vmovupd %zmm10,(%r9,%rdi,1){%k1}
 2609d99:	4c 01 cf             	add    %r9,%rdi
 2609d9c:	62 51 fd 49 11 04 39 	vmovupd %zmm8,(%r9,%rdi,1){%k1}
 2609da3:	4c 01 cf             	add    %r9,%rdi
 2609da6:	62 51 fd 49 11 0c 39 	vmovupd %zmm9,(%r9,%rdi,1){%k1}
 2609dad:	49 01 f6             	add    %rsi,%r14
 2609db0:	48 01 d7             	add    %rdx,%rdi
 2609db3:	48 ff c8             	dec    %rax
 2609db6:	0f 85 84 fc ff ff    	jne    2609a40 <mkl_dft_avx512_mg_colbatch_plain_fwd_16_d+0x540>
 2609dbc:	5b                   	pop    %rbx
 2609dbd:	41 5c                	pop    %r12
 2609dbf:	41 5d                	pop    %r13
 2609dc1:	41 5e                	pop    %r14
 2609dc3:	41 5f                	pop    %r15
 2609dc5:	5d                   	pop    %rbp
 2609dc6:	c5 f8 77             	vzeroupper
 2609dc9:	c3                   	ret
 2609dca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

