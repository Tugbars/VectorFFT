000000000259b920 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d>:
 259b920:	f3 0f 1e fa          	endbr64
 259b924:	53                   	push   %rbx
 259b925:	62 f1 fd 48 10 56 02 	vmovupd 0x80(%rsi),%zmm2
 259b92c:	b0 7f                	mov    $0x7f,%al
 259b92e:	62 f1 fd 48 10 5e 04 	vmovupd 0x100(%rsi),%zmm3
 259b935:	62 f1 fd 48 10 66 06 	vmovupd 0x180(%rsi),%zmm4
 259b93c:	c5 f9 92 c8          	kmovb  %eax,%k1
 259b940:	62 f1 fd c9 10 87 08 	vmovupd 0x8(%rdi),%zmm0{%k1}{z}
 259b947:	00 00 00 
 259b94a:	62 f1 fd c9 10 8f 88 	vmovupd 0x88(%rdi),%zmm1{%k1}{z}
 259b951:	00 00 00 
 259b954:	62 71 fd c9 10 87 08 	vmovupd 0x108(%rdi),%zmm8{%k1}{z}
 259b95b:	01 00 00 
 259b95e:	62 f1 fd c9 10 af 48 	vmovupd 0x48(%rdi),%zmm5{%k1}{z}
 259b965:	00 00 00 
 259b968:	62 f1 fd c9 10 b7 c8 	vmovupd 0xc8(%rdi),%zmm6{%k1}{z}
 259b96f:	00 00 00 
 259b972:	62 71 fd c9 10 8f 48 	vmovupd 0x148(%rdi),%zmm9{%k1}{z}
 259b979:	01 00 00 
 259b97c:	62 71 fd c9 10 1f    	vmovupd (%rdi),%zmm11{%k1}{z}
 259b982:	62 71 fd 48 10 16    	vmovupd (%rsi),%zmm10
 259b988:	62 f1 fd 48 10 7e 01 	vmovupd 0x40(%rsi),%zmm7
 259b98f:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
 259b994:	62 f1 ff 48 12 c0    	vmovddup %zmm0,%zmm0
 259b99a:	62 f1 ff 48 12 ed    	vmovddup %zmm5,%zmm5
 259b9a0:	62 f1 ff 48 12 c9    	vmovddup %zmm1,%zmm1
 259b9a6:	62 f1 ff 48 12 f6    	vmovddup %zmm6,%zmm6
 259b9ac:	62 c1 ff 48 12 d0    	vmovddup %zmm8,%zmm18
 259b9b2:	62 51 ff 48 12 c1    	vmovddup %zmm9,%zmm8
 259b9b8:	62 51 ff 48 12 eb    	vmovddup %zmm11,%zmm13
 259b9be:	62 71 fd c9 10 4f 01 	vmovupd 0x40(%rdi),%zmm9{%k1}{z}
 259b9c5:	62 71 fd c9 10 5f 02 	vmovupd 0x80(%rdi),%zmm11{%k1}{z}
 259b9cc:	62 51 ff 48 12 c9    	vmovddup %zmm9,%zmm9
 259b9d2:	62 51 ff 48 12 f3    	vmovddup %zmm11,%zmm14
 259b9d8:	62 71 fd c9 10 5f 03 	vmovupd 0xc0(%rdi),%zmm11{%k1}{z}
 259b9df:	62 71 fd c9 10 67 04 	vmovupd 0x100(%rdi),%zmm12{%k1}{z}
 259b9e6:	62 51 ff 48 12 db    	vmovddup %zmm11,%zmm11
 259b9ec:	62 51 ff 48 12 fc    	vmovddup %zmm12,%zmm15
 259b9f2:	62 71 fd c9 10 67 05 	vmovupd 0x140(%rdi),%zmm12{%k1}{z}
 259b9f9:	62 51 ff 48 12 e4    	vmovddup %zmm12,%zmm12
 259b9ff:	62 e1 ed 48 c6 c2 55 	vshufpd $0x55,%zmm2,%zmm2,%zmm16
 259ba06:	62 e1 fd 40 59 c0    	vmulpd %zmm0,%zmm16,%zmm16
 259ba0c:	62 f1 e5 48 c6 c3 55 	vshufpd $0x55,%zmm3,%zmm3,%zmm0
 259ba13:	62 e1 fd 48 59 c9    	vmulpd %zmm1,%zmm0,%zmm17
 259ba19:	62 f1 dd 48 c6 c4 55 	vshufpd $0x55,%zmm4,%zmm4,%zmm0
 259ba20:	62 a1 fd 48 59 d2    	vmulpd %zmm18,%zmm0,%zmm18
 259ba26:	4c 89 c9             	mov    %r9,%rcx
 259ba29:	48 c1 e1 04          	shl    $0x4,%rcx
 259ba2d:	4c 8d 52 40          	lea    0x40(%rdx),%r10
 259ba31:	31 c0                	xor    %eax,%eax
 259ba33:	62 f2 fd 48 19 05 c3 	vbroadcastsd 0x269b7c3(%rip),%zmm0        # 4c37200 <row_factorization_db+0x30d0>
 259ba3a:	b7 69 02 
 259ba3d:	62 f2 fd 48 19 0d 41 	vbroadcastsd 0x2650841(%rip),%zmm1        # 4bec288 <convolution.id+0x88>
 259ba44:	08 65 02 
 259ba47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
 259ba4e:	00 00 
 259ba50:	62 b2 85 48 a6 e2    	vfmaddsub213pd %zmm18,%zmm15,%zmm4
 259ba56:	62 b2 8d 48 a6 d9    	vfmaddsub213pd %zmm17,%zmm14,%zmm3
 259ba5c:	62 b2 95 48 a6 d0    	vfmaddsub213pd %zmm16,%zmm13,%zmm2
 259ba62:	4d 8d 1c 02          	lea    (%r10,%rax,1),%r11
 259ba66:	62 71 fd 48 10 6c c6 	vmovupd 0xc0(%rsi,%rax,8),%zmm13
 259ba6d:	03 
 259ba6e:	62 71 fd 48 10 74 c6 	vmovupd 0x140(%rsi,%rax,8),%zmm14
 259ba75:	05 
 259ba76:	62 71 fd 48 10 7c c6 	vmovupd 0x1c0(%rsi,%rax,8),%zmm15
 259ba7d:	07 
 259ba7e:	62 c1 95 48 c6 c5 55 	vshufpd $0x55,%zmm13,%zmm13,%zmm16
 259ba85:	62 b1 d5 48 59 e8    	vmulpd %zmm16,%zmm5,%zmm5
 259ba8b:	62 c1 8d 48 c6 c6 55 	vshufpd $0x55,%zmm14,%zmm14,%zmm16
 259ba92:	62 b1 cd 48 59 f0    	vmulpd %zmm16,%zmm6,%zmm6
 259ba98:	62 c1 85 48 c6 c7 55 	vshufpd $0x55,%zmm15,%zmm15,%zmm16
 259ba9f:	62 31 bd 48 59 c0    	vmulpd %zmm16,%zmm8,%zmm8
 259baa5:	62 e3 ad 48 1a c2 01 	vinsertf64x4 $0x1,%ymm2,%zmm10,%zmm16
 259baac:	62 f3 ad 48 23 d2 ee 	vshuff64x2 $0xee,%zmm2,%zmm10,%zmm2
 259bab3:	62 73 e5 48 1a d4 01 	vinsertf64x4 $0x1,%ymm4,%zmm3,%zmm10
 259baba:	62 f3 e5 48 23 dc ee 	vshuff64x2 $0xee,%zmm4,%zmm3,%zmm3
 259bac1:	62 d3 fd 40 23 e2 dd 	vshuff64x2 $0xdd,%zmm10,%zmm16,%zmm4
 259bac8:	62 53 fd 40 23 d2 88 	vshuff64x2 $0x88,%zmm10,%zmm16,%zmm10
 259bacf:	62 e3 ed 48 23 c3 88 	vshuff64x2 $0x88,%zmm3,%zmm2,%zmm16
 259bad6:	62 f3 ed 48 23 d3 dd 	vshuff64x2 $0xdd,%zmm3,%zmm2,%zmm2
 259badd:	62 d2 95 48 b6 e9    	vfmaddsub231pd %zmm9,%zmm13,%zmm5
 259bae3:	62 d2 8d 48 b6 f3    	vfmaddsub231pd %zmm11,%zmm14,%zmm6
 259bae9:	62 52 85 48 b6 c4    	vfmaddsub231pd %zmm12,%zmm15,%zmm8
 259baef:	62 f3 c5 48 1a dd 01 	vinsertf64x4 $0x1,%ymm5,%zmm7,%zmm3
 259baf6:	62 f3 c5 48 23 ed ee 	vshuff64x2 $0xee,%zmm5,%zmm7,%zmm5
 259bafd:	62 d3 cd 48 1a f8 01 	vinsertf64x4 $0x1,%ymm8,%zmm6,%zmm7
 259bb04:	62 d3 cd 48 23 f0 ee 	vshuff64x2 $0xee,%zmm8,%zmm6,%zmm6
 259bb0b:	62 73 e5 48 23 c7 dd 	vshuff64x2 $0xdd,%zmm7,%zmm3,%zmm8
 259bb12:	62 f3 e5 48 23 df 88 	vshuff64x2 $0x88,%zmm7,%zmm3,%zmm3
 259bb19:	62 f3 d5 48 23 fe 88 	vshuff64x2 $0x88,%zmm6,%zmm5,%zmm7
 259bb20:	62 f3 d5 48 23 ee dd 	vshuff64x2 $0xdd,%zmm6,%zmm5,%zmm5
 259bb27:	62 d1 dd 48 5c f0    	vsubpd %zmm8,%zmm4,%zmm6
 259bb2d:	62 71 ed 48 5c cd    	vsubpd %zmm5,%zmm2,%zmm9
 259bb33:	62 d1 dd 48 58 e0    	vaddpd %zmm8,%zmm4,%zmm4
 259bb39:	62 f1 ed 48 58 d5    	vaddpd %zmm5,%zmm2,%zmm2
 259bb3f:	62 f1 ad 48 58 eb    	vaddpd %zmm3,%zmm10,%zmm5
 259bb45:	62 71 fd 40 58 c7    	vaddpd %zmm7,%zmm16,%zmm8
 259bb4b:	62 f1 fd 40 5c ff    	vsubpd %zmm7,%zmm16,%zmm7
 259bb51:	62 f1 ad 48 5c db    	vsubpd %zmm3,%zmm10,%zmm3
 259bb57:	62 51 cd 48 58 d1    	vaddpd %zmm9,%zmm6,%zmm10
 259bb5d:	62 d1 cd 48 5c f1    	vsubpd %zmm9,%zmm6,%zmm6
 259bb63:	62 71 dd 48 5c ca    	vsubpd %zmm2,%zmm4,%zmm9
 259bb69:	62 f1 dd 48 58 d2    	vaddpd %zmm2,%zmm4,%zmm2
 259bb6f:	62 51 d5 48 58 d8    	vaddpd %zmm8,%zmm5,%zmm11
 259bb75:	62 d1 d5 48 5c e0    	vsubpd %zmm8,%zmm5,%zmm4
 259bb7b:	62 71 fd 48 28 c0    	vmovapd %zmm0,%zmm8
 259bb81:	62 72 ad 48 ac c7    	vfnmadd213pd %zmm7,%zmm10,%zmm8
 259bb87:	62 71 fd 48 28 e0    	vmovapd %zmm0,%zmm12
 259bb8d:	62 72 cd 48 ac e3    	vfnmadd213pd %zmm3,%zmm6,%zmm12
 259bb93:	62 72 fd 48 a8 d7    	vfmadd213pd %zmm7,%zmm0,%zmm10
 259bb99:	62 f2 fd 48 a8 f3    	vfmadd213pd %zmm3,%zmm0,%zmm6
 259bb9f:	62 d1 b5 48 c6 d9 55 	vshufpd $0x55,%zmm9,%zmm9,%zmm3
 259bba6:	62 f1 a5 48 58 fa    	vaddpd %zmm2,%zmm11,%zmm7
 259bbac:	62 f1 a5 48 5c d2    	vsubpd %zmm2,%zmm11,%zmm2
 259bbb2:	62 f1 fd 48 28 ec    	vmovapd %zmm4,%zmm5
 259bbb8:	62 f2 f5 48 a7 eb    	vfmsubadd213pd %zmm3,%zmm1,%zmm5
 259bbbe:	62 f2 f5 48 a6 e3    	vfmaddsub213pd %zmm3,%zmm1,%zmm4
 259bbc4:	62 d1 bd 48 c6 d8 55 	vshufpd $0x55,%zmm8,%zmm8,%zmm3
 259bbcb:	62 51 ad 48 c6 c2 55 	vshufpd $0x55,%zmm10,%zmm10,%zmm8
 259bbd2:	62 51 fd 48 28 cc    	vmovapd %zmm12,%zmm9
 259bbd8:	62 72 f5 48 a7 cb    	vfmsubadd213pd %zmm3,%zmm1,%zmm9
 259bbde:	62 71 fd 48 28 d6    	vmovapd %zmm6,%zmm10
 259bbe4:	62 52 f5 48 a6 d0    	vfmaddsub213pd %zmm8,%zmm1,%zmm10
 259bbea:	62 d1 fd 48 11 7c 02 	vmovupd %zmm7,-0x40(%r10,%rax,1)
 259bbf1:	ff 
 259bbf2:	62 d2 f5 48 a7 f0    	vfmsubadd213pd %zmm8,%zmm1,%zmm6
 259bbf8:	4a 8d 1c 19          	lea    (%rcx,%r11,1),%rbx
 259bbfc:	48 83 c3 c0          	add    $0xffffffffffffffc0,%rbx
 259bc00:	62 31 fd 48 11 4c 19 	vmovupd %zmm9,-0x40(%rcx,%r11,1)
 259bc07:	ff 
 259bc08:	62 f1 fd 48 11 2c 19 	vmovupd %zmm5,(%rcx,%rbx,1)
 259bc0f:	48 01 cb             	add    %rcx,%rbx
 259bc12:	62 71 fd 48 11 14 19 	vmovupd %zmm10,(%rcx,%rbx,1)
 259bc19:	48 01 cb             	add    %rcx,%rbx
 259bc1c:	62 f1 fd 48 11 14 19 	vmovupd %zmm2,(%rcx,%rbx,1)
 259bc23:	48 01 cb             	add    %rcx,%rbx
 259bc26:	62 f1 fd 48 11 34 19 	vmovupd %zmm6,(%rcx,%rbx,1)
 259bc2d:	48 01 cb             	add    %rcx,%rbx
 259bc30:	62 f1 fd 48 11 24 19 	vmovupd %zmm4,(%rcx,%rbx,1)
 259bc37:	62 72 f5 48 a6 e3    	vfmaddsub213pd %zmm3,%zmm1,%zmm12
 259bc3d:	48 01 cb             	add    %rcx,%rbx
 259bc40:	62 71 fd 48 11 24 19 	vmovupd %zmm12,(%rcx,%rbx,1)
 259bc47:	4d 8d 58 fc          	lea    -0x4(%r8),%r11
 259bc4b:	49 83 fb 03          	cmp    $0x3,%r11
 259bc4f:	0f 86 8b 01 00 00    	jbe    259bde0 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d+0x4c0>
 259bc55:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
 259bc59:	49 81 c0 88 01 00 00 	add    $0x188,%r8
 259bc60:	62 e1 fd 48 10 5c c6 	vmovupd 0x200(%rsi,%rax,8),%zmm19
 259bc67:	08 
 259bc68:	62 e1 fd 48 10 64 c6 	vmovupd 0x240(%rsi,%rax,8),%zmm20
 259bc6f:	09 
 259bc70:	62 d1 fd c9 10 28    	vmovupd (%r8),%zmm5{%k1}{z}
 259bc76:	62 d1 fd c9 10 70 01 	vmovupd 0x40(%r8),%zmm6{%k1}{z}
 259bc7d:	62 f1 fd 48 10 54 c6 	vmovupd 0x280(%rsi,%rax,8),%zmm2
 259bc84:	0a 
 259bc85:	62 f1 fd 48 10 5c c6 	vmovupd 0x300(%rsi,%rax,8),%zmm3
 259bc8c:	0c 
 259bc8d:	62 f1 fd 48 10 64 c6 	vmovupd 0x380(%rsi,%rax,8),%zmm4
 259bc94:	0e 
 259bc95:	62 d1 fd c9 10 78 02 	vmovupd 0x80(%r8),%zmm7{%k1}{z}
 259bc9c:	62 71 ff 48 12 d5    	vmovddup %zmm5,%zmm10
 259bca2:	62 e1 ff 48 12 c6    	vmovddup %zmm6,%zmm16
 259bca8:	62 e1 ff 48 12 cf    	vmovddup %zmm7,%zmm17
 259bcae:	62 d1 fd c9 10 68 03 	vmovupd 0xc0(%r8),%zmm5{%k1}{z}
 259bcb5:	62 d1 fd c9 10 70 04 	vmovupd 0x100(%r8),%zmm6{%k1}{z}
 259bcbc:	62 f1 ff 48 12 ed    	vmovddup %zmm5,%zmm5
 259bcc2:	62 e1 ff 48 12 d6    	vmovddup %zmm6,%zmm18
 259bcc8:	62 d1 fd c9 10 70 05 	vmovupd 0x140(%r8),%zmm6{%k1}{z}
 259bccf:	62 d1 fd c9 10 78 06 	vmovupd 0x180(%r8),%zmm7{%k1}{z}
 259bcd6:	62 f1 ff 48 12 f6    	vmovddup %zmm6,%zmm6
 259bcdc:	62 e1 ff 48 12 ef    	vmovddup %zmm7,%zmm21
 259bce2:	62 d1 fd c9 10 78 07 	vmovupd 0x1c0(%r8),%zmm7{%k1}{z}
 259bce9:	62 51 fd c9 10 88 f8 	vmovupd -0x8(%r8),%zmm9{%k1}{z}
 259bcf0:	ff ff ff 
 259bcf3:	62 71 ff 48 12 c7    	vmovddup %zmm7,%zmm8
 259bcf9:	62 c1 ff 48 12 f1    	vmovddup %zmm9,%zmm22
 259bcff:	62 d1 fd c9 10 b8 38 	vmovupd 0x38(%r8),%zmm7{%k1}{z}
 259bd06:	00 00 00 
 259bd09:	62 51 fd c9 10 88 78 	vmovupd 0x78(%r8),%zmm9{%k1}{z}
 259bd10:	00 00 00 
 259bd13:	62 e1 ff 48 12 ff    	vmovddup %zmm7,%zmm23
 259bd19:	62 51 ff 48 12 e9    	vmovddup %zmm9,%zmm13
 259bd1f:	62 d1 fd c9 10 b8 b8 	vmovupd 0xb8(%r8),%zmm7{%k1}{z}
 259bd26:	00 00 00 
 259bd29:	62 51 fd c9 10 98 f8 	vmovupd 0xf8(%r8),%zmm11{%k1}{z}
 259bd30:	00 00 00 
 259bd33:	62 71 ff 48 12 cf    	vmovddup %zmm7,%zmm9
 259bd39:	62 51 ff 48 12 f3    	vmovddup %zmm11,%zmm14
 259bd3f:	62 d1 fd c9 10 b8 38 	vmovupd 0x138(%r8),%zmm7{%k1}{z}
 259bd46:	01 00 00 
 259bd49:	62 51 fd c9 10 a0 78 	vmovupd 0x178(%r8),%zmm12{%k1}{z}
 259bd50:	01 00 00 
 259bd53:	62 71 ff 48 12 df    	vmovddup %zmm7,%zmm11
 259bd59:	62 51 ff 48 12 fc    	vmovddup %zmm12,%zmm15
 259bd5f:	62 d1 fd c9 10 b8 b8 	vmovupd 0x1b8(%r8),%zmm7{%k1}{z}
 259bd66:	01 00 00 
 259bd69:	62 71 ff 48 12 e7    	vmovddup %zmm7,%zmm12
 259bd6f:	62 b1 e5 40 c6 fb 55 	vshufpd $0x55,%zmm19,%zmm19,%zmm7
 259bd76:	62 51 c5 48 59 d2    	vmulpd %zmm10,%zmm7,%zmm10
 259bd7c:	62 b1 dd 40 c6 fc 55 	vshufpd $0x55,%zmm20,%zmm20,%zmm7
 259bd83:	62 b1 c5 48 59 f8    	vmulpd %zmm16,%zmm7,%zmm7
 259bd89:	62 e1 ed 48 c6 c2 55 	vshufpd $0x55,%zmm2,%zmm2,%zmm16
 259bd90:	62 a1 fd 40 59 c1    	vmulpd %zmm17,%zmm16,%zmm16
 259bd96:	62 e1 e5 48 c6 cb 55 	vshufpd $0x55,%zmm3,%zmm3,%zmm17
 259bd9d:	62 a1 f5 40 59 ca    	vmulpd %zmm18,%zmm17,%zmm17
 259bda3:	62 e1 dd 48 c6 d4 55 	vshufpd $0x55,%zmm4,%zmm4,%zmm18
 259bdaa:	62 a1 ed 40 59 d5    	vmulpd %zmm21,%zmm18,%zmm18
 259bdb0:	62 32 e5 40 b6 d6    	vfmaddsub231pd %zmm22,%zmm19,%zmm10
 259bdb6:	62 b2 dd 40 b6 ff    	vfmaddsub231pd %zmm23,%zmm20,%zmm7
 259bdbc:	48 83 c0 40          	add    $0x40,%rax
 259bdc0:	4d 89 d8             	mov    %r11,%r8
 259bdc3:	e9 88 fc ff ff       	jmp    259ba50 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d+0x130>
 259bdc8:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
 259bdcf:	0f 1f 84 00 00 00 00 
 259bdd6:	00 
 259bdd7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
 259bdde:	00 00 
 259bde0:	49 83 f8 04          	cmp    $0x4,%r8
 259bde4:	0f 84 cc 04 00 00    	je     259c2b6 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d+0x996>
 259bdea:	62 f1 fd 48 10 74 c6 	vmovupd 0x200(%rsi,%rax,8),%zmm6
 259bdf1:	08 
 259bdf2:	62 f1 fd 48 10 7c c6 	vmovupd 0x240(%rsi,%rax,8),%zmm7
 259bdf9:	09 
 259bdfa:	62 f1 fd c9 10 94 c7 	vmovupd 0x188(%rdi,%rax,8),%zmm2{%k1}{z}
 259be01:	88 01 00 00 
 259be05:	62 f1 ff 48 12 d2    	vmovddup %zmm2,%zmm2
 259be0b:	62 f1 fd c9 10 9c c7 	vmovupd 0x1c8(%rdi,%rax,8),%zmm3{%k1}{z}
 259be12:	c8 01 00 00 
 259be16:	62 71 fd c9 10 44 c7 	vmovupd 0x180(%rdi,%rax,8),%zmm8{%k1}{z}
 259be1d:	06 
 259be1e:	62 71 ff 48 12 cb    	vmovddup %zmm3,%zmm9
 259be24:	62 51 ff 48 12 c0    	vmovddup %zmm8,%zmm8
 259be2a:	62 f1 fd c9 10 5c c7 	vmovupd 0x1c0(%rdi,%rax,8),%zmm3{%k1}{z}
 259be31:	07 
 259be32:	62 71 ff 48 12 d3    	vmovddup %zmm3,%zmm10
 259be38:	62 f1 cd 48 c6 de 55 	vshufpd $0x55,%zmm6,%zmm6,%zmm3
 259be3f:	62 f1 e5 48 59 da    	vmulpd %zmm2,%zmm3,%zmm3
 259be45:	62 f1 c5 48 c6 d7 55 	vshufpd $0x55,%zmm7,%zmm7,%zmm2
 259be4c:	62 d1 ed 48 59 d1    	vmulpd %zmm9,%zmm2,%zmm2
 259be52:	62 d2 cd 48 b6 d8    	vfmaddsub231pd %zmm8,%zmm6,%zmm3
 259be58:	62 d2 c5 48 b6 d2    	vfmaddsub231pd %zmm10,%zmm7,%zmm2
 259be5e:	49 83 f8 06          	cmp    $0x6,%r8
 259be62:	74 7e                	je     259bee2 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d+0x5c2>
 259be64:	49 83 f8 07          	cmp    $0x7,%r8
 259be68:	0f 85 52 04 00 00    	jne    259c2c0 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d+0x9a0>
 259be6e:	62 f1 fd 48 10 74 c6 	vmovupd 0x300(%rsi,%rax,8),%zmm6
 259be75:	0c 
 259be76:	62 f1 fd 48 10 7c c6 	vmovupd 0x340(%rsi,%rax,8),%zmm7
 259be7d:	0d 
 259be7e:	62 f1 fd c9 10 a4 c7 	vmovupd 0x288(%rdi,%rax,8),%zmm4{%k1}{z}
 259be85:	88 02 00 00 
 259be89:	62 f1 ff 48 12 e4    	vmovddup %zmm4,%zmm4
 259be8f:	62 f1 fd c9 10 ac c7 	vmovupd 0x2c8(%rdi,%rax,8),%zmm5{%k1}{z}
 259be96:	c8 02 00 00 
 259be9a:	62 71 ff 48 12 c5    	vmovddup %zmm5,%zmm8
 259bea0:	62 f1 fd c9 10 6c c7 	vmovupd 0x280(%rdi,%rax,8),%zmm5{%k1}{z}
 259bea7:	0a 
 259bea8:	62 71 ff 48 12 cd    	vmovddup %zmm5,%zmm9
 259beae:	62 f1 fd c9 10 6c c7 	vmovupd 0x2c0(%rdi,%rax,8),%zmm5{%k1}{z}
 259beb5:	0b 
 259beb6:	62 71 ff 48 12 d5    	vmovddup %zmm5,%zmm10
 259bebc:	62 f1 cd 48 c6 ee 55 	vshufpd $0x55,%zmm6,%zmm6,%zmm5
 259bec3:	62 f1 d5 48 59 ec    	vmulpd %zmm4,%zmm5,%zmm5
 259bec9:	62 f1 c5 48 c6 e7 55 	vshufpd $0x55,%zmm7,%zmm7,%zmm4
 259bed0:	62 d1 dd 48 59 e0    	vmulpd %zmm8,%zmm4,%zmm4
 259bed6:	62 d2 cd 48 b6 e9    	vfmaddsub231pd %zmm9,%zmm6,%zmm5
 259bedc:	62 d2 c5 48 b6 e2    	vfmaddsub231pd %zmm10,%zmm7,%zmm4
 259bee2:	62 f1 fd 48 10 74 c6 	vmovupd 0x280(%rsi,%rax,8),%zmm6
 259bee9:	0a 
 259beea:	62 71 fd 48 10 44 c6 	vmovupd 0x2c0(%rsi,%rax,8),%zmm8
 259bef1:	0b 
 259bef2:	62 f1 fd c9 10 bc c7 	vmovupd 0x208(%rdi,%rax,8),%zmm7{%k1}{z}
 259bef9:	08 02 00 00 
 259befd:	62 f1 ff 48 12 ff    	vmovddup %zmm7,%zmm7
 259bf03:	62 71 fd c9 10 8c c7 	vmovupd 0x248(%rdi,%rax,8),%zmm9{%k1}{z}
 259bf0a:	48 02 00 00 
 259bf0e:	62 51 ff 48 12 c9    	vmovddup %zmm9,%zmm9
 259bf14:	62 71 fd c9 10 54 c7 	vmovupd 0x200(%rdi,%rax,8),%zmm10{%k1}{z}
 259bf1b:	08 
 259bf1c:	62 51 ff 48 12 d2    	vmovddup %zmm10,%zmm10
 259bf22:	62 71 fd c9 10 5c c7 	vmovupd 0x240(%rdi,%rax,8),%zmm11{%k1}{z}
 259bf29:	09 
 259bf2a:	62 51 ff 48 12 db    	vmovddup %zmm11,%zmm11
 259bf30:	62 71 cd 48 c6 e6 55 	vshufpd $0x55,%zmm6,%zmm6,%zmm12
 259bf37:	62 71 9d 48 59 e7    	vmulpd %zmm7,%zmm12,%zmm12
 259bf3d:	62 d1 bd 48 c6 f8 55 	vshufpd $0x55,%zmm8,%zmm8,%zmm7
 259bf44:	62 d1 c5 48 59 f9    	vmulpd %zmm9,%zmm7,%zmm7
 259bf4a:	62 52 cd 48 b6 e2    	vfmaddsub231pd %zmm10,%zmm6,%zmm12
 259bf50:	62 d2 bd 48 b6 fb    	vfmaddsub231pd %zmm11,%zmm8,%zmm7
 259bf56:	62 d3 e5 48 1a f4 01 	vinsertf64x4 $0x1,%ymm12,%zmm3,%zmm6
 259bf5d:	62 d3 e5 48 23 dc ee 	vshuff64x2 $0xee,%zmm12,%zmm3,%zmm3
 259bf64:	49 83 f8 06          	cmp    $0x6,%r8
 259bf68:	0f 85 92 01 00 00    	jne    259c100 <mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d+0x7e0>
 259bf6e:	62 f3 ed 48 1a e7 01 	vinsertf64x4 $0x1,%ymm7,%zmm2,%zmm4
 259bf75:	62 f3 ed 48 23 d7 ee 	vshuff64x2 $0xee,%zmm7,%zmm2,%zmm2
 259bf7c:	c5 d1 57 ed          	vxorpd %xmm5,%xmm5,%xmm5
 259bf80:	62 f3 cd 48 23 fd dd 	vshuff64x2 $0xdd,%zmm5,%zmm6,%zmm7
 259bf87:	62 73 e5 48 23 c5 dd 	vshuff64x2 $0xdd,%zmm5,%zmm3,%zmm8
 259bf8e:	62 73 dd 48 23 cd dd 	vshuff64x2 $0xdd,%zmm5,%zmm4,%zmm9
 259bf95:	62 73 ed 48 23 d5 dd 	vshuff64x2 $0xdd,%zmm5,%zmm2,%zmm10
 259bf9c:	62 51 c5 48 5c d9    	vsubpd %zmm9,%zmm7,%zmm11
 259bfa2:	62 51 bd 48 5c e2    	vsubpd %zmm10,%zmm8,%zmm12
 259bfa8:	62 f3 e5 48 23 dd a8 	vshuff64x2 $0xa8,%zmm5,%zmm3,%zmm3
 259bfaf:	62 f3 ed 48 23 d5 a8 	vshuff64x2 $0xa8,%zmm5,%zmm2,%zmm2
 259bfb6:	62 f3 cd 48 23 f5 a8 	vshuff64x2 $0xa8,%zmm5,%zmm6,%zmm6
 259bfbd:	62 f3 dd 48 23 e5 a8 	vshuff64x2 $0xa8,%zmm5,%zmm4,%zmm4
 259bfc4:	62 d1 c5 48 58 e9    	vaddpd %zmm9,%zmm7,%zmm5
 259bfca:	62 d1 bd 48 58 fa    	vaddpd %zmm10,%zmm8,%zmm7
 259bfd0:	62 71 e5 48 5c c2    	vsubpd %zmm2,%zmm3,%zmm8
 259bfd6:	62 51 a5 48 58 cc    	vaddpd %zmm12,%zmm11,%zmm9
 259bfdc:	62 71 cd 48 58 d4    	vaddpd %zmm4,%zmm6,%zmm10
 259bfe2:	62 71 e5 48 58 ea    	vaddpd %zmm2,%zmm3,%zmm13
 259bfe8:	62 f1 cd 48 5c e4    	vsubpd %zmm4,%zmm6,%zmm4
 259bfee:	62 d1 a5 48 5c f4    	vsubpd %zmm12,%zmm11,%zmm6
 259bff4:	62 71 fd 48 28 d8    	vmovapd %zmm0,%zmm11
 259bffa:	62 71 d5 48 5c e7    	vsubpd %zmm7,%zmm5,%zmm12
 259c000:	62 52 b5 48 ac d8    	vfnmadd213pd %zmm8,%zmm9,%zmm11
 259c006:	62 52 fd 48 a8 c8    	vfmadd213pd %zmm8,%zmm0,%zmm9
 259c00c:	62 f1 fd 48 28 d0    	vmovapd %zmm0,%zmm2
 259c012:	62 d1 ad 48 5c dd    	vsubpd %zmm13,%zmm10,%zmm3
 259c018:	c5 d5 58 ef          	vaddpd %ymm7,%ymm5,%ymm5
 259c01c:	c4 c1 2d 58 fd       	vaddpd %ymm13,%ymm10,%ymm7
 259c021:	62 f2 cd 48 ac d4    	vfnmadd213pd %zmm4,%zmm6,%zmm2
 259c027:	62 f2 cd 48 a8 c4    	vfmadd213pd %zmm4,%zmm6,%zmm0
 259c02d:	62 d1 9d 48 c6 e4 55 	vshufpd $0x55,%zmm12,%zmm12,%zmm4
 259c034:	62 d1 a5 48 c6 f3 55 	vshufpd $0x55,%zmm11,%zmm11,%zmm6
 259c03b:	62 51 b5 48 c6 c1 55 	vshufpd $0x55,%zmm9,%zmm9,%zmm8
 259c042:	62 71 fd 48 28 cb    	vmovapd %zmm3,%zmm9
 259c048:	c5 45 58 d5          	vaddpd %ymm5,%ymm7,%ymm10
 259c04c:	c5 c5 5c ed          	vsubpd %ymm5,%ymm7,%ymm5
 259c050:	62 72 f5 48 a7 cc    	vfmsubadd213pd %zmm4,%zmm1,%zmm9
 259c056:	62 f2 f5 48 a6 dc    	vfmaddsub213pd %zmm4,%zmm1,%zmm3
 259c05c:	62 f1 fd 48 28 e2    	vmovapd %zmm2,%zmm4
 259c062:	62 f1 fd 48 28 f8    	vmovapd %zmm0,%zmm7
 259c068:	62 f2 f5 48 a7 e6    	vfmsubadd213pd %zmm6,%zmm1,%zmm4
 259c06e:	62 d2 f5 48 a6 f8    	vfmaddsub213pd %zmm8,%zmm1,%zmm7
 259c074:	62 d2 f5 48 a7 c0    	vfmsubadd213pd %zmm8,%zmm1,%zmm0
 259c07a:	62 f2 f5 48 a6 d6    	vfmaddsub213pd %zmm6,%zmm1,%zmm2
 259c080:	48 8d 34 02          	lea    (%rdx,%rax,1),%rsi
 259c084:	c5 7d 11 54 02 40    	vmovupd %ymm10,0x40(%rdx,%rax,1)
 259c08a:	4c 89 c8             	mov    %r9,%rax
 259c08d:	48 c1 e0 05          	shl    $0x5,%rax
 259c091:	4b 8d 14 49          	lea    (%r9,%r9,2),%rdx
 259c095:	c5 fd 11 64 31 40    	vmovupd %ymm4,0x40(%rcx,%rsi,1)
 259c09b:	4c 89 c9             	mov    %r9,%rcx
 259c09e:	48 c1 e1 06          	shl    $0x6,%rcx
 259c0a2:	c5 7d 11 4c 30 40    	vmovupd %ymm9,0x40(%rax,%rsi,1)
 259c0a8:	48 89 d0             	mov    %rdx,%rax
 259c0ab:	48 c1 e0 04          	shl    $0x4,%rax
 259c0af:	c5 fd 11 7c 30 40    	vmovupd %ymm7,0x40(%rax,%rsi,1)
 259c0b5:	4b 8d 04 89          	lea    (%r9,%r9,4),%rax
 259c0b9:	48 c1 e0 04          	shl    $0x4,%rax
 259c0bd:	c5 fd 11 6c 31 40    	vmovupd %ymm5,0x40(%rcx,%rsi,1)
 259c0c3:	c5 fd 11 44 30 40    	vmovupd %ymm0,0x40(%rax,%rsi,1)
 259c0c9:	48 c1 e2 05          	shl    $0x5,%rdx
 259c0cd:	c5 fd 11 5c 32 40    	vmovupd %ymm3,0x40(%rdx,%rsi,1)
 259c0d3:	49 6b c1 70          	imul   $0x70,%r9,%rax
 259c0d7:	c5 fd 11 54 30 40    	vmovupd %ymm2,0x40(%rax,%rsi,1)
 259c0dd:	5b                   	pop    %rbx
 259c0de:	c5 f8 77             	vzeroupper
 259c0e1:	c3                   	ret
 259c0e2:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
 259c0e9:	0f 1f 84 00 00 00 00 
 259c0f0:	00 
 259c0f1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
 259c0f8:	0f 1f 84 00 00 00 00 
 259c0ff:	00 
 259c100:	62 d3 fd 48 1b e8 01 	vextractf64x4 $0x1,%zmm5,%ymm8
 259c107:	62 73 ed 48 1a cf 01 	vinsertf64x4 $0x1,%ymm7,%zmm2,%zmm9
 259c10e:	62 f3 ed 48 23 d7 ee 	vshuff64x2 $0xee,%zmm7,%zmm2,%zmm2
 259c115:	62 f3 fd 48 1b e7 01 	vextractf64x4 $0x1,%zmm4,%ymm7
 259c11c:	c5 fd 28 ed          	vmovapd %ymm5,%ymm5
 259c120:	c5 fd 28 e4          	vmovapd %ymm4,%ymm4
 259c124:	62 73 cd 48 23 d5 dd 	vshuff64x2 $0xdd,%zmm5,%zmm6,%zmm10
 259c12b:	62 53 e5 48 23 d8 dd 	vshuff64x2 $0xdd,%zmm8,%zmm3,%zmm11
 259c132:	62 f3 cd 48 23 ed 88 	vshuff64x2 $0x88,%zmm5,%zmm6,%zmm5
 259c139:	62 f3 b5 48 23 f4 dd 	vshuff64x2 $0xdd,%zmm4,%zmm9,%zmm6
 259c140:	62 d3 e5 48 23 d8 88 	vshuff64x2 $0x88,%zmm8,%zmm3,%zmm3
 259c147:	62 73 ed 48 23 c7 dd 	vshuff64x2 $0xdd,%zmm7,%zmm2,%zmm8
 259c14e:	62 f3 b5 48 23 e4 88 	vshuff64x2 $0x88,%zmm4,%zmm9,%zmm4
 259c155:	62 71 ad 48 5c ce    	vsubpd %zmm6,%zmm10,%zmm9
 259c15b:	62 f3 ed 48 23 d7 88 	vshuff64x2 $0x88,%zmm7,%zmm2,%zmm2
 259c162:	62 d1 a5 48 5c f8    	vsubpd %zmm8,%zmm11,%zmm7
 259c168:	62 f1 ad 48 58 f6    	vaddpd %zmm6,%zmm10,%zmm6
 259c16e:	62 51 a5 48 58 c0    	vaddpd %zmm8,%zmm11,%zmm8
 259c174:	62 71 e5 48 5c d2    	vsubpd %zmm2,%zmm3,%zmm10
 259c17a:	62 71 b5 48 58 df    	vaddpd %zmm7,%zmm9,%zmm11
 259c180:	62 f1 e5 48 58 da    	vaddpd %zmm2,%zmm3,%zmm3
 259c186:	62 71 d5 48 58 e4    	vaddpd %zmm4,%zmm5,%zmm12
 259c18c:	62 f1 d5 48 5c e4    	vsubpd %zmm4,%zmm5,%zmm4
 259c192:	62 f1 b5 48 5c ef    	vsubpd %zmm7,%zmm9,%zmm5
 259c198:	62 f1 fd 48 28 f8    	vmovapd %zmm0,%zmm7
 259c19e:	62 51 cd 48 5c c8    	vsubpd %zmm8,%zmm6,%zmm9
 259c1a4:	62 d2 a5 48 ac fa    	vfnmadd213pd %zmm10,%zmm11,%zmm7
 259c1aa:	62 52 fd 48 a8 da    	vfmadd213pd %zmm10,%zmm0,%zmm11
 259c1b0:	62 f1 fd 48 28 d0    	vmovapd %zmm0,%zmm2
 259c1b6:	62 d1 cd 48 58 f0    	vaddpd %zmm8,%zmm6,%zmm6
 259c1bc:	62 71 9d 48 58 c3    	vaddpd %zmm3,%zmm12,%zmm8
 259c1c2:	62 f1 9d 48 5c db    	vsubpd %zmm3,%zmm12,%zmm3
 259c1c8:	62 f2 d5 48 ac d4    	vfnmadd213pd %zmm4,%zmm5,%zmm2
 259c1ce:	62 f2 d5 48 a8 c4    	vfmadd213pd %zmm4,%zmm5,%zmm0
 259c1d4:	48 8d 34 02          	lea    (%rdx,%rax,1),%rsi
 259c1d8:	48 83 c6 40          	add    $0x40,%rsi
 259c1dc:	40 b7 3f             	mov    $0x3f,%dil
 259c1df:	48 01 d1             	add    %rdx,%rcx
 259c1e2:	c5 f9 92 cf          	kmovb  %edi,%k1
 259c1e6:	4c 89 cf             	mov    %r9,%rdi
 259c1e9:	62 d1 b5 48 c6 e1 55 	vshufpd $0x55,%zmm9,%zmm9,%zmm4
 259c1f0:	62 f1 bd 48 58 ee    	vaddpd %zmm6,%zmm8,%zmm5
 259c1f6:	62 f1 bd 48 5c f6    	vsubpd %zmm6,%zmm8,%zmm6
 259c1fc:	62 71 fd 48 28 c3    	vmovapd %zmm3,%zmm8
 259c202:	62 f1 c5 48 c6 ff 55 	vshufpd $0x55,%zmm7,%zmm7,%zmm7
 259c209:	62 51 a5 48 c6 cb 55 	vshufpd $0x55,%zmm11,%zmm11,%zmm9
 259c210:	62 72 f5 48 a7 c4    	vfmsubadd213pd %zmm4,%zmm1,%zmm8
 259c216:	62 f2 f5 48 a6 dc    	vfmaddsub213pd %zmm4,%zmm1,%zmm3
 259c21c:	62 f1 fd 48 28 e2    	vmovapd %zmm2,%zmm4
 259c222:	62 71 fd 48 28 d0    	vmovapd %zmm0,%zmm10
 259c228:	62 f2 f5 48 a7 e7    	vfmsubadd213pd %zmm7,%zmm1,%zmm4
 259c22e:	62 52 f5 48 a6 d1    	vfmaddsub213pd %zmm9,%zmm1,%zmm10
 259c234:	62 d2 f5 48 a7 c1    	vfmsubadd213pd %zmm9,%zmm1,%zmm0
 259c23a:	62 f2 f5 48 a6 d7    	vfmaddsub213pd %zmm7,%zmm1,%zmm2
 259c240:	48 c1 e7 05          	shl    $0x5,%rdi
 259c244:	62 f1 fd 49 11 2e    	vmovupd %zmm5,(%rsi){%k1}
 259c24a:	4b 8d 34 49          	lea    (%r9,%r9,2),%rsi
 259c24e:	62 f1 fd 49 11 64 08 	vmovupd %zmm4,0x40(%rax,%rcx,1){%k1}
 259c255:	01 
 259c256:	4c 89 c9             	mov    %r9,%rcx
 259c259:	48 c1 e1 06          	shl    $0x6,%rcx
 259c25d:	48 01 d7             	add    %rdx,%rdi
 259c260:	62 71 fd 49 11 44 38 	vmovupd %zmm8,0x40(%rax,%rdi,1){%k1}
 259c267:	01 
 259c268:	48 89 f7             	mov    %rsi,%rdi
 259c26b:	48 c1 e7 04          	shl    $0x4,%rdi
 259c26f:	48 01 d7             	add    %rdx,%rdi
 259c272:	62 71 fd 49 11 54 38 	vmovupd %zmm10,0x40(%rax,%rdi,1){%k1}
 259c279:	01 
 259c27a:	4b 8d 3c 89          	lea    (%r9,%r9,4),%rdi
 259c27e:	48 01 d1             	add    %rdx,%rcx
 259c281:	48 c1 e7 04          	shl    $0x4,%rdi
 259c285:	48 c1 e6 05          	shl    $0x5,%rsi
 259c289:	48 01 d7             	add    %rdx,%rdi
 259c28c:	48 01 d6             	add    %rdx,%rsi
 259c28f:	62 f1 fd 49 11 74 08 	vmovupd %zmm6,0x40(%rax,%rcx,1){%k1}
 259c296:	01 
 259c297:	62 f1 fd 49 11 44 38 	vmovupd %zmm0,0x40(%rax,%rdi,1){%k1}
 259c29e:	01 
 259c29f:	62 f1 fd 49 11 5c 30 	vmovupd %zmm3,0x40(%rax,%rsi,1){%k1}
 259c2a6:	01 
 259c2a7:	49 6b c9 70          	imul   $0x70,%r9,%rcx
 259c2ab:	48 01 d1             	add    %rdx,%rcx
 259c2ae:	62 f1 fd 49 11 54 08 	vmovupd %zmm2,0x40(%rax,%rcx,1){%k1}
 259c2b5:	01 
 259c2b6:	5b                   	pop    %rbx
 259c2b7:	c5 f8 77             	vzeroupper
 259c2ba:	c3                   	ret
 259c2bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
 259c2c0:	62 f3 fd 48 1b dc 01 	vextractf64x4 $0x1,%zmm3,%ymm4
 259c2c7:	62 f3 fd 48 1b d5 01 	vextractf64x4 $0x1,%zmm2,%ymm5
 259c2ce:	c5 fd 28 f3          	vmovapd %ymm3,%ymm6
 259c2d2:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
 259c2d6:	c5 7d 28 c2          	vmovapd %ymm2,%ymm8
 259c2da:	62 f3 cd 48 23 f7 dd 	vshuff64x2 $0xdd,%zmm7,%zmm6,%zmm6
 259c2e1:	62 73 bd 48 23 c7 dd 	vshuff64x2 $0xdd,%zmm7,%zmm8,%zmm8
 259c2e8:	62 73 dd 48 23 cf dd 	vshuff64x2 $0xdd,%zmm7,%zmm4,%zmm9
 259c2ef:	62 f3 d5 48 23 ff dd 	vshuff64x2 $0xdd,%zmm7,%zmm5,%zmm7
 259c2f6:	62 51 cd 48 5c d0    	vsubpd %zmm8,%zmm6,%zmm10
 259c2fc:	62 71 b5 48 5c df    	vsubpd %zmm7,%zmm9,%zmm11
 259c302:	c5 79 28 e4          	vmovapd %xmm4,%xmm12
 259c306:	c5 79 28 ed          	vmovapd %xmm5,%xmm13
 259c30a:	c5 79 28 f3          	vmovapd %xmm3,%xmm14
 259c30e:	c5 79 28 fa          	vmovapd %xmm2,%xmm15
 259c312:	62 d1 cd 48 58 f0    	vaddpd %zmm8,%zmm6,%zmm6
 259c318:	62 51 9d 48 5c c5    	vsubpd %zmm13,%zmm12,%zmm8
 259c31e:	62 f1 b5 48 58 ff    	vaddpd %zmm7,%zmm9,%zmm7
 259c324:	62 51 ad 48 58 cb    	vaddpd %zmm11,%zmm10,%zmm9
 259c32a:	c5 e1 58 d2          	vaddpd %xmm2,%xmm3,%xmm2
 259c32e:	62 d1 8d 48 58 df    	vaddpd %zmm15,%zmm14,%zmm3
 259c334:	62 51 8d 48 5c f7    	vsubpd %zmm15,%zmm14,%zmm14
 259c33a:	c5 d9 58 e5          	vaddpd %xmm5,%xmm4,%xmm4
 259c33e:	62 d1 9d 48 58 ed    	vaddpd %zmm13,%zmm12,%zmm5
 259c344:	62 51 ad 48 5c d3    	vsubpd %zmm11,%zmm10,%zmm10
 259c34a:	62 71 fd 48 28 d8    	vmovapd %zmm0,%zmm11
 259c350:	62 71 cd 48 5c e7    	vsubpd %zmm7,%zmm6,%zmm12
 259c356:	62 52 b5 48 ac d8    	vfnmadd213pd %zmm8,%zmm9,%zmm11
 259c35c:	62 52 fd 48 a8 c8    	vfmadd213pd %zmm8,%zmm0,%zmm9
 259c362:	62 71 fd 48 28 c0    	vmovapd %zmm0,%zmm8
 259c368:	c5 e9 58 d4          	vaddpd %xmm4,%xmm2,%xmm2
 259c36c:	62 f1 e5 48 5c dd    	vsubpd %zmm5,%zmm3,%zmm3
 259c372:	c5 c9 58 e7          	vaddpd %xmm7,%xmm6,%xmm4
 259c376:	62 52 ad 48 ac c6    	vfnmadd213pd %zmm14,%zmm10,%zmm8
 259c37c:	62 d2 ad 48 a8 c6    	vfmadd213pd %zmm14,%zmm10,%zmm0
 259c382:	62 d1 9d 48 c6 ec 55 	vshufpd $0x55,%zmm12,%zmm12,%zmm5
 259c389:	62 d1 a5 48 c6 f3 55 	vshufpd $0x55,%zmm11,%zmm11,%zmm6
 259c390:	62 d1 b5 48 c6 f9 55 	vshufpd $0x55,%zmm9,%zmm9,%zmm7
 259c397:	62 71 fd 48 28 cb    	vmovapd %zmm3,%zmm9
 259c39d:	c5 69 58 d4          	vaddpd %xmm4,%xmm2,%xmm10
 259c3a1:	c5 e9 5c d4          	vsubpd %xmm4,%xmm2,%xmm2
 259c3a5:	62 72 f5 48 a7 cd    	vfmsubadd213pd %zmm5,%zmm1,%zmm9
 259c3ab:	62 f2 f5 48 a6 dd    	vfmaddsub213pd %zmm5,%zmm1,%zmm3
 259c3b1:	62 d1 fd 48 28 e0    	vmovapd %zmm8,%zmm4
 259c3b7:	62 f1 fd 48 28 e8    	vmovapd %zmm0,%zmm5
 259c3bd:	62 f2 f5 48 a7 e6    	vfmsubadd213pd %zmm6,%zmm1,%zmm4
 259c3c3:	62 f2 f5 48 a6 ef    	vfmaddsub213pd %zmm7,%zmm1,%zmm5
 259c3c9:	62 f2 f5 48 a7 c7    	vfmsubadd213pd %zmm7,%zmm1,%zmm0
 259c3cf:	62 72 f5 48 a6 c6    	vfmaddsub213pd %zmm6,%zmm1,%zmm8
 259c3d5:	48 8d 34 02          	lea    (%rdx,%rax,1),%rsi
 259c3d9:	c5 79 11 54 02 40    	vmovupd %xmm10,0x40(%rdx,%rax,1)
 259c3df:	4c 89 c8             	mov    %r9,%rax
 259c3e2:	48 c1 e0 05          	shl    $0x5,%rax
 259c3e6:	4b 8d 14 49          	lea    (%r9,%r9,2),%rdx
 259c3ea:	c5 f9 11 64 31 40    	vmovupd %xmm4,0x40(%rcx,%rsi,1)
 259c3f0:	4c 89 c9             	mov    %r9,%rcx
 259c3f3:	48 c1 e1 06          	shl    $0x6,%rcx
 259c3f7:	c5 79 11 4c 30 40    	vmovupd %xmm9,0x40(%rax,%rsi,1)
 259c3fd:	48 89 d0             	mov    %rdx,%rax
 259c400:	48 c1 e0 04          	shl    $0x4,%rax
 259c404:	c5 f9 11 6c 30 40    	vmovupd %xmm5,0x40(%rax,%rsi,1)
 259c40a:	4b 8d 04 89          	lea    (%r9,%r9,4),%rax
 259c40e:	48 c1 e0 04          	shl    $0x4,%rax
 259c412:	c5 f9 11 54 31 40    	vmovupd %xmm2,0x40(%rcx,%rsi,1)
 259c418:	c5 f9 11 44 30 40    	vmovupd %xmm0,0x40(%rax,%rsi,1)
 259c41e:	48 c1 e2 05          	shl    $0x5,%rdx
 259c422:	c5 f9 11 5c 32 40    	vmovupd %xmm3,0x40(%rdx,%rsi,1)
 259c428:	49 6b c1 70          	imul   $0x70,%r9,%rax
 259c42c:	c5 79 11 44 30 40    	vmovupd %xmm8,0x40(%rax,%rsi,1)
 259c432:	5b                   	pop    %rbx
 259c433:	c5 f8 77             	vzeroupper
 259c436:	c3                   	ret
 259c437:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
 259c43e:	00 00 

