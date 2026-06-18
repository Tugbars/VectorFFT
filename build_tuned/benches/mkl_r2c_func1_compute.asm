
C:/Program Files (x86)/Intel/oneAPI/mkl/latest/bin/mkl_avx2.2.dll:     file format pei-x86-64


Disassembly of section IPPCODE:

00000001825c3800 <IPPCODE+0x7800>:
   1825c3800:	f3 0f 1e fa          	endbr64
   1825c3804:	53                   	push   rbx
   1825c3805:	56                   	push   rsi
   1825c3806:	57                   	push   rdi
   1825c3807:	55                   	push   rbp
   1825c3808:	41 54                	push   r12
   1825c380a:	41 55                	push   r13
   1825c380c:	41 56                	push   r14
   1825c380e:	41 57                	push   r15
   1825c3810:	48 81 ec 18 01 00 00 	sub    rsp,0x118
   1825c3817:	c5 79 7f 7c 24 70    	vmovdqa XMMWORD PTR [rsp+0x70],xmm15
   1825c381d:	c5 79 7f b4 24 80 00 	vmovdqa XMMWORD PTR [rsp+0x80],xmm14
   1825c3824:	00 00 
   1825c3826:	c5 79 7f ac 24 90 00 	vmovdqa XMMWORD PTR [rsp+0x90],xmm13
   1825c382d:	00 00 
   1825c382f:	c5 79 7f a4 24 a0 00 	vmovdqa XMMWORD PTR [rsp+0xa0],xmm12
   1825c3836:	00 00 
   1825c3838:	c5 79 7f 9c 24 b0 00 	vmovdqa XMMWORD PTR [rsp+0xb0],xmm11
   1825c383f:	00 00 
   1825c3841:	c5 79 7f 94 24 c0 00 	vmovdqa XMMWORD PTR [rsp+0xc0],xmm10
   1825c3848:	00 00 
   1825c384a:	c5 79 7f 8c 24 d0 00 	vmovdqa XMMWORD PTR [rsp+0xd0],xmm9
   1825c3851:	00 00 
   1825c3853:	c5 79 7f 84 24 e0 00 	vmovdqa XMMWORD PTR [rsp+0xe0],xmm8
   1825c385a:	00 00 
   1825c385c:	c5 f9 7f bc 24 f0 00 	vmovdqa XMMWORD PTR [rsp+0xf0],xmm7
   1825c3863:	00 00 
   1825c3865:	c5 f9 7f b4 24 00 01 	vmovdqa XMMWORD PTR [rsp+0x100],xmm6
   1825c386c:	00 00 
   1825c386e:	48 8b f9             	mov    rdi,rcx
   1825c3871:	48 8b f2             	mov    rsi,rdx
   1825c3874:	49 8b d0             	mov    rdx,r8
   1825c3877:	49 8b c9             	mov    rcx,r9
   1825c387a:	4c 8b 84 24 80 01 00 	mov    r8,QWORD PTR [rsp+0x180]
   1825c3881:	00 
   1825c3882:	4c 8b 8c 24 88 01 00 	mov    r9,QWORD PTR [rsp+0x188]
   1825c3889:	00 
   1825c388a:	48 3b fe             	cmp    rdi,rsi
   1825c388d:	74 0c                	je     0x1825c389b
   1825c388f:	48 f7 c6 1f 00 00 00 	test   rsi,0x1f
   1825c3896:	75 03                	jne    0x1825c389b
   1825c3898:	4c 8b ce             	mov    r9,rsi
   1825c389b:	48 81 fa 00 04 00 00 	cmp    rdx,0x400
   1825c38a2:	7f 0a                	jg     0x1825c38ae
   1825c38a4:	48 83 fa 20          	cmp    rdx,0x20
   1825c38a8:	0f 8d 5b 01 00 00    	jge    0x1825c3a09
   1825c38ae:	48 8b ea             	mov    rbp,rdx
   1825c38b1:	4d 8b d8             	mov    r11,r8
   1825c38b4:	4c 8b e7             	mov    r12,rdi
   1825c38b7:	4d 8b e9             	mov    r13,r9
   1825c38ba:	48 d1 e5             	shl    rbp,1
   1825c38bd:	4c 8d 7c 6d 00       	lea    r15,[rbp+rbp*2+0x0]
   1825c38c2:	49 8b dc             	mov    rbx,r12
   1825c38c5:	c5 fd 10 03          	vmovupd ymm0,YMMWORD PTR [rbx]
   1825c38c9:	c5 fd 10 0c ab       	vmovupd ymm1,YMMWORD PTR [rbx+rbp*4]
   1825c38ce:	c5 fd 10 14 6b       	vmovupd ymm2,YMMWORD PTR [rbx+rbp*2]
   1825c38d3:	c4 a1 7d 10 1c 7b    	vmovupd ymm3,YMMWORD PTR [rbx+r15*2]
   1825c38d9:	c5 fd 5c e1          	vsubpd ymm4,ymm0,ymm1
   1825c38dd:	c5 dd c6 e4 05       	vshufpd ymm4,ymm4,ymm4,0x5
   1825c38e2:	c5 fd 58 c1          	vaddpd ymm0,ymm0,ymm1
   1825c38e6:	c5 ed 5c eb          	vsubpd ymm5,ymm2,ymm3
   1825c38ea:	c5 d5 57 2d ee 78 1a 	vxorpd ymm5,ymm5,YMMWORD PTR [rip+0x1a78ee]        # 0x18276b1e0
   1825c38f1:	00 
   1825c38f2:	c5 ed 58 d3          	vaddpd ymm2,ymm2,ymm3
   1825c38f6:	c5 fd 5c ca          	vsubpd ymm1,ymm0,ymm2
   1825c38fa:	c5 fd 58 c2          	vaddpd ymm0,ymm0,ymm2
   1825c38fe:	c5 dd 5c dd          	vsubpd ymm3,ymm4,ymm5
   1825c3902:	c5 dd 58 e5          	vaddpd ymm4,ymm4,ymm5
   1825c3906:	c5 7d 10 43 20       	vmovupd ymm8,YMMWORD PTR [rbx+0x20]
   1825c390b:	c5 7d 10 4c ab 20    	vmovupd ymm9,YMMWORD PTR [rbx+rbp*4+0x20]
   1825c3911:	c5 fd 10 54 6b 20    	vmovupd ymm2,YMMWORD PTR [rbx+rbp*2+0x20]
   1825c3917:	c4 a1 7d 10 7c 7b 20 	vmovupd ymm7,YMMWORD PTR [rbx+r15*2+0x20]
   1825c391e:	41 8b 5b 10          	mov    ebx,DWORD PTR [r11+0x10]
   1825c3922:	c1 e3 05             	shl    ebx,0x5
   1825c3925:	49 83 c3 10          	add    r11,0x10
   1825c3929:	49 03 dc             	add    rbx,r12
   1825c392c:	c4 c1 3d 5c f1       	vsubpd ymm6,ymm8,ymm9
   1825c3931:	c5 cd c6 f6 05       	vshufpd ymm6,ymm6,ymm6,0x5
   1825c3936:	c4 41 3d 58 c1       	vaddpd ymm8,ymm8,ymm9
   1825c393b:	c5 ed 5c ef          	vsubpd ymm5,ymm2,ymm7
   1825c393f:	c5 d5 57 2d 99 78 1a 	vxorpd ymm5,ymm5,YMMWORD PTR [rip+0x1a7899]        # 0x18276b1e0
   1825c3946:	00 
   1825c3947:	c5 ed 58 d7          	vaddpd ymm2,ymm2,ymm7
   1825c394b:	c5 3d 5c ca          	vsubpd ymm9,ymm8,ymm2
   1825c394f:	c5 3d 58 c2          	vaddpd ymm8,ymm8,ymm2
   1825c3953:	c5 cd 5c fd          	vsubpd ymm7,ymm6,ymm5
   1825c3957:	c5 cd 58 f5          	vaddpd ymm6,ymm6,ymm5
   1825c395b:	c5 fd c6 d4 0a       	vshufpd ymm2,ymm0,ymm4,0xa
   1825c3960:	c5 fd c6 c4 05       	vshufpd ymm0,ymm0,ymm4,0x5
   1825c3965:	c5 f5 c6 eb 0a       	vshufpd ymm5,ymm1,ymm3,0xa
   1825c396a:	c5 f5 c6 cb 05       	vshufpd ymm1,ymm1,ymm3,0x5
   1825c396f:	c4 e3 6d 06 e5 20    	vperm2f128 ymm4,ymm2,ymm5,0x20
   1825c3975:	c4 c1 7c 11 65 00    	vmovups YMMWORD PTR [r13+0x0],ymm4
   1825c397b:	c4 e3 7d 06 d9 20    	vperm2f128 ymm3,ymm0,ymm1,0x20
   1825c3981:	c4 c1 7c 11 5d 20    	vmovups YMMWORD PTR [r13+0x20],ymm3
   1825c3987:	c4 e3 6d 06 e5 31    	vperm2f128 ymm4,ymm2,ymm5,0x31
   1825c398d:	c4 c1 7c 11 64 ad 00 	vmovups YMMWORD PTR [r13+rbp*4+0x0],ymm4
   1825c3994:	c4 e3 7d 06 d9 31    	vperm2f128 ymm3,ymm0,ymm1,0x31
   1825c399a:	c4 c1 7c 11 5c ad 20 	vmovups YMMWORD PTR [r13+rbp*4+0x20],ymm3
   1825c39a1:	c5 bd c6 d6 0a       	vshufpd ymm2,ymm8,ymm6,0xa
   1825c39a6:	c5 3d c6 c6 05       	vshufpd ymm8,ymm8,ymm6,0x5
   1825c39ab:	c5 b5 c6 ef 0a       	vshufpd ymm5,ymm9,ymm7,0xa
   1825c39b0:	c5 35 c6 cf 05       	vshufpd ymm9,ymm9,ymm7,0x5
   1825c39b5:	c4 e3 6d 06 f5 20    	vperm2f128 ymm6,ymm2,ymm5,0x20
   1825c39bb:	c4 c1 7c 11 74 6d 00 	vmovups YMMWORD PTR [r13+rbp*2+0x0],ymm6
   1825c39c2:	c4 c3 3d 06 f9 20    	vperm2f128 ymm7,ymm8,ymm9,0x20
   1825c39c8:	c4 c1 7c 11 7c 6d 20 	vmovups YMMWORD PTR [r13+rbp*2+0x20],ymm7
   1825c39cf:	c4 e3 6d 06 f5 31    	vperm2f128 ymm6,ymm2,ymm5,0x31
   1825c39d5:	c4 81 7c 11 74 7d 00 	vmovups YMMWORD PTR [r13+r15*2+0x0],ymm6
   1825c39dc:	c4 c3 3d 06 f9 31    	vperm2f128 ymm7,ymm8,ymm9,0x31
   1825c39e2:	c4 81 7c 11 7c 7d 20 	vmovups YMMWORD PTR [r13+r15*2+0x20],ymm7
   1825c39e9:	49 83 c5 40          	add    r13,0x40
   1825c39ed:	49 3b dc             	cmp    rbx,r12
   1825c39f0:	0f 85 cf fe ff ff    	jne    0x1825c38c5
   1825c39f6:	49 c7 c2 04 00 00 00 	mov    r10,0x4
   1825c39fd:	4c 8b da             	mov    r11,rdx
   1825c3a00:	49 c1 eb 02          	shr    r11,0x2
   1825c3a04:	e9 a2 01 00 00       	jmp    0x1825c3bab
   1825c3a09:	4c 8b e7             	mov    r12,rdi
   1825c3a0c:	4d 8b e9             	mov    r13,r9
   1825c3a0f:	4d 8b dc             	mov    r11,r12
   1825c3a12:	4c 8b d2             	mov    r10,rdx
   1825c3a15:	49 d1 e2             	shl    r10,1
   1825c3a18:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c3a1c:	49 8b d8             	mov    rbx,r8
   1825c3a1f:	c4 c1 7d 10 03       	vmovupd ymm0,YMMWORD PTR [r11]
   1825c3a24:	c4 81 7d 10 1c 93    	vmovupd ymm3,YMMWORD PTR [r11+r10*4]
   1825c3a2a:	c4 81 7d 10 34 53    	vmovupd ymm6,YMMWORD PTR [r11+r10*2]
   1825c3a30:	c4 c1 7d 10 0c 43    	vmovupd ymm1,YMMWORD PTR [r11+rax*2]
   1825c3a36:	4d 03 da             	add    r11,r10
   1825c3a39:	c4 41 7d 10 03       	vmovupd ymm8,YMMWORD PTR [r11]
   1825c3a3e:	c4 81 7d 10 2c 93    	vmovupd ymm5,YMMWORD PTR [r11+r10*4]
   1825c3a44:	c4 01 7d 10 14 53    	vmovupd ymm10,YMMWORD PTR [r11+r10*2]
   1825c3a4a:	c4 41 7d 10 0c 43    	vmovupd ymm9,YMMWORD PTR [r11+rax*2]
   1825c3a50:	c5 fd 5c d3          	vsubpd ymm2,ymm0,ymm3
   1825c3a54:	c5 fd 58 c3          	vaddpd ymm0,ymm0,ymm3
   1825c3a58:	c5 f5 58 de          	vaddpd ymm3,ymm1,ymm6
   1825c3a5c:	c5 f5 5c f6          	vsubpd ymm6,ymm1,ymm6
   1825c3a60:	4c 63 5b 10          	movsxd r11,DWORD PTR [rbx+0x10]
   1825c3a64:	49 c1 e3 04          	shl    r11,0x4
   1825c3a68:	48 83 c3 10          	add    rbx,0x10
   1825c3a6c:	4d 03 dc             	add    r11,r12
   1825c3a6f:	c5 bd 5c fd          	vsubpd ymm7,ymm8,ymm5
   1825c3a73:	c5 3d 58 c5          	vaddpd ymm8,ymm8,ymm5
   1825c3a77:	c4 c1 35 58 ea       	vaddpd ymm5,ymm9,ymm10
   1825c3a7c:	c4 41 35 5c d2       	vsubpd ymm10,ymm9,ymm10
   1825c3a81:	c5 c5 59 3d 17 77 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a7717]        # 0x18276b1a0
   1825c3a88:	00 
   1825c3a89:	c5 2d 59 15 0f 77 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a770f]        # 0x18276b1a0
   1825c3a90:	00 
   1825c3a91:	c5 fd 5c cb          	vsubpd ymm1,ymm0,ymm3
   1825c3a95:	c5 fd 58 c3          	vaddpd ymm0,ymm0,ymm3
   1825c3a99:	c4 41 55 5c c8       	vsubpd ymm9,ymm5,ymm8
   1825c3a9e:	c4 41 55 58 c0       	vaddpd ymm8,ymm5,ymm8
   1825c3aa3:	c4 41 35 c6 c9 05    	vshufpd ymm9,ymm9,ymm9,0x5
   1825c3aa9:	c5 35 57 0d 4f 77 1a 	vxorpd ymm9,ymm9,YMMWORD PTR [rip+0x1a774f]        # 0x18276b200
   1825c3ab0:	00 
   1825c3ab1:	c4 c1 7d 5c e0       	vsubpd ymm4,ymm0,ymm8
   1825c3ab6:	c4 41 7d 58 c0       	vaddpd ymm8,ymm0,ymm8
   1825c3abb:	c4 c1 75 5c c1       	vsubpd ymm0,ymm1,ymm9
   1825c3ac0:	c4 c1 75 58 c9       	vaddpd ymm1,ymm1,ymm9
   1825c3ac5:	c5 ad 5c ef          	vsubpd ymm5,ymm10,ymm7
   1825c3ac9:	c5 ad 58 ff          	vaddpd ymm7,ymm10,ymm7
   1825c3acd:	c5 ed 5c df          	vsubpd ymm3,ymm2,ymm7
   1825c3ad1:	c5 ed 58 d7          	vaddpd ymm2,ymm2,ymm7
   1825c3ad5:	c5 d5 5c fe          	vsubpd ymm7,ymm5,ymm6
   1825c3ad9:	c5 d5 58 ee          	vaddpd ymm5,ymm5,ymm6
   1825c3add:	c5 c5 c6 ff 05       	vshufpd ymm7,ymm7,ymm7,0x5
   1825c3ae2:	c5 c5 57 3d 16 77 1a 	vxorpd ymm7,ymm7,YMMWORD PTR [rip+0x1a7716]        # 0x18276b200
   1825c3ae9:	00 
   1825c3aea:	c5 d5 c6 ed 05       	vshufpd ymm5,ymm5,ymm5,0x5
   1825c3aef:	c5 d5 57 2d 09 77 1a 	vxorpd ymm5,ymm5,YMMWORD PTR [rip+0x1a7709]        # 0x18276b200
   1825c3af6:	00 
   1825c3af7:	c5 6d 5c cd          	vsubpd ymm9,ymm2,ymm5
   1825c3afb:	c5 ed 58 d5          	vaddpd ymm2,ymm2,ymm5
   1825c3aff:	c5 e5 5c ef          	vsubpd ymm5,ymm3,ymm7
   1825c3b03:	c5 e5 58 df          	vaddpd ymm3,ymm3,ymm7
   1825c3b07:	c4 c1 3d 14 f1       	vunpcklpd ymm6,ymm8,ymm9
   1825c3b0c:	c4 41 3d 15 c1       	vunpckhpd ymm8,ymm8,ymm9
   1825c3b11:	c5 fd 14 fd          	vunpcklpd ymm7,ymm0,ymm5
   1825c3b15:	c5 fd 15 c5          	vunpckhpd ymm0,ymm0,ymm5
   1825c3b19:	c4 63 4d 06 cf 20    	vperm2f128 ymm9,ymm6,ymm7,0x20
   1825c3b1f:	c4 41 7c 11 4d 00    	vmovups YMMWORD PTR [r13+0x0],ymm9
   1825c3b25:	c4 e3 3d 06 e8 20    	vperm2f128 ymm5,ymm8,ymm0,0x20
   1825c3b2b:	c4 c1 7c 11 6d 20    	vmovups YMMWORD PTR [r13+0x20],ymm5
   1825c3b31:	c4 63 4d 06 cf 31    	vperm2f128 ymm9,ymm6,ymm7,0x31
   1825c3b37:	c4 01 7c 11 4c 95 00 	vmovups YMMWORD PTR [r13+r10*4+0x0],ymm9
   1825c3b3e:	c4 e3 3d 06 e8 31    	vperm2f128 ymm5,ymm8,ymm0,0x31
   1825c3b44:	c4 81 7c 11 6c 95 20 	vmovups YMMWORD PTR [r13+r10*4+0x20],ymm5
   1825c3b4b:	c5 dd 14 f3          	vunpcklpd ymm6,ymm4,ymm3
   1825c3b4f:	c5 dd 15 e3          	vunpckhpd ymm4,ymm4,ymm3
   1825c3b53:	c5 f5 14 fa          	vunpcklpd ymm7,ymm1,ymm2
   1825c3b57:	c5 f5 15 ca          	vunpckhpd ymm1,ymm1,ymm2
   1825c3b5b:	c4 63 4d 06 cf 20    	vperm2f128 ymm9,ymm6,ymm7,0x20
   1825c3b61:	c4 41 7c 11 4d 40    	vmovups YMMWORD PTR [r13+0x40],ymm9
   1825c3b67:	c4 e3 5d 06 e9 20    	vperm2f128 ymm5,ymm4,ymm1,0x20
   1825c3b6d:	c4 c1 7c 11 6d 60    	vmovups YMMWORD PTR [r13+0x60],ymm5
   1825c3b73:	c4 63 4d 06 cf 31    	vperm2f128 ymm9,ymm6,ymm7,0x31
   1825c3b79:	c4 01 7c 11 4c 95 40 	vmovups YMMWORD PTR [r13+r10*4+0x40],ymm9
   1825c3b80:	c4 e3 5d 06 e9 31    	vperm2f128 ymm5,ymm4,ymm1,0x31
   1825c3b86:	c4 81 7c 11 6c 95 60 	vmovups YMMWORD PTR [r13+r10*4+0x60],ymm5
   1825c3b8d:	49 81 c5 80 00 00 00 	add    r13,0x80
   1825c3b94:	4d 3b dc             	cmp    r11,r12
   1825c3b97:	0f 85 82 fe ff ff    	jne    0x1825c3a1f
   1825c3b9d:	49 c7 c2 08 00 00 00 	mov    r10,0x8
   1825c3ba4:	4c 8b da             	mov    r11,rdx
   1825c3ba7:	49 c1 eb 03          	shr    r11,0x3
   1825c3bab:	48 81 fa 00 04 00 00 	cmp    rdx,0x400
   1825c3bb2:	0f 8e 40 13 00 00    	jle    0x1825c4ef8
   1825c3bb8:	48 89 54 24 50       	mov    QWORD PTR [rsp+0x50],rdx
   1825c3bbd:	48 8b d9             	mov    rbx,rcx
   1825c3bc0:	48 89 5c 24 58       	mov    QWORD PTR [rsp+0x58],rbx
   1825c3bc5:	49 8b d9             	mov    rbx,r9
   1825c3bc8:	48 89 5c 24 60       	mov    QWORD PTR [rsp+0x60],rbx
   1825c3bcd:	4c 89 5c 24 40       	mov    QWORD PTR [rsp+0x40],r11
   1825c3bd2:	4c 89 5c 24 48       	mov    QWORD PTR [rsp+0x48],r11
   1825c3bd7:	49 c7 c3 00 04 00 00 	mov    r11,0x400
   1825c3bde:	49 c1 eb 02          	shr    r11,0x2
   1825c3be2:	4c 89 5c 24 38       	mov    QWORD PTR [rsp+0x38],r11
   1825c3be7:	4c 89 54 24 30       	mov    QWORD PTR [rsp+0x30],r10
   1825c3bec:	49 83 fb 04          	cmp    r11,0x4
   1825c3bf0:	0f 8c 6f 09 00 00    	jl     0x1825c4565
   1825c3bf6:	48 f7 c2 55 55 55 55 	test   rdx,0x55555555
   1825c3bfd:	75 2a                	jne    0x1825c3c29
   1825c3bff:	49 83 fa 04          	cmp    r10,0x4
   1825c3c03:	0f 84 eb 02 00 00    	je     0x1825c3ef4
   1825c3c09:	48 81 fa 00 08 00 00 	cmp    rdx,0x800
   1825c3c10:	74 17                	je     0x1825c3c29
   1825c3c12:	49 83 fb 08          	cmp    r11,0x8
   1825c3c16:	0f 8d d8 02 00 00    	jge    0x1825c3ef4
   1825c3c1c:	48 81 fa 00 20 00 00 	cmp    rdx,0x2000
   1825c3c23:	0f 84 3c 09 00 00    	je     0x1825c4565
   1825c3c29:	49 c1 eb 02          	shr    r11,0x2
   1825c3c2d:	4c 89 54 24 20       	mov    QWORD PTR [rsp+0x20],r10
   1825c3c32:	4c 89 5c 24 28       	mov    QWORD PTR [rsp+0x28],r11
   1825c3c37:	4d 8b e1             	mov    r12,r9
   1825c3c3a:	4d 8b e9             	mov    r13,r9
   1825c3c3d:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c3c42:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c3c47:	49 c1 e2 04          	shl    r10,0x4
   1825c3c4b:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c3c4f:	49 f7 c5 1f 00 00 00 	test   r13,0x1f
   1825c3c56:	0f 85 3b 01 00 00    	jne    0x1825c3d97
   1825c3c5c:	48 8b d9             	mov    rbx,rcx
   1825c3c5f:	4c 8b 64 24 20       	mov    r12,QWORD PTR [rsp+0x20]
   1825c3c64:	c4 01 7d 28 44 55 00 	vmovapd ymm8,YMMWORD PTR [r13+r10*2+0x0]
   1825c3c6b:	c5 fd 28 0b          	vmovapd ymm1,YMMWORD PTR [rbx]
   1825c3c6f:	c4 c1 75 59 c0       	vmulpd ymm0,ymm1,ymm8
   1825c3c74:	c4 81 7d 28 74 55 20 	vmovapd ymm6,YMMWORD PTR [r13+r10*2+0x20]
   1825c3c7b:	c5 f5 59 ce          	vmulpd ymm1,ymm1,ymm6
   1825c3c7f:	c5 fd 28 7b 20       	vmovapd ymm7,YMMWORD PTR [rbx+0x20]
   1825c3c84:	c4 e2 c5 bc c6       	vfnmadd231pd ymm0,ymm7,ymm6
   1825c3c89:	c4 c2 c5 b8 c8       	vfmadd231pd ymm1,ymm7,ymm8
   1825c3c8e:	c4 41 7d 28 44 05 00 	vmovapd ymm8,YMMWORD PTR [r13+rax*1+0x0]
   1825c3c95:	c5 fd 28 ab 80 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0x80]
   1825c3c9c:	00 
   1825c3c9d:	c4 41 55 59 c8       	vmulpd ymm9,ymm5,ymm8
   1825c3ca2:	c4 c1 7d 28 74 05 20 	vmovapd ymm6,YMMWORD PTR [r13+rax*1+0x20]
   1825c3ca9:	c5 55 59 d6          	vmulpd ymm10,ymm5,ymm6
   1825c3cad:	c5 fd 28 bb a0 00 00 	vmovapd ymm7,YMMWORD PTR [rbx+0xa0]
   1825c3cb4:	00 
   1825c3cb5:	c4 62 c5 bc ce       	vfnmadd231pd ymm9,ymm7,ymm6
   1825c3cba:	c4 42 c5 b8 d0       	vfmadd231pd ymm10,ymm7,ymm8
   1825c3cbf:	c4 01 7d 28 44 15 00 	vmovapd ymm8,YMMWORD PTR [r13+r10*1+0x0]
   1825c3cc6:	c5 fd 28 5b 40       	vmovapd ymm3,YMMWORD PTR [rbx+0x40]
   1825c3ccb:	c4 c1 65 59 d0       	vmulpd ymm2,ymm3,ymm8
   1825c3cd0:	c4 81 7d 28 74 15 20 	vmovapd ymm6,YMMWORD PTR [r13+r10*1+0x20]
   1825c3cd7:	c5 e5 59 de          	vmulpd ymm3,ymm3,ymm6
   1825c3cdb:	c5 fd 28 7b 60       	vmovapd ymm7,YMMWORD PTR [rbx+0x60]
   1825c3ce0:	c4 e2 c5 bc d6       	vfnmadd231pd ymm2,ymm7,ymm6
   1825c3ce5:	c4 c2 c5 b8 d8       	vfmadd231pd ymm3,ymm7,ymm8
   1825c3cea:	48 81 c3 c0 00 00 00 	add    rbx,0xc0
   1825c3cf1:	c5 b5 58 e0          	vaddpd ymm4,ymm9,ymm0
   1825c3cf5:	c4 c1 7d 5c c1       	vsubpd ymm0,ymm0,ymm9
   1825c3cfa:	c5 ad 58 e9          	vaddpd ymm5,ymm10,ymm1
   1825c3cfe:	c4 c1 75 5c ca       	vsubpd ymm1,ymm1,ymm10
   1825c3d03:	c4 c1 7d 28 7d 00    	vmovapd ymm7,YMMWORD PTR [r13+0x0]
   1825c3d09:	c5 c5 5c f2          	vsubpd ymm6,ymm7,ymm2
   1825c3d0d:	c5 c5 58 d2          	vaddpd ymm2,ymm7,ymm2
   1825c3d11:	c5 ed 58 fc          	vaddpd ymm7,ymm2,ymm4
   1825c3d15:	c4 c1 7d 29 7d 00    	vmovapd YMMWORD PTR [r13+0x0],ymm7
   1825c3d1b:	c5 ed 5c d4          	vsubpd ymm2,ymm2,ymm4
   1825c3d1f:	c4 81 7d 29 54 55 00 	vmovapd YMMWORD PTR [r13+r10*2+0x0],ymm2
   1825c3d26:	c4 c1 7d 28 7d 20    	vmovapd ymm7,YMMWORD PTR [r13+0x20]
   1825c3d2c:	c5 c5 5c e3          	vsubpd ymm4,ymm7,ymm3
   1825c3d30:	c5 c5 58 db          	vaddpd ymm3,ymm7,ymm3
   1825c3d34:	c5 e5 58 fd          	vaddpd ymm7,ymm3,ymm5
   1825c3d38:	c4 c1 7d 29 7d 20    	vmovapd YMMWORD PTR [r13+0x20],ymm7
   1825c3d3e:	c5 e5 5c dd          	vsubpd ymm3,ymm3,ymm5
   1825c3d42:	c4 81 7d 29 5c 55 20 	vmovapd YMMWORD PTR [r13+r10*2+0x20],ymm3
   1825c3d49:	c5 cd 58 f9          	vaddpd ymm7,ymm6,ymm1
   1825c3d4d:	c4 81 7d 29 7c 15 00 	vmovapd YMMWORD PTR [r13+r10*1+0x0],ymm7
   1825c3d54:	c5 cd 5c f1          	vsubpd ymm6,ymm6,ymm1
   1825c3d58:	c4 c1 7d 29 74 05 00 	vmovapd YMMWORD PTR [r13+rax*1+0x0],ymm6
   1825c3d5f:	c5 dd 5c f8          	vsubpd ymm7,ymm4,ymm0
   1825c3d63:	c4 81 7d 29 7c 15 20 	vmovapd YMMWORD PTR [r13+r10*1+0x20],ymm7
   1825c3d6a:	c5 dd 58 e0          	vaddpd ymm4,ymm4,ymm0
   1825c3d6e:	c4 c1 7d 29 64 05 20 	vmovapd YMMWORD PTR [r13+rax*1+0x20],ymm4
   1825c3d75:	49 83 c5 40          	add    r13,0x40
   1825c3d79:	49 83 ec 04          	sub    r12,0x4
   1825c3d7d:	0f 8f e1 fe ff ff    	jg     0x1825c3c64
   1825c3d83:	4c 03 e8             	add    r13,rax
   1825c3d86:	49 ff cb             	dec    r11
   1825c3d89:	0f 8f cd fe ff ff    	jg     0x1825c3c5c
   1825c3d8f:	48 8b cb             	mov    rcx,rbx
   1825c3d92:	e9 3c 01 00 00       	jmp    0x1825c3ed3
   1825c3d97:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c3d9b:	48 8b fb             	mov    rdi,rbx
   1825c3d9e:	48 8b d9             	mov    rbx,rcx
   1825c3da1:	c4 01 7d 28 04 54    	vmovapd ymm8,YMMWORD PTR [r12+r10*2]
   1825c3da7:	c5 fd 28 0b          	vmovapd ymm1,YMMWORD PTR [rbx]
   1825c3dab:	c4 c1 75 59 c0       	vmulpd ymm0,ymm1,ymm8
   1825c3db0:	c4 81 7d 28 74 54 20 	vmovapd ymm6,YMMWORD PTR [r12+r10*2+0x20]
   1825c3db7:	c5 f5 59 ce          	vmulpd ymm1,ymm1,ymm6
   1825c3dbb:	c5 fd 28 7b 20       	vmovapd ymm7,YMMWORD PTR [rbx+0x20]
   1825c3dc0:	c4 e2 c5 bc c6       	vfnmadd231pd ymm0,ymm7,ymm6
   1825c3dc5:	c4 c2 c5 b8 c8       	vfmadd231pd ymm1,ymm7,ymm8
   1825c3dca:	c4 41 7d 28 04 04    	vmovapd ymm8,YMMWORD PTR [r12+rax*1]
   1825c3dd0:	c5 fd 28 ab 80 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0x80]
   1825c3dd7:	00 
   1825c3dd8:	c4 41 55 59 c8       	vmulpd ymm9,ymm5,ymm8
   1825c3ddd:	c4 c1 7d 28 74 04 20 	vmovapd ymm6,YMMWORD PTR [r12+rax*1+0x20]
   1825c3de4:	c5 55 59 d6          	vmulpd ymm10,ymm5,ymm6
   1825c3de8:	c5 fd 28 bb a0 00 00 	vmovapd ymm7,YMMWORD PTR [rbx+0xa0]
   1825c3def:	00 
   1825c3df0:	c4 62 c5 bc ce       	vfnmadd231pd ymm9,ymm7,ymm6
   1825c3df5:	c4 42 c5 b8 d0       	vfmadd231pd ymm10,ymm7,ymm8
   1825c3dfa:	c4 01 7d 28 04 14    	vmovapd ymm8,YMMWORD PTR [r12+r10*1]
   1825c3e00:	c5 fd 28 5b 40       	vmovapd ymm3,YMMWORD PTR [rbx+0x40]
   1825c3e05:	c4 c1 65 59 d0       	vmulpd ymm2,ymm3,ymm8
   1825c3e0a:	c4 81 7d 28 74 14 20 	vmovapd ymm6,YMMWORD PTR [r12+r10*1+0x20]
   1825c3e11:	c5 e5 59 de          	vmulpd ymm3,ymm3,ymm6
   1825c3e15:	c5 fd 28 7b 60       	vmovapd ymm7,YMMWORD PTR [rbx+0x60]
   1825c3e1a:	c4 e2 c5 bc d6       	vfnmadd231pd ymm2,ymm7,ymm6
   1825c3e1f:	c4 c2 c5 b8 d8       	vfmadd231pd ymm3,ymm7,ymm8
   1825c3e24:	48 81 c3 c0 00 00 00 	add    rbx,0xc0
   1825c3e2b:	c5 b5 58 e0          	vaddpd ymm4,ymm9,ymm0
   1825c3e2f:	c4 c1 7d 5c c1       	vsubpd ymm0,ymm0,ymm9
   1825c3e34:	c5 ad 58 e9          	vaddpd ymm5,ymm10,ymm1
   1825c3e38:	c4 c1 75 5c ca       	vsubpd ymm1,ymm1,ymm10
   1825c3e3d:	c4 c1 7d 28 3c 24    	vmovapd ymm7,YMMWORD PTR [r12]
   1825c3e43:	c5 c5 5c f2          	vsubpd ymm6,ymm7,ymm2
   1825c3e47:	c5 c5 58 d2          	vaddpd ymm2,ymm7,ymm2
   1825c3e4b:	c5 ed 58 fc          	vaddpd ymm7,ymm2,ymm4
   1825c3e4f:	c4 c1 7c 11 7d 00    	vmovups YMMWORD PTR [r13+0x0],ymm7
   1825c3e55:	c5 ed 5c d4          	vsubpd ymm2,ymm2,ymm4
   1825c3e59:	c4 81 7c 11 54 55 00 	vmovups YMMWORD PTR [r13+r10*2+0x0],ymm2
   1825c3e60:	c4 c1 7d 28 7c 24 20 	vmovapd ymm7,YMMWORD PTR [r12+0x20]
   1825c3e67:	c5 c5 5c e3          	vsubpd ymm4,ymm7,ymm3
   1825c3e6b:	c5 c5 58 db          	vaddpd ymm3,ymm7,ymm3
   1825c3e6f:	c5 e5 58 fd          	vaddpd ymm7,ymm3,ymm5
   1825c3e73:	c4 c1 7c 11 7d 20    	vmovups YMMWORD PTR [r13+0x20],ymm7
   1825c3e79:	c5 e5 5c dd          	vsubpd ymm3,ymm3,ymm5
   1825c3e7d:	c4 81 7c 11 5c 55 20 	vmovups YMMWORD PTR [r13+r10*2+0x20],ymm3
   1825c3e84:	c5 cd 58 f9          	vaddpd ymm7,ymm6,ymm1
   1825c3e88:	c4 81 7c 11 7c 15 00 	vmovups YMMWORD PTR [r13+r10*1+0x0],ymm7
   1825c3e8f:	c5 cd 5c f1          	vsubpd ymm6,ymm6,ymm1
   1825c3e93:	c4 c1 7c 11 74 05 00 	vmovups YMMWORD PTR [r13+rax*1+0x0],ymm6
   1825c3e9a:	c5 dd 5c f8          	vsubpd ymm7,ymm4,ymm0
   1825c3e9e:	c4 81 7c 11 7c 15 20 	vmovups YMMWORD PTR [r13+r10*1+0x20],ymm7
   1825c3ea5:	c5 dd 58 e0          	vaddpd ymm4,ymm4,ymm0
   1825c3ea9:	c4 c1 7c 11 64 05 20 	vmovups YMMWORD PTR [r13+rax*1+0x20],ymm4
   1825c3eb0:	49 83 c4 40          	add    r12,0x40
   1825c3eb4:	49 83 c5 40          	add    r13,0x40
   1825c3eb8:	4c 3b e7             	cmp    r12,rdi
   1825c3ebb:	0f 85 e0 fe ff ff    	jne    0x1825c3da1
   1825c3ec1:	4c 03 e0             	add    r12,rax
   1825c3ec4:	4c 03 e8             	add    r13,rax
   1825c3ec7:	49 ff cb             	dec    r11
   1825c3eca:	0f 8f c7 fe ff ff    	jg     0x1825c3d97
   1825c3ed0:	48 8b cb             	mov    rcx,rbx
   1825c3ed3:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c3ed8:	49 c1 e2 02          	shl    r10,0x2
   1825c3edc:	4c 8b 5c 24 48       	mov    r11,QWORD PTR [rsp+0x48]
   1825c3ee1:	49 c1 eb 02          	shr    r11,0x2
   1825c3ee5:	4c 89 5c 24 48       	mov    QWORD PTR [rsp+0x48],r11
   1825c3eea:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c3eef:	e9 f8 fc ff ff       	jmp    0x1825c3bec
   1825c3ef4:	49 c1 eb 03          	shr    r11,0x3
   1825c3ef8:	4c 89 54 24 20       	mov    QWORD PTR [rsp+0x20],r10
   1825c3efd:	4c 89 5c 24 28       	mov    QWORD PTR [rsp+0x28],r11
   1825c3f02:	4d 8b e1             	mov    r12,r9
   1825c3f05:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c3f0a:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c3f0f:	49 c1 e2 04          	shl    r10,0x4
   1825c3f13:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c3f17:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c3f1b:	4d 8b f1             	mov    r14,r9
   1825c3f1e:	4f 8d 3c 96          	lea    r15,[r14+r10*4]
   1825c3f22:	49 f7 c6 1f 00 00 00 	test   r14,0x1f
   1825c3f29:	0f 85 08 03 00 00    	jne    0x1825c4237
   1825c3f2f:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c3f33:	48 8b fb             	mov    rdi,rbx
   1825c3f36:	48 8b d9             	mov    rbx,rcx
   1825c3f39:	c4 81 7d 28 14 54    	vmovapd ymm2,YMMWORD PTR [r12+r10*2]
   1825c3f3f:	c5 fd 28 4b 40       	vmovapd ymm1,YMMWORD PTR [rbx+0x40]
   1825c3f44:	c5 f5 59 c2          	vmulpd ymm0,ymm1,ymm2
   1825c3f48:	c4 01 7d 28 44 54 20 	vmovapd ymm8,YMMWORD PTR [r12+r10*2+0x20]
   1825c3f4f:	c4 c1 75 59 c8       	vmulpd ymm1,ymm1,ymm8
   1825c3f54:	c5 7d 28 4b 60       	vmovapd ymm9,YMMWORD PTR [rbx+0x60]
   1825c3f59:	c4 c2 b5 bc c0       	vfnmadd231pd ymm0,ymm9,ymm8
   1825c3f5e:	c4 e2 b5 b8 ca       	vfmadd231pd ymm1,ymm9,ymm2
   1825c3f63:	c4 c1 7d 28 24 04    	vmovapd ymm4,YMMWORD PTR [r12+rax*1]
   1825c3f69:	c5 fd 28 9b 40 01 00 	vmovapd ymm3,YMMWORD PTR [rbx+0x140]
   1825c3f70:	00 
   1825c3f71:	c5 65 59 f4          	vmulpd ymm14,ymm3,ymm4
   1825c3f75:	c4 41 7d 28 54 04 20 	vmovapd ymm10,YMMWORD PTR [r12+rax*1+0x20]
   1825c3f7c:	c4 c1 65 59 f2       	vmulpd ymm6,ymm3,ymm10
   1825c3f81:	c5 7d 28 9b 60 01 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x160]
   1825c3f88:	00 
   1825c3f89:	c4 42 a5 bc f2       	vfnmadd231pd ymm14,ymm11,ymm10
   1825c3f8e:	c4 e2 a5 b8 f4       	vfmadd231pd ymm6,ymm11,ymm4
   1825c3f93:	c5 8d 58 d0          	vaddpd ymm2,ymm14,ymm0
   1825c3f97:	c4 c1 7d 5c c6       	vsubpd ymm0,ymm0,ymm14
   1825c3f9c:	c5 cd 58 d9          	vaddpd ymm3,ymm6,ymm1
   1825c3fa0:	c5 f5 5c ce          	vsubpd ymm1,ymm1,ymm6
   1825c3fa4:	c4 81 7d 28 34 14    	vmovapd ymm6,YMMWORD PTR [r12+r10*1]
   1825c3faa:	c5 fd 28 ab c0 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0xc0]
   1825c3fb1:	00 
   1825c3fb2:	c5 d5 59 e6          	vmulpd ymm4,ymm5,ymm6
   1825c3fb6:	c4 01 7d 28 64 14 20 	vmovapd ymm12,YMMWORD PTR [r12+r10*1+0x20]
   1825c3fbd:	c4 c1 55 59 ec       	vmulpd ymm5,ymm5,ymm12
   1825c3fc2:	c5 7d 28 ab e0 00 00 	vmovapd ymm13,YMMWORD PTR [rbx+0xe0]
   1825c3fc9:	00 
   1825c3fca:	c4 c2 95 bc e4       	vfnmadd231pd ymm4,ymm13,ymm12
   1825c3fcf:	c4 e2 95 b8 ee       	vfmadd231pd ymm5,ymm13,ymm6
   1825c3fd4:	c4 41 7d 28 45 00    	vmovapd ymm8,YMMWORD PTR [r13+0x0]
   1825c3fda:	c5 fd 28 3b          	vmovapd ymm7,YMMWORD PTR [rbx]
   1825c3fde:	c4 c1 45 59 f0       	vmulpd ymm6,ymm7,ymm8
   1825c3fe3:	c4 41 7d 28 75 20    	vmovapd ymm14,YMMWORD PTR [r13+0x20]
   1825c3fe9:	c4 c1 45 59 fe       	vmulpd ymm7,ymm7,ymm14
   1825c3fee:	c5 7d 28 7b 20       	vmovapd ymm15,YMMWORD PTR [rbx+0x20]
   1825c3ff3:	c4 c2 85 bc f6       	vfnmadd231pd ymm6,ymm15,ymm14
   1825c3ff8:	c4 c2 85 b8 f8       	vfmadd231pd ymm7,ymm15,ymm8
   1825c3ffd:	c4 01 7d 28 54 15 00 	vmovapd ymm10,YMMWORD PTR [r13+r10*1+0x0]
   1825c4004:	c5 7d 28 a3 00 01 00 	vmovapd ymm12,YMMWORD PTR [rbx+0x100]
   1825c400b:	00 
   1825c400c:	c4 41 1d 59 f2       	vmulpd ymm14,ymm12,ymm10
   1825c4011:	c4 01 7d 28 4c 15 20 	vmovapd ymm9,YMMWORD PTR [r13+r10*1+0x20]
   1825c4018:	c4 41 1d 59 e1       	vmulpd ymm12,ymm12,ymm9
   1825c401d:	c5 7d 28 ab 20 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x120]
   1825c4024:	00 
   1825c4025:	c4 42 95 bc f1       	vfnmadd231pd ymm14,ymm13,ymm9
   1825c402a:	c4 42 95 b8 e2       	vfmadd231pd ymm12,ymm13,ymm10
   1825c402f:	c5 0d 58 c6          	vaddpd ymm8,ymm14,ymm6
   1825c4033:	c4 c1 4d 5c f6       	vsubpd ymm6,ymm6,ymm14
   1825c4038:	c5 1d 58 cf          	vaddpd ymm9,ymm12,ymm7
   1825c403c:	c4 c1 45 5c fc       	vsubpd ymm7,ymm7,ymm12
   1825c4041:	c4 01 7d 28 64 55 00 	vmovapd ymm12,YMMWORD PTR [r13+r10*2+0x0]
   1825c4048:	c5 7d 28 9b 80 00 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x80]
   1825c404f:	00 
   1825c4050:	c4 41 25 59 d4       	vmulpd ymm10,ymm11,ymm12
   1825c4055:	c4 01 7d 28 74 55 20 	vmovapd ymm14,YMMWORD PTR [r13+r10*2+0x20]
   1825c405c:	c4 41 25 59 de       	vmulpd ymm11,ymm11,ymm14
   1825c4061:	c5 7d 28 bb a0 00 00 	vmovapd ymm15,YMMWORD PTR [rbx+0xa0]
   1825c4068:	00 
   1825c4069:	c4 42 85 bc d6       	vfnmadd231pd ymm10,ymm15,ymm14
   1825c406e:	c4 42 85 b8 dc       	vfmadd231pd ymm11,ymm15,ymm12
   1825c4073:	c4 41 7d 28 64 05 00 	vmovapd ymm12,YMMWORD PTR [r13+rax*1+0x0]
   1825c407a:	c5 7d 28 ab 80 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x180]
   1825c4081:	00 
   1825c4082:	c4 41 1d 59 e5       	vmulpd ymm12,ymm12,ymm13
   1825c4087:	c4 41 7d 28 74 05 20 	vmovapd ymm14,YMMWORD PTR [r13+rax*1+0x20]
   1825c408e:	c4 41 15 59 ee       	vmulpd ymm13,ymm13,ymm14
   1825c4093:	c5 7d 28 bb a0 01 00 	vmovapd ymm15,YMMWORD PTR [rbx+0x1a0]
   1825c409a:	00 
   1825c409b:	c4 42 85 bc e6       	vfnmadd231pd ymm12,ymm15,ymm14
   1825c40a0:	c4 42 85 b8 6c 05 00 	vfmadd231pd ymm13,ymm15,YMMWORD PTR [r13+rax*1+0x0]
   1825c40a7:	48 81 c3 c0 01 00 00 	add    rbx,0x1c0
   1825c40ae:	c4 41 7d 28 34 24    	vmovapd ymm14,YMMWORD PTR [r12]
   1825c40b4:	c5 0d 5c fc          	vsubpd ymm15,ymm14,ymm4
   1825c40b8:	c4 41 5d 58 f6       	vaddpd ymm14,ymm4,ymm14
   1825c40bd:	c5 8d 5c e2          	vsubpd ymm4,ymm14,ymm2
   1825c40c1:	c4 c1 6d 58 d6       	vaddpd ymm2,ymm2,ymm14
   1825c40c6:	c4 41 2d 5c f4       	vsubpd ymm14,ymm10,ymm12
   1825c40cb:	c4 41 1d 58 e2       	vaddpd ymm12,ymm12,ymm10
   1825c40d0:	c4 41 25 5c d5       	vsubpd ymm10,ymm11,ymm13
   1825c40d5:	c4 41 15 58 eb       	vaddpd ymm13,ymm13,ymm11
   1825c40da:	c4 41 3d 5c dc       	vsubpd ymm11,ymm8,ymm12
   1825c40df:	c4 41 1d 58 e0       	vaddpd ymm12,ymm12,ymm8
   1825c40e4:	c5 05 5c c1          	vsubpd ymm8,ymm15,ymm1
   1825c40e8:	c5 05 58 f9          	vaddpd ymm15,ymm15,ymm1
   1825c40ec:	c4 c1 35 5c cd       	vsubpd ymm1,ymm9,ymm13
   1825c40f1:	c4 41 15 58 e9       	vaddpd ymm13,ymm13,ymm9
   1825c40f6:	c4 41 6d 5c cc       	vsubpd ymm9,ymm2,ymm12
   1825c40fb:	c4 c1 6d 58 d4       	vaddpd ymm2,ymm2,ymm12
   1825c4100:	c4 c1 7d 29 14 24    	vmovapd YMMWORD PTR [r12],ymm2
   1825c4106:	c4 c1 7d 28 54 24 20 	vmovapd ymm2,YMMWORD PTR [r12+0x20]
   1825c410d:	c5 6d 5c e5          	vsubpd ymm12,ymm2,ymm5
   1825c4111:	c5 d5 58 ea          	vaddpd ymm5,ymm5,ymm2
   1825c4115:	c4 c1 4d 5c d2       	vsubpd ymm2,ymm6,ymm10
   1825c411a:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c411f:	c5 cd 59 35 79 70 1a 	vmulpd ymm6,ymm6,YMMWORD PTR [rip+0x1a7079]        # 0x18276b1a0
   1825c4126:	00 
   1825c4127:	c5 ed 59 15 91 70 1a 	vmulpd ymm2,ymm2,YMMWORD PTR [rip+0x1a7091]        # 0x18276b1c0
   1825c412e:	00 
   1825c412f:	c4 41 45 5c d6       	vsubpd ymm10,ymm7,ymm14
   1825c4134:	c5 2d 59 15 64 70 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a7064]        # 0x18276b1a0
   1825c413b:	00 
   1825c413c:	c4 c1 45 58 fe       	vaddpd ymm7,ymm7,ymm14
   1825c4141:	c5 c5 59 3d 77 70 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a7077]        # 0x18276b1c0
   1825c4148:	00 
   1825c4149:	c5 55 5c f3          	vsubpd ymm14,ymm5,ymm3
   1825c414d:	c5 e5 58 dd          	vaddpd ymm3,ymm3,ymm5
   1825c4151:	c5 9d 58 e8          	vaddpd ymm5,ymm12,ymm0
   1825c4155:	c5 1d 5c e0          	vsubpd ymm12,ymm12,ymm0
   1825c4159:	c4 c1 65 5c c5       	vsubpd ymm0,ymm3,ymm13
   1825c415e:	c4 c1 65 58 dd       	vaddpd ymm3,ymm3,ymm13
   1825c4163:	c4 c1 7d 29 5c 24 20 	vmovapd YMMWORD PTR [r12+0x20],ymm3
   1825c416a:	c4 41 7d 29 4d 00    	vmovapd YMMWORD PTR [r13+0x0],ymm9
   1825c4170:	c4 c1 7d 29 45 20    	vmovapd YMMWORD PTR [r13+0x20],ymm0
   1825c4176:	c5 ad 5c c6          	vsubpd ymm0,ymm10,ymm6
   1825c417a:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c417f:	c5 45 58 ca          	vaddpd ymm9,ymm7,ymm2
   1825c4183:	c5 ed 5c d7          	vsubpd ymm2,ymm2,ymm7
   1825c4187:	c5 dd 58 d9          	vaddpd ymm3,ymm4,ymm1
   1825c418b:	c4 81 7d 29 1c 54    	vmovapd YMMWORD PTR [r12+r10*2],ymm3
   1825c4191:	c4 41 0d 5c eb       	vsubpd ymm13,ymm14,ymm11
   1825c4196:	c4 01 7d 29 6c 54 20 	vmovapd YMMWORD PTR [r12+r10*2+0x20],ymm13
   1825c419d:	c5 dd 5c d9          	vsubpd ymm3,ymm4,ymm1
   1825c41a1:	c4 81 7d 29 5c 55 00 	vmovapd YMMWORD PTR [r13+r10*2+0x0],ymm3
   1825c41a8:	c4 41 0d 58 eb       	vaddpd ymm13,ymm14,ymm11
   1825c41ad:	c4 01 7d 29 6c 55 20 	vmovapd YMMWORD PTR [r13+r10*2+0x20],ymm13
   1825c41b4:	c5 85 58 de          	vaddpd ymm3,ymm15,ymm6
   1825c41b8:	c4 81 7d 29 1c 14    	vmovapd YMMWORD PTR [r12+r10*1],ymm3
   1825c41be:	c5 1d 58 e8          	vaddpd ymm13,ymm12,ymm0
   1825c41c2:	c4 01 7d 29 6c 14 20 	vmovapd YMMWORD PTR [r12+r10*1+0x20],ymm13
   1825c41c9:	c5 85 5c de          	vsubpd ymm3,ymm15,ymm6
   1825c41cd:	c4 81 7d 29 5c 15 00 	vmovapd YMMWORD PTR [r13+r10*1+0x0],ymm3
   1825c41d4:	c5 1d 5c e8          	vsubpd ymm13,ymm12,ymm0
   1825c41d8:	c4 01 7d 29 6c 15 20 	vmovapd YMMWORD PTR [r13+r10*1+0x20],ymm13
   1825c41df:	c5 bd 58 da          	vaddpd ymm3,ymm8,ymm2
   1825c41e3:	c4 c1 7d 29 1c 04    	vmovapd YMMWORD PTR [r12+rax*1],ymm3
   1825c41e9:	c4 41 55 58 e9       	vaddpd ymm13,ymm5,ymm9
   1825c41ee:	c4 41 7d 29 6c 04 20 	vmovapd YMMWORD PTR [r12+rax*1+0x20],ymm13
   1825c41f5:	c5 bd 5c da          	vsubpd ymm3,ymm8,ymm2
   1825c41f9:	c4 c1 7d 29 5c 05 00 	vmovapd YMMWORD PTR [r13+rax*1+0x0],ymm3
   1825c4200:	c4 41 55 5c e9       	vsubpd ymm13,ymm5,ymm9
   1825c4205:	c4 41 7d 29 6c 05 20 	vmovapd YMMWORD PTR [r13+rax*1+0x20],ymm13
   1825c420c:	49 83 c4 40          	add    r12,0x40
   1825c4210:	49 83 c5 40          	add    r13,0x40
   1825c4214:	4c 3b e7             	cmp    r12,rdi
   1825c4217:	0f 85 1c fd ff ff    	jne    0x1825c3f39
   1825c421d:	4d 8d 64 05 00       	lea    r12,[r13+rax*1+0x0]
   1825c4222:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c4226:	49 ff cb             	dec    r11
   1825c4229:	0f 8f 00 fd ff ff    	jg     0x1825c3f2f
   1825c422f:	48 8b cb             	mov    rcx,rbx
   1825c4232:	e9 0d 03 00 00       	jmp    0x1825c4544
   1825c4237:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c423b:	48 8b fb             	mov    rdi,rbx
   1825c423e:	48 8b d9             	mov    rbx,rcx
   1825c4241:	c4 81 7d 28 14 54    	vmovapd ymm2,YMMWORD PTR [r12+r10*2]
   1825c4247:	c5 fd 28 4b 40       	vmovapd ymm1,YMMWORD PTR [rbx+0x40]
   1825c424c:	c5 f5 59 c2          	vmulpd ymm0,ymm1,ymm2
   1825c4250:	c4 01 7d 28 44 54 20 	vmovapd ymm8,YMMWORD PTR [r12+r10*2+0x20]
   1825c4257:	c4 c1 75 59 c8       	vmulpd ymm1,ymm1,ymm8
   1825c425c:	c5 7d 28 4b 60       	vmovapd ymm9,YMMWORD PTR [rbx+0x60]
   1825c4261:	c4 c2 b5 bc c0       	vfnmadd231pd ymm0,ymm9,ymm8
   1825c4266:	c4 e2 b5 b8 ca       	vfmadd231pd ymm1,ymm9,ymm2
   1825c426b:	c4 c1 7d 28 24 04    	vmovapd ymm4,YMMWORD PTR [r12+rax*1]
   1825c4271:	c5 fd 28 9b 40 01 00 	vmovapd ymm3,YMMWORD PTR [rbx+0x140]
   1825c4278:	00 
   1825c4279:	c5 65 59 f4          	vmulpd ymm14,ymm3,ymm4
   1825c427d:	c4 41 7d 28 54 04 20 	vmovapd ymm10,YMMWORD PTR [r12+rax*1+0x20]
   1825c4284:	c4 c1 65 59 f2       	vmulpd ymm6,ymm3,ymm10
   1825c4289:	c5 7d 28 9b 60 01 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x160]
   1825c4290:	00 
   1825c4291:	c4 42 a5 bc f2       	vfnmadd231pd ymm14,ymm11,ymm10
   1825c4296:	c4 e2 a5 b8 f4       	vfmadd231pd ymm6,ymm11,ymm4
   1825c429b:	c5 8d 58 d0          	vaddpd ymm2,ymm14,ymm0
   1825c429f:	c4 c1 7d 5c c6       	vsubpd ymm0,ymm0,ymm14
   1825c42a4:	c5 cd 58 d9          	vaddpd ymm3,ymm6,ymm1
   1825c42a8:	c5 f5 5c ce          	vsubpd ymm1,ymm1,ymm6
   1825c42ac:	c4 81 7d 28 34 14    	vmovapd ymm6,YMMWORD PTR [r12+r10*1]
   1825c42b2:	c5 fd 28 ab c0 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0xc0]
   1825c42b9:	00 
   1825c42ba:	c5 d5 59 e6          	vmulpd ymm4,ymm5,ymm6
   1825c42be:	c4 01 7d 28 64 14 20 	vmovapd ymm12,YMMWORD PTR [r12+r10*1+0x20]
   1825c42c5:	c4 c1 55 59 ec       	vmulpd ymm5,ymm5,ymm12
   1825c42ca:	c5 7d 28 ab e0 00 00 	vmovapd ymm13,YMMWORD PTR [rbx+0xe0]
   1825c42d1:	00 
   1825c42d2:	c4 c2 95 bc e4       	vfnmadd231pd ymm4,ymm13,ymm12
   1825c42d7:	c4 e2 95 b8 ee       	vfmadd231pd ymm5,ymm13,ymm6
   1825c42dc:	c4 41 7d 28 45 00    	vmovapd ymm8,YMMWORD PTR [r13+0x0]
   1825c42e2:	c5 fd 28 3b          	vmovapd ymm7,YMMWORD PTR [rbx]
   1825c42e6:	c4 c1 45 59 f0       	vmulpd ymm6,ymm7,ymm8
   1825c42eb:	c4 41 7d 28 75 20    	vmovapd ymm14,YMMWORD PTR [r13+0x20]
   1825c42f1:	c4 c1 45 59 fe       	vmulpd ymm7,ymm7,ymm14
   1825c42f6:	c5 7d 28 7b 20       	vmovapd ymm15,YMMWORD PTR [rbx+0x20]
   1825c42fb:	c4 c2 85 bc f6       	vfnmadd231pd ymm6,ymm15,ymm14
   1825c4300:	c4 c2 85 b8 f8       	vfmadd231pd ymm7,ymm15,ymm8
   1825c4305:	c4 01 7d 28 54 15 00 	vmovapd ymm10,YMMWORD PTR [r13+r10*1+0x0]
   1825c430c:	c5 7d 28 a3 00 01 00 	vmovapd ymm12,YMMWORD PTR [rbx+0x100]
   1825c4313:	00 
   1825c4314:	c4 41 1d 59 f2       	vmulpd ymm14,ymm12,ymm10
   1825c4319:	c4 01 7d 28 4c 15 20 	vmovapd ymm9,YMMWORD PTR [r13+r10*1+0x20]
   1825c4320:	c4 41 1d 59 e1       	vmulpd ymm12,ymm12,ymm9
   1825c4325:	c5 7d 28 ab 20 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x120]
   1825c432c:	00 
   1825c432d:	c4 42 95 bc f1       	vfnmadd231pd ymm14,ymm13,ymm9
   1825c4332:	c4 42 95 b8 e2       	vfmadd231pd ymm12,ymm13,ymm10
   1825c4337:	c5 0d 58 c6          	vaddpd ymm8,ymm14,ymm6
   1825c433b:	c4 c1 4d 5c f6       	vsubpd ymm6,ymm6,ymm14
   1825c4340:	c5 1d 58 cf          	vaddpd ymm9,ymm12,ymm7
   1825c4344:	c4 c1 45 5c fc       	vsubpd ymm7,ymm7,ymm12
   1825c4349:	c4 01 7d 28 64 55 00 	vmovapd ymm12,YMMWORD PTR [r13+r10*2+0x0]
   1825c4350:	c5 7d 28 9b 80 00 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x80]
   1825c4357:	00 
   1825c4358:	c4 41 25 59 d4       	vmulpd ymm10,ymm11,ymm12
   1825c435d:	c4 01 7d 28 74 55 20 	vmovapd ymm14,YMMWORD PTR [r13+r10*2+0x20]
   1825c4364:	c4 41 25 59 de       	vmulpd ymm11,ymm11,ymm14
   1825c4369:	c5 7d 28 bb a0 00 00 	vmovapd ymm15,YMMWORD PTR [rbx+0xa0]
   1825c4370:	00 
   1825c4371:	c4 42 85 bc d6       	vfnmadd231pd ymm10,ymm15,ymm14
   1825c4376:	c4 42 85 b8 dc       	vfmadd231pd ymm11,ymm15,ymm12
   1825c437b:	c4 41 7d 28 64 05 00 	vmovapd ymm12,YMMWORD PTR [r13+rax*1+0x0]
   1825c4382:	c5 7d 28 ab 80 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x180]
   1825c4389:	00 
   1825c438a:	c4 41 1d 59 e5       	vmulpd ymm12,ymm12,ymm13
   1825c438f:	c4 41 7d 28 74 05 20 	vmovapd ymm14,YMMWORD PTR [r13+rax*1+0x20]
   1825c4396:	c4 41 15 59 ee       	vmulpd ymm13,ymm13,ymm14
   1825c439b:	c5 7d 28 bb a0 01 00 	vmovapd ymm15,YMMWORD PTR [rbx+0x1a0]
   1825c43a2:	00 
   1825c43a3:	c4 42 85 bc e6       	vfnmadd231pd ymm12,ymm15,ymm14
   1825c43a8:	c4 42 85 b8 6c 05 00 	vfmadd231pd ymm13,ymm15,YMMWORD PTR [r13+rax*1+0x0]
   1825c43af:	48 81 c3 c0 01 00 00 	add    rbx,0x1c0
   1825c43b6:	c4 41 7d 28 34 24    	vmovapd ymm14,YMMWORD PTR [r12]
   1825c43bc:	c5 0d 5c fc          	vsubpd ymm15,ymm14,ymm4
   1825c43c0:	c4 41 5d 58 f6       	vaddpd ymm14,ymm4,ymm14
   1825c43c5:	c5 8d 5c e2          	vsubpd ymm4,ymm14,ymm2
   1825c43c9:	c4 c1 6d 58 d6       	vaddpd ymm2,ymm2,ymm14
   1825c43ce:	c4 41 2d 5c f4       	vsubpd ymm14,ymm10,ymm12
   1825c43d3:	c4 41 1d 58 e2       	vaddpd ymm12,ymm12,ymm10
   1825c43d8:	c4 41 25 5c d5       	vsubpd ymm10,ymm11,ymm13
   1825c43dd:	c4 41 15 58 eb       	vaddpd ymm13,ymm13,ymm11
   1825c43e2:	c4 41 3d 5c dc       	vsubpd ymm11,ymm8,ymm12
   1825c43e7:	c4 41 1d 58 e0       	vaddpd ymm12,ymm12,ymm8
   1825c43ec:	c5 05 5c c1          	vsubpd ymm8,ymm15,ymm1
   1825c43f0:	c5 05 58 f9          	vaddpd ymm15,ymm15,ymm1
   1825c43f4:	c4 c1 35 5c cd       	vsubpd ymm1,ymm9,ymm13
   1825c43f9:	c4 41 15 58 e9       	vaddpd ymm13,ymm13,ymm9
   1825c43fe:	c4 41 6d 5c cc       	vsubpd ymm9,ymm2,ymm12
   1825c4403:	c4 c1 6d 58 d4       	vaddpd ymm2,ymm2,ymm12
   1825c4408:	c4 c1 7d 11 16       	vmovupd YMMWORD PTR [r14],ymm2
   1825c440d:	c4 c1 7d 28 54 24 20 	vmovapd ymm2,YMMWORD PTR [r12+0x20]
   1825c4414:	c5 6d 5c e5          	vsubpd ymm12,ymm2,ymm5
   1825c4418:	c5 d5 58 ea          	vaddpd ymm5,ymm5,ymm2
   1825c441c:	c4 c1 4d 5c d2       	vsubpd ymm2,ymm6,ymm10
   1825c4421:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c4426:	c5 cd 59 35 72 6d 1a 	vmulpd ymm6,ymm6,YMMWORD PTR [rip+0x1a6d72]        # 0x18276b1a0
   1825c442d:	00 
   1825c442e:	c5 ed 59 15 8a 6d 1a 	vmulpd ymm2,ymm2,YMMWORD PTR [rip+0x1a6d8a]        # 0x18276b1c0
   1825c4435:	00 
   1825c4436:	c4 41 45 5c d6       	vsubpd ymm10,ymm7,ymm14
   1825c443b:	c5 2d 59 15 5d 6d 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a6d5d]        # 0x18276b1a0
   1825c4442:	00 
   1825c4443:	c4 c1 45 58 fe       	vaddpd ymm7,ymm7,ymm14
   1825c4448:	c5 c5 59 3d 70 6d 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a6d70]        # 0x18276b1c0
   1825c444f:	00 
   1825c4450:	c5 55 5c f3          	vsubpd ymm14,ymm5,ymm3
   1825c4454:	c5 e5 58 dd          	vaddpd ymm3,ymm3,ymm5
   1825c4458:	c5 9d 58 e8          	vaddpd ymm5,ymm12,ymm0
   1825c445c:	c5 1d 5c e0          	vsubpd ymm12,ymm12,ymm0
   1825c4460:	c4 c1 65 5c c5       	vsubpd ymm0,ymm3,ymm13
   1825c4465:	c4 c1 65 58 dd       	vaddpd ymm3,ymm3,ymm13
   1825c446a:	c4 c1 7d 11 5e 20    	vmovupd YMMWORD PTR [r14+0x20],ymm3
   1825c4470:	c4 41 7d 11 0f       	vmovupd YMMWORD PTR [r15],ymm9
   1825c4475:	c4 c1 7d 11 47 20    	vmovupd YMMWORD PTR [r15+0x20],ymm0
   1825c447b:	c5 ad 5c c6          	vsubpd ymm0,ymm10,ymm6
   1825c447f:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c4484:	c5 45 58 ca          	vaddpd ymm9,ymm7,ymm2
   1825c4488:	c5 ed 5c d7          	vsubpd ymm2,ymm2,ymm7
   1825c448c:	c5 dd 58 d9          	vaddpd ymm3,ymm4,ymm1
   1825c4490:	c4 41 0d 5c eb       	vsubpd ymm13,ymm14,ymm11
   1825c4495:	c4 81 7d 11 1c 56    	vmovupd YMMWORD PTR [r14+r10*2],ymm3
   1825c449b:	c4 01 7d 11 6c 56 20 	vmovupd YMMWORD PTR [r14+r10*2+0x20],ymm13
   1825c44a2:	c5 dd 5c d9          	vsubpd ymm3,ymm4,ymm1
   1825c44a6:	c4 41 0d 58 eb       	vaddpd ymm13,ymm14,ymm11
   1825c44ab:	c4 81 7d 11 1c 57    	vmovupd YMMWORD PTR [r15+r10*2],ymm3
   1825c44b1:	c4 01 7d 11 6c 57 20 	vmovupd YMMWORD PTR [r15+r10*2+0x20],ymm13
   1825c44b8:	c5 85 58 de          	vaddpd ymm3,ymm15,ymm6
   1825c44bc:	c5 1d 58 e8          	vaddpd ymm13,ymm12,ymm0
   1825c44c0:	c4 81 7d 11 1c 16    	vmovupd YMMWORD PTR [r14+r10*1],ymm3
   1825c44c6:	c4 01 7d 11 6c 16 20 	vmovupd YMMWORD PTR [r14+r10*1+0x20],ymm13
   1825c44cd:	c5 85 5c de          	vsubpd ymm3,ymm15,ymm6
   1825c44d1:	c5 1d 5c e8          	vsubpd ymm13,ymm12,ymm0
   1825c44d5:	c4 81 7d 11 1c 17    	vmovupd YMMWORD PTR [r15+r10*1],ymm3
   1825c44db:	c4 01 7d 11 6c 17 20 	vmovupd YMMWORD PTR [r15+r10*1+0x20],ymm13
   1825c44e2:	c5 bd 58 da          	vaddpd ymm3,ymm8,ymm2
   1825c44e6:	c4 41 55 58 e9       	vaddpd ymm13,ymm5,ymm9
   1825c44eb:	c4 c1 7d 11 1c 06    	vmovupd YMMWORD PTR [r14+rax*1],ymm3
   1825c44f1:	c4 41 7d 11 6c 06 20 	vmovupd YMMWORD PTR [r14+rax*1+0x20],ymm13
   1825c44f8:	c5 bd 5c da          	vsubpd ymm3,ymm8,ymm2
   1825c44fc:	c4 41 55 5c e9       	vsubpd ymm13,ymm5,ymm9
   1825c4501:	c4 c1 7d 11 1c 07    	vmovupd YMMWORD PTR [r15+rax*1],ymm3
   1825c4507:	c4 41 7d 11 6c 07 20 	vmovupd YMMWORD PTR [r15+rax*1+0x20],ymm13
   1825c450e:	49 83 c4 40          	add    r12,0x40
   1825c4512:	49 83 c5 40          	add    r13,0x40
   1825c4516:	49 83 c6 40          	add    r14,0x40
   1825c451a:	49 83 c7 40          	add    r15,0x40
   1825c451e:	4c 3b e7             	cmp    r12,rdi
   1825c4521:	0f 85 1a fd ff ff    	jne    0x1825c4241
   1825c4527:	4d 8d 64 05 00       	lea    r12,[r13+rax*1+0x0]
   1825c452c:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c4530:	4d 8d 34 07          	lea    r14,[r15+rax*1]
   1825c4534:	4f 8d 3c 96          	lea    r15,[r14+r10*4]
   1825c4538:	49 ff cb             	dec    r11
   1825c453b:	0f 8f f6 fc ff ff    	jg     0x1825c4237
   1825c4541:	48 8b cb             	mov    rcx,rbx
   1825c4544:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c4549:	49 c1 e2 03          	shl    r10,0x3
   1825c454d:	4c 8b 5c 24 48       	mov    r11,QWORD PTR [rsp+0x48]
   1825c4552:	49 c1 eb 03          	shr    r11,0x3
   1825c4556:	4c 89 5c 24 48       	mov    QWORD PTR [rsp+0x48],r11
   1825c455b:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c4560:	e9 87 f6 ff ff       	jmp    0x1825c3bec
   1825c4565:	48 8b 5c 24 50       	mov    rbx,QWORD PTR [rsp+0x50]
   1825c456a:	48 81 eb 00 04 00 00 	sub    rbx,0x400
   1825c4571:	48 89 5c 24 50       	mov    QWORD PTR [rsp+0x50],rbx
   1825c4576:	48 83 fb 00          	cmp    rbx,0x0
   1825c457a:	7e 2e                	jle    0x1825c45aa
   1825c457c:	48 8b 5c 24 58       	mov    rbx,QWORD PTR [rsp+0x58]
   1825c4581:	48 8b cb             	mov    rcx,rbx
   1825c4584:	49 8b d9             	mov    rbx,r9
   1825c4587:	48 81 c3 00 40 00 00 	add    rbx,0x4000
   1825c458e:	4c 8b cb             	mov    r9,rbx
   1825c4591:	4c 8b 5c 24 40       	mov    r11,QWORD PTR [rsp+0x40]
   1825c4596:	4c 89 5c 24 48       	mov    QWORD PTR [rsp+0x48],r11
   1825c459b:	4c 8b 54 24 30       	mov    r10,QWORD PTR [rsp+0x30]
   1825c45a0:	4c 8b 5c 24 38       	mov    r11,QWORD PTR [rsp+0x38]
   1825c45a5:	e9 42 f6 ff ff       	jmp    0x1825c3bec
   1825c45aa:	48 8b 5c 24 60       	mov    rbx,QWORD PTR [rsp+0x60]
   1825c45af:	4c 8b cb             	mov    r9,rbx
   1825c45b2:	4c 8b 5c 24 48       	mov    r11,QWORD PTR [rsp+0x48]
   1825c45b7:	49 83 fb 04          	cmp    r11,0x4
   1825c45bb:	0f 84 6d 16 00 00    	je     0x1825c5c2e
   1825c45c1:	48 f7 c2 55 55 55 55 	test   rdx,0x55555555
   1825c45c8:	0f 85 6d 06 00 00    	jne    0x1825c4c3b
   1825c45ce:	49 83 fb 10          	cmp    r11,0x10
   1825c45d2:	0f 84 63 06 00 00    	je     0x1825c4c3b
   1825c45d8:	49 c1 eb 03          	shr    r11,0x3
   1825c45dc:	4c 89 54 24 20       	mov    QWORD PTR [rsp+0x20],r10
   1825c45e1:	4c 89 5c 24 28       	mov    QWORD PTR [rsp+0x28],r11
   1825c45e6:	4d 8b e1             	mov    r12,r9
   1825c45e9:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c45ee:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c45f3:	49 c1 e2 04          	shl    r10,0x4
   1825c45f7:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c45fb:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c45ff:	4d 8b f1             	mov    r14,r9
   1825c4602:	4f 8d 3c 96          	lea    r15,[r14+r10*4]
   1825c4606:	49 f7 c6 1f 00 00 00 	test   r14,0x1f
   1825c460d:	0f 85 08 03 00 00    	jne    0x1825c491b
   1825c4613:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c4617:	48 8b fb             	mov    rdi,rbx
   1825c461a:	48 8b d9             	mov    rbx,rcx
   1825c461d:	c4 81 7d 28 14 54    	vmovapd ymm2,YMMWORD PTR [r12+r10*2]
   1825c4623:	c5 fd 28 4b 40       	vmovapd ymm1,YMMWORD PTR [rbx+0x40]
   1825c4628:	c5 f5 59 c2          	vmulpd ymm0,ymm1,ymm2
   1825c462c:	c4 01 7d 28 44 54 20 	vmovapd ymm8,YMMWORD PTR [r12+r10*2+0x20]
   1825c4633:	c4 c1 75 59 c8       	vmulpd ymm1,ymm1,ymm8
   1825c4638:	c5 7d 28 4b 60       	vmovapd ymm9,YMMWORD PTR [rbx+0x60]
   1825c463d:	c4 c2 b5 bc c0       	vfnmadd231pd ymm0,ymm9,ymm8
   1825c4642:	c4 e2 b5 b8 ca       	vfmadd231pd ymm1,ymm9,ymm2
   1825c4647:	c4 c1 7d 28 24 04    	vmovapd ymm4,YMMWORD PTR [r12+rax*1]
   1825c464d:	c5 fd 28 9b 40 01 00 	vmovapd ymm3,YMMWORD PTR [rbx+0x140]
   1825c4654:	00 
   1825c4655:	c5 65 59 f4          	vmulpd ymm14,ymm3,ymm4
   1825c4659:	c4 41 7d 28 54 04 20 	vmovapd ymm10,YMMWORD PTR [r12+rax*1+0x20]
   1825c4660:	c4 c1 65 59 f2       	vmulpd ymm6,ymm3,ymm10
   1825c4665:	c5 7d 28 9b 60 01 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x160]
   1825c466c:	00 
   1825c466d:	c4 42 a5 bc f2       	vfnmadd231pd ymm14,ymm11,ymm10
   1825c4672:	c4 e2 a5 b8 f4       	vfmadd231pd ymm6,ymm11,ymm4
   1825c4677:	c5 8d 58 d0          	vaddpd ymm2,ymm14,ymm0
   1825c467b:	c4 c1 7d 5c c6       	vsubpd ymm0,ymm0,ymm14
   1825c4680:	c5 cd 58 d9          	vaddpd ymm3,ymm6,ymm1
   1825c4684:	c5 f5 5c ce          	vsubpd ymm1,ymm1,ymm6
   1825c4688:	c4 81 7d 28 34 14    	vmovapd ymm6,YMMWORD PTR [r12+r10*1]
   1825c468e:	c5 fd 28 ab c0 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0xc0]
   1825c4695:	00 
   1825c4696:	c5 d5 59 e6          	vmulpd ymm4,ymm5,ymm6
   1825c469a:	c4 01 7d 28 64 14 20 	vmovapd ymm12,YMMWORD PTR [r12+r10*1+0x20]
   1825c46a1:	c4 c1 55 59 ec       	vmulpd ymm5,ymm5,ymm12
   1825c46a6:	c5 7d 28 ab e0 00 00 	vmovapd ymm13,YMMWORD PTR [rbx+0xe0]
   1825c46ad:	00 
   1825c46ae:	c4 c2 95 bc e4       	vfnmadd231pd ymm4,ymm13,ymm12
   1825c46b3:	c4 e2 95 b8 ee       	vfmadd231pd ymm5,ymm13,ymm6
   1825c46b8:	c4 41 7d 28 45 00    	vmovapd ymm8,YMMWORD PTR [r13+0x0]
   1825c46be:	c5 fd 28 3b          	vmovapd ymm7,YMMWORD PTR [rbx]
   1825c46c2:	c4 c1 45 59 f0       	vmulpd ymm6,ymm7,ymm8
   1825c46c7:	c4 41 7d 28 75 20    	vmovapd ymm14,YMMWORD PTR [r13+0x20]
   1825c46cd:	c4 c1 45 59 fe       	vmulpd ymm7,ymm7,ymm14
   1825c46d2:	c5 7d 28 7b 20       	vmovapd ymm15,YMMWORD PTR [rbx+0x20]
   1825c46d7:	c4 c2 85 bc f6       	vfnmadd231pd ymm6,ymm15,ymm14
   1825c46dc:	c4 c2 85 b8 f8       	vfmadd231pd ymm7,ymm15,ymm8
   1825c46e1:	c4 01 7d 28 54 15 00 	vmovapd ymm10,YMMWORD PTR [r13+r10*1+0x0]
   1825c46e8:	c5 7d 28 a3 00 01 00 	vmovapd ymm12,YMMWORD PTR [rbx+0x100]
   1825c46ef:	00 
   1825c46f0:	c4 41 1d 59 f2       	vmulpd ymm14,ymm12,ymm10
   1825c46f5:	c4 01 7d 28 4c 15 20 	vmovapd ymm9,YMMWORD PTR [r13+r10*1+0x20]
   1825c46fc:	c4 41 1d 59 e1       	vmulpd ymm12,ymm12,ymm9
   1825c4701:	c5 7d 28 ab 20 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x120]
   1825c4708:	00 
   1825c4709:	c4 42 95 bc f1       	vfnmadd231pd ymm14,ymm13,ymm9
   1825c470e:	c4 42 95 b8 e2       	vfmadd231pd ymm12,ymm13,ymm10
   1825c4713:	c5 0d 58 c6          	vaddpd ymm8,ymm14,ymm6
   1825c4717:	c4 c1 4d 5c f6       	vsubpd ymm6,ymm6,ymm14
   1825c471c:	c5 1d 58 cf          	vaddpd ymm9,ymm12,ymm7
   1825c4720:	c4 c1 45 5c fc       	vsubpd ymm7,ymm7,ymm12
   1825c4725:	c4 01 7d 28 64 55 00 	vmovapd ymm12,YMMWORD PTR [r13+r10*2+0x0]
   1825c472c:	c5 7d 28 9b 80 00 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x80]
   1825c4733:	00 
   1825c4734:	c4 41 25 59 d4       	vmulpd ymm10,ymm11,ymm12
   1825c4739:	c4 01 7d 28 74 55 20 	vmovapd ymm14,YMMWORD PTR [r13+r10*2+0x20]
   1825c4740:	c4 41 25 59 de       	vmulpd ymm11,ymm11,ymm14
   1825c4745:	c5 7d 28 bb a0 00 00 	vmovapd ymm15,YMMWORD PTR [rbx+0xa0]
   1825c474c:	00 
   1825c474d:	c4 42 85 bc d6       	vfnmadd231pd ymm10,ymm15,ymm14
   1825c4752:	c4 42 85 b8 dc       	vfmadd231pd ymm11,ymm15,ymm12
   1825c4757:	c4 41 7d 28 64 05 00 	vmovapd ymm12,YMMWORD PTR [r13+rax*1+0x0]
   1825c475e:	c5 7d 28 ab 80 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x180]
   1825c4765:	00 
   1825c4766:	c4 41 1d 59 e5       	vmulpd ymm12,ymm12,ymm13
   1825c476b:	c4 41 7d 28 74 05 20 	vmovapd ymm14,YMMWORD PTR [r13+rax*1+0x20]
   1825c4772:	c4 41 15 59 ee       	vmulpd ymm13,ymm13,ymm14
   1825c4777:	c5 7d 28 bb a0 01 00 	vmovapd ymm15,YMMWORD PTR [rbx+0x1a0]
   1825c477e:	00 
   1825c477f:	c4 42 85 bc e6       	vfnmadd231pd ymm12,ymm15,ymm14
   1825c4784:	c4 42 85 b8 6c 05 00 	vfmadd231pd ymm13,ymm15,YMMWORD PTR [r13+rax*1+0x0]
   1825c478b:	48 81 c3 c0 01 00 00 	add    rbx,0x1c0
   1825c4792:	c4 41 7d 28 34 24    	vmovapd ymm14,YMMWORD PTR [r12]
   1825c4798:	c5 0d 5c fc          	vsubpd ymm15,ymm14,ymm4
   1825c479c:	c4 41 5d 58 f6       	vaddpd ymm14,ymm4,ymm14
   1825c47a1:	c5 8d 5c e2          	vsubpd ymm4,ymm14,ymm2
   1825c47a5:	c4 c1 6d 58 d6       	vaddpd ymm2,ymm2,ymm14
   1825c47aa:	c4 41 2d 5c f4       	vsubpd ymm14,ymm10,ymm12
   1825c47af:	c4 41 1d 58 e2       	vaddpd ymm12,ymm12,ymm10
   1825c47b4:	c4 41 25 5c d5       	vsubpd ymm10,ymm11,ymm13
   1825c47b9:	c4 41 15 58 eb       	vaddpd ymm13,ymm13,ymm11
   1825c47be:	c4 41 3d 5c dc       	vsubpd ymm11,ymm8,ymm12
   1825c47c3:	c4 41 1d 58 e0       	vaddpd ymm12,ymm12,ymm8
   1825c47c8:	c5 05 5c c1          	vsubpd ymm8,ymm15,ymm1
   1825c47cc:	c5 05 58 f9          	vaddpd ymm15,ymm15,ymm1
   1825c47d0:	c4 c1 35 5c cd       	vsubpd ymm1,ymm9,ymm13
   1825c47d5:	c4 41 15 58 e9       	vaddpd ymm13,ymm13,ymm9
   1825c47da:	c4 41 6d 5c cc       	vsubpd ymm9,ymm2,ymm12
   1825c47df:	c4 c1 6d 58 d4       	vaddpd ymm2,ymm2,ymm12
   1825c47e4:	c4 c1 7d 29 14 24    	vmovapd YMMWORD PTR [r12],ymm2
   1825c47ea:	c4 c1 7d 28 54 24 20 	vmovapd ymm2,YMMWORD PTR [r12+0x20]
   1825c47f1:	c5 6d 5c e5          	vsubpd ymm12,ymm2,ymm5
   1825c47f5:	c5 d5 58 ea          	vaddpd ymm5,ymm5,ymm2
   1825c47f9:	c4 c1 4d 5c d2       	vsubpd ymm2,ymm6,ymm10
   1825c47fe:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c4803:	c5 cd 59 35 95 69 1a 	vmulpd ymm6,ymm6,YMMWORD PTR [rip+0x1a6995]        # 0x18276b1a0
   1825c480a:	00 
   1825c480b:	c5 ed 59 15 ad 69 1a 	vmulpd ymm2,ymm2,YMMWORD PTR [rip+0x1a69ad]        # 0x18276b1c0
   1825c4812:	00 
   1825c4813:	c4 41 45 5c d6       	vsubpd ymm10,ymm7,ymm14
   1825c4818:	c5 2d 59 15 80 69 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a6980]        # 0x18276b1a0
   1825c481f:	00 
   1825c4820:	c4 c1 45 58 fe       	vaddpd ymm7,ymm7,ymm14
   1825c4825:	c5 c5 59 3d 93 69 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a6993]        # 0x18276b1c0
   1825c482c:	00 
   1825c482d:	c5 55 5c f3          	vsubpd ymm14,ymm5,ymm3
   1825c4831:	c5 e5 58 dd          	vaddpd ymm3,ymm3,ymm5
   1825c4835:	c5 9d 58 e8          	vaddpd ymm5,ymm12,ymm0
   1825c4839:	c5 1d 5c e0          	vsubpd ymm12,ymm12,ymm0
   1825c483d:	c4 c1 65 5c c5       	vsubpd ymm0,ymm3,ymm13
   1825c4842:	c4 c1 65 58 dd       	vaddpd ymm3,ymm3,ymm13
   1825c4847:	c4 c1 7d 29 5c 24 20 	vmovapd YMMWORD PTR [r12+0x20],ymm3
   1825c484e:	c4 41 7d 29 4d 00    	vmovapd YMMWORD PTR [r13+0x0],ymm9
   1825c4854:	c4 c1 7d 29 45 20    	vmovapd YMMWORD PTR [r13+0x20],ymm0
   1825c485a:	c5 ad 5c c6          	vsubpd ymm0,ymm10,ymm6
   1825c485e:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c4863:	c5 45 58 ca          	vaddpd ymm9,ymm7,ymm2
   1825c4867:	c5 ed 5c d7          	vsubpd ymm2,ymm2,ymm7
   1825c486b:	c5 dd 58 d9          	vaddpd ymm3,ymm4,ymm1
   1825c486f:	c4 81 7d 29 1c 54    	vmovapd YMMWORD PTR [r12+r10*2],ymm3
   1825c4875:	c4 41 0d 5c eb       	vsubpd ymm13,ymm14,ymm11
   1825c487a:	c4 01 7d 29 6c 54 20 	vmovapd YMMWORD PTR [r12+r10*2+0x20],ymm13
   1825c4881:	c5 dd 5c d9          	vsubpd ymm3,ymm4,ymm1
   1825c4885:	c4 81 7d 29 5c 55 00 	vmovapd YMMWORD PTR [r13+r10*2+0x0],ymm3
   1825c488c:	c4 41 0d 58 eb       	vaddpd ymm13,ymm14,ymm11
   1825c4891:	c4 01 7d 29 6c 55 20 	vmovapd YMMWORD PTR [r13+r10*2+0x20],ymm13
   1825c4898:	c5 85 58 de          	vaddpd ymm3,ymm15,ymm6
   1825c489c:	c4 81 7d 29 1c 14    	vmovapd YMMWORD PTR [r12+r10*1],ymm3
   1825c48a2:	c5 1d 58 e8          	vaddpd ymm13,ymm12,ymm0
   1825c48a6:	c4 01 7d 29 6c 14 20 	vmovapd YMMWORD PTR [r12+r10*1+0x20],ymm13
   1825c48ad:	c5 85 5c de          	vsubpd ymm3,ymm15,ymm6
   1825c48b1:	c4 81 7d 29 5c 15 00 	vmovapd YMMWORD PTR [r13+r10*1+0x0],ymm3
   1825c48b8:	c5 1d 5c e8          	vsubpd ymm13,ymm12,ymm0
   1825c48bc:	c4 01 7d 29 6c 15 20 	vmovapd YMMWORD PTR [r13+r10*1+0x20],ymm13
   1825c48c3:	c5 bd 58 da          	vaddpd ymm3,ymm8,ymm2
   1825c48c7:	c4 c1 7d 29 1c 04    	vmovapd YMMWORD PTR [r12+rax*1],ymm3
   1825c48cd:	c4 41 55 58 e9       	vaddpd ymm13,ymm5,ymm9
   1825c48d2:	c4 41 7d 29 6c 04 20 	vmovapd YMMWORD PTR [r12+rax*1+0x20],ymm13
   1825c48d9:	c5 bd 5c da          	vsubpd ymm3,ymm8,ymm2
   1825c48dd:	c4 c1 7d 29 5c 05 00 	vmovapd YMMWORD PTR [r13+rax*1+0x0],ymm3
   1825c48e4:	c4 41 55 5c e9       	vsubpd ymm13,ymm5,ymm9
   1825c48e9:	c4 41 7d 29 6c 05 20 	vmovapd YMMWORD PTR [r13+rax*1+0x20],ymm13
   1825c48f0:	49 83 c4 40          	add    r12,0x40
   1825c48f4:	49 83 c5 40          	add    r13,0x40
   1825c48f8:	4c 3b e7             	cmp    r12,rdi
   1825c48fb:	0f 85 1c fd ff ff    	jne    0x1825c461d
   1825c4901:	4d 8d 64 05 00       	lea    r12,[r13+rax*1+0x0]
   1825c4906:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c490a:	49 ff cb             	dec    r11
   1825c490d:	0f 8f 00 fd ff ff    	jg     0x1825c4613
   1825c4913:	48 8b cb             	mov    rcx,rbx
   1825c4916:	e9 0d 03 00 00       	jmp    0x1825c4c28
   1825c491b:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c491f:	48 8b fb             	mov    rdi,rbx
   1825c4922:	48 8b d9             	mov    rbx,rcx
   1825c4925:	c4 81 7d 28 14 54    	vmovapd ymm2,YMMWORD PTR [r12+r10*2]
   1825c492b:	c5 fd 28 4b 40       	vmovapd ymm1,YMMWORD PTR [rbx+0x40]
   1825c4930:	c5 f5 59 c2          	vmulpd ymm0,ymm1,ymm2
   1825c4934:	c4 01 7d 28 44 54 20 	vmovapd ymm8,YMMWORD PTR [r12+r10*2+0x20]
   1825c493b:	c4 c1 75 59 c8       	vmulpd ymm1,ymm1,ymm8
   1825c4940:	c5 7d 28 4b 60       	vmovapd ymm9,YMMWORD PTR [rbx+0x60]
   1825c4945:	c4 c2 b5 bc c0       	vfnmadd231pd ymm0,ymm9,ymm8
   1825c494a:	c4 e2 b5 b8 ca       	vfmadd231pd ymm1,ymm9,ymm2
   1825c494f:	c4 c1 7d 28 24 04    	vmovapd ymm4,YMMWORD PTR [r12+rax*1]
   1825c4955:	c5 fd 28 9b 40 01 00 	vmovapd ymm3,YMMWORD PTR [rbx+0x140]
   1825c495c:	00 
   1825c495d:	c5 65 59 f4          	vmulpd ymm14,ymm3,ymm4
   1825c4961:	c4 41 7d 28 54 04 20 	vmovapd ymm10,YMMWORD PTR [r12+rax*1+0x20]
   1825c4968:	c4 c1 65 59 f2       	vmulpd ymm6,ymm3,ymm10
   1825c496d:	c5 7d 28 9b 60 01 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x160]
   1825c4974:	00 
   1825c4975:	c4 42 a5 bc f2       	vfnmadd231pd ymm14,ymm11,ymm10
   1825c497a:	c4 e2 a5 b8 f4       	vfmadd231pd ymm6,ymm11,ymm4
   1825c497f:	c5 8d 58 d0          	vaddpd ymm2,ymm14,ymm0
   1825c4983:	c4 c1 7d 5c c6       	vsubpd ymm0,ymm0,ymm14
   1825c4988:	c5 cd 58 d9          	vaddpd ymm3,ymm6,ymm1
   1825c498c:	c5 f5 5c ce          	vsubpd ymm1,ymm1,ymm6
   1825c4990:	c4 81 7d 28 34 14    	vmovapd ymm6,YMMWORD PTR [r12+r10*1]
   1825c4996:	c5 fd 28 ab c0 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0xc0]
   1825c499d:	00 
   1825c499e:	c5 d5 59 e6          	vmulpd ymm4,ymm5,ymm6
   1825c49a2:	c4 01 7d 28 64 14 20 	vmovapd ymm12,YMMWORD PTR [r12+r10*1+0x20]
   1825c49a9:	c4 c1 55 59 ec       	vmulpd ymm5,ymm5,ymm12
   1825c49ae:	c5 7d 28 ab e0 00 00 	vmovapd ymm13,YMMWORD PTR [rbx+0xe0]
   1825c49b5:	00 
   1825c49b6:	c4 c2 95 bc e4       	vfnmadd231pd ymm4,ymm13,ymm12
   1825c49bb:	c4 e2 95 b8 ee       	vfmadd231pd ymm5,ymm13,ymm6
   1825c49c0:	c4 41 7d 28 45 00    	vmovapd ymm8,YMMWORD PTR [r13+0x0]
   1825c49c6:	c5 fd 28 3b          	vmovapd ymm7,YMMWORD PTR [rbx]
   1825c49ca:	c4 c1 45 59 f0       	vmulpd ymm6,ymm7,ymm8
   1825c49cf:	c4 41 7d 28 75 20    	vmovapd ymm14,YMMWORD PTR [r13+0x20]
   1825c49d5:	c4 c1 45 59 fe       	vmulpd ymm7,ymm7,ymm14
   1825c49da:	c5 7d 28 7b 20       	vmovapd ymm15,YMMWORD PTR [rbx+0x20]
   1825c49df:	c4 c2 85 bc f6       	vfnmadd231pd ymm6,ymm15,ymm14
   1825c49e4:	c4 c2 85 b8 f8       	vfmadd231pd ymm7,ymm15,ymm8
   1825c49e9:	c4 01 7d 28 54 15 00 	vmovapd ymm10,YMMWORD PTR [r13+r10*1+0x0]
   1825c49f0:	c5 7d 28 a3 00 01 00 	vmovapd ymm12,YMMWORD PTR [rbx+0x100]
   1825c49f7:	00 
   1825c49f8:	c4 41 1d 59 f2       	vmulpd ymm14,ymm12,ymm10
   1825c49fd:	c4 01 7d 28 4c 15 20 	vmovapd ymm9,YMMWORD PTR [r13+r10*1+0x20]
   1825c4a04:	c4 41 1d 59 e1       	vmulpd ymm12,ymm12,ymm9
   1825c4a09:	c5 7d 28 ab 20 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x120]
   1825c4a10:	00 
   1825c4a11:	c4 42 95 bc f1       	vfnmadd231pd ymm14,ymm13,ymm9
   1825c4a16:	c4 42 95 b8 e2       	vfmadd231pd ymm12,ymm13,ymm10
   1825c4a1b:	c5 0d 58 c6          	vaddpd ymm8,ymm14,ymm6
   1825c4a1f:	c4 c1 4d 5c f6       	vsubpd ymm6,ymm6,ymm14
   1825c4a24:	c5 1d 58 cf          	vaddpd ymm9,ymm12,ymm7
   1825c4a28:	c4 c1 45 5c fc       	vsubpd ymm7,ymm7,ymm12
   1825c4a2d:	c4 01 7d 28 64 55 00 	vmovapd ymm12,YMMWORD PTR [r13+r10*2+0x0]
   1825c4a34:	c5 7d 28 9b 80 00 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x80]
   1825c4a3b:	00 
   1825c4a3c:	c4 41 25 59 d4       	vmulpd ymm10,ymm11,ymm12
   1825c4a41:	c4 01 7d 28 74 55 20 	vmovapd ymm14,YMMWORD PTR [r13+r10*2+0x20]
   1825c4a48:	c4 41 25 59 de       	vmulpd ymm11,ymm11,ymm14
   1825c4a4d:	c5 7d 28 bb a0 00 00 	vmovapd ymm15,YMMWORD PTR [rbx+0xa0]
   1825c4a54:	00 
   1825c4a55:	c4 42 85 bc d6       	vfnmadd231pd ymm10,ymm15,ymm14
   1825c4a5a:	c4 42 85 b8 dc       	vfmadd231pd ymm11,ymm15,ymm12
   1825c4a5f:	c4 41 7d 28 64 05 00 	vmovapd ymm12,YMMWORD PTR [r13+rax*1+0x0]
   1825c4a66:	c5 7d 28 ab 80 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x180]
   1825c4a6d:	00 
   1825c4a6e:	c4 41 1d 59 e5       	vmulpd ymm12,ymm12,ymm13
   1825c4a73:	c4 41 7d 28 74 05 20 	vmovapd ymm14,YMMWORD PTR [r13+rax*1+0x20]
   1825c4a7a:	c4 41 15 59 ee       	vmulpd ymm13,ymm13,ymm14
   1825c4a7f:	c5 7d 28 bb a0 01 00 	vmovapd ymm15,YMMWORD PTR [rbx+0x1a0]
   1825c4a86:	00 
   1825c4a87:	c4 42 85 bc e6       	vfnmadd231pd ymm12,ymm15,ymm14
   1825c4a8c:	c4 42 85 b8 6c 05 00 	vfmadd231pd ymm13,ymm15,YMMWORD PTR [r13+rax*1+0x0]
   1825c4a93:	48 81 c3 c0 01 00 00 	add    rbx,0x1c0
   1825c4a9a:	c4 41 7d 28 34 24    	vmovapd ymm14,YMMWORD PTR [r12]
   1825c4aa0:	c5 0d 5c fc          	vsubpd ymm15,ymm14,ymm4
   1825c4aa4:	c4 41 5d 58 f6       	vaddpd ymm14,ymm4,ymm14
   1825c4aa9:	c5 8d 5c e2          	vsubpd ymm4,ymm14,ymm2
   1825c4aad:	c4 c1 6d 58 d6       	vaddpd ymm2,ymm2,ymm14
   1825c4ab2:	c4 41 2d 5c f4       	vsubpd ymm14,ymm10,ymm12
   1825c4ab7:	c4 41 1d 58 e2       	vaddpd ymm12,ymm12,ymm10
   1825c4abc:	c4 41 25 5c d5       	vsubpd ymm10,ymm11,ymm13
   1825c4ac1:	c4 41 15 58 eb       	vaddpd ymm13,ymm13,ymm11
   1825c4ac6:	c4 41 3d 5c dc       	vsubpd ymm11,ymm8,ymm12
   1825c4acb:	c4 41 1d 58 e0       	vaddpd ymm12,ymm12,ymm8
   1825c4ad0:	c5 05 5c c1          	vsubpd ymm8,ymm15,ymm1
   1825c4ad4:	c5 05 58 f9          	vaddpd ymm15,ymm15,ymm1
   1825c4ad8:	c4 c1 35 5c cd       	vsubpd ymm1,ymm9,ymm13
   1825c4add:	c4 41 15 58 e9       	vaddpd ymm13,ymm13,ymm9
   1825c4ae2:	c4 41 6d 5c cc       	vsubpd ymm9,ymm2,ymm12
   1825c4ae7:	c4 c1 6d 58 d4       	vaddpd ymm2,ymm2,ymm12
   1825c4aec:	c4 c1 7d 11 16       	vmovupd YMMWORD PTR [r14],ymm2
   1825c4af1:	c4 c1 7d 28 54 24 20 	vmovapd ymm2,YMMWORD PTR [r12+0x20]
   1825c4af8:	c5 6d 5c e5          	vsubpd ymm12,ymm2,ymm5
   1825c4afc:	c5 d5 58 ea          	vaddpd ymm5,ymm5,ymm2
   1825c4b00:	c4 c1 4d 5c d2       	vsubpd ymm2,ymm6,ymm10
   1825c4b05:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c4b0a:	c5 cd 59 35 8e 66 1a 	vmulpd ymm6,ymm6,YMMWORD PTR [rip+0x1a668e]        # 0x18276b1a0
   1825c4b11:	00 
   1825c4b12:	c5 ed 59 15 a6 66 1a 	vmulpd ymm2,ymm2,YMMWORD PTR [rip+0x1a66a6]        # 0x18276b1c0
   1825c4b19:	00 
   1825c4b1a:	c4 41 45 5c d6       	vsubpd ymm10,ymm7,ymm14
   1825c4b1f:	c5 2d 59 15 79 66 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a6679]        # 0x18276b1a0
   1825c4b26:	00 
   1825c4b27:	c4 c1 45 58 fe       	vaddpd ymm7,ymm7,ymm14
   1825c4b2c:	c5 c5 59 3d 8c 66 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a668c]        # 0x18276b1c0
   1825c4b33:	00 
   1825c4b34:	c5 55 5c f3          	vsubpd ymm14,ymm5,ymm3
   1825c4b38:	c5 e5 58 dd          	vaddpd ymm3,ymm3,ymm5
   1825c4b3c:	c5 9d 58 e8          	vaddpd ymm5,ymm12,ymm0
   1825c4b40:	c5 1d 5c e0          	vsubpd ymm12,ymm12,ymm0
   1825c4b44:	c4 c1 65 5c c5       	vsubpd ymm0,ymm3,ymm13
   1825c4b49:	c4 c1 65 58 dd       	vaddpd ymm3,ymm3,ymm13
   1825c4b4e:	c4 c1 7d 11 5e 20    	vmovupd YMMWORD PTR [r14+0x20],ymm3
   1825c4b54:	c4 41 7d 11 0f       	vmovupd YMMWORD PTR [r15],ymm9
   1825c4b59:	c4 c1 7d 11 47 20    	vmovupd YMMWORD PTR [r15+0x20],ymm0
   1825c4b5f:	c5 ad 5c c6          	vsubpd ymm0,ymm10,ymm6
   1825c4b63:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c4b68:	c5 45 58 ca          	vaddpd ymm9,ymm7,ymm2
   1825c4b6c:	c5 ed 5c d7          	vsubpd ymm2,ymm2,ymm7
   1825c4b70:	c5 dd 58 d9          	vaddpd ymm3,ymm4,ymm1
   1825c4b74:	c4 41 0d 5c eb       	vsubpd ymm13,ymm14,ymm11
   1825c4b79:	c4 81 7d 11 1c 56    	vmovupd YMMWORD PTR [r14+r10*2],ymm3
   1825c4b7f:	c4 01 7d 11 6c 56 20 	vmovupd YMMWORD PTR [r14+r10*2+0x20],ymm13
   1825c4b86:	c5 dd 5c d9          	vsubpd ymm3,ymm4,ymm1
   1825c4b8a:	c4 41 0d 58 eb       	vaddpd ymm13,ymm14,ymm11
   1825c4b8f:	c4 81 7d 11 1c 57    	vmovupd YMMWORD PTR [r15+r10*2],ymm3
   1825c4b95:	c4 01 7d 11 6c 57 20 	vmovupd YMMWORD PTR [r15+r10*2+0x20],ymm13
   1825c4b9c:	c5 85 58 de          	vaddpd ymm3,ymm15,ymm6
   1825c4ba0:	c5 1d 58 e8          	vaddpd ymm13,ymm12,ymm0
   1825c4ba4:	c4 81 7d 11 1c 16    	vmovupd YMMWORD PTR [r14+r10*1],ymm3
   1825c4baa:	c4 01 7d 11 6c 16 20 	vmovupd YMMWORD PTR [r14+r10*1+0x20],ymm13
   1825c4bb1:	c5 85 5c de          	vsubpd ymm3,ymm15,ymm6
   1825c4bb5:	c5 1d 5c e8          	vsubpd ymm13,ymm12,ymm0
   1825c4bb9:	c4 81 7d 11 1c 17    	vmovupd YMMWORD PTR [r15+r10*1],ymm3
   1825c4bbf:	c4 01 7d 11 6c 17 20 	vmovupd YMMWORD PTR [r15+r10*1+0x20],ymm13
   1825c4bc6:	c5 bd 58 da          	vaddpd ymm3,ymm8,ymm2
   1825c4bca:	c4 41 55 58 e9       	vaddpd ymm13,ymm5,ymm9
   1825c4bcf:	c4 c1 7d 11 1c 06    	vmovupd YMMWORD PTR [r14+rax*1],ymm3
   1825c4bd5:	c4 41 7d 11 6c 06 20 	vmovupd YMMWORD PTR [r14+rax*1+0x20],ymm13
   1825c4bdc:	c5 bd 5c da          	vsubpd ymm3,ymm8,ymm2
   1825c4be0:	c4 41 55 5c e9       	vsubpd ymm13,ymm5,ymm9
   1825c4be5:	c4 c1 7d 11 1c 07    	vmovupd YMMWORD PTR [r15+rax*1],ymm3
   1825c4beb:	c4 41 7d 11 6c 07 20 	vmovupd YMMWORD PTR [r15+rax*1+0x20],ymm13
   1825c4bf2:	49 83 c4 40          	add    r12,0x40
   1825c4bf6:	49 83 c5 40          	add    r13,0x40
   1825c4bfa:	49 83 c6 40          	add    r14,0x40
   1825c4bfe:	49 83 c7 40          	add    r15,0x40
   1825c4c02:	4c 3b e7             	cmp    r12,rdi
   1825c4c05:	0f 85 1a fd ff ff    	jne    0x1825c4925
   1825c4c0b:	4d 8d 64 05 00       	lea    r12,[r13+rax*1+0x0]
   1825c4c10:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c4c14:	4d 8d 34 07          	lea    r14,[r15+rax*1]
   1825c4c18:	4f 8d 3c 96          	lea    r15,[r14+r10*4]
   1825c4c1c:	49 ff cb             	dec    r11
   1825c4c1f:	0f 8f f6 fc ff ff    	jg     0x1825c491b
   1825c4c25:	48 8b cb             	mov    rcx,rbx
   1825c4c28:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c4c2d:	49 c1 e2 03          	shl    r10,0x3
   1825c4c31:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c4c36:	e9 7c f9 ff ff       	jmp    0x1825c45b7
   1825c4c3b:	49 c1 eb 02          	shr    r11,0x2
   1825c4c3f:	4c 89 54 24 20       	mov    QWORD PTR [rsp+0x20],r10
   1825c4c44:	4c 89 5c 24 28       	mov    QWORD PTR [rsp+0x28],r11
   1825c4c49:	4d 8b e1             	mov    r12,r9
   1825c4c4c:	4d 8b e9             	mov    r13,r9
   1825c4c4f:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c4c54:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c4c59:	49 c1 e2 04          	shl    r10,0x4
   1825c4c5d:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c4c61:	49 f7 c5 1f 00 00 00 	test   r13,0x1f
   1825c4c68:	0f 85 3b 01 00 00    	jne    0x1825c4da9
   1825c4c6e:	48 8b d9             	mov    rbx,rcx
   1825c4c71:	4c 8b 64 24 20       	mov    r12,QWORD PTR [rsp+0x20]
   1825c4c76:	c4 01 7d 28 44 55 00 	vmovapd ymm8,YMMWORD PTR [r13+r10*2+0x0]
   1825c4c7d:	c5 fd 28 0b          	vmovapd ymm1,YMMWORD PTR [rbx]
   1825c4c81:	c4 c1 75 59 c0       	vmulpd ymm0,ymm1,ymm8
   1825c4c86:	c4 81 7d 28 74 55 20 	vmovapd ymm6,YMMWORD PTR [r13+r10*2+0x20]
   1825c4c8d:	c5 f5 59 ce          	vmulpd ymm1,ymm1,ymm6
   1825c4c91:	c5 fd 28 7b 20       	vmovapd ymm7,YMMWORD PTR [rbx+0x20]
   1825c4c96:	c4 e2 c5 bc c6       	vfnmadd231pd ymm0,ymm7,ymm6
   1825c4c9b:	c4 c2 c5 b8 c8       	vfmadd231pd ymm1,ymm7,ymm8
   1825c4ca0:	c4 41 7d 28 44 05 00 	vmovapd ymm8,YMMWORD PTR [r13+rax*1+0x0]
   1825c4ca7:	c5 fd 28 ab 80 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0x80]
   1825c4cae:	00 
   1825c4caf:	c4 41 55 59 c8       	vmulpd ymm9,ymm5,ymm8
   1825c4cb4:	c4 c1 7d 28 74 05 20 	vmovapd ymm6,YMMWORD PTR [r13+rax*1+0x20]
   1825c4cbb:	c5 55 59 d6          	vmulpd ymm10,ymm5,ymm6
   1825c4cbf:	c5 fd 28 bb a0 00 00 	vmovapd ymm7,YMMWORD PTR [rbx+0xa0]
   1825c4cc6:	00 
   1825c4cc7:	c4 62 c5 bc ce       	vfnmadd231pd ymm9,ymm7,ymm6
   1825c4ccc:	c4 42 c5 b8 d0       	vfmadd231pd ymm10,ymm7,ymm8
   1825c4cd1:	c4 01 7d 28 44 15 00 	vmovapd ymm8,YMMWORD PTR [r13+r10*1+0x0]
   1825c4cd8:	c5 fd 28 5b 40       	vmovapd ymm3,YMMWORD PTR [rbx+0x40]
   1825c4cdd:	c4 c1 65 59 d0       	vmulpd ymm2,ymm3,ymm8
   1825c4ce2:	c4 81 7d 28 74 15 20 	vmovapd ymm6,YMMWORD PTR [r13+r10*1+0x20]
   1825c4ce9:	c5 e5 59 de          	vmulpd ymm3,ymm3,ymm6
   1825c4ced:	c5 fd 28 7b 60       	vmovapd ymm7,YMMWORD PTR [rbx+0x60]
   1825c4cf2:	c4 e2 c5 bc d6       	vfnmadd231pd ymm2,ymm7,ymm6
   1825c4cf7:	c4 c2 c5 b8 d8       	vfmadd231pd ymm3,ymm7,ymm8
   1825c4cfc:	48 81 c3 c0 00 00 00 	add    rbx,0xc0
   1825c4d03:	c5 b5 58 e0          	vaddpd ymm4,ymm9,ymm0
   1825c4d07:	c4 c1 7d 5c c1       	vsubpd ymm0,ymm0,ymm9
   1825c4d0c:	c5 ad 58 e9          	vaddpd ymm5,ymm10,ymm1
   1825c4d10:	c4 c1 75 5c ca       	vsubpd ymm1,ymm1,ymm10
   1825c4d15:	c4 c1 7d 28 7d 00    	vmovapd ymm7,YMMWORD PTR [r13+0x0]
   1825c4d1b:	c5 c5 5c f2          	vsubpd ymm6,ymm7,ymm2
   1825c4d1f:	c5 c5 58 d2          	vaddpd ymm2,ymm7,ymm2
   1825c4d23:	c5 ed 58 fc          	vaddpd ymm7,ymm2,ymm4
   1825c4d27:	c4 c1 7d 29 7d 00    	vmovapd YMMWORD PTR [r13+0x0],ymm7
   1825c4d2d:	c5 ed 5c d4          	vsubpd ymm2,ymm2,ymm4
   1825c4d31:	c4 81 7d 29 54 55 00 	vmovapd YMMWORD PTR [r13+r10*2+0x0],ymm2
   1825c4d38:	c4 c1 7d 28 7d 20    	vmovapd ymm7,YMMWORD PTR [r13+0x20]
   1825c4d3e:	c5 c5 5c e3          	vsubpd ymm4,ymm7,ymm3
   1825c4d42:	c5 c5 58 db          	vaddpd ymm3,ymm7,ymm3
   1825c4d46:	c5 e5 58 fd          	vaddpd ymm7,ymm3,ymm5
   1825c4d4a:	c4 c1 7d 29 7d 20    	vmovapd YMMWORD PTR [r13+0x20],ymm7
   1825c4d50:	c5 e5 5c dd          	vsubpd ymm3,ymm3,ymm5
   1825c4d54:	c4 81 7d 29 5c 55 20 	vmovapd YMMWORD PTR [r13+r10*2+0x20],ymm3
   1825c4d5b:	c5 cd 58 f9          	vaddpd ymm7,ymm6,ymm1
   1825c4d5f:	c4 81 7d 29 7c 15 00 	vmovapd YMMWORD PTR [r13+r10*1+0x0],ymm7
   1825c4d66:	c5 cd 5c f1          	vsubpd ymm6,ymm6,ymm1
   1825c4d6a:	c4 c1 7d 29 74 05 00 	vmovapd YMMWORD PTR [r13+rax*1+0x0],ymm6
   1825c4d71:	c5 dd 5c f8          	vsubpd ymm7,ymm4,ymm0
   1825c4d75:	c4 81 7d 29 7c 15 20 	vmovapd YMMWORD PTR [r13+r10*1+0x20],ymm7
   1825c4d7c:	c5 dd 58 e0          	vaddpd ymm4,ymm4,ymm0
   1825c4d80:	c4 c1 7d 29 64 05 20 	vmovapd YMMWORD PTR [r13+rax*1+0x20],ymm4
   1825c4d87:	49 83 c5 40          	add    r13,0x40
   1825c4d8b:	49 83 ec 04          	sub    r12,0x4
   1825c4d8f:	0f 8f e1 fe ff ff    	jg     0x1825c4c76
   1825c4d95:	4c 03 e8             	add    r13,rax
   1825c4d98:	49 ff cb             	dec    r11
   1825c4d9b:	0f 8f cd fe ff ff    	jg     0x1825c4c6e
   1825c4da1:	48 8b cb             	mov    rcx,rbx
   1825c4da4:	e9 3c 01 00 00       	jmp    0x1825c4ee5
   1825c4da9:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c4dad:	48 8b fb             	mov    rdi,rbx
   1825c4db0:	48 8b d9             	mov    rbx,rcx
   1825c4db3:	c4 01 7d 28 04 54    	vmovapd ymm8,YMMWORD PTR [r12+r10*2]
   1825c4db9:	c5 fd 28 0b          	vmovapd ymm1,YMMWORD PTR [rbx]
   1825c4dbd:	c4 c1 75 59 c0       	vmulpd ymm0,ymm1,ymm8
   1825c4dc2:	c4 81 7d 28 74 54 20 	vmovapd ymm6,YMMWORD PTR [r12+r10*2+0x20]
   1825c4dc9:	c5 f5 59 ce          	vmulpd ymm1,ymm1,ymm6
   1825c4dcd:	c5 fd 28 7b 20       	vmovapd ymm7,YMMWORD PTR [rbx+0x20]
   1825c4dd2:	c4 e2 c5 bc c6       	vfnmadd231pd ymm0,ymm7,ymm6
   1825c4dd7:	c4 c2 c5 b8 c8       	vfmadd231pd ymm1,ymm7,ymm8
   1825c4ddc:	c4 41 7d 28 04 04    	vmovapd ymm8,YMMWORD PTR [r12+rax*1]
   1825c4de2:	c5 fd 28 ab 80 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0x80]
   1825c4de9:	00 
   1825c4dea:	c4 41 55 59 c8       	vmulpd ymm9,ymm5,ymm8
   1825c4def:	c4 c1 7d 28 74 04 20 	vmovapd ymm6,YMMWORD PTR [r12+rax*1+0x20]
   1825c4df6:	c5 55 59 d6          	vmulpd ymm10,ymm5,ymm6
   1825c4dfa:	c5 fd 28 bb a0 00 00 	vmovapd ymm7,YMMWORD PTR [rbx+0xa0]
   1825c4e01:	00 
   1825c4e02:	c4 62 c5 bc ce       	vfnmadd231pd ymm9,ymm7,ymm6
   1825c4e07:	c4 42 c5 b8 d0       	vfmadd231pd ymm10,ymm7,ymm8
   1825c4e0c:	c4 01 7d 28 04 14    	vmovapd ymm8,YMMWORD PTR [r12+r10*1]
   1825c4e12:	c5 fd 28 5b 40       	vmovapd ymm3,YMMWORD PTR [rbx+0x40]
   1825c4e17:	c4 c1 65 59 d0       	vmulpd ymm2,ymm3,ymm8
   1825c4e1c:	c4 81 7d 28 74 14 20 	vmovapd ymm6,YMMWORD PTR [r12+r10*1+0x20]
   1825c4e23:	c5 e5 59 de          	vmulpd ymm3,ymm3,ymm6
   1825c4e27:	c5 fd 28 7b 60       	vmovapd ymm7,YMMWORD PTR [rbx+0x60]
   1825c4e2c:	c4 e2 c5 bc d6       	vfnmadd231pd ymm2,ymm7,ymm6
   1825c4e31:	c4 c2 c5 b8 d8       	vfmadd231pd ymm3,ymm7,ymm8
   1825c4e36:	48 81 c3 c0 00 00 00 	add    rbx,0xc0
   1825c4e3d:	c5 b5 58 e0          	vaddpd ymm4,ymm9,ymm0
   1825c4e41:	c4 c1 7d 5c c1       	vsubpd ymm0,ymm0,ymm9
   1825c4e46:	c5 ad 58 e9          	vaddpd ymm5,ymm10,ymm1
   1825c4e4a:	c4 c1 75 5c ca       	vsubpd ymm1,ymm1,ymm10
   1825c4e4f:	c4 c1 7d 28 3c 24    	vmovapd ymm7,YMMWORD PTR [r12]
   1825c4e55:	c5 c5 5c f2          	vsubpd ymm6,ymm7,ymm2
   1825c4e59:	c5 c5 58 d2          	vaddpd ymm2,ymm7,ymm2
   1825c4e5d:	c5 ed 58 fc          	vaddpd ymm7,ymm2,ymm4
   1825c4e61:	c4 c1 7c 11 7d 00    	vmovups YMMWORD PTR [r13+0x0],ymm7
   1825c4e67:	c5 ed 5c d4          	vsubpd ymm2,ymm2,ymm4
   1825c4e6b:	c4 81 7c 11 54 55 00 	vmovups YMMWORD PTR [r13+r10*2+0x0],ymm2
   1825c4e72:	c4 c1 7d 28 7c 24 20 	vmovapd ymm7,YMMWORD PTR [r12+0x20]
   1825c4e79:	c5 c5 5c e3          	vsubpd ymm4,ymm7,ymm3
   1825c4e7d:	c5 c5 58 db          	vaddpd ymm3,ymm7,ymm3
   1825c4e81:	c5 e5 58 fd          	vaddpd ymm7,ymm3,ymm5
   1825c4e85:	c4 c1 7c 11 7d 20    	vmovups YMMWORD PTR [r13+0x20],ymm7
   1825c4e8b:	c5 e5 5c dd          	vsubpd ymm3,ymm3,ymm5
   1825c4e8f:	c4 81 7c 11 5c 55 20 	vmovups YMMWORD PTR [r13+r10*2+0x20],ymm3
   1825c4e96:	c5 cd 58 f9          	vaddpd ymm7,ymm6,ymm1
   1825c4e9a:	c4 81 7c 11 7c 15 00 	vmovups YMMWORD PTR [r13+r10*1+0x0],ymm7
   1825c4ea1:	c5 cd 5c f1          	vsubpd ymm6,ymm6,ymm1
   1825c4ea5:	c4 c1 7c 11 74 05 00 	vmovups YMMWORD PTR [r13+rax*1+0x0],ymm6
   1825c4eac:	c5 dd 5c f8          	vsubpd ymm7,ymm4,ymm0
   1825c4eb0:	c4 81 7c 11 7c 15 20 	vmovups YMMWORD PTR [r13+r10*1+0x20],ymm7
   1825c4eb7:	c5 dd 58 e0          	vaddpd ymm4,ymm4,ymm0
   1825c4ebb:	c4 c1 7c 11 64 05 20 	vmovups YMMWORD PTR [r13+rax*1+0x20],ymm4
   1825c4ec2:	49 83 c4 40          	add    r12,0x40
   1825c4ec6:	49 83 c5 40          	add    r13,0x40
   1825c4eca:	4c 3b e7             	cmp    r12,rdi
   1825c4ecd:	0f 85 e0 fe ff ff    	jne    0x1825c4db3
   1825c4ed3:	4c 03 e0             	add    r12,rax
   1825c4ed6:	4c 03 e8             	add    r13,rax
   1825c4ed9:	49 ff cb             	dec    r11
   1825c4edc:	0f 8f c7 fe ff ff    	jg     0x1825c4da9
   1825c4ee2:	48 8b cb             	mov    rcx,rbx
   1825c4ee5:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c4eea:	49 c1 e2 02          	shl    r10,0x2
   1825c4eee:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c4ef3:	e9 bf f6 ff ff       	jmp    0x1825c45b7
   1825c4ef8:	49 83 fb 04          	cmp    r11,0x4
   1825c4efc:	0f 84 2c 0d 00 00    	je     0x1825c5c2e
   1825c4f02:	49 83 fb 08          	cmp    r11,0x8
   1825c4f06:	0f 84 25 09 00 00    	je     0x1825c5831
   1825c4f0c:	49 83 fb 10          	cmp    r11,0x10
   1825c4f10:	0f 84 63 06 00 00    	je     0x1825c5579
   1825c4f16:	49 c1 eb 03          	shr    r11,0x3
   1825c4f1a:	4c 89 54 24 20       	mov    QWORD PTR [rsp+0x20],r10
   1825c4f1f:	4c 89 5c 24 28       	mov    QWORD PTR [rsp+0x28],r11
   1825c4f24:	4d 8b e1             	mov    r12,r9
   1825c4f27:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c4f2c:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c4f31:	49 c1 e2 04          	shl    r10,0x4
   1825c4f35:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c4f39:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c4f3d:	4d 8b f1             	mov    r14,r9
   1825c4f40:	4f 8d 3c 96          	lea    r15,[r14+r10*4]
   1825c4f44:	49 f7 c6 1f 00 00 00 	test   r14,0x1f
   1825c4f4b:	0f 85 08 03 00 00    	jne    0x1825c5259
   1825c4f51:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c4f55:	48 8b fb             	mov    rdi,rbx
   1825c4f58:	48 8b d9             	mov    rbx,rcx
   1825c4f5b:	c4 81 7d 28 14 54    	vmovapd ymm2,YMMWORD PTR [r12+r10*2]
   1825c4f61:	c5 fd 28 4b 40       	vmovapd ymm1,YMMWORD PTR [rbx+0x40]
   1825c4f66:	c5 f5 59 c2          	vmulpd ymm0,ymm1,ymm2
   1825c4f6a:	c4 01 7d 28 44 54 20 	vmovapd ymm8,YMMWORD PTR [r12+r10*2+0x20]
   1825c4f71:	c4 c1 75 59 c8       	vmulpd ymm1,ymm1,ymm8
   1825c4f76:	c5 7d 28 4b 60       	vmovapd ymm9,YMMWORD PTR [rbx+0x60]
   1825c4f7b:	c4 c2 b5 bc c0       	vfnmadd231pd ymm0,ymm9,ymm8
   1825c4f80:	c4 e2 b5 b8 ca       	vfmadd231pd ymm1,ymm9,ymm2
   1825c4f85:	c4 c1 7d 28 24 04    	vmovapd ymm4,YMMWORD PTR [r12+rax*1]
   1825c4f8b:	c5 fd 28 9b 40 01 00 	vmovapd ymm3,YMMWORD PTR [rbx+0x140]
   1825c4f92:	00 
   1825c4f93:	c5 65 59 f4          	vmulpd ymm14,ymm3,ymm4
   1825c4f97:	c4 41 7d 28 54 04 20 	vmovapd ymm10,YMMWORD PTR [r12+rax*1+0x20]
   1825c4f9e:	c4 c1 65 59 f2       	vmulpd ymm6,ymm3,ymm10
   1825c4fa3:	c5 7d 28 9b 60 01 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x160]
   1825c4faa:	00 
   1825c4fab:	c4 42 a5 bc f2       	vfnmadd231pd ymm14,ymm11,ymm10
   1825c4fb0:	c4 e2 a5 b8 f4       	vfmadd231pd ymm6,ymm11,ymm4
   1825c4fb5:	c5 8d 58 d0          	vaddpd ymm2,ymm14,ymm0
   1825c4fb9:	c4 c1 7d 5c c6       	vsubpd ymm0,ymm0,ymm14
   1825c4fbe:	c5 cd 58 d9          	vaddpd ymm3,ymm6,ymm1
   1825c4fc2:	c5 f5 5c ce          	vsubpd ymm1,ymm1,ymm6
   1825c4fc6:	c4 81 7d 28 34 14    	vmovapd ymm6,YMMWORD PTR [r12+r10*1]
   1825c4fcc:	c5 fd 28 ab c0 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0xc0]
   1825c4fd3:	00 
   1825c4fd4:	c5 d5 59 e6          	vmulpd ymm4,ymm5,ymm6
   1825c4fd8:	c4 01 7d 28 64 14 20 	vmovapd ymm12,YMMWORD PTR [r12+r10*1+0x20]
   1825c4fdf:	c4 c1 55 59 ec       	vmulpd ymm5,ymm5,ymm12
   1825c4fe4:	c5 7d 28 ab e0 00 00 	vmovapd ymm13,YMMWORD PTR [rbx+0xe0]
   1825c4feb:	00 
   1825c4fec:	c4 c2 95 bc e4       	vfnmadd231pd ymm4,ymm13,ymm12
   1825c4ff1:	c4 e2 95 b8 ee       	vfmadd231pd ymm5,ymm13,ymm6
   1825c4ff6:	c4 41 7d 28 45 00    	vmovapd ymm8,YMMWORD PTR [r13+0x0]
   1825c4ffc:	c5 fd 28 3b          	vmovapd ymm7,YMMWORD PTR [rbx]
   1825c5000:	c4 c1 45 59 f0       	vmulpd ymm6,ymm7,ymm8
   1825c5005:	c4 41 7d 28 75 20    	vmovapd ymm14,YMMWORD PTR [r13+0x20]
   1825c500b:	c4 c1 45 59 fe       	vmulpd ymm7,ymm7,ymm14
   1825c5010:	c5 7d 28 7b 20       	vmovapd ymm15,YMMWORD PTR [rbx+0x20]
   1825c5015:	c4 c2 85 bc f6       	vfnmadd231pd ymm6,ymm15,ymm14
   1825c501a:	c4 c2 85 b8 f8       	vfmadd231pd ymm7,ymm15,ymm8
   1825c501f:	c4 01 7d 28 54 15 00 	vmovapd ymm10,YMMWORD PTR [r13+r10*1+0x0]
   1825c5026:	c5 7d 28 a3 00 01 00 	vmovapd ymm12,YMMWORD PTR [rbx+0x100]
   1825c502d:	00 
   1825c502e:	c4 41 1d 59 f2       	vmulpd ymm14,ymm12,ymm10
   1825c5033:	c4 01 7d 28 4c 15 20 	vmovapd ymm9,YMMWORD PTR [r13+r10*1+0x20]
   1825c503a:	c4 41 1d 59 e1       	vmulpd ymm12,ymm12,ymm9
   1825c503f:	c5 7d 28 ab 20 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x120]
   1825c5046:	00 
   1825c5047:	c4 42 95 bc f1       	vfnmadd231pd ymm14,ymm13,ymm9
   1825c504c:	c4 42 95 b8 e2       	vfmadd231pd ymm12,ymm13,ymm10
   1825c5051:	c5 0d 58 c6          	vaddpd ymm8,ymm14,ymm6
   1825c5055:	c4 c1 4d 5c f6       	vsubpd ymm6,ymm6,ymm14
   1825c505a:	c5 1d 58 cf          	vaddpd ymm9,ymm12,ymm7
   1825c505e:	c4 c1 45 5c fc       	vsubpd ymm7,ymm7,ymm12
   1825c5063:	c4 01 7d 28 64 55 00 	vmovapd ymm12,YMMWORD PTR [r13+r10*2+0x0]
   1825c506a:	c5 7d 28 9b 80 00 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x80]
   1825c5071:	00 
   1825c5072:	c4 41 25 59 d4       	vmulpd ymm10,ymm11,ymm12
   1825c5077:	c4 01 7d 28 74 55 20 	vmovapd ymm14,YMMWORD PTR [r13+r10*2+0x20]
   1825c507e:	c4 41 25 59 de       	vmulpd ymm11,ymm11,ymm14
   1825c5083:	c5 7d 28 bb a0 00 00 	vmovapd ymm15,YMMWORD PTR [rbx+0xa0]
   1825c508a:	00 
   1825c508b:	c4 42 85 bc d6       	vfnmadd231pd ymm10,ymm15,ymm14
   1825c5090:	c4 42 85 b8 dc       	vfmadd231pd ymm11,ymm15,ymm12
   1825c5095:	c4 41 7d 28 64 05 00 	vmovapd ymm12,YMMWORD PTR [r13+rax*1+0x0]
   1825c509c:	c5 7d 28 ab 80 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x180]
   1825c50a3:	00 
   1825c50a4:	c4 41 1d 59 e5       	vmulpd ymm12,ymm12,ymm13
   1825c50a9:	c4 41 7d 28 74 05 20 	vmovapd ymm14,YMMWORD PTR [r13+rax*1+0x20]
   1825c50b0:	c4 41 15 59 ee       	vmulpd ymm13,ymm13,ymm14
   1825c50b5:	c5 7d 28 bb a0 01 00 	vmovapd ymm15,YMMWORD PTR [rbx+0x1a0]
   1825c50bc:	00 
   1825c50bd:	c4 42 85 bc e6       	vfnmadd231pd ymm12,ymm15,ymm14
   1825c50c2:	c4 42 85 b8 6c 05 00 	vfmadd231pd ymm13,ymm15,YMMWORD PTR [r13+rax*1+0x0]
   1825c50c9:	48 81 c3 c0 01 00 00 	add    rbx,0x1c0
   1825c50d0:	c4 41 7d 28 34 24    	vmovapd ymm14,YMMWORD PTR [r12]
   1825c50d6:	c5 0d 5c fc          	vsubpd ymm15,ymm14,ymm4
   1825c50da:	c4 41 5d 58 f6       	vaddpd ymm14,ymm4,ymm14
   1825c50df:	c5 8d 5c e2          	vsubpd ymm4,ymm14,ymm2
   1825c50e3:	c4 c1 6d 58 d6       	vaddpd ymm2,ymm2,ymm14
   1825c50e8:	c4 41 2d 5c f4       	vsubpd ymm14,ymm10,ymm12
   1825c50ed:	c4 41 1d 58 e2       	vaddpd ymm12,ymm12,ymm10
   1825c50f2:	c4 41 25 5c d5       	vsubpd ymm10,ymm11,ymm13
   1825c50f7:	c4 41 15 58 eb       	vaddpd ymm13,ymm13,ymm11
   1825c50fc:	c4 41 3d 5c dc       	vsubpd ymm11,ymm8,ymm12
   1825c5101:	c4 41 1d 58 e0       	vaddpd ymm12,ymm12,ymm8
   1825c5106:	c5 05 5c c1          	vsubpd ymm8,ymm15,ymm1
   1825c510a:	c5 05 58 f9          	vaddpd ymm15,ymm15,ymm1
   1825c510e:	c4 c1 35 5c cd       	vsubpd ymm1,ymm9,ymm13
   1825c5113:	c4 41 15 58 e9       	vaddpd ymm13,ymm13,ymm9
   1825c5118:	c4 41 6d 5c cc       	vsubpd ymm9,ymm2,ymm12
   1825c511d:	c4 c1 6d 58 d4       	vaddpd ymm2,ymm2,ymm12
   1825c5122:	c4 c1 7d 29 14 24    	vmovapd YMMWORD PTR [r12],ymm2
   1825c5128:	c4 c1 7d 28 54 24 20 	vmovapd ymm2,YMMWORD PTR [r12+0x20]
   1825c512f:	c5 6d 5c e5          	vsubpd ymm12,ymm2,ymm5
   1825c5133:	c5 d5 58 ea          	vaddpd ymm5,ymm5,ymm2
   1825c5137:	c4 c1 4d 5c d2       	vsubpd ymm2,ymm6,ymm10
   1825c513c:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c5141:	c5 cd 59 35 57 60 1a 	vmulpd ymm6,ymm6,YMMWORD PTR [rip+0x1a6057]        # 0x18276b1a0
   1825c5148:	00 
   1825c5149:	c5 ed 59 15 6f 60 1a 	vmulpd ymm2,ymm2,YMMWORD PTR [rip+0x1a606f]        # 0x18276b1c0
   1825c5150:	00 
   1825c5151:	c4 41 45 5c d6       	vsubpd ymm10,ymm7,ymm14
   1825c5156:	c5 2d 59 15 42 60 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a6042]        # 0x18276b1a0
   1825c515d:	00 
   1825c515e:	c4 c1 45 58 fe       	vaddpd ymm7,ymm7,ymm14
   1825c5163:	c5 c5 59 3d 55 60 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a6055]        # 0x18276b1c0
   1825c516a:	00 
   1825c516b:	c5 55 5c f3          	vsubpd ymm14,ymm5,ymm3
   1825c516f:	c5 e5 58 dd          	vaddpd ymm3,ymm3,ymm5
   1825c5173:	c5 9d 58 e8          	vaddpd ymm5,ymm12,ymm0
   1825c5177:	c5 1d 5c e0          	vsubpd ymm12,ymm12,ymm0
   1825c517b:	c4 c1 65 5c c5       	vsubpd ymm0,ymm3,ymm13
   1825c5180:	c4 c1 65 58 dd       	vaddpd ymm3,ymm3,ymm13
   1825c5185:	c4 c1 7d 29 5c 24 20 	vmovapd YMMWORD PTR [r12+0x20],ymm3
   1825c518c:	c4 41 7d 29 4d 00    	vmovapd YMMWORD PTR [r13+0x0],ymm9
   1825c5192:	c4 c1 7d 29 45 20    	vmovapd YMMWORD PTR [r13+0x20],ymm0
   1825c5198:	c5 ad 5c c6          	vsubpd ymm0,ymm10,ymm6
   1825c519c:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c51a1:	c5 45 58 ca          	vaddpd ymm9,ymm7,ymm2
   1825c51a5:	c5 ed 5c d7          	vsubpd ymm2,ymm2,ymm7
   1825c51a9:	c5 dd 58 d9          	vaddpd ymm3,ymm4,ymm1
   1825c51ad:	c4 81 7d 29 1c 54    	vmovapd YMMWORD PTR [r12+r10*2],ymm3
   1825c51b3:	c4 41 0d 5c eb       	vsubpd ymm13,ymm14,ymm11
   1825c51b8:	c4 01 7d 29 6c 54 20 	vmovapd YMMWORD PTR [r12+r10*2+0x20],ymm13
   1825c51bf:	c5 dd 5c d9          	vsubpd ymm3,ymm4,ymm1
   1825c51c3:	c4 81 7d 29 5c 55 00 	vmovapd YMMWORD PTR [r13+r10*2+0x0],ymm3
   1825c51ca:	c4 41 0d 58 eb       	vaddpd ymm13,ymm14,ymm11
   1825c51cf:	c4 01 7d 29 6c 55 20 	vmovapd YMMWORD PTR [r13+r10*2+0x20],ymm13
   1825c51d6:	c5 85 58 de          	vaddpd ymm3,ymm15,ymm6
   1825c51da:	c4 81 7d 29 1c 14    	vmovapd YMMWORD PTR [r12+r10*1],ymm3
   1825c51e0:	c5 1d 58 e8          	vaddpd ymm13,ymm12,ymm0
   1825c51e4:	c4 01 7d 29 6c 14 20 	vmovapd YMMWORD PTR [r12+r10*1+0x20],ymm13
   1825c51eb:	c5 85 5c de          	vsubpd ymm3,ymm15,ymm6
   1825c51ef:	c4 81 7d 29 5c 15 00 	vmovapd YMMWORD PTR [r13+r10*1+0x0],ymm3
   1825c51f6:	c5 1d 5c e8          	vsubpd ymm13,ymm12,ymm0
   1825c51fa:	c4 01 7d 29 6c 15 20 	vmovapd YMMWORD PTR [r13+r10*1+0x20],ymm13
   1825c5201:	c5 bd 58 da          	vaddpd ymm3,ymm8,ymm2
   1825c5205:	c4 c1 7d 29 1c 04    	vmovapd YMMWORD PTR [r12+rax*1],ymm3
   1825c520b:	c4 41 55 58 e9       	vaddpd ymm13,ymm5,ymm9
   1825c5210:	c4 41 7d 29 6c 04 20 	vmovapd YMMWORD PTR [r12+rax*1+0x20],ymm13
   1825c5217:	c5 bd 5c da          	vsubpd ymm3,ymm8,ymm2
   1825c521b:	c4 c1 7d 29 5c 05 00 	vmovapd YMMWORD PTR [r13+rax*1+0x0],ymm3
   1825c5222:	c4 41 55 5c e9       	vsubpd ymm13,ymm5,ymm9
   1825c5227:	c4 41 7d 29 6c 05 20 	vmovapd YMMWORD PTR [r13+rax*1+0x20],ymm13
   1825c522e:	49 83 c4 40          	add    r12,0x40
   1825c5232:	49 83 c5 40          	add    r13,0x40
   1825c5236:	4c 3b e7             	cmp    r12,rdi
   1825c5239:	0f 85 1c fd ff ff    	jne    0x1825c4f5b
   1825c523f:	4d 8d 64 05 00       	lea    r12,[r13+rax*1+0x0]
   1825c5244:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c5248:	49 ff cb             	dec    r11
   1825c524b:	0f 8f 00 fd ff ff    	jg     0x1825c4f51
   1825c5251:	48 8b cb             	mov    rcx,rbx
   1825c5254:	e9 0d 03 00 00       	jmp    0x1825c5566
   1825c5259:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c525d:	48 8b fb             	mov    rdi,rbx
   1825c5260:	48 8b d9             	mov    rbx,rcx
   1825c5263:	c4 81 7d 28 14 54    	vmovapd ymm2,YMMWORD PTR [r12+r10*2]
   1825c5269:	c5 fd 28 4b 40       	vmovapd ymm1,YMMWORD PTR [rbx+0x40]
   1825c526e:	c5 f5 59 c2          	vmulpd ymm0,ymm1,ymm2
   1825c5272:	c4 01 7d 28 44 54 20 	vmovapd ymm8,YMMWORD PTR [r12+r10*2+0x20]
   1825c5279:	c4 c1 75 59 c8       	vmulpd ymm1,ymm1,ymm8
   1825c527e:	c5 7d 28 4b 60       	vmovapd ymm9,YMMWORD PTR [rbx+0x60]
   1825c5283:	c4 c2 b5 bc c0       	vfnmadd231pd ymm0,ymm9,ymm8
   1825c5288:	c4 e2 b5 b8 ca       	vfmadd231pd ymm1,ymm9,ymm2
   1825c528d:	c4 c1 7d 28 24 04    	vmovapd ymm4,YMMWORD PTR [r12+rax*1]
   1825c5293:	c5 fd 28 9b 40 01 00 	vmovapd ymm3,YMMWORD PTR [rbx+0x140]
   1825c529a:	00 
   1825c529b:	c5 65 59 f4          	vmulpd ymm14,ymm3,ymm4
   1825c529f:	c4 41 7d 28 54 04 20 	vmovapd ymm10,YMMWORD PTR [r12+rax*1+0x20]
   1825c52a6:	c4 c1 65 59 f2       	vmulpd ymm6,ymm3,ymm10
   1825c52ab:	c5 7d 28 9b 60 01 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x160]
   1825c52b2:	00 
   1825c52b3:	c4 42 a5 bc f2       	vfnmadd231pd ymm14,ymm11,ymm10
   1825c52b8:	c4 e2 a5 b8 f4       	vfmadd231pd ymm6,ymm11,ymm4
   1825c52bd:	c5 8d 58 d0          	vaddpd ymm2,ymm14,ymm0
   1825c52c1:	c4 c1 7d 5c c6       	vsubpd ymm0,ymm0,ymm14
   1825c52c6:	c5 cd 58 d9          	vaddpd ymm3,ymm6,ymm1
   1825c52ca:	c5 f5 5c ce          	vsubpd ymm1,ymm1,ymm6
   1825c52ce:	c4 81 7d 28 34 14    	vmovapd ymm6,YMMWORD PTR [r12+r10*1]
   1825c52d4:	c5 fd 28 ab c0 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0xc0]
   1825c52db:	00 
   1825c52dc:	c5 d5 59 e6          	vmulpd ymm4,ymm5,ymm6
   1825c52e0:	c4 01 7d 28 64 14 20 	vmovapd ymm12,YMMWORD PTR [r12+r10*1+0x20]
   1825c52e7:	c4 c1 55 59 ec       	vmulpd ymm5,ymm5,ymm12
   1825c52ec:	c5 7d 28 ab e0 00 00 	vmovapd ymm13,YMMWORD PTR [rbx+0xe0]
   1825c52f3:	00 
   1825c52f4:	c4 c2 95 bc e4       	vfnmadd231pd ymm4,ymm13,ymm12
   1825c52f9:	c4 e2 95 b8 ee       	vfmadd231pd ymm5,ymm13,ymm6
   1825c52fe:	c4 41 7d 28 45 00    	vmovapd ymm8,YMMWORD PTR [r13+0x0]
   1825c5304:	c5 fd 28 3b          	vmovapd ymm7,YMMWORD PTR [rbx]
   1825c5308:	c4 c1 45 59 f0       	vmulpd ymm6,ymm7,ymm8
   1825c530d:	c4 41 7d 28 75 20    	vmovapd ymm14,YMMWORD PTR [r13+0x20]
   1825c5313:	c4 c1 45 59 fe       	vmulpd ymm7,ymm7,ymm14
   1825c5318:	c5 7d 28 7b 20       	vmovapd ymm15,YMMWORD PTR [rbx+0x20]
   1825c531d:	c4 c2 85 bc f6       	vfnmadd231pd ymm6,ymm15,ymm14
   1825c5322:	c4 c2 85 b8 f8       	vfmadd231pd ymm7,ymm15,ymm8
   1825c5327:	c4 01 7d 28 54 15 00 	vmovapd ymm10,YMMWORD PTR [r13+r10*1+0x0]
   1825c532e:	c5 7d 28 a3 00 01 00 	vmovapd ymm12,YMMWORD PTR [rbx+0x100]
   1825c5335:	00 
   1825c5336:	c4 41 1d 59 f2       	vmulpd ymm14,ymm12,ymm10
   1825c533b:	c4 01 7d 28 4c 15 20 	vmovapd ymm9,YMMWORD PTR [r13+r10*1+0x20]
   1825c5342:	c4 41 1d 59 e1       	vmulpd ymm12,ymm12,ymm9
   1825c5347:	c5 7d 28 ab 20 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x120]
   1825c534e:	00 
   1825c534f:	c4 42 95 bc f1       	vfnmadd231pd ymm14,ymm13,ymm9
   1825c5354:	c4 42 95 b8 e2       	vfmadd231pd ymm12,ymm13,ymm10
   1825c5359:	c5 0d 58 c6          	vaddpd ymm8,ymm14,ymm6
   1825c535d:	c4 c1 4d 5c f6       	vsubpd ymm6,ymm6,ymm14
   1825c5362:	c5 1d 58 cf          	vaddpd ymm9,ymm12,ymm7
   1825c5366:	c4 c1 45 5c fc       	vsubpd ymm7,ymm7,ymm12
   1825c536b:	c4 01 7d 28 64 55 00 	vmovapd ymm12,YMMWORD PTR [r13+r10*2+0x0]
   1825c5372:	c5 7d 28 9b 80 00 00 	vmovapd ymm11,YMMWORD PTR [rbx+0x80]
   1825c5379:	00 
   1825c537a:	c4 41 25 59 d4       	vmulpd ymm10,ymm11,ymm12
   1825c537f:	c4 01 7d 28 74 55 20 	vmovapd ymm14,YMMWORD PTR [r13+r10*2+0x20]
   1825c5386:	c4 41 25 59 de       	vmulpd ymm11,ymm11,ymm14
   1825c538b:	c5 7d 28 bb a0 00 00 	vmovapd ymm15,YMMWORD PTR [rbx+0xa0]
   1825c5392:	00 
   1825c5393:	c4 42 85 bc d6       	vfnmadd231pd ymm10,ymm15,ymm14
   1825c5398:	c4 42 85 b8 dc       	vfmadd231pd ymm11,ymm15,ymm12
   1825c539d:	c4 41 7d 28 64 05 00 	vmovapd ymm12,YMMWORD PTR [r13+rax*1+0x0]
   1825c53a4:	c5 7d 28 ab 80 01 00 	vmovapd ymm13,YMMWORD PTR [rbx+0x180]
   1825c53ab:	00 
   1825c53ac:	c4 41 1d 59 e5       	vmulpd ymm12,ymm12,ymm13
   1825c53b1:	c4 41 7d 28 74 05 20 	vmovapd ymm14,YMMWORD PTR [r13+rax*1+0x20]
   1825c53b8:	c4 41 15 59 ee       	vmulpd ymm13,ymm13,ymm14
   1825c53bd:	c5 7d 28 bb a0 01 00 	vmovapd ymm15,YMMWORD PTR [rbx+0x1a0]
   1825c53c4:	00 
   1825c53c5:	c4 42 85 bc e6       	vfnmadd231pd ymm12,ymm15,ymm14
   1825c53ca:	c4 42 85 b8 6c 05 00 	vfmadd231pd ymm13,ymm15,YMMWORD PTR [r13+rax*1+0x0]
   1825c53d1:	48 81 c3 c0 01 00 00 	add    rbx,0x1c0
   1825c53d8:	c4 41 7d 28 34 24    	vmovapd ymm14,YMMWORD PTR [r12]
   1825c53de:	c5 0d 5c fc          	vsubpd ymm15,ymm14,ymm4
   1825c53e2:	c4 41 5d 58 f6       	vaddpd ymm14,ymm4,ymm14
   1825c53e7:	c5 8d 5c e2          	vsubpd ymm4,ymm14,ymm2
   1825c53eb:	c4 c1 6d 58 d6       	vaddpd ymm2,ymm2,ymm14
   1825c53f0:	c4 41 2d 5c f4       	vsubpd ymm14,ymm10,ymm12
   1825c53f5:	c4 41 1d 58 e2       	vaddpd ymm12,ymm12,ymm10
   1825c53fa:	c4 41 25 5c d5       	vsubpd ymm10,ymm11,ymm13
   1825c53ff:	c4 41 15 58 eb       	vaddpd ymm13,ymm13,ymm11
   1825c5404:	c4 41 3d 5c dc       	vsubpd ymm11,ymm8,ymm12
   1825c5409:	c4 41 1d 58 e0       	vaddpd ymm12,ymm12,ymm8
   1825c540e:	c5 05 5c c1          	vsubpd ymm8,ymm15,ymm1
   1825c5412:	c5 05 58 f9          	vaddpd ymm15,ymm15,ymm1
   1825c5416:	c4 c1 35 5c cd       	vsubpd ymm1,ymm9,ymm13
   1825c541b:	c4 41 15 58 e9       	vaddpd ymm13,ymm13,ymm9
   1825c5420:	c4 41 6d 5c cc       	vsubpd ymm9,ymm2,ymm12
   1825c5425:	c4 c1 6d 58 d4       	vaddpd ymm2,ymm2,ymm12
   1825c542a:	c4 c1 7d 11 16       	vmovupd YMMWORD PTR [r14],ymm2
   1825c542f:	c4 c1 7d 28 54 24 20 	vmovapd ymm2,YMMWORD PTR [r12+0x20]
   1825c5436:	c5 6d 5c e5          	vsubpd ymm12,ymm2,ymm5
   1825c543a:	c5 d5 58 ea          	vaddpd ymm5,ymm5,ymm2
   1825c543e:	c4 c1 4d 5c d2       	vsubpd ymm2,ymm6,ymm10
   1825c5443:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c5448:	c5 cd 59 35 50 5d 1a 	vmulpd ymm6,ymm6,YMMWORD PTR [rip+0x1a5d50]        # 0x18276b1a0
   1825c544f:	00 
   1825c5450:	c5 ed 59 15 68 5d 1a 	vmulpd ymm2,ymm2,YMMWORD PTR [rip+0x1a5d68]        # 0x18276b1c0
   1825c5457:	00 
   1825c5458:	c4 41 45 5c d6       	vsubpd ymm10,ymm7,ymm14
   1825c545d:	c5 2d 59 15 3b 5d 1a 	vmulpd ymm10,ymm10,YMMWORD PTR [rip+0x1a5d3b]        # 0x18276b1a0
   1825c5464:	00 
   1825c5465:	c4 c1 45 58 fe       	vaddpd ymm7,ymm7,ymm14
   1825c546a:	c5 c5 59 3d 4e 5d 1a 	vmulpd ymm7,ymm7,YMMWORD PTR [rip+0x1a5d4e]        # 0x18276b1c0
   1825c5471:	00 
   1825c5472:	c5 55 5c f3          	vsubpd ymm14,ymm5,ymm3
   1825c5476:	c5 e5 58 dd          	vaddpd ymm3,ymm3,ymm5
   1825c547a:	c5 9d 58 e8          	vaddpd ymm5,ymm12,ymm0
   1825c547e:	c5 1d 5c e0          	vsubpd ymm12,ymm12,ymm0
   1825c5482:	c4 c1 65 5c c5       	vsubpd ymm0,ymm3,ymm13
   1825c5487:	c4 c1 65 58 dd       	vaddpd ymm3,ymm3,ymm13
   1825c548c:	c4 c1 7d 11 5e 20    	vmovupd YMMWORD PTR [r14+0x20],ymm3
   1825c5492:	c4 41 7d 11 0f       	vmovupd YMMWORD PTR [r15],ymm9
   1825c5497:	c4 c1 7d 11 47 20    	vmovupd YMMWORD PTR [r15+0x20],ymm0
   1825c549d:	c5 ad 5c c6          	vsubpd ymm0,ymm10,ymm6
   1825c54a1:	c4 c1 4d 58 f2       	vaddpd ymm6,ymm6,ymm10
   1825c54a6:	c5 45 58 ca          	vaddpd ymm9,ymm7,ymm2
   1825c54aa:	c5 ed 5c d7          	vsubpd ymm2,ymm2,ymm7
   1825c54ae:	c5 dd 58 d9          	vaddpd ymm3,ymm4,ymm1
   1825c54b2:	c4 41 0d 5c eb       	vsubpd ymm13,ymm14,ymm11
   1825c54b7:	c4 81 7d 11 1c 56    	vmovupd YMMWORD PTR [r14+r10*2],ymm3
   1825c54bd:	c4 01 7d 11 6c 56 20 	vmovupd YMMWORD PTR [r14+r10*2+0x20],ymm13
   1825c54c4:	c5 dd 5c d9          	vsubpd ymm3,ymm4,ymm1
   1825c54c8:	c4 41 0d 58 eb       	vaddpd ymm13,ymm14,ymm11
   1825c54cd:	c4 81 7d 11 1c 57    	vmovupd YMMWORD PTR [r15+r10*2],ymm3
   1825c54d3:	c4 01 7d 11 6c 57 20 	vmovupd YMMWORD PTR [r15+r10*2+0x20],ymm13
   1825c54da:	c5 85 58 de          	vaddpd ymm3,ymm15,ymm6
   1825c54de:	c5 1d 58 e8          	vaddpd ymm13,ymm12,ymm0
   1825c54e2:	c4 81 7d 11 1c 16    	vmovupd YMMWORD PTR [r14+r10*1],ymm3
   1825c54e8:	c4 01 7d 11 6c 16 20 	vmovupd YMMWORD PTR [r14+r10*1+0x20],ymm13
   1825c54ef:	c5 85 5c de          	vsubpd ymm3,ymm15,ymm6
   1825c54f3:	c5 1d 5c e8          	vsubpd ymm13,ymm12,ymm0
   1825c54f7:	c4 81 7d 11 1c 17    	vmovupd YMMWORD PTR [r15+r10*1],ymm3
   1825c54fd:	c4 01 7d 11 6c 17 20 	vmovupd YMMWORD PTR [r15+r10*1+0x20],ymm13
   1825c5504:	c5 bd 58 da          	vaddpd ymm3,ymm8,ymm2
   1825c5508:	c4 41 55 58 e9       	vaddpd ymm13,ymm5,ymm9
   1825c550d:	c4 c1 7d 11 1c 06    	vmovupd YMMWORD PTR [r14+rax*1],ymm3
   1825c5513:	c4 41 7d 11 6c 06 20 	vmovupd YMMWORD PTR [r14+rax*1+0x20],ymm13
   1825c551a:	c5 bd 5c da          	vsubpd ymm3,ymm8,ymm2
   1825c551e:	c4 41 55 5c e9       	vsubpd ymm13,ymm5,ymm9
   1825c5523:	c4 c1 7d 11 1c 07    	vmovupd YMMWORD PTR [r15+rax*1],ymm3
   1825c5529:	c4 41 7d 11 6c 07 20 	vmovupd YMMWORD PTR [r15+rax*1+0x20],ymm13
   1825c5530:	49 83 c4 40          	add    r12,0x40
   1825c5534:	49 83 c5 40          	add    r13,0x40
   1825c5538:	49 83 c6 40          	add    r14,0x40
   1825c553c:	49 83 c7 40          	add    r15,0x40
   1825c5540:	4c 3b e7             	cmp    r12,rdi
   1825c5543:	0f 85 1a fd ff ff    	jne    0x1825c5263
   1825c5549:	4d 8d 64 05 00       	lea    r12,[r13+rax*1+0x0]
   1825c554e:	4f 8d 2c 94          	lea    r13,[r12+r10*4]
   1825c5552:	4d 8d 34 07          	lea    r14,[r15+rax*1]
   1825c5556:	4f 8d 3c 96          	lea    r15,[r14+r10*4]
   1825c555a:	49 ff cb             	dec    r11
   1825c555d:	0f 8f f6 fc ff ff    	jg     0x1825c5259
   1825c5563:	48 8b cb             	mov    rcx,rbx
   1825c5566:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c556b:	49 c1 e2 03          	shl    r10,0x3
   1825c556f:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c5574:	e9 7f f9 ff ff       	jmp    0x1825c4ef8
   1825c5579:	49 c1 eb 02          	shr    r11,0x2
   1825c557d:	4c 89 54 24 20       	mov    QWORD PTR [rsp+0x20],r10
   1825c5582:	4c 89 5c 24 28       	mov    QWORD PTR [rsp+0x28],r11
   1825c5587:	4d 8b e1             	mov    r12,r9
   1825c558a:	4d 8b e9             	mov    r13,r9
   1825c558d:	4c 8b 5c 24 28       	mov    r11,QWORD PTR [rsp+0x28]
   1825c5592:	4c 8b 54 24 20       	mov    r10,QWORD PTR [rsp+0x20]
   1825c5597:	49 c1 e2 04          	shl    r10,0x4
   1825c559b:	4b 8d 04 52          	lea    rax,[r10+r10*2]
   1825c559f:	49 f7 c5 1f 00 00 00 	test   r13,0x1f
   1825c55a6:	0f 85 3b 01 00 00    	jne    0x1825c56e7
   1825c55ac:	48 8b d9             	mov    rbx,rcx
   1825c55af:	4c 8b 64 24 20       	mov    r12,QWORD PTR [rsp+0x20]
   1825c55b4:	c4 01 7d 28 44 55 00 	vmovapd ymm8,YMMWORD PTR [r13+r10*2+0x0]
   1825c55bb:	c5 fd 28 0b          	vmovapd ymm1,YMMWORD PTR [rbx]
   1825c55bf:	c4 c1 75 59 c0       	vmulpd ymm0,ymm1,ymm8
   1825c55c4:	c4 81 7d 28 74 55 20 	vmovapd ymm6,YMMWORD PTR [r13+r10*2+0x20]
   1825c55cb:	c5 f5 59 ce          	vmulpd ymm1,ymm1,ymm6
   1825c55cf:	c5 fd 28 7b 20       	vmovapd ymm7,YMMWORD PTR [rbx+0x20]
   1825c55d4:	c4 e2 c5 bc c6       	vfnmadd231pd ymm0,ymm7,ymm6
   1825c55d9:	c4 c2 c5 b8 c8       	vfmadd231pd ymm1,ymm7,ymm8
   1825c55de:	c4 41 7d 28 44 05 00 	vmovapd ymm8,YMMWORD PTR [r13+rax*1+0x0]
   1825c55e5:	c5 fd 28 ab 80 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0x80]
   1825c55ec:	00 
   1825c55ed:	c4 41 55 59 c8       	vmulpd ymm9,ymm5,ymm8
   1825c55f2:	c4 c1 7d 28 74 05 20 	vmovapd ymm6,YMMWORD PTR [r13+rax*1+0x20]
   1825c55f9:	c5 55 59 d6          	vmulpd ymm10,ymm5,ymm6
   1825c55fd:	c5 fd 28 bb a0 00 00 	vmovapd ymm7,YMMWORD PTR [rbx+0xa0]
   1825c5604:	00 
   1825c5605:	c4 62 c5 bc ce       	vfnmadd231pd ymm9,ymm7,ymm6
   1825c560a:	c4 42 c5 b8 d0       	vfmadd231pd ymm10,ymm7,ymm8
   1825c560f:	c4 01 7d 28 44 15 00 	vmovapd ymm8,YMMWORD PTR [r13+r10*1+0x0]
   1825c5616:	c5 fd 28 5b 40       	vmovapd ymm3,YMMWORD PTR [rbx+0x40]
   1825c561b:	c4 c1 65 59 d0       	vmulpd ymm2,ymm3,ymm8
   1825c5620:	c4 81 7d 28 74 15 20 	vmovapd ymm6,YMMWORD PTR [r13+r10*1+0x20]
   1825c5627:	c5 e5 59 de          	vmulpd ymm3,ymm3,ymm6
   1825c562b:	c5 fd 28 7b 60       	vmovapd ymm7,YMMWORD PTR [rbx+0x60]
   1825c5630:	c4 e2 c5 bc d6       	vfnmadd231pd ymm2,ymm7,ymm6
   1825c5635:	c4 c2 c5 b8 d8       	vfmadd231pd ymm3,ymm7,ymm8
   1825c563a:	48 81 c3 c0 00 00 00 	add    rbx,0xc0
   1825c5641:	c5 b5 58 e0          	vaddpd ymm4,ymm9,ymm0
   1825c5645:	c4 c1 7d 5c c1       	vsubpd ymm0,ymm0,ymm9
   1825c564a:	c5 ad 58 e9          	vaddpd ymm5,ymm10,ymm1
   1825c564e:	c4 c1 75 5c ca       	vsubpd ymm1,ymm1,ymm10
   1825c5653:	c4 c1 7d 28 7d 00    	vmovapd ymm7,YMMWORD PTR [r13+0x0]
   1825c5659:	c5 c5 5c f2          	vsubpd ymm6,ymm7,ymm2
   1825c565d:	c5 c5 58 d2          	vaddpd ymm2,ymm7,ymm2
   1825c5661:	c5 ed 58 fc          	vaddpd ymm7,ymm2,ymm4
   1825c5665:	c4 c1 7d 29 7d 00    	vmovapd YMMWORD PTR [r13+0x0],ymm7
   1825c566b:	c5 ed 5c d4          	vsubpd ymm2,ymm2,ymm4
   1825c566f:	c4 81 7d 29 54 55 00 	vmovapd YMMWORD PTR [r13+r10*2+0x0],ymm2
   1825c5676:	c4 c1 7d 28 7d 20    	vmovapd ymm7,YMMWORD PTR [r13+0x20]
   1825c567c:	c5 c5 5c e3          	vsubpd ymm4,ymm7,ymm3
   1825c5680:	c5 c5 58 db          	vaddpd ymm3,ymm7,ymm3
   1825c5684:	c5 e5 58 fd          	vaddpd ymm7,ymm3,ymm5
   1825c5688:	c4 c1 7d 29 7d 20    	vmovapd YMMWORD PTR [r13+0x20],ymm7
   1825c568e:	c5 e5 5c dd          	vsubpd ymm3,ymm3,ymm5
   1825c5692:	c4 81 7d 29 5c 55 20 	vmovapd YMMWORD PTR [r13+r10*2+0x20],ymm3
   1825c5699:	c5 cd 58 f9          	vaddpd ymm7,ymm6,ymm1
   1825c569d:	c4 81 7d 29 7c 15 00 	vmovapd YMMWORD PTR [r13+r10*1+0x0],ymm7
   1825c56a4:	c5 cd 5c f1          	vsubpd ymm6,ymm6,ymm1
   1825c56a8:	c4 c1 7d 29 74 05 00 	vmovapd YMMWORD PTR [r13+rax*1+0x0],ymm6
   1825c56af:	c5 dd 5c f8          	vsubpd ymm7,ymm4,ymm0
   1825c56b3:	c4 81 7d 29 7c 15 20 	vmovapd YMMWORD PTR [r13+r10*1+0x20],ymm7
   1825c56ba:	c5 dd 58 e0          	vaddpd ymm4,ymm4,ymm0
   1825c56be:	c4 c1 7d 29 64 05 20 	vmovapd YMMWORD PTR [r13+rax*1+0x20],ymm4
   1825c56c5:	49 83 c5 40          	add    r13,0x40
   1825c56c9:	49 83 ec 04          	sub    r12,0x4
   1825c56cd:	0f 8f e1 fe ff ff    	jg     0x1825c55b4
   1825c56d3:	4c 03 e8             	add    r13,rax
   1825c56d6:	49 ff cb             	dec    r11
   1825c56d9:	0f 8f cd fe ff ff    	jg     0x1825c55ac
   1825c56df:	48 8b cb             	mov    rcx,rbx
   1825c56e2:	e9 3c 01 00 00       	jmp    0x1825c5823
   1825c56e7:	4b 8d 1c 14          	lea    rbx,[r12+r10*1]
   1825c56eb:	48 8b fb             	mov    rdi,rbx
   1825c56ee:	48 8b d9             	mov    rbx,rcx
   1825c56f1:	c4 01 7d 28 04 54    	vmovapd ymm8,YMMWORD PTR [r12+r10*2]
   1825c56f7:	c5 fd 28 0b          	vmovapd ymm1,YMMWORD PTR [rbx]
   1825c56fb:	c4 c1 75 59 c0       	vmulpd ymm0,ymm1,ymm8
   1825c5700:	c4 81 7d 28 74 54 20 	vmovapd ymm6,YMMWORD PTR [r12+r10*2+0x20]
   1825c5707:	c5 f5 59 ce          	vmulpd ymm1,ymm1,ymm6
   1825c570b:	c5 fd 28 7b 20       	vmovapd ymm7,YMMWORD PTR [rbx+0x20]
   1825c5710:	c4 e2 c5 bc c6       	vfnmadd231pd ymm0,ymm7,ymm6
   1825c5715:	c4 c2 c5 b8 c8       	vfmadd231pd ymm1,ymm7,ymm8
   1825c571a:	c4 41 7d 28 04 04    	vmovapd ymm8,YMMWORD PTR [r12+rax*1]
   1825c5720:	c5 fd 28 ab 80 00 00 	vmovapd ymm5,YMMWORD PTR [rbx+0x80]
   1825c5727:	00 
   1825c5728:	c4 41 55 59 c8       	vmulpd ymm9,ymm5,ymm8
   1825c572d:	c4 c1 7d 28 74 04 20 	vmovapd ymm6,YMMWORD PTR [r12+rax*1+0x20]
   1825c5734:	c5 55 59 d6          	vmulpd ymm10,ymm5,ymm6
   1825c5738:	c5 fd 28 bb a0 00 00 	vmovapd ymm7,YMMWORD PTR [rbx+0xa0]
   1825c573f:	00 
   1825c5740:	c4 62 c5 bc ce       	vfnmadd231pd ymm9,ymm7,ymm6
   1825c5745:	c4 42 c5 b8 d0       	vfmadd231pd ymm10,ymm7,ymm8
   1825c574a:	c4 01 7d 28 04 14    	vmovapd ymm8,YMMWORD PTR [r12+r10*1]
   1825c5750:	c5 fd 28 5b 40       	vmovapd ymm3,YMMWORD PTR [rbx+0x40]
   1825c5755:	c4 c1 65 59 d0       	vmulpd ymm2,ymm3,ymm8
   1825c575a:	c4 81 7d 28 74 14 20 	vmovapd ymm6,YMMWORD PTR [r12+r10*1+0x20]
   1825c5761:	c5 e5 59 de          	vmulpd ymm3,ymm3,ymm6
   1825c5765:	c5 fd 28 7b 60       	vmovapd ymm7,YMMWORD PTR [rbx+0x60]
   1825c576a:	c4 e2 c5 bc d6       	vfnmadd231pd ymm2,ymm7,ymm6
   1825c576f:	c4 c2 c5 b8 d8       	vfmadd231pd ymm3,ymm7,ymm8
   1825c5774:	48 81 c3 c0 00 00 00 	add    rbx,0xc0
   1825c577b:	c5 b5 58 e0          	vaddpd ymm4,ymm9,ymm0
   1825c577f:	c4 c1 7d 5c c1       	vsubpd ymm0,ymm0,ymm9
   1825c5784:	c5 ad 58 e9          	vaddpd ymm5,ymm10,ymm1
   1825c5788:	c4 c1 75 5c ca       	vsubpd ymm1,ymm1,ymm10
   1825c578d:	c4 c1 7d 28 3c 24    	vmovapd ymm7,YMMWORD PTR [r12]
   1825c5793:	c5 c5 5c f2          	vsubpd ymm6,ymm7,ymm2
   1825c5797:	c5 c5 58 d2          	vaddpd ymm2,ymm7,ymm2
   1825c579b:	c5 ed 58 fc          	vaddpd ymm7,ymm2,ymm4
   1825c579f:	c4 c1 7c 11 7d 00    	vmovups YMMWORD PTR [r13+0x0],ymm7
   1825c57a5:	c5 ed 5c d4          	vsubpd ymm2,ymm2,ymm4
   1825c57a9:	c4 81 7c 11 54 55 00 	vmovups YMMWORD PTR [r13+r10*2+0x0],ymm2
   1825c57b0:	c4 c1 7d 28 7c 24 20 	vmovapd ymm7,YMMWORD PTR [r12+0x20]
   1825c57b7:	c5 c5 5c e3          	vsubpd ymm4,ymm7,ymm3
   1825c57bb:	c5 c5 58 db          	vaddpd ymm3,ymm7,ymm3
   1825c57bf:	c5 e5 58 fd          	vaddpd ymm7,ymm3,ymm5
   1825c57c3:	c4 c1 7c 11 7d 20    	vmovups YMMWORD PTR [r13+0x20],ymm7
   1825c57c9:	c5 e5 5c dd          	vsubpd ymm3,ymm3,ymm5
   1825c57cd:	c4 81 7c 11 5c 55 20 	vmovups YMMWORD PTR [r13+r10*2+0x20],ymm3
   1825c57d4:	c5 cd 58 f9          	vaddpd ymm7,ymm6,ymm1
   1825c57d8:	c4 81 7c 11 7c 15 00 	vmovups YMMWORD PTR [r13+r10*1+0x0],ymm7
   1825c57df:	c5 cd 5c f1          	vsubpd ymm6,ymm6,ymm1
   1825c57e3:	c4 c1 7c 11 74 05 00 	vmovups YMMWORD PTR [r13+rax*1+0x0],ymm6
   1825c57ea:	c5 dd 5c f8          	vsubpd ymm7,ymm4,ymm0
   1825c57ee:	c4 81 7c 11 7c 15 20 	vmovups YMMWORD PTR [r13+r10*1+0x20],ymm7
   1825c57f5:	c5 dd 58 e0          	vaddpd ymm4,ymm4,ymm0
   1825c57f9:	c4 c1 7c 11 64 05 20 	vmovups YMMWORD PTR [r13+rax*1+0x20],ymm4
