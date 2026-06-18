
C:/Program Files (x86)/Intel/oneAPI/mkl/latest/bin/mkl_avx2.2.dll:     file format pei-x86-64


Disassembly of section .text:

0000000182549180 <.text+0x2548180>:
   182549180:	f3 0f 1e fa          	endbr64
   182549184:	56                   	push   rsi
   182549185:	57                   	push   rdi
   182549186:	48 81 ec a8 00 00 00 	sub    rsp,0xa8
   18254918d:	c5 78 11 bc 24 90 00 	vmovups XMMWORD PTR [rsp+0x90],xmm15
   182549194:	00 00 
   182549196:	c5 78 11 b4 24 80 00 	vmovups XMMWORD PTR [rsp+0x80],xmm14
   18254919d:	00 00 
   18254919f:	c5 78 11 6c 24 70    	vmovups XMMWORD PTR [rsp+0x70],xmm13
   1825491a5:	c5 78 11 64 24 60    	vmovups XMMWORD PTR [rsp+0x60],xmm12
   1825491ab:	c5 78 11 5c 24 50    	vmovups XMMWORD PTR [rsp+0x50],xmm11
   1825491b1:	c5 78 11 54 24 40    	vmovups XMMWORD PTR [rsp+0x40],xmm10
   1825491b7:	c5 79 11 4c 24 30    	vmovupd XMMWORD PTR [rsp+0x30],xmm9
   1825491bd:	c5 79 11 44 24 20    	vmovupd XMMWORD PTR [rsp+0x20],xmm8
   1825491c3:	c5 f9 11 7c 24 10    	vmovupd XMMWORD PTR [rsp+0x10],xmm7
   1825491c9:	c5 f9 11 34 24       	vmovupd XMMWORD PTR [rsp],xmm6
   1825491ce:	4c 63 c2             	movsxd r8,edx
   1825491d1:	4c 89 c0             	mov    rax,r8
   1825491d4:	48 c1 e0 04          	shl    rax,0x4
   1825491d8:	48 01 c8             	add    rax,rcx
   1825491db:	48 83 c0 c0          	add    rax,0xffffffffffffffc0
   1825491df:	48 83 c1 10          	add    rcx,0x10
   1825491e3:	41 81 f8 ff ff 00 00 	cmp    r8d,0xffff
   1825491ea:	0f 8f e5 00 00 00    	jg     0x1825492d5
   1825491f0:	85 d2                	test   edx,edx
   1825491f2:	0f 8e 3b 02 00 00    	jle    0x182549433
   1825491f8:	45 31 c0             	xor    r8d,r8d
   1825491fb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
   182549200:	c4 a1 7d 10 04 c1    	vmovupd ymm0,YMMWORD PTR [rcx+r8*8]
   182549206:	c4 a1 7d 10 4c c1 20 	vmovupd ymm1,YMMWORD PTR [rcx+r8*8+0x20]
   18254920d:	c4 e3 7d 06 d1 20    	vperm2f128 ymm2,ymm0,ymm1,0x20
   182549213:	c4 e3 7d 06 c1 31    	vperm2f128 ymm0,ymm0,ymm1,0x31
   182549219:	c5 fd 10 08          	vmovupd ymm1,YMMWORD PTR [rax]
   18254921d:	c5 fd 10 58 20       	vmovupd ymm3,YMMWORD PTR [rax+0x20]
   182549222:	c4 e3 65 06 e1 31    	vperm2f128 ymm4,ymm3,ymm1,0x31
   182549228:	c4 e3 65 06 c9 20    	vperm2f128 ymm1,ymm3,ymm1,0x20
   18254922e:	c5 ed 14 d8          	vunpcklpd ymm3,ymm2,ymm0
   182549232:	c5 ed 15 c0          	vunpckhpd ymm0,ymm2,ymm0
   182549236:	c5 dd 14 d1          	vunpcklpd ymm2,ymm4,ymm1
   18254923a:	c5 dd 15 c9          	vunpckhpd ymm1,ymm4,ymm1
   18254923e:	c5 e5 5c e2          	vsubpd ymm4,ymm3,ymm2
   182549242:	c5 fd 58 e9          	vaddpd ymm5,ymm0,ymm1
   182549246:	c4 81 7d 10 34 c1    	vmovupd ymm6,YMMWORD PTR [r9+r8*8]
   18254924c:	c4 81 7d 10 7c c1 20 	vmovupd ymm7,YMMWORD PTR [r9+r8*8+0x20]
   182549253:	c4 63 4d 06 c7 20    	vperm2f128 ymm8,ymm6,ymm7,0x20
   182549259:	c4 e3 4d 06 f7 31    	vperm2f128 ymm6,ymm6,ymm7,0x31
   18254925f:	c5 dd 59 fe          	vmulpd ymm7,ymm4,ymm6
   182549263:	c4 c2 d5 b8 f8       	vfmadd231pd ymm7,ymm5,ymm8
   182549268:	c5 bd 59 e4          	vmulpd ymm4,ymm8,ymm4
   18254926c:	c4 e2 d5 ba e6       	vfmsub231pd ymm4,ymm5,ymm6
   182549271:	c5 ed 58 d7          	vaddpd ymm2,ymm2,ymm7
   182549275:	c5 e5 5c df          	vsubpd ymm3,ymm3,ymm7
   182549279:	c5 dd 5c c9          	vsubpd ymm1,ymm4,ymm1
   18254927d:	c5 dd 5c c0          	vsubpd ymm0,ymm4,ymm0
   182549281:	c5 ed 14 e1          	vunpcklpd ymm4,ymm2,ymm1
   182549285:	c5 ed 15 c9          	vunpckhpd ymm1,ymm2,ymm1
   182549289:	c4 e3 5d 18 d1 01    	vinsertf128 ymm2,ymm4,xmm1,0x1
   18254928f:	c4 a1 7d 11 14 c1    	vmovupd YMMWORD PTR [rcx+r8*8],ymm2
   182549295:	c5 e5 14 d0          	vunpcklpd ymm2,ymm3,ymm0
   182549299:	c4 e3 5d 06 c9 31    	vperm2f128 ymm1,ymm4,ymm1,0x31
   18254929f:	c4 a1 7d 11 4c c1 20 	vmovupd YMMWORD PTR [rcx+r8*8+0x20],ymm1
   1825492a6:	c5 e5 15 c0          	vunpckhpd ymm0,ymm3,ymm0
   1825492aa:	c4 e3 7d 18 ca 01    	vinsertf128 ymm1,ymm0,xmm2,0x1
   1825492b0:	c5 fd 11 48 20       	vmovupd YMMWORD PTR [rax+0x20],ymm1
   1825492b5:	c4 e3 7d 06 c2 31    	vperm2f128 ymm0,ymm0,ymm2,0x31
   1825492bb:	c5 fd 11 00          	vmovupd YMMWORD PTR [rax],ymm0
   1825492bf:	49 83 c0 08          	add    r8,0x8
   1825492c3:	48 83 c0 c0          	add    rax,0xffffffffffffffc0
   1825492c7:	41 39 d0             	cmp    r8d,edx
   1825492ca:	0f 8c 30 ff ff ff    	jl     0x182549200
   1825492d0:	e9 5e 01 00 00       	jmp    0x182549433
   1825492d5:	4d 8d 81 00 20 00 00 	lea    r8,[r9+0x2000]
   1825492dc:	45 31 d2             	xor    r10d,r10d
   1825492df:	c4 e2 7d 19 05 f0 c1 	vbroadcastsd ymm0,QWORD PTR [rip+0xac1f0]        # 0x1825f54d8
   1825492e6:	0a 00 
   1825492e8:	0f 1f 84 00 00 00 00 	nop    DWORD PTR [rax+rax*1+0x0]
   1825492ef:	00 
   1825492f0:	c4 c2 7d 19 08       	vbroadcastsd ymm1,QWORD PTR [r8]
   1825492f5:	c4 c2 7d 19 50 08    	vbroadcastsd ymm2,QWORD PTR [r8+0x8]
   1825492fb:	45 31 db             	xor    r11d,r11d
   1825492fe:	48 89 c6             	mov    rsi,rax
   182549301:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
   182549308:	0f 1f 84 00 00 00 00 
   18254930f:	00 
   182549310:	c4 a1 7d 10 1c d9    	vmovupd ymm3,YMMWORD PTR [rcx+r11*8]
   182549316:	c4 a1 7d 10 64 d9 20 	vmovupd ymm4,YMMWORD PTR [rcx+r11*8+0x20]
   18254931d:	c4 e3 65 06 ec 20    	vperm2f128 ymm5,ymm3,ymm4,0x20
   182549323:	c4 e3 65 06 dc 31    	vperm2f128 ymm3,ymm3,ymm4,0x31
   182549329:	c5 fd 10 26          	vmovupd ymm4,YMMWORD PTR [rsi]
   18254932d:	c5 fd 10 76 20       	vmovupd ymm6,YMMWORD PTR [rsi+0x20]
   182549332:	c4 e3 4d 06 fc 31    	vperm2f128 ymm7,ymm6,ymm4,0x31
   182549338:	c4 e3 4d 06 e4 20    	vperm2f128 ymm4,ymm6,ymm4,0x20
   18254933e:	c5 d5 14 f3          	vunpcklpd ymm6,ymm5,ymm3
   182549342:	c5 d5 15 db          	vunpckhpd ymm3,ymm5,ymm3
   182549346:	c5 c5 14 ec          	vunpcklpd ymm5,ymm7,ymm4
   18254934a:	c5 c5 15 e4          	vunpckhpd ymm4,ymm7,ymm4
   18254934e:	c5 cd 59 f0          	vmulpd ymm6,ymm6,ymm0
   182549352:	c5 e5 59 d8          	vmulpd ymm3,ymm3,ymm0
   182549356:	c5 d5 59 e8          	vmulpd ymm5,ymm5,ymm0
   18254935a:	c5 dd 59 e0          	vmulpd ymm4,ymm4,ymm0
   18254935e:	c5 cd 5c fd          	vsubpd ymm7,ymm6,ymm5
   182549362:	c5 65 58 c4          	vaddpd ymm8,ymm3,ymm4
   182549366:	c5 cd 58 ed          	vaddpd ymm5,ymm6,ymm5
   18254936a:	c5 e5 5c dc          	vsubpd ymm3,ymm3,ymm4
   18254936e:	c4 81 7d 10 24 d9    	vmovupd ymm4,YMMWORD PTR [r9+r11*8]
   182549374:	c4 81 7d 10 74 d9 20 	vmovupd ymm6,YMMWORD PTR [r9+r11*8+0x20]
   18254937b:	c4 63 5d 06 ce 20    	vperm2f128 ymm9,ymm4,ymm6,0x20
   182549381:	c4 e3 5d 06 e6 31    	vperm2f128 ymm4,ymm4,ymm6,0x31
   182549387:	c5 ed 59 f4          	vmulpd ymm6,ymm2,ymm4
   18254938b:	c4 c2 f5 ba f1       	vfmsub231pd ymm6,ymm1,ymm9
   182549390:	c5 f5 59 e4          	vmulpd ymm4,ymm1,ymm4
   182549394:	c4 c2 ed b8 e1       	vfmadd231pd ymm4,ymm2,ymm9
   182549399:	c5 45 59 cc          	vmulpd ymm9,ymm7,ymm4
   18254939d:	c4 62 bd b8 ce       	vfmadd231pd ymm9,ymm8,ymm6
   1825493a2:	c5 c5 59 f6          	vmulpd ymm6,ymm7,ymm6
   1825493a6:	c4 e2 bd ba f4       	vfmsub231pd ymm6,ymm8,ymm4
   1825493ab:	c5 b5 58 e5          	vaddpd ymm4,ymm9,ymm5
   1825493af:	c4 c1 55 5c e9       	vsubpd ymm5,ymm5,ymm9
   1825493b4:	c5 e5 58 fe          	vaddpd ymm7,ymm3,ymm6
   1825493b8:	c5 cd 5c db          	vsubpd ymm3,ymm6,ymm3
   1825493bc:	c5 dd 14 f7          	vunpcklpd ymm6,ymm4,ymm7
   1825493c0:	c5 dd 15 e7          	vunpckhpd ymm4,ymm4,ymm7
   1825493c4:	c4 e3 4d 18 fc 01    	vinsertf128 ymm7,ymm6,xmm4,0x1
   1825493ca:	c4 a1 7d 11 3c d9    	vmovupd YMMWORD PTR [rcx+r11*8],ymm7
   1825493d0:	c5 d5 14 fb          	vunpcklpd ymm7,ymm5,ymm3
   1825493d4:	c4 e3 4d 06 e4 31    	vperm2f128 ymm4,ymm6,ymm4,0x31
   1825493da:	c4 a1 7d 11 64 d9 20 	vmovupd YMMWORD PTR [rcx+r11*8+0x20],ymm4
   1825493e1:	c5 d5 15 db          	vunpckhpd ymm3,ymm5,ymm3
   1825493e5:	c4 e3 65 18 e7 01    	vinsertf128 ymm4,ymm3,xmm7,0x1
   1825493eb:	c5 fd 11 66 20       	vmovupd YMMWORD PTR [rsi+0x20],ymm4
   1825493f0:	c4 e3 65 06 df 31    	vperm2f128 ymm3,ymm3,ymm7,0x31
   1825493f6:	c5 fd 11 1e          	vmovupd YMMWORD PTR [rsi],ymm3
   1825493fa:	49 83 c3 08          	add    r11,0x8
   1825493fe:	41 8d 7b f8          	lea    edi,[r11-0x8]
   182549402:	48 83 c6 c0          	add    rsi,0xffffffffffffffc0
   182549406:	81 ff f8 03 00 00    	cmp    edi,0x3f8
   18254940c:	0f 82 fe fe ff ff    	jb     0x182549310
   182549412:	49 83 c0 10          	add    r8,0x10
   182549416:	48 81 c1 00 20 00 00 	add    rcx,0x2000
   18254941d:	48 05 00 e0 ff ff    	add    rax,0xffffffffffffe000
   182549423:	41 81 c2 00 04 00 00 	add    r10d,0x400
   18254942a:	41 39 d2             	cmp    r10d,edx
   18254942d:	0f 8c bd fe ff ff    	jl     0x1825492f0
   182549433:	c5 f8 77             	vzeroupper
   182549436:	c5 f8 10 34 24       	vmovups xmm6,XMMWORD PTR [rsp]
   18254943b:	c5 f8 10 7c 24 10    	vmovups xmm7,XMMWORD PTR [rsp+0x10]
   182549441:	c5 78 10 44 24 20    	vmovups xmm8,XMMWORD PTR [rsp+0x20]
   182549447:	c5 78 10 4c 24 30    	vmovups xmm9,XMMWORD PTR [rsp+0x30]
   18254944d:	c5 78 10 54 24 40    	vmovups xmm10,XMMWORD PTR [rsp+0x40]
   182549453:	c5 78 10 5c 24 50    	vmovups xmm11,XMMWORD PTR [rsp+0x50]
   182549459:	c5 78 10 64 24 60    	vmovups xmm12,XMMWORD PTR [rsp+0x60]
   18254945f:	c5 78 10 6c 24 70    	vmovups xmm13,XMMWORD PTR [rsp+0x70]
   182549465:	c5 78 10 b4 24 80 00 	vmovups xmm14,XMMWORD PTR [rsp+0x80]
   18254946c:	00 00 
   18254946e:	c5 78 10 bc 24 90 00 	vmovups xmm15,XMMWORD PTR [rsp+0x90]
   182549475:	00 00 
   182549477:	48 81 c4 a8 00 00 00 	add    rsp,0xa8
   18254947e:	5f                   	pop    rdi
   18254947f:	5e                   	pop    rsi
   182549480:	c3                   	ret
   182549481:	cc                   	int3
   182549482:	cc                   	int3
   182549483:	cc                   	int3
   182549484:	cc                   	int3
   182549485:	cc                   	int3
   182549486:	cc                   	int3
   182549487:	cc                   	int3
   182549488:	cc                   	int3
   182549489:	cc                   	int3
   18254948a:	cc                   	int3
   18254948b:	cc                   	int3
   18254948c:	cc                   	int3
   18254948d:	cc                   	int3
   18254948e:	cc                   	int3
   18254948f:	cc                   	int3
   182549490:	f3 0f 1e fa          	endbr64
   182549494:	56                   	push   rsi
   182549495:	57                   	push   rdi
   182549496:	55                   	push   rbp
   182549497:	53                   	push   rbx
   182549498:	48 81 ec a8 00 00 00 	sub    rsp,0xa8
   18254949f:	c5 78 11 bc 24 90 00 	vmovups XMMWORD PTR [rsp+0x90],xmm15
   1825494a6:	00 00 
   1825494a8:	c5 78 11 b4 24 80 00 	vmovups XMMWORD PTR [rsp+0x80],xmm14
   1825494af:	00 00 
   1825494b1:	c5 78 11 6c 24 70    	vmovups XMMWORD PTR [rsp+0x70],xmm13
   1825494b7:	c5 78 11 64 24 60    	vmovups XMMWORD PTR [rsp+0x60],xmm12
   1825494bd:	c5 78 11 5c 24 50    	vmovups XMMWORD PTR [rsp+0x50],xmm11
   1825494c3:	c5 78 11 54 24 40    	vmovups XMMWORD PTR [rsp+0x40],xmm10
   1825494c9:	c5 78 11 4c 24 30    	vmovups XMMWORD PTR [rsp+0x30],xmm9
   1825494cf:	c5 79 11 44 24 20    	vmovupd XMMWORD PTR [rsp+0x20],xmm8
   1825494d5:	c5 f9 11 7c 24 10    	vmovupd XMMWORD PTR [rsp+0x10],xmm7
   1825494db:	c5 f9 11 34 24       	vmovupd XMMWORD PTR [rsp],xmm6
   1825494e0:	48 8b 84 24 f0 00 00 	mov    rax,QWORD PTR [rsp+0xf0]
   1825494e7:	00 
   1825494e8:	4d 63 d0             	movsxd r10,r8d
   1825494eb:	41 81 fa ff ff 00 00 	cmp    r10d,0xffff
   1825494f2:	0f 8f ff 00 00 00    	jg     0x1825495f7
   1825494f8:	45 85 c0             	test   r8d,r8d
   1825494fb:	0f 8e 51 02 00 00    	jle    0x182549752
   182549501:	4e 8d 0c 55 f8 ff ff 	lea    r9,[r10*2-0x8]
   182549508:	ff 
   182549509:	45 31 d2             	xor    r10d,r10d
   18254950c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
   182549510:	c4 a1 7d 10 44 d1 10 	vmovupd ymm0,YMMWORD PTR [rcx+r10*8+0x10]
   182549517:	c4 a1 7d 10 4c d1 30 	vmovupd ymm1,YMMWORD PTR [rcx+r10*8+0x30]
   18254951e:	c4 e3 7d 06 d1 20    	vperm2f128 ymm2,ymm0,ymm1,0x20
   182549524:	c4 e3 7d 06 c1 31    	vperm2f128 ymm0,ymm0,ymm1,0x31
   18254952a:	c4 a1 7d 10 0c c9    	vmovupd ymm1,YMMWORD PTR [rcx+r9*8]
   182549530:	c4 a1 7d 10 5c c9 20 	vmovupd ymm3,YMMWORD PTR [rcx+r9*8+0x20]
   182549537:	c4 e3 65 06 e1 31    	vperm2f128 ymm4,ymm3,ymm1,0x31
   18254953d:	c4 e3 65 06 c9 20    	vperm2f128 ymm1,ymm3,ymm1,0x20
   182549543:	c5 ed 14 d8          	vunpcklpd ymm3,ymm2,ymm0
   182549547:	c5 ed 15 c0          	vunpckhpd ymm0,ymm2,ymm0
   18254954b:	c5 dd 14 d1          	vunpcklpd ymm2,ymm4,ymm1
   18254954f:	c5 dd 15 c9          	vunpckhpd ymm1,ymm4,ymm1
   182549553:	c5 e5 5c e2          	vsubpd ymm4,ymm3,ymm2
   182549557:	c5 fd 58 e9          	vaddpd ymm5,ymm0,ymm1
   18254955b:	c5 e5 58 d2          	vaddpd ymm2,ymm3,ymm2
   18254955f:	c5 fd 5c c1          	vsubpd ymm0,ymm0,ymm1
   182549563:	c4 a1 7d 10 0c d0    	vmovupd ymm1,YMMWORD PTR [rax+r10*8]
   182549569:	c4 a1 7d 10 5c d0 20 	vmovupd ymm3,YMMWORD PTR [rax+r10*8+0x20]
   182549570:	c4 e3 75 06 f3 20    	vperm2f128 ymm6,ymm1,ymm3,0x20
   182549576:	c4 e3 75 06 cb 31    	vperm2f128 ymm1,ymm1,ymm3,0x31
   18254957c:	c5 d5 59 de          	vmulpd ymm3,ymm5,ymm6
   182549580:	c4 e2 dd ba d9       	vfmsub231pd ymm3,ymm4,ymm1
   182549585:	c5 d5 59 c9          	vmulpd ymm1,ymm5,ymm1
   182549589:	c4 e2 dd b8 ce       	vfmadd231pd ymm1,ymm4,ymm6
   18254958e:	c5 ed 58 e3          	vaddpd ymm4,ymm2,ymm3
   182549592:	c5 ed 5c d3          	vsubpd ymm2,ymm2,ymm3
   182549596:	c5 fd 58 d9          	vaddpd ymm3,ymm0,ymm1
   18254959a:	c5 f5 5c c0          	vsubpd ymm0,ymm1,ymm0
   18254959e:	c5 dd 14 cb          	vunpcklpd ymm1,ymm4,ymm3
   1825495a2:	c5 dd 15 db          	vunpckhpd ymm3,ymm4,ymm3
   1825495a6:	c4 e3 75 18 e3 01    	vinsertf128 ymm4,ymm1,xmm3,0x1
   1825495ac:	c4 a1 7d 11 64 d2 10 	vmovupd YMMWORD PTR [rdx+r10*8+0x10],ymm4
   1825495b3:	c5 ed 14 e0          	vunpcklpd ymm4,ymm2,ymm0
   1825495b7:	c4 e3 75 06 cb 31    	vperm2f128 ymm1,ymm1,ymm3,0x31
   1825495bd:	c4 a1 7d 11 4c d2 30 	vmovupd YMMWORD PTR [rdx+r10*8+0x30],ymm1
   1825495c4:	c5 ed 15 c0          	vunpckhpd ymm0,ymm2,ymm0
   1825495c8:	c4 e3 7d 18 cc 01    	vinsertf128 ymm1,ymm0,xmm4,0x1
   1825495ce:	c4 a1 7d 11 4c ca 20 	vmovupd YMMWORD PTR [rdx+r9*8+0x20],ymm1
   1825495d5:	c4 e3 7d 06 c4 31    	vperm2f128 ymm0,ymm0,ymm4,0x31
   1825495db:	c4 a1 7d 11 04 ca    	vmovupd YMMWORD PTR [rdx+r9*8],ymm0
   1825495e1:	49 83 c2 08          	add    r10,0x8
   1825495e5:	49 83 c1 f8          	add    r9,0xfffffffffffffff8
   1825495e9:	45 39 c2             	cmp    r10d,r8d
   1825495ec:	0f 8c 1e ff ff ff    	jl     0x182549510
   1825495f2:	e9 5b 01 00 00       	jmp    0x182549752
   1825495f7:	49 c1 e2 04          	shl    r10,0x4
   1825495fb:	4e 8d 0c 11          	lea    r9,[rcx+r10*1]
   1825495ff:	49 83 c1 c0          	add    r9,0xffffffffffffffc0
   182549603:	48 83 c1 10          	add    rcx,0x10
   182549607:	49 01 d2             	add    r10,rdx
   18254960a:	49 83 c2 c0          	add    r10,0xffffffffffffffc0
   18254960e:	48 83 c2 10          	add    rdx,0x10
   182549612:	4c 8d 98 00 20 00 00 	lea    r11,[rax+0x2000]
   182549619:	31 f6                	xor    esi,esi
   18254961b:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
   182549620:	c4 c2 7d 19 03       	vbroadcastsd ymm0,QWORD PTR [r11]
   182549625:	c4 c2 7d 19 4b 08    	vbroadcastsd ymm1,QWORD PTR [r11+0x8]
   18254962b:	31 ff                	xor    edi,edi
   18254962d:	31 db                	xor    ebx,ebx
   18254962f:	90                   	nop
   182549630:	c5 fd 10 14 d9       	vmovupd ymm2,YMMWORD PTR [rcx+rbx*8]
   182549635:	c5 fd 10 5c d9 20    	vmovupd ymm3,YMMWORD PTR [rcx+rbx*8+0x20]
   18254963b:	c4 e3 6d 06 e3 20    	vperm2f128 ymm4,ymm2,ymm3,0x20
   182549641:	c4 e3 6d 06 d3 31    	vperm2f128 ymm2,ymm2,ymm3,0x31
   182549647:	c4 c1 7d 10 1c 39    	vmovupd ymm3,YMMWORD PTR [r9+rdi*1]
   18254964d:	c4 c1 7d 10 6c 39 20 	vmovupd ymm5,YMMWORD PTR [r9+rdi*1+0x20]
   182549654:	c4 e3 55 06 f3 31    	vperm2f128 ymm6,ymm5,ymm3,0x31
   18254965a:	c4 e3 55 06 db 20    	vperm2f128 ymm3,ymm5,ymm3,0x20
   182549660:	c5 dd 14 ea          	vunpcklpd ymm5,ymm4,ymm2
   182549664:	c5 dd 15 d2          	vunpckhpd ymm2,ymm4,ymm2
   182549668:	c5 cd 14 e3          	vunpcklpd ymm4,ymm6,ymm3
   18254966c:	c5 cd 15 db          	vunpckhpd ymm3,ymm6,ymm3
   182549670:	c5 d5 5c f4          	vsubpd ymm6,ymm5,ymm4
   182549674:	c5 ed 58 fb          	vaddpd ymm7,ymm2,ymm3
   182549678:	c5 d5 58 e4          	vaddpd ymm4,ymm5,ymm4
   18254967c:	c5 ed 5c d3          	vsubpd ymm2,ymm2,ymm3
   182549680:	c5 fd 10 1c d8       	vmovupd ymm3,YMMWORD PTR [rax+rbx*8]
   182549685:	c5 fd 10 6c d8 20    	vmovupd ymm5,YMMWORD PTR [rax+rbx*8+0x20]
   18254968b:	c4 63 65 06 c5 20    	vperm2f128 ymm8,ymm3,ymm5,0x20
   182549691:	c4 e3 65 06 dd 31    	vperm2f128 ymm3,ymm3,ymm5,0x31
   182549697:	c5 f5 59 eb          	vmulpd ymm5,ymm1,ymm3
   18254969b:	c4 c2 fd ba e8       	vfmsub231pd ymm5,ymm0,ymm8
   1825496a0:	c5 fd 59 db          	vmulpd ymm3,ymm0,ymm3
   1825496a4:	c4 c2 f5 b8 d8       	vfmadd231pd ymm3,ymm1,ymm8
   1825496a9:	c5 45 59 c5          	vmulpd ymm8,ymm7,ymm5
   1825496ad:	c4 62 cd ba c3       	vfmsub231pd ymm8,ymm6,ymm3
   1825496b2:	c5 c5 59 db          	vmulpd ymm3,ymm7,ymm3
   1825496b6:	c4 e2 cd b8 dd       	vfmadd231pd ymm3,ymm6,ymm5
   1825496bb:	c5 bd 58 ec          	vaddpd ymm5,ymm8,ymm4
   1825496bf:	c4 c1 5d 5c e0       	vsubpd ymm4,ymm4,ymm8
   1825496c4:	c5 ed 58 f3          	vaddpd ymm6,ymm2,ymm3
   1825496c8:	c5 e5 5c d2          	vsubpd ymm2,ymm3,ymm2
   1825496cc:	c5 d5 14 de          	vunpcklpd ymm3,ymm5,ymm6
   1825496d0:	c5 d5 15 ee          	vunpckhpd ymm5,ymm5,ymm6
   1825496d4:	c4 e3 65 18 f5 01    	vinsertf128 ymm6,ymm3,xmm5,0x1
   1825496da:	c5 fd 11 34 da       	vmovupd YMMWORD PTR [rdx+rbx*8],ymm6
   1825496df:	c5 dd 14 f2          	vunpcklpd ymm6,ymm4,ymm2
   1825496e3:	c4 e3 65 06 dd 31    	vperm2f128 ymm3,ymm3,ymm5,0x31
   1825496e9:	c5 fd 11 5c da 20    	vmovupd YMMWORD PTR [rdx+rbx*8+0x20],ymm3
   1825496ef:	c5 dd 15 d2          	vunpckhpd ymm2,ymm4,ymm2
   1825496f3:	c4 e3 6d 18 de 01    	vinsertf128 ymm3,ymm2,xmm6,0x1
   1825496f9:	c4 c1 7d 11 5c 3a 20 	vmovupd YMMWORD PTR [r10+rdi*1+0x20],ymm3
   182549700:	c4 e3 6d 06 d6 31    	vperm2f128 ymm2,ymm2,ymm6,0x31
   182549706:	c4 c1 7d 11 14 3a    	vmovupd YMMWORD PTR [r10+rdi*1],ymm2
   18254970c:	48 83 c3 08          	add    rbx,0x8
   182549710:	8d 6b f8             	lea    ebp,[rbx-0x8]
   182549713:	48 83 c7 c0          	add    rdi,0xffffffffffffffc0
   182549717:	81 fd f8 03 00 00    	cmp    ebp,0x3f8
   18254971d:	0f 82 0d ff ff ff    	jb     0x182549630
   182549723:	49 83 c3 10          	add    r11,0x10
   182549727:	48 81 c1 00 20 00 00 	add    rcx,0x2000
   18254972e:	49 81 c1 00 e0 ff ff 	add    r9,0xffffffffffffe000
   182549735:	48 81 c2 00 20 00 00 	add    rdx,0x2000
   18254973c:	49 81 c2 00 e0 ff ff 	add    r10,0xffffffffffffe000
   182549743:	81 c6 00 04 00 00    	add    esi,0x400
   182549749:	44 39 c6             	cmp    esi,r8d
   18254974c:	0f 8c ce fe ff ff    	jl     0x182549620
   182549752:	c5 f8 77             	vzeroupper
   182549755:	c5 f8 10 34 24       	vmovups xmm6,XMMWORD PTR [rsp]
   18254975a:	c5 f8 10 7c 24 10    	vmovups xmm7,XMMWORD PTR [rsp+0x10]
   182549760:	c5 78 10 44 24 20    	vmovups xmm8,XMMWORD PTR [rsp+0x20]
   182549766:	c5 78 10 4c 24 30    	vmovups xmm9,XMMWORD PTR [rsp+0x30]
   18254976c:	c5 78 10 54 24 40    	vmovups xmm10,XMMWORD PTR [rsp+0x40]
   182549772:	c5 78 10 5c 24 50    	vmovups xmm11,XMMWORD PTR [rsp+0x50]
   182549778:	c5 78 10 64 24 60    	vmovups xmm12,XMMWORD PTR [rsp+0x60]
   18254977e:	c5 78 10 6c 24 70    	vmovups xmm13,XMMWORD PTR [rsp+0x70]
   182549784:	c5 78 10 b4 24 80 00 	vmovups xmm14,XMMWORD PTR [rsp+0x80]
   18254978b:	00 00 
   18254978d:	c5 78 10 bc 24 90 00 	vmovups xmm15,XMMWORD PTR [rsp+0x90]
   182549794:	00 00 
   182549796:	48 81 c4 a8 00 00 00 	add    rsp,0xa8
   18254979d:	5b                   	pop    rbx
   18254979e:	5d                   	pop    rbp
   18254979f:	5f                   	pop    rdi
   1825497a0:	5e                   	pop    rsi
   1825497a1:	c3                   	ret
   1825497a2:	cc                   	int3
   1825497a3:	cc                   	int3
   1825497a4:	cc                   	int3
   1825497a5:	cc                   	int3
   1825497a6:	cc                   	int3
   1825497a7:	cc                   	int3
   1825497a8:	cc                   	int3
   1825497a9:	cc                   	int3
   1825497aa:	cc                   	int3
   1825497ab:	cc                   	int3
   1825497ac:	cc                   	int3
   1825497ad:	cc                   	int3
   1825497ae:	cc                   	int3
   1825497af:	cc                   	int3
