================================================================
  DFT-8, no-twiddle (n1) — DAG
================================================================

  t0   = x[7].re
  t2   = x[7].im
  t5   = x[3].re
  t6   = x[3].im
  t8   = t6 - t2
  t10  = t5 - t0
  t13  = x[5].re
  t14  = x[5].im
  t16  = x[1].re
  t17  = x[1].im
  t18  = t17 - t14
  t20  = t16 - t13
  t21  = t10 + t18
  t22  = t20 - t8
  t23  = t22 - t21
  t24  = 0.707107
  t28  = t21 + t22
  t31  = x[6].re
  t32  = x[6].im
  t34  = x[2].re
  t35  = x[2].im
  t36  = t35 - t32
  t38  = t34 - t31
  t41  = x[4].re
  t42  = x[4].im
  t44  = x[0].re
  t45  = x[0].im
  t46  = t45 - t42
  t48  = t44 - t41
  t49  = t38 + t46
  t50  = t48 - t36
  t53  = t2 + t6
  t54  = t0 + t5
  t56  = t14 + t17
  t57  = t13 + t16
  t58  = t56 - t53
  t60  = t57 - t54
  t63  = t32 + t35
  t64  = t31 + t34
  t66  = t42 + t45
  t67  = t41 + t44
  t68  = t66 - t63
  t70  = t67 - t64
  t71  = t70 - t58
  t72  = t60 + t68
  t73  = t18 - t10
  t75  = t8 + t20
  t76  = t73 + t75
  t79  = t73 - t75
  t82  = t46 - t38
  t83  = t36 + t48
  t87  = t53 + t56
  t88  = t54 + t57
  t90  = t63 + t66
  t91  = t64 + t67
  t92  = t91 - t88
  t94  = t90 - t87
  t97  = t58 + t70
  t98  = t68 - t60
  t101 = t88 + t91
  t102 = t87 + t90
  t103 = fma(+t23*t24, +t50)
  t104 = fma(+t24*t28, +t49)
  t105 = fma(-t24*t76, +t83)
  t106 = fma(-t24*t79, +t82)
  t107 = fma(-t23*t24, +t50)
  t108 = fma(-t24*t28, +t49)
  t109 = fma(+t24*t76, +t83)
  t110 = fma(+t24*t79, +t82)

  X[7].re      = t103
  X[7].im      = t104
  X[6].re      = t71
  X[6].im      = t72
  X[5].re      = t105
  X[5].im      = t106
  X[4].re      = t92
  X[4].im      = t94
  X[3].re      = t107
  X[3].im      = t108
  X[2].re      = t97
  X[2].im      = t98
  X[1].re      = t109
  X[1].im      = t110
  X[0].re      = t101
  X[0].im      = t102

================================================================
  Stats
================================================================

DAG nodes: 69 total
  Loads:  16
  Consts: 1
  Negs:   0
  Adds:   22
  Subs:   22
  Muls:   0
  Cmuls:  0   (each = 1 mul + 1 fmadd/fnmadd = 2 instructions)
  Fmas:   8   (each = 1 fmadd/fmsub/fnmadd/fnmsub = 1 instruction)

Vector instructions (FMA-fused, ISA-independent): 52
  Breakdown: 44 add/sub/mul/neg + 0 cmul-pair instructions + 8 fma

Scalar-equivalent ops (each Cmul = 3 ops, each Fma = 2 ops): 60
  AVX-512 work (×8 lanes): 480 ops/iter
  AVX-2   work (×4 lanes): 240 ops/iter
