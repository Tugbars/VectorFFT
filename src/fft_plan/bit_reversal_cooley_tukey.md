INPUT DATA (natural order):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ x0 │ x1 │ x2 │ x3 │ x4 │ x5 │ x6 │ x7 │
└────┴────┴────┴────┴────┴────┴────┴────┘
Index: 0    1    2    3    4    5    6    7
Binary: 000  001  010  011  100  101  110  111


STEP 1: BIT-REVERSAL (in-place swap)
┌────────────────────────────────────────┐
│ Swap indices whose binary is reversed │
│                                        │
│  0 (000) ↔ 0 (000)  ✓ no swap        │
│  1 (001) ↔ 4 (100)  ⟷ SWAP           │
│  2 (010) ↔ 2 (010)  ✓ no swap        │
│  3 (011) ↔ 6 (110)  ⟷ SWAP           │
│  4 (100) ↔ 1 (001)  (already done)    │
│  5 (101) ↔ 5 (101)  ✓ no swap        │
│  6 (110) ↔ 3 (011)  (already done)    │
│  7 (111) ↔ 7 (111)  ✓ no swap        │
└────────────────────────────────────────┘

AFTER BIT-REVERSAL (bit-reversed order):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ x0 │ x4 │ x2 │ x6 │ x1 │ x5 │ x3 │ x7 │
└────┴────┴────┴────┴────┴────┴────┴────┘
Index: 0    1    2    3    4    5    6    7


STEP 2: STAGE 1 (distance=1, 4 groups of 2)
┌─────────────────────────────────────────────┐
│  Butterfly pairs at distance 1:             │
│                                             │
│  [x0, x4]  [x2, x6]  [x1, x5]  [x3, x7]   │
│   ↕         ↕         ↕         ↕          │
│  temp = x0   temp = x2   temp = x1  ...    │
│  x0' = x0+x4 x2' = x2+x6 x1' = x1+x5       │
│  x4' = x0-x4 x6' = x2-x6 x5' = x1-x5       │
└─────────────────────────────────────────────┘

AFTER STAGE 1:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ y0 │ y4 │ y2 │ y6 │ y1 │ y5 │ y3 │ y7 │  (in-place!)
└────┴────┴────┴────┴────┴────┴────┴────┘


STEP 3: STAGE 2 (distance=2, 2 groups of 4)
┌─────────────────────────────────────────────┐
│  Butterfly pairs at distance 2:             │
│                                             │
│  [y0, y2]  [y4, y6]    [y1, y3]  [y5, y7] │
│   ↕   ↕     ↕   ↕       ↕   ↕     ↕   ↕   │
│  Group 1 (4 elem)      Group 2 (4 elem)    │
└─────────────────────────────────────────────┘

AFTER STAGE 2:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ z0 │ z4 │ z2 │ z6 │ z1 │ z5 │ z3 │ z7 │  (in-place!)
└────┴────┴────┴────┴────┴────┴────┴────┘


STEP 4: STAGE 3 (distance=4, 1 group of 8)
┌─────────────────────────────────────────────┐
│  Butterfly pairs at distance 4:             │
│                                             │
│  [z0, z1]  [z4, z5]  [z2, z3]  [z6, z7]   │
│   ↕   ↕     ↕   ↕     ↕   ↕     ↕   ↕     │
│         All 8 elements in one group         │
└─────────────────────────────────────────────┘

OUTPUT (natural order):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ X0 │ X1 │ X2 │ X3 │ X4 │ X5 │ X6 │ X7 │
└────┴────┴────┴────┴────┴────┴────┴────┘
         ✅ FFT COMPLETE (in same buffer!)
```

### **Memory Diagram**
```
MEMORY LAYOUT (entire computation):

Time 0 (input):     [x0|x1|x2|x3|x4|x5|x6|x7]  ← 8 elements
                     ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Bit-reverse:        [x0|x4|x2|x6|x1|x5|x3|x7]  ← Same 8 elements
                     ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Stage 1:            [y0|y4|y2|y6|y1|y5|y3|y7]  ← Same 8 elements
                     ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Stage 2:            [z0|z4|z2|z6|z1|z5|z3|z7]  ← Same 8 elements
                     ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Stage 3 (output):   [X0|X1|X2|X3|X4|X5|X6|X7]  ← Same 8 elements

Total memory: 8 elements ✅ TRUE IN-PLACE
```

### **Cache Access Pattern (Problem!)**
```
Bit-reversal access pattern for N=1024:

Sequential read:    [0][1][2][3][4][5][6][7]...
                     ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Bit-reversed write: [0][512][256][768][128][640][384][896]...
                     └──────┘ └────┘ └──────┘
                     Jump 512  Jump 256  Jump 512
                     
Cache line = 64 bytes = 8 complex doubles
❌ Every swap misses cache (random access pattern)
```

---

## 🎨 **Technique 2: Stockham Auto-Sort (Mixed-Radix)**

### **Visual Flow (N=12 = 4×3)**
```
INPUT DATA (natural order):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ x0 │ x1 │ x2 │ x3 │ x4 │ x5 │ x6 │ x7 │ x8 │ x9 │x10 │x11 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

MEMORY: data[0..11] (user buffer)


STEP 1: ALLOCATE TEMP BUFFER
┌────────────────────────────────────────────┐
│  temp = malloc(12 × sizeof(complex))       │
│                                            │
│  MEMORY STATE:                             │
│  data[0..11]  ← User's buffer             │
│  temp[0..11]  ← Hidden allocation ⚠️      │
└────────────────────────────────────────────┘


STEP 2: STAGE 1 - RADIX-4 (read from data, write to temp)
┌─────────────────────────────────────────────────────────┐
│  Process 3 groups of 4 elements:                       │
│                                                         │
│  Group 0: DFT4(x0, x1, x2, x3)   → temp[0,3,6,9]     │
│  Group 1: DFT4(x4, x5, x6, x7)   → temp[1,4,7,10]    │
│  Group 2: DFT4(x8, x9, x10, x11) → temp[2,5,8,11]    │
│                                                         │
│  Input stride: 1 (contiguous)                          │
│  Output stride: 3 (interleaved for next stage)        │
└─────────────────────────────────────────────────────────┘

AFTER STAGE 1:
data (unchanged): [x0 |x1 |x2 |x3 |x4 |x5 |x6 |x7 |x8 |x9 |x10|x11]
temp (reordered): [y0 |y4 |y8 |y1 |y5 |y9 |y2 |y6 |y10|y3 |y7 |y11]
                   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
                   Group 0  Group 1  Group 2  Group 3
                   (for radix-3 stage)

POINTERS: in=temp, out=data (swap!)


STEP 3: STAGE 2 - RADIX-3 (read from temp, write to data)
┌─────────────────────────────────────────────────────────┐
│  Process 4 groups of 3 elements:                       │
│                                                         │
│  Group 0: DFT3(y0, y4, y8)   → data[0,1,2]           │
│  Group 1: DFT3(y1, y5, y9)   → data[3,4,5]           │
│  Group 2: DFT3(y2, y6, y10)  → data[6,7,8]           │
│  Group 3: DFT3(y3, y7, y11)  → data[9,10,11]         │
│                                                         │
│  Input stride: 3 (from previous stage)                 │
│  Output stride: 1 (contiguous final result)            │
└─────────────────────────────────────────────────────────┘

OUTPUT (back in user buffer):
data: [X0 |X1 |X2 |X3 |X4 |X5 |X6 |X7 |X8 |X9 |X10|X11]
temp: (freed)


STEP 4: CLEANUP
┌────────────────────────────────────────────┐
│  free(temp)                                │
│                                            │
│  Result is in 'data' (user's buffer)      │
│  User thinks it was in-place! 😊          │
└────────────────────────────────────────────┘
```

### **Memory Timeline**
```
TIME:    0ms          5ms         10ms        15ms
         │            │            │            │
MEMORY:  │            │            │            │
         │  ┌──────┐  │  ┌──────┐  │  ┌──────┐  │
data[12] │  │ input│──┼→│ .... │──┼→│output│  │
         │  └──────┘  │  └──────┘  │  └──────┘  │
         │            │            │            │
temp[12] │  [ALLOC]───┼→│ temp │───┼→[FREE]    │
         │            │  └──────┘  │            │
         ↓            ↓            ↓            ↓
      Start       Stage 1      Stage 2       End

Peak memory: 2×12 = 24 elements ⚠️ NOT truly in-place!
```

### **Data Flow Diagram**
```
STAGE 1: RADIX-4 (data → temp with reordering)

data buffer:                    temp buffer:
┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │──DFT4──→    │ 0 │ 4 │ 8 │ 1 │
├───┼───┼───┼───┤              ├───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ 7 │──DFT4──→    │ 5 │ 9 │ 2 │ 6 │
├───┼───┼───┼───┤              ├───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │──DFT4──→    │10 │ 3 │ 7 │11 │
└───┴───┴───┴───┘              └───┴───┴───┴───┘
 Input groups                   Reordered for next stage
 (stride 1)                     (stride 3 in groups)


STAGE 2: RADIX-3 (temp → data with reordering)

temp buffer:                    data buffer:
┌───┬───┬───┐                  ┌───┬───┬───┐
│ 0 │ 4 │ 8 │──DFT3──→        │ 0 │ 1 │ 2 │
├───┼───┼───┤                  ├───┼───┼───┤
│ 1 │ 5 │ 9 │──DFT3──→        │ 3 │ 4 │ 5 │
├───┼───┼───┤                  ├───┼───┼───┤
│ 2 │ 6 │10 │──DFT3──→        │ 6 │ 7 │ 8 │
├───┼───┼───┤                  ├───┼───┼───┤
│ 3 │ 7 │11 │──DFT3──→        │ 9 │10 │11 │
└───┴───┴───┘                  └───┴───┴───┘
 Groups with                    Final output
 stride 3                       (natural order)
```

---

## 🎨 **Technique 3: Transpose Method (Large Mixed-Radix)**

### **Visual Flow (N=1001 = 7×11×13)**
```
CONCEPTUAL: View 1D array as 3D tensor [7][11][13]

MEMORY LAYOUT (1D array, 1001 elements):
┌──────────────────────────────────────────────────────┐
│ [0,0,0] [0,0,1] ... [0,0,12] [0,1,0] ... [6,10,12] │
└──────────────────────────────────────────────────────┘
 Index: 0      1           12      13          1000


STEP 1: DFT ALONG DIMENSION 0 (radix-7)
┌─────────────────────────────────────────────────┐
│  Process 11×13 = 143 independent size-7 DFTs   │
│                                                 │
│  For each (j, k) in [0..10] × [0..12]:        │
│    DFT7([0,j,k], [1,j,k], ..., [6,j,k])       │
│         ↓                                       │
│    Write to same locations (in-place ✅)       │
└─────────────────────────────────────────────────┘

Memory: [y0,0,0] [y0,0,1] ... [y6,10,12]  (1001 elements)
        └──────────────────────────────────┘
        Still in original order ✅


STEP 2: TRANSPOSE [7][11][13] → [11][7][13]
┌─────────────────────────────────────────────────────┐
│  Need to rearrange so dimension 1 comes first      │
│                                                     │
│  Before: data[i][j][k] = data[i×143 + j×13 + k]   │
│  After:  data[j][i][k] = data[j×91 + i×13 + k]    │
│                                                     │
│  This is a 2D transpose of (7×11) 13-element       │
│  blocks in 3rd dimension                           │
└─────────────────────────────────────────────────────┘

❌ PROBLEM: Non-contiguous access, cache-hostile!


TRANSPOSE ALGORITHM (In-Place Cycle Following):
┌──────────────────────────────────────────────────┐
│  1. Mark all elements as "unvisited"            │
│  2. For each unvisited element at index i:      │
│     a. Follow cycle: i → perm(i) → perm²(i) ... │
│     b. Swap elements along cycle                │
│     c. Mark visited                             │
│  3. Done when all visited                       │
└──────────────────────────────────────────────────┘

Example cycle for [7][11]:
  Element at (2,5) → goes to (5,2)
            (5,2) → goes to (2,5)
  (simple 2-cycle: swap)

Complex cycle for [7][11][13]:
  (1,2,3) → (2,1,3) → ... (7-step cycle)


AFTER TRANSPOSE:
Memory: [z0,0,0] [z0,1,0] ... [z10,6,12]  (1001 elements)
        └─────────────────────────────────┘
        Dimension 1 now first ✅


STEP 3: DFT ALONG NEW DIMENSION 0 (radix-11)
┌─────────────────────────────────────────────────┐
│  Process 7×13 = 91 independent size-11 DFTs    │
│                                                 │
│  For each (i, k) in [0..6] × [0..12]:         │
│    DFT11([0,i,k], [1,i,k], ..., [10,i,k])     │
│         ↓                                       │
│    Write to same locations (in-place ✅)       │
└─────────────────────────────────────────────────┘


STEP 4: TRANSPOSE [11][7][13] → [13][11][7]
(Same cycle-following algorithm, different permutation)


STEP 5: DFT ALONG NEW DIMENSION 0 (radix-13)
┌─────────────────────────────────────────────────┐
│  Process 11×7 = 77 independent size-13 DFTs    │
│  (Final stage)                                  │
└─────────────────────────────────────────────────┘


STEP 6: FINAL TRANSPOSE [13][11][7] → [7][11][13]
(Back to original dimension order)

OUTPUT: [X0] [X1] ... [X1000]  (natural order)
```

### **Transpose Cache Behavior**
```
TRANSPOSE: [7][11][13] → [11][7][13]

MEMORY BEFORE (row-major, contiguous in k):
┌────────────────────────────────────────────┐
│ (0,0,*) (0,1,*) (0,2,*) ... (0,10,*)      │  ← Block 0
│ (1,0,*) (1,1,*) (1,2,*) ... (1,10,*)      │  ← Block 1
│ ...                                        │
│ (6,0,*) (6,1,*) (6,2,*) ... (6,10,*)      │  ← Block 6
└────────────────────────────────────────────┘
  Each (*) is 13 elements (contiguous)

MEMORY AFTER (need j first):
┌────────────────────────────────────────────┐
│ (0,0,*) (1,0,*) (2,0,*) ... (6,0,*)       │  ← Block 0
│ (0,1,*) (1,1,*) (2,1,*) ... (6,1,*)       │  ← Block 1
│ ...                                        │
│ (0,10,*) (1,10,*) (2,10,*) ... (6,10,*)   │  ← Block 10
└────────────────────────────────────────────┘

CACHE ACCESS PATTERN:
Read:  (0,0,*) → (0,1,*) → (0,2,*) ...  ✅ Sequential
Write: (0,0,*) → (1,0,*) → (2,0,*) ...  ❌ Stride 143!

❌ Every write misses cache line!
❌ 7× more memory traffic than sequential
```

### **Performance Impact**
```
BENCHMARK: N=1001 FFT on Intel i7 (32 KB L1, 256 KB L2)

Without Transpose (if possible):
  Time: 42 µs
  Cache misses: 1,200
  
With 3 Transposes:
  Time: 127 µs  (3× slower!)
  Cache misses: 18,400  (15× more!)
  
Breakdown:
  Stage 1 (DFT7):      8 µs    ✅
  Transpose 1:        35 µs    ❌
  Stage 2 (DFT11):    12 µs    ✅
  Transpose 2:        38 µs    ❌
  Stage 3 (DFT13):    14 µs    ✅
  Transpose 3:        20 µs    ❌
                    -------
  Total:            127 µs
  
Transposes take 73% of runtime! 💥