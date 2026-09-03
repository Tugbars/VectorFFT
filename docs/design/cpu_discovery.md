# CPU Discovery and Its Touchpoints
### What the library learns about the machine, which decisions that feeds, and the switch that decides whether it is used

**Scope.** This paper describes the structure of the CPU discovery layer
(`src/core/support/cpu_cache.h`): what it reads, the `seen` / `used`
split, and every place its answers are consumed. It is a map of the
mechanism, not a design history. Companions:
[`planning_model.md`](planning_model.md),
[`measurement_arms.md`](measurement_arms.md),
`src/core/wisdom2/README.md` §4.3 (field portability classes).

**The thesis in one sentence:** the library never computes a tuning
parameter from a cache size — widths and chains are found by racing —
so discovery exists for three narrower jobs: size a few *candidate
fences*, *stamp identity* onto measured verdicts, and *refuse* a verdict
that was measured on different silicon.

---

## 1. The probe

`vfft_cpu_cache()` runs CPUID once per process, caches the result in a
static, and hands back a read-only struct. It is **planning-only by
law** — nothing on an execute path may call it.

Both vendors publish the same cache encoding under different leaf
numbers, so one decoder (`_vfft_cpu_walk_cache_leaf`) serves both; the
only per-vendor decision is which leaf number to hand it.

```
  CPUID leaf 0 ──────────── vendor string
  CPUID leaf 4 ──┐
                 ├───────── cache geometry (walked over subleaves)
  0x8000001D ────┘            Intel leaf / AMD twin
  CPUID leaf 0xB ─┐
                  ├──────── SMT width (threads per physical core)
  0x8000001E ─────┘           Intel leaf / AMD twin
  CPUID leaf 0x1A ────────── hybrid core type (P / E; absent = not hybrid)
```

The AMD twins are gated three ways — vendor is AMD *and* leaf 4 returned
nothing *and* TOPOEXT (`0x80000001` ECX bit 22) is set *and* the
extended max-leaf covers them — so a part that lacks them reports 0 and
takes the fallback rather than decoding garbage.

### 1.1 The OS tier — hosts CPUID cannot describe

CPUID is an x86 instruction. Below it sits a second source that fills
**only the fields CPUID left at zero**, so on a working x86 host it is a
no-op:

```
  host                                   why CPUID fails        OS source used instead
  ARM64: Apple silicon, Windows on        no CPUID at all        macOS  sysctl hw.perflevel0.l1dcachesize / l2 / l3
         Snapdragon, Graviton                                    Windows GetLogicalProcessorInformationEx
                                                                  Linux  sysconf(_SC_LEVEL1_DCACHE_SIZE), then sysfs
  virtualized guests                      hypervisor masks or    the topology the hypervisor actually granted
                                          fabricates leaf 4
  sandboxes that trap CPUID               instruction faults     same calls; build with -DVFFT_CPU_DISABLE_CPUID
```

It reports the same four quantities (L1d, L2, L3, SMT) through the same
struct, so no consumer changes. One limit, stated once: the OS reports the
cache of *a* core without saying which type. Hybrid parts are x86 and
always have CPUID, so this tier never sizes a hybrid host.

`-DVFFT_CPU_DISABLE_CPUID` skips the instruction and forces this tier; it
exists for trapped sandboxes and for verifying the tier on hardware that
has CPUID. Without CPUID the host tag degrades to the architecture
(`arm64`, `x86-nocpuid`): coarse, never wrong.

### 1.1 `seen` vs `used` — the one fork that matters

Discovery **always runs** and always fills the `*_seen` fields. A
compile-time switch decides whether consumers get that answer or a
pinned constant:

```
                          ┌─────────────────────────────────────────────┐
   l1d_seen  l2_seen      │            VFFT_L1D_DISCOVER ?              │
   l3_seen   smt          └─────────────────────────────────────────────┘
   vendor    core_type          │ = 1                       │ = 0 (default)
   geometry_ok                  ▼                           ▼
   (always filled)      used ← seen                 used ← pinned constants
                        if is_pcore &&              l1d_used = 48 KB
                           geometry_ok,             l2_used  =  2 MB
                        else small fallback
                        (32 KB / 1 MB)
                                │                           │
                                └───────────┬───────────────┘
                                            ▼
                              l1d_used / l2_used
                    what every sizing decision reads, and what
                    gets stamped beside any verdict that used it
```

Guard rails inside the probe:

- **E-core refusal.** CPUID answers for whichever core the query runs
  on. On a hybrid part, a P-core label that fails its geometry
  cross-check — or an E-core answer — is refused, and sizing falls to
  the *smaller* fallback: undershooting a tile degrades gracefully;
  overshooting loses the whole benefit at once.
- **L3 is an aggregate.** `l3_seen` is shared by every core. Its only
  legitimate use is a whole-machine budget question ("do T concurrent
  working sets fit?"), never per-core sizing.

---

## 2. The tile-width lifecycle (the pattern every cache-sensitive verdict follows)

The largest consumer is the **tcut tile width** of the K=1 ZTURN
cascade. Note where L1 does *not* appear: candidate generation
(`vfft_zturn2_tile_candidates`, `zturn.h`) enumerates every width that
divides `N/4` and is legal for the chain, and **benches all of them**.
There is no occupancy filter, by decision: an excluded width is never
timed and leaves no trace, so a wrong filter would be undetectable from
its own output. Occupancy is computed and reported because it explains
results; it gates nothing.

```
 1 · RACE                  2 · BANK                    3 · REPLAY
 dp_planner_il.h           wisdom2_oop.txt             k1_commit.h

 bench EVERY legal   ──►   @cell … zt_tw=1024    ──►   vfft_cpu_l1d_matches(zt_l1)
 width w | (N/4),          zt_l1=<l1d_used>            i.e. stamped ≤ 0
 chain-legal;                                              || stamped == l1d_used
 L1 not consulted          the width AND the
                           cache it was raced               │ match      │ foreign
                           against                          ▼            ▼
                                                        TILED        UNTILED,
                                                        w=1024       said loudly
```

**L1 is identity, not input.** The width is chosen by measurement alone;
the cache size rides along as a stamp so replay can ask one question —
*was this verdict measured on silicon shaped like mine?* A mismatch
never re-derives a width; it disables tiling, prints why, and the cell
stays correct-but-slower until it is re-raced on this machine. Two more
checks sit behind the fence: an explicit `VFFT_TCUT` env override beats
wisdom, and a width illegal for the banked chain is refused even when
the cache matches.

The fork of §1.1 lands exactly here. With `VFFT_L1D_DISCOVER=0`,
`l1d_used` is the same pinned 48 KB on every machine, so the replay
comparison degenerates to `49152 == 49152` everywhere and the fence can
never fire. With discovery on, a store stamped on one machine is refused
on another — which is the fence's whole purpose.

---

## 3. All consumers, by field

Four fields carry all the load; everything else in the struct is
diagnostics.

| field | consumer | role |
|---|---|---|
| `l1d_used` | `dp_planner_il.h:1857` (bank), `k1_commit.h:820` (replay) | the tcut stamp + fence of §2 |
| `l2_used` | `il2d_tier.h:909, :1325` | 2D IL band fence: a band width `w` is a *candidate* only if `w · N2 · 16 ≤ l2_used`; the race still decides |
| `smt` | `threads.h:176` | worker-pool **pin stride** — one worker per physical core, derived, never hard-coded; unknown (0) keeps the historical stride 2 |
| `host_tag()` + `l1d_used` + ISA | `vfft.c` (store open) | the wisdom `@meta host= isa= l1d=` stamp: an unstamped store adopts this host's identity; a store stamped for a different host warns once, loudly |

Both fences shape only which *candidates* exist or replay — measurement
always makes the final choice. Nothing anywhere converts a cache size
directly into a tuning parameter.

The fence itself has a gate: `oop_width_gate.h`
(driver `benches/zturn_wisdom_width_gate.c`) injects synthetic verdicts
— no width, native-L1 width, foreign-L1 width, illegal width — through
the public front door and asserts the engagement of each. Its CASES
derive the native/foreign stamps from `vfft_cpu_l1d_bytes()` at run
time, so the gate asserts *behaviour* (native engages, foreign refused)
rather than assuming any particular host.

---

## 4. The two modes, honestly stated

| | `DISCOVER=0` pinned (default) | `DISCOVER=1` live |
|---|---|---|
| `l1d_used` / `l2_used` | 48 KB / 2 MB on *every* machine | this machine's real geometry (guarded) |
| replay fence | can never fire — every host claims the calibration host's cache | fires exactly on cross-machine replay |
| L2 band candidates | sized for a 2 MB L2 regardless of hardware | sized for the real L2 |
| protects against | a mid-campaign query answered from an E-core resizing a benchmark on a hybrid part | foreign verdicts replaying silently on the wrong silicon |
| appropriate when | running *on* the calibration host, pinned to its P-cores | running anywhere else |

The two protections are aimed at different failure modes; the switch
chooses which one you get. It is `#ifndef`-guarded, so
`-DVFFT_L1D_DISCOVER=1` on a build selects live mode without touching
the source.

---

## 5. API quick reference

| call | returns | rule attached to it |
|---|---|---|
| `vfft_cpu_l1d_bytes()` | `l1d_used` | the value every L1-sized decision must use, and the value stamped beside any verdict that depended on it |
| `vfft_cpu_l2_bytes()` | `l2_used` | the L2 twin, same contract |
| `vfft_cpu_l3_bytes()` | `l3_seen` | aggregate budgets only; 0 = unknown, and the caller must then refuse the question |
| `vfft_cpu_smt()` | threads per core | pin stride derives from this, never hard-coded; 0 = unknown |
| `vfft_cpu_l1d_matches(s)` | bool | the replay fence: `s ≤ 0 \|\| s == l1d_used` — an unstamped verdict always passes |
| `vfft_cpu_l2_matches(s)` | bool | L2 twin of the fence |
| `vfft_cpu_host_tag()` | e.g. `amd-f25m117` | identity for the `@meta host=` stamp — vendor + display family/model, whitespace-free; nothing may branch on it |

---

## 6. The design rule under all of it

A chain or a radix is a property of the transform and ports to any CPU.
A tile width is a property of one machine's cache, and on the wrong
machine it fails as a mild slowdown — the worst thing to inherit
silently. So machine-shaped values are **discovered, stamped, and
re-checked**; they are never ported, and they are never the thing that
picks the winner. Measurement picks the winner.
