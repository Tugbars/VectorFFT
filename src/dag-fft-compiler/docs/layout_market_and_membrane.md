# Layouts and their owners: who needs split, who needs transform-contiguous

Companion to lab notebook sections 52-53c. Two questions in one
document: the engineering decision (fused membrane vs an interleaved
codelet family) and the market map (which FFT users live on which
layout, and therefore who we serve natively, who we serve through the
membrane, and who stays on the incumbents' turf).

## 1. Terms, precisely

Two ORTHOGONAL layout axes get conflated under "interleaved":

- **Batch axis.** Transform-contiguous (record-major): sample n of
  record r at `r*N + n`; each transform is one contiguous run; the
  incumbents' contract. Batch-split (ours): `n*K + k`; sample n of
  ALL records adjacent; SIMD lanes = records; zero shuffles in the
  engine.
- **Complex axis.** Interleaved complex: (re, im) pairs adjacent (C99
  complex, RF I/Q streams). Split complex: separate re[] and im[]
  arrays (ours).

A user crossing into our engine may need conversion on either axis or
both. The same 8x8 boundary tile performs both conversions in one
pass, which is why the membrane is one mechanism, not two.

## 2. The decision: membrane, not interleaved codelets

Evidence (sections 53b/53c): every SHUFFLE-FREE layout crossing
measured ~3x overhead — the explicit corner-turn (+199-241%, DRAM
passes) and the L2-sliced scalar form (+219-222%, scalar store
throughput). That pair of falsifications (#13, #14) also explains the
incumbents' measured 3-3.7x adapter tax as physics, not sloth. The
only cheap crossing is the in-register 8x8 shuffle tile at the
boundary: O(NK) shuffle work against the FFT's O(NK logN) compute,
landed on port 5, which our shuffle-free kernels leave idle.

Interleaved codelets pay the shuffle tax inside every butterfly of
every stage, forever (FFTW: 21-29% of the vector stream); the
membrane pays it once per element at the surface. Same coin, two
prices, ratio ~ log N. An interleaved family would cost months
(re-deriving genfft's SIMD layer: lane mapping, inter-stage shuffle
networks, twiddle layout, permutation synthesis) for a ceiling of
FFTW-parity — their home format is exactly where we measured 1.01x.
The membrane v1 is a hand-written tile kernel, days, engine
untouched.

DECISION RULE (standing): build membrane v1, measure. The predicted
+5-15% whole-plan overhead is held loosely (two pre-registrations
died establishing this section). The interleaved family is justified
only if v1 measures above ~40%, and even then only for the
K=1-small-N niche that nothing else serves natively (the scalar
cascade serves it respectably; four-step covers K=1 at large N and
reuses the same membrane tiles).

## 3. Who needs SPLIT-BATCHED (our shape, native wins)

The unifying trait: the user owns many records at once, or the
problem manufactures them. Where the producer can be told what to
write, the speed is pure profit; where it can't, one membrane pass
serves.

- **Wireless infrastructure.** 5G/LTE basestation PHY: OFDM symbols x
  subcarrier transforms x antenna chains (massive MIMO = 64-256
  chains). Thousands of same-N FFTs per slot is the definition of
  the workload.
- **Radar / EW / SAR.** Pulse-Doppler processing batches FFTs across
  range gates and channels; "corner-turn" is native radar vocabulary
  — the radar cube is already stored and turned this way. SAR
  imaging is massive 2D batch work.
- **Sonar and medical ultrasound.** Array channels and scan lines =
  batch axis by construction.
- **Seismic (oil & gas).** Thousands of geophone traces per shot;
  FK filtering and migration are batched-1D-FFT factories.
- **Radio astronomy.** Correlators and polyphase filterbanks
  channelize across hundreds of antennas/beams (SKA, ALMA class).
- **Weather / climate (the user's example, correct).** Spectral
  transform models (ECMWF IFS class) run FFTs along latitude circles
  for every field and every vertical level per timestep: the
  operational workload IS a giant batched 1D FFT, repeated forever.
- **Semiconductor / chip makers (the user's example, correct).**
  Computational lithography (OPC/ILT, Hopkins/TCC imaging models)
  burns datacenter-scale CPU on tiled 2D FFTs over mask layouts;
  wafer inspection and scatterometry metrology add more. Fabs are
  among the largest FFT consumers on earth.
- **Every multidimensional FFT user, internally.** A 2D/3D FFT IS
  batched 1D transforms along each axis (pencil decomposition in
  HPC). MD codes (PME electrostatics), CFD spectral solvers, CT/MRI
  reconstruction (projections, coils, slices) — all batch at the
  engine level even when the API call looks singular.
- **ML and offline signal pipelines.** Fourier Neural Operator
  layers (batch x channels), dataset-scale STFT feature extraction,
  spectral augmentation.
- **Quant finance / HFT.** The purest "owns the producer" case.
  Data arrives transform-contiguous (per-symbol columnar time
  series), but the workload batches along the UNIVERSE axis: the
  same window transform across hundreds-thousands of symbols per
  update, cross-sectional features, batched FFT convolution for
  signal kernels, Carr-Madan strike grids across expiries x
  underlyings x recalibrations, dataset-scale research/backtests.
  The membrane tax never gets charged: the feed handler is yours, so
  a rolling window written as window[t mod N][k] maintains split
  layout INCREMENTALLY at ingestion - one store per tick that was
  happening anyway. The latency-critical K=1 residue is doubly
  evacuated: HFT hot paths use Kalman/particle/IIR estimators rather
  than per-tick transforms, and single-stream per-tick spectral bins
  belong to the sliding DFT / Goertzel (O(1) per bin per tick), not
  to a batch FFT of any layout.

## 4. Who needs TRANSFORM-CONTIGUOUS (their shape; membrane or niche)

The unifying trait: records arrive one at a time and latency is the
product, or the code is written against the singular-array idiom.

- **Real-time audio.** Plugin/callback processing of one frame per
  period; ANC headphones, hearing aids, speech codecs. K=1, small N,
  latency-bound. (Scalar-cascade territory; absolute times are
  microseconds.)
- **Single-link comms / SDR endpoints.** A consumer modem or
  single-channel demodulator transforms each symbol as it lands.
  (The basestation on the other end of the same link batches.)
- **Control and condition monitoring.** Single-sensor vibration or
  motor-current spectral analysis per revolution/tick — the embedded
  K=1 small-N world.
- **Interactive instruments and scripting.** Oscilloscope/analyzer
  single-trace updates; the numpy/MATLAB `fft(x)` one-array idiom —
  ergonomics, not architecture.
- **Legacy interop.** Code written against FFTW/MKL conventions, C99
  complex, record-major files (WAV, packet payloads, I/Q captures).
  Convertible in principle; contiguous in practice because that is
  the shape the producer wrote.

## 5. The strategy map

- Native wins (no conversion, full margin): every batch-rich domain
  in section 3 where the producer is ours to specify — and all of
  section 3 once the fused membrane exists, at the membrane's
  measured tax.
- Membrane-served: transform-contiguous archives and pipelines with
  batchable work (offline audio/image, captured I/Q, legacy arrays).
- Cascade-served: any K >= 1, bit-exact (section 53), killing the
  K-multiplicity objection outright.
- Theirs, for now: hard-realtime K=1 small-N (audio callbacks,
  control loops) where within-transform SIMD is the only
  vectorization that exists — the one niche an interleaved family
  would uniquely serve, deliberately conceded pending the membrane
  v1 measurement and the four-step plan node.

One sentence for the README: batched transforms are not a niche of
the FFT market — they are the basestations, the radars, the weather
models, the fabs, and the inside of every multidimensional transform;
the singular streaming transform is the niche, and it is the only
ground we have not already taken.
