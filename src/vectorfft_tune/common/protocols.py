"""
protocols.py — twiddle layout definitions per protocol.

A "protocol" is the contract between the planner (which populates the twiddle
buffer) and the codelet (which reads from it). Three protocols exist, matching
the registry's parallel slots:

    flat:  W_re has (R-1) * me doubles.
           W_re[(j-1)*me + m] is the j-th leg twiddle at position m.
           Used by: ct_t1_dit, ct_t1_dit_u2, ct_t1_dit_log1 (reads a subset).

    log3:  W_re has 1 * me doubles (w1 only).
           W_re[m] is the base twiddle at position m.
           Codelet derives w2 = w1 * w1, w3 = w2 * w1.
           Used by: ct_t1_dit_log3.

    t1s:   W_re has (R-1) doubles (scalars).
           W_re[j-1] is the j-th leg twiddle (same for all m in this call).
           Codelet broadcasts each scalar to a SIMD register before the m-loop.
           Used by: ct_t1s_dit. K-blocked execution guarantees the scalar
           invariant at the call site.

    log1_tight: experimental — same codelet body as log1 but the harness
           allocates only 2*me doubles (w1+w2, no wasted w3 column). Tests
           whether log1 in flat protocol is handicapped by a full-size
           twiddle table. Only used by the handicap experiment; not a
           real protocol in production.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    twiddle_cols_expr: str        # python expression in terms of R and me -> doubles per component
    description: str


PROTOCOLS = {
    'flat':       ProtocolSpec(
        name='flat',
        twiddle_cols_expr='(R - 1) * me',
        description='Full per-leg per-m twiddles.'),
    'log3':       ProtocolSpec(
        name='log3',
        twiddle_cols_expr='1 * me',
        description='Base twiddle only; derive w2, w3 via cmul.'),
    't1s':        ProtocolSpec(
        name='t1s',
        twiddle_cols_expr='(R - 1)',
        description='Scalar per leg; broadcast before the m-loop.'),
    'log1_tight': ProtocolSpec(
        name='log1_tight',
        twiddle_cols_expr='2 * me',
        description='log1 variant — w1+w2 only, derive w3. Handicap experiment.'),
}


def twiddle_doubles(protocol: str, R: int, me: int) -> int:
    """Return the number of doubles per component (re OR im) for a given
    (protocol, R, me). Multiply by 2 for total re+im buffer size."""
    spec = PROTOCOLS[protocol]
    return int(eval(spec.twiddle_cols_expr, {'R': R, 'me': me}))


def known_protocols() -> list[str]:
    return list(PROTOCOLS.keys())
