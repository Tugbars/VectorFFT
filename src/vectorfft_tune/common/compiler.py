"""
compiler.py — platform-aware compiler detection and flag sets.

Supports:
  Linux   + gcc   (default container)
  Linux   + clang
  Linux   + icx
  Windows + gcc (mingw)
  Windows + icx (Tugbars's production target)
  Windows + msvc (fallback)

Flag sets are:
  COMMON_FLAGS: optimization + warnings applicable everywhere
  ISA_FLAGS:    ISA-specific flags per {gcc, clang, icx, msvc}
  LINK_FLAGS:   platform/compiler-specific link extras (lld, libm, etc.)

Callers should use the `build_command` helper rather than assembling argv
lists manually — it routes through the right compiler class.

Environment:
  CC       — if set, overrides auto-detection.
  VFFT_CC  — same, higher priority (avoids collision with system CC).
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


WINDOWS = (platform.system() == 'Windows')


def _which(name: str) -> str | None:
    p = shutil.which(name)
    return p if p else None


def detect_cc() -> str:
    """Return an absolute compiler path, or a symbolic name if on PATH.
    Order: VFFT_CC, CC, icx, gcc, clang, cl."""
    for env_var in ('VFFT_CC', 'CC'):
        v = os.environ.get(env_var)
        if v:
            return v
    for candidate in ('icx', 'gcc', 'clang', 'cl'):
        if _which(candidate):
            return candidate
    raise RuntimeError(
        'No C compiler found. Set VFFT_CC or CC, or install gcc/clang/icx.')


def compiler_kind(cc_path: str) -> str:
    """Classify the compiler: 'gcc', 'clang', 'icx', 'msvc', 'unknown'."""
    base = Path(cc_path).name.lower()
    if base.startswith('icx') or base.startswith('icc'): return 'icx'
    if base.startswith('clang'):                         return 'clang'
    if base == 'cl.exe' or base == 'cl':                 return 'msvc'
    if base.startswith('gcc'):                           return 'gcc'
    # Probe by asking the compiler.
    for kind, probe in [('icx', ['--version']),
                        ('gcc', ['--version']),
                        ('clang', ['--version']),
                        ('msvc', ['/?'])]:
        try:
            r = subprocess.run([cc_path] + probe, capture_output=True,
                               text=True, timeout=5)
            txt = (r.stdout + r.stderr).lower()
            if 'intel' in txt or 'icx' in txt: return 'icx'
            if 'clang' in txt:                 return 'clang'
            if 'gcc' in txt or 'g++' in txt:   return 'gcc'
            if 'microsoft' in txt:             return 'msvc'
        except Exception:
            continue
    return 'unknown'


@dataclass
class CompilerSpec:
    cc: str
    kind: str      # 'gcc' | 'clang' | 'icx' | 'msvc'
    on_windows: bool

    def base_flags(self, opt_level: str = 'O3') -> list[str]:
        if self.kind == 'msvc':
            return [f'/{opt_level}' if opt_level.startswith('O') else f'/{opt_level}',
                    '/nologo', '/W2']
        return [f'-{opt_level}', '-Wall']

    def isa_flags(self, isa: str) -> list[str]:
        """Flags to enable a target ISA for code in the TU being compiled.
        The generated codelets use `__attribute__((target(...)))` for function-
        level targeting (gcc/clang/icx) so we only need broad ISA enable."""
        if self.kind in ('gcc', 'clang', 'icx'):
            if isa == 'avx2':   return ['-mavx2', '-mfma']
            if isa == 'avx512': return ['-mavx512f', '-mavx512dq', '-mfma']
            if isa == 'scalar': return []
        if self.kind == 'msvc':
            if isa == 'avx2':   return ['/arch:AVX2']
            if isa == 'avx512': return ['/arch:AVX512']
            if isa == 'scalar': return []
        return []

    def compile_only_flag(self) -> str:
        return '/c' if self.kind == 'msvc' else '-c'

    def output_flag(self, path: str) -> list[str]:
        if self.kind == 'msvc':
            return [f'/Fo{path}'] if path.endswith('.obj') else [f'/Fe{path}']
        return ['-o', path]

    def define(self, name: str, value: str | None = None) -> list[str]:
        if self.kind == 'msvc':
            return [f'/D{name}' if value is None else f'/D{name}={value}']
        return [f'-D{name}' if value is None else f'-D{name}={value}']

    def include(self, path: str) -> list[str]:
        if self.kind == 'msvc':
            return [f'/I{path}']
        return ['-I', path]

    def link_math(self) -> list[str]:
        """Math library linker flag."""
        if self.kind == 'msvc':
            return []  # msvcrt has math
        if self.on_windows:
            # Any compiler on Windows links against msvcrt which provides libm.
            # ICX in clang-cl mode rejects -lm outright; mingw-gcc doesn't need it.
            return []
        return ['-lm']

    def link_flags(self) -> list[str]:
        if self.kind == 'msvc':
            return []
        # ICX on Windows with LLD produces smaller exes, less Intel-runtime fuss
        if self.kind == 'icx' and self.on_windows:
            return ['/Qoption,link,/opt:noref']  # placeholder; LLD already default
        return []


def detect() -> CompilerSpec:
    cc = detect_cc()
    kind = compiler_kind(cc)
    return CompilerSpec(cc=cc, kind=kind, on_windows=WINDOWS)


# ── runtime ISA probe (for skipping AVX-512 candidates on AVX2-only chips) ──

def host_supports_avx512() -> bool:
    """Return True if this machine can execute AVX-512 instructions.
    Used by harness for runtime-gating candidate skips."""
    try:
        if sys.platform == 'linux':
            with open('/proc/cpuinfo') as f:
                return 'avx512f' in f.read()
        # Windows / macOS: compile+run a tiny probe if needed, else fall back to
        # a conservative guess based on compiler kind. Most Windows Raptor Lake
        # machines do NOT support AVX-512 (fused off).
        return False  # safe default; overridden by harness at measurement time
    except Exception:
        return False


def host_supports_avx2() -> bool:
    try:
        if sys.platform == 'linux':
            with open('/proc/cpuinfo') as f:
                return 'avx2' in f.read()
        return True  # AVX2 has been baseline on Intel/AMD x86 for ~10 years
    except Exception:
        return True
    