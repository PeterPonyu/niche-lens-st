"""Synthetic niche-recovery benchmark generator.

Implements the schema documented in ``docs/SYNTHETIC_BENCHMARK.md``.
No performance is reported here; this generates evidence inputs.
"""

from .generator import SynthInstance, generate_instance

__all__ = ["SynthInstance", "generate_instance"]
