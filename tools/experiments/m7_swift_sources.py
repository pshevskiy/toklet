"""Explicit transitive Swift sources for standalone NativeWorker programs.

Keep diagnostics and their entry points at each caller. This common list tracks
worker dependencies without pulling the application UI into numerical checks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER_NAMES = (
    'NativeWorker.swift', 'NativeCompatibility.swift', 'NativeResourcePlan.swift',
    'NativeContext.swift', 'NativeContextLimits.swift', 'NativeCapacityPlan.swift',
    'NativePlanningPreview.swift', 'NativeSessionTransition.swift', 'NativeProcessMemory.swift',
)


def worker_sources(*additional):
    """Return a new list, preserving caller entry-point ordering and no duplicates."""
    names = (*WORKER_NAMES, *additional)
    return [ROOT / 'apple/Harness' / name for name in dict.fromkeys(names)]
