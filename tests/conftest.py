"""Gemeinsame Fixtures: die Mini-Stichproben, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import os
import sys
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Fünf Phase-I-Beobachtungen (1, 3, 2, 5, 4). Handrechnung: x̄ = 3, Summe der Abweichungsquadrate 10, s² = 5/2; gleitende Spannweiten 2, 1, 3, 1 (Mittel 7/4)."""
    return (1.0, 3.0, 2.0, 5.0, 4.0)
