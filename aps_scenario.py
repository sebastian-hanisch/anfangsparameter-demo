"""Szenario: Phase-I-Stichproben aus dem SplitMix64-Strom (Zufalls-Seed), Schätzer, Störungen der Phase I.

Zufall: `uniforms` rechnet die Ziehungen von `aps_rng.SplitMix64` vektorisiert (bitgleich mit `SplitMix64(seed).random()`, Test). Die Normalzahlen entstehen per Box-Muller wie in
`SplitMix64.normal()`, aber mit den numpy-Funktionen statt der `decimal`-Fassung: sie können sich in der letzten Binärstelle von der Klasse unterscheiden (Test auf 1e-12), darum
prüfen die Messreihen der README Bänder, nicht Bits. Phase-I-Stichproben für die Verteilungs-Experimente werden direkt als (x̄, s) gezogen (x̄ = Φ⁻¹(u)/√m, s² = χ²-Quantil(u′; m − 1)/(m − 1);
x̄ und s sind unabhängig, Satz von Cochran), das ist gleich in der Verteilung zu Rohnormalzahlen, aber m-mal billiger (Test gegen Rohnormalzahlen)."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2

import aps_constants as C

_MASK = (1 << 64) - 1
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_TWO53 = float(1 << 53)
D2 = 2.0 / np.sqrt(np.pi)       # 1,1284: Erwartung der gleitenden Spannweite von zwei N(0, 1)-Werten, d₂ für n = 2


def uniforms(seed: int, start: int, count: int) -> np.ndarray:
    """Die Ziehungen Nr. start … start + count − 1 (ab 0) des SplitMix64-Stroms mit diesem Zufalls-Seed, gleichverteilt in [0, 1) mit 53 Bit.
    Bitgleich mit `SplitMix64(seed).random()`: Der Zustand nach der i-ten Ziehung ist seed + (i + 1)·Γ (Rest modulo 2⁶⁴), das Mischen ist elementweise."""
    idx = np.arange(start, start + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & _MASK) + (idx + np.uint64(1)) * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / _TWO53


def box_muller(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Standardnormalverteilt wie `SplitMix64.normal()` (u1 nach unten durch 1e-12 begrenzt, nur die Kosinus-Hälfte)."""
    return np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-12))) * np.cos(2.0 * np.pi * u2)


def normals(seed: int, start: int, count: int) -> np.ndarray:
    """Die Normalzahlen Nr. start … start + count − 1 des Stroms: Nr. i nutzt die Ziehungen 2i und 2i + 1 (wie `SplitMix64.normal()`, das zwei Ziehungen verbraucht)."""
    u = uniforms(seed, 2 * start, 2 * count)
    return box_muller(u[0::2], u[1::2])


def open_unit(u: np.ndarray) -> np.ndarray:
    """Ziehungen aus [0, 1) in das offene Intervall (0, 1) schieben (halbe Schrittweite 2⁻⁵⁴), damit Φ⁻¹ und χ²-Quantile endlich bleiben."""
    return np.minimum(u + 0.5 / _TWO53, 1.0 - 1.0 / _TWO53)       # u + 2⁻⁵⁴ rundet für u = 1 − 2⁻⁵³ auf 1,0: auf den größten Wert unter 1 begrenzen


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------------------

def estimate_s(x: np.ndarray) -> np.ndarray:
    """Stichproben-Standardabweichung mit Nenner m − 1 (je Zeile, letzte Achse)."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] < 2:
        raise ValueError("s braucht mindestens zwei Beobachtungen")
    return x.std(axis=-1, ddof=1)


def estimate_mr(x: np.ndarray) -> np.ndarray:
    """Mittlere gleitende Spannweite geteilt durch d₂ = 1,1284 (üblich für Einzelwerte; Zwetsloot/Woodall 2017 nennen ihn)."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] < 2:
        raise ValueError("MR braucht mindestens zwei Beobachtungen")
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1) / D2


def estimate(x: np.ndarray, estimator: str = "s"):
    """(x̄, σ̂) je Zeile; estimator „s“ oder „MR“."""
    x = np.asarray(x, dtype=float)
    if estimator == "s":
        return x.mean(axis=-1), estimate_s(x)
    if estimator == "MR":
        return x.mean(axis=-1), estimate_mr(x)
    raise ValueError(f"Schätzer „{estimator}“ unbekannt (erlaubt: s, MR)")


# --- Stichproben --------------------------------------------------------------------------------------------------------------------------------------

def phase1_sample(seed: int, m: int) -> np.ndarray:
    """EINE Phase-I-Stichprobe aus m Rohnormalzahlen (wahre Reihe N(0, 1)), Zufalls-Seed `seed`."""
    if m < 0:
        raise ValueError("m darf nicht negativ sein")
    return normals(seed, 0, m)


def phase1_direct(seed: int, m: int, count: int):
    """`count` Phase-I-Stichproben als (x̄, s), direkt aus der Verteilung gezogen (Zufalls-Seed `seed`): x̄ = Φ⁻¹(u)/√m, s² = χ²-Quantil(u′; m − 1)/(m − 1)."""
    if m < 2:
        raise ValueError("m muss mindestens 2 sein")
    u = open_unit(uniforms(seed, 0, 2 * count))
    xb = ndtri(u[:count]) / np.sqrt(m)
    s = np.sqrt(chi2.ppf(u[count:], m - 1.0) / (m - 1.0))
    return xb, s


def phase1_raw_batch(seed: int, m: int, count: int) -> np.ndarray:
    """`count` Phase-I-Stichproben aus Rohnormalzahlen, Form (count, m) (für die Störungen)."""
    return normals(seed, 0, count * m).reshape(count, m)


# --- Störungen der Phase I (Experiment 5) ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Contaminated:
    data: np.ndarray        # (count, m)
    mu_phase2: float        # wahres Mittel in Phase II (Stufe: Zustand der zweiten Hälfte; sonst 0)


def contaminate(x: np.ndarray, kind: str, size: float) -> Contaminated:
    """Störung der Phase I. „keine“; „Ausreißer“: eine Beobachtung in der Mitte (Index m // 2) um `size` erhöht, Phase II ohne Störung; „Stufe“: die zweite Hälfte (Index ≥ m // 2)
    um `size` erhöht, Phase II läuft im Zustand der zweiten Hälfte (Setzung dieses Stücks)."""
    x = np.array(x, dtype=float)
    m = x.shape[-1]
    if kind == "keine":
        return Contaminated(x, 0.0)
    if kind == "Ausreißer":
        x[..., m // 2] += size
        return Contaminated(x, 0.0)
    if kind == "Stufe":
        x[..., m // 2:] += size
        return Contaminated(x, float(size))
    raise ValueError(f"Störung „{kind}“ unbekannt (erlaubt: keine, Ausreißer, Stufe)")
