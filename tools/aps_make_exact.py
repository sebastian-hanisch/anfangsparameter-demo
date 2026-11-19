"""Rechnet die eingefrorenen Tabellen von `aps_exact.json` (Stück „Parameter schätzen“) und prüft sie.

    python tools/aps_make_exact.py --write            # rechnet alles neu und schreibt aps_exact.json (rund 3 bis 6 Minuten, ein Thread)
    python tools/aps_make_exact.py --check            # rechnet alles neu und vergleicht mit aps_exact.json auf 1e-6 relativ (rund 3 bis 6 Minuten)
    python tools/aps_make_exact.py --check --quick    # nur m = 100 und 300 (wenige Sekunden)

Alle Felder sind Rechnungen mit dem DOPPELINTEGRAL über (x̄, s) (Gauss-Legendre-Gitter, im Logarithmus gerechnet): ein anderer Rechenweg als der der App (`aps_exact.mean_carl`
integriert den Tail der exakten Verteilungsfunktion). Die Tests vergleichen beide Wege. Das Skript kopiert keinen Code der App; es benutzt nur `aps_constants`.

Voraussetzungen: Einzelwerte unabhängig N(0, 1), Phase I mit m Beobachtungen in Kontrolle, Grenzen x̄ ± L·s (zweiseitig), ARL₀-Ziel 370,398 (= 1/p der 3σ-Karte).
Felder: `mean_carl` (E[1/p], das Mittel der bedingten ARL₀, m ≥ 11), `unbedingt` (unbedingte Lauflängenverteilung E[(1 − p)ⁿ], jede Lauflänge mit frischer Phase I),
`weiten` (Grenzfaktoren L(a), L(b), L(c) und ARL₁ bei bekannten Parametern), `abhaengig` (Tabelle I aus der Vormessung V13b, nicht von diesem Skript gerechnet)."""

import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, log_ndtr, ndtr
from scipy.stats import t as t_dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import aps_constants as C  # noqa: E402

PATH = os.path.join(ROOT, "aps_exact.json")
STAND = "2026-11-18"
M_MEAN = tuple(m for m in C.M_OPTIONS if m >= 11)           # E[1/p] ist endlich genau für m >= 11
M_UNBEDINGT = (30, 100, 500)
M_WEITEN = (30, 50, 100, 300, 1000)
QUICK_M = (100, 300)

# Tabelle I (Phase I bei autokorrelierten Werten, AR(1), Randverteilung N(0, 1), Rohdatenkarte x̄ ± 3 s als wären die Werte unabhängig): aus der Vormessung V13b
# (PLAN-04 Befund I; Simulation, 4000 Phase-I-Stichproben je Zeile, feste Zufalls-Seeds; bedingte ARL₀ je Stichprobe exakt über die Integralgleichung des AR(1)-Kerns).
# Nicht von diesem Skript gerechnet und in diesem Stück nicht neu gemessen; Stück 4b misst neu.
TABELLE_I = {
    "m": [30, 100, 300, 1000],
    "phi": [0.0, 0.3, 0.6, 0.9],
    "arl0_bekannt": [370.4, 376.4, 419.4, 831.8],
    "anteil_unter_halber_arl0": [[0.356, 0.186, 0.042, 0.001], [0.427, 0.226, 0.056, 0.002], [0.548, 0.298, 0.089, 0.003], [0.793, 0.440, 0.134, 0.004]],
    "median": [[287, 336, 358, 371], [236, 317, 356, 370], [155, 289, 360, 402], [50, 225, 480, 696]],
}


# --- Doppelintegral ---------------------------------------------------------------------------------------------------------------------------------

def _panels(a, b, panels, order):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def _expect(m, fun, L=3.0, s_hi=None, scale=1, logmode=False):
    """E[fun(log p)] über x̄ ~ N(0, 1/m) und s = χ_{m−1}/√(m − 1) (unabhängig); logmode: fun gibt den LOGARITHMUS der Werte (verhindert Überlauf bei 1/p)."""
    nu = m - 1.0
    hi = s_hi if s_hi is not None else min(12.0, 1.0 + 16.0 / math.sqrt(nu))
    s, ws = _panels(0.0, hi, int(max(80, hi / 0.02) * scale), 12)
    X = 9.0 / math.sqrt(m)
    x, wx = _panels(-X, X, 40 * scale, 8)
    lw_s = math.log(2.0) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1.0) * np.log(s) - 0.5 * nu * s * s
    lw_x = 0.5 * math.log(m / (2.0 * math.pi)) - 0.5 * m * x * x
    total = 0.0
    for i in range(0, s.size, 200):
        sb = s[i:i + 200]
        lp = np.logaddexp(log_ndtr(x[None, :] - L * sb[:, None]), log_ndtr(-x[None, :] - L * sb[:, None]))
        val = fun(lp)
        logw = lw_s[i:i + 200, None] + lw_x[None, :]
        total += (ws[i:i + 200, None] * wx[None, :] * (np.exp(logw + val) if logmode else np.exp(logw) * val)).sum()
    return float(total)


def e_inv_p(m, L=3.0, scale=1):
    return _expect(m, lambda lp: -lp, L, s_hi=12.0 if m < 30 else None, scale=scale, logmode=True)


def e_alarm_within(m, n, L=3.0):
    """E[1 − (1 − p)ⁿ]: Wahrscheinlichkeit eines Alarms in den ersten n Beobachtungen (Mittel über Phase-I-Stichproben; n darf reell sein)."""
    return _expect(m, lambda lp: -np.expm1(n * np.log1p(-np.exp(lp))), L)


def arl1(L, delta=1.0):
    return 1.0 / (float(ndtr(-L - delta)) + float(ndtr(delta - L)))


# --- Felder -----------------------------------------------------------------------------------------------------------------------------------------

def field_mean_carl(ms):
    out = {}
    for m in ms:
        a, b = e_inv_p(m), e_inv_p(m, scale=2)
        assert abs(a / b - 1.0) < 1e-6, (m, a, b)
        out[str(m)] = b
        print(f"  mean_carl m = {m}: {b:.6f}", flush=True)
    return out


def field_unbedingt(ms):
    out = {}
    for m in ms:
        def quantile(q):
            return brentq(lambda n: e_alarm_within(m, n) - q, 0.5, 20000.0, xtol=1e-3)
        out[str(m)] = {"p_alarm_100": e_alarm_within(m, 100), "median": quantile(0.5), "q05": quantile(0.05), "q95": quantile(0.95)}
        print(f"  unbedingt m = {m}: {out[str(m)]}", flush=True)
    return out


def _tail_share(m, L, f=0.5):
    """P(bedingte ARL₀ < f·370,398) für den Wert L (aus dem Doppelintegral: Anteil der (x̄, s) mit p > 1/a); hier nur über die Gauss-Gitter-Summe der Indikatorfunktion
    ist zu grob, darum der exakte Weg über die Schranke s*(x̄) mit Bisektion. Die App rechnet denselben Ausdruck mit `aps_exact.cdf_carl`; dieses Skript hat seine eigene Fassung."""
    from scipy.stats import chi2
    a = f * C.TARGET_ARL0
    p0 = 1.0 / a
    nu = m - 1.0
    x, w = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, 9.0, 91)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    z = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    wz = (half[:, None] * w[None, :]).ravel()
    xb = z / math.sqrt(m)
    lo, hi = np.zeros_like(xb), np.full_like(xb, 10.0)
    for _ in range(60):
        md = 0.5 * (lo + hi)
        big = np.logaddexp(log_ndtr(xb - L * md), log_ndtr(-xb - L * md)) > math.log(p0)
        lo, hi = np.where(big, md, lo), np.where(big, hi, md)
    return float((wz * 2.0 * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * chi2.cdf(nu * (0.5 * (lo + hi)) ** 2, nu)).sum())


def field_weiten(ms):
    out = {}
    for m in ms:
        La = math.sqrt(1.0 + 1.0 / m) * float(t_dist.isf(C.P_NOMINAL / 2.0, m - 1))
        Lb = brentq(lambda L: e_inv_p(m, L, scale=1) - C.TARGET_ARL0, 1.5, 3.0, xtol=1e-9)
        Lc = brentq(lambda L: _tail_share(m, L) - 0.05, 1.5, 8.0, xtol=1e-9)
        out[str(m)] = {"L_a": La, "L_b": Lb, "L_c": Lc, "arl1_L3": arl1(3.0), "arl1_a": arl1(La), "arl1_b": arl1(Lb), "arl1_c": arl1(Lc)}
        print(f"  weiten m = {m}: L(a) {La:.4f}, L(b) {Lb:.4f}, L(c) {Lc:.4f}", flush=True)
    return out


def compute(quick=False):
    t0 = time.time()
    quick_ms = lambda ms: tuple(m for m in ms if m in QUICK_M) if quick else ms
    print("mean_carl …", flush=True)
    mean_carl = field_mean_carl(quick_ms(M_MEAN))
    print("unbedingt …", flush=True)
    unbedingt = field_unbedingt(quick_ms(M_UNBEDINGT))
    print("weiten …", flush=True)
    weiten = field_weiten(quick_ms(M_WEITEN))
    print(f"fertig in {time.time() - t0:.0f} s (nur zur Kontrolle, kein Befund)")
    return mean_carl, unbedingt, weiten


def build(mean_carl, unbedingt, weiten):
    return {
        "mean_carl": {
            "quelle": "Doppelintegral über (x̄, s), Gauss-Legendre im Logarithmus, Gitter und doppeltes Gitter gleich auf 1e-6 relativ (tools/aps_make_exact.py); E[1/p] = Mittel der bedingten ARL₀ der Shewhart-Karte, L = 3, Einzelwerte iid N(0, 1)",
            "stand": STAND, "werte": mean_carl},
        "unbedingt": {
            "quelle": "Doppelintegral E[1 − (1 − p)ⁿ] (jede Lauflänge mit frischer Phase I); Median und Quantile als reelles n mit P(RL ≤ n) = q (Bisektion, xtol 1e-3), L = 3 (tools/aps_make_exact.py)",
            "stand": STAND, "werte": unbedingt},
        "weiten": {
            "quelle": "L(a) geschlossen (t-Formel), L(b) E[1/p] = 370,398 (Doppelintegral), L(c) P(bedingte ARL₀ < 185,2) = 5 % (exakte Verteilungsfunktion mit eigener Bisektion); ARL₁(δ = 1) der zentrierten Karte mit bekannten Parametern und diesem L (Preis, nicht die ARL₁ der geschätzten Karte) (tools/aps_make_exact.py)",
            "stand": STAND, "werte": weiten},
        "abhaengig": {
            "quelle": "PLAN-04 Befund I = Vormessung V13b (Simulation, 4000 Phase-I-Stichproben je Zeile, Rohdatenkarte x̄ ± 3 s auf AR(1)-Reihen mit Randverteilung N(0, 1)); in diesem Stück nicht gerechnet, Stück 4b misst neu",
            "stand": STAND, "werte": TABELLE_I},
    }


def compare(old, new, path=""):
    """Rekursiver Vergleich auf 1e-6 relativ; liefert die Liste der Abweichungen."""
    bad = []
    if isinstance(new, dict):
        for k, v in new.items():
            if k not in old:
                bad.append(f"{path}/{k} fehlt in der Datei")
            else:
                bad += compare(old[k], v, f"{path}/{k}")
    elif isinstance(new, (int, float)):
        if abs(old - new) > 1e-6 * max(1.0, abs(new)):
            bad.append(f"{path}: Datei {old}, neu gerechnet {new}")
    return bad


def main(argv):
    quick = "--quick" in argv
    if "--write" in argv and "--check" in argv or not ({"--write", "--check"} & set(argv)):
        print(__doc__)
        return 2
    if "--write" in argv and quick:
        print("--write mit --quick würde die Datei verstümmeln")
        return 2
    mean_carl, unbedingt, weiten = compute(quick)
    fresh = build(mean_carl, unbedingt, weiten)
    if "--write" in argv:
        with open(PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(fresh, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print(f"{PATH} geschrieben")
        return 0
    with open(PATH, encoding="utf-8") as fh:
        stored = json.load(fh)
    bad = []
    for key in ("mean_carl", "unbedingt", "weiten"):
        bad += compare(stored[key]["werte"], fresh[key]["werte"], key)
    bad += compare(stored["abhaengig"]["werte"], TABELLE_I, "abhaengig")
    if bad:
        print("ABWEICHUNGEN:\n" + "\n".join(bad))
        return 1
    print("aps_exact.json stimmt mit der Neurechnung überein (1e-6 relativ)" + (" (schneller Modus: nur m = 100 und 300)" if quick else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
