"""Kern: die Shewhart-Einzelkarte mit aus Phase I geschätzten Grenzen, exakt gerechnet (numpy, scipy).

Modell: Einzelwerte unabhängig N(0, 1) (Einheit: die wahre Streuung; die Karte kennt sie nicht). Phase I: m Beobachtungen in Kontrolle, daraus x̄ und s (Nenner m − 1).
Phase II: Alarm, wenn |x − x̄| > L·s. Für EINE feste Phase-I-Stichprobe ist die Alarmwahrscheinlichkeit je Beobachtung
    p(x̄, s) = Φ(x̄ − L s) + Φ(−x̄ − L s)        (wahre Beobachtung N(0, 1); bei Verschiebung μ: Φ(x̄ − L s − μ) + Φ(μ − x̄ − L s))
und die Lauflänge bis zum Alarm ist geometrisch mit diesem p: die BEDINGTE ARL₀ ist 1/p. Über Phase-I-Stichproben ist x̄ ~ N(0, 1/m) und s² ~ χ²(m − 1)/(m − 1), unabhängig.

Vier Maße für „die“ ARL₀ (Namensfalle der Linie): (1) die Verteilung der bedingten ARL₀ (Median, Quantile, Anteil unter einer Schwelle) = `cdf_carl`; (2) ihr Mittel E[1/p]
(= `mean_carl`); (3) der Kehrwert der mittleren Alarmrate 1/E[p] (= `mean_alarm_rate`, geschlossen über die t-Verteilung); (4) die unbedingte Lauflängenverteilung E[(1 − p)ⁿ]
(eingefroren, `aps_reference`).

Die Verteilungsfunktion `cdf_carl` ist zufallsfrei: Für jedes x̄ fällt p in s, p > 1/a heißt s < s*(x̄); P(bedingte ARL₀ < a) = ∫ 2φ(z) F_χ²(ν s*²; ν) dz mit z = √m·x̄, ν = m − 1.
Der Integrand ist glatt (anders als beim Integral über s, das am Wurzelknick bei s₀ Gauss-Legendre-Knoten verliert: die erste Fassung der Vormessung war deshalb ab m ≈ 1000 um bis zu 10⁻³ ungenau)."""

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import chi2, f as f_dist, t as t_dist

import aps_constants as C

# Gauss-Legendre-Gitter über z = √m·x̄ ≥ 0 (90 Teilstücke à 16 Knoten bis z = 9; die Dichte hinter 9 liegt unter 1e-18)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_EDGES = np.linspace(0.0, 9.0, 91)
_MID = 0.5 * (_EDGES[1:] + _EDGES[:-1])
_HALF = 0.5 * (_EDGES[1:] - _EDGES[:-1])
_Z = (_MID[:, None] + _HALF[:, None] * _GL_X[None, :]).ravel()
_WZ = (_HALF[:, None] * _GL_W[None, :]).ravel()
_PHI_Z = 2.0 * np.exp(-0.5 * _Z * _Z) / math.sqrt(2.0 * math.pi)       # 2 φ(z): zweiseitig über ±z


def _check_m(m):
    if int(m) != m or m < 2:
        raise ValueError(f"m = {m}: die Standardabweichung s braucht mindestens m = 2 Beobachtungen (m − 1 Freiheitsgrade)")
    return int(m)


# --- Einzelne Karte ---------------------------------------------------------------------------------------------------------------------------------

def p_alarm(xb, s, L=C.L_NOMINAL, mu=0.0, sigma=1.0):
    """Alarmwahrscheinlichkeit je Beobachtung der Karte mit Grenzen x̄ ± L·s, wahre Beobachtung N(mu, sigma²) (numpy-Felder erlaubt; die App rechnet in Einheiten sigma = 1)."""
    xb, s = np.asarray(xb), np.asarray(s)
    return ndtr((xb - L * s - mu) / sigma) + ndtr((mu - xb - L * s) / sigma)


def log_p_alarm(xb, s, L=C.L_NOMINAL, mu=0.0):
    return np.logaddexp(log_ndtr(np.asarray(xb) - L * np.asarray(s) - mu), log_ndtr(mu - np.asarray(xb) - L * np.asarray(s)))


def conditional_arl(xb, s, L=C.L_NOMINAL, mu=0.0):
    """Bedingte ARL gegeben die Phase-I-Stichprobe: 1/p (Beobachtungen bis zum Alarm, geometrisch)."""
    return 1.0 / p_alarm(xb, s, L, mu)


def alarm_within(p, n):
    """Wahrscheinlichkeit eines Alarms in den ersten n Beobachtungen bei Alarmwahrscheinlichkeit p je Beobachtung."""
    with np.errstate(divide="ignore"):                       # p = 1: log1p(−1) = −∞, das Ergebnis ist 1
        return -np.expm1(n * np.log1p(-np.asarray(p, dtype=float)))


def geometric_quantile(p, q):
    """Quantil q der geometrischen Lauflänge (kleinstes n mit P(RL ≤ n) ≥ q), stetig gerechnet: ln(1 − q)/ln(1 − p)."""
    return math.log1p(-q) / math.log1p(-p)


def known_alarm_prob(L=C.L_NOMINAL):
    """Alarmwahrscheinlichkeit je Beobachtung bei bekannten Parametern: 2 Φ(−L)."""
    return 2.0 * float(ndtr(-L))


def nominal_arl(L=C.L_NOMINAL):
    """ARL₀ der Karte mit bekannten Parametern: 1/(2 Φ(−L)); für L = 3: 370,398."""
    return 1.0 / known_alarm_prob(L)


def arl1_known(L, delta):
    """ARL der zentrierten Karte mit bekannten Parametern bei einem Sprung um delta σ."""
    return 1.0 / (float(ndtr(-L - delta)) + float(ndtr(delta - L)))


def effective_limit(p):
    """Grenzfaktor L_eff einer zentrierten Karte mit Alarmwahrscheinlichkeit p je Beobachtung (Kompromisskurve ARL₀ gegen ARL₁)."""
    return -ndtri(np.asarray(p) / 2.0)


# --- Verteilung der bedingten ARL₀ (zufallsfrei) ----------------------------------------------------------------------------------------------------

def _s_star(xb, p0, L):
    """Zu jedem x̄ die Schwelle s*, bei der p(x̄, s*) = p0 gilt (p fällt in s; Bisektion, vektorisiert)."""
    lo = np.zeros_like(xb)
    hi = np.full_like(xb, max(10.0, 4.0 + abs(ndtri(min(p0, 0.5) / 2.0)) / L))
    lp0 = math.log(p0)
    for _ in range(60):
        md = 0.5 * (lo + hi)
        big = log_p_alarm(xb, md, L) > lp0
        lo = np.where(big, md, lo)
        hi = np.where(big, hi, md)
    return 0.5 * (lo + hi)


def _integral_over_xbar(a, m, L, survival):
    p0 = 1.0 / a
    nu = m - 1.0
    sstar = _s_star(_Z / math.sqrt(m), p0, L)
    g = chi2.sf(nu * sstar ** 2, nu) if survival else chi2.cdf(nu * sstar ** 2, nu)
    return float((_WZ * _PHI_Z * g).sum())


def _mu_threshold(p0, L):
    """Nur x̄ geschätzt (s = 1): kleinstes g mit p(g, 1) = p0, oder None, wenn schon das Zentrum p0 übersteigt."""
    if 2.0 * float(ndtr(-L)) >= p0:
        return None
    return brentq(lambda g: float(ndtr(g - L) + ndtr(-g - L)) - p0, 0.0, L + 40.0, xtol=1e-14)


def cdf_carl(a, m, L=C.L_NOMINAL, mode="both"):
    """P(bedingte ARL₀ < a) über die Phase-I-Stichproben, exakt (zufallsfrei). mode: "both" (x̄ und s geschätzt), "sigma" (nur s, x̄ = 0), "mu" (nur x̄, s = 1)."""
    m = _check_m(m)
    if a <= 1.0:
        return 0.0
    if math.isinf(a):
        return 1.0
    p0 = 1.0 / a
    nu = m - 1.0
    if mode == "sigma":
        s0 = -float(ndtri(p0 / 2.0)) / L
        return float(chi2.cdf(nu * s0 * s0, nu))
    if mode == "mu":
        g = _mu_threshold(p0, L)
        return 1.0 if g is None else float(2.0 * ndtr(-math.sqrt(m) * g))
    return _integral_over_xbar(a, m, L, survival=False)


def sf_carl(a, m, L=C.L_NOMINAL, mode="both"):
    """P(bedingte ARL₀ ≥ a), am Rand genau gerechnet (nicht 1 − cdf)."""
    m = _check_m(m)
    if a <= 1.0:
        return 1.0
    if math.isinf(a):
        return 0.0
    p0 = 1.0 / a
    nu = m - 1.0
    if mode == "sigma":
        s0 = -float(ndtri(p0 / 2.0)) / L
        return float(chi2.sf(nu * s0 * s0, nu))
    if mode == "mu":
        g = _mu_threshold(p0, L)
        return 0.0 if g is None else float(2.0 * ndtr(math.sqrt(m) * g) - 1.0)
    return _integral_over_xbar(a, m, L, survival=True)


def quantile_carl(q, m, L=C.L_NOMINAL, mode="both"):
    """Quantil q der bedingten ARL₀ (Zähleinheit: Phase-I-Stichproben)."""
    m = _check_m(m)
    if not 0.0 < q < 1.0:
        raise ValueError("q muss zwischen 0 und 1 liegen")
    return float(math.exp(brentq(lambda la: cdf_carl(math.exp(la), m, L, mode) - q, math.log(1.0 + 1e-9), math.log(1e15), xtol=1e-11)))


def mean_alarm_rate(m, L=C.L_NOMINAL):
    """E[p] geschlossen: (X − x̄)/(s·√(1 + 1/m)) ist t(m − 1), also E[p] = 2 P(t(m − 1) > L/√(1 + 1/m)). Die Grenzen sind ein Vorhersageintervall, kein 3σ-Intervall."""
    m = _check_m(m)
    return float(2.0 * t_dist.sf(L / math.sqrt(1.0 + 1.0 / m), m - 1))


def inverse_mean_alarm_rate(m, L=C.L_NOMINAL):
    """1/E[p]: der Kehrwert der mittleren Alarmrate (viele Karten gleichzeitig, jede mit eigener Phase I)."""
    return 1.0 / mean_alarm_rate(m, L)


def mean_carl_is_finite(m, L=C.L_NOMINAL):
    """Ist E[1/p] endlich? 1/p ~ exp((L s − |x̄|)²/2) gegen die Dichte exp(−(m − 1)s²/2 − m x̄²/2): am Rand x̄ = 0 hat s² den Koeffizienten (L² − m + 1)/2,
    negativ genau für m > L² + 1 (L = 3: m ≥ 11). Eigene Herleitung, numerisch bestätigt (abgeschnittenes Integral, Test)."""
    return m - 1.0 > L * L + 1e-12


def mean_carl(m, L=C.L_NOMINAL, rel=1e-7):
    """E[1/p], das Mittel der bedingten ARL₀ (= unbedingte ARL im Sinn von Lombard/Hawkins). Über den Tail: E = 1 + ∫₁^∞ P(ARL ≥ a) da, in t = ln a integriert.
    Das ist ein ANDERER Weg als das Doppelintegral in tools/aps_make_exact.py (dort stehen die eingefrorenen Werte). Unendlich für m ≤ L² + 1."""
    m = _check_m(m)
    if not mean_carl_is_finite(m, L):
        return math.inf
    f = lambda t: math.exp(t) * sf_carl(math.exp(t), m, L)
    total, upper = 1.0, 0.0
    # Abschnitte der Länge 6 in t bis der Beitrag verschwindet; je Abschnitt quad mit relativer Toleranz
    for _ in range(60):
        part, _err = quad(f, upper, upper + 6.0, epsabs=0.0, epsrel=rel, limit=100)
        total += part
        upper += 6.0
        if part < 1e-11 * total:
            break
    return total


# --- Mindestzahl, Grenzen weiten ----------------------------------------------------------------------------------------------------------------------

def min_m(f, eps, L=C.L_NOMINAL, mode="both", lo=2, hi=100000):
    """Kleinstes ganzzahliges m mit P(bedingte ARL₀ < f·370,398) ≤ eps (Bisektion; die Verteilungsfunktion fällt in m). None, wenn schon m = hi nicht reicht."""
    ok = lambda m: cdf_carl(f * C.TARGET_ARL0, m, L, mode) <= eps
    if ok(lo):
        return lo
    if not ok(hi):
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def limit_equal_alarm_rate(m):
    """L(a): Grenzfaktor, bei dem die mittlere Alarmrate 1/370,398 ist: L = √(1 + 1/m)·t(m − 1; 1 − p₀/2), geschlossen."""
    m = _check_m(m)
    return math.sqrt(1.0 + 1.0 / m) * float(t_dist.isf(C.P_NOMINAL / 2.0, m - 1))


def limit_guarantee(m, f=0.5, eps=0.05, mode="both"):
    """L(c): Grenzfaktor, bei dem P(bedingte ARL₀ < f·370,398) = eps ist (Garantie mit Wahrscheinlichkeit)."""
    m = _check_m(m)
    g = lambda L: cdf_carl(f * C.TARGET_ARL0, m, L, mode) - eps
    if g(80.0) > 0.0:
        raise ValueError(f"m = {m}: auch bei sehr weiten Grenzen bleibt der Anteil über {eps}")
    return float(brentq(g, 1.5, 80.0, xtol=1e-10))


# --- Hotelling-T² mit geschätztem Mittel und geschätzter Kovarianz (nur die Zeile J der Annahmen-Tabelle; ganze Rechnung: Stück 4b) --------------------

def _t2_scale(p, m):
    """T² = [p (m + 1)(m − 1)/(m (m − p))]·F(p, m − p) für p Kanäle, m Phase-I-Vektoren (Standardsatz zu z′S⁻¹z; eigene Herleitung, Simulation bestätigt sie)."""
    if p < 1 or m <= p:
        raise ValueError(f"p = {p}, m = {m}: nötig ist m > p ≥ 1")
    return p * (m + 1.0) * (m - 1.0) / (m * (m - p))


def t2_mean_alarm_rate(p, m, ucl):
    """Mittlere Alarmrate je Phase-II-Vektor (über Phase-I-Stichproben) bei Grenze ucl für T²; p = 1 ist die t-Formel von `mean_alarm_rate` (ucl = L²)."""
    return float(f_dist.sf(ucl / _t2_scale(p, m), p, m - p))


def t2_chi2_limit(p):
    """χ²-Regel (Σ bekannt): UCL = χ²-Quantil 1 − p₀; ARL₀ 370,398 nur für bekanntes Σ."""
    return float(chi2.isf(C.P_NOMINAL, p))


def t2_f_limit(p, m):
    """F-Regel: UCL so, dass die unbedingte ARL₀ (Kehrwert der mittleren Alarmrate) 370,398 ist."""
    return float(f_dist.isf(C.P_NOMINAL, p, m - p)) * _t2_scale(p, m)


# --- Endlich viele Phase-I-Ausgänge: die vier Maße als Summen (Spielzeug des Handbeispiels und diskretes Gegenstück der Integrale) -------------------------------

def mixture_measures(ps, weights=None):
    """Die vier Maße für endlich viele mögliche Phase-I-Ausgänge mit Alarmwahrscheinlichkeiten `ps` (je Beobachtung) und Gewichten `weights` (Summe 1; gleich verteilt, wenn None).
    Rechnet mit Brüchen, wenn die Eingaben `fractions.Fraction` sind. Rückgabe: `mean_carl` (Mittel der bedingten ARL, Σ w/p), `inv_mean_alarm` (1/Σ w p) und `survival(n)`
    (P(Lauflänge > n) = Σ w (1 − p)ⁿ, jede Lauflänge mit frischer Phase I). Das Integral über (x̄, s) ist derselbe Ausdruck mit unendlich vielen Ausgängen."""
    ps = list(ps)
    unit = ps[0] * 0 + 1                                    # Eins im Typ der Eingabe (Fraction bleibt Fraction)
    w = list(weights) if weights is not None else [unit / len(ps)] * len(ps)
    return {
        "mean_carl": sum(wi / pi for wi, pi in zip(w, ps)),
        "inv_mean_alarm": 1 / sum(wi * pi for wi, pi in zip(w, ps)),
        "survival": lambda n: sum(wi * (1 - pi) ** n for wi, pi in zip(w, ps)),
    }
