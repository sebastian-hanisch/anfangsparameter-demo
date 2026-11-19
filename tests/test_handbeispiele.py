"""Handbeispiele H 1 bis H 6 (Plan 04) als exakte Tests: Brüche, Ganzzahlen, Tafelwerte, geschlossene Formeln; die Demo-Funktionen werden gegen sie gehalten.

Die Phase-I-Stichproben mit m = 3 sind ausgedacht (zum Nachrechnen von Hand, ohne Aussage über übliche Stichprobengrößen). Zweite Rechenwege: Doppelintegral (scipy.dblquad),
Student-t-Verteilung, lineares System in Brüchen gegen numpy, Summe der Überlebenswahrscheinlichkeiten."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammaln, log_ndtr, ndtr
from scipy.stats import chi2, norm, t as t_dist

import aps_constants as C
import aps_cusum as CU
import aps_evaluation as ev
import aps_exact as E
import aps_scenario as S


# --- H 1: die bekannte Karte ------------------------------------------------------------------------------------------------------------------------

def test_h1_known_card():
    p0 = E.known_alarm_prob(3.0)
    assert 2 * 0.001350 == pytest.approx(p0, abs=1e-6)          # Tafelwert Φ(−3) = 0,001350
    assert p0 == pytest.approx(0.002699796, abs=1e-9)
    assert E.nominal_arl(3.0) == pytest.approx(370.3983, abs=1e-3)
    assert C.TARGET_ARL0 == pytest.approx(370.3983, abs=1e-3)
    assert float(E.alarm_within(p0, 100)) == pytest.approx(0.2369, abs=5e-5)   # 23,69 % in den ersten 100 Beobachtungen


# --- H 2: die vier Maße in Brüchen (Spielzeug, ausgedacht) ------------------------------------------------------------------------------------------------

def test_h2_toy_measures_in_fractions():
    mm = E.mixture_measures([F(1, 10), F(1, 100)])
    assert mm["mean_carl"] == 55                                # Mittel der bedingten ARL 10 und 100
    assert mm["inv_mean_alarm"] == F(200, 11)                   # 18,18: Kehrwert der mittleren Alarmrate 11/200
    assert mm["mean_carl"] > mm["inv_mean_alarm"]               # Jensen
    surv = mm["survival"]
    assert surv(0) == 1 and 1 - surv(1) == F(11, 200)           # P(Lauflänge = 1) ist die mittlere Alarmrate


def test_h2_median_is_18_and_sum_of_survival_is_the_mean():
    surv = E.mixture_measures([F(1, 10), F(1, 100)])["survival"]
    assert surv(17) > F(1, 2) >= surv(18)
    assert surv(18) < F(1, 2) and float(surv(17)) == pytest.approx(0.505, abs=5e-4)
    assert sum(float(surv(n)) for n in range(6000)) == pytest.approx(55.0, abs=1e-2)       # zweiter Weg: E[N] = Σ P(N > n)
    assert float(1 - surv(55)) > 0.7                            # 70 % aller Läufe alarmieren spätestens beim Mittel 55


# --- H 3: drei Phase-I-Beobachtungen, drei Karten -----------------------------------------------------------------------------------------------------------

def test_h3_three_samples_mean_and_std_exactly():
    for x, (xb, s) in (((F(-1, 2), F(0), F(1, 2)), (F(0), F(1, 2))), ((F(0), F(1, 2), F(1)), (F(1, 2), F(1, 2))), ((F(-2), F(0), F(2)), (F(0), F(2)))):
        mean = sum(x) / 3
        s2 = sum((v - mean) ** 2 for v in x) / 2
        assert mean == xb and s2 == s * s                       # Brüche
        a = ev.analyse_values([float(v) for v in x])
        assert a.xb == pytest.approx(float(xb), abs=1e-15) and a.sigma == pytest.approx(float(s), abs=1e-15)


def test_h3_conditional_arl_against_table_values():
    rows = {r["name"][0]: r for r in ev.hand_analysis()}
    assert 2 * 0.066807 == pytest.approx(2 * ndtr(-1.5), abs=1e-6)      # Tafel Φ(−1,5)
    assert rows["A"]["arl0"] == pytest.approx(1 / (2 * 0.066807), abs=1e-3) and rows["A"]["arl0"] == pytest.approx(7.4846, abs=1e-3)
    assert (rows["A"]["lower"], rows["A"]["upper"]) == (-1.5, 1.5)
    assert 0.158655 + 0.022750 == pytest.approx(ndtr(-1.0) + ndtr(-2.0), abs=1e-6)      # Tafel Φ(−1), Φ(−2)
    assert rows["B"]["arl0"] == pytest.approx(1 / 0.181405, abs=1e-3) and rows["B"]["arl0"] == pytest.approx(5.5126, abs=1e-3)
    assert (rows["B"]["lower"], rows["B"]["upper"]) == (-1.0, 2.0)
    assert rows["C"]["arl0"] / 5.0679e8 == pytest.approx(1.0, abs=1e-3)                 # Φ(−6) = 9,866e-10
    assert (rows["C"]["lower"], rows["C"]["upper"]) == (-6.0, 6.0)


def test_h3_m3_variance_is_exponential_one():
    """s² ~ χ²(2)/2 = Exp(1): P(s < 1) = 1 − 1/e. Mit a = Ziel-ARL₀ ist s₀ = 1 (nur σ geschätzt: p = 2Φ(−3s) > p₀ genau für s < 1)."""
    assert E.cdf_carl(C.TARGET_ARL0, 3, mode="sigma") == pytest.approx(1 - math.exp(-1), abs=1e-12)
    assert 1 - math.exp(-1) == pytest.approx(0.632121, abs=1e-6)


def test_h3_half_target_share_with_only_sigma_estimated():
    """P(bedingte ARL₀ < 185,2, nur σ) = P(s < 0,9274) = 1 − exp(−0,8601) = 0,5769 (früher fälschlich 0,5772 notiert; −Φ⁻¹(p₀) = 2,7822, nicht 2,7782)."""
    s_half = -norm.ppf(C.P_NOMINAL) / 3
    assert -norm.ppf(C.P_NOMINAL) == pytest.approx(2.7822, abs=5e-5)
    assert s_half == pytest.approx(0.92739, abs=1e-5) and round(s_half ** 2, 4) == 0.8601
    assert E.cdf_carl(0.5 * C.TARGET_ARL0, 3, mode="sigma") == pytest.approx(1 - math.exp(-s_half ** 2), abs=1e-12)      # zwei Wege: χ²(2)-Verteilungsfunktion gegen 1 − e^(−s²)
    assert round(1 - math.exp(-s_half ** 2), 4) == 0.5769 and 1 - math.exp(-s_half ** 2) == pytest.approx(0.57686, abs=1e-5)


# --- H 4: mittlere Alarmrate bei m = 3 geschlossen; das Mittel der bedingten ARL₀ ist unendlich ----------------------------------------------------------

def _fs(s, m):
    nu = m - 1.0
    return math.exp(math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * math.log(s) - 0.5 * nu * s * s)


def _fx(x, m):
    return math.sqrt(m / (2 * math.pi)) * math.exp(-0.5 * m * x * x)


def test_h4_mean_alarm_rate_m3_closed_form_and_second_ways():
    c = 3 / math.sqrt(1 + 1 / 3)
    assert c * c == pytest.approx(27 / 4, abs=1e-12)
    closed = 1 - c / math.sqrt(2 + c * c)                       # t₂: P(|t| > c) = 1 − c/√(2 + c²)
    assert closed == pytest.approx(1 - math.sqrt(27 / 35), abs=1e-12)
    assert closed == pytest.approx(2 * t_dist.sf(c, 2), abs=1e-12)                    # zweiter Weg: Student-t-Verteilung
    assert E.mean_alarm_rate(3) == pytest.approx(closed, abs=1e-12)                    # die Demo-Funktion
    assert 1 / closed == pytest.approx(8.2177, abs=1e-3)
    num, _ = integrate.dblquad(lambda x, s: float(E.p_alarm(x, s)) * _fs(s, 3) * _fx(x, 3), 0, 14, lambda s: -8, lambda s: 8, epsabs=1e-10, epsrel=1e-10)
    assert num == pytest.approx(closed, abs=1e-6)                                      # dritter Weg: Doppelintegral


def _truncated(m, S):
    """E[1/p; s ≤ S] auf einem Gauss-Legendre-Gitter im Logarithmus (unabhängig von der Demo)."""
    xs, ws = np.polynomial.legendre.leggauss(40)

    def nodes(lo, hi, panels):
        e = np.linspace(lo, hi, panels + 1)
        return (((e[1:] + e[:-1]) / 2)[:, None] + ((e[1:] - e[:-1]) / 2)[:, None] * xs[None, :]).ravel(), (((e[1:] - e[:-1]) / 2)[:, None] * ws[None, :]).ravel()
    sn, sw = nodes(0, S, 30)
    xn, xw = nodes(-6.0, 6.0, 30)
    nu = m - 1.0
    lfs = math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * np.log(sn) - 0.5 * nu * sn * sn
    lfx = 0.5 * math.log(m / (2 * math.pi)) - 0.5 * m * xn * xn
    lp = np.logaddexp(log_ndtr(xn[None, :] - 3 * sn[:, None]), log_ndtr(-xn[None, :] - 3 * sn[:, None]))
    return float((sw[:, None] * xw[None, :] * np.exp(lfs[:, None] + lfx[None, :] - lp)).sum())


def test_h4_mean_of_conditional_arl_is_infinite_for_m3_and_m10_but_not_m11():
    assert _truncated(3, 6.0) > 10 * _truncated(3, 3.0)         # wächst ohne Grenze
    r10 = _truncated(10, 7.5) / _truncated(10, 5.0)
    assert 30 < r10 < 48                                        # S⁹ gäbe 1,5⁹ = 38,4
    assert _truncated(11, 12.0) / _truncated(11, 8.0) - 1 < 1e-6 and _truncated(11, 12.0) == pytest.approx(459653.27, rel=1e-4)
    assert not E.mean_carl_is_finite(3) and not E.mean_carl_is_finite(10) and E.mean_carl_is_finite(11)
    assert E.mean_carl(3) == math.inf and E.mean_carl(10) == math.inf


def _exponent(t, m):
    """Exponent je Radius² in Polarform (s = r cos t, |x̄| = r sin t): ((3 cos t − sin t)² − (m − 1) cos² t − m sin² t)/2."""
    return ((3 * math.cos(t) - math.sin(t)) ** 2 - (m - 1) * math.cos(t) ** 2 - m * math.sin(t) ** 2) / 2


def test_h4_hand_derivation_of_the_infinity_boundary():
    ts = [i * (math.pi / 2) / 2000 for i in range(2001)]
    assert _exponent(0.0, 3) == pytest.approx(3.5)                                    # (10 − m)/2 am Rand x̄ = 0
    assert all(_exponent(t, 3) <= _exponent(0.0, 3) + 1e-12 for t in ts)             # das Maximum liegt am Rand
    assert all(_exponent(0.0, m) > 0 for m in range(2, 10))
    assert _exponent(0.0, 10) == pytest.approx(0.0, abs=1e-12) and all(_exponent(t, 10) <= 1e-12 for t in ts)
    assert all(max(_exponent(t, m) for t in ts) == _exponent(0.0, m) < 0 for m in range(11, 400))
    for m in range(2, 40):
        assert (_exponent(0.0, m) < 0) == E.mean_carl_is_finite(m), m                 # die Demo-Regel m > L² + 1 stimmt mit dem Exponenten überein


# --- H 5: zwei Schätzer der Streuung -----------------------------------------------------------------------------------------------------------------

def test_h5_estimators_on_five_values(mini):
    x = [F(v) for v in mini]
    mean = sum(x) / 5
    assert mean == 3 and sum((v - mean) ** 2 for v in x) == 10 and sum((v - mean) ** 2 for v in x) / 4 == F(5, 2)
    mr = [abs(x[i + 1] - x[i]) for i in range(4)]
    assert mr == [2, 1, 3, 1] and sum(mr) / 4 == F(7, 4)
    assert float(S.estimate_s(np.array(mini))) == pytest.approx(math.sqrt(2.5), abs=1e-15) and math.sqrt(2.5) == pytest.approx(1.5811, abs=1e-4)
    assert S.D2 == pytest.approx(1.1284, abs=1e-4)
    assert float(S.estimate_mr(np.array(mini))) == pytest.approx(1.75 / S.D2, abs=1e-15) and 1.75 / S.D2 == pytest.approx(1.5508, abs=1e-4)


def test_h5_a_step_is_seen_by_s_and_hardly_by_mr():
    x = np.array([0.0, 0, 0, 0, 2, 2, 2, 2])
    assert float(S.estimate_s(x)) == pytest.approx(math.sqrt(8 / 7), abs=1e-12) and float(S.estimate_s(x)) == pytest.approx(1.07, abs=5e-3)
    assert float(S.estimate_mr(x)) == pytest.approx((2 / 7) / S.D2, abs=1e-12) and float(S.estimate_mr(x)) == pytest.approx(0.25, abs=5e-3)


# --- H 6: Karte mit drei Zuständen in Brüchen ------------------------------------------------------------------------------------------------------------

def _chain(u, d, h=3):
    """Schritte +1 (Wahrscheinlichkeit u), −1 (d), sonst 0; reflektiert bei 0; Alarm beim Erreichen von h (Rest der Wahrscheinlichkeit verlässt die Kette)."""
    Q = [[F(0)] * h for _ in range(h)]
    for i in range(h):
        stay = 1 - u - d
        if i + 1 < h:
            Q[i][i + 1] += u
        if i - 1 >= 0:
            Q[i][i - 1] += d
        else:
            stay += d
        Q[i][i] += stay
    return Q


def _solve_fractions(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def _arl_fractions(u, d):
    Q = _chain(u, d)
    return _solve_fractions([[(F(1) if i == j else F(0)) - Q[i][j] for j in range(3)] for i in range(3)], [F(1)] * 3)


def test_h6_three_state_chain_in_fractions_against_the_demo_chain_functions():
    L = _arl_fractions(F(1, 4), F(1, 4))
    assert L == [24, 20, 12]
    Q = np.array([[float(v) for v in row] for row in _chain(F(1, 4), F(1, 4))])
    for start in range(3):
        assert CU.chain_arl(Q, start) == pytest.approx(float(L[start]), abs=1e-12)    # Demo-Löser gegen Brüche
    surv = CU.chain_survival(Q, 4000)
    assert surv[0] == 1.0 and surv.sum() == pytest.approx(24.0, abs=1e-9)             # ARL = Σ P(Lauflänge > n)
    assert surv[1] == pytest.approx(float(sum(_chain(F(1, 4), F(1, 4))[0])), abs=1e-15)


def test_h6_the_up_probability_drives_the_arl():
    assert _arl_fractions(F(1, 8), F(1, 4))[0] == 88
    assert _arl_fractions(F(3, 8), F(1, 4))[0] == F(344, 27) and float(F(344, 27)) == pytest.approx(12.74, abs=5e-3)
    assert _arl_fractions(F(1, 8), F(1, 4))[0] / _arl_fractions(F(1, 4), F(1, 4))[0] > 3       # u um die Hälfte kleiner: mehr als dreifach (88 / 24)
