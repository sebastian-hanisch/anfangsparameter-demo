"""Orakel-Tests für die exakte Verteilung der bedingten ARL₀ (aps_exact.py): jeder Wert gegen einen ANDEREN Rechenweg (ORACLE-PLAYBOOK §2, §5, §6).

Wege: (a) mittlere Alarmrate geschlossen (t-Formel) gegen ein eigenes Doppelintegral; (b) die Verteilungsfunktion `cdf_carl` gegen die Simulation aus ROHNORMALZAHLEN (numpy) und gegen den zweiten
exakten Weg (Integral über s mit Wurzelsubstitution am Knick s₀); (c) Zerlegung (nur σ, nur μ) gegen numerische Integration mit Indikator; (d) Mindestzahl: P(m − 1) > ε ≥ P(m) mit beiden exakten Wegen
und 10⁷ direkt gezogene (x̄, s) bei m = 273 und 274; (e) Lauflängenverteilung gegen die Simulation des Alarmprozesses; (f) unbedingte Lauflänge gegen die Simulation; (g) T² gegen Wishart-Simulation.
Die Referenz im Test ist numpy, nicht der SplitMix-Strom der Demo (2·10⁵ Stichproben × m = 100 sind 2·10⁷ Normalzahlen)."""

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import gammaln, ndtr, ndtri
from scipy.stats import chi2

import aps_constants as C
import aps_exact as E

TARGET = C.TARGET_ARL0


def _grid(lo, hi, panels, order):
    x, w = np.polynomial.legendre.leggauss(order)
    e = np.linspace(lo, hi, panels + 1)
    return (((e[1:] + e[:-1]) / 2)[:, None] + ((e[1:] - e[:-1]) / 2)[:, None] * x[None, :]).ravel(), (((e[1:] - e[:-1]) / 2)[:, None] * w[None, :]).ravel()


def _mean_alarm_rate_by_double_integral(m, L=3.0):
    """E[p] über die gemeinsame Dichte von (x̄, s): eigenes Gauss-Legendre-Gitter (unabhängig vom t-Formel-Weg)."""
    nu = m - 1.0
    hi = min(14.0, 1.0 + 16.0 / math.sqrt(nu)) if m > 3 else 14.0
    s, ws = _grid(0.0, hi, 300, 12)
    x, wx = _grid(-9.0 / math.sqrt(m), 9.0 / math.sqrt(m), 80, 8)
    lfs = math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * np.log(s) - 0.5 * nu * s * s
    fx = np.sqrt(m / (2 * np.pi)) * np.exp(-0.5 * m * x * x)
    p = ndtr(x[None, :] - L * s[:, None]) + ndtr(-x[None, :] - L * s[:, None])
    return float((ws[:, None] * np.exp(lfs)[:, None] * wx[None, :] * fx[None, :] * p).sum())


@pytest.mark.parametrize("m", [3, 4, 10, 30, 100, 1000])
def test_mean_alarm_rate_t_formula_equals_the_double_integral(m):
    assert E.mean_alarm_rate(m) == pytest.approx(_mean_alarm_rate_by_double_integral(m), rel=1e-7)


def _cdf_via_s_integral(a, m, L=3.0):
    """Zweiter exakter Weg: Integral über s (χ-Dichte) mit der Innenwahrscheinlichkeit 2Φ(−√m·g(s)); bis s₀ ist sie 1, darüber fällt g wie √(s − s₀) (Wurzelknick): Zerlegung bei s₀
    und Substitution s = s₀ + u²."""
    p0 = 1.0 / a
    nu = m - 1.0
    s0 = -ndtri(p0 / 2.0) / L
    sd = 1.0 / math.sqrt(2.0 * nu)
    lo_s, hi_s = max(1e-6, 1.0 - 14 * sd), 1.0 + 14 * sd

    def lfs(s):
        return math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * np.log(s) - 0.5 * nu * s * s

    def inner(s):
        lo, hi = np.zeros_like(s), L * s + 12.0
        for _ in range(100):
            md = 0.5 * (lo + hi)
            big = ndtr(md - L * s) + ndtr(-md - L * s) > p0
            hi, lo = np.where(big, md, hi), np.where(big, lo, md)
        g = np.where(2 * ndtr(-L * s) >= p0, 0.0, 0.5 * (lo + hi))
        return 2.0 * ndtr(-math.sqrt(m) * g)

    total = 0.0
    segs_lin = [(lo_s, min(s0, hi_s))] if s0 > lo_s else []
    segs_sq = [(max(s0, lo_s), hi_s)] if s0 < hi_s else []
    for a_, b_ in segs_lin:
        s, w = _grid(a_, b_, 60, 16)
        total += (w * np.exp(lfs(s)) * inner(s)).sum()
    for a_, b_ in segs_sq:
        u, w = _grid(0.0, math.sqrt(b_ - a_), 80, 16)
        s = a_ + u * u
        total += (w * 2.0 * u * np.exp(lfs(s)) * inner(s)).sum()
    return float(total)


@pytest.mark.parametrize("m,f", [(30, 0.5), (100, 0.5), (500, 0.5), (1000, 0.75), (1588, 0.75), (11828, 0.9)])
def test_cdf_x_route_equals_the_s_route(m, f):
    assert E.cdf_carl(f * TARGET, m) == pytest.approx(_cdf_via_s_integral(f * TARGET, m), abs=1e-6)


def _simulate_shares(m, thresholds, rng, total=200_000, chunk=20_000, L=3.0):
    counts = {a: 0 for a in thresholds}
    meds = []
    for _ in range(total // chunk):
        X = rng.standard_normal((chunk, m))
        xb, s = X.mean(1), X.std(1, ddof=1)
        arl = 1.0 / (ndtr(xb - L * s) + ndtr(-xb - L * s))
        meds.append(arl)
        for a in thresholds:
            counts[a] += int((arl < a).sum())
    return counts, np.concatenate(meds)


@pytest.mark.parametrize("m", [30, 100, 500])
def test_cdf_against_simulation_from_raw_normals(m):
    rng = np.random.default_rng(4711)
    thr = (TARGET / 2, TARGET, 2 * TARGET)
    counts, arl = _simulate_shares(m, thr, rng)
    for a in thr:
        exact = E.cdf_carl(a, m)
        se = math.sqrt(exact * (1 - exact) / 200_000)
        assert abs(counts[a] / 200_000 - exact) <= 4 * se, (m, a)
    assert np.median(arl) == pytest.approx(E.quantile_carl(0.5, m), rel=0.01)


def test_decomposition_only_sigma_and_only_mu_against_numerical_integration_with_indicator():
    for m in (30, 100):
        nu = m - 1.0
        for a in (TARGET / 2, 250.0):
            # nur σ geschätzt (x̄ = 0): P(p > 1/a) = P(2Φ(−3 s) > 1/a), Integral über die χ-Dichte mit Indikator
            dens = lambda s: math.exp(math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * math.log(s) - 0.5 * nu * s * s)
            s_cut = brentq(lambda s: 2 * ndtr(-3 * s) - 1 / a, 1e-9, 5.0)
            by_quad, _ = integrate.quad(dens, 1e-9, s_cut, epsabs=1e-13, epsrel=1e-12, limit=200)
            assert E.cdf_carl(a, m, mode="sigma") == pytest.approx(by_quad, abs=1e-9)
            # nur μ geschätzt (s = 1): P(p > 1/a) = P(|x̄| > g), Integral über die Normaldichte von x̄
            g = brentq(lambda g_: ndtr(g_ - 3) + ndtr(-g_ - 3) - 1 / a, 0.0, 15.0, xtol=1e-14)
            by_quad_mu, _ = integrate.quad(lambda x: math.sqrt(m / (2 * math.pi)) * math.exp(-0.5 * m * x * x), g, 12.0, epsabs=1e-14, epsrel=1e-12)
            assert E.cdf_carl(a, m, mode="mu") == pytest.approx(2 * by_quad_mu, abs=1e-9)


def test_quantile_inverts_the_cdf_and_edges_are_defined():
    for m in (11, 30, 100, 1000):
        for q in (0.05, 0.5, 0.95):
            assert E.cdf_carl(E.quantile_carl(q, m), m) == pytest.approx(q, abs=1e-9)
    assert E.cdf_carl(1.0, 30) == 0.0 and E.cdf_carl(0.5, 30) == 0.0 and E.cdf_carl(math.inf, 30) == 1.0
    assert E.sf_carl(5.0, 30) + E.cdf_carl(5.0, 30) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        E.quantile_carl(1.0, 30)


NINE = {(0.5, 0.10): 175, (0.5, 0.05): 274, (0.5, 0.01): 525, (0.75, 0.10): 986, (0.75, 0.05): 1588, (0.75, 0.01): 3119, (0.9, 0.10): 7241, (0.9, 0.05): 11828, (0.9, 0.01): 23504}


@pytest.mark.parametrize("key", list(NINE))
def test_minimum_m_boundary_with_both_exact_routes(key):
    f, eps = key
    mm = E.min_m(f, eps)
    assert mm == NINE[key]
    for route in (lambda a, m: E.cdf_carl(a, m), _cdf_via_s_integral):
        assert route(f * TARGET, mm) <= eps < route(f * TARGET, mm - 1)


def test_minimum_m_second_zufalls_seed_simulation_of_the_boundary_with_10_million_direct_draws():
    """m = 273 und 274 liegen 3,4·10⁻⁴ auseinander; mit 10⁷ direkt gezogenen (x̄, s) ist der Standardfehler 7·10⁻⁵ (2·10⁵ Stichproben mit 5·10⁻⁴ trennten sie nicht)."""
    rng = np.random.default_rng(20261118)
    for m in (273, 274):
        cnt, tot = 0, 0
        for _ in range(10):
            n = 1_000_000
            xb = rng.standard_normal(n) / math.sqrt(m)
            s = np.sqrt(rng.chisquare(m - 1, n) / (m - 1))
            cnt += int((1.0 / (ndtr(xb - 3 * s) + ndtr(-xb - 3 * s)) < 0.5 * TARGET).sum())
            tot += n
        exact = E.cdf_carl(0.5 * TARGET, m)
        assert abs(cnt / tot - exact) <= 4 * math.sqrt(exact * (1 - exact) / tot), m


def test_minimum_m_decomposition_sigma_261_mu_21():
    assert E.min_m(0.5, 0.05, mode="sigma") == 261 and E.min_m(0.5, 0.05, mode="mu") == 21
    assert E.min_m(0.5, 0.05, mode="sigma") > 10 * E.min_m(0.5, 0.05, mode="mu")


def test_run_length_of_a_fixed_card_is_geometric_with_p_by_simulating_the_alarm_process():
    """Phase II simuliert: Beobachtungen N(0, 1) bis |x − x̄| > 3 s; Mittel der Lauflängen gegen 1/p (feste Phase-I-Stichprobe, 3 Fälle)."""
    rng = np.random.default_rng(99)
    for xb, s in ((0.0, 1.0), (0.4, 0.7), (-0.3, 0.55)):
        p = float(E.p_alarm(xb, s))
        runs = 20_000
        length = np.zeros(runs, dtype=int)
        active = np.ones(runs, dtype=bool)
        step = 0
        while active.any() and step < 5000:
            step += 1
            hit = active & (np.abs(rng.standard_normal(runs) - xb) > 3 * s)
            length[hit] = step
            active &= ~hit
        assert not active.any()
        se = length.std(ddof=1) / math.sqrt(runs)
        assert abs(length.mean() - 1 / p) <= 4 * se, (xb, s)
        assert float((length <= 100).mean()) == pytest.approx(float(E.alarm_within(p, 100)), abs=4 * math.sqrt(0.25 / runs))


def test_unconditional_run_length_against_simulation_from_raw_normals():
    """P(Lauflänge ≤ 100) mit frischer Phase I je Lauf (m = 30): Mittel von 1 − (1 − p)¹⁰⁰ über 2·10⁵ Phase-I-Stichproben gegen die Doppelintegral-Tabelle (aps_exact.json)."""
    import aps_reference as R
    rng = np.random.default_rng(31337)
    vals = []
    for _ in range(10):
        X = rng.standard_normal((20_000, 30))
        xb, s = X.mean(1), X.std(1, ddof=1)
        vals.append(E.alarm_within(E.p_alarm(xb, s), 100))
    vals = np.concatenate(vals)
    assert abs(vals.mean() - R.unconditional(30)["p_alarm_100"]) <= 4 * vals.std(ddof=1) / math.sqrt(vals.size)


def test_t2_mean_alarm_rate_against_wishart_simulation_and_closed_forms():
    # p = 1: die t-Formel
    for m in (5, 30, 100):
        assert E.t2_mean_alarm_rate(1, m, 9.0) == pytest.approx(E.mean_alarm_rate(m), rel=1e-12)
    # p = 2: geschlossen (1 + 2y/ν)^(−ν/2), ν = m − 2, y = UCL / Skala
    for m in (10, 30, 100):
        nu = m - 2.0
        y = 11.829158 * m * (m - 2) / (2 * (m + 1) * (m - 1))
        assert E.t2_mean_alarm_rate(2, m, 11.829158) == pytest.approx((1 + 2 * y / nu) ** (-nu / 2), rel=1e-7)
    # Simulation: Phase I aus m = 30 Vektoren N(0, I_p), Phase II ein neuer Vektor; Anteil T² > UCL gegen die F-Formel
    rng = np.random.default_rng(777)
    for p in (2, 5):
        m, R = 30, 120_000
        ucl = E.t2_chi2_limit(p)
        X = rng.standard_normal((R, m, p))
        xb = X.mean(1)
        D = X - xb[:, None, :]
        S = np.einsum("rmi,rmj->rij", D, D) / (m - 1)
        new = rng.standard_normal((R, p)) - xb
        t2 = np.einsum("ri,ri->r", new, np.linalg.solve(S, new[..., None])[..., 0])
        share = float((t2 > ucl).mean())
        exact = E.t2_mean_alarm_rate(p, m, ucl)
        assert abs(share - exact) <= 4 * math.sqrt(exact * (1 - exact) / R), p
    # die F-Regel trifft das Ziel genau
    for p in (2, 5, 10):
        for m in (30, 100):
            assert 1.0 / E.t2_mean_alarm_rate(p, m, E.t2_f_limit(p, m)) == pytest.approx(TARGET, rel=1e-10)
    assert E.t2_chi2_limit(1) == pytest.approx(chi2.isf(C.P_NOMINAL, 1)) and E.t2_chi2_limit(1) == pytest.approx(9.0, abs=1e-9)
    assert math.sqrt(E.t2_f_limit(1, 100)) == pytest.approx(E.limit_equal_alarm_rate(100), rel=1e-10) and E.limit_equal_alarm_rate(100) == pytest.approx(3.0929, abs=5e-5)
