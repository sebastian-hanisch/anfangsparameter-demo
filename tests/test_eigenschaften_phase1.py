"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Invarianzen und Monotonie über je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed.

Nicht als Eigenschaft behauptet: Monotonie der bedingten ARL₀ beim Vergrößern der Phase-I-Stichprobe durch Kopien derselben Werte (x̄ bleibt, s sinkt leicht, die bedingte ARL₀ kann in
beide Richtungen gehen)."""


import numpy as np
import pytest
from scipy.stats import spearmanr

import aps_constants as C
import aps_evaluation as ev
import aps_exact as E
import aps_scenario as S

TARGET = C.TARGET_ARL0


def _instances(count=40, m_range=(3, 60), seed=2026):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        m = int(rng.integers(*m_range))
        yield rng.standard_normal(m) * rng.uniform(0.3, 3.0) + rng.uniform(-2, 2)


def test_s_is_permutation_invariant_and_mr_is_not():
    rng = np.random.default_rng(5)
    mr_changed = 0
    for x in _instances():
        perm = rng.permutation(x)
        assert S.estimate_s(perm) == pytest.approx(S.estimate_s(x), rel=1e-12)
        assert S.estimate_mr(perm) >= 0
        mr_changed += not np.isclose(S.estimate_mr(perm), S.estimate_mr(x), rtol=1e-9)
    assert mr_changed >= 30                                  # MR hängt an der Reihenfolge (in fast jeder Instanz)


def test_affine_transformation_of_data_and_true_series_leaves_the_conditional_arl_unchanged():
    rng = np.random.default_rng(6)
    for x in _instances():
        a, b = float(rng.uniform(0.2, 5.0)), float(rng.uniform(-10, 10))
        mu = float(rng.uniform(-1, 1))
        xb, s = float(x.mean()), float(x.std(ddof=1))
        base = E.p_alarm(xb, s, 3.0, mu=mu)
        moved = E.p_alarm(a * xb + b, a * s, 3.0, mu=a * mu + b, sigma=a)        # Daten und wahre Reihe gleich umgerechnet
        assert moved == pytest.approx(base, rel=1e-9, abs=1e-300)


def test_mirroring_all_data_leaves_the_two_sided_conditional_arl_unchanged():
    for x in _instances():
        a = ev.analyse_values(x)
        b = ev.analyse_values(-x)
        assert b.arl0 == pytest.approx(a.arl0, rel=1e-12) and b.xb == -a.xb and b.sigma == pytest.approx(a.sigma, rel=1e-15)


def test_conditional_arl_grows_with_the_limit_factor_for_a_fixed_sample():
    for x in _instances(20):
        arls = [ev.analyse_values(x, L=L).arl0 for L in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)]
        assert arls == sorted(arls)
    assert [E.nominal_arl(L) for L in (2.0, 3.0, 4.0)] == sorted(E.nominal_arl(L) for L in (2.0, 3.0, 4.0))


def test_distribution_function_falls_in_m_and_in_the_limit_factor():
    grid = [11, 15, 20, 30, 50, 91, 100, 200, 300, 500, 1000, 5000]
    for f in (0.5, 0.75):
        vals = [E.cdf_carl(f * TARGET, m) for m in grid]
        assert vals == sorted(vals, reverse=True)                              # mehr Phase I, weniger kurze Karten
    for m in (30, 100, 300):
        by_L = [E.cdf_carl(0.5 * TARGET, m, L) for L in (2.6, 2.8, 3.0, 3.2, 3.4)]
        assert by_L == sorted(by_L, reverse=True)                              # weitere Grenzen, weniger kurze Karten
    for m in (30, 100):
        by_a = [E.cdf_carl(a, m) for a in (10, 50, 100, 200, 400, 800, 1600)]
        assert by_a == sorted(by_a) and 0.0 <= by_a[0] and by_a[-1] <= 1.0     # Verteilungsfunktion


def test_quantiles_are_ordered_and_mean_exceeds_inverse_mean_alarm_rate():
    for m in (15, 30, 100, 1000):
        q = [E.quantile_carl(p, m) for p in (0.05, 0.5, 0.95)]
        assert q == sorted(q)
        assert E.mean_carl(m) >= E.inverse_mean_alarm_rate(m)                  # Jensen
    inv = [E.inverse_mean_alarm_rate(m) for m in (3, 10, 30, 100, 300, 1000, 5000, 100000)]
    assert inv == sorted(inv) and all(v < TARGET for v in inv) and inv[-1] == pytest.approx(TARGET, rel=1e-3)     # wächst in m und nähert sich 370,40


def test_minimum_m_grows_when_the_criterion_gets_stricter():
    assert E.min_m(0.5, 0.10) < E.min_m(0.5, 0.05) < E.min_m(0.5, 0.01)
    assert E.min_m(0.5, 0.05) < E.min_m(0.75, 0.05) < E.min_m(0.9, 0.05)
    assert E.min_m(0.5, 0.05, L=3.17) < E.min_m(0.5, 0.05) < E.min_m(0.5, 0.05, L=2.9)      # weitere Grenzen brauchen weniger Phase I


def test_rank_correlation_of_conditional_arl0_and_arl1_is_strongly_positive():
    for m in (30, 100, 300):
        for seed in (1, 2):
            xb, s = S.phase1_direct(seed, m, 3000)
            rho = spearmanr(1 / E.p_alarm(xb, s), 1 / E.p_alarm(xb, s, mu=1.0))[0]
            assert rho > 0.8, (m, seed, rho)


def test_duplicating_the_phase_i_sample_keeps_the_mean_and_shrinks_s():
    for x in _instances(30, (5, 40)):
        x2 = np.concatenate([x, x])
        assert x2.mean() == pytest.approx(x.mean(), rel=1e-12, abs=1e-12)
        m = x.size
        assert S.estimate_s(x2) ** 2 == pytest.approx(S.estimate_s(x) ** 2 * 2 * (m - 1) / (2 * m - 1), rel=1e-12)
        assert S.estimate_s(x2) < S.estimate_s(x)


def test_contamination_a_larger_outlier_widens_the_limits_monotonically():
    x = S.phase1_raw_batch(3, 100, 200)
    sizes = (0.0, 2.0, 5.0, 10.0)
    sig = [float(np.median(S.estimate(S.contaminate(x, "Ausreißer", d).data, "s")[1])) for d in sizes]
    assert sig == sorted(sig)
    mr = [float(np.median(S.estimate(S.contaminate(x, "Ausreißer", d).data, "MR")[1])) for d in sizes]
    assert (mr[-1] - mr[0]) < (sig[-1] - sig[0])           # der Ausreißer geht in s quadratisch, in MR linear ein


def test_verdicts_never_fail_over_random_settings():
    rng = np.random.default_rng(11)
    kinds = {"success", "info", "warning", "error"}
    for _ in range(5):
        s = ev.Settings(m=int(rng.choice(C.M_OPTIONS)), seed=int(rng.integers(0, 1000)), estimator=str(rng.choice(C.ESTIMATORS)), L=float(rng.uniform(1.5, 6.0)),
                        zerl=str(rng.choice(C.ZERLEGUNG)), f=str(rng.choice(C.F_OPTIONS)), eps=str(rng.choice(C.EPS_OPTIONS)), k=float(rng.uniform(0.25, 1.5)),
                        n_stich=1000, stoerung=str(rng.choice(C.STOERUNG)))
        for kind, text in (ev.verdict_sample(ev.analyse_sample(s)), ev.verdict_minimum(ev.analyse_minimum(s)), ev.verdict_cusum(ev.analyse_cusum(s)),
                           ev.verdict_contamination(ev.analyse_contamination(s)), ev.verdict_spread(ev.analyse_spread(s))):
            assert kind in kinds and isinstance(text, str) and text and "nan" not in text.lower(), (s, text)
