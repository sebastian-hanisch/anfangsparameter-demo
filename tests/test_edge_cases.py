"""Randfälle, an denen die Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): m = 0, 1, 2, das Mittel der bedingten ARL₀ bei m ≤ 10, alle Phase-I-Werte gleich (s = 0),
sehr großes m, Grenzfaktor unter 1, Zufalls-Seed 0, Schätzer MR bei m = 2, unerreichbare Mindestzahl."""

import math

import numpy as np
import pytest

import aps_constants as C
import aps_cusum as CU
import aps_evaluation as ev
import aps_exact as E
import aps_scenario as S

TARGET = C.TARGET_ARL0


def test_m_zero_and_one_have_no_standard_deviation():
    for bad in (0, 1):
        with pytest.raises(ValueError, match="mindestens"):
            E.cdf_carl(TARGET, bad)
        with pytest.raises(ValueError):
            E.mean_alarm_rate(bad)
        with pytest.raises(ValueError):
            S.phase1_direct(1, bad, 10)
    with pytest.raises(ValueError, match="nicht definiert"):
        ev.analyse_values([1.0])
    with pytest.raises(ValueError, match="nicht definiert"):
        ev.analyse_values([])
    with pytest.raises(ValueError):
        S.estimate_s(np.array([1.0]))
    assert S.phase1_sample(1, 0).size == 0


def test_m_two_has_one_degree_of_freedom_and_the_mean_alarm_rate_is_cauchy():
    """t(1) ist die Cauchy-Verteilung: P(|t| > c) = 1 − (2/π)·arctan(c), c = 3/√(1 + 1/2)."""
    c = 3 / math.sqrt(1.5)
    assert E.mean_alarm_rate(2) == pytest.approx(1 - 2 / math.pi * math.atan(c), rel=1e-12)
    assert 0.0 < E.cdf_carl(0.5 * TARGET, 2) < 1.0
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400_000, 2))
    xb, s = X.mean(1), X.std(1, ddof=1)
    share = float((1 / E.p_alarm(xb, s) < 0.5 * TARGET).mean())
    assert abs(share - E.cdf_carl(0.5 * TARGET, 2)) < 4 * math.sqrt(0.25 / 400_000)


def test_mean_of_the_conditional_arl_is_infinite_up_to_m_10_and_the_text_says_so():
    for m in (3, 10):
        assert E.mean_carl(m) == math.inf
        sp = ev.analyse_spread(ev.Settings(m=m, n_stich=1000))
        assert sp.mean_carl == math.inf and sp.mean_source == "Herleitung"
        kind, text = ev.verdict_spread(sp)
        assert "ist unendlich" in text and "Mittel der bedingten ARL₀" in text
    assert math.isfinite(E.mean_carl(11))


def test_all_phase_i_values_equal_collapses_the_limits():
    a = ev.analyse_values([2.0, 2.0, 2.0, 2.0])
    assert a.sigma == 0.0 and a.lower == a.upper == 2.0 and a.arl0 == 1.0 and a.median == a.q05 == a.q95 == 1.0 and a.p100 == pytest.approx(1.0)
    kind, text = ev.verdict_sample(a)
    assert kind == "warning" and "Alle Phase-I-Werte sind gleich" in text and "ARL₀ = 1" in text


def test_card_that_never_alarms_is_infinite_not_a_crash():
    a = ev.analyse_values([-100.0, 100.0])        # s = 141: die Grenzen liegen bei ±424, p unterläuft zu 0
    assert a.p == 0.0 and a.arl0 == math.inf and a.median == math.inf
    kind, text = ev.verdict_sample(a)
    assert kind == "info" and "unendlich" in text


def test_very_large_m_has_a_vanishing_share_below_half_the_target():
    assert E.cdf_carl(0.5 * TARGET, 5000) < 5e-5            # aus der exakten Verteilung, nicht aus einem Anteil
    assert E.cdf_carl(0.5 * TARGET, 5000) > 0.0


def test_limit_factor_below_one_and_zero_still_work():
    for L in (0.5, 1.0):
        assert 0.0 <= E.cdf_carl(2.0, 30, L) <= 1.0
        assert E.nominal_arl(L) > 1.0
    assert float(E.p_alarm(0.3, 1.0, 0.0)) == pytest.approx(1.0, abs=1e-12)    # L = 0: jede abweichende Beobachtung alarmiert (die Regler beginnen bei 1,5)


def test_seed_zero_and_the_largest_seed():
    for seed in (0, C.SEED_MAX):
        a = ev.analyse_sample(ev.Settings(seed=seed))
        assert math.isfinite(a.arl0) and len(a.history) == C.HISTORY_LENGTH and a.history[-1][0] == seed
    assert [sd for sd, _ in ev.analyse_sample(ev.Settings(seed=3)).history][-1] == 3
    assert ev.history_seeds(3)[0] == (3 - 19) % (C.SEED_MAX + 1)           # zyklisch, nie negativ


def test_mr_estimator_with_two_observations():
    assert float(S.estimate_mr(np.array([1.0, 4.0]))) == pytest.approx(3.0 / S.D2, rel=1e-15)
    a = ev.analyse_values([1.0, 4.0], "MR")
    assert a.sigma == pytest.approx(3.0 / S.D2, rel=1e-15) and math.isfinite(a.arl0)
    with pytest.raises(ValueError):
        S.estimate_mr(np.array([1.0]))
    with pytest.raises(ValueError, match="unbekannt"):
        S.estimate(np.array([1.0, 2.0]), "Median")


def test_unreachable_minimum_returns_none_and_the_verdict_says_so():
    assert E.min_m(0.5, 0.0, hi=1000) is None
    mn = ev.analyse_minimum(ev.Settings(f="¾", eps="1 %", zerl="nur μ geschätzt"))
    assert mn.m_min is not None and mn.m_min < 3000
    kind, text = ev.verdict_minimum(ev.MinimumAnalysis(mn.settings, None, None, None, None, None, mn.m_grid, mn.curve, mn.threshold))
    assert kind == "warning" and "nicht erreichbar" in text


def test_contamination_input_validation():
    x = np.zeros((3, 10))
    with pytest.raises(ValueError, match="unbekannt"):
        S.contaminate(x, "Rampe", 1.0)
    cont = S.contaminate(x, "keine", 5.0)
    assert np.array_equal(cont.data, x) and cont.mu_phase2 == 0.0
    out = S.contaminate(x, "Ausreißer", 5.0)
    assert out.data[0, 5] == 5.0 and out.data.sum() == 15.0 and x.sum() == 0.0        # Ausgangsdaten bleiben unverändert
    step = S.contaminate(x, "Stufe", 2.0)
    assert step.data[0, :5].sum() == 0.0 and step.data[0, 5:].sum() == 10.0 and step.mu_phase2 == 2.0


def test_set_t2_needs_more_vectors_than_channels():
    with pytest.raises(ValueError):
        E.t2_mean_alarm_rate(5, 5, 20.0)
    with pytest.raises(ValueError):
        E.t2_mean_alarm_rate(0, 30, 5.0)


def test_estimated_cusum_with_extreme_sample_does_not_fail():
    """Sehr kleines s (aus m = 3): z = (x − x̄)/s ist breit; die Integralgleichung bleibt lösbar."""
    arl = CU.cusum_arl_integral(np.array([0.0, 3.0, -3.0]), np.array([20.0, 0.5, 0.5]), 0.5, 4.0965, 40)
    assert np.all(np.isfinite(arl)) and np.all(arl >= 1.0)
