"""Orakel für den einseitigen CUSUM mit geschätzten Parametern (aps_cusum.py): Integralgleichung (Nyström) gegen das Brook/Evans-Gitter, die Simulation und die Summe der Überlebenswahrscheinlichkeiten
einer Markov-Kette (ORACLE-PLAYBOOK §2: Warteschlangen und Ketten gegen Simulation im Band). Die Simulation nutzt den SplitMix64-Strom der Demo (aps_scenario.normals)."""

import numpy as np
import pytest
from scipy.special import ndtr

import aps_constants as C
import aps_cusum as CU
import aps_scenario as S

H = 4.0965


def _fixed_samples(count=4, m=30):
    """Die ersten `count` Phase-I-Stichproben (Zufalls-Seed 1, 2, …) mit bedingter ARL zwischen 15 und 400 (Rohnormalzahlen)."""
    out = []
    seed = 0
    while len(out) < count:
        seed += 1
        x = S.phase1_sample(seed, m)
        xb, s = float(x.mean()), float(x.std(ddof=1))
        arl = float(CU.cusum_arl_integral(-xb / s, 1.0 / s, 0.5, H, 64)[0])
        if 15.0 < arl < 400.0:
            out.append((seed, xb, s, arl))
    return out


def test_h_for_the_known_card_is_4_0965_for_370_398_and_4_0954_for_370():
    assert CU.solve_h(0.5) == pytest.approx(4.0965, abs=5e-5)
    assert CU.solve_h(0.5, target=370.0) == pytest.approx(4.0954, abs=5e-5)
    assert CU.arl_known(0.5, 4.0965) == pytest.approx(C.TARGET_ARL0, abs=0.05)
    assert CU.shewhart_one_sided_limit() == pytest.approx(2.7822, abs=5e-5)
    assert float(CU.shewhart_one_sided_arl(0.0, 1.0, CU.shewhart_one_sided_limit())) == pytest.approx(C.TARGET_ARL0, rel=1e-12)


def test_integral_equation_equals_the_brook_evans_grid_for_four_fixed_samples():
    for seed, xb, s, arl in _fixed_samples():
        mu_z, sd_z = -xb / s, 1.0 / s
        grid = CU.chain_arl(CU.cusum_chain(lambda v: ndtr((v - mu_z) / sd_z), 0.5, H, 400))
        assert arl == pytest.approx(grid, rel=1e-4), seed                 # zwei exakte Wege (Vormessung: 2e-5)
        assert float(CU.cusum_arl_integral(mu_z, sd_z, 0.5, H, 40)[0]) == pytest.approx(arl, rel=1e-4), seed     # Knotenzahl der App gegen 64


def test_known_parameters_grid_equals_integral_equation():
    cdf = lambda v: ndtr(v)
    assert CU.chain_arl(CU.cusum_chain(cdf, 0.5, H, 400)) == pytest.approx(CU.arl_known(0.5, H), rel=1e-4)


def test_simulation_with_the_demo_stream_is_in_the_band():
    for seed, xb, s, arl in _fixed_samples(2):
        sim_seed = 4000 + seed                      # eigener Strom, nicht der der Stichprobe
        runs = 4000
        mean, se, cens = CU.simulate_cusum(lambda step: S.normals(sim_seed, (step - 1) * runs, runs), xb, s, 0.5, H, runs, 6000)
        assert cens == 0
        assert abs(mean - arl) <= 4 * se, (seed, mean, arl, se)


def test_run_length_distribution_of_the_chain_against_simulation():
    """P(Lauflänge > n) = e₀ᵀQⁿ1 gegen den simulierten Anteil (bekannte Parameter, 4000 Läufe)."""
    Q = CU.cusum_chain(lambda v: ndtr(v), 0.5, H, 200)
    surv = CU.chain_survival(Q, 400)
    runs = 4000
    C_t = np.zeros(runs)
    length = np.full(runs, 10 ** 6)
    active = np.ones(runs, dtype=bool)
    for step in range(1, 401):
        C_t = np.where(active, np.maximum(0.0, C_t + S.normals(777, (step - 1) * runs, runs) - 0.5), C_t)
        fired = active & (C_t > H)
        length[fired] = step
        active &= ~fired
    for n in (50, 100, 200, 400):
        share = float((length > n).mean())
        assert abs(share - surv[n]) <= 4 * np.sqrt(surv[n] * (1 - surv[n]) / runs), n
    assert surv.sum() < CU.chain_arl(Q) < surv.sum() + 400                  # die Summe bis 400 ist ein Teil des Mittels (der Rest ist der Tail)


def test_conditional_arl_of_the_one_sided_shewhart_card_against_simulation():
    rng = np.random.default_rng(55)
    xb, s, c = 0.2, 0.9, CU.shewhart_one_sided_limit()
    exact = float(CU.shewhart_one_sided_arl(xb, s, c))
    runs = 20000
    length = np.zeros(runs)
    active = np.ones(runs, dtype=bool)
    for step in range(1, 4000):
        hit = active & (rng.standard_normal(runs) > xb + c * s)
        length[hit] = step
        active &= ~hit
        if not active.any():
            break
    assert abs(length.mean() - exact) <= 4 * length.std(ddof=1) / np.sqrt(runs)
