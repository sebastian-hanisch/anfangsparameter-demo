"""Zufall: der vektorisierte SplitMix64-Strom (aps_scenario.uniforms) ist bitgleich mit der Klasse aps_rng.SplitMix64; die Normalzahlen stimmen auf 1e-12; die direkt gezogenen (x̄, s)
haben dieselbe Verteilung wie Mittel und Standardabweichung aus Rohnormalzahlen."""

import math

import numpy as np
import pytest
from scipy.stats import kstest, norm

import aps_scenario as S
from aps_rng import SplitMix64


@pytest.mark.parametrize("seed", [0, 1, 35, 999999, 2 ** 63 + 5])
def test_vectorized_uniforms_are_bitwise_equal_to_the_class(seed):
    g = SplitMix64(seed)
    assert [g.random() for _ in range(500)] == list(S.uniforms(seed, 0, 500))


def test_stream_blocks_are_consistent():
    whole = S.uniforms(7, 0, 300)
    assert list(S.uniforms(7, 100, 50)) == list(whole[100:150])
    assert S.uniforms(7, 0, 0).size == 0


def test_normals_equal_the_class_to_1e_12():
    g = SplitMix64(123)
    ref = np.array([g.normal() for _ in range(300)])
    assert np.max(np.abs(ref - S.normals(123, 0, 300))) < 1e-12
    assert list(S.normals(123, 100, 20)) == list(S.normals(123, 0, 120)[100:])         # Blöcke des Stroms


def test_normals_are_standard_normal():
    z = S.normals(5, 0, 20000)
    assert kstest(z, "norm").pvalue > 0.001
    assert abs(z.mean()) < 4 / math.sqrt(z.size) and abs(z.std() - 1) < 4 / math.sqrt(2 * z.size)


def test_same_seed_same_sample_different_seed_different_sample():
    assert np.array_equal(S.phase1_sample(35, 100), S.phase1_sample(35, 100))
    assert not np.array_equal(S.phase1_sample(35, 100), S.phase1_sample(36, 100))
    assert S.phase1_sample(35, 0).size == 0
    with pytest.raises(ValueError):
        S.phase1_sample(35, -1)


def test_direct_draws_are_distributed_like_mean_and_std_of_raw_normals():
    """(x̄, s) direkt gezogen: x̄·√m ~ N(0, 1) und (m − 1)s² ~ χ²(m − 1), unabhängig. Gegen Rohnormalzahlen (numpy) mit Zwei-Stichproben-Test auf 30 000 Stichproben."""
    from scipy.stats import ks_2samp, spearmanr
    m, count = 30, 30000
    xb, s = S.phase1_direct(2026, m, count)
    X = np.random.default_rng(11).standard_normal((count, m))
    assert ks_2samp(xb, X.mean(1)).pvalue > 0.001 and ks_2samp(s, X.std(1, ddof=1)).pvalue > 0.001
    assert kstest(xb * math.sqrt(m), "norm").pvalue > 0.001
    assert abs(spearmanr(xb, s)[0]) < 4 / math.sqrt(count)                  # unabhängig (Satz von Cochran)
    assert np.all(np.isfinite(xb)) and np.all(s > 0)


def test_direct_draws_need_two_observations():
    with pytest.raises(ValueError):
        S.phase1_direct(1, 1, 10)


def test_open_unit_never_gives_0_or_1():
    u = S.open_unit(np.array([0.0, 1 - 2.0 ** -53]))
    assert 0.0 < u[0] < 1.0 and 0.0 < u[1] < 1.0 and np.isfinite(norm.ppf(u)).all()
