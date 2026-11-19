"""Orakel für die vier Maße (Mittel der bedingten ARL₀, Kehrwert der mittleren Alarmrate, unbedingte Lauflänge): die Integrale als gewichtete Summe über ein (x̄, s)-Gitter,
gerechnet mit `aps_exact.mixture_measures` (derselbe Ausdruck wie im Handbeispiel H 2 mit endlich vielen Ausgängen), gegen den Tail-Weg `aps_exact.mean_carl`, die t-Formel und die eingefrorene
Tabelle (Doppelintegral des Skripts tools/aps_make_exact.py)."""

import math

import numpy as np
import pytest
from scipy.special import gammaln, log_ndtr

import aps_exact as E
import aps_reference as R


def _grid(lo, hi, panels, order):
    x, w = np.polynomial.legendre.leggauss(order)
    e = np.linspace(lo, hi, panels + 1)
    return (((e[1:] + e[:-1]) / 2)[:, None] + ((e[1:] - e[:-1]) / 2)[:, None] * x[None, :]).ravel(), (((e[1:] - e[:-1]) / 2)[:, None] * w[None, :]).ravel()


def _mixture(m, s_hi=None):
    """Das (x̄, s)-Gitter als endlich viele Ausgänge: Alarmwahrscheinlichkeiten und Gewichte (Dichte × Quadraturgewicht)."""
    nu = m - 1.0
    hi = s_hi or min(12.0, 1.0 + 16.0 / math.sqrt(nu))
    s, ws = _grid(0.0, hi, max(80, int(hi / 0.02)), 12)
    x, wx = _grid(-9.0 / math.sqrt(m), 9.0 / math.sqrt(m), 40, 8)
    lfs = math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * np.log(s) - 0.5 * nu * s * s
    lfx = 0.5 * math.log(m / (2 * math.pi)) - 0.5 * m * x * x
    lp = np.logaddexp(log_ndtr(x[None, :] - 3 * s[:, None]), log_ndtr(-x[None, :] - 3 * s[:, None]))
    w = ws[:, None] * wx[None, :] * np.exp(lfs[:, None] + lfx[None, :])
    return np.exp(lp).ravel(), w.ravel()


@pytest.mark.parametrize("m", [30, 100])
def test_measures_as_weighted_sums_equal_the_closed_and_tail_routes(m):
    ps, w = _mixture(m)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    mm = E.mixture_measures(ps, w)
    assert mm["inv_mean_alarm"] == pytest.approx(E.inverse_mean_alarm_rate(m), rel=1e-7)                  # t-Formel
    assert mm["mean_carl"] == pytest.approx(E.mean_carl(m), rel=1e-6)                                      # Tail-Weg
    assert mm["mean_carl"] == pytest.approx(R.mean_carl(m), rel=1e-6)                                      # eingefrorene Tabelle
    assert mm["mean_carl"] >= mm["inv_mean_alarm"]                                                          # Jensen


def test_unconditional_survival_as_a_weighted_sum_equals_the_frozen_table():
    ps, w = _mixture(100)
    surv = E.mixture_measures(ps, w)["survival"]
    fr = R.unconditional(100)
    assert 1 - surv(100) == pytest.approx(fr["p_alarm_100"], rel=1e-6)
    # Median und Quantile als reelles n mit P(RL ≤ n) = q; die Summe läuft mit reellem n weiter
    for q, key in ((0.5, "median"), (0.05, "q05"), (0.95, "q95")):
        assert 1 - surv(fr[key]) == pytest.approx(q, abs=1e-5)


def test_mean_carl_tail_route_equals_the_frozen_double_integral_on_the_whole_table():
    for m in (15, 91, 274, 500):
        assert E.mean_carl(m) == pytest.approx(R.mean_carl(m), rel=1e-7), m


def test_mean_carl_is_defined_live_for_other_limits():
    """Auch für einen anderen Grenzfaktor (L = 3,17) gilt: endlich genau für m > L² + 1 = 11,05, also m ≥ 12; Tail-Weg gegen die gewichtete Summe."""
    assert E.mean_carl(11, 3.17) == math.inf and math.isfinite(E.mean_carl(12, 3.17))
    nu = 99.0
    s, ws = _grid(0.0, 2.6, 130, 12)
    x, wx = _grid(-0.9, 0.9, 40, 8)
    lfs = math.log(2) + 0.5 * nu * math.log(0.5 * nu) - gammaln(0.5 * nu) + (nu - 1) * np.log(s) - 0.5 * nu * s * s
    lfx = 0.5 * math.log(100 / (2 * math.pi)) - 50 * x * x
    lp = np.logaddexp(log_ndtr(x[None, :] - 3.17 * s[:, None]), log_ndtr(-x[None, :] - 3.17 * s[:, None]))
    direct = float((ws[:, None] * wx[None, :] * np.exp(lfs[:, None] + lfx[None, :] - lp)).sum())
    assert E.mean_carl(100, 3.17) == pytest.approx(direct, rel=1e-6)
