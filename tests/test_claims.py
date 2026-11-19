"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App. Exakte Werte (ohne Zufall) auf die genannte Stelle,
Simulationszahlen als Band mit zwei Zufalls-Seeds (35 und 2026).

Nicht im Test, sondern Messungen dieser README: die Live-Rechenzeiten; die Aussagen zur Vormessung („1593 und 11.253“, Vormessung-Zufalls-Seeds); Tabelle I (eingefroren, Vormessung V13b, nur auf Aufbau und
Monotonie geprüft, die Zahlen selbst stehen als Quelle in der Datei); die Dauer des Neurechnens der Tabellen. Jede Testfunktion prüft auch, dass die genannte Zahl in der README steht."""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import aps_constants as C
import aps_cusum as CU
import aps_evaluation as ev
import aps_exact as E
import aps_reference as R
import aps_scenario as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
TARGET = C.TARGET_ARL0
HALF = 0.5 * TARGET


def _has(*phrases):
    for ph in phrases:
        assert ph in README, f"in der README fehlt: {ph}"


def _dec(x, digits=1):
    return ev.de(x, digits)


# --- exakte Verteilung ----------------------------------------------------------------------------------------------------------------------------------

def test_exact_distribution_m100():
    assert E.quantile_carl(0.5, 100) == pytest.approx(342.0, abs=0.05) and E.quantile_carl(0.05, 100) == pytest.approx(117.9, abs=0.05) and E.quantile_carl(0.95, 100) == pytest.approx(1162.5, abs=0.05)
    assert E.cdf_carl(HALF, 100) == pytest.approx(0.179, abs=5e-4) and HALF == pytest.approx(185.2, abs=0.05) and TARGET == pytest.approx(370.398, abs=5e-4)
    _has("**342,0**", "**117,9**", "**1162,5**", "**17,9 %**", "**185,2**", "**370,398**")


def test_four_measures_m100():
    assert E.mean_carl(100) == pytest.approx(455.9, abs=0.05) and E.inverse_mean_alarm_rate(100) == pytest.approx(280.0, abs=0.05)
    assert R.unconditional(100)["median"] == pytest.approx(229.2, abs=0.05)
    assert E.mean_carl(100) > E.inverse_mean_alarm_rate(100)
    _has("**455,9**", "**280,0**", "**342,0**", "**229,2**")


def test_four_measures_m30_and_m91():
    assert E.mean_carl(30) == pytest.approx(885.8, abs=0.05) and E.inverse_mean_alarm_rate(30) == pytest.approx(161.1, abs=0.05) and E.quantile_carl(0.5, 30) == pytest.approx(287.5, abs=0.05)
    assert R.unconditional(30)["median"] == pytest.approx(186.7, abs=0.05) and E.cdf_carl(HALF, 30) == pytest.approx(0.360, abs=5e-4)
    assert E.quantile_carl(0.5, 91) == pytest.approx(339.4, abs=0.05) and E.cdf_carl(HALF, 91) == pytest.approx(0.194, abs=5e-4)
    _has("885,8 / 161,1 / 287,5 / 186,7", "**36,0 %**", "Median 339,4", "**19,4 %**")


def test_mean_infinite_below_11():
    assert E.mean_carl_is_finite(11) and not E.mean_carl_is_finite(10)
    assert R.mean_carl(11) == pytest.approx(459653.3, abs=0.1) and E.mean_carl(12) == pytest.approx(52198.3, abs=0.1)
    # abgeschnittenes Integral bei m = 10: E[1/p; s <= S] auf einem feinen Gitter (S = 5 und 30)
    from test_handbeispiele import _truncated
    assert _truncated(10, 5.0) == pytest.approx(1.5e8, rel=0.02) and _truncated(10, 30.0) == pytest.approx(1.6e15, rel=0.03)
    _has("459.653,3", "52.198,3", "1,5·10⁸", "1,6·10¹⁵")


def test_decomposition_quantiles():
    assert E.quantile_carl(0.05, 30, mode="mu") == pytest.approx(221.6, abs=0.05) and E.quantile_carl(0.05, 30, mode="sigma") == pytest.approx(52.5, abs=0.05)
    assert E.quantile_carl(0.05, 100, mode="mu") == pytest.approx(310.5, abs=0.05) and E.quantile_carl(0.05, 100, mode="sigma") == pytest.approx(123.0, abs=0.05)
    _has("221,6 / 52,5", "310,5 / 123,0")


NINE = {(0.5, 0.10): 175, (0.5, 0.05): 274, (0.5, 0.01): 525, (0.75, 0.10): 986, (0.75, 0.05): 1588, (0.75, 0.01): 3119, (0.9, 0.10): 7241, (0.9, 0.05): 11828, (0.9, 0.01): 23504}


def test_minimum_m_table():
    for (f, eps), expected in NINE.items():
        assert E.min_m(f, eps) == expected, (f, eps)
    assert E.min_m(0.5, 0.05, mode="sigma") == 261 and E.min_m(0.5, 0.05, mode="mu") == 21 and round(261 / 21) == 12
    _has("**m = 274**", "**m = 1588**", "**m = 11.828**", "**261**", "**21**", "Faktor 12", "| ½ | 175 | **274** | 525 |", "| ¾ | 986 | **1588** | 3119 |", "| 0,9 | 7241 | **11.828** | 23.504 |")


def test_widening_table_in_readme():
    for r in ev.widen_rows():
        cells = [str(r["m"]), ev.de(r["L_a"], 4), ev.de(r["L_b"], 4), ev.de(r["L_c"], 4), ev.de(r["arl1_L3"], 2), ev.de(r["arl1_a"], 2), ev.de(r["arl1_b"], 2), ev.de(r["arl1_c"], 2),
                 ev.pct(r["share_3"]), ev.pct(r["share_b"]), ev.pct(r["share_c"])]
        assert "| " + " | ".join(cells) + " |" in README, cells
    _has("Faktor **3,0929**", "**2,9414**", "**3,1706**", "**66,7 statt 43,9**", "Ab m = 274 liegt L(c) unter 3", "m = 300: 2,9891")
    assert E.limit_guarantee(300) == pytest.approx(2.9891, abs=5e-5) and E.min_m(0.5, 0.05) == 274
    assert E.limit_guarantee(274) <= 3.0 < E.limit_guarantee(273)                  # ab m = 274 liegt L(c) unter 3


def test_calibrating_the_mean_makes_short_cards_more_frequent():
    rows = ev.widen_rows()
    assert len(rows) == 5 and all(r["share_b"] > r["share_3"] and r["median_b"] < r["median_3"] for r in rows)
    r = next(x for x in rows if x["m"] == 100)
    assert r["share_3"] == pytest.approx(0.179, abs=5e-4) and r["share_b"] == pytest.approx(0.257, abs=5e-4) and r["median_3"] == pytest.approx(342.0, abs=0.05) and r["median_b"] == pytest.approx(283.4, abs=0.05)
    _has("**25,7 %**", "von 17,9 % auf", "von 342,0 auf 283,4", "in allen fünf gemessenen Zeilen m = 30 bis 1000")


def test_widen_preset_numbers():
    assert C.PRESETS["Grenzen weiten (m = 100)"]["L"] == 3.18
    assert E.cdf_carl(HALF, 100, 3.18) == pytest.approx(0.046, abs=5e-4) and E.min_m(0.5, 0.05, L=3.18) == 97
    assert E.cdf_carl(HALF, 100, 3.17) == pytest.approx(0.0503, abs=5e-5) and E.min_m(0.5, 0.05, L=3.17) == 101
    _has("Faktor 3,18: 4,6 % unter 185,2, kleinstes m = 97", "Faktor 3,17: 5,03 % (m = 101)")


# --- Simulationszahlen mit zwei Zufalls-Seeds ------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _coupling(seed, m):
    xb, s = S.phase1_direct(seed, m, 10000)
    a0, a1 = 1 / E.p_alarm(xb, s), 1 / E.p_alarm(xb, s, mu=1.0)
    from scipy.stats import spearmanr
    a1c = np.array([E.arl1_known(float(l), 1.0) for l in E.effective_limit(1 / a0)])
    return float(spearmanr(a0, a1)[0]), np.percentile(a1 / a1c, [5, 50, 95]), float(np.median(a1))


def test_coupling_band():
    expected = {30: (0.67, 2.78, 40.3), 100: (0.75, 1.64, 42.7), 300: (0.83, 1.30, 43.5)}
    for seed in (35, 2026):
        for m, (lo, hi, med1) in expected.items():
            rho, (q05, q50, q95), median_arl1 = _coupling(seed, m)
            assert 0.885 <= rho <= 0.905, (seed, m)
            assert q05 == pytest.approx(lo, abs=0.015) and q95 == pytest.approx(hi, abs=0.03) and 0.985 <= q50 <= 1.002, (seed, m)
            assert median_arl1 == pytest.approx(med1, abs=0.6) and median_arl1 < E.arl1_known(3.0, 1.0)
    _has("Rangkorrelation 0,89 bis 0,90", "Median 0,99 bis 1,00", "m = 30: 0,67 bis 2,78", "m = 100: 0,75 bis 1,64", "m = 300: 0,83 bis 1,30")


@lru_cache(maxsize=None)
def _cusum(seed, m):
    return ev.analyse_cusum(ev.Settings(m=m, seed=seed, n_stich=6000))


def test_cusum_against_shewhart_band():
    expected = {30: (0.355, 0.313), 100: (0.211, 0.156), 300: (0.072, 0.032), 1000: (0.004, 0.0003)}
    for seed in (35, 2026):
        for m, (c_share, w_share) in expected.items():
            st_ = _cusum(seed, m).stats
            tol_c = 4 * math.sqrt(c_share * (1 - c_share) / 6000) + 1e-3
            tol_w = 4 * math.sqrt(w_share * (1 - w_share) / 6000) + 1e-3
            assert abs(st_["cusum"][2] - c_share) <= tol_c and abs(st_["shewhart"][2] - w_share) <= tol_w, (seed, m, st_)
            assert st_["cusum"][2] > st_["shewhart"][2], (seed, m)                         # in allen vier Zeilen und beiden Zufalls-Seeds ist der CUSUM stärker betroffen
    _has("35,5 / 31,3 %", "21,1 / 15,6 %", "7,2 / 3,2 %", "0,4 / 0,03 %", "**21,1 %**", "**15,6 %**", "ist der CUSUM stärker betroffen")
    m100 = _cusum(35, 100).stats
    assert m100["cusum"][1] == pytest.approx(102.4, abs=4.0) and m100["shewhart"][1] == pytest.approx(127.6, abs=4.0)
    assert _cusum(35, 100).stats["cusum"][0] == pytest.approx(349.2, abs=0.05) and _cusum(2026, 100).stats["cusum"][0] == pytest.approx(360.4, abs=0.05)
    _has("349,2 (Zufalls-Seed 35) und 360,4 (Zufalls-Seed 2026) gegen 357,5")


def test_cusum_threshold_and_check():
    assert CU.solve_h(0.5) == pytest.approx(4.0965, abs=5e-5) and CU.solve_h(0.5, 370.0) == pytest.approx(4.0954, abs=5e-5)
    ck = ev.cusum_check(ev.Settings(m=30, seed=35, n_stich=1000))
    assert ck["integral"] == pytest.approx(ck["grid"], rel=1e-4) and abs(ck["sim_mean"] - ck["integral"]) <= 4 * ck["sim_se"] and ck["sim_censored"] == 0
    assert 15.0 < ck["integral"] < 400.0
    _has("h = 4,0965", "4,0954")


@lru_cache(maxsize=None)
def _estimators(seed, m):
    X = S.phase1_raw_batch(seed, m, 40000)
    out = {}
    for est in C.ESTIMATORS:
        xb, sg = S.estimate(X, est)
        out[est] = (float((1 / E.p_alarm(xb, sg) < HALF).mean()), float(sg.std() * math.sqrt(2 * (m - 1))))
    return out


def test_estimators_s_against_mr():
    expected = {30: (0.357, 0.378), 100: (0.180, 0.232), 300: (0.042, 0.085)}
    for seed in (35, 2026):
        for m, (s_share, mr_share) in expected.items():
            r = _estimators(seed, m)
            assert abs(r["s"][0] - s_share) <= 4 * math.sqrt(s_share * (1 - s_share) / 40000) + 1e-3, (seed, m)
            assert abs(r["MR"][0] - mr_share) <= 4 * math.sqrt(mr_share * (1 - mr_share) / 40000) + 1e-3, (seed, m)
            assert r["MR"][0] > r["s"][0]                                                  # MR ist hier der schlechtere Schätzer für die Streuung der bedingten ARL₀
            assert 0.975 <= r["s"][1] <= 1.02 and 1.26 <= r["MR"][1] <= 1.30
            assert E.cdf_carl(HALF, m) == pytest.approx(r["s"][0], abs=4 * math.sqrt(0.25 / 40000))      # s gegen die exakte Verteilung
    _has("s 35,7 / 18,0 / 4,2 %, MR 37,8 / 23,2 / 8,5 %", "s 0,99 bis 1,00, MR 1,27 bis 1,28")


README_CONTAMINATION = {
    ("keine", 0.0): ((340.9, 42.9, 42.8), (346.9, 43.4, 43.3)),
    ("Ausreißer", 5.0): ((1121.8, 119.5, 91.3), (686.7, 82.2, 63.5)),
    ("Stufe", 1.0): ((439.3, 30.9, 436.5), (154.8, 15.0, 154.2)),
    ("Stufe", 2.0): ((1701.0, 80.7, 41716.4), (46.1, 6.5, 378.7)),
}


@lru_cache(maxsize=None)
def _contamination(seed):
    X = S.phase1_raw_batch(seed, 100, 40000)
    return {(kind, size): ev.contamination_row(X, kind, size) for kind, size in ev.CONTAMINATION_CASES}


def test_contamination_table_band():
    for seed in (35, 2026):
        rows = _contamination(seed)
        for key, (s_vals, mr_vals) in README_CONTAMINATION.items():
            for got, want in zip(rows[key]["s"] + rows[key]["MR"], s_vals + mr_vals):
                assert got == pytest.approx(want, rel=0.03), (seed, key)
    for key, (s_vals, mr_vals) in README_CONTAMINATION.items():
        _has(" / ".join(ev.de(v, 1) for v in s_vals), " / ".join(ev.de(v, 1) for v in mr_vals))
    r0, ro, r2 = (_contamination(35)[k] for k in (("keine", 0.0), ("Ausreißer", 5.0), ("Stufe", 2.0)))
    assert ro["s"][0] > 3.2 * r0["s"][0] and ro["s"][1] > 2.7 * r0["s"][1]                  # „das 3,3fache“ und „das 2,8fache“ (Vormessung) gelten auch hier
    assert r2["s"][0] > 4 * TARGET and r2["MR"][0] < TARGET / 7 and r2["s"][0] > 2 * TARGET and r2["MR"][0] < 0.5 * TARGET
    _has("ARL₀ im Median 1122 statt 341", "119,5 statt 42,9", "ARL₀ 1701, MR: 46,1", "über das Vierfache", "unter einem Siebtel", "**Weder s noch MR bringt die ARL₀ in die Nähe des Ziels**")
    for seed in (35, 2026):
        base = _contamination(seed)[("keine", 0.0)]
        assert base["s"][0] == pytest.approx(340.9, rel=0.02) and base["MR"][0] == pytest.approx(345.0, rel=0.02)         # ohne Störung nahe an der exakten Verteilung


# --- Handbeispiel, Annahmen, Literatur ----------------------------------------------------------------------------------------------------------------

def test_hand_samples():
    rows = {r["name"][0]: r for r in ev.hand_analysis()}
    assert rows["A"]["arl0"] == pytest.approx(7.48, abs=5e-3) and rows["B"]["arl0"] == pytest.approx(5.51, abs=5e-3) and rows["C"]["arl0"] == pytest.approx(5.07e8, rel=2e-3)
    assert 1 / E.mean_alarm_rate(3) == pytest.approx(8.22, abs=5e-3) and E.cdf_carl(TARGET, 3, mode="sigma") == pytest.approx(0.632, abs=5e-4)
    _has("ARL₀ 7,48 / 5,51 / 5,07·10⁸", "1/E[p] = 8,22", "63,2 %")


def test_assumption_rows():
    ti = R.table_i()
    j100, j30 = ti["m"].index(100), ti["m"].index(30)
    assert ti["anteil_unter_halber_arl0"][3][j100] == 0.440 and ti["anteil_unter_halber_arl0"][0][j100] == 0.186
    assert ti["anteil_unter_halber_arl0"][3][j30] == 0.793 and ti["anteil_unter_halber_arl0"][0][j30] == 0.356
    assert [round(r[100], 1) for r in ev.t2_rows(ps=(2, 5, 10))] == [239.6, 154.7, 75.0]
    assert [round(E.t2_f_limit(p, 100), 2) for p in (2, 5, 10)] == [12.83, 20.77, 33.19]
    assert [round(E.t2_chi2_limit(p), 2) for p in (2, 5, 10)] == [11.83, 18.21, 26.90]
    _has("44,0 %", "18,6 %", "**239,6 / 154,7 / 75,0**", "12,83 / 20,77 / 33,19")


def test_app_text_numbers_come_from_the_tables():
    """Die Zahlen der Annahmen-Tabelle in der App (aps_texts.ASSUMPTIONS_MD) stehen in der eingefrorenen Tabelle I."""
    import aps_texts as T
    ti = R.table_i()
    j = ti["m"].index(100)
    assert ev.de(100 * ti["anteil_unter_halber_arl0"][3][j], 1) + " %" in T.ASSUMPTIONS_MD and ev.de(100 * ti["anteil_unter_halber_arl0"][0][j], 1) + " %" in T.ASSUMPTIONS_MD


def test_the_numbers_in_the_app_texts_of_the_formula_block_match_the_exact_values():
    import aps_texts as T
    assert "370{,}398" in T.FORMULAS_MD and "185{,}2" in T.FORMULAS_MD and C.TARGET_ARL0 == pytest.approx(370.398, abs=5e-4) and HALF == pytest.approx(185.2, abs=0.05)
    assert "m \\ge 11" in T.FORMULAS_MD and E.mean_carl_is_finite(11) and not E.mean_carl_is_finite(10)
    import re
    assert not re.search(r"(?<!\\)%", T.FORMULAS_MD), "ungeschütztes % im Formelblock (KaTeX liest es als Kommentar)"
