"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Die Verdikte sind reine Funktionen (Art, Text) und bekommen Zeilen, nicht Seeds."""

import math

import numpy as np
import pytest

import aps_constants as C
import aps_evaluation as ev

T = C.TARGET_ARL0


# --- Zahlenformate -----------------------------------------------------------------------------------------------------------------------------------

def test_german_number_formats():
    assert ev.de(3.14159, 2) == "3,14" and ev.de(-1.5) == "−1,5" and ev.de(-0.04, 1) == "0,0" and ev.de(0.0, 2) == "0,00"
    assert ev.de(12345.67, 1) == "12.345,7" and ev.de(9999.94, 1) == "9999,9" and ev.de(math.inf) == "unendlich"
    assert ev.pct(0.1792) == "17,9 %" and ev.pct(0.04983, 2) == "4,98 %"
    assert ev.sci(5.0679e8) == "5,07·10⁸" and ev.sci(7.4846) == "7,48" and ev.sci(999999.0) == "999.999,00" and ev.sci(1e6) == "1,00·10⁶"
    assert ev.interval(-1.5, 1.5, 1) == "[−1,5; 1,5]"


# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------------------

def test_verdict_hand_names_every_sample_and_the_orders_of_magnitude():
    rows = [{"name": "A (−½, 0, ½)", "lower": -1.5, "upper": 1.5, "arl0": 7.4846}, {"name": "B (0, ½, 1)", "lower": -1.0, "upper": 2.0, "arl0": 5.5126},
            {"name": "C (−2, 0, 2)", "lower": -6.0, "upper": 6.0, "arl0": 5.0679e8}]
    kind, text = ev.verdict_hand(rows)
    assert kind == "info"
    for part in ("A (−½, 0, ½): Grenzen [−1,5; 1,5], bedingte ARL₀ 7,48", "B (0, ½, 1): Grenzen [−1,0; 2,0], bedingte ARL₀ 5,51", "C (−2, 0, 2): Grenzen [−6,0; 6,0], bedingte ARL₀ 5,07·10⁸",
                 "ARL₀ von 5,51 bis 5,07·10⁸ (8 Größenordnungen)"):
        assert part in text


def _sample(arl0, sigma=1.0, estimator="s", m=100, seed=35):
    p = 1.0 / arl0
    return ev.SampleAnalysis(m=m, seed=seed, estimator=estimator, L=3.0, x=np.zeros(m), xb=0.0, sigma=sigma, lower=-3 * sigma, upper=3 * sigma, p=p, arl0=arl0, p100=0.031,
                             median=2191.8, q05=162.2, q95=9472.9, history=())


def test_verdict_sample_threshold_is_the_half_target():
    kind, text = ev.verdict_sample(_sample(0.5 * T - 0.01))
    assert kind == "warning" and "weniger als die Hälfte der nominellen ARL₀" in text
    assert ev.verdict_sample(_sample(0.5 * T))[0] == "info"
    kind, text = ev.verdict_sample(_sample(3162.6))
    assert kind == "info" and "m = 100, Zufalls-Seed 35" in text and "Bedingte ARL₀ 3162,6" in text and "[−3,00; 3,00]" in text
    assert "Median 2191,8, 5 % 162,2, 95 % 9472,9" in text and "in den ersten 100 Beobachtungen 3,1 %" in text and "Ziel 370,4" in text


def test_verdict_sample_names_the_estimator():
    assert "Stichproben-Standardabweichung s" in ev.verdict_sample(_sample(300.0))[1]
    assert "mittlere gleitende Spannweite (MR)" in ev.verdict_sample(_sample(300.0, estimator="MR"))[1]


# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------------

def _spread(share, m=100, zerl="beide geschätzt", mean_carl=455.9, uncond=None, eps="5 %", median=342.0, q05=117.9, q95=1162.5, inv=280.0):
    s = ev.Settings(m=m, zerl=zerl, eps=eps)
    z = np.zeros(2)
    return ev.SpreadAnalysis(settings=s, a_grid=z, cdf=z, threshold=185.2, share=share, median=median, q05=q05, q95=q95, mean_carl=mean_carl, mean_source="eingefroren",
                             inv_mean_alarm=inv, uncond=uncond, arl0_samples=np.ones(5), arl1_samples=np.ones(5), rho=0.89, ratio_q=(0.75, 0.99, 1.61), frontier=(z, z), arl1_known=43.9)


def test_verdict_spread_names_the_distribution_and_the_four_measures():
    kind, text = ev.verdict_spread(_spread(0.1792, uncond={"median": 229.2}))
    assert kind == "warning"
    for part in ("m = 100 Phase-I-Beobachtungen, beide geschätzt, Grenzfaktor 3,00", "Median der bedingten ARL₀ 342,0 (Ziel 370,4), 5 % 117,9, 95 % 1162,5", "17,9 % der Phase-I-Stichproben",
                 "weniger als ½ der Ziel-ARL₀ (185,2)", "Mittel der bedingten ARL₀ 455,9", "Kehrwert der mittleren Alarmrate 280,0", "unbedingte Lauflängenverteilung: Median 229,2"):
        assert part in text


def test_verdict_spread_threshold_follows_epsilon():
    assert ev.verdict_spread(_spread(0.05))[0] == "success" and ev.verdict_spread(_spread(0.0501))[0] == "warning"
    assert ev.verdict_spread(_spread(0.0501, eps="10 %"))[0] == "success" and ev.verdict_spread(_spread(0.1001, eps="10 %"))[0] == "warning"


def test_verdict_spread_infinite_mean_and_other_decompositions():
    text = ev.verdict_spread(_spread(0.49, m=10, mean_carl=math.inf))[1]
    assert "Mittel der bedingten ARL₀ ist unendlich" in text and "unbedingte Lauflängenverteilung" not in text
    only_mu = ev.verdict_spread(_spread(0.1, zerl="nur μ geschätzt"))[1]
    assert "nur μ geschätzt" in only_mu and "Mittel der bedingten ARL₀" not in only_mu          # die Maße 2 bis 4 gelten nur für beide geschätzt


def test_verdict_coupling_names_correlation_and_ratio_quantiles():
    kind, text = ev.verdict_coupling(_spread(0.1))
    assert kind == "info" and "Rangkorrelation" in text and "0,89" in text and "Median 0,99, 5 % bis 95 % 0,75 bis 1,61" in text and "43,9" in text and "Über 5 Phase-I-Stichproben" in text


# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------------------

def _minimum(m_min, p_before=0.050166, p_at=0.049828, m_sigma=261, m_mu=21, zerl="beide geschätzt", eps="5 %", f="½"):
    return ev.MinimumAnalysis(ev.Settings(zerl=zerl, eps=eps, f=f), m_min, p_before, p_at, m_sigma, m_mu, np.array([10, 100]), np.array([0.5, 0.1]), 185.2)


def test_verdict_minimum_names_m_both_probabilities_and_the_decomposition():
    kind, text = ev.verdict_minimum(_minimum(274))
    assert kind == "info"
    for part in ("höchstens 5 % der Phase-I-Stichproben", "weniger als ½ der Ziel-ARL₀ (185,2)", "m = 274 Phase-I-Beobachtungen", "bei m = 273: 5,02 %", "bei m = 274: 4,98 %",
                 "Nur σ geschätzt: m = 261; nur μ geschätzt: m = 21", "Faktor 12"):
        assert part in text
    assert "Nur σ geschätzt" not in ev.verdict_minimum(_minimum(21, zerl="nur μ geschätzt"))[1]


def test_verdict_minimum_at_the_lower_edge_and_unreachable():
    kind, text = ev.verdict_minimum(_minimum(2, p_before=None, p_at=0.04))
    assert kind == "info" and "m = 2 ist die kleinste Zahl" in text and "bei m = 1" not in text
    kind, text = ev.verdict_minimum(_minimum(None, None, None, None, None))
    assert kind == "warning" and "nicht erreichbar" in text


def test_verdict_widen_uses_the_row_of_the_requested_m():
    rows = [{"m": 100, "L_a": 3.0929, "L_b": 2.9414, "L_c": 3.1706, "arl1_L3": 43.89, "arl1_a": 54.94, "arl1_b": 38.25, "arl1_c": 66.69, "share_3": 0.1792, "share_b": 0.257, "median_3": 342.0, "median_b": 283.4},
            {"m": 30, "L_a": 3.3347, "L_b": 2.8019, "L_c": 3.6222, "arl1_L3": 43.89, "arl1_a": 102.16, "arl1_b": 27.89, "arl1_c": 228.81, "share_3": 0.36, "share_b": 0.4, "median_3": 287.5, "median_b": 250.0}]
    kind, text = ev.verdict_widen(rows)
    assert kind == "info"
    for part in ("Bei m = 100", "Faktor 3,17", "ARL₁(δ = 1) = 66,7 statt 43,9", "(Faktor 2,94)", "von 17,9 % auf 25,7 %", "Median 342,0 → 283,4"):
        assert part in text
    assert "Bei m = 30" in ev.verdict_widen(rows, 30)[1] and "Faktor 3,62" in ev.verdict_widen(rows, 30)[1]


# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------------------

def _cusum(c_share, w_share, se=0.005, n=6000):
    return ev.CusumAnalysis(ev.Settings(n_stich=n), 4.0965, 2.7822, np.ones(2), np.ones(2),
                            {"cusum": (357.5, 100.2, c_share, se), "shewhart": (360.9, 124.9, w_share, se)})


def test_verdict_cusum_compares_with_a_two_standard_error_band():
    se = math.hypot(0.005, 0.005)
    assert "stärker betroffen als die Shewhart-Karte" in ev.verdict_cusum(_cusum(0.21, 0.156))[1]
    assert "weniger stark betroffen als die Shewhart-Karte" in ev.verdict_cusum(_cusum(0.15, 0.20))[1]
    assert "ist der CUSUM von der Shewhart-Karte nicht zu unterscheiden;" in ev.verdict_cusum(_cusum(0.15, 0.15))[1]
    assert "stärker betroffen" in ev.verdict_cusum(_cusum(0.15 + 2.01 * se, 0.15))[1] and "nicht zu unterscheiden" in ev.verdict_cusum(_cusum(0.15 + 1.99 * se, 0.15))[1]
    assert "nicht zu unterscheiden" in ev.verdict_cusum(_cusum(0.15 - 1.99 * se, 0.15))[1] and "weniger stark" in ev.verdict_cusum(_cusum(0.15 - 2.01 * se, 0.15))[1]


def test_verdict_cusum_names_numbers_and_the_contrary_source():
    kind, text = ev.verdict_cusum(_cusum(0.2097, 0.156))
    assert kind == "info"
    for part in ("m = 100", "k = 0,50", "h = 4,0965", "6000 Phase-I-Stichproben", "CUSUM Median 357,5, 5 % 100,2, 21,0 % unter 185,2", "Shewhart-Karte Median 360,9, 5 % 124,9, 15,6 % unter 185,2",
                 "Zwetsloot/Woodall 2017", "das Gegenteil"):
        assert part in text


# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------------------

def _row(kind, size, s, mr):
    return {"kind": kind, "size": size, "s": s, "MR": mr}


def _contamination(chosen_row, stoer):
    base = _row("keine", 0.0, (342.3, 42.8, 42.9), (347.5, 43.5, 43.5))
    return {"cases": [base], "chosen": chosen_row, "settings": ev.Settings(stoerung=stoer, n_stich=40000)}


def test_verdict_contamination_without_disturbance_just_lists_the_baseline():
    kind, text = ev.verdict_contamination(_contamination(_row("keine", 0.0, (342.3, 42.8, 42.9), (347.5, 43.5, 43.5)), "keine"))
    assert kind == "info" and "Ohne Störung" in text and "342,3" in text and "347,5" in text and "Wähle eine Störung" in text


def test_verdict_contamination_outlier_makes_the_card_quiet_and_slow():
    res = _contamination(_row("Ausreißer", 5.0, (1129.0, 120.0, 91.7), (688.1, 82.5, 64.0)), "Ausreißer +5σ")
    kind, text = ev.verdict_contamination(res)
    assert kind == "warning" and "einem Ausreißer +5,0σ" in text and "ARL₀ 1129,0 statt 342,3 (Schätzer s)" in text and "688,1 statt 347,5 (MR)" in text
    assert "ARL₁ für den Aufwärtssprung 1: 120,0 statt 42,8 (s), 82,5 statt 43,5 (MR)" in text and "im Zustand ohne Störung" in text and "ruhiger und träger" in text
    assert "bringt weder s noch MR" not in text


def test_verdict_contamination_step_two_sigma_no_estimator_saves_it_and_thresholds():
    res = _contamination(_row("Stufe", 2.0, (1707.0, 80.9, 41913.9), (46.2, 6.5, 378.3)), "Stufe 2σ in der Mitte")
    kind, text = ev.verdict_contamination(res)
    assert kind == "warning" and "einer Stufe von 2,0σ in der Mitte" in text and "41.913,9" in text and "im Zustand der zweiten Hälfte" in text and "bringt weder s noch MR die ARL₀ in die Nähe des Ziels" in text
    assert "Segmentierung (Stück 7)" in text
    # Schwellen: s über das Doppelte des Ziels UND MR unter der Hälfte
    assert "bringt weder s noch MR" not in ev.verdict_contamination(_contamination(_row("Stufe", 2.0, (2 * T - 1, 1, 1), (0.5 * T - 1, 1, 1)), "Stufe 2σ in der Mitte"))[1]
    assert "bringt weder s noch MR" in ev.verdict_contamination(_contamination(_row("Stufe", 2.0, (2 * T + 1, 1, 1), (0.5 * T - 1, 1, 1)), "Stufe 2σ in der Mitte"))[1]
    assert "bringt weder s noch MR" not in ev.verdict_contamination(_contamination(_row("Stufe", 2.0, (2 * T + 1, 1, 1), (0.5 * T + 1, 1, 1)), "Stufe 2σ in der Mitte"))[1]


def test_verdict_contamination_small_effect_is_only_info():
    res = _contamination(_row("Stufe", 0.5, (400.0, 40.0, 100.0), (330.0, 41.0, 60.0)), "Stufe 0,5σ in der Mitte")
    assert ev.verdict_contamination(res)[0] == "info"
