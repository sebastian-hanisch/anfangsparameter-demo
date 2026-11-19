"""Literaturwerte (nur gelesene Quellen, Stufe W), mit Quelle und Stand im Docstring (ORACLE-PLAYBOOK §5).

Quelle: Jones-Farmer, Woodall, Steiner, Champ 2014, „An Overview of Phase I Analysis for Process Improvement and Monitoring“, Journal of Quality Technology 46, 265–280 (Abschnitt 2, gelesen).
Die Wahrscheinlichkeit mindestens eines Fehlalarms einer rückblickenden Phase-I-Karte mit 3σ-Grenzen und bekannten Parametern ist FAP = 1 − (1 − p)^m: die Quelle nennt 0,078 für m = 30 und 0,126
für m = 50 Kartenpunkte; die Nachrechnung gibt 0,0779 und 0,1264. Die 3σ-ARL₀ 370,40 ist der Lehrbuchwert (1/(2Φ(−3))).

Nicht getestet, weil nicht gelesen: Jensen u. a. 2006, die Tabelle 1 von Zwetsloot/Woodall 2017 (nicht lesbar), die Tabellen von Diko u. a. 2019 (nur der Verweis „offen“)."""

import pytest

import aps_constants as C
import aps_exact as E


def test_false_alarm_probability_of_a_phase_i_chart_matches_jones_farmer_2014():
    fap = lambda m: 1.0 - (1.0 - C.P_NOMINAL) ** m
    assert fap(30) == pytest.approx(0.078, abs=5e-4) and fap(30) == pytest.approx(0.0779, abs=5e-5)
    assert fap(50) == pytest.approx(0.126, abs=5e-4) and fap(50) == pytest.approx(0.1264, abs=5e-5)
    assert fap(100) == pytest.approx(0.2369, abs=5e-5) and fap(300) == pytest.approx(0.5556, abs=5e-5)
    assert float(E.alarm_within(C.P_NOMINAL, 100)) == pytest.approx(fap(100), abs=1e-12)           # die Demo-Funktion gegen die Formel der Quelle


def test_three_sigma_arl0_is_370_40():
    assert E.nominal_arl(3.0) == pytest.approx(370.40, abs=5e-3)
