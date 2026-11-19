"""Die eingefrorenen Tabellen (aps_exact.json): Aufbau, Quelle und Stand je Feld, Querprüfungen gegen die Live-Rechnung der App und die Neurechnung des Skripts (schneller Modus)."""

import importlib.util
import math
from pathlib import Path

import pytest

import aps_constants as C
import aps_evaluation as ev
import aps_exact as E
import aps_reference as R

ROOT = Path(__file__).resolve().parent.parent


def _maker():
    spec = importlib.util.spec_from_file_location("aps_make_exact", ROOT / "tools" / "aps_make_exact.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_field_has_source_and_date():
    data = R.load()
    assert set(data) == {"mean_carl", "unbedingt", "weiten", "abhaengig"}
    for name, entry in data.items():
        assert set(entry) == {"quelle", "stand", "werte"}, name
        assert entry["quelle"] and entry["stand"] == "2026-11-18", name
        assert entry["quelle"] in R.source(name) and entry["stand"] in R.source(name)


def test_mean_carl_is_tabulated_exactly_for_the_finite_slider_values():
    for m in C.M_OPTIONS:
        assert (R.mean_carl(m) is not None) == (m >= 11), m                  # genau die m mit endlichem Mittel
    assert R.mean_carl(10) is None and R.mean_carl(3) is None


def test_inverse_mean_alarm_rate_is_below_the_mean_of_the_conditional_arl_everywhere():
    for m in C.M_OPTIONS:
        if m >= 11:
            assert R.mean_carl(m) > E.inverse_mean_alarm_rate(m), m           # Jensen
    assert all(R.mean_carl(m) > C.TARGET_ARL0 for m in C.M_OPTIONS if m >= 11)


def test_widen_table_is_consistent_with_the_exact_distribution():
    for key, v in R.widen_table().items():
        m = int(key)
        assert v["L_a"] == pytest.approx(E.limit_equal_alarm_rate(m), abs=1e-12)
        assert E.cdf_carl(0.5 * C.TARGET_ARL0, m, v["L_c"]) == pytest.approx(0.05, abs=1e-6)             # L(c): 5 %
        assert E.inverse_mean_alarm_rate(m, v["L_a"]) == pytest.approx(C.TARGET_ARL0, rel=1e-9)          # L(a): gleiche mittlere Alarmrate
        assert E.arl1_known(v["L_c"], 1.0) == pytest.approx(v["arl1_c"], rel=1e-12)
        assert v["L_b"] < 3.0 < v["L_a"]


def test_mean_carl_at_L_b_is_the_target():
    for m in (30, 100, 300):
        assert E.mean_carl(m, R.widen_table()[str(m)]["L_b"]) == pytest.approx(C.TARGET_ARL0, rel=1e-5), m


def test_table_i_shape_and_monotony_in_phi():
    ti = R.table_i()
    assert ti["m"] == [30, 100, 300, 1000] and ti["phi"] == [0.0, 0.3, 0.6, 0.9]
    share = ti["anteil_unter_halber_arl0"]
    for j in range(len(ti["m"])):
        col = [share[i][j] for i in range(len(ti["phi"]))]
        assert col == sorted(col), j                                         # die Autokorrelation verschlechtert die Verteilung monoton in φ (diese Messung)
    for row in share:
        assert row == sorted(row, reverse=True)                              # mehr Phase I hilft in jeder Zeile
    assert ti["arl0_bekannt"][3] == 831.8                                    # Shewhart bei AR(1), φ = 0,9: 831,8 (nicht 831,5)


def test_make_exact_quick_check_passes():
    """tools/aps_make_exact.py --check --quick rechnet m = 100 und 300 mit dem DOPPELINTEGRAL neu und vergleicht auf 1e-6 relativ mit der Datei (wenige Sekunden)."""
    assert _maker().main(["--check", "--quick"]) == 0


def test_make_exact_refuses_writing_in_quick_mode_and_prints_usage_without_arguments(capsys):
    mod = _maker()
    assert mod.main(["--write", "--quick"]) == 2
    assert mod.main([]) == 2 and "--check" in capsys.readouterr().out


def test_make_exact_compare_detects_a_changed_value():
    mod = _maker()
    assert mod.compare({"a": {"b": 1.0}}, {"a": {"b": 1.0 + 1e-9}}) == []
    assert len(mod.compare({"a": {"b": 1.0}}, {"a": {"b": 1.1}})) == 1
    assert len(mod.compare({}, {"a": 1.0})) == 1


def test_widen_rows_of_the_evaluation_use_the_exact_distribution():
    rows = ev.widen_rows()
    r100 = next(r for r in rows if r["m"] == 100)
    assert r100["share_3"] == pytest.approx(E.cdf_carl(0.5 * C.TARGET_ARL0, 100), abs=1e-15)
    assert math.isclose(r100["share_c"], 0.05, abs_tol=1e-6)
