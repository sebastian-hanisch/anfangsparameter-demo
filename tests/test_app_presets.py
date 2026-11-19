"""App (AppTest): je Preset die erwartete Verdikt-Zeile (kein Preset ohne Wirkung meldet eine Wirkung), jeder Regler an beiden Rändern, Permalink, Expander und Tabellen.

Die Smoke-Tests (jeder Button, jeder Slider an Minimum und Maximum) stehen in tests/test_app.py; hier kommen die Regler dazu, die `at.slider` nicht erfasst (Auswahl-Regler, Zahleneingaben)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import aps_constants as C
import aps_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at) -> dict:
    return {kind: [e.value for e in getattr(at, kind)] for kind in ("success", "info", "warning", "error")}


def _all(at) -> str:
    return "\n".join(v for vals in _texts(at).values() for v in vals)


def _preset(name) -> AppTest:
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, (name, [str(e) for e in at.exception])
    return at


def test_standard_preset_gives_the_anchor_verdicts():
    at = _app()
    texts, everything = _texts(at), _all(at)
    assert not texts["error"]
    for part in ("Median der bedingten ARL₀ 342,0 (Ziel 370,4), 5 % 117,9, 95 % 1162,5", "17,9 % der Phase-I-Stichproben", "Mittel der bedingten ARL₀ 455,9", "Kehrwert der mittleren Alarmrate 280,0",
                 "unbedingte Lauflängenverteilung: Median 229,2", "m = 274 Phase-I-Beobachtungen", "Nur σ geschätzt: m = 261; nur μ geschätzt: m = 21",
                 "A (−½, 0, ½): Grenzen [−1,5; 1,5], bedingte ARL₀ 7,48", "5,07·10⁸", "h = 4,0965", "Faktor 3,17", "ARL₁(δ = 1) = 66,7 statt 43,9", "von 17,9 % auf 25,7 %"):
        assert part in everything, part
    assert any("Diese Phase-I-Stichprobe (m = 100, Zufalls-Seed 35" in v for v in texts["info"] + texts["warning"])
    assert any(v.startswith("m = 100 Phase-I-Beobachtungen") for v in texts["warning"])                 # 17,9 % > 5 %: Warnung


def test_preset_three_observations_shows_the_infinite_mean_and_the_hand_samples():
    at = _preset("Drei Beobachtungen (Handbeispiel)")
    everything = _all(at)
    assert "m = 3 Phase-I-Beobachtungen" in everything and "Mittel der bedingten ARL₀ ist unendlich" in everything and "ARL₀ von 5,51 bis 5,07·10⁸ (8 Größenordnungen)" in everything
    assert "Kehrwert der mittleren Alarmrate 8,2" in everything                         # H 4: 1/E[p] = 8,22 bei m = 3
    assert any(v.startswith("m = 3 Phase-I-Beobachtungen") for v in _texts(at)["warning"])


def test_preset_m30_and_m91_anchor_numbers():
    everything = _all(_preset("m = 30"))
    assert "Mittel der bedingten ARL₀ 885,8" in everything and "Median der bedingten ARL₀ 287,5" in everything and "36,0 % der Phase-I-Stichproben" in everything
    everything = _all(_preset("m = 91 (Referenzfenster)"))
    assert "Median der bedingten ARL₀ 339,4" in everything and "19,4 % der Phase-I-Stichproben" in everything


def test_preset_guarantee_m274_is_the_first_m_below_five_percent():
    at = _preset("Garantie: m = 274")
    texts = _texts(at)
    assert any(v.startswith("m = 274 Phase-I-Beobachtungen") and "5,0 % der Phase-I-Stichproben" in v for v in texts["success"])       # 4,98 % <= ε = 5 %
    assert not any(v.startswith("m = 274 Phase-I-Beobachtungen") for v in texts["warning"])
    assert "m = 274 Phase-I-Beobachtungen (bei m = 273: 5,02 %, bei m = 274: 4,98 %)" in _all(at)


def test_preset_mean_is_infinite_at_m10():
    at = _preset("Mittel ist unendlich (m = 10)")
    assert any("Mittel der bedingten ARL₀ ist unendlich" in v for v in _all(at).split("\n"))
    assert "m = 10 Phase-I-Beobachtungen" in _all(at)


def test_preset_widen_limits_reaches_the_guarantee():
    at = _preset("Grenzen weiten (m = 100)")
    texts = _texts(at)
    assert at.sidebar.number_input(key="L_input").value == 3.18
    assert any(v.startswith("m = 100 Phase-I-Beobachtungen") and "Grenzfaktor 3,18" in v for v in texts["success"])        # 4,6 % <= 5 %
    assert "Grenzfaktor 3,18 und beide geschätzt braucht das m = 97 Phase-I-Beobachtungen" in _all(at)                       # kleinstes m bei diesem L


def test_preset_cusum_uses_6000_samples():
    at = _preset("CUSUM, m = 100")
    assert at.sidebar.slider(key="nst_slider").value == 6000
    assert any("6000 Phase-I-Stichproben: CUSUM Median" in v and "stärker betroffen als die Shewhart-Karte" in v for v in _texts(at)["info"])


def test_preset_step_in_phase_i_says_no_estimator_saves_it():
    at = _preset("Stufe in Phase I")
    assert at.sidebar.selectbox(key="stoer_select").value == "Stufe 2σ in der Mitte"
    assert any("einer Stufe von 2,0σ in der Mitte" in v and "bringt weder s noch MR die ARL₀ in die Nähe des Ziels" in v for v in _texts(at)["warning"])


def test_without_disturbance_the_contamination_experiment_reports_no_effect():
    at = _app()
    assert any(v.startswith("Ohne Störung") and "Wähle eine Störung" in v for v in _texts(at)["info"])
    assert not any("bringt weder s noch MR" in v for v in _all(at).split("\n"))


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
@pytest.mark.parametrize("edge", ["first", "last"])
def test_every_regler_at_both_edges_does_not_raise(state_key, edge):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    _regler(at, state_key).set_value(spec.first if edge == "first" else spec.last).run()
    assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


def test_all_regler_at_the_lower_and_upper_edge_together():
    for edge in ("first", "last"):
        at = _app()
        for state_key, spec in P.SETTING_SPECS.items():
            _regler(at, state_key).set_value(getattr(spec, edge))
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
        assert not _texts(at)["error"]


def test_next_sample_button_advances_the_seed_and_the_history():
    at = _app()
    at.sidebar.button[-1].click().run()
    assert not at.exception and at.sidebar.number_input(key="seed_input").value == C.DEFAULT_SEED + 1
    assert any("Zufalls-Seed 36" in v for v in _all(at).split("\n"))


def test_permalink_limits_values_and_ignores_unknown_choices():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["m"] = "77"
    at.query_params["L"] = "99"
    at.query_params["est"] = "Median"
    at.query_params["nst"] = "-5"
    at.run()
    assert not at.exception
    assert at.sidebar.select_slider(key="m_select").value == 91                         # rastet auf den nächsten Listenwert ein
    assert at.sidebar.number_input(key="L_input").value == C.L_MAX
    assert at.sidebar.radio(key="est_radio").value == C.DEFAULT_ESTIMATOR               # unbekannte Auswahl wird ignoriert
    assert at.sidebar.slider(key="nst_slider").value == C.N_STICH_MIN


def test_expanders_and_dataframes_are_present():
    at = _app()
    labels = [e.label for e in at.expander]
    assert "So funktioniert Parameter schätzen" in labels and "📐 Mathematische Formulierung" in labels
    assert any(label.startswith("❓ Und wenn die Annahmen nicht gelten?") for label in labels)
    assert len(at.dataframe) == 7                                    # Handbeispiel, vier Maße, Grenzen weiten, Gegenprobe CUSUM, Störungen, Tabelle I, Zeile J
    assert [s.value.split(":")[0] for s in at.subheader if s.value.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)]
    assert any(s.value == "🚧 Wo die Annahmen enden" for s in at.subheader)
