"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, erweitert um Auswahl-Regler).

Zahlenregler haben Grenzen (`lo`, `hi`), Auswahlregler eine Optionsliste (`options`); der Permalink begrenzt Zahlen auf die Grenzen und übernimmt Auswahlwerte nur,
wenn sie in der Liste stehen (die Phase-I-Größe m rastet auf den nächsten Listenwert ein)."""

from dataclasses import dataclass

import streamlit as st

import aps_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None

    @property
    def first(self):
        return self.options[0] if self.options else self.lo

    @property
    def last(self):
        return self.options[-1] if self.options else self.hi


SETTING_SPECS = {
    "m_select": SettingSpec("m", int, C.DEFAULT_M, options=C.M_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "est_radio": SettingSpec("est", str, C.DEFAULT_ESTIMATOR, options=C.ESTIMATORS),
    "L_input": SettingSpec("L", float, C.DEFAULT_L, C.L_MIN, C.L_MAX),
    "zerl_radio": SettingSpec("zerl", str, C.DEFAULT_ZERLEGUNG, options=C.ZERLEGUNG),
    "f_radio": SettingSpec("f", str, C.DEFAULT_F, options=C.F_OPTIONS),
    "eps_radio": SettingSpec("eps", str, C.DEFAULT_EPS, options=C.EPS_OPTIONS),
    "k_input": SettingSpec("k", float, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "nst_slider": SettingSpec("nst", int, C.DEFAULT_N_STICH, C.N_STICH_MIN, C.N_STICH_MAX),
    "stoer_select": SettingSpec("stoer", str, C.DEFAULT_STOERUNG, options=C.STOERUNG),
}
PRESET_KEYS = {"m": "m_select", "seed": "seed_input", "estimator": "est_radio", "L": "L_input", "zerl": "zerl_radio", "f": "f_radio", "eps": "eps_radio",
               "k": "k_input", "n_stich": "nst_slider", "stoerung": "stoer_select"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp_value(spec: SettingSpec, value):
    """Wert auf die Grenzen bzw. die Optionen des Reglers begrenzen; None, wenn eine Auswahl nicht in der Liste steht."""
    if spec.options is not None:
        if spec.caster is int:
            return min(spec.options, key=lambda o: abs(o - value))
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = clamp_value(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: was das Preset nicht nennt, geht auf den Standardwert (so stimmt `aktives_preset`, das fehlende Einträge als Standard liest)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def next_sample():
    """Weitere Phase-I-Stichprobe: der nächste Zufalls-Seed (zyklisch)."""
    st.session_state["seed_input"] = (int(st.session_state.get("seed_input", C.DEFAULT_SEED)) + 1) % (C.SEED_MAX + 1)
