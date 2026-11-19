"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ein eigener Zustand; Permalink-Begrenzung und Einrasten."""

import pytest

import aps_constants as C
import aps_evaluation as ev
import aps_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.STANDARD_PRESET]) == ev.Settings()
    assert next(name for name in C.PRESETS if ev.Settings(**C.PRESETS[name]) == ev.Settings()) == C.STANDARD_PRESET     # genau ein Preset ist der Standard


def test_presets_are_pairwise_different_states():
    states = [ev.Settings(**preset) for preset in C.PRESETS.values()]
    assert len(set(states)) == len(states)                     # sonst könnte aktives_preset nicht eindeutig hervorheben


def test_setting_specs_cover_every_preset_key_and_default_to_the_settings_defaults():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    default = ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == getattr(default, key), key


def test_m_options_contain_the_numbers_of_the_presets_and_the_frozen_grid():
    for m in (3, 10, 30, 91, 100, 274):
        assert m in C.M_OPTIONS
    assert list(C.M_OPTIONS) == sorted(C.M_OPTIONS) and C.M_MIN == 3 and C.M_MAX == 5000


def test_clamp_value_limits_numbers_snaps_m_and_rejects_unknown_choices():
    spec_L, spec_m, spec_est = P.SETTING_SPECS["L_input"], P.SETTING_SPECS["m_select"], P.SETTING_SPECS["est_radio"]
    assert P.clamp_value(spec_L, 99.0) == C.L_MAX and P.clamp_value(spec_L, -1.0) == C.L_MIN and P.clamp_value(spec_L, 3.3) == 3.3
    assert P.clamp_value(spec_m, 77) == 91 and P.clamp_value(spec_m, 1) == 3 and P.clamp_value(spec_m, 10 ** 6) == 5000 and P.clamp_value(spec_m, 274) == 274
    assert P.clamp_value(spec_est, "MR") == "MR" and P.clamp_value(spec_est, "Median") is None


def test_every_option_value_of_a_regler_is_valid_for_the_evaluation():
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is not None and state_key not in ("m_select",):
            assert len(set(spec.options)) == len(spec.options)
    assert set(C.STOERUNG_FAELLE) == set(C.STOERUNG) and C.STOERUNG[0] == "keine"
    assert set(C.F_VALUES) == set(C.F_OPTIONS) and set(C.EPS_VALUES) == set(C.EPS_OPTIONS)
    for name in C.STOERUNG:
        kind, size = ev.Settings(stoerung=name).stoerung_fall
        assert kind in ("keine", "Ausreißer", "Stufe") and size >= 0
    assert [ev.Settings(zerl=z).mode for z in C.ZERLEGUNG] == ["both", "mu", "sigma"]


def test_history_seeds_end_with_the_seed_and_have_the_declared_length():
    seeds = ev.history_seeds(35)
    assert seeds[-1] == 35 and len(seeds) == C.HISTORY_LENGTH and seeds == list(range(16, 36))
