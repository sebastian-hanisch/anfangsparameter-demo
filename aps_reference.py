"""Eingefrorene Tabellen (aps_exact.json): schwere Rechnungen, die die App nicht live macht, und die Tabelle I aus der Vormessung V13b.

Jedes Feld der Datei hat `quelle` (Rechenweg und Voraussetzungen), `stand` (Datum) und `werte`. Neu rechnen und prüfen: `python tools/aps_make_exact.py --check`."""

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent / "aps_exact.json"


@lru_cache(maxsize=1)
def load() -> dict:
    with open(PATH, encoding="utf-8") as fh:
        return json.load(fh)


def source(field: str) -> str:
    """Quelle und Stand eines Feldes, für Fußnoten der App."""
    entry = load()[field]
    return f"{entry['quelle']} (Stand {entry['stand']})"


def mean_carl(m: int):
    """E[1/p] (L = 3) aus der Tabelle; None, wenn m nicht in der Tabelle steht."""
    return load()["mean_carl"]["werte"].get(str(m))


def unconditional(m: int):
    """Unbedingte Lauflängenverteilung (L = 3): {p_alarm_100, median, q05, q95}; None, wenn m nicht in der Tabelle steht."""
    return load()["unbedingt"]["werte"].get(str(m))


def widen_table() -> dict:
    """Grenzfaktoren L(a), L(b), L(c) und ARL₁-Preise je m (str → dict)."""
    return load()["weiten"]["werte"]


def table_i() -> dict:
    """Tabelle I (AR(1)-Phase-I, Rohdatenkarte): m, phi, arl0_bekannt, anteil_unter_halber_arl0, median."""
    return load()["abhaengig"]["werte"]
