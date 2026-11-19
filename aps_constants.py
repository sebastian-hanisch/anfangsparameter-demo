"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Alle Beobachtungen sind in Einheiten der wahren Streuung angegeben (wahre Reihe N(0, 1), nach einer Änderung N(δ, 1)); die Karte kennt das nicht und schätzt Mittel und
Streuung aus den m Beobachtungen der Phase I."""

import math

# --- Karte und Ziel -------------------------------------------------------------------------------------------------------------------------------
L_NOMINAL = 3.0                                              # Grenzfaktor der Standardkarte (zweiseitig, Grenzen x̄ ± L·s)
P_NOMINAL = math.erfc(L_NOMINAL / math.sqrt(2.0))            # 2 Φ(−3) = 0,0026998: Alarmwahrscheinlichkeit je Beobachtung bei bekannten Parametern
TARGET_ARL0 = 1.0 / P_NOMINAL                                # 370,398: das Ziel ARL₀ (gesetzt wären 370; die Linie rechnet mit 1/p der 3σ-Karte)
CUSUM_K = 0.5                                                # Referenzwert des CUSUM (σ-Einheiten)
SHIFT_DELTA = 1.0                                            # Sprunggröße für die bedingte ARL₁ (σ-Einheiten)
N_SIM_HORIZON = 100                                          # „Alarm in den ersten 100 Beobachtungen“

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------------
# Phase-I-Umfang m: feste Auswahl (logarithmisch verteilt, enthält die Zahlen der Presets und der Messreihe)
M_OPTIONS = (3, 4, 5, 6, 8, 10, 11, 15, 20, 30, 50, 91, 100, 150, 200, 274, 300, 500, 1000, 2000, 5000)
DEFAULT_M = 100
M_MIN, M_MAX = M_OPTIONS[0], M_OPTIONS[-1]
SEED_MAX, DEFAULT_SEED = 999999, 35
L_MIN, L_MAX, DEFAULT_L = 1.5, 6.0, L_NOMINAL
ESTIMATORS = ("s", "MR")                                     # Stichproben-Standardabweichung (Nenner m − 1) / mittlere gleitende Spannweite ÷ 1,1284
DEFAULT_ESTIMATOR = "s"
ZERLEGUNG = ("beide geschätzt", "nur μ geschätzt", "nur σ geschätzt")
DEFAULT_ZERLEGUNG = ZERLEGUNG[0]
F_OPTIONS = ("½", "¾", "0,9")
F_VALUES = {"½": 0.5, "¾": 0.75, "0,9": 0.9}
DEFAULT_F = "½"
EPS_OPTIONS = ("10 %", "5 %", "1 %")
EPS_VALUES = {"10 %": 0.10, "5 %": 0.05, "1 %": 0.01}
DEFAULT_EPS = "5 %"
K_MIN, K_MAX, DEFAULT_K = 0.25, 1.5, CUSUM_K
N_STICH_MIN, N_STICH_MAX, N_STICH_STEP, DEFAULT_N_STICH = 1000, 10000, 1000, 4000   # Zahl der Phase-I-Stichproben (Experimente 2, 4, 5)
# Störung der Phase I als EINE Auswahl (Art und Größe in σ): eine getrennte Größenwahl wäre bei „keine“ ein toter Regler.
STOERUNG_FAELLE = {
    "keine": ("keine", 0.0),
    "Ausreißer +3σ": ("Ausreißer", 3.0),
    "Ausreißer +5σ": ("Ausreißer", 5.0),
    "Ausreißer +10σ": ("Ausreißer", 10.0),
    "Stufe 0,5σ in der Mitte": ("Stufe", 0.5),
    "Stufe 1σ in der Mitte": ("Stufe", 1.0),
    "Stufe 2σ in der Mitte": ("Stufe", 2.0),
    "Stufe 3σ in der Mitte": ("Stufe", 3.0),
}
STOERUNG = tuple(STOERUNG_FAELLE)
DEFAULT_STOERUNG = STOERUNG[0]
M_PHASE_CONTAMINATION = 100                                  # Experiment 5 rechnet immer mit m = 100 (Plan 04)

# --- Gitter der Rechnung --------------------------------------------------------------------------------------------------------------------------
HISTORY_LENGTH = 20                                          # Experiment 1: die letzten 20 Stichproben (Zufalls-Seeds s − 19 … s)
SCATTER_POINTS = 3000                                        # Experiment 2: Punkte im Streudiagramm (aus den n_stich Stichproben)
CUSUM_NODES = 40                                             # Gauss-Legendre-Knoten der Integralgleichung je Stichprobe (Plan Z3)
CUSUM_NODES_TARGET = 64                                      # Knoten für die Schwelle h bei bekannten Parametern
CHECK_RUNS = 4000                                            # Läufe der Gegenprobe im Experiment 4
CHECK_GRID = 400                                             # Zellen des Brook/Evans-Gitters der Gegenprobe

# --- Festgelegte Stichproben des Handbeispiels (H 3; ausgedacht, nur zum Nachrechnen von Hand) ----------------------------------------------------
HAND_SAMPLES = {
    "A (−½, 0, ½)": (-0.5, 0.0, 0.5),
    "B (0, ½, 1)": (0.0, 0.5, 1.0),
    "C (−2, 0, 2)": (-2.0, 0.0, 2.0),
}

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------------
COLOR_CHART = "#2F6B65"
COLOR_TRUE = "#c44e52"
COLOR_LINE = "#4c78a8"
COLOR_ACCENT = "#dd8452"
COLOR_GREY = "rgba(150,150,150,0.55)"

# --- Presets (Name -> Reglerwerte; fehlende Regler meinen ihren Standardwert) -------------------------------------------------------------------------
STANDARD_PRESET = "m = 100"
PRESETS = {
    "Drei Beobachtungen (Handbeispiel)": {"m": 3},
    "m = 30": {"m": 30},
    "m = 91 (Referenzfenster)": {"m": 91},
    STANDARD_PRESET: {"m": 100},
    "Garantie: m = 274": {"m": 274},
    "Mittel ist unendlich (m = 10)": {"m": 10},
    "Grenzen weiten (m = 100)": {"m": 100, "L": 3.18},
    "CUSUM, m = 100": {"m": 100, "n_stich": 6000},
    "Stufe in Phase I": {"m": 100, "stoerung": "Stufe 2σ in der Mitte"},
}
PRESET_HELP = {
    "Drei Beobachtungen (Handbeispiel)": "Phase I mit nur drei Beobachtungen: die drei ausgedachten Stichproben des Handbeispiels ergeben Grenzen, die um Größenordnungen auseinanderliegen.",
    "m = 30": "Eine kleine Phase I mit 30 Beobachtungen: die bedingte ARL₀ streut breit, das Mittel der bedingten ARL₀ ist weit über 370.",
    "m = 91 (Referenzfenster)": "91 Beobachtungen (so groß ist das Referenzfenster Tage 639 bis 729 im Niveausprung-Detektor); hier nur als Zeile der Einzelwert-Karte, nicht übertragbar.",
    STANDARD_PRESET: "Voreinstellung: 100 Beobachtungen in Phase I, Grenzen x̄ ± 3 s, zweiseitig.",
    "Garantie: m = 274": "Das kleinste m, bei dem höchstens 5 % der Phase-I-Stichproben eine Karte mit weniger als der halben nominellen ARL₀ ergeben.",
    "Mittel ist unendlich (m = 10)": "Bei m ≤ 10 ist das Mittel der bedingten ARL₀ unendlich (eigene Herleitung); die Verteilung selbst ist endlich.",
    "Grenzen weiten (m = 100)": "Grenzfaktor 3,18 statt 3 (der Wert für die Garantie ist 3,1706, auf zwei Stellen gerundet 3,17 gäbe 5,03 %): weniger kurze bedingte ARL₀, aber auf Kosten der Erkennungsgeschwindigkeit.",
    "CUSUM, m = 100": "Einseitiger CUSUM mit geschätzten Parametern, 6000 Phase-I-Stichproben, daneben die einseitige Shewhart-Karte.",
    "Stufe in Phase I": "Eine Stufe von 2σ in der Mitte der Phase I (m = 100): weder s noch MR bringt die ARL₀ in die Nähe des Ziels; die Stufe zu finden ist die Aufgabe der Segmentierung (Stück 7).",
}
