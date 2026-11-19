# Parameter schätzen: wie viel Phase I braucht eine Karte? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-anfangsparameter-demo.streamlit.app/)**

Stück 4 der Linie **„Änderungserkennung“** (Konzepte-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Block „online“: Karten, die im laufenden Betrieb Alarm geben). Der Vorgänger (Stück 3) vergleicht Karten
mit **bekanntem** Mittelwert und bekannter Streuung; dieses Stück hebt genau diese Annahme auf (Kante „fix“): Die Karte schätzt x̄ und s aus **m Beobachtungen einer Phase I** und baut daraus ihre Grenzen. Der Nachfolger (Stück 5) lässt die
Unabhängigkeit der Werte fallen; ein Schwesterstück (4b) rechnet Phase I bei abhängigen Werten und mehreren Kanälen. Das Vehikel ist eine **ausgedachte Messreihe** (unabhängige, normalverteilte Einzelwerte, wahres Mittel 0 und wahre Streuung 1, die Karte kennt beides nicht);
in den gelesenen Quellen gibt es keine reale Phase-I-Stichprobe aus Logistik, Lager oder Fertigung mit bekannter Wahrheit. Was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet.

## Kernfrage

Die Zahl **„ARL₀ = 370“** gehört zu einer Karte mit bekannten Parametern. Mit Grenzen aus einer Phase-I-Stichprobe ist die Zeit bis zum ersten Fehlalarm eine **Zufallsgröße**, die an der Stichprobe hängt. Drei Fragen: (1) Welche Karte entsteht aus einer Phase-I-Stichprobe,
und wie lange hält sie? (2) Wie breit streut die ARL₀, und welche „mittlere“ ARL₀ ist gemeint? (3) Wie viele Beobachtungen braucht Phase I, oder wie weit müssen die Grenzen werden? Gezählt wird immer getrennt: **Beobachtungen bis zum Alarm** (ARL),
**Phase-I-Stichproben** (Wiederholungen des Schätzens) und **Läufe** (Simulation). „Lauflänge“ heißt hier Lauflänge bis zum Alarm.

## Ergebnis in Kürze

Alle Zahlen gelten für die zweiseitige Shewhart-Einzelkarte mit Grenzen x̄ ± 3 s, unabhängige N(0, 1)-Beobachtungen in Phase I und II, ν = 0 (die Änderung liegt vor der ersten Beobachtung von Phase II; die Karte hat kein Gedächtnis, jedes ν gibt dieselbe bedingte ARL bei fester Phase-I-Stichprobe) und das ARL₀-Ziel
**370,398** (= 1/p der 3σ-Karte, nicht das gesetzte 370), soweit nichts anderes dasteht. „Exakt“ heißt ohne Zufall gerechnet.

- **m = 100 (exakt, Zähleinheit Phase-I-Stichproben):** Median der bedingten ARL₀ **342,0**, 5 %-Quantil **117,9**, 95 %-Quantil **1162,5**; **17,9 %** der Phase-I-Stichproben ergeben eine Karte mit bedingter ARL₀ unter **185,2**, der Hälfte des Ziels. Mit bekannten Parametern wäre es ARL₀ 370,4.
- **Welche „mittlere“ ARL₀?** Für dieselbe Karte (m = 100) nennt das Mittel der bedingten ARL₀ **455,9**, der Kehrwert der mittleren Alarmrate (viele Karten gleichzeitig) **280,0**, der Median der bedingten **342,0** und der Median der unbedingten Lauflänge (jede Lauflänge mit frischer Phase I) **229,2**.
  Bei m = 30: 885,8 / 161,1 / 287,5 / 186,7, und **36,0 %** der Stichproben unter 185,2; bei m = 91 (so groß ist das Referenzfenster Tage 639 bis 729 im Niveausprung-Detektor, nur als Zeile, nicht übertragbar): Median 339,4, **19,4 %** unter 185,2.
- **Das Mittel der bedingten ARL₀ ist für m ≤ 10 unendlich** (eigene Herleitung, numerisch bestätigt, Quelle nicht gelesen): endlich genau für m ≥ 11 (m = 11: 459.653,3; m = 12: 52.198,3). Das abgeschnittene Integral wächst bei m = 10 von 1,5·10⁸ (s ≤ 5) auf 1,6·10¹⁵ (s ≤ 30) ohne Grenze.
- **Mindestzahl** (kleinstes m mit höchstens ε der Phase-I-Stichproben unter f mal 370,4): halbe ARL₀ mit 95 %: **m = 274**; drei Viertel: **m = 1588**; 0,9: **m = 11.828**. Nur σ geschätzt: **261**, nur μ geschätzt: **21**: der Streuungsfehler überwiegt (Faktor 12) in diesem Kriterium.
  Alle neun Werte (f = ½, ¾, 0,9; ε = 10, 5, 1 %) stehen in der Tabelle unten.
- **Die Grenzen weiten** hat bei m = 100 drei Antworten: Faktor **3,0929** (gleiche mittlere Alarmrate), **2,9414** (Mittel der bedingten ARL₀ = 370,4), **3,1706** (höchstens 5 % unter 185,2). Die Garantie kostet bei bekannten Parametern ARL₁(δ = 1) = **66,7 statt 43,9** Beobachtungen.
  **Wer auf das Mittel eicht (2,9414), macht kurze bedingte ARL₀ häufiger:** der Anteil unter 185,2 steigt von 17,9 % auf **25,7 %**, der Median fällt von 342,0 auf 283,4 (in allen fünf gemessenen Zeilen m = 30 bis 1000 steigt der Anteil und fällt der Median).
- **Kopplung:** je Phase-I-Stichprobe hängen bedingte ARL₀ und bedingte ARL₁ (Aufwärtssprung 1σ) zusammen (Rangkorrelation 0,89 bis 0,90 bei m = 30, 100, 300): wer eine kurze ARL₀ erwischt, erkennt auch schneller; die Schätzung wandert überwiegend auf der Kompromisskurve (bei m = 100 liegen rund 80 % der Streuung von log ARL₁ entlang der Kurve), mit deutlichen Abweichungen bei kleinem m. Zählung, kein Urteil: die Streuung ist die Aussage.
- **CUSUM (einseitig, k = 0,5, h = 4,0965):** bei m = 100 liegen **21,1 %** der Phase-I-Stichproben unter 185,2, bei der einseitigen Shewhart-Karte **15,6 %** (Simulation mit 6000 Phase-I-Stichproben, Standardfehler rund 0,5 Punkte); **in diesem Modell, bei diesen Parametern und diesem Maß ist der CUSUM stärker betroffen.** Eine gelesene Quelle (Zwetsloot/Woodall 2017) nennt für ihre Maße (SDARL) und Einstellungen das Gegenteil: beide Aussagen gelten für ihr Maß.
- **Verunreinigte Phase I (m = 100):** ein Ausreißer +5σ macht die Karte ruhig **und** träge, ohne Meldung (ARL₀ im Median 1122 statt 341, ARL₁ für den Aufwärtssprung 119,5 statt 42,9). Bei einer Stufe von 2σ in der Mitte bringt **weder s noch MR** die ARL₀ in die Nähe des Ziels (s: ARL₀ 1701, MR: 46,1 gegen 370,4, im Zustand der zweiten Hälfte).
- **Wenn die Annahmen nicht gelten** (Tabelle, nicht neu gemessen): bei AR(1)-Phase-I-Werten (φ = 0,9) hat die Karte in 44,0 % der Phase-I-Stichproben eine bedingte ARL₀ unter der Hälfte (m = 100; bei unabhängigen Werten 18,6 % in derselben Simulation); die T²-Karte mit geschätzter Kovarianz und χ²-Grenze hat bei m = 100 die unbedingte ARL₀ **239,6 / 154,7 / 75,0** bei p = 2 / 5 / 10 Kanälen statt 370,4.

## Modell

- **Reihe:** Einzelwerte x₁, x₂, … unabhängig N(μ, σ²), angegeben in Einheiten der wahren Streuung (wahre Reihe N(0, 1), nach einer Änderung N(δ, 1)). **Phase I:** m Beobachtungen in Kontrolle, daraus x̄ und s (Nenner m − 1). **Phase II:** Alarm, wenn |x − x̄| > L·s (L = 3, zweiseitig); die Karte wird nicht neu geschätzt.
- **Eine feste Phase-I-Stichprobe:** p(x̄, s) = Φ(x̄ − L s) + Φ(−x̄ − L s); die Lauflänge bis zum Alarm ist geometrisch mit diesem p, die **bedingte ARL₀** ist 1/p. Über Phase-I-Stichproben: x̄ ~ N(0, 1/m), s² ~ χ²(m − 1)/(m − 1), unabhängig.
- **Vier Maße für „die“ ARL₀:** (1) Verteilung der bedingten ARL₀; (2) ihr Mittel E[1/p]; (3) Kehrwert der mittleren Alarmrate 1/E[p] (geschlossen über die t-Verteilung: die Phase-I-Grenzen sind ein Vorhersageintervall, kein 3σ-Intervall); (4) unbedingte Lauflängenverteilung E[(1 − p)ⁿ].
  Jensen: E[1/p] ≥ 1/E[p]. Streit der Quellen (nur als Streit): Lombard und Hawkins halten gerade die bedingte ARL für „notional“, Gandy/Kvaløy und Diko u. a. garantieren sie mit Wahrscheinlichkeit; beide rechnen richtig.
- **CUSUM:** C_t = max(0, C_{t−1} + (x_t − x̄)/s − k), Alarm bei C_t > h; ARL ab Null per Integralgleichung. Einseitig; Schwellen zu verschiedenen Zielen werden nicht vermischt (h = 4,0965 für 370,398, 4,0954 für 370).
- Alle Formeln stehen im 📐-Expander der App (vollständig für die Einzelwert-Karte; CUSUM-Gleichung für den einseitigen CUSUM nach oben).

## Methodik

- **Exakte Verteilung** (`aps_exact.cdf_carl`): P(bedingte ARL₀ < a) = ∫ 2φ(z)·F_χ²(ν s*(z/√m, a)²) dz, ein Gauss-Legendre-Integral über z mit Bisektion für s* je Knoten (rund 4 ms je Aufruf). Der Integrand ist glatt; das Integral über s hätte einen Wurzelknick bei s₀ (siehe „Befunde und Korrekturen“).
- **Mittel der bedingten ARL₀** (`aps_exact.mean_carl`): 1 + ∫ P(ARL ≥ a) da in ln a (Tail-Weg, bis etwa 5 s); die eingefrorene Tabelle (`aps_exact.json`) stammt aus dem **Doppelintegral** über (x̄, s) (`tools/aps_make_exact.py`, ein anderer Rechenweg; `--check` rechnet neu und vergleicht auf 1e-6 relativ).
  Eingefroren sind außerdem die unbedingte Lauflängenverteilung (m = 30, 100, 500), die Grenzfaktoren L(a), L(b), L(c) (m = 30 bis 1000) und die Tabelle I der Annahmen (aus der Vormessung, Simulation, nicht in diesem Stück neu gemessen).
- **Zufall:** `aps_rng.py` ist die SplitMix64-Quelle der Website (wörtliche Kopie, `tests/test_copies.py`); `aps_scenario.uniforms` rechnet dieselben Ziehungen vektorisiert (bitgleich mit der Klasse, Test), die Normalzahlen per Box-Muller mit numpy (gleich der Klasse auf 1e-12). Phase-I-Stichproben für die Verteilungs-Experimente werden direkt als
  (x̄, s) gezogen (x̄ = Φ⁻¹(u)/√m, s² = χ²-Quantil(u′; m − 1)/(m − 1)); die Verunreinigung braucht Rohnormalzahlen. Zufallsanteile sind Bänder; bitgleiche Werte sind nicht garantiert (die letzte Binärstelle von Φ⁻¹ und χ²-Quantil kann sich zwischen Plattformen unterscheiden).
- **CUSUM** (`aps_cusum.py`): Nyström-Integralgleichung mit 40 Knoten je Phase-I-Stichprobe (die Schwelle h mit 64 Knoten gelöst); zweiter exakter Weg das Brook/Evans-Gitter (400 Zellen), dritter die Simulation (4000 Läufe, zensiert bei 6000 Beobachtungen). Die Gegenprobe der App zeigt alle drei für eine gezogene Stichprobe.
- **Experimente:** (1) eine Phase-I-Stichprobe, eine Karte, dazu die drei ausgedachten Stichproben des Handbeispiels und der Verlauf der letzten 20 Stichproben; (2) die Verteilungsfunktion der bedingten ARL₀, die vier Maße und die Kopplung von ARL₀ und ARL₁; (3) Mindestzahl m (live) und Grenzen weiten;
  (4) CUSUM neben der einseitigen Shewhart-Karte; (5) Schätzer s gegen MR und verunreinigte Phase I (immer m = 100).

## Befunde (gemessen, keine Behauptungen)

Zufalls-Seed 35 (Voreinstellung); Simulationszahlen mit zweitem Zufalls-Seed 2026 als Band. „Test“ nennt die Testfunktion in `tests/test_claims.py` (Band oder exakt).

| Frage | Befund | Test |
|---|---|---|
| m = 100: Verteilung der bedingten ARL₀ (exakt) | Median 342,0; 5 % 117,9; 95 % 1162,5; Anteil unter 185,2: 17,9 % | `test_exact_distribution_m100` |
| vier Maße, m = 100 | 455,9 / 280,0 / 342,0 / unbedingter Median 229,2 | `test_four_measures_m100` |
| vier Maße, m = 30 und 91 | m = 30: 885,8 / 161,1 / 287,5 / 186,7, 36,0 % unter 185,2; m = 91: 339,4, 19,4 % | `test_four_measures_m30_and_m91` |
| Mittel unendlich | endlich genau für m ≥ 11; m = 11: 459.653,3; m = 12: 52.198,3; m = 10: 1,5·10⁸ → 1,6·10¹⁵ (s ≤ 5 → 30) | `test_mean_infinite_below_11` |
| Zerlegung (nur μ / nur σ) | 5 %-Quantil m = 30: 221,6 / 52,5; m = 100: 310,5 / 123,0 | `test_decomposition_quantiles` |
| Mindestzahl m | neun Werte der Tabelle; nur σ 261, nur μ 21 | `test_minimum_m_table` |
| Grenzen weiten | L(a), L(b), L(c), ARL₁-Preise, Anteile (Tabelle) | `test_widening_table_in_readme` |
| Eichen auf das Mittel | Anteil steigt, Median fällt in allen fünf Zeilen; m = 100: 17,9 → 25,7 %, 342,0 → 283,4 | `test_calibrating_the_mean_makes_short_cards_more_frequent` |
| Preset „Grenzen weiten“ | Faktor 3,18: 4,6 % unter 185,2, kleinstes m = 97; Faktor 3,17: 5,03 % (m = 101) | `test_widen_preset_numbers` |
| Kopplung ARL₀/ARL₁ | Rangkorrelation 0,89 bis 0,90; Verhältnis zur zentrierten Karte: Median 0,99 bis 1,00, 5 % bis 95 % m = 30: 0,67 bis 2,78, m = 100: 0,75 bis 1,64, m = 300: 0,83 bis 1,30 (Band) | `test_coupling_band` |
| CUSUM gegen Shewhart einseitig | Anteil unter 185,2 (CUSUM / Shewhart), m = 30: 35,5 / 31,3 %; 100: 21,1 / 15,6 %; 300: 7,2 / 3,2 %; 1000: 0,4 / 0,03 % (6000 Phase-I-Stichproben, Band) | `test_cusum_against_shewhart_band` |
| CUSUM-Schwelle und Gegenprobe | h = 4,0965 (Ziel 370,398), 4,0954 (Ziel 370); drei Wege für eine Stichprobe | `test_cusum_threshold_and_check` |
| Schätzer s gegen MR | Anteil unter 185,2, m = 30 / 100 / 300: s 35,7 / 18,0 / 4,2 %, MR 37,8 / 23,2 / 8,5 %; Streuung von σ̂ (mal √(2(m−1))): s 0,99 bis 1,00, MR 1,27 bis 1,28 | `test_estimators_s_against_mr` |
| Verunreinigte Phase I | Tabelle unten (m = 100, 40.000 Phase-I-Stichproben, Band) | `test_contamination_table_band` |
| Handbeispiel (m = 3) | ARL₀ 7,48 / 5,51 / 5,07·10⁸; 1/E[p] = 8,22; s < 1 mit 63,2 % | `test_hand_samples` |
| Annahmen: Tabelle I und Zeile J | 44,0 % und 18,6 % (φ = 0,9 und 0) bei m = 100; T²: 239,6 / 154,7 / 75,0, Grenzen der F-Regel 12,83 / 20,77 / 33,19 | `test_assumption_rows` |
| Literatur | FAP einer Phase-I-Karte mit bekannten Parametern 0,0779 bei m = 30 und 0,1264 bei m = 50 Kartenpunkten (Jones-Farmer u. a. 2014: 0,078 und 0,126) | `tests/test_literatur_phase1.py` |

### Mindestzahl m (exakt; kleinstes m mit P(bedingte ARL₀ < f · 370,4) ≤ ε)

| f | ε = 10 % | ε = 5 % | ε = 1 % |
|---|---|---|---|
| ½ | 175 | **274** | 525 |
| ¾ | 986 | **1588** | 3119 |
| 0,9 | 7241 | **11.828** | 23.504 |

### Grenzen weiten (Ziel 370,398; ARL₁(δ = 1) bei bekannten Parametern mit diesem Faktor: ein Preis, nicht die ARL₁ der geschätzten Karte)

| m | L(a) gleiche Alarmrate | L(b) Mittel = 370,4 | L(c) Garantie 5 % | ARL₁ bei L = 3 | bei L(a) | bei L(b) | bei L(c) | Anteil < 185,2 bei L = 3 | bei L(b) | bei L(c) |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 3,3347 | 2,8019 | 3,6222 | 43,89 | 102,16 | 27,89 | 228,81 | 36,0 % | 55,8 % | 5,0 % |
| 50 | 3,1920 | 2,8819 | 3,3790 | 43,89 | 70,40 | 33,36 | 115,12 | 28,5 % | 42,7 % | 5,0 % |
| 100 | 3,0929 | 2,9414 | 3,1706 | 43,89 | 54,94 | 38,25 | 66,69 | 17,9 % | 25,7 % | 5,0 % |
| 300 | 3,0303 | 2,9806 | 2,9891 | 43,89 | 47,19 | 41,93 | 42,78 | 4,2 % | 5,7 % | 5,0 % |
| 1000 | 3,0090 | 2,9942 | 2,8903 | 43,89 | 44,85 | 43,30 | 34,00 | 0,1 % | 0,1 % | 5,0 % |

Ab m = 274 liegt L(c) unter 3 (die Garantie gilt dann schon ohne Weiten; m = 300: 2,9891).

### Verunreinigte Phase I (m = 100; Median der bedingten ARL₀ / ARL₁ Aufwärtssprung 1 / ARL₁ Abwärtssprung 1; Phase II im Zustand der zweiten Hälfte bei der Stufe, sonst ohne Störung)

| Fall | Schätzer s | Schätzer MR |
|---|---|---|
| ohne Störung | 340,9 / 42,9 / 42,8 | 346,9 / 43,4 / 43,3 |
| ein Ausreißer +5σ | 1121,8 / 119,5 / 91,3 | 686,7 / 82,2 / 63,5 |
| Stufe 1σ in der Mitte | 439,3 / 30,9 / 436,5 | 154,8 / 15,0 / 154,2 |
| Stufe 2σ in der Mitte | 1701,0 / 80,7 / 41.716,4 | 46,1 / 6,5 / 378,7 |

Bei der Stufe 2σ macht s die Karte in der einen Richtung taub (ARL₀ über das Vierfache der Ziel-ARL₀), MR zentriert falsch und alarmiert oft (ARL₀ unter einem Siebtel). **Weder s noch MR bringt die ARL₀ in die Nähe des Ziels** (1701 bzw. 46 gegen 370): das gilt hier nur für diese Stufe, m = 100 und diese beiden Schätzer. Phase I ist selbst eine Suche nach Stufen: Segmentierung (Stück 7), hier nicht gerechnet.

## Befunde und Korrekturen gegenüber Plan und Vormessung

- **Mindestzahl:** das grobe Gitter der ersten Vormessung (300 und 2000) ist nach unten präzisiert: exakt 274 und 1588. Die erste exakte Fassung hatte für ¾ und 0,9 falsche Werte (1593, 11.253): ihr Integral über s lief über einen Wurzelknick und war ab m ≈ 1000 um bis 10⁻³ ungenau. Der Bau rechnet das Integral über x̄ (glatt) und prüft den Rand mit dem zweiten Weg (Integral über s mit Substitution) und mit 10⁷ direkt gezogenen (x̄, s).
- **Zufall:** der Plan sah für Experiment 1 `normal()` der Vorlage vor. Gebaut: ein vektorisierter Strom (bitgleich mit der Klasse) mit Box-Muller in numpy (gleich der Klasse auf 1e-12), weil die Historie von 20 Stichproben und m bis 5000 mit `decimal` zu langsam wären. Eine Zufallsziehung kann sich daher zwischen Plattformen in der letzten Stelle unterscheiden: Anteile sind Bänder.
- **Störung als eine Auswahl** statt Art plus Größe: ein getrennter Größen-Regler wäre bei „keine“ ein toter Regler. Die Auswahl enthält Ausreißer +3σ, +5σ, +10σ und Stufen 0,5σ bis 3σ; Experiment 5 zeigt die vier festen Fälle der Vormessung immer mit.
- **Preset „Grenzen weiten“** hat den Faktor 3,18, nicht 3,17: mit zwei Stellen gerundet gibt 3,17 (L(c) = 3,1706) 5,03 % und verfehlt das Garantie-Kriterium knapp.
- **Mittel der bedingten ARL₀ live:** der Plan sah nur die eingefrorene Tabelle vor. Gebaut: Tabelle für L = 3 und die Schieberwerte, sonst live über den Tail-Weg (bis etwa 5 s); beide Wege stimmen auf 1e-7 relativ überein (Test). Das Neurechnen der Tabelle dauert rund 11 Sekunden (ein Thread), nicht fünf Minuten.
- **Zweiter Zufalls-Seed der Simulationszahlen** (Plan E 11): Kopplung, CUSUM, Schätzer und Verunreinigung sind mit den Zufalls-Seeds 35 und 2026 gemessen. Verschoben gegenüber der Vormessung (Zufalls-Seed 13601 usw.): CUSUM-Median bei m = 100: 349,2 (Zufalls-Seed 35) und 360,4 (Zufalls-Seed 2026) gegen 357,5 der Vormessung, die Mediane der zwei Zufalls-Seeds
  unterscheiden sich um 11; der Anteil unter 185,2 blieb bei 21,1 und 20,4 % (Vormessung 21,0 %). Ausreißer +5σ, ARL₀ s: 1121,8 und 1122,6 gegen 1129,0; Stufe 2σ, ARL₁ abwärts (s): 41.716 und 41.606 gegen 41.914; Rangkorrelation 0,893 bis 0,901 gegen 0,892 bis 0,896. Bei diesen Simulationszahlen (Kopplung, CUSUM, Schätzer, Verunreinigung) hat sich keine Richtung der Vormessung umgekehrt.
- **Preis der F-Regel und Mindestzahl der T²-Karte** (Vormessung J) sind in diesem Stück nicht gerechnet; die unbedingte ARL₀ der χ²-Regel und die Grenzen der F-Regel sind exakt (F-Verteilung).

## Ehrliche Grenzen

- Nur **Einzelwerte, unabhängig und normalverteilt**, Phase I in Kontrolle, **eine** Reihe; die Reihen sind ausgedacht. Die Zahlen sind keine Aussage über Logistik, Lager oder Fertigung.
- **Nicht gemessen:** schiefe Phase-I-Werte; Untergruppen (n > 1) und X̄-Karten mit c₄; EWMA und Shiryaev–Roberts mit geschätzten Parametern; zweiseitiger CUSUM exakt (nur eine Näherung in der Vormessung); Verzögerungen (ARL₁) der geschätzten Karte (nur der Preis bei bekannten Parametern);
  Self-Starting-Karten und Bootstrap-Grenzen nach Gandy/Kvaløy (nur als Quellenaussage); Wirkung auf Betriebsentscheidungen. Tabelle I (autokorrelierte Phase I) und die T²-Zeile sind Verweise auf das Schwesterstück, nicht neu gemessen.
- **Die Quellen:** Jensen u. a. 2006 und Keefe/Woodall/Jones-Farmer 2015 sind **nicht gelesen** (ihre Position steht nur über die gelesenen Arbeiten fest), Zwetsloot/Woodall 2017 nur im Textteil (Tabelle 1 nicht lesbar). Die Mindestzahlen sind **eigene Rechnung in diesem Modell**, keine Quellenzahl.
- In diesem Modell (Shewhart-Einzelkarte, Aufwärtssprung 1σ) heißt eine kurze bedingte ARL₀ nicht, dass die Karte schlechter erkennt: die Kopplung zeigt, dass kurze ARL₀ und kurze ARL₁ zusammengehen. Welche Verteilung der Anwender erlebt (eine Phase I, oder viele), ist ein Streit der Quellen, kein Ergebnis dieser Demo.

## Quellen

Stufe: **W** = gelesen, sonst „nicht gelesen“.

- Gandy/Kvaløy 2013, *Guaranteed conditional performance of control charts via bootstrap methods*, Scandinavian Journal of Statistics 40, arXiv 1111.4180 (W: Einleitung, Abschnitte 2 und 4).
- Diko/Chakraborti/Does 2019, *Guaranteed in-control performance of the EWMA chart for monitoring the mean*, Quality and Reliability Engineering International 35, 1144–1160 (W: Einleitung, Abschnitt 3).
- Zwetsloot/Woodall 2017, *A head-to-head comparative study of the conditional performance of control charts based on estimated parameters*, Quality Engineering 29 (W: Textteil; Tabelle 1 nicht lesbar).
- Does/Goedhart/Woodall 2020, *On the design of control charts with guaranteed conditional performance under estimated parameters*, Quality and Reliability Engineering International 36, 2610–2620 (W).
- Jones-Farmer/Woodall/Steiner/Champ 2014, *An overview of Phase I analysis for process improvement and monitoring*, Journal of Quality Technology 46, 265–280 (W: Abschnitte 1 bis 3).
- Lombard/Hawkins, *CUSUM average run length: conditional or unconditional?*, arXiv 1904.00340 (W).
- Nicht gelesen: Jensen/Jones-Farmer/Champ/Woodall 2006; Keefe/Woodall/Jones-Farmer 2015; Brook/Evans 1972 (das Gitter ist nach der in der Vormessung beschriebenen Konstruktion gebaut: Markov-Kette auf Zellen).

## Tests

**212 Tests** in 18 Dateien; die ganze Suite läuft seriell in **220 s** (Windows, Python 3.14.6, ein Thread, unter paralleler Rechnerlast), mit `pytest -n 8` in 71 bis 151 s (je nach paralleler Last). Die schwersten Tests sind die AppTest-Läufe (jeder Button und jeder Regler an beiden Rändern, bis zu 25 s) und die Gegenprobe des Mittels der bedingten ARL₀ (bis 11 s).

| Art | Tests | Datei | Was geprüft wird |
|---|---|---|---|
| Handbeispiele | 14 | `test_handbeispiele.py` | H 1 bis H 6 des Plans exakt: Brüche, Tafelwerte, Student-t, Doppelintegral, Hand-Herleitung der Unendlichkeits-Grenze, Markov-Kette in Brüchen gegen die Löser der Demo |
| Orakel (anderer Rechenweg) | 44 | `test_oracle_verteilung.py` (31), `test_oracle_mass.py` (5), `test_oracle_cusum.py` (6), `test_literatur_phase1.py` (2) | Mittlere Alarmrate geschlossen gegen Doppelintegral; `cdf_carl` gegen Simulation aus Rohnormalzahlen und gegen den zweiten exakten Weg; Zerlegung; Mindestzahl (neun Werte, beide Wege, 10⁷ direkt gezogene (x̄, s)); geometrische und unbedingte Lauflänge gegen Simulation; T² gegen Wishart-Simulation; die Maße als gewichtete Summen; CUSUM: Integralgleichung gegen Brook/Evans-Gitter, Simulation, Überlebenswahrscheinlichkeit; Literaturwerte (Stufe W) |
| Eigenschaften | 11 | `test_eigenschaften_phase1.py` | s permutationsinvariant, MR nicht; affine und Spiegel-Invarianz; Monotonie in L, m und ε; Jensen; Rangkorrelation; Verdikte über zufällige Einstellungen |
| Randfälle | 13 | `test_edge_cases.py` | m = 0, 1, 2; m ≤ 10; alle Werte gleich; Karte ohne Alarm; m = 5000; L unter 1; Zufalls-Seed 0 und Maximum; MR bei m = 2; unerreichbare Mindestzahl; Störungen |
| Claims (Zahl → Test) | 18 | `test_claims.py` | jede Zahl der README (exakt, oder Band mit zwei Zufalls-Seeds) und ihr Vorkommen in der README |
| Verdikte | 17 | `test_verdicts.py` | jede Schwelle und jede Zahl im Text an handgesetzten Werten |
| Zufall | 12 + 2 | `test_rng.py`, `test_copies.py` | vektorisierter Strom bitgleich mit SplitMix64; Normalzahlen; direkte (x̄, s) gegen Rohnormalzahlen; die kopierte Quelle |
| Eingefrorene Tabellen | 10 | `test_reference.py` | Aufbau, Quelle und Stand je Feld, Querprüfung gegen die Live-Rechnung, schnelle Neurechnung des Skripts |
| Oberfläche | 36 | `test_app.py` (3), `test_app_presets.py` (33) | AppTest: jeder Button, jeder Regler an beiden Rändern, alle Ränder zusammen, je Preset die erwartete Verdikt-Zeile, Permalink, Expander |
| Layout, Presets, Footer, Laufzeit-Importe | 23 + 9 + 2 + 1 | `test_layout.py`, `test_presets.py`, `test_footer.py`, `test_runtime_imports.py` | Schnellstart in der Seitenleiste, aktives Preset hervorgehoben, jede Reglerwahl hebt die Markierung auf; Presets vollständig, eindeutig und innerhalb der Grenzen; Footer; jede Laufzeit-Bibliothek in `requirements.txt` |

**Live-Rechenzeiten** (Messung dieser README, nicht im Test; ein Thread): Experiment 1 mit der Historie von 20 Stichproben 2 bis 4 ms (m = 100 bis 5000); `cdf_carl` 5 ms je Aufruf; Experiment 2 (Verteilungsfunktion, Quantile, 4000 Stichproben für die Kopplung) 1,2 s, mit dem Mittel live bei L ≠ 3 bis 2,5 s,
beim Tail-Weg für m = 11 5,9 s; Experiment 3 0,4 s (die Tabelle „Grenzen weiten“ mit fünf Zeilen 1,4 s); Experiment 4 mit 4000 Integralgleichungen 0,2 s (10.000: 0,4 s), die Gegenprobe 0,6 s; Experiment 5 mit 4000 Stichproben à 100 Werten 0,05 s (10.000: 0,12 s). Der kalte Lauf der ganzen Seite im AppTest dauert 5 bis 6 s.
Die App zeigt für die Experimente 2 bis 4 einen Spinner mit Zeitangabe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | venv, Caches, Secrets; Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2: Kopf, Schnellstart, Ergebnis, Kernfragen, fünf Experimente, Annahmen, Expander) |
| `aps_exact.py` | Kern: bedingte ARL₀, exakte Verteilung, Mittel, Mindestzahl, Grenzfaktoren, T²-Zeile, Summenform der Maße |
| `aps_cusum.py` | einseitiger CUSUM: Integralgleichung, Brook/Evans-Gitter, Simulation |
| `aps_scenario.py` | Zufallsstrom (vektorisiert), Phase-I-Stichproben, Schätzer s und MR, Störungen |
| `aps_rng.py` | SplitMix64 (wörtliche Kopie der Website-Vorlage) |
| `aps_evaluation.py` | Einstellungen, Auswertung der Experimente, alle Verdikte als reine Funktionen |
| `aps_reference.py`, `aps_exact.json` | eingefrorene Tabellen mit Quelle und Stand je Feld |
| `aps_constants.py`, `aps_presets.py`, `aps_layout.py`, `aps_visualization.py`, `aps_texts.py` | Regler und Presets, Permalink, fester Kopf, Plotly-Abbildungen, lange Texte |
| `tools/aps_make_exact.py` | rechnet `aps_exact.json` neu (`--write`) oder prüft sie (`--check`, `--quick`) |
| `tests/` | siehe Abschnitt Tests |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, plotly) und pytest |

## Bewusst nicht umgesetzt

- **Kein PDF-Export** (der Plan nennt keinen).
- **Keine Bootstrap-Grenzen, keine Self-Starting-Karte, keine Untergruppen:** nur als Aussage der Quellen oder in der Liste der Grenzen.
- **Keine Wiederholung der Rechnung zu abhängigen Werten und mehreren Kanälen:** Tabelle I und Zeile J sind Verweise; die Prüfung steht im Schwesterstück.
- **Kein Import aus anderen Demos**; der Niveausprung-Detektor (`niveausprung-demo`) hat ein anderes Kennmaß (Fehlmeldungen je Jahr, Verzögerung in Tagen, σ aus einem festen Referenzfenster) und wird nur genannt. Das `forecast-interval-demo` zeigt dieselbe Endlichkeitsfrage für ein Quantil.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Cloud-Einstellung der Python-Version steht im Streamlit-Dashboard (Tests laufen unter Python 3.12).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html).
