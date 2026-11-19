"""Längere Texte der App (Einleitung, Verfahren, Annahmen, Formeln, Konventionen, Quellen). Die README wiederholt die Zahlen, nicht diese Texte.

Im Formelblock steht jede Display-Gleichung in einem eigenen $$-Block (KaTeX), Prozentzeichen als \\%."""

INTRO_MD = """
Stück der Linie **„Änderungserkennung“**. Die Zahl **„ARL₀ = 370“** gehört zu einer Karte mit **bekannten** Parametern. Hier schätzt die Karte Mittelwert und Streuung aus **m Beobachtungen einer Phase I**
und baut daraus ihre Grenzen x̄ ± 3 s: dann ist ihre Zeit bis zum ersten Fehlalarm selbst eine **Zufallsgröße**, die an der Phase-I-Stichprobe hängt. Gezählt werden **Beobachtungen bis zum Alarm**
(ARL, *Average Run Length*) und **Phase-I-Stichproben** (Wiederholungen des Schätzens); die Messreihen sind **ausgedacht** (unabhängige, normalverteilte Einzelwerte), keine Betriebsdaten.

Weiter unten: fünf Experimente, der Abschnitt **🚧 Wo die Annahmen enden** und die Expander **❓ Und wenn die Annahmen nicht gelten?** und **📐 Mathematische Formulierung**.
"""

HOW_MD = """
1. **Phase I:** Die Karte sieht m Beobachtungen des Prozesses in Kontrolle und rechnet daraus das Mittel x̄ und die Standardabweichung s (Nenner m − 1).
2. **Phase II:** Neue Beobachtungen werden gegen die Grenzen x̄ ± L·s geprüft (L = 3, zweiseitig, Einzelwerte); die Karte wird nicht neu geschätzt. Für **eine feste Phase-I-Stichprobe** ist die Lauflänge bis zum Alarm
   geometrisch mit der Alarmwahrscheinlichkeit p = Φ(x̄ − L·s) + Φ(−x̄ − L·s); die **bedingte ARL₀** ist 1/p.
3. **Warum „die“ ARL₀ eine Verteilung ist:** x̄ und s sind Zufallsgrößen (x̄ ~ N(0, 1/m), s² ~ χ²(m − 1)/(m − 1), unabhängig), p wandert mit ihnen. Welche „mittlere“ ARL₀ man nennt, ändert die Zahl
   (Mittel der bedingten, Kehrwert der mittleren Alarmrate, Median, unbedingte Lauflänge): zwei verschiedene Fragen haben hier zwei verschiedene Antworten.
"""

ASSUMPTIONS_MD = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Einzelwerte, unabhängig, normalverteilt** (auch in Phase I) | Bei autokorrelierten Phase-I-Werten (AR(1)) nimmt der Anteil der Karten mit bedingter ARL₀ unter der Hälfte des Ziels zu: m = 100, φ = 0,9: 44,0 % statt 18,6 % bei unabhängigen Werten (gleiche Simulation; Tabelle im Expander, nicht in diesem Stück neu gemessen). Schiefe Phase-I-Werte: nicht neu gemessen. | Autokorrelierte Phase I: Stück 4b der Linie; abhängige Werte allgemein: Stück 5 |
| **Phase I ist in Kontrolle** | Ein Ausreißer oder eine Stufe in Phase I verunreinigt x̄ und s: Experiment 5. Bei einer Stufe von 2σ in der Mitte (m = 100) bringt weder s noch MR die ARL₀ in die Nähe des Ziels. | Die Stufe finden: Segmentierung (Stück 7) |
| **Ein Kanal** | Mit mehreren Kanälen und geschätzter Kovarianz fällt die unbedingte ARL₀ der T²-Karte nach der χ²-Regel stark (Zeile J im Expander, exakt gerechnet). | Stück 4b und Stück 6 |
| **Einzelwerte, keine Untergruppen** | X̄-Karten mit Untergruppen (n > 1) und c₄-Schätzer sind nicht gerechnet. | – |
| **Shewhart und einseitiger CUSUM** | EWMA und Shiryaev–Roberts mit geschätzten Parametern, zweiseitiger CUSUM (nur Näherung in der Vormessung) und Verzögerungen (ARL₁) der geschätzten Karte sind nicht gemessen. | – |
| **Keine reale Datenquelle** | Die Reihen sind ausgedacht; in den gelesenen Quellen gibt es keine reale Phase-I-Stichprobe aus Logistik, Lager oder Fertigung mit bekannter Wahrheit. Was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet. | – |
"""

CONVENTIONS_MD = """
| Größe | Bedeutung in diesem Stück |
|---|---|
| **Lauflänge** | immer „Lauflänge bis zum Alarm“ in **Beobachtungen**; nicht ein Simulationslauf. Zähleinheiten werden getrennt genannt: Beobachtungen (bis zum Alarm), Phase-I-Stichproben (Wiederholungen des Schätzens), Läufe (Simulation). |
| **ARL₀-Ziel** | 370,398 = 1/p der 3σ-Karte (zweiseitig); nicht das gesetzte 370. Einseitiger CUSUM k = 0,5: h = 4,0965 für 370,398 (4,0954 für 370). |
| **bedingte ARL₀** | 1/p für eine feste Phase-I-Stichprobe (die Karte, die man gebaut hat). Ihre Verteilung über Phase-I-Stichproben ist die Aussage. |
| **vier Maße** | (1) Verteilung der bedingten ARL₀ (Median, Quantile, Anteil unter einer Schwelle); (2) Mittel der bedingten ARL₀, E[1/p]; (3) Kehrwert der mittleren Alarmrate 1/E[p] (viele Karten gleichzeitig); (4) unbedingte Lauflängenverteilung E[(1 − p)ⁿ] (jede Lauflänge mit frischer Phase I). Jensen: E[1/p] ≥ 1/E[p]. |
| **Fehlalarm** | Alarm vor der Änderung, je Zeit gezählt (ARL₀), nicht der Anteil falscher unter positiven. |
| **ν** | die letzte Beobachtung **vor** der Änderung; „Änderung vor der ersten Beobachtung“ ist ν = 0, die Alarmbeobachtung zählt mit. Die Shewhart-Karte hat kein Gedächtnis: Lorden-, Pollak- und Shiryaev-Maß fallen zusammen, zero-state gilt für jedes ν. |
"""

FORMULAS_MD = r"""
**Lesehilfe.** Gegeben: eine Phase-I-Stichprobe $x_1, \dots, x_m$ aus dem Prozess in Kontrolle und der Grenzfaktor $L$. Gesucht: die Verteilung der Lauflänge bis zum Alarm in Phase II und, über Phase-I-Stichproben, ihre Streuung.

| Symbol | Bedeutung |
|---|---|
| $m$, $\nu = m - 1$ | Zahl der Phase-I-Beobachtungen, Freiheitsgrade der Streuungsschätzung |
| $\bar x$, $s$ | Mittel und Stichproben-Standardabweichung der Phase I |
| $L$ | Grenzfaktor; Grenzen $\bar x \pm L s$ (Voreinstellung $L = 3$, zweiseitig) |
| $p(\bar x, s)$ | Alarmwahrscheinlichkeit je Beobachtung der Karte mit diesen Grenzen |
| $N$ | Lauflänge bis zum Alarm, in Beobachtungen |
| $a$ | Schwelle für die bedingte ARL₀ |
| $s^*(\bar x, a)$ | der Wert von $s$ mit $p(\bar x, s^*) = 1/a$ |
| $f$, $\varepsilon$ | Schwellenanteil (der Ziel-ARL₀) und zulässiger Anteil der Phase-I-Stichproben |
| $k$, $h$ | Referenzwert und Schwelle des CUSUM |
| $\delta$ | Sprunggröße in Einheiten der wahren Streuung |

**Modell.** Die Beobachtungen sind unabhängig $N(0, 1)$ (standardisiert; nach einer Änderung $N(\delta, 1)$), die Karte kennt Mittel und Streuung nicht. Phase I:

$$\bar x = \frac{1}{m} \sum_{i=1}^{m} x_i, \qquad s^2 = \frac{1}{m-1} \sum_{i=1}^{m} (x_i - \bar x)^2$$

$$\bar x \sim N\!\left(0, \tfrac{1}{m}\right), \qquad \nu s^2 \sim \chi^2_{\nu}, \qquad \bar x \text{ und } s \text{ unabhängig}$$

**Eine feste Phase-I-Stichprobe.** Alarm bei $|X - \bar x| > L s$; die wahre Beobachtung $X \sim N(\delta, 1)$ hat die Alarmwahrscheinlichkeit

$$p(\bar x, s) = \Phi(\bar x - L s - \delta) + \Phi(\delta - \bar x - L s)$$

Die Lauflänge ist geometrisch, die bedingte ARL ist ihr Mittel:

$$P(N > n \mid \bar x, s) = (1 - p)^n, \qquad \mathrm{ARL}(\bar x, s) = \frac{1}{p}$$

**Vier Maße für „die“ ARL₀** ($\delta = 0$). Verteilungsfunktion, Mittel der bedingten ARL₀, Kehrwert der mittleren Alarmrate, unbedingte Lauflänge:

$$M_1(a) = P\!\left(\tfrac{1}{p} < a\right)$$

$$M_2 = E\!\left[\tfrac{1}{p}\right]$$

$$M_3 = \frac{1}{E[p]}$$

$$M_4(n) = P(N > n) = E\!\left[(1 - p)^n\right]$$

Mittlere Alarmrate geschlossen: $(X - \bar x)/(s\sqrt{1 + 1/m})$ ist $t_{m-1}$-verteilt, die Phase-I-Grenzen sind also ein **Vorhersageintervall**, kein $3\sigma$-Intervall:

$$E[p] = 2\, P\!\left(t_{m-1} > \frac{L}{\sqrt{1 + 1/m}}\right)$$

**Die exakte Verteilung der bedingten ARL₀.** Für festes $\bar x$ fällt $p$ in $s$; $p > 1/a$ heißt $s < s^*(\bar x, a)$. Mit $z = \sqrt{m}\,\bar x$:

$$P\!\left(\tfrac{1}{p} < a\right) = \int_{-\infty}^{\infty} \varphi(z)\, F_{\chi^2_\nu}\!\left(\nu\, s^*\!\left(z/\sqrt{m}, a\right)^2\right) dz$$

Nur $\sigma$ geschätzt ($\bar x = 0$, $s_0 = -\Phi^{-1}(1/(2a))/L$) und nur $\mu$ geschätzt ($s = 1$, $g$ mit $p(g, 1) = 1/a$):

$$P_\sigma = F_{\chi^2_\nu}\!\left(\nu s_0^2\right), \qquad P_\mu = 2\,\Phi\!\left(-\sqrt{m}\, g\right)$$

**Wann ist das Mittel der bedingten ARL₀ unendlich?** $1/p$ wächst wie $\exp\!\big((L s - |\bar x|)^2/2\big)$, die gemeinsame Dichte fällt wie $\exp\!\big(-\nu s^2/2 - m \bar x^2/2\big)$. Der Kreuzterm $-L s |\bar x|$ ist für $s, |\bar x| \ge 0$ negativ; maßgeblich ist der Rand $\bar x = 0$, an dem $s^2$ im Exponenten den Koeffizienten hat:

$$\frac{L^2 - (m - 1)}{2} < 0 \iff m > L^2 + 1$$

Für $L = 3$ ist $E[1/p]$ also genau für $m \ge 11$ endlich (eigene Herleitung, numerisch bestätigt; Quelle nicht gelesen).

**Mindestzahl und weitere Grenzen.** Das kleinste $m$ mit höchstens dem Anteil $\varepsilon$ unter dem Bruchteil $f$ der Ziel-ARL₀:

$$m_{\min} = \min\left\{ m : P\!\left(\tfrac{1}{p} < f \cdot 370{,}398\right) \le \varepsilon \right\}$$

Grenzfaktor $L_a$ mit gleicher mittlerer Alarmrate $1/370{,}398$ (geschlossen; $p_0 = 2\Phi(-3)$); $L_b$ löst $E[1/p] = 370{,}398$, $L_c$ löst $P(1/p < 185{,}2) = 5\,\%$ (Zahlenwerte der Tabelle: Experiment 3):

$$L_a = \sqrt{1 + \tfrac{1}{m}}\; t_{m-1;\, 1 - p_0/2}$$

**Schätzer der Streuung.** Neben $s$ die mittlere gleitende Spannweite ($d_2 = 2/\sqrt{\pi}$):

$$\hat\sigma_{MR} = \frac{1}{d_2} \cdot \frac{1}{m - 1} \sum_{i=1}^{m-1} |x_{i+1} - x_i|$$

**Einseitiger CUSUM mit geschätzten Parametern.** Mit $z_t = (x_t - \bar x)/s$ und Alarm bei $C_t > h$:

$$C_t = \max\!\left(0,\; C_{t-1} + z_t - k\right), \qquad C_0 = 0$$

Die ARL $\mathcal{L}(u)$ ab dem Zustand $u \in [0, h]$ löst die Integralgleichung zweiter Art ($z \sim N(\mu, \sigma^2)$ mit $\mu = -\bar x / s$, $\sigma = 1/s$; Nyström mit Gauss-Legendre, zweiter Weg: das Gitter von Brook und Evans):

$$\mathcal{L}(u) = 1 + \mathcal{L}(0)\, \Phi\!\left(\frac{k - u - \mu}{\sigma}\right) + \int_0^h \mathcal{L}(v)\, \frac{1}{\sigma}\, \varphi\!\left(\frac{v + k - u - \mu}{\sigma}\right) dv$$

**Vollständigkeit.** Der Block ist für die Einzelwert-Karte vollständig (Verteilung, Alarmwahrscheinlichkeit, vier Maße, exakte Verteilung, Mindestzahl). Der CUSUM-Teil gilt für den einseitigen CUSUM nach oben mit Start null; die Zeile zur T²-Karte (Expander „Und wenn die Annahmen nicht gelten?“) ist nur eine Skizze der Verteilung, ihre Herleitung steht nicht hier.

**Literatur (Stufe: W = gelesen, sonst nicht gelesen).**
- Gandy/Kvaløy 2013, *Guaranteed conditional performance of control charts via bootstrap methods*, Scandinavian Journal of Statistics 40, arXiv 1111.4180 (W: Einleitung, Abschnitte 2 und 4).
- Diko/Chakraborti/Does 2019, *Guaranteed in-control performance of the EWMA chart for monitoring the mean*, QREI 35, 1144–1160 (W: Einleitung, Abschnitt 3).
- Zwetsloot/Woodall 2017, *A head-to-head comparative study of the conditional performance of control charts based on estimated parameters*, Quality Engineering 29 (W: Textteil; die Zahlen der Tabelle 1 sind nicht lesbar).
- Does/Goedhart/Woodall 2020, *On the design of control charts with guaranteed conditional performance under estimated parameters*, QREI 36, 2610–2620 (W).
- Jones-Farmer/Woodall/Steiner/Champ 2014, *An overview of Phase I analysis for process improvement and monitoring*, Journal of Quality Technology 46, 265–280 (W: Abschnitte 1 bis 3).
- Lombard/Hawkins, *CUSUM average run length: conditional or unconditional?*, arXiv 1904.00340 (W). Keefe/Woodall/Jones-Farmer 2015 und Jensen/Jones-Farmer/Champ/Woodall 2006: nicht gelesen; ihre Position steht nur über die gelesenen Arbeiten fest, Zahlen daraus werden hier nicht verwendet.
"""

FOLLOWUP_MD = """
Die Zahlen dieser Seite gelten für **unabhängige** normalverteilte Einzelwerte in Phase I und II. Die beiden Tabellen zeigen, was beim Aufheben der Annahmen passiert; sie sind hier **Verweis**, die Rechnung und ihre Prüfung stehen in Stück 4b der Linie (Phase I bei abhängigen Werten und mehreren Kanälen).
"""

TABLE_I_NOTE = (
    "**Tabelle I (eingefroren, Simulation, nicht in diesem Stück neu gemessen).** Phase-I-Werte aus einer AR(1)-Reihe mit Randverteilung N(0, 1) und Autokorrelation φ; die Karte baut ihre Grenzen x̄ ± 3 s "
    "**als wären die Werte unabhängig** (Rohdatenkarte). Gezeigt: Anteil der Phase-I-Stichproben mit bedingter ARL₀ (auf der wahren AR(1)-Reihe, exakt gerechnet) unter der Hälfte des Ziels, und in Klammern "
    "der Median der bedingten ARL₀. 4000 Phase-I-Stichproben je Zeile (Standardfehler der Anteile höchstens 0,008); Zählung: Phase-I-Stichproben, ARL in Beobachtungen bis zum Alarm."
)

TABLE_J_NOTE = (
    "**Zeile J (exakt gerechnet, F-Verteilung).** T²-Karte mit p Kanälen, Phase I aus m unabhängigen Vektoren, geschätztes Mittel **und** geschätzte Kovarianz. Gezeigt: die unbedingte ARL₀ "
    "(Kehrwert der mittleren Alarmrate je Phase-II-Vektor) bei der **χ²-Regel** (Grenze = χ²-Quantil, gilt für bekanntes Σ mit ARL₀ 370,398) und die Grenze der **F-Regel** (so, dass die unbedingte ARL₀ 370,398 ist). "
    "T² = [p (m + 1)(m − 1) / (m (m − p))]·F(p, m − p) (eigene Herleitung aus dem Standardsatz zu z′S⁻¹z, Quelle nicht gelesen; Simulation aus Rohnormalzahlen bestätigt sie in der Vormessung). "
    "p = 1 ist die t-Formel von oben. Die Verteilung der bedingten ARL₀ der T²-Karte und der Preis der F-Regel bei Verzögerungen stehen in Stück 4b."
)
