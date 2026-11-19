"""Parameter schätzen: wie viel Phase I braucht eine Karte? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Linie „Änderungserkennung“: eine Karte, die ihre Grenzen aus m Beobachtungen einer Phase I schätzt. Die ARL₀ ist dann eine Verteilung über Phase-I-Stichproben.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2). Rechnung: aps_exact.py, aps_cusum.py, aps_scenario.py; Verdikte: aps_evaluation.py; Texte: aps_texts.py.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import aps_constants as C
import aps_evaluation as ev
import aps_exact as E
import aps_reference as R
import aps_texts as T
from aps_evaluation import Settings, de, pct, sci
from aps_layout import dunkles_design, kopf_css
from aps_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    next_sample,
    sidebar_presets,
    sync_query_params,
)
from aps_visualization import (
    build_cdf,
    build_contamination,
    build_coupling,
    build_cusum,
    build_history,
    build_minimum,
    build_sample,
    case_label,
)

st.set_page_config(page_title="Parameter schätzen – Sebastian Hanisch", layout="wide")


# Rechnungen je Experiment, zwischengespeichert nach den Reglern, die sie wirklich brauchen (die übrigen stehen auf ihrem Standardwert).
@st.cache_data(show_spinner=False)
def _sample(m, seed, estimator, L):
    return ev.analyse_sample(Settings(m=m, seed=seed, estimator=estimator, L=L))


@st.cache_data(show_spinner=False)
def _hand(L):
    return ev.hand_analysis(L)


@st.cache_data(show_spinner=False)
def _spread(m, seed, L, zerl, f, eps, n_stich):
    return ev.analyse_spread(Settings(m=m, seed=seed, L=L, zerl=zerl, f=f, eps=eps, n_stich=n_stich))


@st.cache_data(show_spinner=False)
def _minimum(m, L, zerl, f, eps):
    return ev.analyse_minimum(Settings(m=m, L=L, zerl=zerl, f=f, eps=eps))


@st.cache_data(show_spinner=False)
def _widen():
    return ev.widen_rows()


@st.cache_data(show_spinner=False)
def _widen_live(m, f_value, eps_value):
    return ev.widen_live_row(m, f_value, eps_value)


@st.cache_data(show_spinner=False)
def _cusum(m, seed, k, n_stich):
    return ev.analyse_cusum(Settings(m=m, seed=seed, k=k, n_stich=n_stich))


@st.cache_data(show_spinner=False)
def _cusum_check(m, seed, k, n_stich):
    return ev.cusum_check(Settings(m=m, seed=seed, k=k, n_stich=n_stich))


@st.cache_data(show_spinner=False)
def _contamination(seed, L, n_stich, stoerung):
    return ev.analyse_contamination(Settings(seed=seed, L=L, n_stich=n_stich, stoerung=stoerung))


@st.cache_data(show_spinner=False)
def _t2():
    return ev.t2_rows()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in aps_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📏 Parameter schätzen: wie viel Phase I braucht eine Karte?")
st.markdown(T.INTRO_MD)

with st.expander("So funktioniert Parameter schätzen", expanded=True):
    st.markdown(T.HOW_MD)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Phase I**")
    m = st.select_slider("Phase-I-Beobachtungen m", options=C.M_OPTIONS, key="m_select", help="Wie viele Beobachtungen die Karte zum Schätzen von Mittel und Streuung bekommt (alle Experimente; Experiment 5 rechnet immer mit m = 100).")
    seed = st.number_input("Zufalls-Seed der Phase-I-Stichprobe", *bounds("seed_input"), key="seed_input", step=1, help="Welche Phase-I-Stichprobe gezogen wird (Experimente 1, 2, 4, 5).")
    est = st.radio("Schätzer der Streuung", C.ESTIMATORS, key="est_radio", horizontal=True,
                   help="s: Stichproben-Standardabweichung (Nenner m − 1); MR: mittlere gleitende Spannweite geteilt durch 1,1284 (Experiment 1; Experiment 5 zeigt beide).")
    st.markdown("**Karte**")
    L = st.number_input("Grenzfaktor L", *bounds("L_input"), key="L_input", step=0.05, format="%.2f", help="Grenzen x̄ ± L·s, zweiseitig (Experimente 1, 2, 3, 5). Das Ziel bleibt die ARL₀ 370,398 der 3σ-Karte bei bekannten Parametern.")
    zerl = st.radio("Was wird geschätzt", C.ZERLEGUNG, key="zerl_radio", help="Beide Parameter, nur das Mittel (σ bekannt) oder nur die Streuung (μ bekannt) (Experimente 2 und 3).")
    st.markdown("**Kriterium**")
    f = st.radio("Schwellenanteil f der Ziel-ARL₀", C.F_OPTIONS, key="f_radio", horizontal=True, help="Eine Karte gilt als zu kurz, wenn ihre bedingte ARL₀ unter f mal 370,398 liegt (Experimente 2 und 3).")
    eps = st.radio("Zulässiger Anteil ε", C.EPS_OPTIONS, key="eps_radio", horizontal=True, help="Höchstens dieser Anteil der Phase-I-Stichproben darf unter die Schwelle fallen (Experiment 3 und Ampel in Experiment 2).")
    st.markdown("**Viele Stichproben, CUSUM, Störung**")
    n_stich = st.slider("Phase-I-Stichproben", *bounds("nst_slider"), key="nst_slider", step=C.N_STICH_STEP, help="Wie viele Phase-I-Stichproben für Streudiagramm, CUSUM und Störungs-Experiment gezogen werden (Experimente 2, 4, 5).")
    k = st.number_input("CUSUM-Referenzwert k", *bounds("k_input"), key="k_input", step=0.05, format="%.2f", help="Einseitiger CUSUM nach oben; die Schwelle h wird so gelöst, dass die ARL₀ bei bekannten Parametern 370,398 ist (Experiment 4).")
    stoer = st.selectbox("Störung der Phase I", C.STOERUNG, key="stoer_select", help="Ein Ausreißer oder eine Stufe in der Mitte der Phase I (m = 100, Experiment 5).")
    st.button("➕ Weitere Phase-I-Stichprobe", width="stretch", on_click=next_sample, help="Setzt den Zufalls-Seed um eins weiter.")

sync_query_params({"m_select": int(m), "seed_input": int(seed), "est_radio": est, "L_input": float(L), "zerl_radio": zerl, "f_radio": f, "eps_radio": eps,
                   "k_input": float(k), "nst_slider": int(n_stich), "stoer_select": stoer})

settings = Settings(int(m), int(seed), est, float(L), zerl, f, eps, float(k), int(n_stich), stoer)

# --- Ergebnis: eine Phase-I-Stichprobe, eine Karte ------------------------------------------------------------------------------------------------------

a = _sample(settings.m, settings.seed, settings.estimator, settings.L)
st.markdown("## 🎯 Ergebnis: diese Phase-I-Stichprobe baut diese Karte")
m1, m2, m3, m4 = st.columns(4)
m1.metric("bedingte ARL₀ dieser Karte", de(a.arl0, 1), delta=f"{de(a.arl0 - C.TARGET_ARL0, 1)} gegen das Ziel {de(C.TARGET_ARL0, 1)}", delta_color="off", help="Beobachtungen bis zum Alarm im Mittel, gegeben diese Phase-I-Stichprobe (1/p).")
m2.metric("Grenzen der Karte", f"{de(a.lower, 2)} bis {de(a.upper, 2)}", delta=f"wahr: ±{de(a.L, 1)}", delta_color="off")
m3.metric(f"Alarm in den ersten {C.N_SIM_HORIZON}", pct(a.p100), delta=f"bekannte Parameter: {pct(float(E.alarm_within(C.P_NOMINAL, C.N_SIM_HORIZON)))}", delta_color="off", help="Wahrscheinlichkeit mindestens eines Fehlalarms in den ersten 100 Beobachtungen von Phase II.")
m4.metric("Median der Lauflänge", de(a.median, 1), delta=f"5 %: {de(a.q05, 1)}, 95 %: {de(a.q95, 1)}", delta_color="off", help="Lauflänge bis zum Alarm, geometrisch verteilt.")
st.plotly_chart(build_sample(a), width="stretch", key="sample_chart")
st.caption("Punkte: die Phase-I-Beobachtungen (ausgedacht, unabhängig N(0, 1)). Orange: die Grenzen, die die Karte daraus baut; rot gepunktet: die Grenzen, die sie bei bekannten Parametern hätte.")

# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
with st.spinner("Rechne die exakte Verteilung der bedingten ARL₀ (etwa 1 bis 3 s, bei weiteren Grenzen und kleinem m bis 6 s) …"):
    sp = _spread(settings.m, settings.seed, settings.L, settings.zerl, settings.f, settings.eps, settings.n_stich)
    mn = _minimum(settings.m, settings.L, settings.zerl, settings.f, settings.eps)
st.markdown("**1. Welche Karte entsteht aus einer Phase-I-Stichprobe, und wie lange hält sie?**")
kind, text = ev.verdict_sample(a)
getattr(st, kind)(text)
st.markdown("**2. Wie breit streut die ARL₀, und welche „mittlere“ ist gemeint?**")
kind, text = ev.verdict_spread(sp)
getattr(st, kind)(text)
st.markdown("**3. Wie viele Beobachtungen braucht Phase I, oder wie weit müssen die Grenzen werden?**")
kind, text = ev.verdict_minimum(mn)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Eine Phase-I-Stichprobe, eine Karte")
st.caption("Regler: m, Zufalls-Seed, Schätzer, Grenzfaktor L; „Weitere Phase-I-Stichprobe“ zieht die nächste. Die Beobachtungen stammen aus einem SplitMix64-Strom (Zufalls-Seed), die Reihen sind ausgedacht.")
hand = _hand(settings.L)
st.dataframe([{"Stichprobe (m = 3)": r["name"], "x̄": de(r["xb"], 2), "s": de(r["s"], 2), "Grenzen": ev.interval(r["lower"], r["upper"], 1), "bedingte ARL₀": sci(r["arl0"])} for r in hand], hide_index=True, width="stretch")
kind, text = ev.verdict_hand(hand)
getattr(st, kind)(text)
st.markdown(f"**Die letzten {C.HISTORY_LENGTH} Phase-I-Stichproben** (Zufalls-Seeds {a.history[0][0]} bis {a.history[-1][0]}, m = {a.m}): die bedingte ARL₀ springt von Stichprobe zu Stichprobe.")
st.plotly_chart(build_history(a), width="stretch", key="history_chart")

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie breit streut die ARL₀, und welche mittlere?")
st.caption("Regler: m, Grenzfaktor L, Was wird geschätzt, Schwellenanteil f, Zufalls-Seed und Zahl der Phase-I-Stichproben (für das Streudiagramm). Die Verteilungsfunktion ist exakt gerechnet, ohne Zufall.")
st.plotly_chart(build_cdf(sp), width="stretch", key="cdf_chart")
rows = [
    {"Maß": "1. Verteilung der bedingten ARL₀: Median (5 % / 95 %)", "Wert": f"{de(sp.median, 1)} ({de(sp.q05, 1)} / {de(sp.q95, 1)})", "Zähleinheit": "Phase-I-Stichproben"},
    {"Maß": "2. Mittel der bedingten ARL₀, E[1/p]", "Wert": "unendlich" if sp.mean_carl == float("inf") else de(sp.mean_carl, 1), "Zähleinheit": f"Phase-I-Stichproben ({sp.mean_source})"},
    {"Maß": "3. Kehrwert der mittleren Alarmrate, 1/E[p]", "Wert": de(sp.inv_mean_alarm, 1), "Zähleinheit": "Beobachtungen über viele Karten"},
    {"Maß": "4. Unbedingte Lauflänge (frische Phase I je Lauf): Median", "Wert": de(sp.uncond["median"], 1) if sp.uncond else "nur für L = 3 und m = 30, 100, 500 eingefroren", "Zähleinheit": "Beobachtungen bis zum Alarm"},
    {"Maß": "bekannte Parameter (Vergleich): ARL₀ / Median der Lauflänge", "Wert": f"{de(E.nominal_arl(sp.settings.L), 1)} / {de(E.geometric_quantile(E.known_alarm_prob(sp.settings.L), 0.5), 1)}", "Zähleinheit": "Beobachtungen bis zum Alarm"},
]
st.markdown("**Vier Maße für dieselbe Karte** (Maße 2 bis 4: beide Parameter geschätzt)")
st.dataframe(rows, hide_index=True, width="stretch")
st.markdown("**Kopplung von ARL₀ und ARL₁ je Phase-I-Stichprobe** (beide Parameter geschätzt)")
st.plotly_chart(build_coupling(sp), width="stretch", key="coupling_chart")
kind, text = ev.verdict_coupling(sp)
getattr(st, kind)(text)
st.caption("Ehrliche Grenze: Lombard und Hawkins halten gerade die bedingte ARL für ein „notionales“ Konstrukt (der Anwender erlebt eine Phase I, die unbedingte ARL sei das Maß); Gandy und Kvaløy sowie Diko u. a. garantieren die bedingte ARL mit Wahrscheinlichkeit. Beide Seiten rechnen richtig, sie streiten über das Maß.")

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie viele Beobachtungen, oder die Grenzen weiten?")
st.caption("Regler: Schwellenanteil f, zulässiger Anteil ε, Was wird geschätzt, Grenzfaktor L (für die Kurve und das kleinste m). Die Tabelle der Grenzfaktoren gilt fest für f = ½ und ε = 5 %.")
st.plotly_chart(build_minimum(mn), width="stretch", key="minimum_chart")
wrows = _widen()
st.markdown("**Grenzen weiten: drei Kriterien** (ARL₀-Ziel 370,398; ARL₁ = bei bekannten Parametern, Sprung +1σ: ein Preis, nicht die ARL₁ der geschätzten Karte)")
st.dataframe([{
    "m": r["m"], "L(a) gleiche Alarmrate": de(r["L_a"], 4), "L(b) Mittel = 370,4": de(r["L_b"], 4), "L(c) Garantie 5 %": de(r["L_c"], 4),
    "ARL₁ bei L = 3": de(r["arl1_L3"], 2), "bei L(a)": de(r["arl1_a"], 2), "bei L(b)": de(r["arl1_b"], 2), "bei L(c)": de(r["arl1_c"], 2),
    "Anteil < 185,2 bei L = 3": pct(r["share_3"]), "bei L(b)": pct(r["share_b"]), "bei L(c)": pct(r["share_c"]),
} for r in wrows], hide_index=True, width="stretch")
kind, text = ev.verdict_widen(wrows)
getattr(st, kind)(text)
if str(settings.m) not in R.widen_table():
    live = _widen_live(settings.m, settings.f_value, settings.eps_value)
    lc = "nicht erreichbar" if live["L_c"] is None else de(live["L_c"], 4)
    st.caption(f"Für Ihr m = {settings.m} (nicht in der Tabelle; f und ε wie eingestellt): L(a) = {de(live['L_a'], 4)}, L(c) = {lc}; L(b) steht nur für die Tabellenzeilen.")

st.markdown("---")

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: CUSUM mit geschätzten Parametern")
st.caption("Regler: m, k, Zahl der Phase-I-Stichproben, Zufalls-Seed. Einseitiger CUSUM nach oben (zero-state), daneben die einseitige Shewhart-Karte; die bedingte ARL₀ je Stichprobe ist exakt (Integralgleichung), nicht simuliert.")
with st.spinner(f"Löse {settings.n_stich} Integralgleichungen (je nach Zahl 0,1 bis 0,5 s) …"):
    ca = _cusum(settings.m, settings.seed, settings.k, settings.n_stich)
st.plotly_chart(build_cusum(ca), width="stretch", key="cusum_chart")
kind, text = ev.verdict_cusum(ca)
getattr(st, kind)(text)
with st.expander("Gegenprobe für eine Phase-I-Stichprobe: Integralgleichung, Gitter, Simulation"):
    with st.spinner("Simuliere 4000 Läufe (etwa 1 s) …"):
        ck = _cusum_check(settings.m, settings.seed, settings.k, settings.n_stich)
    st.markdown(f"Die erste gezogene Stichprobe mit bedingter ARL zwischen 15 und 400 (Nr. {ck['index'] + 1}; x̄ = {de(ck['xb'], 3)}, s = {de(ck['s'], 3)}):")
    st.dataframe([
        {"Rechenweg": "Integralgleichung (Nyström, 64 Knoten)", "ARL ab Null": de(ck["integral"], 3)},
        {"Rechenweg": f"Brook/Evans-Gitter ({C.CHECK_GRID} Zellen)", "ARL ab Null": de(ck["grid"], 3)},
        {"Rechenweg": f"Simulation ({ck['runs']} Läufe, zensiert bei 6000: {ck['sim_censored']})", "ARL ab Null": f"{de(ck['sim_mean'], 1)} ± {de(ck['sim_se'], 1)}"},
    ], hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 5 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Schätzer und verunreinigte Phase I")
st.caption(f"Regler: Störung, Grenzfaktor L, Zahl der Phase-I-Stichproben, Zufalls-Seed. Immer m = {C.M_PHASE_CONTAMINATION}. Phase II läuft bei der Stufe im Zustand der zweiten Hälfte (Setzung dieses Stücks), sonst im Zustand ohne Störung.")
ct = _contamination(settings.seed, settings.L, settings.n_stich, settings.stoerung)
st.plotly_chart(build_contamination(ct), width="stretch", key="contamination_chart")
fmt = lambda v: f"{de(v[0], 1)} / {de(v[1], 1)} / {de(v[2], 1)}"
cases = list(ct["cases"])
if (ct["chosen"]["kind"], ct["chosen"]["size"]) not in {(r["kind"], r["size"]) for r in cases}:
    cases.append(ct["chosen"])
st.dataframe([{"Fall": case_label(r), "Schätzer s: ARL₀ / ARL₁ auf / ARL₁ ab": fmt(r["s"]), "Schätzer MR: ARL₀ / ARL₁ auf / ARL₁ ab": fmt(r["MR"])} for r in cases], hide_index=True, width="stretch")
kind, text = ev.verdict_contamination(ct)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MD)

with st.expander("❓ Und wenn die Annahmen nicht gelten? (Verweis auf Stück 4b)"):
    st.markdown(T.FOLLOWUP_MD)
    ti = R.table_i()
    st.markdown(T.TABLE_I_NOTE)
    head = ["φ (ARL₀ bei bekannten Parametern)"] + [f"m = {mm}" for mm in ti["m"]]
    st.dataframe([
        {head[0]: f"{de(phi, 1)} ({de(a0, 1)})", **{head[i + 1]: f"{de(100 * ti['anteil_unter_halber_arl0'][j][i], 1)} % ({ti['median'][j][i]})" for i in range(len(ti["m"]))}}
        for j, (phi, a0) in enumerate(zip(ti["phi"], ti["arl0_bekannt"]))
    ], hide_index=True, width="stretch")
    st.caption(R.source("abhaengig"))
    st.markdown(T.TABLE_J_NOTE)
    t2 = _t2()
    ms = (30, 100, 300, 1000)
    st.dataframe([
        {"Kanäle p": r["p"], "χ²-Grenze": de(r["chi2_limit"], 3), **{f"ARL₀ (χ²-Regel), m = {mm}": de(r[mm], 1) for mm in ms}, **{f"Grenze F-Regel, m = {mm}": de(r[f"ucl_f_{mm}"], 2) for mm in ms}}
        for r in t2
    ], hide_index=True, width="stretch")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown("**Konventionen** (vor der ersten Gleichung):")
    st.markdown(T.CONVENTIONS_MD)
    st.markdown(T.FORMULAS_MD)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
