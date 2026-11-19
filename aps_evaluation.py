"""Auswertung der fünf Experimente und alle Verdikte (reine Funktionen: `verdict_*(…) -> (Art, Text)`, die App ruft `getattr(st, Art)(Text)`).

Zähleinheiten, immer getrennt genannt: Beobachtungen (bis zum Alarm), Phase-I-Stichproben (Wiederholungen des Schätzens), Läufe (Simulation). „ARL₀-Ziel“ heißt hier 370,398
(= 1/p der 3σ-Karte, nicht das gesetzte 370). Alle Zahlen gelten für Einzelwerte, unabhängig N(0, 1), Phase I in Kontrolle, ν = 0 (die Änderung liegt vor der ersten Beobachtung
von Phase II), zero-state."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import ndtri
from scipy.stats import spearmanr

import aps_constants as C
import aps_cusum as CU
import aps_exact as E
import aps_reference as R
import aps_scenario as S


# --- Zahlenformate (deutsch: Komma, echtes Minus, Tausenderpunkt ab 10.000) -----------------------------------------------------------------------------

def de(x: float, digits: int = 1) -> str:
    if math.isinf(x):
        return "unendlich" if x > 0 else "−unendlich"
    text = f"{x:,.{digits}f}" if abs(x) >= 10000 else f"{x:.{digits}f}"
    text = text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return text.replace("-", "−") if float(f"{x:.{digits}f}") != 0.0 else text.replace("-", "")


def pct(share: float, digits: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Komma."""
    return f"{de(100.0 * share, digits)} %"


def sci(x: float, digits: int = 2) -> str:
    """Große Zahlen ab 10⁶ als „5,07·10⁸“, sonst mit Komma."""
    if x < 1e6:
        return de(x, digits)
    exp = int(math.floor(math.log10(x)))
    mant = x / 10 ** exp
    sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"{de(mant, digits)}·10{sup}"


def interval(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{de(lo, digits)}; {de(hi, digits)}]"


# --- Einstellungen ------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    m: int = C.DEFAULT_M
    seed: int = C.DEFAULT_SEED
    estimator: str = C.DEFAULT_ESTIMATOR
    L: float = C.DEFAULT_L
    zerl: str = C.DEFAULT_ZERLEGUNG
    f: str = C.DEFAULT_F
    eps: str = C.DEFAULT_EPS
    k: float = C.DEFAULT_K
    n_stich: int = C.DEFAULT_N_STICH
    stoerung: str = C.DEFAULT_STOERUNG

    @property
    def f_value(self) -> float:
        return C.F_VALUES[self.f]

    @property
    def eps_value(self) -> float:
        return C.EPS_VALUES[self.eps]

    @property
    def stoerung_fall(self) -> tuple:
        """(Art, Größe in σ) der gewählten Störung."""
        return C.STOERUNG_FAELLE[self.stoerung]

    @property
    def mode(self) -> str:
        return {"beide geschätzt": "both", "nur μ geschätzt": "mu", "nur σ geschätzt": "sigma"}[self.zerl]


# --- Experiment 1: eine Phase-I-Stichprobe, eine Karte --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleAnalysis:
    m: int
    seed: int
    estimator: str
    L: float
    x: np.ndarray                 # die m Phase-I-Beobachtungen
    xb: float
    sigma: float                  # geschätzte Streuung (s oder MR)
    lower: float
    upper: float
    p: float                      # Alarmwahrscheinlichkeit je Beobachtung (wahre Reihe N(0, 1))
    arl0: float                   # bedingte ARL₀ = 1/p
    p100: float                   # Alarm in den ersten 100 Beobachtungen
    median: float                 # Median, 5 %, 95 % der geometrischen Lauflänge
    q05: float
    q95: float
    history: tuple                # ((Zufalls-Seed, bedingte ARL₀), …) der letzten HISTORY_LENGTH Stichproben


def _rl_quantile(p: float, q: float) -> float:
    """Quantil der geometrischen Lauflänge; bei p = 1 ist die Lauflänge 1, bei p = 0 gibt es nie einen Alarm."""
    if p >= 1.0:
        return 1.0
    return math.inf if p <= 0.0 else E.geometric_quantile(p, q)


def analyse_values(x, estimator: str = "s", L: float = C.L_NOMINAL, seed: int = 0) -> SampleAnalysis:
    """Auswertung für gegebene Phase-I-Werte (auch für das Handbeispiel und die Randfälle)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Phase I braucht mindestens zwei Beobachtungen: s ist mit einer Beobachtung nicht definiert")
    xb, sigma = S.estimate(x, estimator)
    xb, sigma = float(xb), float(sigma)
    p = float(E.p_alarm(xb, sigma, L))
    arl0 = math.inf if p == 0.0 else 1.0 / p
    return SampleAnalysis(
        m=int(x.size), seed=seed, estimator=estimator, L=L, x=x, xb=xb, sigma=sigma, lower=xb - L * sigma, upper=xb + L * sigma, p=p, arl0=arl0,
        p100=float(E.alarm_within(p, C.N_SIM_HORIZON)),
        median=_rl_quantile(p, 0.5), q05=_rl_quantile(p, 0.05), q95=_rl_quantile(p, 0.95),
        history=())


def history_seeds(seed: int, length: int = C.HISTORY_LENGTH) -> list[int]:
    """Die Zufalls-Seeds s − (length − 1) … s (zyklisch in 0 … SEED_MAX), älteste zuerst."""
    return [(seed - i) % (C.SEED_MAX + 1) for i in range(length - 1, -1, -1)]


def analyse_sample(settings: Settings) -> SampleAnalysis:
    a = analyse_values(S.phase1_sample(settings.seed, settings.m), settings.estimator, settings.L, settings.seed)
    hist = []
    for sd in history_seeds(settings.seed):
        h = analyse_values(S.phase1_sample(sd, settings.m), settings.estimator, settings.L, sd)
        hist.append((sd, h.arl0))
    return SampleAnalysis(**{**a.__dict__, "history": tuple(hist)})


def hand_analysis(L: float = C.L_NOMINAL) -> list[dict]:
    """Die drei ausgedachten Stichproben des Handbeispiels (m = 3, Schätzer s): Grenzen und bedingte ARL₀."""
    rows = []
    for name, x in C.HAND_SAMPLES.items():
        a = analyse_values(x, "s", L)
        rows.append({"name": name, "xb": a.xb, "s": a.sigma, "lower": a.lower, "upper": a.upper, "arl0": a.arl0})
    return rows


def verdict_hand(rows: list[dict]) -> tuple[str, str]:
    parts = [f"{r['name']}: Grenzen {interval(r['lower'], r['upper'], 1)}, bedingte ARL₀ {sci(r['arl0'])}" for r in rows]
    orders = [int(math.floor(math.log10(r["arl0"]))) for r in rows]
    return "info", ("Drei Beobachtungen, drei Karten (Handbeispiel, ausgedacht, Schätzer s): " + "; ".join(parts) +
                    f". Dieselbe Karte mit denselben Regeln, drei Stichproben, ARL₀ von {de(min(r['arl0'] for r in rows), 2)} bis {sci(max(r['arl0'] for r in rows))} "
                    f"({max(orders) - min(orders)} Größenordnungen).")


def verdict_sample(a: SampleAnalysis) -> tuple[str, str]:
    est = "Stichproben-Standardabweichung s" if a.estimator == "s" else "mittlere gleitende Spannweite (MR)"
    base = (f"Diese Phase-I-Stichprobe (m = {a.m}, Zufalls-Seed {a.seed}, Schätzer {est}) ergibt die Grenzen {interval(a.lower, a.upper, 2)} statt ±{de(a.L, 1)}. "
            f"Bedingte ARL₀ {de(a.arl0, 1)} Beobachtungen bis zum Alarm (Ziel {de(C.TARGET_ARL0, 1)} bei bekannten Parametern bei L = 3); "
            f"Lauflänge geometrisch: Median {de(a.median, 1)}, 5 % {de(a.q05, 1)}, 95 % {de(a.q95, 1)}; Wahrscheinlichkeit eines Alarms in den ersten {C.N_SIM_HORIZON} Beobachtungen {pct(a.p100)}.")
    if a.sigma == 0.0:
        return "warning", "Alle Phase-I-Werte sind gleich: s = 0, die Grenzen fallen zusammen, und jede abweichende Beobachtung alarmiert (ARL₀ = 1). " + base
    if a.arl0 < 0.5 * C.TARGET_ARL0:
        return "warning", base + f" Das ist weniger als die Hälfte der nominellen ARL₀: bei einer anderen Phase-I-Stichprobe wäre es anders."
    return "info", base + " Eine andere Phase-I-Stichprobe gibt eine andere Zahl (siehe den Verlauf der letzten Stichproben)."


# --- Experiment 2: Verteilung der bedingten ARL₀, vier Maße, Kopplung ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadAnalysis:
    settings: Settings
    a_grid: np.ndarray
    cdf: np.ndarray
    threshold: float              # f · 370,398
    share: float                  # P(bedingte ARL₀ < threshold), exakt
    median: float
    q05: float
    q95: float
    mean_carl: float              # E[1/p] (beide geschätzt), unendlich für m ≤ L² + 1
    mean_source: str              # "eingefroren" | "live"
    inv_mean_alarm: float         # 1/E[p]
    uncond: dict | None           # unbedingte Lauflänge (eingefroren, nur L = 3 und m in der Tabelle)
    arl0_samples: np.ndarray      # bedingte ARL₀ der gezogenen Phase-I-Stichproben (beide geschätzt)
    arl1_samples: np.ndarray      # bedingte ARL₁ bei Aufwärtssprung δ = 1
    rho: float                    # Rangkorrelation ARL₀ / ARL₁
    ratio_q: tuple                # 5 %, 50 %, 95 % des Verhältnisses ARL₁ / ARL₁ der zentrierten Karte gleicher ARL₀
    frontier: tuple               # (ARL₀-Gitter, ARL₁ der zentrierten Karte)
    arl1_known: float


def mean_carl_for(m: int, L: float):
    """(E[1/p], Quelle): eingefrorene Tabelle für L = 3 und m aus der Tabelle, sonst live über den Tail der exakten Verteilung (bis etwa 5 s)."""
    if not E.mean_carl_is_finite(m, L):
        return math.inf, "Herleitung"
    if L == C.L_NOMINAL:
        frozen = R.mean_carl(m)
        if frozen is not None:
            return frozen, "eingefroren"
    return E.mean_carl(m, L), "live"


def analyse_spread(settings: Settings) -> SpreadAnalysis:
    m, L, mode = settings.m, settings.L, settings.mode
    thr = settings.f_value * C.TARGET_ARL0
    lo_q, hi_q = E.quantile_carl(0.0005, m, L, mode), E.quantile_carl(0.9995, m, L, mode)
    a_grid = np.geomspace(max(1.0 + 1e-6, 0.9 * lo_q), 1.1 * hi_q, 160)
    cdf = np.array([E.cdf_carl(a, m, L, mode) for a in a_grid])
    mean_val, mean_src = mean_carl_for(m, L)
    uncond = R.unconditional(m) if L == C.L_NOMINAL else None
    xb, s = S.phase1_direct(settings.seed, m, settings.n_stich)
    p0, p1 = E.p_alarm(xb, s, L), E.p_alarm(xb, s, L, mu=C.SHIFT_DELTA)
    a0, a1 = 1.0 / p0, 1.0 / p1
    rho = float(spearmanr(a0, a1)[0])
    leff = E.effective_limit(p0)
    a1c = np.array([E.arl1_known(float(l), C.SHIFT_DELTA) for l in leff])
    ratio = a1 / a1c
    grid0 = np.geomspace(max(2.0, float(a0.min())), float(a0.max()), 80)
    frontier = (grid0, np.array([E.arl1_known(float(-ndtri(1.0 / (2.0 * g))), C.SHIFT_DELTA) for g in grid0]))
    return SpreadAnalysis(
        settings=settings, a_grid=a_grid, cdf=cdf, threshold=thr, share=E.cdf_carl(thr, m, L, mode), median=E.quantile_carl(0.5, m, L, mode),
        q05=E.quantile_carl(0.05, m, L, mode), q95=E.quantile_carl(0.95, m, L, mode), mean_carl=mean_val, mean_source=mean_src,
        inv_mean_alarm=E.inverse_mean_alarm_rate(m, L), uncond=uncond, arl0_samples=a0, arl1_samples=a1, rho=rho,
        ratio_q=tuple(float(v) for v in np.percentile(ratio, [5, 50, 95])), frontier=frontier, arl1_known=E.arl1_known(L, C.SHIFT_DELTA))


def verdict_spread(sp: SpreadAnalysis) -> tuple[str, str]:
    s = sp.settings
    head = (f"m = {s.m} Phase-I-Beobachtungen, {s.zerl}, Grenzfaktor {de(s.L, 2)}: Median der bedingten ARL₀ {de(sp.median, 1)} (Ziel {de(C.TARGET_ARL0, 1)}), 5 % {de(sp.q05, 1)}, 95 % {de(sp.q95, 1)}; "
            f"{pct(sp.share)} der Phase-I-Stichproben (exakt, ohne Zufall gerechnet) ergeben eine Karte mit weniger als {s.f} der Ziel-ARL₀ ({de(sp.threshold, 1)}).")
    measures = ""
    if s.zerl == C.DEFAULT_ZERLEGUNG:
        un = f"; unbedingte Lauflängenverteilung: Median {de(sp.uncond['median'], 1)}" if sp.uncond else ""
        mean_txt = "ist unendlich (die Herleitung steht im Formel-Expander)" if math.isinf(sp.mean_carl) else de(sp.mean_carl, 1)
        measures = (f" Für dieselbe Karte nennt das Mittel der bedingten ARL₀ {mean_txt}, der Kehrwert der mittleren Alarmrate {de(sp.inv_mean_alarm, 1)}{un}.")
    kind = "warning" if sp.share > s.eps_value else "success"
    return kind, head + measures


# --- Experiment 3: Mindestzahl m, Grenzen weiten ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MinimumAnalysis:
    settings: Settings
    m_min: int | None
    p_before: float | None        # P bei m_min − 1
    p_at: float | None            # P bei m_min
    m_sigma: int | None           # Zerlegung bei gleichem f, ε, L
    m_mu: int | None
    m_grid: np.ndarray
    curve: np.ndarray             # P(bedingte ARL₀ < f·370,398) über m_grid
    threshold: float


def analyse_minimum(settings: Settings) -> MinimumAnalysis:
    f, eps, L = settings.f_value, settings.eps_value, settings.L
    mm = E.min_m(f, eps, L, settings.mode)
    grid = np.unique(np.round(np.geomspace(C.M_MIN, C.M_MAX, 60)).astype(int))
    curve = np.array([E.cdf_carl(f * C.TARGET_ARL0, int(m), L, settings.mode) for m in grid])
    return MinimumAnalysis(
        settings=settings, m_min=mm, p_before=None if mm is None or mm <= 2 else E.cdf_carl(f * C.TARGET_ARL0, mm - 1, L, settings.mode),
        p_at=None if mm is None else E.cdf_carl(f * C.TARGET_ARL0, mm, L, settings.mode),
        m_sigma=E.min_m(f, eps, L, "sigma"), m_mu=E.min_m(f, eps, L, "mu"), m_grid=grid, curve=curve, threshold=f * C.TARGET_ARL0)


def verdict_minimum(mn: MinimumAnalysis) -> tuple[str, str]:
    s = mn.settings
    if mn.m_min is None:
        return "warning", f"Mit höchstens 100.000 Phase-I-Beobachtungen ist das Kriterium (höchstens {s.eps} der Stichproben unter {s.f} der Ziel-ARL₀) bei Grenzfaktor {de(s.L, 2)} nicht erreichbar."
    edge = (f"(bei m = {mn.m_min - 1}: {pct(mn.p_before, 2)}, bei m = {mn.m_min}: {pct(mn.p_at, 2)})" if mn.p_before is not None
            else f"(schon bei der kleinsten möglichen Zahl m = {mn.m_min} sind es nur {pct(mn.p_at, 2)}; m = 2 ist die kleinste Zahl, mit der s definiert ist)")
    text = (f"Kriterium: höchstens {s.eps} der Phase-I-Stichproben ergeben eine Karte mit weniger als {s.f} der Ziel-ARL₀ ({de(mn.threshold, 1)}). Bei Grenzfaktor {de(s.L, 2)} und {s.zerl} braucht das "
            f"m = {mn.m_min} Phase-I-Beobachtungen {edge}.")
    if s.zerl == C.DEFAULT_ZERLEGUNG and mn.m_sigma and mn.m_mu:
        text += f" Nur σ geschätzt: m = {mn.m_sigma}; nur μ geschätzt: m = {mn.m_mu}: der Streuungsfehler überwiegt (Faktor {de(mn.m_sigma / mn.m_mu, 0)}) in diesem Kriterium."
    return "info", text


def widen_rows(max_f: float = 0.5) -> list[dict]:
    """Befund D für m = 30 … 1000: Grenzfaktoren L(a), L(b), L(c) (eingefroren), Preis ARL₁(δ = 1) bei bekannten Parametern und die Verteilung der bedingten ARL₀ bei jedem L
    (Anteil unter f·370,398 und Median, live aus der exakten Verteilung)."""
    rows = []
    for key, v in R.widen_table().items():
        m = int(key)
        row = {"m": m, **v}
        for tag, L in (("3", 3.0), ("a", v["L_a"]), ("b", v["L_b"]), ("c", v["L_c"])):
            row[f"share_{tag}"] = E.cdf_carl(max_f * C.TARGET_ARL0, m, L)
            row[f"median_{tag}"] = E.quantile_carl(0.5, m, L)
        rows.append(row)
    return rows


def verdict_widen(rows: list[dict], m: int = 100) -> tuple[str, str]:
    r = next(x for x in rows if x["m"] == m)
    return "info", (f"Bei m = {m}: Wer die Grenzen für die Garantie (höchstens 5 % unter 185,2) weitet, braucht den Faktor {de(r['L_c'], 2)} und zahlt bei bekannten Parametern "
                    f"ARL₁(δ = 1) = {de(r['arl1_c'], 1)} statt {de(r['arl1_L3'], 1)} Beobachtungen. Wer stattdessen auf das Mittel der bedingten ARL₀ eicht (Faktor {de(r['L_b'], 2)}), senkt dieses Mittel auf "
                    f"{de(C.TARGET_ARL0, 1)}, erhöht aber den Anteil der Karten unter 185,2 von {pct(r['share_3'])} auf {pct(r['share_b'])} (Median {de(r['median_3'], 1)} → {de(r['median_b'], 1)}).")


def widen_live_row(m: int, f: float = 0.5, eps: float = 0.05) -> dict:
    """Für ein m ohne eingefrorene Zeile: L(a) und L(c) live, L(b) nur, wenn m in der Tabelle steht."""
    La = E.limit_equal_alarm_rate(m)
    try:
        Lc = E.limit_guarantee(m, f, eps)
    except ValueError:
        Lc = None
    frozen = R.widen_table().get(str(m))
    return {"m": m, "L_a": La, "L_c": Lc, "L_b": None if frozen is None else frozen["L_b"],
            "arl1_L3": E.arl1_known(3.0, 1.0), "arl1_a": E.arl1_known(La, 1.0), "arl1_c": None if Lc is None else E.arl1_known(Lc, 1.0),
            "arl1_b": None if frozen is None else frozen["arl1_b"]}


def verdict_coupling(sp: "SpreadAnalysis") -> tuple[str, str]:
    """Befund E: Kopplung von bedingter ARL₀ und ARL₁ je Phase-I-Stichprobe (Aufwärtssprung 1σ)."""
    q05, q50, q95 = sp.ratio_q
    return "info", (f"Über {sp.arl0_samples.size} Phase-I-Stichproben (m = {sp.settings.m}, Zufalls-Seed {sp.settings.seed}) beträgt die Rangkorrelation zwischen bedingter ARL₀ und bedingter ARL₁ (Sprung +1σ) {de(sp.rho, 2)}: "
                    "wer eine kurze ARL₀ erwischt, erkennt auch schneller, die Schätzung wandert vor allem auf der Kompromisskurve. Verhältnis der bedingten ARL₁ zur ARL₁ einer zentrierten Karte mit gleicher ARL₀: "
                    f"Median {de(q50, 2)}, 5 % bis 95 % {de(q05, 2)} bis {de(q95, 2)}. Zählung, kein Urteil: die Streuung ist die Aussage (ARL₁ der zentrierten Karte mit bekannten Parametern: {de(sp.arl1_known, 1)}).")


# --- Experiment 4: CUSUM mit geschätzten Parametern -----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def cusum_h(k: float) -> float:
    return CU.solve_h(k)


@dataclass(frozen=True)
class CusumAnalysis:
    settings: Settings
    h: float
    c1: float
    arl_cusum: np.ndarray
    arl_shewhart: np.ndarray
    stats: dict                   # {"cusum": (Median, 5 %, Anteil unter der halben Ziel-ARL₀, Standardfehler des Anteils), "shewhart": (…)}


def analyse_cusum(settings: Settings) -> CusumAnalysis:
    h = cusum_h(settings.k)
    c1 = CU.shewhart_one_sided_limit()
    xb, s = S.phase1_direct(settings.seed, settings.m, settings.n_stich)
    ac = CU.cusum_arl_integral(-xb / s, 1.0 / s, settings.k, h, C.CUSUM_NODES)
    asw = CU.shewhart_one_sided_arl(xb, s, c1)
    half = C.TARGET_ARL0 / 2.0

    def stat(a):
        share = float((a < half).mean())
        return (float(np.median(a)), float(np.percentile(a, 5)), share, math.sqrt(max(share * (1 - share), 0.0) / a.size))
    return CusumAnalysis(settings, h, c1, ac, asw, {"cusum": stat(ac), "shewhart": stat(asw)})


def cusum_check(settings: Settings, analysis: CusumAnalysis | None = None) -> dict:
    """Gegenprobe für EINE Phase-I-Stichprobe (die erste gezogene mit ARL zwischen 15 und 400): Integralgleichung (64 Knoten), Brook/Evans-Gitter (400 Zellen) und Simulation
    (4000 Läufe, zensiert bei 6000 Beobachtungen)."""
    a = analysis or analyse_cusum(settings)
    xb, s = S.phase1_direct(settings.seed, settings.m, settings.n_stich)
    idx = np.flatnonzero((a.arl_cusum > 15.0) & (a.arl_cusum < 400.0))
    i = int(idx[0]) if idx.size else 0
    x_i, s_i = float(xb[i]), float(s[i])
    k, h = settings.k, a.h
    exact = float(CU.cusum_arl_integral(-x_i / s_i, 1.0 / s_i, k, h, C.CUSUM_NODES_TARGET)[0])
    from scipy.special import ndtr
    mu_z, sd_z = -x_i / s_i, 1.0 / s_i
    grid = CU.chain_arl(CU.cusum_chain(lambda v: ndtr((v - mu_z) / sd_z), k, h, C.CHECK_GRID))
    sim_seed = (settings.seed + 1) % (C.SEED_MAX + 1)
    mean, se, cens = CU.simulate_cusum(lambda step: S.normals(sim_seed, (step - 1) * C.CHECK_RUNS, C.CHECK_RUNS), x_i, s_i, k, h, C.CHECK_RUNS, 6000)
    return {"index": i, "xb": x_i, "s": s_i, "integral": exact, "grid": grid, "sim_mean": mean, "sim_se": se, "sim_censored": cens, "runs": C.CHECK_RUNS}


def verdict_cusum(ca: CusumAnalysis) -> tuple[str, str]:
    c, w = ca.stats["cusum"], ca.stats["shewhart"]
    s = ca.settings
    diff = c[2] - w[2]
    se = math.hypot(c[3], w[3])
    if diff > 2.0 * se:
        rel = "stärker betroffen als die Shewhart-Karte"
    elif diff < -2.0 * se:
        rel = "weniger stark betroffen als die Shewhart-Karte"
    else:
        rel = "von der Shewhart-Karte nicht zu unterscheiden"
    return "info", (f"Einseitig, m = {s.m}, k = {de(s.k, 2)}, h = {de(ca.h, 4)} (ARL₀ = {de(C.TARGET_ARL0, 3)} bei bekannten Parametern), {s.n_stich} Phase-I-Stichproben: CUSUM Median {de(c[0], 1)}, 5 % {de(c[1], 1)}, "
                    f"{pct(c[2])} unter {de(C.TARGET_ARL0 / 2, 1)}; einseitige Shewhart-Karte Median {de(w[0], 1)}, 5 % {de(w[1], 1)}, {pct(w[2])} unter {de(C.TARGET_ARL0 / 2, 1)}. "
                    f"In diesem Modell, bei diesen Parametern und diesem Maß (Anteil unter der halben Ziel-ARL₀) ist der CUSUM {rel}; "
                    "eine gelesene Quelle (Zwetsloot/Woodall 2017) nennt für ihre Maße (SDARL) das Gegenteil: beide Aussagen gelten für ihr Maß und ihre Einstellungen.")


# --- Experiment 5: Schätzer und verunreinigte Phase I ---------------------------------------------------------------------------------------------------

CONTAMINATION_CASES = (("keine", 0.0), ("Ausreißer", 5.0), ("Stufe", 1.0), ("Stufe", 2.0))


def contamination_row(x_clean: np.ndarray, kind: str, size: float, L: float = C.L_NOMINAL) -> dict:
    """Median der bedingten ARL₀, ARL₁ (Sprung +1) und ARL₁ (Sprung −1) für beide Schätzer. Phase II läuft im wahren Zustand (Stufe: der zweiten Hälfte, sonst 0)."""
    cont = S.contaminate(x_clean, kind, size)
    out = {"kind": kind, "size": size}
    for est in C.ESTIMATORS:
        xb, sg = S.estimate(cont.data, est)
        mu = cont.mu_phase2
        out[est] = tuple(float(np.median(1.0 / E.p_alarm(xb, sg, L, mu + d))) for d in (0.0, +1.0, -1.0))
    return out


def analyse_contamination(settings: Settings) -> dict:
    """Die vier festen Fälle und der gewählte Fall; m = 100, `n_stich` Phase-I-Stichproben aus Rohnormalzahlen."""
    x = S.phase1_raw_batch(settings.seed, C.M_PHASE_CONTAMINATION, settings.n_stich)
    cases = [contamination_row(x, kind, size, settings.L) for kind, size in CONTAMINATION_CASES]
    chosen = contamination_row(x, *settings.stoerung_fall, settings.L)
    return {"cases": cases, "chosen": chosen, "settings": settings}


def verdict_contamination(res: dict) -> tuple[str, str]:
    ch, base = res["chosen"], res["cases"][0]
    s = res["settings"]
    if ch["kind"] == "keine":
        return "info", (f"Ohne Störung (m = {C.M_PHASE_CONTAMINATION}, {s.n_stich} Phase-I-Stichproben, Median der bedingten Werte): ARL₀ {de(ch['s'][0], 1)} (Schätzer s) und {de(ch['MR'][0], 1)} (MR), "
                        f"ARL₁ bei Aufwärts- und Abwärtssprung 1: {de(ch['s'][1], 1)} / {de(ch['s'][2], 1)} (s) und {de(ch['MR'][1], 1)} / {de(ch['MR'][2], 1)} (MR). Wähle eine Störung, um die Wirkung zu sehen.")
    name = f"einem Ausreißer +{de(ch['size'], 1)}σ" if ch["kind"] == "Ausreißer" else f"einer Stufe von {de(ch['size'], 1)}σ in der Mitte"
    text = (f"Mit {name} in {C.M_PHASE_CONTAMINATION} Phase-I-Beobachtungen (Median über {s.n_stich} Phase-I-Stichproben): ARL₀ {de(ch['s'][0], 1)} statt {de(base['s'][0], 1)} (Schätzer s), "
            f"{de(ch['MR'][0], 1)} statt {de(base['MR'][0], 1)} (MR); ARL₁ für den Aufwärtssprung 1: {de(ch['s'][1], 1)} statt {de(base['s'][1], 1)} (s), {de(ch['MR'][1], 1)} statt {de(base['MR'][1], 1)} (MR); "
            f"für den Abwärtssprung 1: {de(ch['s'][2], 1)} (s), {de(ch['MR'][2], 1)} (MR). Phase II läuft "
            + ("im Zustand der zweiten Hälfte." if ch["kind"] == "Stufe" else "im Zustand ohne Störung."))
    if ch["s"][0] > 2.0 * C.TARGET_ARL0 and ch["MR"][0] < 0.5 * C.TARGET_ARL0:
        text += (" Hier bringt weder s noch MR die ARL₀ in die Nähe des Ziels: s macht die Karte taub (ARL₀ über das Doppelte der Ziel-ARL₀), MR zentriert falsch und alarmiert oft (ARL₀ unter der Hälfte). "
                 "Die Stufe zu finden ist die Aufgabe der Segmentierung (Stück 7), nicht des Schätzers.")
        return "warning", text
    if ch["s"][0] > 2.0 * base["s"][0]:
        return "warning", text + " Die Karte wird ruhiger und träger, ohne dass sie es meldet."
    return "info", text


# --- Annahmen-Tabelle: Zeile J (Hotelling-T² mit geschätztem Σ) und Tabelle I (eingefroren) ----------------------------------------------------------------

def t2_rows(ps=(1, 2, 5, 10), ms=(30, 100, 300, 1000)) -> list[dict]:
    """Unbedingte ARL₀ (Kehrwert der mittleren Alarmrate) der T²-Karte, χ²-Regel (Σ bekannt) gegen F-Regel; je p und m. Exakt über die F-Verteilung."""
    rows = []
    for p in ps:
        row = {"p": p, "chi2_limit": E.t2_chi2_limit(p)}
        for m in ms:
            row[m] = 1.0 / E.t2_mean_alarm_rate(p, m, E.t2_chi2_limit(p))
            row[f"ucl_f_{m}"] = E.t2_f_limit(p, m)
        rows.append(row)
    return rows
