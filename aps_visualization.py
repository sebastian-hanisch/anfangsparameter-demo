"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Waagerechte und senkrechte Marken auf logarithmischen Achsen sind Zwei-Punkte-Spuren, nie `add_hline`/`add_vline` (Plotly zieht sie dort bis 10^194)."""

import numpy as np
import plotly.graph_objects as go

import aps_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _hline(fig, y, x0, x1, name, color, dash="dot", width=2):
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", line=dict(color=color, dash=dash, width=width), name=name, hoverinfo="skip"))


def _vline(fig, x, y0, y1, name, color, dash="dot", width=2):
    fig.add_trace(go.Scatter(x=[x, x], y=[y0, y1], mode="lines", line=dict(color=color, dash=dash, width=width), name=name, hoverinfo="skip"))


# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

def build_sample(a):
    """Die m Phase-I-Beobachtungen, die geschätzten Grenzen x̄ ± L·s (gestrichelt) und die wahren Grenzen ±L (rot, gepunktet)."""
    m = a.m
    xs = list(range(1, m + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=list(a.x), mode="markers", marker=dict(color=C.COLOR_CHART, size=6 if m <= 200 else 3), name="Phase-I-Beobachtungen",
                             hovertemplate="Nr. %{x}: %{y:.2f}<extra></extra>"))
    for y, name, color, dash in ((a.upper, "geschätzte Grenzen x̄ ± L·s", C.COLOR_ACCENT, "dash"), (a.lower, None, C.COLOR_ACCENT, "dash"),
                                 (a.xb, "geschätztes Mittel x̄", C.COLOR_ACCENT, "solid"), (a.L, "wahre Grenzen ±L", C.COLOR_TRUE, "dot"), (-a.L, None, C.COLOR_TRUE, "dot")):
        _hline(fig, y, 1, max(m, 2), name or "", color, dash)
        if name is None:
            fig.data[-1].showlegend = False
    fig.update_xaxes(title_text="Nummer der Phase-I-Beobachtung")
    fig.update_yaxes(title_text="Beobachtung (Einheit: wahre Streuung)")
    return _base(fig)


def build_history(a):
    """Bedingte ARL₀ der letzten Stichproben (logarithmische Achse) gegen das Ziel."""
    seeds = [str(sd) for sd, _ in a.history]
    vals = [v for _, v in a.history]
    colors = [C.COLOR_ACCENT if sd == a.seed else C.COLOR_CHART for sd, _ in a.history]
    fig = go.Figure(go.Bar(x=seeds, y=vals, marker_color=colors, hovertemplate="Zufalls-Seed %{x}: bedingte ARL₀ %{y:.1f}<extra></extra>", name="bedingte ARL₀"))
    _hline(fig, C.TARGET_ARL0, seeds[0], seeds[-1], "Ziel 370,4 (bekannte Parameter, L = 3)", C.COLOR_TRUE)      # auf der Kategorieachse mit Kategorien als Enden, nicht −0,5 und n − 0,5
    fig.update_xaxes(title_text="Zufalls-Seed der Phase-I-Stichprobe (orange: die gezeigte)", type="category")
    fig.update_yaxes(title_text="bedingte ARL₀ (Beobachtungen bis zum Alarm)", type="log")
    return _base(fig, 320)


# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

def build_cdf(sp):
    """Exakte Verteilungsfunktion der bedingten ARL₀ (log-Achse) mit Marken Ziel und f·Ziel."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp.a_grid, y=sp.cdf, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="P(bedingte ARL₀ < a)", hovertemplate="a = %{x:.1f}: %{y:.3f}<extra></extra>"))
    _vline(fig, C.TARGET_ARL0, 0, 1, f"Ziel {C.TARGET_ARL0:.1f}".replace(".", ","), C.COLOR_TRUE)
    _vline(fig, sp.threshold, 0, 1, f"{sp.settings.f} des Ziels ({sp.threshold:.1f})".replace(".", ","), C.COLOR_ACCENT)
    fig.update_xaxes(title_text="a = bedingte ARL₀ (Beobachtungen bis zum Alarm)", type="log")
    fig.update_yaxes(title_text="Anteil der Phase-I-Stichproben unter a", range=[0, 1.02])
    return _base(fig)


def build_coupling(sp):
    """Bedingte ARL₀ gegen bedingte ARL₁ (Sprung +1σ) je Phase-I-Stichprobe, mit der Kompromisskurve der zentrierten Karte."""
    n = min(C.SCATTER_POINTS, sp.arl0_samples.size)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=sp.arl0_samples[:n], y=sp.arl1_samples[:n], mode="markers", marker=dict(color="rgba(47,107,101,0.35)", size=4), name=f"{n} Phase-I-Stichproben"))
    fig.add_trace(go.Scatter(x=sp.frontier[0], y=sp.frontier[1], mode="lines", line=dict(color=C.COLOR_TRUE, width=3), name="zentrierte Karte, bekannte Parameter (Kompromisskurve)"))
    fig.update_xaxes(title_text="bedingte ARL₀", type="log")
    fig.update_yaxes(title_text="bedingte ARL₁ (Sprung +1σ)", type="log")
    return _base(fig, 380)


# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

def build_minimum(mn):
    """P(bedingte ARL₀ < f·Ziel) über m (log-Achse), mit der Grenze ε und dem kleinsten m."""
    s = mn.settings
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mn.m_grid, y=mn.curve, mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), marker=dict(size=4),
                             name=f"Anteil unter {s.f} des Ziels ({mn.threshold:.1f})".replace(".", ","), hovertemplate="m = %{x}: %{y:.4f}<extra></extra>"))
    _hline(fig, s.eps_value, float(mn.m_grid[0]), float(mn.m_grid[-1]), f"ε = {s.eps}", C.COLOR_TRUE)
    if mn.m_min is not None:
        _vline(fig, mn.m_min, 0, float(max(mn.curve.max(), 0.01)), f"kleinstes m = {mn.m_min}", C.COLOR_ACCENT)
    fig.update_xaxes(title_text="Phase-I-Beobachtungen m", type="log")
    fig.update_yaxes(title_text="Anteil der Phase-I-Stichproben", range=[0, float(min(1.0, mn.curve.max() * 1.05))])
    return _base(fig)


# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------------

def build_cusum(ca):
    """Verteilungsfunktionen (Anteil der Phase-I-Stichproben unter a) der bedingten ARL₀ von CUSUM und einseitiger Shewhart-Karte."""
    fig = go.Figure()
    for arr, name, color in ((ca.arl_cusum, f"CUSUM (k = {ca.settings.k:g}, h = {ca.h:.4f})".replace(".", ","), C.COLOR_LINE), (ca.arl_shewhart, f"Shewhart einseitig (c = {ca.c1:.4f})".replace(".", ","), C.COLOR_CHART)):
        xs = np.sort(arr)
        fig.add_trace(go.Scatter(x=xs, y=np.arange(1, xs.size + 1) / xs.size, mode="lines", line=dict(color=color, width=3), name=name, hovertemplate="a = %{x:.1f}: %{y:.3f}<extra></extra>"))
    _vline(fig, C.TARGET_ARL0 / 2, 0, 1, f"halbe Ziel-ARL₀ ({C.TARGET_ARL0 / 2:.1f})".replace(".", ","), C.COLOR_ACCENT)
    _vline(fig, C.TARGET_ARL0, 0, 1, "Ziel 370,4", C.COLOR_TRUE)
    fig.update_xaxes(title_text="bedingte ARL₀ der Phase-I-Stichprobe", type="log")
    fig.update_yaxes(title_text="Anteil der Phase-I-Stichproben unter a", range=[0, 1.02])
    return _base(fig)


# --- Experiment 5 -----------------------------------------------------------------------------------------------------------------------------------

def case_label(row):
    if row["kind"] == "keine":
        return "ohne Störung"
    if row["kind"] == "Ausreißer":
        return f"Ausreißer +{row['size']:g}σ".replace(".", ",")
    return f"Stufe {row['size']:g}σ".replace(".", ",")


def build_contamination(res):
    """Median der bedingten ARL₀ je Fall und Schätzer (logarithmische Achse) gegen das Ziel."""
    rows = list(res["cases"])
    chosen = res["chosen"]
    if (chosen["kind"], chosen["size"]) not in {(r["kind"], r["size"]) for r in rows}:
        rows.append(chosen)
    labels = [case_label(r) for r in rows]
    fig = go.Figure()
    for est, color in (("s", C.COLOR_CHART), ("MR", C.COLOR_ACCENT)):
        fig.add_trace(go.Bar(x=labels, y=[r[est][0] for r in rows], marker_color=color, name=f"Schätzer {est}", hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    _hline(fig, C.TARGET_ARL0, labels[0], labels[-1], "Ziel 370,4", C.COLOR_TRUE)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Median der bedingten ARL₀", type="log")
    return _base(fig)
