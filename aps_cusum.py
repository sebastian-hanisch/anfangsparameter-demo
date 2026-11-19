"""Einseitiger CUSUM (nach oben) mit geschätzten Parametern: bedingte ARL₀ je Phase-I-Stichprobe, exakt (numpy, scipy).

Statistik C_t = max(0, C_{t−1} + z_t − k), Start C_0 = 0, Alarm bei C_t > h. Mit Grenzen aus Phase I wird die Beobachtung x zu z = (x − x̄)/s; für die wahre Reihe N(0, 1) ist
z ~ N(−x̄/s, 1/s²). Die ARL ab Null löst eine Integralgleichung zweiter Art (Nyström, Gauss-Legendre); der zweite exakte Weg ist das Gitter von Brook und Evans (Markov-Kette auf
Zellen), der dritte die Simulation. Alle Zahlen gelten für zero-state (Start bei 0) und den einseitigen CUSUM nach oben; „zweiseitig“ wäre hier nur eine Näherung und ist nicht gerechnet."""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

import aps_constants as C


def gl_nodes(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def cusum_arl_integral(mu, sd, k, h, n=C.CUSUM_NODES, chunk=400):
    """ARL ab C_0 = 0 des einseitigen CUSUM, Beobachtungen z ~ N(mu, sd²), je Eintrag von mu/sd (Stapel). Unbekannt: L(0) und L an den n Knoten (Nyström)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sd = np.atleast_1d(np.asarray(sd, dtype=float))
    if sd.size == 1 and mu.size > 1:
        sd = np.full(mu.size, sd[0])
    v, w = gl_nodes(0.0, h, n)
    u = np.concatenate([[0.0], v])
    out = np.empty(mu.size)
    for a in range(0, mu.size, chunk):
        m_ = mu[a:a + chunk]
        s_ = sd[a:a + chunk]
        A = np.empty((m_.size, n + 1, n + 1))
        A[:, :, 0] = ndtr((k - u[None, :] - m_[:, None]) / s_[:, None])
        A[:, :, 1:] = (w[None, None, :] * np.exp(-0.5 * ((v[None, None, :] + k - u[None, :, None] - m_[:, None, None]) / s_[:, None, None]) ** 2)
                       / (math.sqrt(2.0 * math.pi) * s_[:, None, None]))
        L = np.linalg.solve(np.eye(n + 1)[None] - A, np.ones((m_.size, n + 1, 1)))[:, :, 0]
        out[a:a + chunk] = L[:, 0]
    return out


def cusum_chain(cdf, k, h, n):
    """Brook/Evans: Zustand 0 (C = 0) und n gleich breite Zellen über (0, h], Zellenmitte als Vertreter; Übergangsmatrix Q der Größe (n + 1) × (n + 1) zwischen den Zuständen,
    der Alarm ist der Rest. `cdf` ist die Verteilungsfunktion von z."""
    width = h / n
    reps = np.concatenate([[0.0], (np.arange(n) + 0.5) * width])
    edges = np.arange(n + 1) * width
    F = cdf(k - reps[:, None] + edges[None, :])
    Q = np.empty((n + 1, n + 1))
    Q[:, 0] = cdf(k - reps)
    Q[:, 1:] = F[:, 1:] - F[:, :-1]
    return np.maximum(Q, 0.0)


def chain_arl(Q, start=0):
    """ARL ab dem Zustand start: (I − Q) L = 1."""
    return float(np.linalg.solve(np.eye(Q.shape[0]) - Q, np.ones(Q.shape[0]))[start])


def chain_survival(Q, n_max, start=0):
    """P(Lauflänge > n) für n = 0 … n_max aus e₀ᵀQⁿ1 (Überlebenswahrscheinlichkeit der Kette)."""
    s = np.zeros(n_max + 1)
    v = np.ones(Q.shape[0])
    s[0] = v[start]
    for i in range(1, n_max + 1):
        v = Q @ v
        s[i] = v[start]
    return s


def arl_known(k, h, n=C.CUSUM_NODES_TARGET):
    """ARL₀ bei bekannten Parametern (z ~ N(0, 1))."""
    return float(cusum_arl_integral(0.0, 1.0, k, h, n)[0])


def solve_h(k, target=C.TARGET_ARL0, n=C.CUSUM_NODES_TARGET):
    """Schwelle h mit ARL₀ = target bei bekannten Parametern (einseitig; k = 0,5, Ziel 370,398: h = 4,0965; Ziel 370: 4,0954)."""
    return float(brentq(lambda h: arl_known(k, h, n) - target, 0.5, 12.0, xtol=1e-10))


def shewhart_one_sided_limit(target=C.TARGET_ARL0):
    """Grenzfaktor c der einseitigen Shewhart-Karte (Grenze x̄ + c·s) mit ARL₀ = target bei bekannten Parametern: −Φ⁻¹(1/target) = 2,7822."""
    return -float(ndtri(1.0 / target))


def shewhart_one_sided_arl(xb, s, c):
    """Bedingte ARL der einseitigen Shewhart-Karte (Alarm bei x > x̄ + c·s), wahre Beobachtung N(0, 1): 1/Φ(−x̄ − c s)."""
    return 1.0 / ndtr(-np.asarray(xb) - c * np.asarray(s))


def simulate_cusum(normals, xb, s, k, h, runs, horizon):
    """Simulation der Karte mit festen Grenzen (x̄, s) und wahrer Reihe N(0, 1): Lauflängen von `runs` Läufen. `normals(step)` liefert die Normalzahlen
    (Länge `runs`) des Schritts; Läufe ohne Alarm bis `horizon` werden mit `horizon` gezählt (zensiert). Rückgabe: (Mittel, Standardfehler, Zahl zensierter Läufe)."""
    C_t = np.zeros(runs)
    length = np.full(runs, float(horizon))
    active = np.ones(runs, dtype=bool)
    for step in range(1, horizon + 1):
        x = normals(step)
        z = (x - xb) / s
        C_t = np.where(active, np.maximum(0.0, C_t + z - k), C_t)
        fired = active & (C_t > h)
        length[fired] = step
        active &= ~fired
        if not active.any():
            break
    censored = int(active.sum())
    return float(length.mean()), float(length.std(ddof=1) / math.sqrt(runs)), censored
