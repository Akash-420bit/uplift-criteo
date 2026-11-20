"""Uplift evaluation: Qini curve, AUQC, uplift at top-k, bootstrap, profit.

Notation (test set of N users sorted by score, highest first):
  Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k)                 (Qini curve, paper Eq. 2)
  U(k) = Q(k) * k / N_T(k) = k * (ybar_T(k) - ybar_C(k))   (incremental outcomes if
                                                            all top-k users are treated)
  AUQC = integral_0^1 [ Q(phi N)/N - phi Q(N)/N ] d phi     (area between model and
                                                            random Qini curves)
  Profit per 1,000 users: Pi(phi) = 1000 * (V * U(phi N)/N - c * phi)  (paper Eq. 3)

Every function accepts optional bootstrap weights w (counts per user); the
ranking is kept fixed, the user population is resampled.
"""
from __future__ import annotations

import numpy as np


def rank_order(score, seed):
    """Indices sorting users by score, highest first; ties broken at random.

    Bootstrap weights must be drawn in the ORIGINAL user order and then
    permuted with this order, so that all models see the same resample
    (paired bootstrap).
    """
    rng = np.random.default_rng(seed)
    return np.lexsort((rng.random(len(score)), -np.asarray(score, dtype=float)))


def cumulative(y, t, w=None):
    w = np.ones_like(y) if w is None else w.astype(float)
    cw = np.cumsum(w)
    cnt_t = np.cumsum(w * t)
    cnt_c = cw - cnt_t
    y_t = np.cumsum(w * y * t)
    y_c = np.cumsum(w * y * (1 - t))
    return cw, cnt_t, cnt_c, y_t, y_c


def qini_and_uplift_curves(cw, cnt_t, cnt_c, y_t, y_c):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cnt_c > 0, cnt_t / cnt_c, 0.0)
        q = y_t - y_c * ratio
        u = np.where(cnt_t > 0, q * cw / cnt_t, 0.0)
    return q, u


def summarize(y, t, w=None, top_k=(0.1, 0.2, 0.3)):
    """AUQC and uplift among the top-k fractions, for pre-sorted y, t."""
    cw, nt, nc, yt, yc = cumulative(y, t, w)
    q, _ = qini_and_uplift_curves(cw, nt, nc, yt, yc)
    total = cw[-1]
    x = np.concatenate([[0.0], cw / total])
    qn = np.concatenate([[0.0], q / total])
    auqc = np.trapezoid(qn - x * qn[-1], x)
    out = {"auqc": auqc}
    for k in top_k:
        i = min(np.searchsorted(cw, k * total), len(cw) - 1)
        out[f"uplift_top{int(round(k * 100))}"] = yt[i] / nt[i] - yc[i] / nc[i]
    out["ate"] = yt[-1] / nt[-1] - yc[-1] / nc[-1]
    return out


def curve_points(y, t, n_points=101):
    """Qini curve Q/N and uplift curve U/N at evenly spaced fractions."""
    cw, nt, nc, yt, yc = cumulative(y, t)
    q, u = qini_and_uplift_curves(cw, nt, nc, yt, yc)
    n = len(y)
    phi = np.linspace(0, 1, n_points)
    idx = np.clip(np.round(phi * n).astype(int) - 1, 0, n - 1)
    qn, un = q[idx] / n, u[idx] / n
    qn[0] = un[0] = 0.0
    return phi, qn, un


def profit_curve(y, t, V, c, phi_grid, w=None):
    """Profit per 1,000 users at each targeting fraction (pre-sorted y, t)."""
    cw, nt, nc, yt, yc = cumulative(y, t, w)
    _, u = qini_and_uplift_curves(cw, nt, nc, yt, yc)
    total = cw[-1]
    idx = np.clip(np.searchsorted(cw, phi_grid * total), 0, len(cw) - 1)
    un = np.where(phi_grid > 0, u[idx] / total, 0.0)
    return 1000.0 * (V * un - c * phi_grid)


def bootstrap_weights(n, n_boot, seed):
    """Yield multinomial resampling counts (classic non-parametric bootstrap)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n)


def ci(values, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))
