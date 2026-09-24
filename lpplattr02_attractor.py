"""lpplattr02_attractor.py — 3D Attraktor: BTC-Residuen (links) vs Sim y1+z (rechts)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import expm as _expm
from lpplattr02_params import (Z_C, Z_D, Z_E, SIGN_OU, HALVING_INTERVAL_DAYS, M2,
                               Z_A, Z_B, Z_MIX, ALPHA, GAMMA,
                               M as _M_lppl, N as _N_lppl)

TAU_ATT      = 30
M_ATT        = 50
START_ATT    = 1164
SMOOTH_SIGMA = 60
CYCLES_JSON  = 'cycles.json'   # Kopie von gold/cycles.json im Repo
HALVING_DAYS = [1425, 2744, 4146, 5586, 7044, 7044 + round(3.9*365)]  # H5 geschätzt

# Zyklusfarben
_CYCLE_COLS = ['grey', '#0000FF', '#90EE90', '#FF69B4', '#FFD700', '#00CED1', '#FF8844']


# ── FTLE (Finite-Time Lyapunov Exponent) ─────────────────────────────────────

def _lppl_jac(y1, y2, z):
    """Analytischer Jacobian des LPPL-ODE-Systems an (y1, y2, z)."""
    J = np.zeros((3, 3))
    J[0, 0] = Z_MIX
    J[0, 1] = 1.0 - Z_A * z - Z_MIX
    J[0, 2] = -Z_A * y2
    J[1, 0] = (-GAMMA * _N_lppl * abs(y1)**(_N_lppl - 1) * np.sign(y1)
               + Z_B * z)
    J[1, 1] = ALPHA * _M_lppl * abs(y2)**(_M_lppl - 1)
    J[1, 2] = Z_B * y1
    J[2, 0] = Z_D + Z_E * y2
    J[2, 1] = Z_E * y1
    J[2, 2] = -Z_C
    return J


def compute_ftle_lppl(y1_arr, y2_arr, z_arr, dt=1.0):
    """FTLE λ1 via QR-Gram-Schmidt. Gibt λ1 in yr⁻¹ zurück."""
    Q = np.eye(3)
    log_sum = 0.0
    for i in range(len(y1_arr)):
        J = _lppl_jac(y1_arr[i], y2_arr[i], z_arr[i])
        Q, R = np.linalg.qr(_expm(dt * J) @ Q)
        log_sum += np.log(max(abs(R[0, 0]), 1e-300))
    return log_sum / (len(y1_arr) * dt / 365.0)


def _cycle_color(day):
    for i, hd in enumerate(HALVING_DAYS):
        if day < hd:
            return _CYCLE_COLS[i]
    return _CYCLE_COLS[len(HALVING_DAYS)]


def _embed(sig, W, M, tau):
    N = len(sig)
    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = sig[W - j*tau : N - j*tau]
    return D


def _pca(D):
    Dc = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(Dc, full_matrices=False)
    return Dc @ Vt.T, s**2 / (s**2).sum()


def _smooth(pc, sigma):
    out = pc.copy()
    for j in range(3):
        out[:, j] = gaussian_filter1d(pc[:, j], sigma=sigma)
    return out


def _style3d(ax):
    ax.set_facecolor('#0a0a0a')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.zaxis.label.set_color('#cccccc')


def _plot3d(ax, pc_s, days_v, last_real_day, title):
    """3D-Plot mit Zyklusfarben, gestrichelter Zukunft, Halving-Markierungen."""
    _style3d(ax)

    # Zyklusfarben pro Punkt
    cols = [_cycle_color(d) for d in days_v]

    # Vergangenheit (solid) und Zukunft (dashed) trennen
    past_mask = days_v <= last_real_day

    # Solid past: farbige Segmente
    pts  = pc_s[:, :3].reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_cols = cols[:-1]
    # Vergangene Segmente
    past_segs  = [s for s, m in zip(segs,  past_mask[:-1]) if m]
    past_cols  = [c for c, m in zip(seg_cols, past_mask[:-1]) if m]
    if past_segs:
        ax.add_collection3d(Line3DCollection(past_segs, colors=past_cols,
                                             lw=2.2, alpha=0.90))
    # Zukünftige Segmente (gestrichelt, gedämpft)
    fut_segs = [s for s, m in zip(segs, past_mask[:-1]) if not m]
    fut_cols = [c for c, m in zip(seg_cols, past_mask[:-1]) if not m]
    if fut_segs:
        ax.add_collection3d(Line3DCollection(fut_segs, colors=fut_cols,
                                             lw=1.4, alpha=0.45, linestyle='dashed'))

    # Achsen-Limits
    for arr, setter in [(pc_s[:, 0], ax.set_xlim),
                        (pc_s[:, 1], ax.set_ylim),
                        (pc_s[:, 2], ax.set_zlim)]:
        p = (arr.max() - arr.min()) * 0.05
        setter(arr.min()-p, arr.max()+p)

    # Halvings markieren
    for i, hd in enumerate(HALVING_DAYS):
        idx = np.argmin(np.abs(days_v - hd))
        if np.abs(days_v[idx] - hd) < 200:
            ax.scatter(pc_s[idx, 0], pc_s[idx, 1], pc_s[idx, 2],
                       color='white', s=35, zorder=10)
            ax.text(pc_s[idx, 0], pc_s[idx, 1], pc_s[idx, 2],
                    f' H{i+1}', color='white', fontsize=7, fontweight='bold')

    with plt.rc_context({'text.usetex': False}):
        ax.set_title(title, color='#cccccc', fontsize=9, pad=4)


def _add_phase_spans(ax, x_max):
    """P1/P2/P3 farbige axvspans + Offset-Labels. 24.09.2026: gleiche Phasen-Uhr wie das Modell
    (echte Halvings, proportional skaliert, lpplattr02_ode.get_days_in_cycle) und nur dort, wo mu(t)
    wirklich aus der Phasentabelle kommt: bei --mu-data gar keine, mit --mu-until nur ab dem Stichtag."""
    from lpplattr02_ode import get_days_in_cycle, get_cycle_number
    from lpplattr02_params import MU_FROM_DATA

    def _table_active(t):
        if not MU_FROM_DATA["enabled"]:
            return True
        return MU_FROM_DATA["until"] is not None and t >= MU_FROM_DATA["until"]

    def _phase(t):
        d = get_days_in_cycle(t)
        return 'P1' if 100 <= d <= 550 else ('P2' if 550 < d <= 950 else 'P3')

    COLORS = {'P1': '#00CC44', 'P2': '#FF4444', 'P3': '#4488FF'}
    trans  = ax.get_xaxis_transform()   # x=Daten, y=Achsenfrakt. 0-1

    segs, start, key = [], None, None
    for t in np.arange(1.0, x_max + 1.0):
        k = (_phase(t), get_cycle_number(t)) if _table_active(t) else None
        if k != key:
            if key is not None:
                segs.append((start, t, key))
            start, key = t, k
    if key is not None:
        segs.append((start, x_max, key))

    for s, e, (phase, cyc) in segs:
        col = '#888888' if cyc == 0 else COLORS[phase]   # 24.09.2026: pre-H1 grau
        ax.axvspan(s, e, color=col, alpha=0.15, lw=0)
        if (e - s) > 150:
            offset = SIGN_OU["mu_offset_by_cycle"].get(cyc, {}).get(phase, 0)
            ax.text((s + e) / 2, 0.97, f'{offset:+.1f}', transform=trans,
                    fontsize=8, ha='center', va='top',
                    color=col, alpha=0.9, clip_on=True)


def build_attractor_figure(days_real, x_values_real, days, x_arr_sim, z_arr, y2_arr_sim=None, zero_z=False):
    """
    Linkes 3D:  echte BTC-Residuen
    Rechtes 3D: Simulation y1 + z
    """
    _W          = (M_ATT - 1) * TAU_ATT
    last_real   = int(days_real[-1])

    # ── Embedding BTC (real) ─────────────────────────────────────────────────
    mask_r      = days_real >= START_ATT
    lr          = x_values_real[mask_r]
    days_r      = days_real[mask_r]
    Dr          = _embed(lr, _W, M_ATT, TAU_ATT)
    pc_r, var_r = _pca(Dr)
    pc_rs       = _smooth(pc_r, SMOOTH_SIGMA)
    days_rv     = days_r[_W:]

    # ── Embedding Sim y1 + z (multivariat, volle Zeitachse inkl. Zukunft) ────
    mask_s   = days >= START_ATT
    days_s   = days[mask_s]
    y1_s     = x_arr_sim[mask_s]
    z_s      = z_arr[mask_s]
    N_s      = len(y1_s)

    # ── FTLE λ1 (beide Attraktoren) ──────────────────────────────────────────
    ftle_real = None
    ftle_sim  = None

    # Real BTC: y2 ≈ numerische Ableitung, z ≈ Forward-Euler
    try:
        y2_r = np.gradient(lr, 1.0)
        z_r  = np.zeros_like(lr)
        for i in range(1, len(lr)):
            z_r[i] = (z_r[i-1]
                      + (-Z_C * z_r[i-1]
                         + Z_D * lr[i-1]
                         + Z_E * lr[i-1] * y2_r[i-1]))
        ftle_real = compute_ftle_lppl(lr, y2_r, z_r)
    except Exception as _fe:
        print(f'FTLE real Fehler: {_fe}')

    # Sim: y2 direkt aus Simulation
    if y2_arr_sim is not None:
        try:
            past_s = days[mask_s] <= last_real
            ftle_sim = compute_ftle_lppl(
                y1_s[past_s], y2_arr_sim[mask_s][past_s], z_s[past_s])
        except Exception as _fe:
            print(f'FTLE sim Fehler: {_fe}')

    Dsim = np.empty((N_s - _W, 2 * M_ATT))
    for j in range(M_ATT):
        Dsim[:, j]       = y1_s[_W - j*TAU_ATT : N_s - j*TAU_ATT]
        Dsim[:, M_ATT+j] = z_s [_W - j*TAU_ATT : N_s - j*TAU_ATT]
    pc_sim, var_sim = _pca(Dsim)
    pc_sims  = _smooth(pc_sim, SMOOTH_SIGMA)
    days_sv  = days_s[_W:]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig   = plt.figure(figsize=(20, 10), facecolor='#0a0a0a')
    ax_r  = fig.add_axes([0.03, 0.74, 0.94, 0.22])          # Residuen oben, flach
    ax_3b = fig.add_axes([0.02, 0.03, 0.46, 0.66], projection='3d')   # BTC real
    ax_3s = fig.add_axes([0.52, 0.03, 0.46, 0.66], projection='3d')   # Sim y1+z

    # ── Residuals oben ────────────────────────────────────────────────────────
    ax_r.set_facecolor('#1a1a1a')
    ax_r.plot(days_real, np.exp(x_values_real), color='orange',  lw=0.8, alpha=0.85, label='BTC')
    ax_r.plot(days,      np.exp(x_arr_sim),      color='#6688cc', lw=0.6, alpha=0.55, label='Sim')
    ax_r.set_yscale('log')
    ax_r.axhline(1, color='#555', lw=0.8, ls='--')
    ax_r.set_ylabel('rel. Value', color='#ccc', fontsize=9)
    ax_r.set_xlabel('Day', color='#ccc', fontsize=9)
    ax_r.set_title('', color='#ccc', fontsize=10)
    ax_r.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='#ccc', edgecolor='#444')
    ax_r.grid(True, alpha=0.3, ls='--', lw=0.4)
    ax_r.tick_params(colors='#999', labelsize=8)
    for hd in HALVING_DAYS:
        ax_r.axvline(hd, color='#555', lw=0.7, ls='--', alpha=0.7)

    try:
        _add_phase_spans(ax_r, float(days[-1]))
    except Exception as _e:
        print(f"Phase-Spans Fehler (ignoriert): {_e}")

    # ── 3D Plots ──────────────────────────────────────────────────────────────
    _plot3d(ax_3b, pc_rs,   days_rv, last_real,
            f'BTC real  PC1={var_r[0]*100:.1f}%  PC2={var_r[1]*100:.1f}%  PC3={var_r[2]*100:.1f}%')
    _plot3d(ax_3s, pc_sims, days_sv, last_real,
            f'Sim y1+z  PC1={var_sim[0]*100:.1f}%  PC2={var_sim[1]*100:.1f}%  PC3={var_sim[2]*100:.1f}%')
    ax_3s.view_init(elev=34, azim=-135)

    _title2 = "Attractor from Price-Residuals" + (" (decoupled from z)" if zero_z else "")
    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(_title2,
                     color='#CCCCCC', fontsize=13, fontname='Comfortaa',
                     fontweight='bold', y=0.99)
    return fig
