#!/usr/bin/env python3
"""lppl-simple.py — Test: vereinfachte DGL, eigenstaendig, KEIN Import aus lpplattr02*.

Minimalmodell (24.09.2026):
    y1 = ln(P / PLM),   y2 = dy1/dt,   n = M2 + t*y2   (lokaler Exponent)
    Kern (1 Tag, RK4):      dy1 = y2
                            dy2 = -gamma * y1 * |y1|^(N-1)           [+ alpha*y2*|y2|^(M-1) mit --with-alpha]
    Sign-OU (2 x 1/2 Tag):  dn  = -kappa * sign(n - mu(t)) dt + sigma dW,   sigma = sqrt(2*kappa*b_cycle)
                            in y2-Form: y2' = (y2 - dt*kappa*sign(X)/t) / (1 + dt/t),  y1 unveraendert
    mu(t) = M2 + Offset(Zyklus, Phase), Phasen 100/550/950 auf echten Halvings, proportional skaliert.
Kein z, keine Daempfung (optional), keine Kicks. Gleicher Strang-Split wie lpplattr02 (1/2 sign, Kern, 1/2 sign),
damit das Ergebnis vergleichbar ist.

Aufruf:  ./lppl-simple.py [--zero-sigma] [--seed N] [--b-half|--b-quarter] [--with-alpha] [--no-plot] [--save out.npy]
"""
import argparse, csv, os
import numpy as np

# ── Konstanten (Werte wie lpplattr02_params.py, hier bewusst abgeschrieben) ──────────────────────
C2, M2 = -37.7440, 5.6574                     # Power-Law Mean (Basis der Simulation)
C2_PLB, M2_PLB = -40.4617, 5.90573            # Power-Law Bottom (nur Plot)
GAMMA, N_EXP = 0.003, 3.0                     # Feder
ALPHA, M_EXP = -0.00078, 1.05 * 1.02          # Daempfung (nur mit --with-alpha)
KAPPA = 108.62                                # Sign-Drift
B_BY_CYCLE = {0: 58.2, 1: 58.2, 2: 96.8, 3: 108.9, 4: 103.4, 5: 103.4}
OFFSETS = {0: (0.0, 3.29, -1.28), 1: (4.15, -9.29, 2.42), 2: (15.30, -16.11, -0.51),
           3: (9.43, -17.29, 7.50), 4: (-0.43, -2.0, 2.0), 5: (4.0, -4.0, 1.0)}
HALVING = [1425, 2744, 4146, 5586, 7044]
L_REF = 3.815 * 365
DT_SUB, N_SUB = 0.05, 10                      # Sign-Substeps je halber Tag


def cycle_number(t):
    return sum(1 for h in HALVING if t >= h)


def days_in_cycle(t):
    """Position im Zyklus in Referenz-Tagen: echte Halvings, proportional auf L_REF skaliert."""
    edges = [0] + HALVING
    for i in range(len(edges) - 1, -1, -1):
        if t >= edges[i]:
            L = (edges[i + 1] - edges[i]) if i + 1 < len(edges) else L_REF
            return (t - edges[i]) * L_REF / L
    return t


def mu_of(t):
    p1, p2, p3 = OFFSETS.get(cycle_number(t), (0.0, 0.0, 0.0))
    d = days_in_cycle(t)
    if 100 <= d <= 550:
        return M2 + p1
    if 550 < d <= 950:
        return M2 + p2
    return M2 + p3


def sigma_of(t, bscale):
    return np.sqrt(2.0 * KAPPA * B_BY_CYCLE.get(cycle_number(t), 58.2) * bscale)


def sign_half_day(y1, y2, t0, bscale, rng):
    t = t0
    for _ in range(N_SUB):
        X = (M2 + t * y2) - mu_of(t)
        s = np.sign(X)
        y2 = (y2 - DT_SUB * KAPPA * s / t) / (1.0 + DT_SUB / t)     # y1 bleibt (kein Doppelintegral)
        sig = sigma_of(t, bscale)
        if sig > 0.0:
            y2 += (sig / (t + DT_SUB)) * np.sqrt(DT_SUB) * rng.normal()   # /(t+dt): n bekommt exakt sigma*sqrt(dt)*xi
        t += DT_SUB
    return y1, y2


def core_rhs(y1, y2, with_alpha):
    dy2 = -GAMMA * y1 * abs(y1) ** (N_EXP - 1)
    if with_alpha:
        dy2 += ALPHA * y2 * abs(y2) ** (M_EXP - 1)
    return y2, dy2


def core_day(y1, y2, with_alpha, h=1.0):
    """Ein Tag RK4 fuer den Kern."""
    k1 = core_rhs(y1, y2, with_alpha)
    k2 = core_rhs(y1 + 0.5 * h * k1[0], y2 + 0.5 * h * k1[1], with_alpha)
    k3 = core_rhs(y1 + 0.5 * h * k2[0], y2 + 0.5 * h * k2[1], with_alpha)
    k4 = core_rhs(y1 + h * k3[0], y2 + h * k3[1], with_alpha)
    y1 += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y2 += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return y1, y2


def load_data():
    for f in ["ziel.csv", "../ziel.csv", "../../ziel.csv"]:
        if os.path.exists(f):
            rows = [(float(r[0]), float(r[1])) for r in csv.reader(open(f), delimiter=' ') if float(r[1]) > 0]
            t = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
            return f, t, p
    raise FileNotFoundError("ziel.csv not found")


def main():
    ap = argparse.ArgumentParser(description='Minimalmodell: Sign-OU auf dem Exponenten + Feder, eigenstaendig')
    ap.add_argument('--zero-sigma', action='store_true', help='Rauschen aus (deterministisch)')
    ap.add_argument('--b-half', action='store_true', help='b_by_cycle / 2')
    ap.add_argument('--b-quarter', action='store_true', help='b_by_cycle / 4')
    ap.add_argument('--seed', type=int, default=None, help='Seed (Standard: zufaellig)')
    ap.add_argument('--with-alpha', action='store_true', help='Daempfungsterm alpha*y2|y2|^(M-1) mitnehmen')
    ap.add_argument('--days', type=int, default=9000, help='Horizont in Tagen')
    ap.add_argument('--no-plot', action='store_true', help='Plotfenster unterdruecken (Standard: Plot zeigen)')
    ap.add_argument('--save', type=str, default=None, help='y1(t) als .npy speichern (Spalten t, y1, n)')
    a = ap.parse_args()

    bscale = 0.0 if a.zero_sigma else (0.25 if a.b_quarter else (0.5 if a.b_half else 1.0))
    seed = a.seed if a.seed is not None else int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    f, t_real, p_real = load_data()
    x_real = np.log(p_real / (np.exp(C2) * t_real ** M2))
    days = np.concatenate([t_real, np.arange(int(t_real[-1]) + 1, a.days + 1)])

    y1, y2 = 0.0, 0.0
    Y1 = np.empty(len(days)); NN = np.empty(len(days))
    for i, t in enumerate(days):
        Y1[i] = y1; NN[i] = M2 + t * y2
        y1, y2 = sign_half_day(y1, y2, t, bscale, rng)
        y1, y2 = core_day(y1, y2, a.with_alpha)
        y1, y2 = sign_half_day(y1, y2, t + 0.5, bscale, rng)

    n_real = len(t_real); m1 = t_real >= 1425
    def r2(s, r): return 1 - np.sum((s - r) ** 2) / np.sum((r - r.mean()) ** 2)
    print(f"Daten: {f}  ({int(t_real[0])}..{int(t_real[-1])})   Seed={seed}   b-Skalierung={bscale}   alpha={'an' if a.with_alpha else 'aus'}")
    # R2 aufgespalten in pre-H1 (Tag < 1425) und post-H1; Residuum und Preis (log-Raum, PLM*e^y1 bzw. PLM-Linie)
    pre = ~m1; post = m1
    lnP_btc = np.log(p_real); lnPLM = C2 + M2 * np.log(t_real); ys = Y1[:n_real]
    R2 = {'res sim': (r2(ys[pre], x_real[pre]), r2(ys[post], x_real[post])),
          'price sim': (r2(lnPLM[pre] + ys[pre], lnP_btc[pre]), r2(lnPLM[post] + ys[post], lnP_btc[post])),
          'price PLM': (r2(lnPLM[pre], lnP_btc[pre]), r2(lnPLM[post], lnP_btc[post]))}
    print("R2  pre-H1 | post-H1:  " + "   ".join(f"{k}: {v[0]:.3f} | {v[1]:.3f}" for k, v in R2.items()))
    print("y1 an Peaks 2011/2013/2017/2021: " + "  ".join(f"{Y1[np.argmin(np.abs(days - p))]:+.2f}" for p in [888, 1796, 3269, 4682])
          + "   (BTC " + "  ".join(f"{x_real[np.argmin(np.abs(t_real - p))]:+.2f}" for p in [888, 1796, 3269, 4682]) + ")")
    if a.save:
        np.save(a.save, np.column_stack([days, Y1, NN]))
        print(f"gespeichert: {a.save}")

    if not a.no_plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        plt.style.use(next(f for f in ['hz.mplstyle', '../hz.mplstyle', '/home/hz/Data/hz.mplstyle'] if os.path.exists(f)))
        mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']
        fig = plt.figure(figsize=(14, 9.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8])
        ax1, ax2 = [fig.add_subplot(gs[i]) for i in range(2)]
        for ax in (ax1, ax2):
            ax.set_facecolor('#1a1a1a')
        # Panel 1: Preis
        plm = np.exp(C2) * days ** M2
        plb = np.exp(C2_PLB) * days ** M2_PLB
        ax1.plot(t_real, p_real, color='orange', lw=1.0, label='BTC')
        ax1.plot(days, plm * np.exp(Y1), color='#44AAFF', lw=1.0, label='Simple model')
        ax1.plot(days, plm, color='#CCCCCC', ls='--', lw=0.9, alpha=0.8, label='PLM (base)')
        ax1.plot(days, plb, color='orange', ls=':', lw=0.9, alpha=0.8, label='PLB')
        ax1.set_yscale('log'); ax1.set_xlim(0, a.days); ax1.set_ylim(1e-2, 1e6)
        ax1.set_ylabel('Price [USD]')
        ax1.set_title(r'\textbf{Price: BTC vs. Simple Model on the Power-Law Mean}', fontsize=12, pad=4)
        ax1.plot([], [], ' ', label=f'Seed={seed}')
        ax1.plot([], [], ' ', label=r'$R^2$: pre-H1 $|$ post-H1')
        ax1.plot([], [], ' ', label=r'$R^2_{\rm price}$ (sim) = ' + f'{R2["price sim"][0]:.3f} $|$ {R2["price sim"][1]:.3f}')
        ax1.plot([], [], ' ', label=r'$R^2_{\rm price}$ (PL Mean) = ' + f'{R2["price PLM"][0]:.3f} $|$ {R2["price PLM"][1]:.3f}')
        ax1.legend(loc='upper left', fontsize=10, facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
        # Panel 2: Residuen
        ax2.plot(t_real, x_real, color='orange', lw=0.8, alpha=0.8, label='BTC residual $y=\\ln(P/\\mathrm{PLM})$')
        ax2.plot(days, Y1, color='#44AAFF', lw=1.0, label='Simple model $y_1$')
        ax2.axhline(0, color='#555555', lw=0.5, ls='--')
        ax2.plot([], [], ' ', label=r'$R^2_{\rm res}$ = ' + f'{R2["res sim"][0]:.2f} $|$ {R2["res sim"][1]:.2f}  (pre $|$ post H1)')
        ax2.set_xlim(0, a.days); ax2.set_xlabel('Days since Genesis Block'); ax2.set_ylabel(r'$y_1$ (log residual)')
        ax2.set_title(r'\textbf{Residuals}', fontsize=12, pad=4)
        ax2.legend(loc='upper left', fontsize=10, facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
        for ax in (ax1, ax2):
            for h in HALVING:
                ax.axvline(h, color='#808080', ls='--', lw=0.8, alpha=0.6)
        ax2.sharex(ax1)
        with plt.rc_context({'text.usetex': False}):
            plt.suptitle('Simple Model: Sign-OU Exponent Drive + Cubic Spring', color='#CCCCCC',
                         fontsize=14, y=0.985, fontname='Comfortaa', fontweight='bold')
        plt.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.93, hspace=0.3)
        plt.show()


if __name__ == '__main__':
    main()
