"""lpplattr02_util.py — Hilfsfunktionen: I/O, Preise, Maxima, n-Scatter."""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lpplattr02_params import (
    C2, M2, ALPHA, GAMMA, M, N, DAMPING, SIGN_OU,
    HALVING_INTERVAL_YEARS, HALVING_INTERVAL_DAYS, FORCING_INITIAL,
    FORCING_DECAY, SKIP_FIRST_HALVING, HALVING_ENABLED, NO_HALVING,
    FORK_DAY, FORK_Y1_AMPLITUDE, FORK_Y2_AMPLITUDE, FORK_ENABLED,
    noise_std, base_values, y1_init_value, y2_init_value,
    BATCH_MODE, N_SIMULATIONS, OUTPUT_DIR
)


def calculate_fundamental_price(t):
    return np.exp(C2) * t**M2


def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            data.append((int(row[0]), float(row[1])))
    return np.array(data)


def smooth_prices(prices, window=1):
    return np.convolve(prices, np.ones(window) / window, mode='same')


def tukey(window_length, alpha=0.5):
    if alpha <= 0:   return np.ones(window_length)
    if alpha >= 1:   return np.hanning(window_length)
    x = np.linspace(0, 1, window_length)
    w = np.ones(window_length)
    lo = x < alpha / 2
    hi = x > (1 - alpha / 2)
    w[lo] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[lo] - alpha/2)))
    w[hi] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[hi] - 1 + alpha/2)))
    return w


def find_local_maxima(values, days, threshold=0, min_distance=365):
    indices = (np.diff(np.sign(np.diff(values))) < 0).nonzero()[0] + 1
    indices = indices[values[indices] > threshold]
    if len(indices) <= 1:
        return indices
    filtered, i = [], 0
    while i < len(indices):
        cur_idx, cur_val, cur_day = indices[i], values[indices[i]], days[indices[i]]
        j = i + 1
        while j < len(indices):
            nxt_idx, nxt_day, nxt_val = indices[j], days[indices[j]], values[indices[j]]
            if nxt_day - cur_day < min_distance:
                if nxt_val > cur_val:
                    cur_idx, cur_val, cur_day = nxt_idx, nxt_val, nxt_day
                j += 1
            else:
                break
        filtered.append(cur_idx)
        i = j if j < len(indices) else len(indices)
    return np.array(filtered)


def generate_filename(index):
    return f"{datetime.now().strftime('%y-%m-%d-%H-%M')}-btc-sim-{index:05d}.png"


def save_parameters(output_dir, filename_base, y1_init, y2_init):
    from lpplattr02_params import Z_C, Z_D, Z_E
    with open(os.path.join(output_dir, f"{filename_base}.txt"), 'w') as f:
        f.write(f"C2={C2}  M2={M2}  ALPHA={ALPHA}  GAMMA={GAMMA}  M={M}  N={N}\n")
        f.write(f"Z_C={Z_C}  Z_D={Z_D}  Z_E={Z_E}\n")
        f.write(f"DAMPING={DAMPING}\nSIGN_OU={SIGN_OU}\n")
        f.write(f"HALVING_INTERVAL_DAYS={HALVING_INTERVAL_DAYS}\n")
        f.write(f"FORCING_INITIAL={FORCING_INITIAL}  SKIP_FIRST={SKIP_FIRST_HALVING}\n")
        f.write(f"HALVING_ENABLED={HALVING_ENABLED}  NO_HALVING={NO_HALVING}\n")
        f.write(f"FORK_DAY={FORK_DAY}  FORK_ENABLED={FORK_ENABLED}\n")
        f.write(f"noise_std={noise_std}\ny1_init={y1_init}  y2_init={y2_init}\n")


def process_prices_to_daily_n(days, prices, label):
    result = []
    for i in range(len(days) - 1):
        t1, t2, p1, p2 = days[i], days[i+1], prices[i], prices[i+1]
        if p1 > 0 and p2 > 0 and t2 > t1:
            result.append((t1, np.log(p2/p1) / np.log(t2/t1)))
    return result


def create_daily_n_scatter(days, prices_btc, prices_sim, output_dir, filename_base):
    HALVING_DAYS  = [1425, 2744, 4146, 5586, 7044]
    cycle_labels  = {1: "'13", 2: "'17", 3: "'21", 4: "'25", 5: "'28"}
    colors        = {"'13": '#0000FF', "'17": '#90EE90', "'21": '#FF69B4',
                     "'25": '#FFD700', "'28": '#00CED1'}

    n = min(len(days), len(prices_btc), len(prices_sim))
    days_c = days[:n]
    daily_btc = process_prices_to_daily_n(days_c, prices_btc[:n], "BTC")
    daily_sim = process_prices_to_daily_n(days_c, prices_sim[:n], "Sim")

    def get_cycle(t):
        if t < HALVING_DAYS[0]: return 0
        for i, hd in enumerate(HALVING_DAYS):
            if i == len(HALVING_DAYS)-1: return i+1
            if hd <= t < HALVING_DAYS[i+1]: return i+1
        return len(HALVING_DAYS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    axes = axes.flatten()

    for idx, cycle_num in enumerate([1, 2, 3, 4]):
        ax  = axes[idx]
        ax.set_facecolor('#1a1a1a')
        lbl = cycle_labels[cycle_num]
        btc = [n_ for t, n_ in daily_btc if get_cycle(t) == cycle_num]
        sim = [n_ for t, n_ in daily_sim if get_cycle(t) == cycle_num]
        all_n = btc + sim
        if not all_n: continue
        lo, hi = np.percentile(all_n, [0.5, 99.5])
        bins   = np.linspace(lo, hi, 80)
        bc     = (bins[:-1] + bins[1:]) / 2
        for vals, col, lab, z in [(btc, colors[lbl], f'BTC {lbl}', 5),
                                   (sim, '#808080',    f'Sim {lbl}', 4)]:
            if vals:
                h, _ = np.histogram(vals, bins=bins, density=True)
                mk = h > 0
                ax.scatter(bc[mk], h[mk], facecolor=col, edgecolor='white',
                           linewidth=0.3, alpha=0.9 if z==5 else 0.5,
                           s=15 if z==5 else 10, label=lab, zorder=z)
        ax.set_yscale('log')
        ax.set_xlabel(r'Daily $\Delta n$', fontsize=11)
        ax.set_ylabel('PDF', fontsize=11)
        ax.axvline(0, color='red', lw=0.5, alpha=0.8)
        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(loc='upper right', fontsize=9,
                  facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
        ax.set_title(f"Cycle {lbl} (N={len(btc)}/{len(sim)})",
                     fontsize=11, color='#CCCCCC')

    with plt.rc_context({'text.usetex': False}):
        plt.suptitle("Daily n Distribution by Cycle: BTC vs Simulation",
                     color='#CCCCCC', fontsize=13, y=0.98,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.92, left=0.06, right=0.98, bottom=0.06,
                        hspace=0.25, wspace=0.15)
    out = os.path.join(output_dir, f"{filename_base}_daily_n.png")
    plt.savefig(out, dpi=300, facecolor='#0a0a0a')
    print(f"Daily n scatter saved to: {out}")
    plt.show(block=False)
