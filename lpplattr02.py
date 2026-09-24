#!/usr/bin/env python3
"""lpplattr02.py — Hauptprogramm: Simulation + Plot + Attraktor."""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta

from lpplattr02_params import (
    C2, C2_PLB, M2, M2_PLB, Z_C, Z_D, Z_E,
    y1_init_value, y2_init_value, z_init_value,
    SIGN_OU, HALVING_INTERVAL_DAYS, FORCING_INITIAL, NO_HALVING,
    SKIP_FIRST_HALVING, HALVING_ENABLED, FORK_ENABLED, FORK_DAY,
    FORK_Y1_AMPLITUDE, FORK_Y2_AMPLITUDE,
    base_values, noise_std, BATCH_MODE, N_SIMULATIONS, OUTPUT_DIR, M2, MU_FROM_DATA
)
from lpplattr02_ode    import rk8_step, sign_step_substeps
from lpplattr02_util   import (read_data, smooth_prices, calculate_fundamental_price,
                                find_local_maxima, save_parameters, create_daily_n_scatter)
from lpplattr02_attractor import build_attractor_figure

try:
    plt.style.use(next(f for f in ['hz.mplstyle', '../hz.mplstyle', '/home/hz/Data/hz.mplstyle'] if os.path.exists(f)))
    plt.rcParams['figure.facecolor'] = '#0a0a0a'
    plt.rcParams['axes.facecolor']   = '#1a1a1a'
    mpl.rcParams['font.sans-serif']  = ['Comfortaa', 'DejaVu Sans', 'Arial']
except Exception:
    pass


def resolve_data_file():
    data_file = next((f for f in ["ziel.csv", "../ziel.csv", "../../ziel.csv"]
                      if os.path.exists(f)), None)
    if data_file is None:
        raise FileNotFoundError("ziel.csv not found")
    return data_file


def prepare_reference_data(days_end=9000):
    # ── Daten laden ───────────────────────────────────────────────────────────
    data_file = resolve_data_file()
    data                 = read_data(data_file)
    days_real            = data[:, 0]
    prices_real          = data[:, 1]
    smoothed_prices_real = smooth_prices(prices_real)

    days_mask            = days_real <= days_end
    days_real            = days_real[days_mask]
    prices_real          = prices_real[days_mask]
    smoothed_prices_real = smoothed_prices_real[days_mask]

    max_day_real = int(days_real[-1])
    days = np.concatenate([days_real, np.arange(max_day_real+1, days_end + 1)]) \
           if max_day_real < days_end else days_real

    fundamental_prices      = calculate_fundamental_price(days)
    fundamental_prices_real = fundamental_prices[:len(days_real)]
    x_values                = np.log(smoothed_prices_real / fundamental_prices_real)

    return {
        'data_file': data_file,
        'days_real': days_real,
        'prices_real': prices_real,
        'smoothed_prices_real': smoothed_prices_real,
        'days': days,
        'fundamental_prices': fundamental_prices,
        'fundamental_prices_real': fundamental_prices_real,
        'x_values': x_values,
    }


def generate_noise_sequences(days, seed):
    np.random.seed(seed)
    noise_sequences = [
        {k: np.random.normal(base_values[k], noise_std[k]) for k in base_values}
        for _ in days
    ]
    _ = [np.random.normal(0, 1) for _ in days]   # sign_ou_noise alignment
    return noise_sequences


def simulate_without_forcing(days, noise_sequences):
    y_nf = np.array([y1_init_value, y2_init_value, z_init_value], dtype=float)
    sim_x_nf = []
    state_history = []
    for i, t in enumerate(days):
        state_history.append(y_nf.copy())
        sim_x_nf.append(y_nf[0])
        y_nf = rk8_step(y_nf, 1.0, noise_sequences[i], float(t))
    return {
        'x': np.array(sim_x_nf),
        'states': np.array(state_history),
        'final_state': y_nf.copy(),
    }


def simulate_with_forcing(days, noise_sequences):
    y_wf = np.array([y1_init_value, y2_init_value, z_init_value], dtype=float)
    sim_x_wf, z_wf_list, y2_wf_list = [], [], []
    state_history = []

    for i, t_day in enumerate(days):
        # Halving-Kicks
        for n in range(10):
            if abs(t_day - n * HALVING_INTERVAL_DAYS) < 0.5:
                if not (n == 0 and SKIP_FIRST_HALVING) and HALVING_ENABLED[n]:
                    f = FORCING_INITIAL if NO_HALVING else FORCING_INITIAL * 0.5**n
                    y_wf[1] += f
                break
        if FORK_ENABLED and abs(t_day - FORK_DAY) < 0.5:
            y_wf[0] += FORK_Y1_AMPLITUDE
            y_wf[1] += FORK_Y2_AMPLITUDE

        state_history.append(y_wf.copy())
        sim_x_wf.append(y_wf[0])
        z_wf_list.append(y_wf[2])
        y2_wf_list.append(y_wf[1])

        tc = float(t_day)
        n_sub = 10
        if SIGN_OU["enabled"]:
            y_wf = sign_step_substeps(y_wf, tc, 0.5/n_sub, n_sub)
        y_wf = rk8_step(y_wf, 1.0, noise_sequences[i], tc,
                        use_damping=True, use_sign_ou=False)
        if SIGN_OU["enabled"]:
            y_wf = sign_step_substeps(y_wf, tc + 0.5, 0.5/n_sub, n_sub)

    return {
        'x': np.array(sim_x_wf),
        'z': np.array(z_wf_list),
        'y2': np.array(y2_wf_list),
        'states': np.array(state_history),
        'final_state': y_wf.copy(),
    }


def run_simulation_and_plot(
    sim_index=None,
    seed=5,
    zero_z=False,
    days_end=9000,
    show_plot=True,
    save_plot=True,
    return_data=False,
):
    reference = prepare_reference_data(days_end=days_end)
    data_file = reference['data_file']
    days_real = reference['days_real']
    smoothed_prices_real = reference['smoothed_prices_real']
    days = reference['days']
    fundamental_prices = reference['fundamental_prices']
    fundamental_prices_real = reference['fundamental_prices_real']
    x_values = reference['x_values']

    data_file = next((f for f in ["ziel.csv", "../ziel.csv", "../../ziel.csv"]
                      if os.path.exists(f)), None)
    if data_file is None:
        raise FileNotFoundError("ziel.csv not found")
    print(f"Using data file: {data_file}")
    noise_sequences = generate_noise_sequences(days, seed)
    y1_init, y2_init = y1_init_value, y2_init_value

    try:
        no_forcing = simulate_without_forcing(days, noise_sequences)
        with_forcing = simulate_with_forcing(days, noise_sequences)

        # ── 24.09.2026: R² der Residuen, mit z-Puffer (dieser Lauf) und ohne (Z_A=Z_B=Z_MIX=0),
        #    zweiter Lauf mit identischem Seed/RNG-Zustand; R² gegen ln(P/PLM) der echten Daten ──
        def _r2(sim, ref):
            return 1.0 - np.sum((sim - ref)**2) / np.sum((ref - ref.mean())**2)
        _nreal = len(days_real)
        r2_z = _r2(with_forcing['x'][:_nreal], x_values)
        import lpplattr02_ode as _ode
        _zsave = (_ode.Z_A, _ode.Z_B, _ode.Z_MIX)
        _ode.Z_A = _ode.Z_B = _ode.Z_MIX = 0.0
        _xnz = simulate_with_forcing(days, generate_noise_sequences(days, seed))['x'][:_nreal]
        _ode.Z_A, _ode.Z_B, _ode.Z_MIX = _zsave
        # aufgespalten in pre-H1 (Tag < 1425) und post-H1 (Tag >= 1425)
        _pre = days_real < 1425; _post = ~_pre
        _xz = with_forcing['x'][:_nreal]
        _lnP = np.log(smoothed_prices_real); _lnPLM = np.log(fundamental_prices_real)
        R2 = {}
        for _lab, _sim, _ref in [('res z on', _xz, x_values), ('res z off', _xnz, x_values),
                                 ('price sim', _lnPLM + _xz, _lnP), ('price PLM', _lnPLM, _lnP)]:
            R2[_lab] = (_r2(_sim[_pre], _ref[_pre]), _r2(_sim[_post], _ref[_post]))
        print(">>> R²  pre-H1 | post-H1:  " + "   ".join(f"{k}: {v[0]:.3f} | {v[1]:.3f}" for k, v in R2.items()))

        # ── Arrays ────────────────────────────────────────────────────────────
        x_arr   = with_forcing['x']
        z_arr   = with_forcing['z']
        y2_arr  = with_forcing['y2']
        sim_p   = fundamental_prices * np.exp(x_arr)
        bub     = (sim_p - fundamental_prices) / fundamental_prices
        bub_nf  = (fundamental_prices * np.exp(no_forcing['x']) - fundamental_prices) \
                  / fundamental_prices

        # CSV export (Sim-Lap/)
        sim_dir = "Sim-Lap"
        os.makedirs(sim_dir, exist_ok=True)
        csv_mask = days <= days_end
        with open(os.path.join(sim_dir, "sim_oscillation.csv"), "w") as f:
            for d, v in zip(days[csv_mask], (np.exp(x_arr)-1)[csv_mask]):
                f.write(f"{int(d)} {v}\n")
        with open(os.path.join(sim_dir, "sim_z.csv"), "w") as f:
            for d, v in zip(days[csv_mask], z_arr[csv_mask]):
                f.write(f"{int(d)} {v}\n")
        # ziel.csv-Format pro Seed (für SSM-Analyse): "day price DD.MM.YYYY"
        # Kalibriert an Attractor/ziel.csv: day 394 = 01.02.2010
        # → Day 1 = 04.01.2009 = Tag NACH Bitcoin Genesis (3 Jan 2009)
        _day1_date = datetime(2009, 1, 4)
        ziel_csv = os.path.join(sim_dir, f"sim_ziel_seed{seed}.csv")
        with open(ziel_csv, "w") as f:
            for d, p in zip(days[csv_mask], sim_p[csv_mask]):
                _date = _day1_date + timedelta(days=int(d) - 1)
                f.write(f"{int(d)} {p:.10g} {_date.strftime('%d.%m.%Y')}\n")

        # ── Bubble fit ────────────────────────────────────────────────────────
        top_idx = find_local_maxima(bub, days, threshold=0.5, min_distance=600)
        if len(top_idx) >= 2:
            lf        = np.polyfit(days[top_idx], np.log(bub[top_idx]), 1)
            fit_line  = np.exp(np.polyval(lf, days))
            half_life = -np.log(2) / lf[0] / 365
        else:
            fit_line  = np.ones_like(days);  half_life = np.nan

        bub_real     = (smoothed_prices_real - fundamental_prices_real) / fundamental_prices_real
        pk_days_real = days_real[np.clip(np.searchsorted(days_real, [1796, 3269, 4682]),
                                         0, len(days_real)-1)]
        pk_bub_real  = bub_real[np.searchsorted(days_real, pk_days_real)]
        lf_real      = np.polyfit(pk_days_real, np.log(pk_bub_real), 1)
        fit_real     = np.exp(np.polyval(lf_real, days_real))
        hl_real      = -np.log(2) / lf_real[0] / 365

        # ── Plot 1: Simulation ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(12, 8), facecolor='#0a0a0a')
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_xlim(0, days_end);  ax1.grid(False)
        ax1.set_xlabel("Days since Genesis Block")
        ax1.yaxis.set_visible(False)

        ax2 = ax1.twinx()
        ax2.set_zorder(ax1.get_zorder()+1);  ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('none')
        ax2.plot(days_real, smoothed_prices_real, color="orange",   lw=1, label="BTC")
        ax2.plot(days,      sim_p,                 color="darkgray", lw=1, label="Sim")
        plb = np.exp(C2_PLB) * days**M2_PLB
        ax2.plot(days, plb, color="orange", ls="--", label="PLB")
        ax2.set_yscale('log');  ax2.set_xlim(0, days_end);  ax2.set_ylim(1e-2, 1e6)
        ax2.grid(True, which="major", ls="-", alpha=0.2)
        ax1.plot([], [], ' ', label=f'Seed={seed}')   # 24.09.2026: Seed in der Legende (wie Lap-dark-v4)
        ax1.plot([], [], ' ', label=r'$R^2$: pre-H1 $|$ post-H1')
        ax1.plot([], [], ' ', label=r'$R^2_{\rm res}$ (z on) = ' + f'{R2["res z on"][0]:.2f} $|$ {R2["res z on"][1]:.2f}')
        ax1.plot([], [], ' ', label=r'$R^2_{\rm res}$ (z off) = ' + f'{R2["res z off"][0]:.2f} $|$ {R2["res z off"][1]:.2f}')
        ax1.plot([], [], ' ', label=r'$R^2_{\rm price}$ (sim) = ' + f'{R2["price sim"][0]:.3f} $|$ {R2["price sim"][1]:.3f}')
        ax1.plot([], [], ' ', label=r'$R^2_{\rm price}$ (PL Mean) = ' + f'{R2["price PLM"][0]:.3f} $|$ {R2["price PLM"][1]:.3f}')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h2+h1, l2+l1, loc="upper left", fontsize=10,
                   facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
        for i, hd in enumerate([1425, 2744, 4146, 5586, 7044]):
            ax2.axvline(hd, color='#808080', ls='--', lw=0.8, alpha=0.6)
            ax2.text(hd+50, 1e5, f'H{i+1}', color='#808080', fontsize=8, alpha=0.8)
        # Letzter realer Datenpunkt — grüner Punkt
        ax1.set_title('LPPL-Simulation vs Real Data', fontsize=10, color='#AAAAAA', pad=6)

        ax3 = plt.subplot(2, 1, 2)
        z_centered = z_arr - np.mean(z_arr)
        ax3.plot(days, z_centered, color="#44AAFF", lw=0.7, alpha=0.85, label="Feedback Memory",
                 ls='--' if zero_z else '-')
        ax3.axhline(0, color='#444', lw=0.5)
        ax3.set_ylabel("z(t)", color="#44AAFF")
        ax3.set_xlim(0, days_end);  ax3.set_xlabel("Days since Genesis Block")
        ax3.grid(True, which="major", ls="-", alpha=0.2)
        ax3.sharex(ax1)
        ax3.set_title('Residuals and Feedback Memory',
                      fontsize=10, color='#AAAAAA', pad=6)
        ax3r = ax3.twinx()
        ax3r.plot(days_real, np.exp(x_values), color="orange",   lw=0.8, alpha=0.7, label="BTC Res.")
        ax3r.plot(days,      np.exp(x_arr),    color="dimgray",  lw=0.6, alpha=0.5, label="Sim Res.")
        ax3r.axhline(1, color='#555', lw=0.5, ls='--')
        ax3r.set_yscale('log')
        ax3r.set_ylabel("rel. Value", color="#ccc")
        _h3,  _l3  = ax3.get_legend_handles_labels()
        _h3r, _l3r = ax3r.get_legend_handles_labels()
        ax3r.legend(_h3 + _h3r, _l3 + _l3r,
                    fontsize=7, facecolor='#1a1a1a', labelcolor='#ccc', edgecolor='#444', loc='upper left')
        _title1 = "BTC Bubble Simulation" + (" (decoupled from z)" if zero_z else "")
        with plt.rc_context({'text.usetex': False}):
            plt.suptitle(_title1, color='#CCCCCC', fontsize=13,
                         fontname='Comfortaa', fontweight='bold')
        plt.subplots_adjust(top=0.93, bottom=0.06, left=0.06, right=0.94, hspace=0.35)   # 24.09.2026: Rand links/rechts symmetrisch schmal

        # ── Plot 2: Attraktor ─────────────────────────────────────────────────
        if np.all(np.isfinite(x_arr)) and np.all(np.isfinite(z_arr)):
            build_attractor_figure(days_real, x_values, days, x_arr, z_arr,
                                   y2_arr_sim=y2_arr, zero_z=zero_z)
        else:
            print("Attraktor übersprungen: overflow in x_arr/z_arr")

        # ── Speichern ─────────────────────────────────────────────────────────
        filename_base = None
        if save_plot:
            now           = datetime.now()
            filename_base = f"LPPL-Lap_{now.strftime('%Y%m%d_%H%M%S')}"
            fig.savefig(os.path.join(sim_dir, f"{filename_base}.png"), dpi=300, facecolor='#0a0a0a')
            save_parameters(sim_dir, filename_base, y1_init, y2_init)
            # if not BATCH_MODE:
            #     create_daily_n_scatter(days_real, smoothed_prices_real, sim_p,
            #                            sim_dir, filename_base)
            print(f"Simulation saved to: {os.path.join(sim_dir, filename_base)}.*")

        if BATCH_MODE or not show_plot:
            plt.close('all')
        else:
            plt.show()

        if return_data:
            return {
                'half_life': half_life,
                'days_real': days_real,
                'days': days,
                'smoothed_prices_real': smoothed_prices_real,
                'fundamental_prices': fundamental_prices,
                'fundamental_prices_real': fundamental_prices_real,
                'x_values': x_values,
                'no_forcing': no_forcing,
                'with_forcing': with_forcing,
                'x_arr': x_arr,
                'z_arr': z_arr,
                'y2_arr': y2_arr,
                'sim_p': sim_p,
                'filename_base': filename_base,
            }

        return half_life

    except (RuntimeWarning, np.linalg.LinAlgError) as e:
        print(f"Error in simulation {sim_index}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='LPPL Attractor Simulation')
    parser.add_argument('--b-half', action='store_true', dest='b_half',
                        help='Rauschen halbieren: b_by_cycle / 2 (Einfluss der Exponenten besser sichtbar)')
    parser.add_argument('--b-quarter', action='store_true', dest='b_quarter',
                        help='Rauschen vierteln: b_by_cycle / 4')
    parser.add_argument('--zero-sigma', action='store_true',
                        help='Brownian Motion ausschalten (sigma=0 für alle Zyklen)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (Standard: zufaellig, wird in der Legende gezeigt)')
    parser.add_argument('--zero-z', action='store_true',
                        help='Wang-Kopplung ausschalten: Z_A=Z_B=Z_MIX=0 (z passiv)')
    parser.add_argument('--days', type=int, default=9000,
                        help='Simulationshorizont in Tagen (Standard: 9000)')
    parser.add_argument('--mu-data', type=int, nargs='?', const=30, default=None, dest='mu_data',
                        help='mu(t) aus echten Daten: zentriertes Fenster in Tagen (ohne Wert: 30)')
    parser.add_argument('--mu-until', type=int, default=None, dest='mu_until',
                        help='Stichtag: ab diesem Tag wieder Phasentabelle (z.B. 5586 = H4)')
    args = parser.parse_args()

    if args.mu_data is not None:               # 24.09.2026: datengetriebenes mu(t)
        import lpplattr02_params as _pm
        _pm.MU_FROM_DATA.update(enabled=True, window=args.mu_data, until=args.mu_until)
        print(f">>> --mu-data aktiv: mu(t) = {args.mu_data}d-Mittel des echten Exponenten"
              + (f" bis Tag {args.mu_until}, danach Phasentabelle" if args.mu_until else " bis Datenende"))

    if args.seed is None:                      # 24.09.2026: Zufall, Seed in der Legende
        args.seed = int(np.random.randint(0, 2**31 - 1))
    print(f">>> Seed = {args.seed}")

    if args.zero_sigma:
        import lpplattr02_params as _pm
        _pm.SIGN_OU['b'] = 0.0
        for k in _pm.SIGN_OU['b_by_cycle']:
            _pm.SIGN_OU['b_by_cycle'][k] = 0.0
        print('>>> --zero-sigma aktiv: Brownian Motion aus (b_by_cycle = 0)')

    if args.b_half or args.b_quarter:            # 24.09.2026: Rauschen halbieren / vierteln
        import lpplattr02_params as _pm
        _f = 4.0 if args.b_quarter else 2.0
        _pm.SIGN_OU['b'] /= _f
        for k in _pm.SIGN_OU['b_by_cycle']:
            _pm.SIGN_OU['b_by_cycle'][k] /= _f
        print(f'>>> --b-{"quarter" if args.b_quarter else "half"} aktiv: b_by_cycle / {_f:.0f} -> '
              + ', '.join(f"{k}: {v:.1f}" for k, v in _pm.SIGN_OU['b_by_cycle'].items()))

    if args.zero_z:
        import lpplattr02_params as _pm
        import lpplattr02_ode as _ode
        _pm.Z_A  = 0.0; _pm.Z_B  = 0.0; _pm.Z_MIX  = 0.0
        _ode.Z_A = 0.0; _ode.Z_B = 0.0; _ode.Z_MIX = 0.0
        print('>>> --zero-z aktiv: Z_A=Z_B=Z_MIX=0 (orig: 8e-3 / 8e-3 / 0.0002)')

    from lpplattr02_params import M2
    half_lives = []
    for i in range(N_SIMULATIONS):
        hl = run_simulation_and_plot(
            i + 1,
            seed=args.seed,
            zero_z=args.zero_z,
            days_end=args.days,
        )
        if hl is not None:
            half_lives.append(hl)
        if BATCH_MODE and (i+1) % 100 == 0:
            print(f"Completed {i+1} simulations")
    if BATCH_MODE:
        arr = np.array(half_lives)
        np.save(os.path.join(OUTPUT_DIR, 'half_lives.npy'), arr)
        print(f"All {N_SIMULATIONS} done  avg={np.mean(arr):.2f}y  std={np.std(arr):.2f}y")


if __name__ == "__main__":
    main()
