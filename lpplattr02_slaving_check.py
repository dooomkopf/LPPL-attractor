#!/usr/bin/env python3
"""Phase-1 diagnostics for lpplattr02: slaving and slow-fast checks."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hz")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lpplattr02 import (
    run_simulation_and_plot,
)
from lpplattr02_ode import system_of_equations
from lpplattr02_params import (
    DAMPING,
    SIGN_OU,
    Z_C,
    Z_D,
    Z_E,
    base_values,
)


try:
    plt.style.use(next(f for f in ['hz.mplstyle', '../hz.mplstyle', '/home/hz/Data/hz.mplstyle'] if os.path.exists(f)))
    plt.rcParams["figure.facecolor"] = "#0a0a0a"
    plt.rcParams["axes.facecolor"] = "#1a1a1a"
    plt.rcParams["text.usetex"] = False
    mpl.rcParams["font.sans-serif"] = ["Comfortaa", "DejaVu Sans", "Arial"]
except Exception:
    pass


MODE_SPECS = {
    "deterministic": {"label": "deterministic"},
    "forced": {"label": "forced"},
}


def compute_slave(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return (Z_D * y1 + Z_E * y1 * y2) / Z_C


def finite_difference_jacobian(point: np.ndarray, eps: float) -> np.ndarray:
    jacobian = np.zeros((point.size, point.size), dtype=float)
    for idx in range(point.size):
        direction = np.zeros_like(point)
        direction[idx] = eps
        fp = system_of_equations(
            point + direction,
            base_values,
            t_current=1.0,
            use_damping=False,
            use_sign_ou=False,
        )
        fm = system_of_equations(
            point - direction,
            base_values,
            t_current=1.0,
            use_damping=False,
            use_sign_ou=False,
        )
        jacobian[:, idx] = (fp - fm) / (2.0 * eps)
    return jacobian


def analyze_spectrum(jacobian: np.ndarray) -> dict[str, object]:
    eigenvalues = np.linalg.eigvals(jacobian)
    real_parts = eigenvalues.real
    fast_index = int(np.argmin(real_parts))
    slow_mask = np.ones(len(eigenvalues), dtype=bool)
    slow_mask[fast_index] = False
    slow_real_parts = real_parts[slow_mask]
    fast_real = float(real_parts[fast_index])
    slow_scale = float(np.max(np.abs(slow_real_parts))) if slow_real_parts.size else 0.0
    gap_ratio = abs(fast_real) / max(slow_scale, 1e-12)
    return {
        "jacobian": jacobian,
        "eigenvalues": eigenvalues,
        "fast_real": fast_real,
        "slow_scale": slow_scale,
        "gap_ratio": gap_ratio,
        "has_stable_fast_mode": fast_real < 0.0,
        "unstable_count": int(np.sum(real_parts > 0.0)),
    }


def compute_metrics(z: np.ndarray, z_slave: np.ndarray) -> dict[str, float]:
    valid_mask = np.isfinite(z) & np.isfinite(z_slave)
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return {
            "valid_count": 0,
            "max_abs_delta": float("nan"),
            "rms_abs_delta": float("nan"),
            "max_rel_signal": float("nan"),
            "rms_rel_signal": float("nan"),
            "median_pointwise_rel": float("nan"),
            "corr_z_vs_slave": float("nan"),
        }

    z = z[valid_mask]
    z_slave = z_slave[valid_mask]
    delta = z - z_slave
    abs_delta = np.abs(delta)
    z_rms = float(np.sqrt(np.mean(np.square(z))))
    slave_rms = float(np.sqrt(np.mean(np.square(z_slave))))
    signal_rms = max(z_rms, slave_rms, 1e-12)
    signal_peak = max(float(np.max(np.abs(z))), float(np.max(np.abs(z_slave))), 1e-12)
    pointwise_scale = np.maximum(np.maximum(np.abs(z), np.abs(z_slave)), 1e-12)
    correlation = np.nan
    if np.std(z) > 0.0 and np.std(z_slave) > 0.0:
        correlation = float(np.corrcoef(z, z_slave)[0, 1])

    return {
        "valid_count": valid_count,
        "max_abs_delta": float(np.max(abs_delta)),
        "rms_abs_delta": float(np.sqrt(np.mean(np.square(delta)))),
        "max_rel_signal": float(np.max(abs_delta) / signal_peak),
        "rms_rel_signal": float(np.sqrt(np.mean(np.square(delta))) / signal_rms),
        "median_pointwise_rel": float(np.median(abs_delta / pointwise_scale)),
        "corr_z_vs_slave": correlation,
    }


def classify_run(run: dict[str, object], spectrum: dict[str, object]) -> str:
    if run["breakdown_day"] is not None:
        return "NUMERICAL BREAKDOWN"

    metrics = run["metrics"]
    rms_rel = metrics["rms_rel_signal"]
    gap_ratio = float(spectrum["gap_ratio"])
    stable_fast = bool(spectrum["has_stable_fast_mode"])

    if stable_fast and gap_ratio >= 3.0 and rms_rel <= 0.05:
        return "SFD JA"
    if stable_fast and gap_ratio >= 1.5 and rms_rel <= 0.15:
        return "SFD MARGINAL"
    return "SFD NEIN"


def detect_breakdown_day(days: np.ndarray, states: np.ndarray, z_slave: np.ndarray) -> float | None:
    finite_mask = np.all(np.isfinite(states), axis=1) & np.isfinite(z_slave)
    bad_idx = np.flatnonzero(~finite_mask)
    if bad_idx.size == 0:
        return None
    return float(days[bad_idx[0]])


def build_run(mode: str, days: np.ndarray, states: np.ndarray) -> dict[str, object]:
    z_slave = compute_slave(states[:, 0], states[:, 1])
    metrics = compute_metrics(states[:, 2], z_slave)
    breakdown_day = detect_breakdown_day(days, states, z_slave)
    return {
        "mode": mode,
        "days": days,
        "states": states,
        "z_slave": z_slave,
        "metrics": metrics,
        "breakdown_day": breakdown_day,
    }


def format_complex(value: complex) -> str:
    return f"{value.real:+.6e}{value.imag:+.6e}j"


def format_matrix(matrix: np.ndarray) -> str:
    return "\n".join(
        "[" + ", ".join(f"{entry:+.4e}" for entry in row) + "]" for row in matrix
    )


def print_run_summary(run: dict[str, object], spectrum: dict[str, object], summary: str) -> None:
    metrics = run["metrics"]
    print(f"\n[{run['mode']}] {summary}")
    print(f"  valid samples     = {metrics['valid_count']}")
    if run["breakdown_day"] is not None:
        print(f"  breakdown day     = {run['breakdown_day']:.1f}")
    print(f"  max|Delta|       = {metrics['max_abs_delta']:.6e}")
    print(f"  rms|Delta|       = {metrics['rms_abs_delta']:.6e}")
    print(f"  max|Delta|/signal= {metrics['max_rel_signal']:.6%}")
    print(f"  rms|Delta|/signal= {metrics['rms_rel_signal']:.6%}")
    print(f"  median pointwise rel = {metrics['median_pointwise_rel']:.6%}")
    print(f"  corr(z, h)   = {metrics['corr_z_vs_slave']:.6f}")
    print(f"  fast eigenvalue real part = {spectrum['fast_real']:+.6e}")
    print(f"  slow-fast gap ratio       = {float(spectrum['gap_ratio']):.6f}")


def plot_run(
    run: dict[str, object],
    spectrum: dict[str, object],
    summary: str,
    output_dir: str,
    show_plot: bool,
) -> str:
    days = run["days"]
    states = run["states"]
    z_slave = run["z_slave"]
    metrics = run["metrics"]
    delta = states[:, 2] - z_slave
    finite_mask = np.all(np.isfinite(states), axis=1) & np.isfinite(z_slave)
    if not np.any(finite_mask):
        finite_mask = np.zeros_like(days, dtype=bool)
    finite_days = days[finite_mask]
    finite_states = states[finite_mask]
    finite_z_slave = z_slave[finite_mask]
    finite_delta = delta[finite_mask]

    figure = plt.figure(figsize=(14, 18), facecolor="#0a0a0a")
    axes = figure.subplots(5, 1, sharex=False)
    figure.subplots_adjust(top=0.95, hspace=0.35)

    mode_title = "Forced run (sign_OU on)" if run["mode"] == "forced" else "Deterministic run"
    figure.suptitle(
        f"lpplattr02 Phase-1 Slaving Check: {mode_title}",
        color="#CCCCCC",
        fontsize=14,
        fontname="Comfortaa",
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(days, states[:, 0], color="#ffaa33", lw=0.9, label="y1")
    ax.plot(days, states[:, 1], color="#55ddaa", lw=0.9, label="y2")
    ax.plot(days, states[:, 2], color="#44aaff", lw=0.8, label="z")
    if run["breakdown_day"] is not None:
        ax.axvline(run["breakdown_day"], color="#ff4444", lw=0.8, ls="--", label="breakdown")
    ax.set_title("1. Full 3D integration", color="#AAAAAA", fontsize=10, pad=6)
    ax.set_ylabel("state")
    ax.grid(True, alpha=0.2)
    ax.legend(
        loc="upper left",
        fontsize=8,
        facecolor="#1A1A1A",
        edgecolor="#555555",
        labelcolor="#E0E0E0",
    )

    ax = axes[1]
    ax.plot(days, states[:, 2], color="#44aaff", lw=0.9, label="z(t)")
    ax.plot(days, z_slave, color="#ff6699", lw=0.9, ls="--", label="h(y1,y2)")
    if run["breakdown_day"] is not None:
        ax.axvline(run["breakdown_day"], color="#ff4444", lw=0.8, ls="--")
    ax.set_title("2. Slaving overlay", color="#AAAAAA", fontsize=10, pad=6)
    ax.set_ylabel("z / h")
    ax.grid(True, alpha=0.2)
    ax.legend(
        loc="upper left",
        fontsize=8,
        facecolor="#1A1A1A",
        edgecolor="#555555",
        labelcolor="#E0E0E0",
    )

    ax = axes[2]
    ax.plot(days, delta, color="#ff7777", lw=0.8)
    ax.axhline(0.0, color="#777777", lw=0.6, ls="--")
    if run["breakdown_day"] is not None:
        ax.axvline(run["breakdown_day"], color="#ff4444", lw=0.8, ls="--")
    ax.set_title("3. Slaving residual Delta(t) = z - h(y1,y2)", color="#AAAAAA", fontsize=10, pad=6)
    ax.set_ylabel("Delta(t)")
    ax.grid(True, alpha=0.2)
    hist_ax = ax.inset_axes([0.76, 0.18, 0.21, 0.68])
    hist_ax.hist(finite_delta, bins=40, color="#ff7777", alpha=0.75)
    hist_ax.set_title("hist", fontsize=8, color="#BBBBBB")
    hist_ax.tick_params(axis="both", labelsize=7)
    hist_ax.set_facecolor("#111111")
    hist_ax.grid(True, alpha=0.15)
    ax.text(
        0.015,
        0.98,
        (
            f"max|Delta| = {metrics['max_abs_delta']:.3e}\n"
            f"rms|Delta| = {metrics['rms_abs_delta']:.3e}\n"
            f"rms rel = {metrics['rms_rel_signal']:.2%}\n"
            f"corr(z,h) = {metrics['corr_z_vs_slave']:.4f}\n"
            f"valid n = {metrics['valid_count']}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#DDDDDD",
        bbox={"facecolor": "#111111", "edgecolor": "#333333", "alpha": 0.9},
    )

    ax = axes[3]
    step = max(1, len(finite_days) // 5000) if len(finite_days) else 1
    ax.plot(
        finite_states[::step, 0],
        finite_states[::step, 1],
        color="#ffaa33",
        lw=0.7,
        alpha=0.9,
    )
    ax.set_title("4. Phase portrait projections", color="#AAAAAA", fontsize=10, pad=6)
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.grid(True, alpha=0.2)
    yz_ax = ax.inset_axes([0.62, 0.57, 0.33, 0.32])
    yz_ax.plot(
        finite_states[::step, 0],
        finite_states[::step, 2],
        color="#44aaff",
        lw=0.6,
        alpha=0.85,
    )
    yz_ax.set_title("y1 vs z", fontsize=8, color="#BBBBBB")
    yz_ax.tick_params(axis="both", labelsize=7)
    yz_ax.grid(True, alpha=0.15)
    yz_ax.set_facecolor("#111111")
    xz_ax = ax.inset_axes([0.62, 0.12, 0.33, 0.32])
    xz_ax.plot(
        finite_states[::step, 1],
        finite_states[::step, 2],
        color="#55ddaa",
        lw=0.6,
        alpha=0.85,
    )
    xz_ax.set_title("y2 vs z", fontsize=8, color="#BBBBBB")
    xz_ax.tick_params(axis="both", labelsize=7)
    xz_ax.grid(True, alpha=0.15)
    xz_ax.set_facecolor("#111111")

    ax = axes[4]
    eigenvalues = spectrum["eigenvalues"]
    ax.scatter(eigenvalues.real, eigenvalues.imag, color="#ffcc66", s=60, zorder=3)
    ax.axhline(0.0, color="#777777", lw=0.6, ls="--")
    ax.axvline(0.0, color="#777777", lw=0.6, ls="--")
    ax.grid(True, alpha=0.2)
    ax.set_title("5. Linearization at origin (finite differences)", color="#AAAAAA", fontsize=10, pad=6)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    text = (
        f"{summary}\n"
        + (
            f"breakdown day = {run['breakdown_day']:.1f}\n"
            if run["breakdown_day"] is not None
            else ""
        )
        + f"gap ratio = {float(spectrum['gap_ratio']):.3f}\n"
        + f"stable fast mode = {bool(spectrum['has_stable_fast_mode'])}\n"
        + f"unstable linear modes = {int(spectrum['unstable_count'])}\n\n"
        + "eigenvalues:\n"
        + "\n".join(f"  {format_complex(value)}" for value in eigenvalues)
        + "\n\nJacobian:\n"
        + format_matrix(spectrum["jacobian"])
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#DDDDDD",
        family="monospace",
        bbox={"facecolor": "#111111", "edgecolor": "#333333", "alpha": 0.9},
    )

    for axis in axes:
        axis.tick_params(colors="#CCCCCC")
        axis.xaxis.label.set_color("#CCCCCC")
        axis.yaxis.label.set_color("#CCCCCC")
        axis.title.set_color("#AAAAAA")

    axes[-1].set_xlim(
        float(np.min(eigenvalues.real)) - 0.001,
        float(np.max(eigenvalues.real)) + 0.001,
    )
    axes[0].set_xlim(float(days[0]), float(days[-1]))
    axes[-1].set_aspect("auto")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir,
        f"lpplattr02_slaving_check_{run['mode']}_{timestamp}.png",
    )
    figure.savefig(output_path, dpi=220, facecolor="#0a0a0a")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 slaving diagnostics for lpplattr02.")
    parser.add_argument(
        "--mode",
        choices=["all", "deterministic", "forced"],
        default="all",
        help="Which run to execute. Default: all.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=8000,
        help="Number of simulation days. Default: 8000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Random seed used for reproducibility. Default: 5.",
    )
    parser.add_argument(
        "--fd-eps",
        type=float,
        default=1e-6,
        help="Finite-difference step for the Jacobian at the origin. Default: 1e-6.",
    )
    parser.add_argument(
        "--output-dir",
        default="Sim-Lap",
        help="Directory for the generated figures. Default: Sim-Lap.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a plot window. Default: show the slaving figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modes = (
        ["deterministic", "forced"]
        if args.mode == "all"
        else [args.mode]
    )

    simulation = run_simulation_and_plot(
        seed=args.seed,
        days_end=args.days,
        show_plot=False,
        save_plot=False,
        return_data=True,
    )
    days = simulation["days"]

    jacobian = finite_difference_jacobian(np.zeros(3, dtype=float), args.fd_eps)
    spectrum = analyze_spectrum(jacobian)

    print("lpplattr02 Phase-1 Slaving Check")
    print(f"  DAMPING.enabled = {DAMPING['enabled']}")
    print(f"  SIGN_OU.enabled = {SIGN_OU['enabled']}")
    print(f"  finite-difference eps = {args.fd_eps:.1e}")
    print(f"  first/last simulated day = {int(days[0])} .. {int(days[-1])}")

    available_runs = {}
    if "deterministic" in modes:
        available_runs["deterministic"] = build_run(
            "deterministic",
            days,
            simulation["no_forcing"]["states"],
        )
    if "forced" in modes:
        available_runs["forced"] = build_run(
            "forced",
            days,
            simulation["with_forcing"]["states"],
        )

    for mode in modes:
        run = available_runs[mode]
        summary = classify_run(run, spectrum)
        print_run_summary(run, spectrum, summary)
        output_path = plot_run(
            run,
            spectrum,
            summary,
            args.output_dir,
            show_plot=not args.no_show,
        )
        print(f"  figure saved to: {output_path}")


if __name__ == "__main__":
    main()
