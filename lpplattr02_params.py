"""lpplattr02_params.py — Alle Konstanten und Parameter."""

import numpy as np
import os

BATCH_MODE    = False
N_SIMULATIONS = 1000 if BATCH_MODE else 1
OUTPUT_DIR    = "btc-sim-cDGL"
if BATCH_MODE and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Power Law
C2      = -37.7440
M2      =   5.6574
C2_PLB  = -40.4617
M2_PLB  =   5.90573

# LPPL
k      = 1.02
ALPHA  = -0.00078   # 24.09.2026: wie LPPL-Lap-dark-v4 (vorher -0.00074)
GAMMA  =  0.003
M      =  1.05 * k
N      =  3.0

# Anfangsbedingungen
y1_init_value = 0.0
y2_init_value = 0.0
z_init_value  = 0

# ── Stufe A: passives z ───────────────────────────────────────────────────────
Z_C = 0.0039  # Relaxation 0.004
Z_D = 1e-6   # Kopplung an y1 # 1e-6
Z_E = 0.5   # Kopplung an y1·y2  (erzeugt 2ω)   24.09.2026: 2.0 -> 0.5, z-Amplitude schrumpft mit

# ── Stufe B: aktives z (Wang-analog: Produkt-Rückkopplung) ───────────────────
Z_A   = 2e-3  # Kopplung −y2·z → ẏ1  (analog Wang: −y·z in ẋ)   24.09.2026: 8e-3 -> 2e-3 (Faktor 4, nach y1-Fix)
Z_B   = 2e-3  # Kopplung  y1·z → ẏ2  (analog Wang:  x·z in ẏ)   24.09.2026: 8e-3 -> 2e-3 (Faktor 4, nach y1-Fix)
Z_MIX = 0.0002 # Mischterm (y1-y2) → ẏ1  (analog Wang: A*(x-y) in ẋ)
Z_START_DAY = 1425  # 24.09.2026: z-Puffer (Stufe A+B) erst ab diesem Tag (H1); None = immer an

# Sign-OU
SIGN_OU = {
    "enabled": True,
    "mu_base": M2,
    "kappa": 108.62,
    # 24.09.2026: Rauschen direkt als gemessene Laplace-Skala b des Tages-n pro Zyklus
    # (Data/ziel.csv, prop. Fenster). Der Code rechnet intern sigma = sqrt(2*kappa*b).
    # Vorher sigma_by_cycle 40 / 90 (entsprach b = 7.4 / 37.3).
    "b": 58.2,
    "b_by_cycle": {
        0: 58.2,    # vor Halving, wie '13 (nicht messbar)
        1: 58.2,    # '13
        2: 96.8,    # '17
        3: 108.9,   # '21
        4: 103.4,   # '25
        5: 103.4,   # '28 wie '25
    },
    "mu_offset_by_cycle": {
        # Messung 24.09.2026: mean(n) - M2, prop. Fenster auf echten Halvings (get_days_in_cycle),
        # Data/ziel.csv bis Tag 6473. Gleiche Werte wie LPPL-Lap-dark-v4-fix.py.
        # Vorher handgetunt: 0: 0/0/0, 1: 4.72/-10.33/0.75, 2: 15/-8/0, 3: 6/-10/2, 4: 1/-8/2
        # ACHTUNG Zyklus 0 P1 = 13.30 aus nur 4 Monatspunkten mit Platzhalterpreisen; mit z AN
        # (ohne --zero-z) laeuft das System damit vor H1 weg (deterministisch geprueft).
        0: {"P1":  0.0,  "P2":  3.29, "P3": -1.28},   # vor Halving; P1 neutral (Messung 13.30 aus 4 Platzhalterpunkten)
        1: {"P1":  4.15, "P2": -9.29, "P3":  2.42},   # '13
        2: {"P1": 15.30, "P2": -16.11, "P3": -0.51},  # '17
        3: {"P1":  9.43, "P2": -17.29, "P3":  7.50},  # '21
        4: {"P1": -0.43, "P2": -2.0,  "P3":  2.0 },   # '25: P1 gemessen; P2, P3 gesetzt 24.09.2026 (Daten unvollst.)
        5: {"P1":  4.0,  "P2":  -4.0,  "P3":  1.0 },  # '28: gesetzt 24.09.2026 (keine Daten)
    },
    "t_min": 0.0,
}

# Datengetriebenes mu(t) (24.09.2026): statt Phasentabelle das zentrierte W-Tage-Mittel des ECHTEN
# Tages-Exponenten n = ln(P2/P1)/ln(t2/t1) aus ziel.csv (absolut, kein M2-Offset), bis Stichtag
# "until" (Tag; None = bis Datenende), danach und ausserhalb der Daten wieder die Phasentabelle.
# CLI: ./lpplattr02.py --mu-data [W] [--mu-until TAG]
MU_FROM_DATA = {
    "enabled": False,
    "window": 30,       # Tage, zentriert (Leiter 24.09.: 10-30 optimal, R2 ab H1 0.87 statt 0.44)
    "until": None,      # z.B. 5586 = H4 -> Warm-up bis H4, '25 mit Tabelle vorhergesagt
}

# Halving
HALVING_INTERVAL_YEARS = 3.815
HALVING_INTERVAL_DAYS  = HALVING_INTERVAL_YEARS * 365
FORCING_INITIAL        = 0.006
FORCING_DECAY          = 0.5
SKIP_FIRST_HALVING     = True
HALVING_ENABLED        = [False]*10
NO_HALVING             = False

# Fork
FORK_DAY          = 590
FORK_Y1_AMPLITUDE = -1.1 * 0.115 / 10
FORK_Y2_AMPLITUDE =  2.0 * 0.0022 / 10
FORK_ENABLED      = False

# PLB-Dämpfung
DAMPING = {
    "enabled": False,
    "kappa0": 2.1 * 0.0467,
    "t_min": 120.0,
}

# Rauschen
noise_std = {'alpha': 0.0, 'gamma': 0.0, 'M': 0.0, 'N': 0.0}
base_values = {'alpha': ALPHA, 'gamma': GAMMA, 'M': M, 'N': N}
