"""lpplattr02_ode.py — ODE-Kern: Butcher, system_of_equations, rk8_step, sign_step."""

import numpy as np
from lpplattr02_params import (
    ALPHA, GAMMA, M, N, M2, DAMPING, SIGN_OU, Z_C, Z_D, Z_E, Z_A, Z_B, Z_MIX, Z_START_DAY,
    MU_FROM_DATA
)

# ── Butcher-Tableau RK8 (Dormand-Prince) ─────────────────────────────────────
A_RK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0],
    [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0],
    [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0, 0, 0, 0],
    [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0],
    [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0],
    [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0],
    [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0],
    [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0]
])
C_RK = np.array([0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200,
                 5490023248/9719169821, 13/20, 1201146811/1299019798, 1, 1])
B_RK = np.array([14005451/335480064, 0, 0, 0, 0, -59238493/1068277825,
                 181606767/758867731, 561292985/797845732, -1041891430/1371343529,
                 760417239/1151165299, 118820643/751138087,
                 -528747749/2220607170, 1/4])


def kappa_and_stiffness(t_current, M2_val, kappa0, t_min):
    t = max(float(t_current), float(t_min))
    kappa     = kappa0 * (M2_val / t)
    dkappa_dt = -kappa0 * (M2_val / t**2)
    return kappa, dkappa_dt + kappa**2


def get_cycle_number(t):
    HALVING_DAYS = [1425, 2744, 4146, 5586, 7044]
    for i, h in enumerate(reversed(HALVING_DAYS)):
        if t >= h:
            return len(HALVING_DAYS) - i
    return 0


def get_days_in_cycle(t):
    """Position im Zyklus in Referenz-Tagen (0..HALVING_INTERVAL_DAYS).
    FIX 24.09.2026: echte Halving-Tage statt theoretischer Uhr n*HALVING_INTERVAL_DAYS;
    jeder echte Zyklus proportional auf die Referenzlaenge skaliert, so dass die
    Phasengrenzen 100/550/950 in get_mu_t als Bruchteile gelten (wie LPPL-Lap-dark-v4-fix.py).
    Vor H1: Zyklus 0..1425. Nach H5 (est.): keine Skalierung."""
    from lpplattr02_params import HALVING_INTERVAL_DAYS
    edges = [0, 1425, 2744, 4146, 5586, 7044]
    for i in range(len(edges) - 1, -1, -1):
        h = edges[i]
        if t >= h:
            L = (edges[i + 1] - h) if i + 1 < len(edges) else HALVING_INTERVAL_DAYS
            return (t - h) * HALVING_INTERVAL_DAYS / L
    return t


_MU_DATA = {"t": None, "n": None, "W": None}   # Cache: geglaetteter echter Exponent (MU_FROM_DATA)


def _mu_data_value(t):
    """Zentriertes W-Tage-Mittel des echten Tages-Exponenten an Tag t; None ausserhalb der Daten."""
    W = int(MU_FROM_DATA["window"])
    if _MU_DATA["n"] is None or _MU_DATA["W"] != W:
        import os, csv
        path = next((f for f in ["ziel.csv", "../ziel.csv", "../../ziel.csv"]
                     if os.path.exists(f)), None)
        if path is None:
            raise FileNotFoundError("MU_FROM_DATA: ziel.csv not found")
        rows = [(float(r[0]), float(r[1])) for r in csv.reader(open(path), delimiter=' ')
                if float(r[1]) > 0]
        td = np.array([r[0] for r in rows]); pd_ = np.array([r[1] for r in rows])
        n = np.log(pd_[1:] / pd_[:-1]) / np.log(td[1:] / td[:-1])
        _MU_DATA["t"] = td[:-1]
        _MU_DATA["n"] = np.convolve(n, np.ones(W) / W, mode='same')
        _MU_DATA["W"] = W
    td = _MU_DATA["t"]
    if t < td[0] or t > td[-1]:
        return None
    return float(_MU_DATA["n"][min(int(np.searchsorted(td, t)), len(td) - 1)])


def get_mu_t(t):
    # 24.09.2026: datengetriebenes mu(t) per Flag, sonst Phasentabelle wie bisher
    if MU_FROM_DATA["enabled"] and (MU_FROM_DATA["until"] is None or t < MU_FROM_DATA["until"]):
        v = _mu_data_value(t)
        if v is not None:
            return v
    cycle   = get_cycle_number(t)
    offsets = SIGN_OU["mu_offset_by_cycle"].get(cycle, {"P1": 0, "P2": 0, "P3": 0})
    dic     = get_days_in_cycle(t)
    if 100 <= dic <= 550:  return SIGN_OU["mu_base"] + offsets["P1"]
    if 550 <  dic <= 950:  return SIGN_OU["mu_base"] + offsets["P2"]
    return SIGN_OU["mu_base"] + offsets["P3"]


def get_sigma_t(t):
    """24.09.2026: Parameter ist die gemessene Laplace-Skala b; sigma = sqrt(2*kappa*b)."""
    cycle = get_cycle_number(t)
    b = SIGN_OU["b_by_cycle"].get(cycle, SIGN_OU["b"])
    return np.sqrt(2.0 * SIGN_OU["kappa"] * b)


def calculate_local_exponent(t, y2):
    n   = M2 + t * y2
    X   = n - get_mu_t(t)
    return n, X


def system_of_equations(y, noisy_values, t_current=1,
                        use_damping=False, use_sign_ou=False, noise_xi=0.0):
    y1, y2, z = y
    # 24.09.2026: z-Puffer vor Z_START_DAY (H1) komplett aus (Stufe A+B), danach wie bisher
    zon = 0.0 if (Z_START_DAY is not None and t_current < Z_START_DAY) else 1.0
    dy1 = y2 - zon * Z_A * y2 * z + zon * Z_MIX * (y1 - y2)
    dy2 = (noisy_values['alpha'] * y2 * abs(y2)**(noisy_values['M'] - 1)
           - noisy_values['gamma'] * y1 * abs(y1)**(noisy_values['N'] - 1)
           + zon * Z_B * y1 * z)

    if use_damping and DAMPING["enabled"]:
        M2_local = noisy_values.get('M2', M2)
        kappa, stiff = kappa_and_stiffness(t_current, M2_local,
                                           DAMPING["kappa0"], DAMPING["t_min"])
        dy2 += (-2.0 * kappa * y2) - (stiff * y1)

    dz = zon * (-Z_C * z + Z_D * y1 + Z_E * y1 * y2)
    return np.array([dy1, dy2, dz])


def sign_step_substeps(y, t_start, dt_sub, n_sub):
    kappa = SIGN_OU["kappa"]
    t_min = SIGN_OU["t_min"]
    y1, y2, z = y
    t_sub = t_start
    for _ in range(n_sub):
        t_safe  = max(t_sub, t_min)
        sigma   = get_sigma_t(t_safe)
        X       = (M2 + t_safe * y2) - get_mu_t(t_safe)
        sign_X  = np.sign(X) if X != 0 else 0.0
        y2_new  = (y2 + dt_sub * (-(kappa * sign_X) / t_safe)) / (1.0 + dt_sub / t_safe)
        # FIX 24.09.2026 (Codex-bestaetigt): y1 NICHT hier integrieren (Doppelintegration mit
        # dy1 im RK8-Kern). Strang-Split: F = (dy1, base) im Kern, G = (0, Sign-OU) hier.
        y1_new  = y1
        if sigma != 0.0:
            y2_new += (sigma / (t_safe + dt_sub)) * np.sqrt(dt_sub) * np.random.normal()   # 24.09.2026: /(t+dt) -> n bekommt exakt sigma*sqrt(dt)*xi
        y1, y2 = y1_new, y2_new
        t_sub  += dt_sub
    return np.array([y1, y2, z])   # z eingefroren (Stufe A)


def rk8_step(y, dt, noisy_values, t_current=1,
             use_damping=False, use_sign_ou=False, noise_xi=0.0):
    k = np.zeros((13, 3))
    for i in range(13):
        y_temp = y.copy()
        for j in range(i):
            y_temp += dt * A_RK[i, j] * k[j]
        k[i] = system_of_equations(y_temp, noisy_values,
                                   t_current + C_RK[i] * dt,
                                   use_damping, use_sign_ou,
                                   noise_xi if i == 12 else 0.0)
    return y + dt * sum(b * ki for b, ki in zip(B_RK, k))
