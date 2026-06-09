import numpy as np
import mujoco
"""
Controllers for Mobile Manipulator
Includes:
1. PID Control
2. Time-Delay Estimation (TDE)
3. Sliding Mode Control (SMC)
3. Adaptive Sliding Mode Control (ASMC)

Note on MuJoCo Dynamics (Engineering Note):
- Armature: Represents the inertia of the rotor/gears. 
  - High armature (>0.3) makes the system sluggish but stable against high-frequency noise.
  - Low armature (<0.05, e.g., wrist joints) makes the system agile but prone to overshoot and oscillation.
- Damping: Simulates joint friction and viscosity.
  - High damping helps suppress oscillation but increases steady-state error and torque demand.
  - Low damping allows fast response but risks instability.
  
Key Strategy: Since joints 4 and 5 have low inertia and limited torque, 
tracking errors and oscillations are handled via adaptive gains in the ASMC scheme.
"""

def pid_control(sim,error_p,error_d,integral_error): 
    Kp = np.array([100, 120, 100, 100, 120, 130]) 
    Ki = np.array([0, 0, 0, 0, 0, 0])       
    Kd = np.array([20, 25, 25, 25, 26, 20])    

    u = Kp * error_p + Kd * error_d + Ki * integral_error
    torque_limits = np.array([100, 144, 59, 22, 30, 20])
    sim.ctrl[:6] = np.clip(u, -torque_limits, torque_limits)

def traditional_tdc(sim,tau,qd_ddot,error,error_dot): 
    K1 = np.diag([20, 25, 25, 25, 25, 25])
    K2 = np.diag([10, 14, 14, 15, 15, 15])
    M_hat = np.diag([0.008, 0.006, 0.004, 0.005, 0.005, 0.001])
    M_hat_inv = np.linalg.inv(M_hat)
    tau_t_minus_L = tau
    
    phi_hat = sim.qacc[:6] - M_hat_inv @ tau_t_minus_L
    desired_term = qd_ddot + K2 @ error_dot + K1 @ error
    tau = -M_hat @ phi_hat + M_hat @ desired_term
    return tau

def adaptive_integral_smc(sim, tau ,sum_error,qd_ddot, error, error_dot, K, dt=0.001):

    M_hat = np.diag([0.002,0.003,0.002,0.002,0.002,0.001])
    Kp    = np.diag([20,25,30,30,30,30])
    Kd    = np.diag([10,10,10,15,15,15])
    alpha = np.array([12,12,12,12,15,15])
    beta  = np.array([15,18,18,16,12,12])
    M_hat_inv = np.linalg.inv(M_hat)
    tau_t_minus_L = tau

    phi_hat = sim.qacc[:6] - M_hat_inv @ tau_t_minus_L
    
    for i in range(6):
        sum_error[i]+= error[i] * dt  #
    s = error_dot + Kd @ error + Kp @ (sum_error.flatten())

    for i in range(len(K)):
        s_abs = abs(s[i])
        dot_K = alpha[i] * s_abs * np.sign(s_abs - (K[i]**2)/beta[i])
        K[i] += dot_K * dt           
        if K[i] < 0.0:                
            K[i] = 0.0
    
    desired = qd_ddot + Kd @ error_dot + Kp @ error + np.diag(K) @ s
    tau = -M_hat @ phi_hat + M_hat @ desired
    return tau, K,sum_error

def floating_timedelay_asmc(sum_sign, sim,tau,qd_ddot,error,error_dot,error_t_minus_L,error_dot_t_minus_L,phi_hat_t_minus_L,adaptive_param):
    K1 = np.diag([20, 35, 40, 18, 16.5, 24]) 
    K2 = np.diag([20, 35, 30, 15, 12.5, 15])

    alpha = 0.3
    epsilon = 0.3
    psi = np.array([15, 25, 25, 15, 15, 15])
    M_hat = np.diag([0.0015, 0.0015, 0.001, 0.001, 0.001, 0.001])
    M_hat_inv = np.linalg.inv(M_hat)
    tau_t_minus_L = tau
    dt = 0.001
    

    phi_hat = sim.qacc[1:7] - M_hat_inv @ tau_t_minus_L
    eta_t = phi_hat - phi_hat_t_minus_L

    s = error_dot + K1 @ error + error_dot_t_minus_L + K1 @ error_t_minus_L

    phi_hat_term = phi_hat + alpha * eta_t
    desired_term = qd_ddot + K1 @ error_dot + K1 @ error_dot_t_minus_L + K2 @ s
    tau_line_t = -M_hat @ phi_hat_term + M_hat @ desired_term

   
    for i in range(6):
        sum_sign[i] += np.sign(s[i])*dt
    smc_term = adaptive_param * (np.sign(s)+ sum_sign.flatten())
    tau = tau_line_t + M_hat @ smc_term

    
    adaptive_param_new = adaptive_param.copy()
    for j in range(6):
        if abs(s[j]) >= epsilon:
            adaptive_param_new[j] += psi[j] * abs(s[j]) * dt
        else:
            adaptive_param_new[j] -= (1.0 / dt) * abs(s[j]) * adaptive_param_new[j] / epsilon
        if adaptive_param_new[j] < 0:
            adaptive_param_new[j] = 0.0
    return adaptive_param_new, phi_hat,tau,sum_sign,s
    
def online_timedelay_asmc(sum_sign, sim,tau,qd_ddot,error,error_dot,error_t_minus_L,error_dot_t_minus_L,phi_hat_t_minus_L,adaptive_param):
    """
    Proposed Method: ASMC based on TDE (Online/Real-time version).
    """
    K1 = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
    K2 = np.diag([0.5, 0.5, 1.0, 1.5, 2.5, 2.5])
    alpha = 0.5
    epsilon = 0.3
    psi = np.array([6, 6.5, 6, 5, 5.5, 4.5])
    M_hat = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]) 
    M_hat_inv = np.linalg.inv(M_hat)
    tau_t_minus_L = tau
    dt = 0.001
    
    phi_hat = sim.qacc[:6] - M_hat_inv @ tau_t_minus_L
    eta_t = phi_hat - phi_hat_t_minus_L

    s = error_dot + K1 @ error + error_dot_t_minus_L + K1 @ error_t_minus_L


    phi_hat_term = phi_hat + alpha * eta_t
    desired_term = qd_ddot + K1 @ error_dot + K1 @ error_dot_t_minus_L + K2 @ s
    tau_line_t = -M_hat @ phi_hat_term + M_hat @ desired_term


    for i in range(6):
        sum_sign[i] += np.sign(s[i])*dt
    smc_term = adaptive_param * (np.sign(s)+ sum_sign.flatten())
    tau = tau_line_t + M_hat @ smc_term

    
    adaptive_param_new = adaptive_param.copy()
    for j in range(6):
        if abs(s[j]) >= epsilon:
            adaptive_param_new[j] += psi[j] * abs(s[j]) * dt
        else:
            adaptive_param_new[j] -= (1.0 / dt) * abs(s[j]) * adaptive_param_new[j] / epsilon
        if adaptive_param_new[j] < 0:
            adaptive_param_new[j] = 0.0
    return adaptive_param_new, phi_hat,tau,sum_sign, s

## NFTSMC
def _vec(x, n: int = 6) -> np.ndarray:
    """Convert input to a finite 1-D vector of length n."""
    y = np.asarray(x, dtype=float).reshape(-1)
    if y.size < n:
        z = np.zeros(n, dtype=float)
        z[: y.size] = y
        y = z
    y = y[:n]
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def sat(x, delta: float = 0.03) -> np.ndarray:
    """Boundary-layer saturation function from Eq. (26)."""
    delta = max(float(delta), 1.0e-12)
    return np.clip(np.asarray(x, dtype=float) / delta, -1.0, 1.0)


def _abs_power(x, p: float, eps: float = 1.0e-12) -> np.ndarray:
    """Regularized |x|^p."""
    return np.power(np.abs(np.asarray(x, dtype=float)) + eps, p)


def _paper_power_sign(x, p: float, delta: float = 0.03) -> np.ndarray:
    """Regularized |x|^p * sat(x), used to reduce chattering."""
    return _abs_power(x, p) * sat(x, delta)


def _mass_and_bias(model, data, n: int = 6):
    """
    Get M(q) and h(q,qdot)=C(q,qdot)qdot+G(q) from MuJoCo for the first n DOFs.
    Falls back to a nominal diagonal inertia when model is not supplied.
    """
    if model is not None and mujoco is not None:
        M_full = np.zeros((model.nv, model.nv), dtype=float)
        mujoco.mj_fullM(model, M_full, data.qM)
        M = M_full[:n, :n].copy()
        h = _vec(data.qfrc_bias[:n], n)
        # Small regularization in case the XML has nearly massless axes.
        M = M + 1.0e-9 * np.eye(n)
        return M, h

    # Fallback only for compatibility with the old call signature.
    M = np.diag([0.008, 0.006, 0.004, 0.003, 0.0025, 0.0015])
    h = np.zeros(n, dtype=float)
    return M, h


def _ctrl_limits(model, n: int = 6, default=None) -> np.ndarray:
    """Read actuator limits from MuJoCo if available; otherwise use defaults."""
    if default is None:
        default = np.array([100.0, 144.0, 59.0, 22.0, 30.0, 20.0], dtype=float)
    default = _vec(default, n)

    if model is None:
        return default

    try:
        ctrlrange = np.asarray(model.actuator_ctrlrange[:n], dtype=float)
        ctrllimited = np.asarray(model.actuator_ctrllimited[:n], dtype=bool)
        if ctrlrange.shape == (n, 2):
            lim = np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1]))
            valid = ctrllimited & np.isfinite(lim) & (lim > 0.0)
            return np.where(valid, lim, default)
    except Exception:
        pass

    return default


# ---------------------------------------------------------------------------
# Paper NFTSM controller, Eqs. (23)--(26)
# ---------------------------------------------------------------------------

def paper_nftsm_control_law(
    data,
    qd_ddot,
    error_q,
    error_q_dot,
    *,
    model=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    r1: float = 1.8,
    r2: float = 1.6,
    r3: float = 1.0,
    c1: float | np.ndarray = 20.0,
    c2: float | np.ndarray = 0.6,
    delta: float = 0.03,
    torque_limits=None,
    n: int = 6,
):
    """
    Exact dynamic-level NFTSM law adapted to your error convention.

    Paper definitions:
        e1 = q_m - q_md
        e2 = qdot_m - qdot_md
        s  = e1 + alpha*sat(e1)|e1|^r1 + beta*sat(e2)|e2|^r2       (23)
        tau = -M(q) * (u_eq + u_sw)                                (24)
        u_eq = |e2|^(2-r2) sign(e2)/(beta*r2)
               * (1 + alpha*r1*|e1|^(r1-1)) + H(q,qdot)
        u_sw = c1*|s|^r3 sign(s) + c2*s                             (25)

    In this implementation sign(.) is replaced by sat(.) according to Eq. (26).
    """
    qd_ddot = _vec(qd_ddot, n)

    # Convert your simulation's desired-minus-actual errors into the paper's
    # actual-minus-desired errors.
    e1 = -_vec(error_q, n)
    e2 = -_vec(error_q_dot, n)

    c1 = np.broadcast_to(np.asarray(c1, dtype=float), (n,))
    c2 = np.broadcast_to(np.asarray(c2, dtype=float), (n,))

    M, h = _mass_and_bias(model, data, n=n)

    # Eq. (5): H = -M^{-1}(Cqdot + G) - qddot_d.
    H = -np.linalg.solve(M, h) - qd_ddot

    # Eq. (23), using sat(.) instead of sign(.) to reduce chattering.
    s = e1 + alpha * _paper_power_sign(e1, r1, delta) + beta * _paper_power_sign(e2, r2, delta)

    # Eq. (25): equivalent term and switching term.
    ueq = (_abs_power(e2, 2.0 - r2) * sat(e2, delta) / (beta * r2)) * (
        1.0 + alpha * r1 * _abs_power(e1, r1 - 1.0)
    ) + H
    usw = c1 * _abs_power(s, r3) * sat(s, delta) + c2 * s

    tau = -M @ (ueq + usw)

    limits = _ctrl_limits(model, n=n, default=torque_limits)
    tau = np.clip(tau, -limits, limits)

    return tau, s, H


##FT-CNT-MAG-STC
def _sat_sign(x, eps=0.0):
    """sign(x), or a saturated sign when eps > 0 for numerical smoothing."""
    x = np.asarray(x, dtype=float)
    if eps is None or eps <= 0:
        return np.sign(x)
    return np.clip(x / eps, -1.0, 1.0)


def init_ft_cnt_mag_stc_state(n=6, k1_init=1.5):
    """
    Initialize controller states for FT-CNT-MAG-STC.
    """
    return {
        "g": np.zeros(n, dtype=float),
        "k1": np.ones(n, dtype=float) * k1_init,
    }


def ft_cnt_mag_stc(
    sim,
    qd_ddot,
    error,
    error_dot,
    state=None,
    dt=0.001,
    M_hat_diag=None,
    torque_limits=None,
    params=None,
    smooth_sign_eps=0.0,
):
    """
    Finite-Time Continuous Nonsingular Terminal Modified Adaptive-Gain
    Super-Twisting Control (FT-CNT-MAG-STC), adapted as a 6-DOF baseline.

    Error convention in your code:
        error     = q_d - q
        error_dot = dq_d - dq

    Returns
    -------
    state : dict
        Updated controller state.
    tau : ndarray, shape (6,)
        Joint torque command.
    s : ndarray, shape (6,)
        Sliding variable.
    """
    qd_ddot = np.asarray(qd_ddot, dtype=float).reshape(-1)
    error = np.asarray(error, dtype=float).reshape(-1)
    error_dot = np.asarray(error_dot, dtype=float).reshape(-1)
    n = error.size

    if state is None:
        state = init_ft_cnt_mag_stc_state(n=n)

    default_params = {
        "lambda1": np.ones(n) * 2.0,
        "lambda2": np.ones(n) * 2.0,
        "r1":      np.ones(n) * 3.0,
        "r2":      np.ones(n) * (3.0 / 5.0),
        "Delta":   np.ones(n) * 0.05,
        "omega1":  np.ones(n) * 9.0,
        "gamma1":  np.ones(n) * 5.0,
        "mu":      np.ones(n) * 0.03,
        "km":      np.ones(n) * 0.5,
        "eta":     np.ones(n) * 2.0,
        "epsilon": np.ones(n) * 1.0,
        "beta0":   np.ones(n) * 0.5,
        "k3":      np.ones(n) * 2.0,
    }

    if params is not None:
        for key, value in params.items():
            default_params[key] = np.asarray(value, dtype=float) + np.zeros(n)

    lam1 = default_params["lambda1"]
    lam2 = default_params["lambda2"]
    r1 = default_params["r1"]
    r2 = default_params["r2"]
    Delta = default_params["Delta"]
    omega1 = default_params["omega1"]
    gamma1 = default_params["gamma1"]
    mu = default_params["mu"]
    km = default_params["km"]
    eta = default_params["eta"]
    eps_gain = default_params["epsilon"]
    beta0 = default_params["beta0"]
    k3 = default_params["k3"]

    g = np.asarray(state.get("g", np.zeros(n)), dtype=float).reshape(-1)
    k1_gain = np.asarray(state.get("k1", np.ones(n) * 1.5), dtype=float).reshape(-1)

    Delta = np.maximum(Delta, 1e-6)
    abs_e = np.abs(error)
    sign_e = _sat_sign(error, smooth_sign_eps)

    c1 = (2.0 - r2) * (Delta ** (r2 - 1.0))
    c2 = (r2 - 1.0) * (Delta ** (r2 - 2.0))

    far_region = abs_e >= Delta

    beta = np.where(
        far_region,
        (abs_e ** r2) * sign_e,
        c1 * error + c2 * (error ** 2) * sign_e
    )

    beta_dot = np.where(
        far_region,
        r2 * (abs_e ** (r2 - 1.0)) * error_dot,
        c1 * error_dot + 2.0 * c2 * abs_e * error_dot
    )

    s = error_dot + lam1 * (abs_e ** r1) * sign_e + lam2 * beta

    sign_s = _sat_sign(s, smooth_sign_eps)
    sqrt_abs_s = np.sqrt(np.abs(s) + 1e-12)

    phi1 = sqrt_abs_s * sign_s + k3 * s
    phi2 = 0.5 * sign_s + 1.5 * k3 * sqrt_abs_s * sign_s + (k3 ** 2) * s

    k1_dot = np.empty(n, dtype=float)
    active = k1_gain > km
    k1_dot[active] = (
        omega1[active]
        * np.sqrt(gamma1[active] / 2.0)
        * np.sign(np.abs(s[active]) - mu[active])
    )
    k1_dot[~active] = eta[~active]

    k1_gain = k1_gain + k1_dot * dt
    k1_gain = np.maximum(k1_gain, km + 1e-6)

    k2_gain = 2.0 * eps_gain * k1_gain + beta0 + 4.0 * (eps_gain ** 2)

    g_dot = -k2_gain * phi2
    g = g + g_dot * dt

    terminal_comp = (
        lam1 * r1 * (abs_e ** (r1 - 1.0)) * error_dot
        + lam2 * beta_dot
    )

    qddot_cmd = qd_ddot + terminal_comp + k1_gain * phi1 - g

    if M_hat_diag is None:
        M_hat_diag = np.array([0.0015, 0.0015, 0.001, 0.001, 0.001, 0.001], dtype=float)

    M_hat_diag = np.asarray(M_hat_diag, dtype=float).reshape(-1)
    if M_hat_diag.size != n:
        M_hat_diag = np.ones(n) * float(M_hat_diag[0])

    tau = M_hat_diag * qddot_cmd

    if torque_limits is not None:
        torque_limits = np.asarray(torque_limits, dtype=float).reshape(-1)
        tau = np.clip(tau, -torque_limits, torque_limits)

    state["g"] = g
    state["k1"] = k1_gain
    state["k2"] = k2_gain

    return state, tau, s