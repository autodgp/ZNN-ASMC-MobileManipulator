import numpy as np
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
    return adaptive_param_new, phi_hat,tau,sum_sign
    
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
    return adaptive_param_new, phi_hat,tau,sum_sign
