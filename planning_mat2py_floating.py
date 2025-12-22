import mujoco
import numpy as np
from mujoco import viewer
import matplotlib.pyplot as plt
import time
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import os
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['font.family'] = 'Times New Roman'         
plt.rcParams['mathtext.fontset'] = 'stix'               
plt.rcParams['mathtext.rm'] = 'Times New Roman'         

plot_data = {
    'joint_angles': [], 'joint_velocities': [],
    'timesteps': []
}

def add_trajectory_line(viewer, position, color=[0, 0, 0, 1], width=2.5):
    if len(position) < 2: 
        return
    mujoco.mjv_connector(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=width, from_=position[-2], to=position[-1]
    )
    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = color
    viewer.user_scn.ngeom += 1

def compute_jacobian(model, data, ee_site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    return np.vstack((jacp[:, 1:7], jacr[:, 1:7]))

def PT_AF(alpha):
    gamma, a1, a2, w = 1, 1, 1, 3
    k1 = 2 / gamma * w
    k2 = (0.5)**(1 - gamma/2) * (2*a1) / (gamma * w * np.sqrt(a1*a2))
    k3 = (0.5)**(1 + gamma/2) * 3**(gamma/2) * 2 * a2 / (gamma * w * np.sqrt(a1*a2))
    return k1*alpha + k2*np.abs(alpha)**(1 - gamma/2)*np.sign(alpha) + k3*np.abs(alpha)**(1 + gamma/2)*np.sign(alpha)

def L_AF(alpha):
    return alpha

def PE_AF(alpha):
    k = 0.8
    return alpha +  abs(alpha)**k * np.sign(alpha)

def DPE_AF(alpha):
    k = 2/3
    return alpha +  abs(alpha)**k * np.sign(alpha) + abs(alpha)**(1/k) * np.sign(alpha)

def SBP_AF(alpha):
    beta = 0.65
    return 0.5 * (np.abs(alpha)**beta + np.abs(alpha)**(1/beta)) * np.sign(alpha)

def Adaptive_SBP_AF(alpha, adaptive_param, dt=0.001):
    beta = 0.8
    for j in range(np.size(alpha)):
        if np.abs(alpha[j]) < 1e-6:
            adaptive_param[j] = 0.0
            continue
        if np.abs(alpha[j]) <= beta:
            adaptive_param[j] += beta * np.abs(alpha[j]) * dt
        else:
            adaptive_param[j] -= (1 - beta) * np.abs(alpha[j]) * dt
        
        adaptive_param[j] = max(adaptive_param[j], 0.0)
    
    result = adaptive_param * (np.abs(alpha)**beta + np.abs(alpha)**(1/beta)) * np.sign(alpha)
    return adaptive_param,result

def compute_joint_velocity_bounds(theta,theta_min, theta_max, theta_dot_min, theta_dot_max):
    beta1, beta2 = 0.8, 0.9
    delta1 = beta1 * theta_min
    delta2 = beta2 * theta_max
    delta3 = theta_min - delta1
    delta4 = theta_max - delta2

    kappa1 = 1 - np.square(np.square(np.sin(0.5*np.pi*np.sin(0.5*np.pi*(theta - delta1)/delta3))) )
    kappa2 = 1 - np.square(np.square(np.sin(0.5*np.pi*np.sin(0.5*np.pi*(theta - delta2)/delta4))) )

    J_minus = np.where(theta >= delta1, theta_dot_min,
                           np.where(theta >= theta_min, kappa1 * theta_dot_min, 0))
    J_plus = np.where(theta <= delta2, theta_dot_max,
                          np.where(theta <= theta_max, kappa2 * theta_dot_max, 0))
    return J_minus, J_plus

dt = 0.001
n = 6
m = 6
n_points = 10000 # number of points
N = 10000 # simulation steps

K = 50 
lambda_factor = 0.7
delta = 0.6
xi = np.zeros((m, 1))
zeta = np.zeros((2 * n, 1))
nu = 1e-6

C = np.vstack((np.eye(n), -np.eye(n)))
F = np.zeros((3*n+m, 3*n+m))
s = np.zeros((3*n+m, 1))
AF_adaptive_param = np.full((24, 1), 0.5, dtype=float) 

# Mujoco model
model = mujoco.MjModel.from_xml_path("arm_basemove.xml")  
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
end_effector_id = model.site('ee_tip').id

theta0 = np.array([0,  -4.81809265e-06 ,-3.40180323e-01,  6.40422295e-01, -3.66864295e-07,1.47054028e+00, -8.49525403e-06]) 
data.qpos[:7] = theta0
mujoco.mj_forward(model, data)
model.opt.timestep = dt  

theta = theta0[1:7].copy()
theta_dot = np.zeros((n, 1))
q = np.vstack((theta_dot, xi, zeta))
theta0_star = np.array([1.88869226e-08 ,-1.17356586e+00 , 1.07977379e+00 ,-3.27183860e-08,7.62041335e-01, -3.02992341e-08 ])

theta_max = np.array([2.8973, 1.8, 1.6057, 3.0718, 2.23402, 3.1])
theta_min = -theta_max
theta_dot_max = np.array([0.6]*6)
theta_dot_min = -theta_dot_max

r = 0.1
theta_r = np.linspace(0, 2*np.pi, n_points)
x_traj = -0.3 + r * np.cos(theta_r)
y_traj = 0. + r * np.sin(theta_r)
z_traj = np.full(n_points, 0.5)

position_history = np.zeros((6, N))
theta_history = np.zeros((6, N))
theta_dot_history = np.zeros((6, N))
theta_ddot_history = np.zeros((6, N))
theta_history[:, 0] = theta0[1:7]

actual_traj, ideal_traj = [], []
epsilon_history = []

# floating base
base_amp = 0.05
base_freq = 1
desired_base_pos = 0.0
base_position = []



for k in range(N - 1):
    current_time = data.time
    desired_base_pos = base_amp * np.sin(2 * current_time) + 0.02 * np.cos(6 * current_time) + 0.01 * np.exp(-3*current_time)  # floating-base disturbances
    data.qpos[0] = desired_base_pos
    base_position.append(data.qpos[0])

    r_des = np.array([x_traj[k], y_traj[k], z_traj[k], 0, 0, 0])
    r_dot = np.array([(x_traj[k+1]-x_traj[k])/dt,
                      (y_traj[k+1]-y_traj[k])/dt,
                      (z_traj[k+1]-z_traj[k])/dt, 0, 0, 0])
    J = compute_jacobian(model, data, end_effector_id)
    print(J, J.shape)
    position = data.site_xpos[end_effector_id].copy()
    pos = np.hstack((position, [0, 0, 0]))
    position_history[:, k] = pos.copy()
    
    b = r_dot + K * (r_des - pos)
    phi = delta * (theta - theta0_star)
    J_minus, J_plus = compute_joint_velocity_bounds(theta, theta_min, theta_max, theta_dot_min, theta_dot_max)
    d = np.hstack((J_plus, -J_minus))
    
    F_prev, s_prev = F.copy(), s.copy()
    F = np.block([[np.eye(n), J.T, C.T],
                  [J, np.zeros((6, 6)), np.zeros((6, 2*n))],
                  [-C, np.zeros((2*n, 6)), np.eye(2*n)]])

    theta_dot = theta_dot.flatten()
    x = d - C @ theta_dot
    x = x.reshape(-1, 1)
    l = np.sqrt(x**2 + zeta**2 + nu)  #(12,1)
    d = d.reshape(-1, 1)
    temp_l_d = l - d
    temp_l_d = temp_l_d.flatten()
    s = np.hstack((-phi, b, temp_l_d))
    s = s.reshape(-1, 1)
    
    epsilon = F @ q - s 
    epsilon_history.append(epsilon.flatten())
    F_dot = (F - F_prev) / dt
    s_dot = (s - s_prev) / dt
    AF_adaptive_param, AF_result = Adaptive_SBP_AF(epsilon, AF_adaptive_param)

    q_dot = np.linalg.solve(F, -F_dot @ q - lambda_factor * AF_result + s_dot)
    q = q + dt * q_dot

    theta_dot = q[:n]
    xi = q[n:n+m]
    zeta = q[n+m:]
    theta += theta_dot.flatten() * dt
    theta_history[:, k+1] = theta
    theta_dot_history[:, k+1] = theta_dot.flatten()
    theta_ddot_history[:,k+1] = (theta_dot.flatten() - theta_dot_history[:, k]) / dt
    

    data.qpos[1:7] = theta
    mujoco.mj_step(model, data)
    viewer.sync()
    # time.sleep(dt)
    ee_pos = data.site_xpos[model.site('ee_tip').id].copy()
    ideal_traj.append(ee_pos)
    plot_data['joint_angles'].append(data.qpos[1:7].copy())
    plot_data['joint_velocities'].append(data.qvel[1:7].copy())
    plot_data['timesteps'].append(data.time)
    add_trajectory_line(viewer, ideal_traj, [0, 0, 1, 1])

print(data.qpos[1:7])
ideal_traj = np.array(ideal_traj)
base_position = np.array(base_position)
scipy.io.savemat("planning_mat2py_floating/theta_history.mat", {"theta_history": theta_history})
scipy.io.savemat("planning_mat2py_floating/theta_dot_history.mat", {"theta_dot_history": theta_dot_history})
scipy.io.savemat("planning_mat2py_floating/theta_ddot_history.mat", {"theta_ddot_history": theta_ddot_history})

save_dir = 'planning_mat2py_floating'
os.makedirs(save_dir, exist_ok=True)

fig = plt.figure(figsize=(6, 6))  
ax = fig.add_subplot(111, projection='3d')

# plot 3d trajectory
ax.plot(x_traj, y_traj, z_traj, color='#ff7f0e', linestyle='-', linewidth=1.2, label='Target Trajectory')
ax.plot(ideal_traj[:, 0], ideal_traj[:, 1], ideal_traj[:, 2], color='blue',linestyle='-.',  linewidth=1.2, label='Desired Trajectory')

ax.scatter(ideal_traj[0, 0], ideal_traj[0, 1], ideal_traj[0, 2], color='green', s=40, label='Start', marker='o')
ax.text(ideal_traj[0, 0], ideal_traj[0, 1], ideal_traj[0, 2]+0.0005, 'Start', fontsize=9, color='green')
ax.scatter(ideal_traj[-1, 0], ideal_traj[-1, 1], ideal_traj[-1, 2], color='blue', s=40, label='End', marker='s')
ax.text(ideal_traj[-1, 0], ideal_traj[-1, 1], ideal_traj[-1, 2]+0.0005, 'End', fontsize=9, color='blue')

ax.set_xlabel('X(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_ylabel('Y(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_zlabel('Z(m)', fontsize=15, labelpad=-35,fontname='Times New Roman')

ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='z', pad=8)
legend_font = FontProperties(family='Times New Roman', size=14)
ax.legend(loc='upper right',
          bbox_to_anchor=(0.90, 0.8),
          frameon=True,
          prop=legend_font,
          edgecolor='black')  

ax.view_init(elev=25, azim=45)
save_path = os.path.join(save_dir, 'floating_trajectory')
plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')
print(f'图像已保存到: {save_path}.png')

error_x = ideal_traj[:, 0] - x_traj[:-1]
error_y = ideal_traj[:, 1] - y_traj[:-1]
error_z = ideal_traj[:, 2] - z_traj[:-1]

legend_font = FontProperties(family='Times New Roman', size=15) 
line_styles = ['-', '--', '-.', ':', '-', '--']

# --------- Trajectory Error X ---------
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=300, sharex=True)

# Plot actual and reference trajectory
ax1.plot(plot_data['timesteps'], x_traj[:-1], linestyle='-.', linewidth=1.6, label='Reference end-effector', color='blue', zorder=3)
ax1.plot(plot_data['timesteps'], ideal_traj[:, 0], linestyle='--', linewidth=1.6, label='Actual end-effector', color='orange')
ax1.set_xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
ax1.set_ylabel('X (m)', fontsize=15, fontname='Times New Roman')
ax1.grid(True, linestyle=':', linewidth=0.5)

# Secondary axis for tracking error
ax2 = ax1.twinx()
ax2.plot(plot_data['timesteps'], error_x, linestyle='-', linewidth=1.6, color='purple', label='Tracking error', zorder=1)
ax2.set_ylabel('Position Error (m)', fontsize=15, fontname='Times New Roman')
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))               
ax2.yaxis.set_major_formatter(fmt)
for item in list(ax2.get_yticklabels()) + [ax2.yaxis.get_offset_text()]:
    item.set_fontname('Times New Roman')
    item.set_fontsize(12)

for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(12)

for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(12)

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right',  bbox_to_anchor=(0.98, 0.65), prop={'size': 12, 'family': 'Times New Roman'})


# Axis limits and formatting
ax1.set_xlim([0, 10])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'floating_trajectory_errors_x.png'),dpi=300, bbox_inches='tight', pad_inches=0.1)

# --------- Trajectory Error Y ---------
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=300, sharex=True)

# Plot actual and reference trajectory
ax1.plot(plot_data['timesteps'], y_traj[:-1], linestyle='-.', linewidth=1.6, label='Reference end-effector', color='blue', zorder=3)
ax1.plot(plot_data['timesteps'], ideal_traj[:, 1], linestyle='--', linewidth=1.6, label='Actual end-effector', color='orange')
ax1.set_xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
ax1.set_ylabel('Y (m)', fontsize=15, fontname='Times New Roman')
ax1.grid(True, linestyle=':', linewidth=0.5)
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
# Secondary axis for tracking error
ax2 = ax1.twinx()
ax2.plot(plot_data['timesteps'], error_x, linestyle='-', linewidth=1.6, color='purple', label='Tracking error', zorder=1)
ax2.set_ylabel('Position Error (m)', fontsize=15, fontname='Times New Roman')
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))              
ax2.yaxis.set_major_formatter(fmt)
for item in list(ax2.get_yticklabels()) + [ax2.yaxis.get_offset_text()]:
    item.set_fontname('Times New Roman')
    item.set_fontsize(12)
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(12)
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(12)

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right',  bbox_to_anchor=(0.98, 0.65), prop={'size': 12, 'family': 'Times New Roman'})


# Axis limits and formatting
ax1.set_xlim([0, 10])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'floating_trajectory_errors_y.png'),dpi=300, bbox_inches='tight', pad_inches=0.1)

# --------- Trajectory Error Z ---------
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=300, sharex=True)

# Plot actual and reference trajectory
ax1.plot(plot_data['timesteps'], z_traj[:-1], linestyle='-.', linewidth=1.6, label='Reference end-effector', color='blue')
ax1.plot(plot_data['timesteps'], ideal_traj[:, 2], linestyle='--', linewidth=1.6, label='Actual end-effector', color='orange')
ax1.plot(plot_data['timesteps'], base_position[:N-1] + 0.5, linestyle='-', linewidth=1.6, label='Floating disturbance', color='red')
ax1.set_xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
ax1.set_ylabel('Z (m)', fontsize=15, fontname='Times New Roman')
ax1.grid(True, linestyle=':', linewidth=0.5)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f')) 
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(12)
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(12)

left_ylim = ax1.get_ylim()
left_y_center = 0.5

ax2 = ax1.twinx()
ax2.plot(plot_data['timesteps'], error_z, linestyle='-', linewidth=1.6, color='purple', label='Tracking error')

right_ylim_upper = max(abs(error_z))
ax2.set_ylim(-right_ylim_upper, right_ylim_upper)

ax2.set_ylabel('Position Error (m)', fontsize=15, fontname='Times New Roman')
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))              
ax2.yaxis.set_major_formatter(fmt)
ax2.yaxis.offsetText.set_fontname('Times New Roman')
ax2.yaxis.offsetText.set_fontsize(12)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2,labels_1 + labels_2, bbox_to_anchor=(0.55, 1), loc='upper left', frameon=True, prop={'size': 12, 'family': 'Times New Roman'})

ax1.set_xlim([0, 10])
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
ax2.tick_params(axis='y', labelsize=12)
# Axis limits and formatting
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'floating_trajectory_errors_z.png'),dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.figure(figsize=(6, 4.5), dpi=300)  
for i in range(6):
    values = theta_history
    plt.plot(plot_data['timesteps'], values[i,:-1],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=1.5,
                label=rf'$q_{{{i + 1}}}$')

plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
plt.ylabel(r'$q$ (rad)', fontsize=15, fontname='Times New Roman')
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.xlim([0, 10])
plt.legend(loc='upper right', prop=legend_font)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'floating_joint_angles.png'))

plt.figure(figsize=(6, 4.5), dpi=300)  
for i in range(6):
    values = theta_dot_history
    plt.plot(plot_data['timesteps'], values[i,:-1],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=1.5,
                label=rf'$\dot{{q}}_{{{i + 1}}}$')

plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
plt.ylabel(r'$\dot{q}$ (rad/s)', fontsize=15, fontname='Times New Roman')
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.xlim([0, 10])
plt.legend(loc='upper right', prop=legend_font)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'floating_joint_velocities.png'))
plt.show()