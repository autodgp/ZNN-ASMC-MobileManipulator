import mujoco
import numpy as np
from mujoco import viewer
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import os
import scipy.io
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from ctrl import floating_timedelay_asmc

plt.rcParams['font.family'] = 'Times New Roman'         
plt.rcParams['mathtext.fontset'] = 'stix'               
plt.rcParams['mathtext.rm'] = 'Times New Roman'        

plot_data = {
    'joint_angles': [], 'joint_velocities': [],
    'joint_torques': [], 'position_errors': [], 'timesteps': []
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


# load data
theta_history = scipy.io.loadmat('planning_mat2py_floating/theta_history.mat')['theta_history']  
theta_dot_history = scipy.io.loadmat('planning_mat2py_floating/theta_dot_history.mat')['theta_dot_history']
theta_ddot_history = scipy.io.loadmat('planning_mat2py_floating/theta_ddot_history.mat')['theta_ddot_history']
t = theta_history.shape[1]  

dt = 0.001
n = 6
m = 6
n_points = 10000 # number of points
N = 10000 # simulation steps

# Mujoco model
model = mujoco.MjModel.from_xml_path("arm_basemove.xml")  
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
end_effector_id = model.site('ee_tip').id
model.opt.timestep = dt  

# initial joint state
theta0 = np.array([0,  -4.81809265e-06 ,-3.40180323e-01,  6.40422295e-01, -3.66864295e-07,1.47054028e+00, -8.49525403e-06]) 

data.qpos[:7] = theta0
mujoco.mj_forward(model, data)
model.opt.timestep = dt  

theta = theta0[1:7].copy()
theta_dot = np.zeros((n, 1))
theta_ddot = np.zeros((n, 1))

r = 0.1
theta_r = np.linspace(0, 2*np.pi, t)
x_traj = -0.3 + r * np.cos(theta_r)
y_traj = 0. + r * np.sin(theta_r)
z_traj = np.full(t, 0.5)

position_history = np.zeros((6, N))
actual_traj, ideal_traj = [], []
adaptive_param = np.ones(6)
phi_hat_t_minus_L = np.zeros(6)
error_q_t_minus_L = np.zeros(6)
error_q_dot_t_minus_L = np.zeros(6)
tau = np.zeros(6)
sum_sign = np.zeros((6, 1))  # 初始化sum_sign

# floating base
base_amp = 0.05
base_freq = 1
desired_base_pos = 0.0
base_position = []


for i in range(t):
    current_time = data.time
    desired_base_pos = base_amp * np.sin(2 * current_time) + 0.02 * np.cos(6 * current_time) + 0.01 * np.exp(-3*current_time)  # floating-base disturbances
    data.qpos[0] = desired_base_pos
    base_position.append(data.qpos[0]+0.5)    

    q_target, q_dot_target, q_ddot_target = theta_history[:, i], theta_dot_history[:, i], theta_ddot_history[:, i]

    error_q = q_target - data.qpos[1:7]
    error_q_dot = q_dot_target - data.qvel[1:7]

    adaptive_param, phi_hat_t_minus_L, tau,sum_sign = floating_timedelay_asmc(
        sum_sign, data, tau, q_ddot_target, error_q, error_q_dot,
        error_q_t_minus_L, error_q_dot_t_minus_L, phi_hat_t_minus_L, adaptive_param)

    data.ctrl[1:7] = tau
    mujoco.mj_step(model, data) 

    ee_pos = data.site_xpos[model.site('ee_tip').id].copy()
    actual_traj.append(ee_pos)
    add_trajectory_line(viewer, actual_traj, [1, 0, 0, 1])

    viewer.sync()
    #time.sleep(dt)

    error_q_t_minus_L, error_q_dot_t_minus_L = error_q.copy(), error_q_dot.copy()
    plot_data['joint_angles'].append(data.qpos[1:7].copy())
    plot_data['joint_velocities'].append(data.qvel[1:7].copy())
    plot_data['joint_torques'].append(data.ctrl[1:7].copy())
    plot_data['position_errors'].append(error_q.copy())
    plot_data['timesteps'].append(current_time)



base_position = np.array(base_position)
actual_traj = np.array(actual_traj)

save_dir = 'planning_mat2py_floating_dynamic'
save_path = os.path.join(save_dir, 'floating_dynamic')

error_x = actual_traj[:, 0] - x_traj
error_y = actual_traj[:, 1] - y_traj
error_z = actual_traj[:, 2] - z_traj

actual_traj = np.array(actual_traj)
np.save('planning_control_realtime/theta_history.npy', theta_history)
fig = plt.figure(figsize=(6, 6))  
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_traj, y_traj, z_traj, color='#ff7f0e', linestyle='-', linewidth=1.8, label='Target Trajectory')
ax.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2], color='red', linestyle='-.',linewidth=1.2, label='Actual Trajectory')
ax.scatter(actual_traj[0, 0], actual_traj[0, 1], actual_traj[0, 2], color='green', s=40, label='Start', marker='o')
ax.text(actual_traj[0, 0], actual_traj[0, 1], actual_traj[0, 2]+0.0002, 'Start', fontsize=12, color='green',fontname='Times New Roman')
ax.scatter(actual_traj[-1, 0], actual_traj[-1, 1], actual_traj[-1, 2], color='red', s=40, label='End', marker='s')
ax.text(actual_traj[-1, 0]+0.001, actual_traj[-1, 1]+0.008, actual_traj[-1, 2]-0.001, 'End', fontsize=12, color='red',fontname='Times New Roman')
ax.set_xlabel('X(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_ylabel('Y(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_zlabel('Z(m)', fontsize=15, labelpad=-35,fontname='Times New Roman')
#ax.set_title('End-effector Trajectory Tracking', fontsize=25,fontname='Times New Roman')
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
plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')
legend_font = FontProperties(family='Times New Roman', size=15) 
line_styles = ['-', '--', '-.', ':', '-', '--']

# --------- Trajectory Error XYZ ---------
fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4.5), dpi=300, sharex=True)

ax1.plot(plot_data['timesteps'], error_x, label='X error', linestyle='-.', linewidth=1.6)
ax1.plot(plot_data['timesteps'], error_y, label='Y error', linestyle='--', linewidth=1.6)
ax1.plot(plot_data['timesteps'], error_z, label='Y error', linestyle='--', linewidth=1.6)
ax1.set_xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
ax1.set_ylabel('Position Error (m)', fontsize=15, fontname='Times New Roman')
ax1.legend(prop=legend_font)
ax1.grid(True, linestyle=':', linewidth=0.5)
ax1.set_xlim([0, 10])
#ax1.set_title('End-Effector Trajectory Errors', fontsize=14, fontname='Times New Roman')

plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'floating_dyanamics_errors_xyz_.png'))


# --------- Joint Angles and Velocities ---------
theta_history= np.array(plot_data['joint_angles']).T  # shape: (6, T)
theta_dot_history = np.array(plot_data['joint_velocities']).T
plt.figure(figsize=(6, 4.5), dpi=300)  
for i in range(6):
    values = theta_history
    plt.plot(plot_data['timesteps'], values[i],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=1.5,
                label=rf'$q_{{{i + 1}}}$')

plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
plt.ylabel(r'${q}$ (rad)', fontsize=15, fontname='Times New Roman')
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
    plt.plot(plot_data['timesteps'], values[i],
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

# --------- Joint Torques ---------
plt.figure(figsize=(6, 4.5), dpi=300)
for i in range(6):
        values = [d[i] for d in plot_data['joint_torques']]
        plt.plot(plot_data['timesteps'], values,
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=1.5,
                    label=rf'$\tau_{i + 1}$')
            
plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
plt.ylabel(r'$\tau$ (N·m)', fontsize=15, fontname='Times New Roman')
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')

plt.grid(True, linestyle=':', linewidth=0.5)
plt.xlim([0, 10])
plt.legend(loc='upper right',prop=legend_font)
plt.tight_layout()
plt.savefig(f'{save_dir}/floating_joint_torques.png') 



# --------- Joint Angle Tracking Errors ---------
joint_errors = np.array(plot_data['position_errors'])  # shape: (N, 6)

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
for i in range(6):
    ax.plot(plot_data['timesteps'], joint_errors[:, i], linestyle=line_styles[i % len(line_styles)],
            linewidth=1.5, label=rf'$e_{i + 1}$')
ax.set_xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
ax.set_ylabel(r'$e$ (rad)', fontsize=15, fontname='Times New Roman')
#ax.set_title('Joint Tracking Errors', fontsize=15, fontname='Times New Roman')
ax.grid(True, linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', prop=legend_font)
ax.set_xlim([0, 10])
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.tight_layout()
ax = plt.gca()
axins = inset_axes(ax, width="30%", height="30%", loc='center',
            bbox_to_anchor=(0.0, 0.3, 1,1), bbox_transform=ax.transAxes
                    )
for i in range(6):
        axins.plot(plot_data['timesteps'], joint_errors[:, i], linestyle=line_styles[i], linewidth=1.8)
        x1, x2, y1, y2 = 7,8, -0.001, 0.001
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(True, linestyle=':', linewidth=0.5)
        for tick in axins.get_xticklabels() + axins.get_yticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(12)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", linewidth=0.6, alpha=0.5)
plt.savefig(os.path.join(save_dir, f'floating_dynamics_joints_error.png'))

plt.show()

