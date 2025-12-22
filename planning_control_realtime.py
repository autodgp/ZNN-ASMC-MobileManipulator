import mujoco
import numpy as np
from mujoco import viewer
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from ctrl import online_timedelay_asmc
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'         
plt.rcParams['mathtext.fontset'] = 'stix'               
plt.rcParams['mathtext.rm'] = 'Times New Roman'         

plot_data = {
    'joint_angles': [], 'joint_velocities': [],
    'joint_torques': [], 'position_errors': [], 'timesteps': []
}

def plot_joint_data(data, save_fig=True,save_dir='planning_control_realtime',filename_prefix='joint_data'):
    titles = ['Joint Angles', 'Joint Velocities', 'Joint Torques', 'Joints Tracking Errors']
    keys = ['joint_angles', 'joint_velocities', 'joint_torques', 'position_errors']
    ylabels = [r'$q$ (rad)', r'$\dot{q}$ (rad/s)', r'$\tau$ (N·m) ', r'$e$ (rad) ']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'x', 'D', '*']
    legend_font = FontProperties(family='Times New Roman', size=15) 

    #joint angles
    plt.figure(figsize=(6, 4.5), dpi=300)  
    for i in range(6):
            values = [d[i] for d in data['joint_angles']]
            plt.plot(data['timesteps'], values,
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=1.5,
                     label=rf'$q_{i + 1}$')
            
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel(ylabels[0], fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')

    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlim([0, 10])
    plt.legend(loc='upper right',prop=legend_font)
    plt.tight_layout()


    #joint velocities
    plt.figure(figsize=(6, 4.5), dpi=300) 
    for i in range(6):
            values = [d[i] for d in data['joint_velocities']]
            plt.plot(data['timesteps'], values,
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=1.5,
             label=rf'$\dot{{{{q}}}}_{{{i + 1}}}$')
            
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel(ylabels[1], fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')

    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlim([0, 10])
    plt.legend(loc='upper right',prop=legend_font)
    plt.tight_layout()
    # inset
    ax = plt.gca()
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right',
                   bbox_to_anchor=(-0.8, -0.93, 1.5, 1.5), bbox_transform=ax.transAxes)
    for i in range(6):
        values = [d[i] for d in data['joint_velocities']]
        axins.plot(data['timesteps'], values,
               linestyle=line_styles[i % len(line_styles)], linewidth=1.)
    axins.set_xlim(4, 6)
    axins.set_ylim(-0.2, 0.2)
    axins.grid(True, linestyle=':', linewidth=0.5)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", linewidth=0.6, alpha=0.5)


    #joint torques
    plt.figure(figsize=(6, 4.5), dpi=300)
    for i in range(6):
            values = [d[i] for d in data['joint_torques']]
            plt.plot(data['timesteps'], values,
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=1.5,
                     label=rf'$\tau_{i + 1}$')
            
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel(ylabels[2], fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')

    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlim([0, 10])
    plt.legend(loc='upper right',prop=legend_font)
    # inset
    ax = plt.gca()
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right',
                   bbox_to_anchor=(-0.8, -0.96, 1.5, 1.5), bbox_transform=ax.transAxes)
    for i in range(6):
        values = [d[i] for d in data['joint_torques']]
        axins.plot(data['timesteps'], values,
               linestyle=line_styles[i % len(line_styles)], linewidth=1.)
    axins.set_xlim(4, 6)
    axins.set_ylim(-0.5, 1.5)
    axins.grid(True, linestyle=':', linewidth=0.5)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    if save_fig:
         plt.savefig(f'{save_dir}/realtime_joint_torques.png') 

    #Joint Tracking Errors
    plt.figure(figsize=(6, 4.5), dpi=300)
    for i in range(6):
            values = [d[i] for d in data['position_errors']]
            plt.plot(data['timesteps'], values,
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=1.5,
                     label=rf'$e_{i + 1}$')
            
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel(ylabels[3], fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')

    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlim([0, 10])
    plt.legend(loc='upper right',prop=legend_font)
    # inset
    ax = plt.gca()
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right',
                   bbox_to_anchor=(-0.7, -0.9, 1.4, 1.4), bbox_transform=ax.transAxes)
    for i in range(6):
        values = [d[i] for d in data['position_errors']]
        axins.plot(data['timesteps'], values,
               linestyle=line_styles[i % len(line_styles)], linewidth=1.)
    axins.set_xlim(4, 6)
    axins.set_ylim(-0.0003, 0.0003)
    axins.grid(True, linestyle=':', linewidth=0.5)
    axins.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axins.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
    offset = axins.yaxis.get_offset_text()
    offset.set_fontname('Times New Roman')
    offset.set_fontsize(10)
    axins.grid(True, linestyle=':', linewidth=0.5)
    for tick in axins.get_xticklabels() + axins.get_yticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontsize(10)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    if save_fig:
         plt.savefig(f'{save_dir}/realtime_position_errors.png') 

    # Planned Joint Angles
    plt.figure(figsize=(6, 4.5), dpi=300)
    for i in range(6):
        values = theta_history[i, :len(data['timesteps'])]  
        plt.plot(data['timesteps'], values,
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=1.5,
                 label=rf'$q_{{d{i + 1}}}$')
    
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel(r'$q_d$ (rad)', fontsize=15, fontname='Times New Roman')
    #plt.title('Planned Joint Angles', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')
    plt.xlim([0, 10])
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'planning_joint_angle.png'))


    # Planned Joint Velocities
    plt.figure(figsize=(6, 4.5), dpi=300)
    for i in range(6):
        values = theta_dot_history[i, :len(data['timesteps'])]  
        plt.plot(data['timesteps'], values,
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=1.5,
                 label = rf'$\dot{{q}}_{{d{i + 1}}}$')
    
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel(r'$\dot{q}_d$ (rad/s)', fontsize=15, fontname='Times New Roman')
    #plt.title('Planned Joint Velocities', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')
    plt.xlim([0, 10])
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', prop=legend_font)
    # inset
    ax = plt.gca()
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right',
                   bbox_to_anchor=(-0.8, -0.96, 1.5, 1.5), bbox_transform=ax.transAxes)
    for i in range(6):
        values = theta_dot_history[i, :len(data['timesteps'])]  
        axins.plot(data['timesteps'], values,
               linestyle=line_styles[i % len(line_styles)], linewidth=1.)
    axins.set_xlim(4, 6)
    axins.set_ylim(-0.2, 0.25)
    axins.grid(True, linestyle=':', linewidth=0.5)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'planning_joint_vel.png'))

    # End-effector Position Errors
    target_traj = np.array([
        x_traj[:9999],  
        y_traj[:9999],  
        z_traj[:9999]   
    ]).T  # (9999, 3)
    errors_xyz = np.array(actual_traj) - target_traj
    labels_xyz = ['X error', 'Y error', 'Z error']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    plt.figure(figsize=(6, 4.5), dpi=300)
    for i in range(3):
            plt.plot(data['timesteps'], errors_xyz[:, i], linestyle=line_styles[i], linewidth=1.8, label=labels_xyz[i])
    plt.xlabel('Time (s)', fontsize=15, fontname='Times New Roman')
    plt.ylabel('Position Error (m)', fontsize=15, fontname='Times New Roman')
    #plt.title('End-effector Position Errors', fontsize=15, fontname='Times New Roman')
    plt.xticks(fontsize=15, fontname='Times New Roman')
    plt.yticks(fontsize=15, fontname='Times New Roman')
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlim([0, 10])
    plt.legend(prop=legend_font)
    plt.tight_layout()
    ax = plt.gca()
    axins = inset_axes(ax, width="30%", height="30%", loc='center',
            bbox_to_anchor=(-0.4, 0.022, 1.5 ,1.4), bbox_transform=ax.transAxes
                   )
    for i in range(3):
        axins.plot(data['timesteps'], errors_xyz[:, i], linestyle=line_styles[i], linewidth=1.0, label=labels_xyz[i])
        x1, x2, y1, y2 = 8.,8.5, -3e-5, 2e-5
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axins.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        offset = axins.yaxis.get_offset_text()
        offset.set_fontname('Times New Roman')
        offset.set_fontsize(10)
        axins.grid(True, linestyle=':', linewidth=0.5)
        for tick in axins.get_xticklabels() + axins.get_yticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", linewidth=0.6, alpha=0.5)

    plt.savefig(f'{save_dir}/realtime_target_errors.png')

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
    return np.vstack((jacp[:, :6], jacr[:, :6]))


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
lambda_factor = 0.8
delta = 0.4
xi = np.zeros((m, 1))
zeta = np.zeros((2 * n, 1))
nu = 1e-6

C = np.vstack((np.eye(n), -np.eye(n)))
F = np.zeros((3*n+m, 3*n+m))
s = np.zeros((3*n+m, 1))
AF_adaptive_param = np.full((24, 1), 0.5, dtype=float)  

# Mujoco model
model = mujoco.MjModel.from_xml_path("arm_scene_ground.xml")  
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
end_effector_id = model.site('ee_tip').id
model.opt.timestep = dt  

theta0 = np.array([0.3218  , -0.5534  ,  0.3358  ,  0.0000    ,1.7884 ,   0.3218])  
data.qpos[:n] = np.copy(theta0)

theta = theta0.copy()
theta_dot = np.zeros((n, 1))
theta_ddot = np.zeros((n, 1))
q = np.vstack((theta_dot, xi, zeta))
theta0_star = np.array([1.88869226e-08 ,-1.17356586e+00 , 1.07977379e+00 ,-3.27183860e-08,7.62041335e-01, -3.02992341e-08 ])

# Limits
theta_max = np.array([2.8798, 1.8962, 1.6057, 3.0925, 2.29402, 3.1415])
theta_min = -theta_max
theta_dot_max = np.array([0.6]*6)
theta_dot_min = -theta_dot_max

# Reference Trajectory Generation
r = 0.1
theta_r = np.linspace(0, 2*np.pi, n_points)
x_traj = -0.3 + r * np.cos(theta_r)
y_traj = 0. + r * np.sin(theta_r)
z_traj = np.full(n_points, 0.2)

position_history = np.zeros((6, N))
theta_history = np.zeros((6, N))
theta_dot_history = np.zeros((6, N))
theta_history[:, 0] = theta0
tau_history = []
actual_traj, ideal_traj = [], []

# Controller State Variables
adaptive_param = np.ones(6)
phi_hat_t_minus_L = np.zeros(6)
error_q_t_minus_L = np.zeros(6)
error_q_dot_t_minus_L = np.zeros(6)
tau = np.zeros(6)
sum_sign = np.zeros((6, 1))  

print("Starting simulation...")
for k in range(N-1):
    r_des = np.array([x_traj[k], y_traj[k], z_traj[k], 0, 0, 0])
    r_dot = np.array([(x_traj[k+1]-x_traj[k])/dt,
                      (y_traj[k+1]-y_traj[k])/dt,
                      (z_traj[k+1]-z_traj[k])/dt, 0, 0, 0])

    mujoco.mj_forward(model, data)
    J = compute_jacobian(model, data, end_effector_id)
    position = data.site_xpos[end_effector_id].copy()
    pos = np.hstack((position, [0, 0, 0]))
    position_history[:, k] = pos.copy()
    
    theta = data.qpos[:6].copy()
    theta_dot = data.qvel[:6].copy()
    theta_dot = theta_dot.reshape(-1, 1)
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
    l = np.sqrt(x**2 + zeta**2 + nu)  #(12,1)W
    d = d.reshape(-1, 1)
    temp_l_d = l - d
    temp_l_d = temp_l_d.flatten()
    s = np.hstack((-phi, b, temp_l_d))
    s = s.reshape(-1, 1)
    
    epsilon = F @ q - s 
    F_dot = (F - F_prev) / dt
    s_dot = (s - s_prev) / dt

    # ZNN Motion Planner
    AF_adaptive_param, AF_result = Adaptive_SBP_AF(epsilon, AF_adaptive_param)
    q_dot = np.linalg.solve(F, -F_dot @ q - lambda_factor * AF_result + s_dot)
    q = q + dt * q_dot

    theta_dot = q[:n]
    xi = q[n:n+m]
    zeta = q[n+m:]
    theta += theta_dot.flatten() * dt
    theta_history[:, k+1] = theta
    theta_dot_history[:, k+1] = theta_dot.flatten()
    theta_ddot = (theta_dot.flatten() - theta_dot_history[:, k]) / dt
    
    # Dynamic Control (ASMC)
    q_target, q_dot_target, q_ddot_target = theta, theta_dot.flatten(), theta_ddot
    error_q = q_target - data.qpos[:6]
    error_q_dot = q_dot_target - data.qvel[:6]

    adaptive_param, phi_hat_t_minus_L, tau,sum_sign = online_timedelay_asmc(
        sum_sign, data, tau, q_ddot_target, error_q, error_q_dot,
        error_q_t_minus_L, error_q_dot_t_minus_L, phi_hat_t_minus_L, adaptive_param)
    
    data.ctrl[:6] = tau
    mujoco.mj_step(model, data)
    ee_pos = data.site_xpos[model.site('ee_tip').id].copy()
    actual_traj.append(ee_pos)
    add_trajectory_line(viewer, actual_traj, [1, 0, 0, 1], width=4)
    viewer.sync()
    #time.sleep(dt)

    # Data Logging
    error_q_t_minus_L, error_q_dot_t_minus_L = error_q.copy(), error_q_dot.copy()
    plot_data['joint_angles'].append(data.qpos[:6].copy())
    plot_data['joint_velocities'].append(data.qvel[:6].copy())
    plot_data['joint_torques'].append(data.ctrl[:6].copy())
    plot_data['position_errors'].append(error_q.copy())
    plot_data['timesteps'].append(data.time)

for k in range(N-1):
    data.qpos[:6] = theta_history[:, k]
    mujoco.mj_forward(model, data)
    ee_pos = data.site_xpos[model.site('ee_tip').id].copy()
    ideal_traj.append(ee_pos)


# Trajectory for comparison
actual_traj = np.array(actual_traj)
ideal_traj = np.array(ideal_traj)

# Target and Desired Trajectory Plot
fig = plt.figure(figsize=(6, 6)) 
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_traj, y_traj, z_traj, color='#ff7f0e', linestyle='-', linewidth=1.8, label='Target Trajectory')
ax.plot(ideal_traj[:, 0], ideal_traj[:, 1], ideal_traj[:, 2], color='blue', linestyle='-.',linewidth=1.8, label='Desired Trajectory')

# start-end
ax.scatter(ideal_traj[0, 0], ideal_traj[0, 1], ideal_traj[0, 2], color='green', s=40, label='Start', marker='o')
ax.text(ideal_traj[0, 0], ideal_traj[0, 1], ideal_traj[0, 2]+0.0002, 'Start', fontsize=12, color='green',fontname='Times New Roman')
ax.scatter(ideal_traj[-1, 0], ideal_traj[-1, 1], ideal_traj[-1, 2], color='red', s=40, label='End', marker='s')
ax.text(ideal_traj[-1, 0]+0.001, ideal_traj[-1, 1]+0.008, ideal_traj[-1, 2]-0.001, 'End', fontsize=12, color='red',fontname='Times New Roman')

# axis and font
ax.set_xlabel('X(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_ylabel('Y(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_zlabel('Z(m)', fontsize=15, labelpad=-35,fontname='Times New Roman')
#ax.set_title('End-effector Trajectory Tracking', fontsize=25,fontname='Times New Roman')
for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(15)

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
save_dir = 'planning_control_realtime'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'realtimg_planing')
plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')

# plot actual trajectory and desired trajectory
fig = plt.figure(figsize=(6, 6))  
ax = fig.add_subplot(111, projection='3d')

ax.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2], color='red', linestyle='--',linewidth=1.8, label='Actual Trajectory')
ax.plot(ideal_traj[:, 0], ideal_traj[:, 1], ideal_traj[:, 2], color='blue', linestyle='-.',linewidth=1.8, label='Desired Trajectory')

# Start-end
ax.scatter(actual_traj[0, 0], actual_traj[0, 1], actual_traj[0, 2], color='green', s=40, label='Start', marker='o')
ax.text(actual_traj[0, 0], actual_traj[0, 1], actual_traj[0, 2]+0.0002, 'Start', fontsize=12, color='green',fontname='Times New Roman')

ax.scatter(actual_traj[-1, 0], actual_traj[-1, 1], actual_traj[-1, 2], color='red', s=40, label='End', marker='s')
ax.text(actual_traj[-1, 0]+0.001, actual_traj[-1, 1]+0.008, actual_traj[-1, 2]-0.001, 'End', fontsize=12, color='red',fontname='Times New Roman')

ax.set_xlabel('X(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_ylabel('Y(m)', fontsize=15, labelpad=8,fontname='Times New Roman')
ax.set_zlabel('Z(m)', fontsize=15, labelpad=-35,fontname='Times New Roman')
for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(15)

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
save_dir = 'planning_control_realtime'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'realtime_trajectory_tracking_error')
plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')

# plot joint data here
plot_joint_data(plot_data, save_fig=True, save_dir='planning_control_realtime')
plt.tight_layout()
plt.show()


