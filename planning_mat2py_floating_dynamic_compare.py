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
from ctrl import paper_nftsm_control_law
from ctrl import ft_cnt_mag_stc, init_ft_cnt_mag_stc_state


plt.rcParams['font.family'] = 'Times New Roman'         
plt.rcParams['mathtext.fontset'] = 'stix'               
plt.rcParams['mathtext.rm'] = 'Times New Roman'        

plot_data = {
    'joint_angles': [], 'joint_velocities': [],
    'joint_torques': [], 'position_errors': [], 'timesteps': [],
    'sliding_variables': []
}

def save_plot_data(plot_data, save_root='floating_compare_data', method_name='nftsm'):
    """
    Save simulation data for later comparison plotting.
    Saved data:
        timesteps
        joint_angles
        joint_velocities
        joint_torques
        position_errors
        sliding_variables
    """
    save_dir = os.path.join(save_root, method_name)
    os.makedirs(save_dir, exist_ok=True)

    data_to_save = {
        'timesteps': np.array(plot_data['timesteps']),
        'joint_angles': np.array(plot_data['joint_angles']),
        'joint_velocities': np.array(plot_data['joint_velocities']),
        'joint_torques': np.array(plot_data['joint_torques']),
        'position_errors': np.array(plot_data['position_errors']),
        'sliding_variables': np.array(plot_data['sliding_variables'])
    }

    # Python 后续画图最方便
    np.savez(os.path.join(save_dir, 'plot_data.npz'), **data_to_save)

    # 单独保存 error_q_history 和 tau_history，方便你后面直接调用
    np.save(os.path.join(save_dir, 'error_q_history.npy'), data_to_save['position_errors'])
    np.save(os.path.join(save_dir, 'tau_history.npy'), data_to_save['joint_torques'])
    np.save(os.path.join(save_dir, 'time_history.npy'), data_to_save['timesteps'])

    print(f'[Saved] plot_data saved to: {save_dir}')

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

ft_state = init_ft_cnt_mag_stc_state(n=6, k1_init=1.5)


for i in range(t):
    current_time = data.time
    desired_base_pos = base_amp * np.sin(2 * current_time) + 0.02 * np.cos(6 * current_time) + 0.01 * np.exp(-3*current_time)  # floating-base disturbances
    data.qpos[0] = desired_base_pos
    base_position.append(data.qpos[0]+0.5)    

    q_target, q_dot_target, q_ddot_target = theta_history[:, i], theta_dot_history[:, i], theta_ddot_history[:, i]

    error_q = q_target - data.qpos[1:7]
    error_q_dot = q_dot_target - data.qvel[1:7]

    # adaptive_param, phi_hat_t_minus_L, tau,sum_sign,s_val = floating_timedelay_asmc(
    #     sum_sign, data, tau, q_ddot_target, error_q, error_q_dot,
    #     error_q_t_minus_L, error_q_dot_t_minus_L, phi_hat_t_minus_L, adaptive_param)

    tau, s_val, H_val = paper_nftsm_control_law(
       data,
       q_ddot_target,
       error_q,
      error_q_dot,
       model=model
    )
    
    # ft_state, tau, s_val = ft_cnt_mag_stc(
    # data,
    # q_ddot_target,
    # error_q,
    # error_q_dot,
    # state=ft_state,
    # dt=dt,
    # smooth_sign_eps=0.0)

    data.ctrl[1:7] = tau
    mujoco.mj_step(model, data) 

    ee_pos = data.site_xpos[model.site('ee_tip').id].copy()
    actual_traj.append(ee_pos)
    add_trajectory_line(viewer, actual_traj, [1, 0, 0, 1])

    #viewer.sync()
    #time.sleep(dt)

    error_q_t_minus_L, error_q_dot_t_minus_L = error_q.copy(), error_q_dot.copy()
    plot_data['sliding_variables'].append(s_val.copy())
    plot_data['joint_angles'].append(data.qpos[1:7].copy())
    plot_data['joint_velocities'].append(data.qvel[1:7].copy())
    plot_data['joint_torques'].append(data.ctrl[1:7].copy())
    plot_data['position_errors'].append(error_q.copy())
    plot_data['timesteps'].append(current_time)

save_plot_data(
    plot_data,
    save_root='floating_compare_data',
    method_name='online_timedelay_asmc'
)

base_position = np.array(base_position)
actual_traj = np.array(actual_traj)

save_dir = 'floating_compare_data'
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




