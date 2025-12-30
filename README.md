# ZNN-ASMC-Mobile-Manipulator

Official code for the paper (Under Review. The manuscript title has been updated in the revised resubmission) :**"Zeroing Neural Network-Based Motion Planning and Adaptive Sliding Mode Tracking Control for Mobile Manipulators"**

The project simulates the motion planning and tracking control of a mobile manipulator under floating-base disturbances using **MuJoCo**. It features a **Zeroing Neural Network (ZNN)** for real-time kinematic planning and an **Adaptive Sliding Mode Controller (ASMC)** based on Time-Delay Estimation (TDE) for robust dynamic control.

## Prerequisites
The code is written in Python 3. To run the simulations, you need to install the following dependencies:

```bash
pip install mujoco numpy matplotlib scipy
```

## Usage
To run the complete simulation showing both ZNN planning and ASMC tracking:

```bash
python planning_control_realtime.py
```

This will open the MuJoCo viewer and generate performance plots in the `results_realtime` folder.

To test the ZNN planner and dynamic controller under floating-base disturbances:

```bash
python planning_mat2py_floating.py
python planning_mat2py_floating_dynamic.py
```
