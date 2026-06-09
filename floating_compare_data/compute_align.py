import numpy as np
import pandas as pd
from pathlib import Path


def compute_time_weighted_metrics_1d(error_1d, dt):
    """
    计算单个关节误差的 ISE 和 IAE
    ISE = ∫ e(t)^2 dt
    IAE = ∫ |e(t)| dt
    """
    error_1d = np.asarray(error_1d).reshape(-1)

    ise = np.sum(error_1d ** 2) * dt
    iae = np.sum(np.abs(error_1d)) * dt

    return ise, iae


def load_error_q(path):
    """
    加载关节误差数据，保证形状为 N × joint_num
    """
    error_q = np.load(path)

    # 去掉多余维度
    error_q = np.squeeze(error_q)

    if error_q.ndim != 2:
        raise ValueError(f"{path} 的维度不是二维数组，当前 shape = {error_q.shape}")

    # 如果数据是 joint_num × N，则转置成 N × joint_num
    if error_q.shape[0] <= 10 and error_q.shape[1] > 10:
        error_q = error_q.T

    if error_q.shape[1] < 5:
        raise ValueError(f"{path} 的关节数不足 5 个，当前 shape = {error_q.shape}")

    return error_q


# ===================== 基本参数 =====================
dt = 0.001

base_dir = Path(__file__).resolve().parent

# 五种方法对应的误差文件路径
method_files = {
    "TDC": base_dir / "tdc" / "tdc_error_q_history.npy",
    "AISMC": base_dir / "aismc" / "aismc_error_q_history.npy",
    "ASMC": base_dir / "asmc" / "asmc_error_q_history.npy",
    "MAG-STC": base_dir / "MAG-STC" / "error_q_history.npy",
    "NFTSM": base_dir / "nftsm" / "error_q_history.npy",
}

# Python 索引从 0 开始：
# 第 3 关节 -> 索引 2
# 第 5 关节 -> 索引 4
joint3_idx = 2
joint5_idx = 4


# ===================== 计算 ISE 和 IAE =====================
results = {
    "Controller": [],
    "ISE (Joint 3)": [],
    "IAE (Joint 3)": [],
    "ISE (Joint 5)": [],
    "IAE (Joint 5)": [],
}

for method_name, file_path in method_files.items():
    error_q = load_error_q(file_path)

    error_joint3 = error_q[:, joint3_idx]
    error_joint5 = error_q[:, joint5_idx]

    joint3_ise, joint3_iae = compute_time_weighted_metrics_1d(error_joint3, dt)
    joint5_ise, joint5_iae = compute_time_weighted_metrics_1d(error_joint5, dt)

    results["Controller"].append(method_name)
    results["ISE (Joint 3)"].append(joint3_ise)
    results["IAE (Joint 3)"].append(joint3_iae)
    results["ISE (Joint 5)"].append(joint5_ise)
    results["IAE (Joint 5)"].append(joint5_iae)


df = pd.DataFrame(results)

print(df)

# 保存结果
df.to_csv(base_dir / "joint35_ise_iae_5methods.csv", index=False)
print("\n结果已保存为 joint35_ise_iae_5methods.csv")
