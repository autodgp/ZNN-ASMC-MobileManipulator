"""
compare_dyn_floating_5methods.py

Five-method floating-base dynamic comparison script.

This script follows the plotting style of compare_dyn.py:
- Times New Roman font
- Joint 3 / Joint 5 torque comparison
- Joint 3 / Joint 5 tracking-error comparison
- Inset zoom windows
- ISE / IAE numerical output

Expected folder structure:
.
├── aismc/
│   ├── aismc_error_q_history.npy
│   └── aismc_tau_history.npy
├── asmc/
│   ├── asmc_error_q_history.npy
│   └── asmc_tau_history.npy
├── MAG-STC/
│   ├── error_q_history.npy
│   ├── tau_history.npy
│   └── time_history.npy        # optional
├── nftsm/
│   ├── error_q_history.npy
│   ├── tau_history.npy
│   └── time_history.npy        # optional
└── tdc/
    ├── tdc_error_q_history.npy
    └── tdc_tau_history.npy

Output folder:
compare_floating_5methods_joint35/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import ScalarFormatter


# =========================
# User settings
# =========================
DATA_ROOT = Path(".")
SAVE_DIR = Path("compare_floating_5methods_joint35")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.001
T_MAX = 10.0

# Compared joints: joint 3 and joint 5 in paper notation
JOINTS_TO_PLOT = {
    3: 2,   # joint label -> Python index
    5: 4,
}

# Inset windows. Adjust if your local magnified ranges are not ideal.
INSET_CONFIG = {
    "tau_3": {"xlim": (0.05, 0.1), "ylim": (-1, 1.5), "bbox": (-0.25, 0.10, 1.20, 1.20), "loc1": 3, "loc2": 4},
    "tau_5": {"xlim": (0.05, 0.1), "ylim": (-1, 3), "bbox": (-0.30, -0.35, 1.20, 1.20), "loc1": 1, "loc2": 2},
    "err_3": {"xlim": (0.0, 1.0), "ylim": (-0.01, 0.01), "bbox": (-0.25, 0.12, 1.20, 1.20), "loc1": 3, "loc2": 4},
    "err_5": {"xlim": (0.0, 1.0), "ylim": (-0.01, 0.01), "bbox": (-0.25, 0.12, 1.20, 1.20), "loc1": 3, "loc2": 4},
}

METHODS = [
    {
        "key": "tdc",
        "label": "TDC",
        "folder": "tdc",
        "tau_file": "tdc_tau_history.npy",
        "err_file": "tdc_error_q_history.npy",
        "time_file": None,
    },
    {
        "key": "aismc",
        "label": "AISMC",
        "folder": "aismc",
        "tau_file": "aismc_tau_history.npy",
        "err_file": "aismc_error_q_history.npy",
        "time_file": None,
    },
    {
        "key": "nftsm",
        "label": "NFTSM",
        "folder": "nftsm",
        "tau_file": "tau_history.npy",
        "err_file": "error_q_history.npy",
        "time_file": "time_history.npy",
    },
    {
        "key": "magstc",
        "label": "MAG-STC",
        "folder": "MAG-STC",
        "tau_file": "tau_history.npy",
        "err_file": "error_q_history.npy",
        "time_file": "time_history.npy",
    },
    {
        "key": "asmc",
        "label": "ASMC",
        "folder": "asmc",
        "tau_file": "asmc_tau_history.npy",
        "err_file": "asmc_error_q_history.npy",
        "time_file": None,
    },
]

LINE_STYLES = ["--", "-.", "-", ":", (0, (3, 1, 1, 1))]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# =========================
# Plot style
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"

legend_font = FontProperties(family="Times New Roman", size=12)


# =========================
# Utility functions
# =========================
def load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_method_data(method: dict) -> dict:
    folder = DATA_ROOT / method["folder"]

    tau = load_array(folder / method["tau_file"], f"{method['label']} tau history")
    err = load_array(folder / method["err_file"], f"{method['label']} error_q history")

    n = min(len(tau), len(err))
    tau = tau[:n]
    err = err[:n]

    time_file = method.get("time_file")
    if time_file is not None and (folder / time_file).exists():
        t = np.load(folder / time_file, allow_pickle=True)
        t = np.asarray(t).reshape(-1)[:n]
    else:
        t = np.arange(n) * DT

    n = min(len(t), len(tau), len(err))
    t = t[:n]
    tau = tau[:n]
    err = err[:n]

    # Ensure only data within T_MAX is used.
    mask = t <= T_MAX
    if np.any(mask):
        t = t[mask]
        tau = tau[mask]
        err = err[mask]

    return {
        "key": method["key"],
        "label": method["label"],
        "t": t,
        "tau": tau,
        "err": err,
    }


def common_time_length(data_dict: dict) -> int:
    return min(len(v["t"]) for v in data_dict.values())


def trim_to_common_length(data_dict: dict) -> dict:
    n_common = common_time_length(data_dict)
    for item in data_dict.values():
        item["t"] = item["t"][:n_common]
        item["tau"] = item["tau"][:n_common]
        item["err"] = item["err"][:n_common]
    return data_dict


def auto_ylim_for_inset(series_list, margin_ratio=0.15):
    values = np.concatenate([np.asarray(s).reshape(-1) for s in series_list])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if np.isclose(ymin, ymax):
        delta = max(abs(ymin), 1.0) * 0.1
        return ymin - delta, ymax + delta
    margin = (ymax - ymin) * margin_ratio
    return ymin - margin, ymax + margin


def compute_metrics_1d(error_1d: np.ndarray, dt: float) -> dict:
    error_1d = np.asarray(error_1d).reshape(-1)
    return {
        "ISE": np.sum(error_1d ** 2) * dt,
        "IAE": np.sum(np.abs(error_1d)) * dt,
        "MAE": np.mean(np.abs(error_1d)),
        "RMSE": np.sqrt(np.mean(error_1d ** 2)),
        "MaxAbs": np.max(np.abs(error_1d)),
    }


def compute_torque_metrics_1d(tau_1d: np.ndarray, dt: float) -> dict:
    tau_1d = np.asarray(tau_1d).reshape(-1)
    d_tau = np.diff(tau_1d)
    return {
        "RMS_Torque": np.sqrt(np.mean(tau_1d ** 2)),
        "MaxAbs_Torque": np.max(np.abs(tau_1d)),
        "MeanAbs_Torque": np.mean(np.abs(tau_1d)),
        "Torque_TV": np.sum(np.abs(d_tau)),
        "Torque_Rate_RMS": np.sqrt(np.mean((d_tau / dt) ** 2)) if len(d_tau) > 0 else 0.0,
    }


def plot_joint_compare(data_dict, data_key, joint_idx, joint_label, ylabel, filename, inset_key):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    for i, method in enumerate(METHODS):
        item = data_dict[method["key"]]
        t = item["t"]
        y = item[data_key][:, joint_idx]
        ax.plot(
            t,
            y,
            label=method["label"],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.5,
            color=COLORS[i % len(COLORS)],
        )

    ax.set_xlabel("Time (s)", fontsize=15, fontname="Times New Roman")
    ax.set_ylabel(ylabel, fontsize=15, fontname="Times New Roman")
    ax.set_xlim([0, T_MAX])
    ax.tick_params(labelsize=15)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right", prop=legend_font)

    # Inset
    cfg = INSET_CONFIG[inset_key]
    axins = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="center",
        bbox_to_anchor=cfg["bbox"],
        bbox_transform=ax.transAxes,
    )

    x1, x2 = cfg["xlim"]
    inset_series = []
    for i, method in enumerate(METHODS):
        item = data_dict[method["key"]]
        t = item["t"]
        y = item[data_key][:, joint_idx]
        axins.plot(
            t,
            y,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.0,
            color=COLORS[i % len(COLORS)],
        )
        mask = (t >= x1) & (t <= x2)
        if np.any(mask):
            inset_series.append(y[mask])

    axins.set_xlim(x1, x2)
    if cfg["ylim"] is None:
        ylim = auto_ylim_for_inset(inset_series)
        if ylim is not None:
            axins.set_ylim(*ylim)
    else:
        axins.set_ylim(*cfg["ylim"])

    axins.grid(True, linestyle=":", linewidth=0.5)
    axins.tick_params(labelsize=10)

    if data_key == "err":
        axins.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axins.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
        offset = axins.yaxis.get_offset_text()
        offset.set_fontname("Times New Roman")
        offset.set_fontsize(10)

    mark_inset(
        ax,
        axins,
        loc1=cfg["loc1"],
        loc2=cfg["loc2"],
        fc="none",
        ec="0.5",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.tight_layout()
    out_path = SAVE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


# =========================
# Main workflow
# =========================
def main():
    print("Loading five-method floating-base comparison data...")
    data = {}
    for method in METHODS:
        data[method["key"]] = load_method_data(method)

    data = trim_to_common_length(data)

    print("\nLoaded data summary:")
    for method in METHODS:
        item = data[method["key"]]
        print(
            f"{method['label']:>15s}: "
            f"steps={len(item['t'])}, "
            f"tau_shape={item['tau'].shape}, "
            f"err_shape={item['err'].shape}, "
            f"time=[{item['t'][0]:.4f}, {item['t'][-1]:.4f}] s"
        )

    # Plots: Joint 3 and Joint 5 torques and tracking errors.
    for joint_label, joint_idx in JOINTS_TO_PLOT.items():
        plot_joint_compare(
            data,
            data_key="tau",
            joint_idx=joint_idx,
            joint_label=joint_label,
            ylabel=rf"$\tau_{joint_label}$ (N·m)",
            filename=f"realtime_joint{joint_label}_torque_compare_5methods.png",
            inset_key=f"tau_{joint_label}",
        )

        plot_joint_compare(
            data,
            data_key="err",
            joint_idx=joint_idx,
            joint_label=joint_label,
            ylabel=rf"$e_{joint_label}$ (rad)",
            filename=f"joint{joint_label}_error_compare_5methods.png",
            inset_key=f"err_{joint_label}",
        )

    # Metrics for Joint 3 and Joint 5, matching the original compare_dyn.py style.
    rows_joint35 = []
    for method in METHODS:
        item = data[method["key"]]
        row = {"Controller": method["label"]}
        for joint_label, joint_idx in JOINTS_TO_PLOT.items():
            metrics = compute_metrics_1d(item["err"][:, joint_idx], DT)
            row[f"ISE (Joint {joint_label})"] = metrics["ISE"]
            row[f"IAE (Joint {joint_label})"] = metrics["IAE"]
            row[f"MAE (Joint {joint_label})"] = metrics["MAE"]
            row[f"RMSE (Joint {joint_label})"] = metrics["RMSE"]
            row[f"MaxAbs (Joint {joint_label})"] = metrics["MaxAbs"]
        rows_joint35.append(row)

    df_joint35 = pd.DataFrame(rows_joint35)
    print("\nJoint 3 and Joint 5 tracking-error metrics:")
    print(df_joint35.to_string(index=False))
    df_joint35.to_csv(SAVE_DIR / "joint35_error_metrics.csv", index=False)

    # Metrics for all joints.
    rows_all = []
    for method in METHODS:
        item = data[method["key"]]
        n_joints = item["err"].shape[1]
        for j in range(n_joints):
            metrics = compute_metrics_1d(item["err"][:, j], DT)
            rows_all.append({
                "Controller": method["label"],
                "Joint": j + 1,
                **metrics,
            })

    df_all = pd.DataFrame(rows_all)
    df_all.to_csv(SAVE_DIR / "all_joint_error_metrics.csv", index=False)

    # Torque metrics for all joints.
    rows_tau = []
    for method in METHODS:
        item = data[method["key"]]
        n_joints = item["tau"].shape[1]
        for j in range(n_joints):
            metrics = compute_torque_metrics_1d(item["tau"][:, j], DT)
            rows_tau.append({
                "Controller": method["label"],
                "Joint": j + 1,
                **metrics,
            })

    df_tau = pd.DataFrame(rows_tau)
    df_tau.to_csv(SAVE_DIR / "all_joint_torque_metrics.csv", index=False)

    # Optional compact summary by controller.
    summary_rows = []
    for method in METHODS:
        item = data[method["key"]]
        err_norm = np.linalg.norm(item["err"], axis=1)
        tau_norm = np.linalg.norm(item["tau"], axis=1)

        summary_rows.append({
            "Controller": method["label"],
            "ISE_joint_norm": np.sum(err_norm ** 2) * DT,
            "IAE_joint_norm": np.sum(np.abs(err_norm)) * DT,
            "MAE_joint_norm": np.mean(np.abs(err_norm)),
            "RMSE_joint_norm": np.sqrt(np.mean(err_norm ** 2)),
            "Max_joint_norm": np.max(np.abs(err_norm)),
            "RMS_tau_norm": np.sqrt(np.mean(tau_norm ** 2)),
            "Max_tau_norm": np.max(np.abs(tau_norm)),
            "Mean_tau_norm": np.mean(np.abs(tau_norm)),
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\nCompact all-joint summary:")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(SAVE_DIR / "summary_metrics.csv", index=False)

    print(f"\nAll figures and CSV files have been saved to: {SAVE_DIR.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
