import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

true_path = "Data/orion_data/orion_data_AoA0.18726272_Mach28.55732221.csv"
pred_path = "Outputs/orion_sequential/predicted_output.csv"

true_df = pd.read_csv(true_path)
pred_df = pd.read_csv(pred_path)

vi_true = true_df["Velocity[i] (m/s)"].to_numpy()
vj_true = true_df["Velocity[j] (m/s)"].to_numpy()
vi_pred = pred_df["Velocity[i] (m/s)"].to_numpy()
vj_pred = pred_df["Velocity[j] (m/s)"].to_numpy()

abs_vi_true = np.abs(vi_true)
abs_vj_true = np.abs(vj_true)
abs_err_vi = np.abs(vi_pred - vi_true)
abs_err_vj = np.abs(vj_pred - vj_true)

plt.figure(figsize=(10, 6))
bins_mag = np.linspace(0, max(abs_vi_true.max(), abs_vj_true.max()), 120)
plt.hist(abs_vi_true, bins=bins_mag, alpha=0.6, label="|Velocity[i]| true", edgecolor="black")
plt.hist(abs_vj_true, bins=bins_mag, alpha=0.6, label="|Velocity[j]| true", edgecolor="black")
plt.yscale("log")
plt.xlabel("Velocity Magnitude (m/s)")
plt.ylabel("Cell Count")
plt.title("True Velocity Magnitude Distribution")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("velocity_magnitude_histogram.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
bins_err = np.linspace(0, max(abs_err_vi.max(), abs_err_vj.max()), 120)
plt.hist(abs_err_vi, bins=bins_err, alpha=0.6, label="|Velocity[i] error|", edgecolor="black")
plt.hist(abs_err_vj, bins=bins_err, alpha=0.6, label="|Velocity[j] error|", edgecolor="black")
plt.yscale("log")
plt.xlabel("Absolute Error (m/s)")
plt.ylabel("Cell Count")
plt.title("Velocity Error Distribution")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("velocity_error_histogram.png", dpi=200)
plt.close()