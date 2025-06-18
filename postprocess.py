import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# Load and clean data
df_true = pd.read_csv("ellipse_data/ellipse_data_test1.csv").drop(columns=["a", "b"])
df_pred = pd.read_csv("predicted_test1.csv").drop(columns=["a", "b"], errors="ignore")
df_pred = df_pred[[
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Absolute Pressure (Pa)",
    "Density (kg/m^3)",
    "X (m)",
    "Y (m)",
    "Z (m)",
]]

# Calculate error
error = np.abs(df_true - df_pred)
error["X (m)"] = df_true["X (m)"]
error["Y (m)"] = df_true["Y (m)"]
error.to_csv("error_test1.csv", index=False)

# Output folder
os.makedirs("figures", exist_ok=True)

# Fields to compare
fields = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
          "Absolute Pressure (Pa)", "Density (kg/m^3)"]

# Create interpolation and plotting per field
for field in fields:
    x = df_true["X (m)"].values
    y = df_true["Y (m)"].values

    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate true, predicted, and error values
    zi_true = griddata((x, y), df_true[field].values, (xi, yi), method='cubic')
    zi_pred = griddata((x, y), df_pred[field].values, (xi, yi), method='cubic')
    zi_error = griddata((x, y), error[field].values, (xi, yi), method='cubic')

    # Plot all three as subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["True", "Predicted", "Error"]
    datasets = [zi_true, zi_pred, zi_error]
    cmaps = ["viridis", "plasma", "inferno"]

    for ax, data, title, cmap in zip(axs, datasets, titles, cmaps):
        contour = ax.contourf(xi, yi, data, levels=100, cmap=cmap)
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"{field} - {title}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.tight_layout()
    safe_field = field.replace(' ', '_')\
                  .replace('[','')\
                  .replace(']','')\
                  .replace('(','')\
                  .replace(')','')\
                  .replace('/','_')
    plt.savefig(f"error_figures/{safe_field}_comparison.png")
    plt.close()

