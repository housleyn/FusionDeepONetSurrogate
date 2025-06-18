import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# Load and clean data
df_true = pd.read_csv("ellipse_data/ellipse_data_test3.csv").drop(columns=["a", "b"])
df_pred = pd.read_csv("predicted_test3.csv").drop(columns=["Sphere Radius"])
df_pred = df_pred[["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
                   "Absolute Pressure (Pa)", "Density (kg/m^3)", "X (m)", "Y (m)", "Z (m)"]]

# Calculate error
error = (df_true - df_pred)
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

    xi = np.linspace(x.min(), x.max(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate true, predicted, and error values
    zi_true = griddata((x, y), df_true[field].values, (xi, yi), method='cubic')
    zi_pred = griddata((x, y), df_pred[field].values, (xi, yi), method='cubic')
    zi_error = griddata((x, y), error[field].values, (xi, yi), method='cubic')
    zi_error = np.abs(zi_error)

        # Define semi-ellipse mask
    a = 1.44  # replace with actual
    b = 1.16  # replace with actual
    x0, y0 = -2.5, 0
    mask = ((xi - x0)**2 / a**2 + (yi - y0)**2 / b**2) <= 1

    # Mask the data
    zi_true[mask] = np.nan
    zi_pred[mask] = np.nan
    zi_error[mask] = np.nan

    

    # Plot all three as subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Predicted", "True", "Error"]
    datasets = [zi_pred, zi_true, zi_error]
    cmaps = ["inferno", "inferno", "inferno"]
    # Set shared vmin, vmax for true/pred
    vmin_tp = np.nanmin([zi_true, zi_pred])
    vmax_tp = np.nanmax([zi_true, zi_pred])
    ticks_tp = np.linspace(vmin_tp, vmax_tp, 9)
    tick_labels_tp = [f"{t:.3f}" for t in ticks_tp]  # same format

    # Separate error settings
    vmin_err = np.nanmin(zi_error)
    vmax_err = np.nanmax(zi_error)
    ticks_err = np.linspace(vmin_err, vmax_err, 9)
    tick_labels_err = [f"{t:.3f}" for t in ticks_err]

    for ax, data, title, cmap in zip(axs, datasets, titles, cmaps):
        if title == "Error":
            contour = ax.contourf(xi, yi, data, levels=100, cmap=cmap, vmin=vmin_err, vmax=vmax_err)
            cbar = fig.colorbar(contour, ax=ax, ticks=ticks_err)
            cbar.ax.set_yticklabels(tick_labels_err)
            cbar.set_label("Error Color Scale")
            cbar.ax.tick_params(labelsize=14)
        else:
            contour = ax.contourf(xi, yi, data, levels=100, cmap=cmap, vmin=vmin_tp, vmax=vmax_tp)
            cbar = fig.colorbar(contour, ax=ax, ticks=ticks_tp)
            cbar.ax.set_yticklabels(tick_labels_tp)
            cbar.set_label(f"{field} Color Scale")
            cbar.ax.tick_params(labelsize=14)

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
