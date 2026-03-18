import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as mcolors
import os 
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

def plot_surface_temperature_profile(
    self,
    save_path=None,
    area_threshold=1e100,
    temperature_col="Temperature (K)",
):
    """
    Plot predicted vs true surface temperature profile using the same
    surface-cell extraction logic as compute_surface_percent_differences().

    The surface cells are ordered by polar angle around the body centroid
    in the X-Y plane, which works well for a closed 2D body profile.

    Parameters
    ----------
    save_path : str or None
        Output path for the plot. If None, saves into figures_dir/surface_figures.
    area_threshold : float
        Upper threshold for valid area magnitudes.
    temperature_col : str
        Column name for temperature in df_true/df_pred.
    """

    # --- Basic checks ---
    if self.df_true is None:
        raise ValueError("df_true is None; surface profile requires true data.")
    if self.df_pred is None:
        raise ValueError("df_pred is None; surface profile requires predicted data.")

    required_true_cols = [
        "X (m)", "Y (m)", "Z (m)",
        "is_on_surface",
        "Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)",
        temperature_col,
    ]
    for col in required_true_cols:
        if col not in self.df_true.columns:
            raise ValueError(f"True data missing required column: {col}")

    if temperature_col not in self.df_pred.columns:
        raise ValueError(f"Predicted data missing required column: {temperature_col}")

    # --- Surface mask from normals (same logic as compute_surface_percent_differences) ---
    area_vec_all = self.df_true[
        ["Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)"]
    ].to_numpy()
    area_norm_all = np.linalg.norm(area_vec_all, axis=1)

    is_surface = (pd.to_numeric(self.df_true["is_on_surface"], errors="coerce")
                    .fillna(0).astype(int).to_numpy() == 1)

    valid_area = (
        np.isfinite(area_norm_all)
        & (area_norm_all > 0.0)
        & (area_norm_all < area_threshold)
    )
    surface_mask = is_surface & valid_area

    if not np.any(surface_mask):
        raise ValueError("No valid surface cells found for temperature profile.")

    # --- Extract surface coordinates and values ---
    x_s = self.df_true["X (m)"].to_numpy()[surface_mask]
    y_s = self.df_true["Y (m)"].to_numpy()[surface_mask]

    T_true_s = self.df_true[temperature_col].to_numpy()[surface_mask]
    T_pred_s = self.df_pred[temperature_col].to_numpy()[surface_mask]

    finite_mask = (
        np.isfinite(x_s) & np.isfinite(y_s)
        & np.isfinite(T_true_s) & np.isfinite(T_pred_s)
    )

    x_s = x_s[finite_mask]
    y_s = y_s[finite_mask]
    T_true_s = T_true_s[finite_mask]
    T_pred_s = T_pred_s[finite_mask]

    if len(x_s) == 0:
        raise ValueError("No finite surface cells found for temperature profile.")

    # --- Order points around body centroid ---
    x_c = np.mean(x_s)
    y_c = np.mean(y_s)
    theta = np.arctan2(y_s - y_c, x_s - x_c)
    order = np.argsort(theta)

    x_ord = x_s[order]
    y_ord = y_s[order]
    T_true_ord = T_true_s[order]
    T_pred_ord = T_pred_s[order]

    # --- Build cumulative arc length along ordered surface ---
    ds = np.sqrt(np.diff(x_ord)**2 + np.diff(y_ord)**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # Optional closure if first/last point are duplicated or nearly duplicated
    if len(s) > 1 and np.isclose(x_ord[0], x_ord[-1], atol=1e-12) and np.isclose(y_ord[0], y_ord[-1], atol=1e-12):
        x_ord = x_ord[:-1]
        y_ord = y_ord[:-1]
        T_true_ord = T_true_ord[:-1]
        T_pred_ord = T_pred_ord[:-1]
        ds = np.sqrt(np.diff(x_ord)**2 + np.diff(y_ord)**2)
        s = np.concatenate(([0.0], np.cumsum(ds)))

    # --- Save path ---
    if save_path is None:
        out_dir = os.path.join(self.figures_dir, "surface_figures")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "surface_temperature_profile.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, T_true_ord, label="True Temperature", linewidth=2)
    ax.plot(s, T_pred_ord, label="Predicted Temperature", linewidth=2, linestyle="--")

    ax.set_xlabel("Surface Arc Length")
    ax.set_ylabel(temperature_col)
    ax.set_title("Surface Temperature Profile")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface temperature profile to {save_path}")