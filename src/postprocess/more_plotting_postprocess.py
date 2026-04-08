import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def _get_ordered_surface_temperature_data(
    self,
    area_threshold=1e100,
    temperature_col="Temperature (K)",
):
    """
    Extract and order surface points and temperature values using the same logic
    as the surface temperature profile plot.

    Returns
    -------
    dict
        Contains ordered x, y, true/pred temperature, and cumulative arc length s.
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

    # --- Surface mask ---
    area_vec_all = self.df_true[
        ["Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)"]
    ].to_numpy()
    area_norm_all = np.linalg.norm(area_vec_all, axis=1)

    is_surface = (
        pd.to_numeric(self.df_true["is_on_surface"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy() == 1
    )

    valid_area = (
        np.isfinite(area_norm_all)
        & (area_norm_all > 0.0)
        & (area_norm_all < area_threshold)
    )
    surface_mask = is_surface & valid_area

    if not np.any(surface_mask):
        raise ValueError("No valid surface cells found.")

    # --- Extract ---
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
        raise ValueError("No finite surface cells found.")

    # --- Order by polar angle ---
    x_c = np.mean(x_s)
    y_c = np.mean(y_s)
    theta = np.arctan2(y_s - y_c, x_s - x_c)
    order = np.argsort(theta)

    x_ord = x_s[order]
    y_ord = y_s[order]
    T_true_ord = T_true_s[order]
    T_pred_ord = T_pred_s[order]

    # --- Remove duplicate closure point if present ---
    if (
        len(x_ord) > 1
        and np.isclose(x_ord[0], x_ord[-1], atol=1e-12)
        and np.isclose(y_ord[0], y_ord[-1], atol=1e-12)
    ):
        x_ord = x_ord[:-1]
        y_ord = y_ord[:-1]
        T_true_ord = T_true_ord[:-1]
        T_pred_ord = T_pred_ord[:-1]

    # --- Arc length ---
    ds = np.sqrt(np.diff(x_ord)**2 + np.diff(y_ord)**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    return {
        "x": x_ord,
        "y": y_ord,
        "T_true": T_true_ord,
        "T_pred": T_pred_ord,
        "s": s,
    }


def plot_surface_temperature_profile(
    self,
    save_path=None,
    area_threshold=1e100,
    temperature_col="Temperature (K)",
):
    """
    Plot predicted vs true surface temperature profile.
    """

    data = _get_ordered_surface_temperature_data(
        self,
        area_threshold=area_threshold,
        temperature_col=temperature_col,
    )

    x_ord = data["x"]
    y_ord = data["y"]
    T_true_ord = data["T_true"]
    T_pred_ord = data["T_pred"]
    s = data["s"]

    if save_path is None:
        out_dir = os.path.join(self.figures_dir, "surface_figures")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "surface_temperature_profile.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, T_true_ord, label="True Temperature", linewidth=2)
    ax.plot(s, T_pred_ord, label="Predicted Temperature", linewidth=2, linestyle="--")

    ax.set_xlabel("Surface Arc Length (m)")
    ax.set_ylabel(temperature_col)
    ax.set_title("Surface Temperature Profile")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface temperature profile to {save_path}")


def plot_surface_arclength_map(
    self,
    save_path=None,
    area_threshold=1e100,
    temperature_col="Temperature (K)",
    num_markers=11,
    annotate=True,
    label_offset_points=24,
    tick_length_data=0.08,
    x_pad_frac=.1,
    y_pad_frac=2.0,
):
    """
    Plot the body surface in physical space and mark selected cumulative arc-length
    locations with black tick marks and legible labels placed off the body.
    """

    data = _get_ordered_surface_temperature_data(
        self,
        area_threshold=area_threshold,
        temperature_col=temperature_col,
    )

    x = data["x"]
    y = data["y"]
    s = data["s"]

    if len(s) < 2:
        raise ValueError("Not enough surface points to create arc-length map.")

    s_total = s[-1]
    if s_total <= 0:
        raise ValueError("Total arc length is zero or invalid.")

    if save_path is None:
        out_dir = os.path.join(self.figures_dir, "surface_figures")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "surface_arclength_map.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Body outline only: smooth line, no dots ---
    ax.plot(x, y, color="0.35", linewidth=1.5, zorder=1)

    # --- Marker locations at selected arc-length fractions ---
    marker_s = np.linspace(0.0, s_total, num_markers, endpoint=False)**1.2
    marker_s = marker_s / marker_s.max() * s_total
    marker_x = np.interp(marker_s, s, x)
    marker_y = np.interp(marker_s, s, y)

    # centroid for outward label direction
    x_c = np.mean(x)
    y_c = np.mean(y)

    # black marker points
    ax.scatter(marker_x, marker_y, color="k", s=24, zorder=3)

    if annotate:
        for xm, ym, sm in zip(marker_x, marker_y, marker_s):
            dx = xm - x_c
            dy = ym - y_c
            norm = np.hypot(dx, dy)

            if norm < 1e-12:
                dx, dy = 1.0, 0.0
                norm = 1.0

            ux = dx / norm
            uy = dy / norm

            # outward tick in data units
            tick_x2 = xm + tick_length_data * ux
            tick_y2 = ym + tick_length_data * uy
            ax.plot([xm, tick_x2], [ym, tick_y2], color="k", linewidth=1.3, zorder=4)

            # label offset in points
            xoff = label_offset_points * ux
            yoff = label_offset_points * uy

            ha = "left" if ux >= 0 else "right"
            va = "bottom" if uy >= 0 else "top"

            ax.annotate(
                f"{int(round(sm))}",
                xy=(tick_x2, tick_y2),
                xytext=(xoff, yoff),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=11,
                color="k",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95),
                zorder=5,
                clip_on=False,
            )

    # --- Expand limits so labels are clearly outside the body and unobstructed ---
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_range = x_max - x_min
    y_range = y_max - y_min

    ax.set_xlim(x_min - x_pad_frac * x_range, x_max + x_pad_frac * x_range)
    ax.set_ylim(y_min - y_pad_frac * y_range, y_max + y_pad_frac * y_range)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved surface arc-length map to {save_path}")