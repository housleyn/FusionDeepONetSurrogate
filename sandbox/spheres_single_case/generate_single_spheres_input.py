
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path


FLOAT_MAX = np.finfo(np.float64).max


def unsigned_circle_distance(x, y, center, radius=1.0):
    cx, cy = center
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.abs(r - radius)


def starccm_two_sphere_distance(x, y, center_main, center_secondary, radius=1.0):
    d1 = unsigned_circle_distance(x, y, center_main, radius=radius)
    d2 = unsigned_circle_distance(x, y, center_secondary, radius=radius)
    return np.sqrt(d1**2 + d2**2)


def inside_any_circle(x, y, centers, radius=1.0, tol=1e-12):
    mask = np.zeros_like(x, dtype=bool)
    for cx, cy in centers:
        mask |= ((x - cx) ** 2 + (y - cy) ** 2) < (radius - tol) ** 2
    return mask


def sample_surface(center, n_surface, radius=1.0, angle_offset=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n_surface, endpoint=False) + angle_offset
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack([x, y])


def sample_ring(center, r_inner, r_outer, n_samples, rng):
    theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    # area-uniform annulus sampling
    rho = np.sqrt(rng.uniform(r_inner**2, r_outer**2, n_samples))
    x = center[0] + rho * np.cos(theta)
    y = center[1] + rho * np.sin(theta)
    return np.column_stack([x, y])


def sample_background(x_lim, y_lim, n_samples, rng):
    x = rng.uniform(x_lim[0], x_lim[1], n_samples)
    y = rng.uniform(y_lim[0], y_lim[1], n_samples)
    return np.column_stack([x, y])


def filter_valid_points(points, centers, x_lim, y_lim, radius=1.0):
    if len(points) == 0:
        return points
    x = points[:, 0]
    y = points[:, 1]
    in_domain = (
        (x >= x_lim[0]) & (x <= x_lim[1]) &
        (y >= y_lim[0]) & (y <= y_lim[1])
    )
    not_inside = ~inside_any_circle(x, y, centers, radius=radius)
    return points[in_domain & not_inside]


def build_single_case(
    x_secondary,
    y_secondary,
    output_csv,
    output_plot=None,
    mach=7.5,
    total_points=30000,
    n_surface_each=180,
    seed=42,
    x_lim=(-2.5, 17.5),
    y_lim=(-15.0, 15.0),
    radius=1.0,
):
    center_main = (0.0, 0.0)
    center_secondary = (float(x_secondary), float(y_secondary))
    centers = [center_main, center_secondary]

    # Basic geometry checks
    center_dist = np.sqrt((center_secondary[0] - center_main[0])**2 + (center_secondary[1] - center_main[1])**2)
    if center_dist <= 2.0 * radius:
        raise ValueError("Spheres overlap or touch; choose a larger separation.")
    if not (center_secondary[1] > 0.0):
        raise ValueError("Secondary sphere must be in the top half-plane (y > 0).")
    if not (x_lim[0] + radius < center_secondary[0] < x_lim[1] - radius):
        raise ValueError("Secondary sphere is too close to the left/right domain wall.")
    if not (y_lim[0] + radius < center_secondary[1] < y_lim[1] - radius):
        raise ValueError("Secondary sphere is too close to the top/bottom domain wall.")

    rng = np.random.default_rng(seed)

    # Fixed explicit surface points
    surf_main = sample_surface(center_main, n_surface_each, radius=radius, angle_offset=0.0)
    surf_secondary = sample_surface(center_secondary, n_surface_each, radius=radius, angle_offset=np.pi / n_surface_each)
    surface_points = np.vstack([surf_main, surf_secondary])

    # Remaining budget
    n_surface_total = surface_points.shape[0]
    if total_points <= n_surface_total:
        raise ValueError("total_points must exceed total number of surface points.")

    remaining = total_points - n_surface_total

    # Ring-heavy sampling around each sphere
    # Fractions are chosen to give strong near-body refinement with a lighter far field.
    ring_specs = [
        (1.02, 1.35, 0.24),  # very near wall
        (1.35, 2.20, 0.18),  # near field
        (2.20, 4.00, 0.12),  # medium field
    ]
    n_ring_total = int(remaining * 0.80)  # 80% ring-biased, 20% background
    n_background = remaining - n_ring_total

    ring_points_all = []
    assigned = 0
    for i, (r0, r1, frac) in enumerate(ring_specs):
        n_this = int(n_ring_total * frac)
        assigned += n_this
        per_sphere = n_this // 2
        for center in centers:
            pts = sample_ring(center, r0, r1, per_sphere, rng)
            ring_points_all.append(pts)

    # absorb any leftover ring points into the closest annulus
    leftover = n_ring_total - assigned
    if leftover > 0:
        extra_main = sample_ring(center_main, 1.02, 1.35, leftover // 2 + leftover % 2, rng)
        extra_sec = sample_ring(center_secondary, 1.02, 1.35, leftover // 2, rng)
        ring_points_all.extend([extra_main, extra_sec])

    ring_points = np.vstack(ring_points_all) if ring_points_all else np.empty((0, 2))
    ring_points = filter_valid_points(ring_points, centers, x_lim, y_lim, radius=radius)

    # Background points
    bg_try = sample_background(x_lim, y_lim, n_background * 3, rng)
    bg_try = filter_valid_points(bg_try, centers, x_lim, y_lim, radius=radius)
    if len(bg_try) < n_background:
        raise RuntimeError("Not enough valid background points after filtering.")
    background_points = bg_try[:n_background]

    # Combine and trim/fill to exact total
    non_surface = np.vstack([ring_points, background_points])

    # De-duplicate approximately while preserving order
    rounded = np.round(non_surface, 10)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    non_surface = non_surface[np.sort(unique_idx)]

    need_non_surface = total_points - n_surface_total
    if len(non_surface) < need_non_surface:
        deficit = need_non_surface - len(non_surface)
        extra_try = sample_background(x_lim, y_lim, deficit * 5, rng)
        extra_try = filter_valid_points(extra_try, centers, x_lim, y_lim, radius=radius)
        rounded_extra = np.round(extra_try, 10)
        _, unique_idx = np.unique(rounded_extra, axis=0, return_index=True)
        extra_try = extra_try[np.sort(unique_idx)]
        non_surface = np.vstack([non_surface, extra_try[:deficit]])

    non_surface = non_surface[:need_non_surface]
    points = np.vstack([surface_points, non_surface])

    is_surface = np.zeros(len(points), dtype=int)
    is_surface[:n_surface_total] = 1

    x = points[:, 0]
    y = points[:, 1]
    z = np.zeros_like(x)

    dist = starccm_two_sphere_distance(x, y, center_main, center_secondary, radius=radius)

    # Match original CSV titles exactly
    df = pd.DataFrame({
        "Velocity[i] (m/s)": np.nan,
        "Velocity[j] (m/s)": np.nan,
        "Velocity[k] (m/s)": np.nan,
        "Absolute Pressure (Pa)": np.nan,
        "Density (kg/m^3)": np.nan,
        "Temperature (K)": np.nan,
        "x_field": np.full(len(points), x_secondary, dtype=float),
        "y_field": np.full(len(points), y_secondary, dtype=float),
        "Mach_field": np.full(len(points), mach, dtype=float),
        "distanceToSurface": dist,
        "is_on_surface": is_surface,
        "Area[i] (m^2)": np.full(len(points), FLOAT_MAX, dtype=float),
        "Area[j] (m^2)": np.full(len(points), FLOAT_MAX, dtype=float),
        "Area[k] (m^2)": np.full(len(points), FLOAT_MAX, dtype=float),
        "X (m)": x,
        "Y (m)": y,
        "Z (m)": z,
    })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if output_plot is not None:
        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)

        triang = tri.Triangulation(x, y)
        surf_tri_mask = is_surface[triang.triangles].any(axis=1)
        triang.set_mask(surf_tri_mask)

        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.tricontourf(triang, dist, levels=100, cmap="viridis")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("distanceToSurface")
        ax.scatter(surface_points[:, 0], surface_points[:, 1], s=4, c="white", linewidths=0, alpha=0.8, label="Surface points")
        ax.set_title("Generated single-case input: tricontourf(distanceToSurface)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_plot, dpi=250, bbox_inches="tight")
        plt.close(fig)

    return df


if __name__ == "__main__":
    build_single_case(
        x_secondary=2.747362847147431,
        y_secondary=0.8419290887304742,
        output_csv="generated_single_case.csv",
        output_plot="generated_single_case_distance_plot.png",
        mach=7.5,
        total_points=30000,
        n_surface_each=180,
        seed=42,
    )
