from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import imageio.v2 as imageio


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


def _validate_geometry(center_main, center_secondary, x_lim, y_lim, radius):
    center_dist = np.sqrt(
        (center_secondary[0] - center_main[0])**2 +
        (center_secondary[1] - center_main[1])**2
    )

    if center_dist <= 2.0 * radius:
        raise ValueError("Spheres overlap or touch; choose a larger separation.")

    if center_secondary[1] <= 0.0:
        raise ValueError("Secondary sphere must be in the top half-plane (y > 0).")

    if not (x_lim[0] + radius < center_secondary[0] < x_lim[1] - radius):
        raise ValueError("Secondary sphere is too close to the left/right domain wall.")

    if not (y_lim[0] + radius < center_secondary[1] < y_lim[1] - radius):
        raise ValueError("Secondary sphere is too close to the top/bottom domain wall.")


def build_single_case(
    x_secondary,
    y_secondary,
    output_csv,
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

    _validate_geometry(center_main, center_secondary, x_lim, y_lim, radius)

    rng = np.random.default_rng(seed)

    surf_main = sample_surface(center_main, n_surface_each, radius=radius, angle_offset=0.0)
    surf_secondary = sample_surface(
        center_secondary,
        n_surface_each,
        radius=radius,
        angle_offset=np.pi / n_surface_each,
    )
    surface_points = np.vstack([surf_main, surf_secondary])

    n_surface_total = surface_points.shape[0]
    if total_points <= n_surface_total:
        raise ValueError("total_points must exceed total number of surface points.")

    remaining = total_points - n_surface_total

    ring_specs = [
        (1.02, 1.35, 0.24),
        (1.35, 2.20, 0.18),
        (2.20, 4.00, 0.12),
    ]

    n_ring_total = int(remaining * 0.80)
    n_background = remaining - n_ring_total

    ring_points_all = []
    assigned = 0

    for r0, r1, frac in ring_specs:
        n_this = int(n_ring_total * frac)
        assigned += n_this
        per_sphere = n_this // 2

        for center in centers:
            pts = sample_ring(center, r0, r1, per_sphere, rng)
            ring_points_all.append(pts)

    leftover = n_ring_total - assigned
    if leftover > 0:
        extra_main = sample_ring(center_main, 1.02, 1.35, leftover // 2 + leftover % 2, rng)
        extra_sec = sample_ring(center_secondary, 1.02, 1.35, leftover // 2, rng)
        ring_points_all.extend([extra_main, extra_sec])

    ring_points = np.vstack(ring_points_all) if ring_points_all else np.empty((0, 2))
    ring_points = filter_valid_points(ring_points, centers, x_lim, y_lim, radius=radius)

    bg_try = sample_background(x_lim, y_lim, n_background * 3, rng)
    bg_try = filter_valid_points(bg_try, centers, x_lim, y_lim, radius=radius)

    if len(bg_try) < n_background:
        raise RuntimeError("Not enough valid background points after filtering.")

    background_points = bg_try[:n_background]

    non_surface = np.vstack([ring_points, background_points])

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

    dist = starccm_two_sphere_distance(
        x, y,
        center_main=center_main,
        center_secondary=center_secondary,
        radius=radius,
    )

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

    return {
        "df": df,
        "points": points,
        "surface_points": surface_points,
        "is_surface": is_surface,
    }


def plot_distance_field(
    df,
    is_surface,
    surface_points,
    output_plot,
    x_lim=(-2.5, 17.5),
    y_lim=(-15.0, 15.0),
    levels=100,
):
    x = df["X (m)"].to_numpy()
    y = df["Y (m)"].to_numpy()
    dist = df["distanceToSurface"].to_numpy()

    triang = tri.Triangulation(x, y)
    surf_tri_mask = is_surface[triang.triangles].any(axis=1)
    triang.set_mask(surf_tri_mask)

    output_plot = Path(output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.tricontourf(triang, dist, levels=levels, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("distanceToSurface")

    ax.scatter(
        surface_points[:, 0],
        surface_points[:, 1],
        s=4,
        c="white",
        linewidths=0,
        alpha=0.8,
        label="Surface points",
    )

    ax.set_title("Generated single-case input: tricontourf(distanceToSurface)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_plot, dpi=250)
    plt.close(fig)

    return output_plot


def build_position_path(start, end, n_frames):
    x_vals = np.linspace(start[0], end[0], n_frames)
    y_vals = np.linspace(start[1], end[1], n_frames)
    return list(zip(x_vals, y_vals))


def save_gif_from_frames(frame_paths, gif_path, fps=10):
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    duration = 1.0 / fps
    imageio.mimsave(gif_path, images, duration=duration, loop=0)

    return gif_path