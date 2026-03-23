import numpy as np
import pandas as pd


def compute_circle_surface_area_vectors(center, surface_points, radius=1.0):
    """
    Compute 2D line-element area vectors for circle surface points.

    For a circle in 2D:
        A_vec = n * ds
    where n is the outward unit normal and ds is the arc length assigned
    to each surface point.

    Parameters
    ----------
    center : tuple[float, float]
        Circle center (cx, cy).
    surface_points : ndarray, shape (N, 2)
        Surface point coordinates on the circle.
    radius : float
        Circle radius.

    Returns
    -------
    area_vec : ndarray, shape (N, 3)
        Area vectors [Ax, Ay, Az].
    """
    cx, cy = center
    x = surface_points[:, 0]
    y = surface_points[:, 1]

    rx = x - cx
    ry = y - cy
    rmag = np.sqrt(rx**2 + ry**2) + 1e-15

    nx = rx / rmag
    ny = ry / rmag

    ds = 2.0 * np.pi * radius / len(surface_points)

    area_vec = np.zeros((len(surface_points), 3), dtype=float)
    area_vec[:, 0] = nx * ds
    area_vec[:, 1] = ny * ds
    area_vec[:, 2] = 0.0

    return area_vec


def assign_surface_area_vectors(
    df,
    n_surface_each,
    center_main=(0.0, 0.0),
    center_secondary=None,
    radius=1.0,
):
    """
    Replace placeholder area columns with real surface area vectors.

    Assumes the first:
        - n_surface_each rows belong to main sphere surface
        - next n_surface_each rows belong to secondary sphere surface
    which matches your Explorer generator ordering.

    Non-surface rows are assigned zero area vectors.

    Parameters
    ----------
    df : pandas.DataFrame
        Explorer-generated input dataframe.
    n_surface_each : int
        Number of explicit surface points per sphere.
    center_main : tuple
        Main sphere center.
    center_secondary : tuple
        Secondary sphere center.
    radius : float
        Circle radius.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of df with updated Area[i/j/k] columns.
    """
    if center_secondary is None:
        raise ValueError("center_secondary must be provided.")

    df_out = df.copy()

    n_surface_total = 2 * n_surface_each
    if len(df_out) < n_surface_total:
        raise ValueError("Dataframe has fewer rows than expected surface points.")

    # initialize all to zero
    df_out["Area[i] (m^2)"] = 0.0
    df_out["Area[j] (m^2)"] = 0.0
    df_out["Area[k] (m^2)"] = 0.0

    pts = df_out[["X (m)", "Y (m)"]].to_numpy()

    main_surface_pts = pts[:n_surface_each]
    sec_surface_pts = pts[n_surface_each:n_surface_total]

    A_main = compute_circle_surface_area_vectors(
        center=center_main,
        surface_points=main_surface_pts,
        radius=radius,
    )
    A_sec = compute_circle_surface_area_vectors(
        center=center_secondary,
        surface_points=sec_surface_pts,
        radius=radius,
    )

    A_all = np.vstack([A_main, A_sec])

    df_out.loc[:n_surface_total - 1, "Area[i] (m^2)"] = A_all[:, 0]
    df_out.loc[:n_surface_total - 1, "Area[j] (m^2)"] = A_all[:, 1]
    df_out.loc[:n_surface_total - 1, "Area[k] (m^2)"] = A_all[:, 2]

    # enforce surface flag consistency for the explicit surface block
    df_out.loc[:n_surface_total - 1, "is_on_surface"] = 1

    return df_out


def compute_pressure_force_from_dataframe(
    df,
    pressure_col="Absolute Pressure (Pa)",
    area_cols=("Area[i] (m^2)", "Area[j] (m^2)", "Area[k] (m^2)"),
    surface_col="is_on_surface",
    area_threshold=1e100,
):
    """
    Compute net pressure-force proxy from dataframe.

    Uses:
        F = - sum( p * A_vec )

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing pressure, surface flags, and area vectors.
    pressure_col : str
        Pressure column name.
    area_cols : tuple[str, str, str]
        Area vector column names.
    surface_col : str
        Surface marker column.
    area_threshold : float
        Upper threshold for valid area-vector norms.

    Returns
    -------
    F : ndarray, shape (3,)
        Net pressure force vector.
    surface_mask : ndarray, shape (N,)
        Boolean mask used for surface integration.
    """
    required_cols = [pressure_col, surface_col, *area_cols]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    A_vec = df[list(area_cols)].to_numpy(dtype=float)
    A_norm = np.linalg.norm(A_vec, axis=1)

    is_surface = df[surface_col].to_numpy().astype(int) == 1
    valid_area = (
        np.isfinite(A_norm) &
        (A_norm > 0.0) &
        (A_norm < area_threshold)
    )
    surface_mask = is_surface & valid_area

    if not np.any(surface_mask):
        raise ValueError("No valid surface points found for pressure integration.")

    p_s = df[pressure_col].to_numpy(dtype=float)[surface_mask]
    A_s = A_vec[surface_mask]

    F = -np.sum(p_s[:, None] * A_s, axis=0)
    return F, surface_mask


def compute_cd_from_dataframe(
    df,
    pressure_col="Absolute Pressure (Pa)",
    area_threshold=1e100,
):
    """
    Compute drag proxy Cd from predicted pressure and area vectors.

    For your current no-AoA case:
        e_drag = [1, 0, 0]

    Parameters
    ----------
    df : pandas.DataFrame
        Predicted dataframe with pressure and area vectors.
    pressure_col : str
        Pressure column to integrate.
    area_threshold : float
        Upper threshold for valid area-vector norms.

    Returns
    -------
    cd_proxy : float
        Drag proxy from pressure integration.
    force_vector : ndarray
        Net integrated force vector.
    surface_mask : ndarray
        Surface mask used.
    """
    F, surface_mask = compute_pressure_force_from_dataframe(
        df=df,
        pressure_col=pressure_col,
        area_threshold=area_threshold,
    )

    e_drag = np.array([1.0, 0.0, 0.0], dtype=float)
    cd_proxy = float(np.dot(F, e_drag))

    return cd_proxy, F, surface_mask


def merge_input_geometry_into_prediction_df(pred_df, input_df):
    """
    Merge geometry-only columns from input_df into pred_df so Cd can be computed
    from predictions on the same coordinates.

    Parameters
    ----------
    pred_df : pandas.DataFrame
        Model prediction dataframe.
    input_df : pandas.DataFrame
        Explorer-generated input dataframe containing geometry columns.

    Returns
    -------
    merged_df : pandas.DataFrame
        Prediction dataframe augmented with surface and area columns.
    """
    needed_cols = [
        "is_on_surface",
        "Area[i] (m^2)",
        "Area[j] (m^2)",
        "Area[k] (m^2)",
    ]

    for col in needed_cols:
        if col not in input_df.columns:
            raise ValueError(f"Input dataframe missing required column: {col}")

    merged_df = pred_df.copy()
    for col in needed_cols:
        merged_df[col] = input_df[col].values

    return merged_df