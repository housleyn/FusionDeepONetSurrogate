import polars as pl
import numpy as np
import open3d as o3d
import pyvista as pv


def reconstruct_body_surface(
    surface_file: str,
    save_path: str | None = None,
    area_threshold: float = 1e5,
    poisson_depth: int = 9,
    density_keep_frac: float = 0.98,
    half_smooth_iter: int = 70,
    half_smooth_relax: float = 0.1,
    full_smooth_iter: int = 100,
    full_smooth_relax: float = 0.08,
) -> pv.PolyData:
    cols = [
        "is_on_surface",
        "Area[i] (m^2)",
        "Area[j] (m^2)",
        "Area[k] (m^2)",
        "X (m)",
        "Y (m)",
        "Z (m)",
    ]

    df = pl.scan_csv(surface_file).select(cols).collect()

    df = df.filter(pl.col("is_on_surface") == 1)
    df = df.filter(
        (pl.col("Area[i] (m^2)").abs() < area_threshold) &
        (pl.col("Area[j] (m^2)").abs() < area_threshold) &
        (pl.col("Area[k] (m^2)").abs() < area_threshold)
    )

    points = df.select(["X (m)", "Y (m)", "Z (m)"]).to_numpy()

    area_vectors = df.select([
        "Area[i] (m^2)",
        "Area[j] (m^2)",
        "Area[k] (m^2)"
    ]).to_numpy()

    areas = np.linalg.norm(area_vectors, axis=1)
    valid = areas > 0.0

    points = points[valid]
    area_vectors = area_vectors[valid]
    areas = areas[valid]

    normals = area_vectors / areas[:, None]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth
    )

    densities = np.asarray(densities)
    density_cutoff = np.quantile(densities, 1.0 - density_keep_frac)
    vertices_to_remove = densities < density_cutoff
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()

    surface = pv.PolyData(verts, faces_pv)

    surface = surface.clean()
    surface = surface.triangulate()
    surface = surface.clip(normal="z", origin=(0, 0, 0), invert=True)
    surface = surface.extract_largest()
    surface = surface.smooth(
        n_iter=half_smooth_iter,
        relaxation_factor=half_smooth_relax
    )

    mirrored = surface.copy()
    mirrored.points[:, 2] *= -1
    mirrored = mirrored.compute_normals(auto_orient_normals=True, inplace=False)

    full_surface = surface.merge(mirrored, merge_points=True)
    full_surface = full_surface.clean()
    full_surface = full_surface.triangulate()
    full_surface = full_surface.smooth(
        n_iter=full_smooth_iter,
        relaxation_factor=full_smooth_relax,
        boundary_smoothing=True,
        feature_smoothing=False
    )

    full_surface = full_surface.compute_normals(
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False
    )

    if save_path is not None:
        full_surface.save(save_path)

    return full_surface