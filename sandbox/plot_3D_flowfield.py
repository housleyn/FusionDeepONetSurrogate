from pathlib import Path

import polars as pl
import numpy as np
import pyvista as pv

from reconstruct_surface import reconstruct_body_surface


flow_file = r"Outputs\waverider_low_fi_fusion_HF1000_HF1000_B8_H128\predicted_output_3d.csv"
surface_file = r"Data\processed_full_3d_all_waveriders\Mach_6.380430_shape_1_full_field.csv"
body_mesh_file = r"reconstructed_body.vtp"

cols = ["X (m)", "Y (m)", "Z (m)", "Absolute Pressure (Pa)"]

sample_frac = 0.05
nx, ny, nz = 200, 200, 200
interp_radius = .25
interp_sharpness = 3.0


# ============================================================
# LOAD OR BUILD BODY
# ============================================================
if Path(body_mesh_file).exists():
    body = pv.read(body_mesh_file)
else:
    body = reconstruct_body_surface(
        surface_file,
        save_path=body_mesh_file
    )


# ============================================================
# LOAD FLOW DATA
# ============================================================
lf = pl.scan_csv(flow_file).select(cols)
df = lf.collect()
df = df.sample(fraction=sample_frac)

points = df.select(["X (m)", "Y (m)", "Z (m)"]).to_numpy()
pressure = df["Absolute Pressure (Pa)"].to_numpy()

cloud = pv.PolyData(points)
cloud["pressure"] = pressure


# ============================================================
# INTERPOLATE TO GRID
# ============================================================
xmin, ymin, zmin = points.min(axis=0)
xmax, ymax, zmax = points.max(axis=0)

grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.origin = (xmin, ymin, zmin)
grid.spacing = (
    (xmax - xmin) / (nx - 1),
    (ymax - ymin) / (ny - 1),
    (zmax - zmin) / (nz - 1),
)

grid = grid.interpolate(
    cloud,
    radius=interp_radius,
    sharpness=interp_sharpness
)


# ============================================================
# ISOSURFACES
# ============================================================
iso_values = np.quantile(pressure, .5)

isosurfaces = grid.contour(
    isosurfaces=iso_values,
    scalars="pressure"
)

isosurfaces = isosurfaces.clean()
isosurfaces = isosurfaces.extract_largest()

isosurfaces = isosurfaces.smooth(
    n_iter=30,
    relaxation_factor=0.1
)

# mirror across z = 0
isosurfaces_mirror = isosurfaces.copy()
isosurfaces_mirror.points[:, 2] *= -1

isosurfaces_mirror = isosurfaces_mirror.compute_normals(
    auto_orient_normals=True,
    consistent_normals=True,
    inplace=False
)

isosurfaces_full = isosurfaces.merge(isosurfaces_mirror, merge_points=True)
isosurfaces_full = isosurfaces_full.clean()
isosurfaces_full = isosurfaces_full.triangulate()


# ============================================================
# OPTIONAL (keep slices commented)
# ============================================================
slices = grid.slice_orthogonal()


# ============================================================
# PLOT
# ============================================================
plotter = pv.Plotter()

plotter.add_mesh(
    body,
    color="white",
    opacity=1.0,
    smooth_shading=True,
    show_edges=False
)

# plotter.add_mesh(
#     slices,
#     scalars="pressure",
#     cmap="turbo",
#     opacity=1.0,
#     smooth_shading=True
# )

plotter.add_mesh(
    isosurfaces_full,
    scalars="pressure",
    cmap="turbo",
    opacity=0.6,
    smooth_shading=True
)

plotter.show()