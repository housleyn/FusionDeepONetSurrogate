import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

df = pd.read_csv('Data/ellipse_data/ellipse_data_test2.csv')

x = df["X (m)"].values
y = df["Y (m)"].values
z = df["Z (m)"].values
density = df["Density (kg/m^3)"].values
velocity_i = df["Velocity[i] (m/s)"].values
velocity_j = df["Velocity[j] (m/s)"].values
velocity_k = df["Velocity[k] (m/s)"].values
absolute_pressure = df["Absolute Pressure (Pa)"].values
temperature = df["Temperature (K)"].values
distance = df["distanceToEllipse"].values

tolerance = 0.0010609098358303332

surface = distance <= tolerance 

x_surface = x[surface]
y_surface = y[surface]

density_surface = density[surface]
velocity_i_surface = velocity_i[surface]
velocity_j_surface = velocity_j[surface]
velocity_k_surface = velocity_k[surface]
absolute_pressure_surface = absolute_pressure[surface]
temperature_surface = temperature[surface]


def calculate_pressure_force_components(nx,ny,ds,p):
    Fx = -p * (nx) * ds
    Fy = -p * (ny) * ds
    return Fx, Fy


def calculate_surface_normals_tangents(x, y, k=6):
    coords = np.column_stack((x, y))
    tree = KDTree(coords)
    tangents = np.zeros_like(coords)

    for i, (xi, yi) in enumerate(coords):
        dists, idxs = tree.query([xi, yi], k=k)
        A = np.c_[x[idxs] - xi, y[idxs] - yi]
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        tangent = Vt[0]
        tangent /= np.linalg.norm(tangent)
        tangents[i] = tangent

    for i in range(1, len(tangents)):
        if np.dot(tangents[i], tangents[i - 1]) < 0:
            tangents[i] *= -1

    
    tx = tangents[:, 0]
    ty = tangents[:, 1]
    nx = -ty.copy()
    ny = tx.copy()

    centroid = np.array([x.mean(), y.mean()])
    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        normal = np.array([nx[i], ny[i]])
        to_centroid = centroid - point
        if np.dot(normal, to_centroid) < 0:
            nx[i] *= -1
            ny[i] *= -1

    return tx, ty, nx, ny


tx, ty, nx, ny = calculate_surface_normals_tangents(x_surface, y_surface)


ds = np.zeros_like(x_surface)
for i in range(1, len(x_surface) - 1):
    dx = x_surface[i+1] - x_surface[i-1]
    dy = y_surface[i+1] - y_surface[i-1]
    ds[i] = 0.5 * np.sqrt(dx**2 + dy**2)


ds[0] = np.sqrt((x_surface[1] - x_surface[0])**2 + (y_surface[1] - y_surface[0])**2)
ds[-1] = np.sqrt((x_surface[-1] - x_surface[-2])**2 + (y_surface[-1] - y_surface[-2])**2)

Fx, Fy = calculate_pressure_force_components(nx, ny, ds, absolute_pressure_surface)

total_drag = Fx.sum()
total_lift = Fy.sum()

print(f"Total Pressure Drag Force: {total_drag:.4e} N")
print(f"Total Pressure Lift Force: {total_lift:.4e} N")

plt.figure(figsize=(10, 6))
plt.scatter(x_surface, y_surface, c=absolute_pressure_surface, cmap='viridis', s=10)
plt.colorbar(label='Absolute Pressure (Pa)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
# Scale factor for arrow length
scale = 5e-2  

# Plot tangents (in red)
plt.quiver(x_surface, y_surface, tx, ty, color='red', scale=1/scale, label='Tangents')

# Plot normals (in blue)
plt.quiver(x_surface, y_surface, nx, ny, color='blue', scale=1/scale, label='Normals')

plt.legend()
plt.title("Surface Points with Tangents (red) and Normals (blue)")
plt.savefig('ellipse_surface_with_normals_tangents.png')
# plt.show()