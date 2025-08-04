import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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

epsilon = .1
interior_mask = (distance < epsilon)
exterior_mask = (distance >= epsilon)

from scipy.spatial import KDTree

def compute_gradient_unstructured(x, y, f, k=6):
    coords = np.column_stack((x, y))
    tree = KDTree(coords)
    grad = np.zeros_like(f)

    for i, (xi, yi) in enumerate(coords):
        dists, idxs = tree.query([xi, yi], k=k)
        A = np.c_[x[idxs] - xi, y[idxs] - yi]
        b = f[idxs] - f[i]
        g, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        grad[i] = np.linalg.norm(g)
    return grad

grad_p = compute_gradient_unstructured(x, y, absolute_pressure)
grad_p_corrected=grad_p.copy()
grad_p_corrected[interior_mask] = 0.0


threshold = 10000000
high_grad_mask = grad_p_corrected > threshold 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3})

sc1 = ax1.scatter(x, y, c=grad_p_corrected, s=8, cmap='inferno', vmin=0, vmax=9.56e7)
ax1.set_title("Gradient Magnitude Scatter Plot")
ax1.set_ylabel("Y (m)")
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-5, 5)

sc2 = ax2.scatter(x[high_grad_mask], y[high_grad_mask], c=grad_p_corrected[high_grad_mask], s=8, cmap='inferno', vmin=0, vmax=9.56e7)
ax2.set_title("High Gradient Regions (Shock)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-5, 5)

cbar = fig.colorbar(sc1, ax=[ax1, ax2], orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label("Gradient Magnitude")

# plt.tight_layout()
# plt.savefig('ellipse_gradient_combined.png', dpi=300)
# plt.show()


print(high_grad_mask.size)
pd.DataFrame(high_grad_mask).to_csv('ellipse_high_grad_mask.csv', index=False)
print("High gradient mask saved to 'ellipse_high_grad_mask.csv'.")