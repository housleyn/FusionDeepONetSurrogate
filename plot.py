import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load the data ===
data = np.load("test_processed_data.npz")
coords = data["coords"]      # shape: (3, npts_max, 3)
outputs = data["outputs"]    # shape: (3, npts_max, 5)
params = data["params"]      # shape: (3, 1)

# === Pick dataset index ===
i = 0  # 0, 1, or 2 for the 3 radii

X = coords[i, :, 0]
Y = coords[i, :, 1]
Z = coords[i, :, 2]

rho     = outputs[i, :, 0]
vx      = outputs[i, :, 1]
vy      = outputs[i, :, 2]
vz      = outputs[i, :, 3]
p       = outputs[i, :, 4]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = outputs[i, :, 4]

sc = ax.scatter(X, Y, Z, c=p, cmap='viridis', s=3)
plt.colorbar(sc, ax=ax, label="Pressure (Pa)")
ax.set_title(f"Pressure Field (Radius = {params[i, 0]})")
plt.show()
