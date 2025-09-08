import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

df = pd.read_csv('Data/ellipse_data/ellipse_data_test5.csv')

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

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3})

# sc1 = ax1.scatter(x, y, c=grad_p_corrected, s=8, cmap='inferno', vmin=0, vmax=9.56e7)
# ax1.set_title("Gradient Magnitude Scatter Plot")
# ax1.set_ylabel("Y (m)")
# ax1.set_xlim(-2.5, 2.5)
# ax1.set_ylim(-5, 5)

# sc2 = ax2.scatter(x[high_grad_mask], y[high_grad_mask], c=grad_p_corrected[high_grad_mask], s=8, cmap='inferno', vmin=0, vmax=9.56e7)
# ax2.set_title("High Gradient Regions (Shock)")
# ax2.set_xlabel("X (m)")
# ax2.set_ylabel("Y (m)")
# ax2.set_xlim(-2.5, 2.5)
# ax2.set_ylim(-5, 5)

# cbar = fig.colorbar(sc1, ax=[ax1, ax2], orientation='vertical', fraction=0.025, pad=0.02)
# cbar.set_label("Gradient Magnitude")

# plt.tight_layout()
# plt.savefig('ellipse_gradient_combined.png', dpi=300)
# plt.show()


# print(high_grad_mask.size)
# pd.DataFrame(high_grad_mask).to_csv('ellipse_high_grad_mask.csv', index=False)
# print("High gradient mask saved to 'ellipse_high_grad_mask.csv'.")


X = x[high_grad_mask]
Y = y[high_grad_mask]

import numpy as np
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import medfilt, savgol_filter

# 1) Robust front extraction using percentile
def extract_front_percentile(X, Y, bin_dy, q, min_pts):
    y_min, y_max = Y.min(), Y.max()
    edges = np.arange(y_min, y_max + bin_dy, bin_dy)
    xs, ys = [], []
    which = np.digitize(Y, edges) - 1
    for b in range(len(edges)-1):
        sel = (which == b)
        if np.count_nonzero(sel) >= min_pts:
            xs.append(np.quantile(X[sel], q))
            ys.append(0.5*(edges[b]+edges[b+1]))
    ys = np.array(ys); xs = np.array(xs)
    o = np.argsort(ys)
    return ys[o], xs[o]

ys, xs = extract_front_percentile(X, Y, bin_dy=0.05, q=0.98, min_pts=3)

# 2) Denoise x(y): choose ONE of these
xs_smooth = medfilt(xs, kernel_size=7)                     # robust median
# xs_smooth = savgol_filter(xs, window_length=21, polyorder=3)  # or Savitzky–Golay

# 3) Optional downsample
ys_d, xs_d = ys[::2], xs_smooth[::2]

# 4A) Smooth spline (tune s upward for smoother)
S = UnivariateSpline(ys_d, xs_d, k=3, s=len(xs_d)*1e-2)

# 4B) Or LSQ spline with few knots (often cleaner)
# t = np.quantile(ys_d, [0.2, 0.4, 0.6, 0.8])  # interior knots
# S = LSQUnivariateSpline(ys_d, xs_d, t, k=3)

# usage as before:
dS = S.derivative()
def shock_x(yq): return S(yq)
def shock_angle_beta(yq):
    dx_dy = dS(yq)
    dy_dx = np.where(np.abs(dx_dy)<1e-12, np.inf, 1.0/dx_dy)
    return np.arctan(dy_dx)


dS = S.derivative()  # dx/dy
def shock_x(yq): return S(yq)
def shock_angle_beta(yq):
    dx_dy = dS(yq)
    dy_dx = np.where(np.abs(dx_dy) < 1e-12, np.inf, 1.0/dx_dy)
    return np.arctan(dy_dx)  # radians; angle of shock to freestream (+x)

def angle_deg(yq): return np.degrees(shock_angle_beta(yq))

def angle_for_point(xq, yq):
    xs = shock_x(yq)
    beta = shock_angle_beta(yq)
    return beta, (xq < xs)  # (angle, is_behind_shock)


# --- Plot overlay
yy = np.linspace(ys.min(), ys.max(), 600)
plt.figure(figsize=(7,4))
plt.scatter(X, Y, s=4, c='k', alpha=0.25, label='High-gradient pts')
plt.plot(S(yy), yy, lw=2, label='Shock spline')

plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.title('Fitted Shock Spline & Tangents'); plt.legend(); plt.tight_layout()
plt.savefig('sandbox/ellipse_shock_spline.png', dpi=300)
# plt.show()

