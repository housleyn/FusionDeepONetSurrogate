import torch
import numpy as np
from model import FusionDeepONet
import pandas as pd

def load_model(path="fusion_deeponet.pt", device="cpu"):
    model = FusionDeepONet(coord_dim=3, param_dim=1, hidden_size=32, num_hidden_layers=3, out_dim=5)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_stats(npz_path="processed_data.npz"):
    data = np.load(npz_path)
    return {
        "coords_mean": torch.tensor(data["coords_mean"], dtype=torch.float32),
        "coords_std": torch.tensor(data["coords_std"], dtype=torch.float32),
        "radii_mean": torch.tensor(data["radii_mean"], dtype=torch.float32),
        "radii_std": torch.tensor(data["radii_std"], dtype=torch.float32),
        "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
        "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
    }

def normalize(coords, radius, stats):
    coords = (coords - stats["coords_mean"]) / stats["coords_std"].float()
    radius = (radius - stats["radii_mean"]) / stats["radii_std"].float()
    return coords, radius

def denormalize(output, stats):
    return output * stats["outputs_std"] + stats["outputs_mean"]

def predict(coords_np, radius_val, model, stats, device="cpu"):
    coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, n_pts, 3)
    radius = torch.tensor([[radius_val]], dtype=torch.float32).to(device)          # shape: (1, 1)

    coords_norm, radius_norm = normalize(coords, radius, stats)
    with torch.no_grad():
        pred_norm = model(coords_norm, radius_norm)
    pred = denormalize(pred_norm, stats)
    return pred.squeeze(0).cpu().numpy()  # shape: (n_pts, 5)


def load_csv_input(csv_path):
    df = pd.read_csv(csv_path)
    
    # Extract node coordinates
    coords_np = df[["X (m)", "Y (m)", "Z (m)"]].values  # shape: (n_pts, 3)

    # Extract radius (assuming it's the same for all rows)
    if "Sphere Radius" in df.columns:
        radius_val = np.float32(df["Sphere Radius"].iloc[0])
    else:
        raise ValueError("CSV must contain a 'radius' column.")

    return coords_np, radius_val

import csv

def save_to_csv(coords_np, output_np, radius_val, out_path="predicted_output.csv"):
    headers = [
        '"Velocity[i] (m/s)"', '"Velocity[j] (m/s)"', '"Velocity[k] (m/s)"',
        '"Absolute Pressure (Pa)"', '"Density (kg/m^3)"',
        '"Sphere Radius"', '"X (m)"', '"Y (m)"', '"Z (m)"'
    ]
    
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(coords_np.shape[0]):
            vel = output_np[i, :3]
            pressure = output_np[i, 3]
            density = output_np[i, 4]

            row = list(vel) + [pressure, density, radius_val]
            row += list(coords_np[i])
            writer.writerow(row)


import pyvista as pv

def save_to_vtk(coords_np, output_np, out_path="predicted_output.vtk"):
    # Create PyVista point cloud
    cloud = pv.PolyData(coords_np)

    # Add vector and scalar fields
    cloud["velocity"] = output_np[:, :3]                   # u, v, w
    cloud["pressure"] = output_np[:, 3]                    # pressure
    cloud["density"] = output_np[:, 4]                     # density

    # Save to VTK
    cloud.save(out_path)
    print(f"✅ VTK file saved to: {out_path}")
