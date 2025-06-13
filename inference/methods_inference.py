import torch
from model import FusionDeepONet
import csv
import numpy as np 
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

class MethodsInference:
    def _load_model(self, path):
        model = FusionDeepONet(coord_dim=3, param_dim=1, hidden_size=32, num_hidden_layers=3, out_dim=5)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _load_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "coords_mean": torch.tensor(data["coords_mean"], dtype=torch.float32),
            "coords_std": torch.tensor(data["coords_std"], dtype=torch.float32),
            "radii_mean": torch.tensor(data["radii_mean"], dtype=torch.float32),
            "radii_std": torch.tensor(data["radii_std"], dtype=torch.float32),
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }

    def _normalize(self, coords, radius):
        # Coordinates and radius were not normalized during training,
        # so simply return them unchanged
        return coords, radius

    def _denormalize(self, output):
        return output * self.stats["outputs_std"] + self.stats["outputs_mean"]

    def predict(self, coords_np, radius_val):
        coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        radius = torch.tensor([[radius_val]], dtype=torch.float32).to(self.device)

        coords_norm, radius_norm = self._normalize(coords, radius)
        with torch.no_grad():
            pred_norm = self.model(coords_norm, radius_norm)
        pred = self._denormalize(pred_norm)
        return pred.squeeze(0).cpu().numpy()

    def load_csv_input(self, csv_path):
        df = pd.read_csv(csv_path)
        coords_np = df[["X (m)", "Y (m)", "Z (m)"]].values
        if "Sphere Radius" in df.columns:
            radius_val = np.float32(df["Sphere Radius"].iloc[0])
        elif "sphere_radius" in df.columns:
            radius_val = np.float32(df["sphere_radius"].iloc[0])
        else:
            raise ValueError("CSV must contain a 'Sphere Radius' column.")
        return coords_np, radius_val

    def save_to_csv(self, coords_np, output_np, radius_val, out_path="predicted_output.csv"):
        headers = [
            'Density (kg/m^3)', 'Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)',
            'Absolute Pressure (Pa)', "Sphere Radius", "X (m)", "Y (m)", "Z (m)"
        ]
        with open(out_path, mode='w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)
            for i in range(coords_np.shape[0]):
                vel = output_np[i, :3]
                pressure = output_np[i, 3]
                density = output_np[i, 4]
                row = list(vel) + [pressure, density, radius_val] + list(coords_np[i])
                writer.writerow(row)

    def save_to_vtk(self, coords_np, output_np, out_path="predicted_output.vtk"):
        cloud = pv.PolyData(coords_np)
        cloud["velocity"] = output_np[:, 1:4]
        cloud["pressure"] = output_np[:, 4]
        cloud["density"] = output_np[:, 0]
        cloud.save(out_path)
        print(f"✅ VTK file saved to: {out_path}")

    def relative_L2_norm_error_per_field(csv_true, csv_pred, columns):
        df_true = pd.read_csv(csv_true)
        df_pred = pd.read_csv(csv_pred)

        assert np.allclose(
            df_true[["X (m)", "Y (m)", "Z (m)"]].values,
            df_pred[["X (m)", "Y (m)", "Z (m)"]].values
        ), "Coordinate mismatch"

        data = []
        for col in columns:
            error = df_pred[col].values - df_true[col].values
            rel_rmse = np.sqrt(np.sum(error**2) / np.sum(df_true[col].values**2))
            data.append([col, f"{rel_rmse * 100:.2f}%"])

        return data

    def plot_table(data):
        fig, ax = plt.subplots()
        ax.axis('off')
        table = ax.table(
            cellText=data,
            colLabels=["Field", "% Error"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.tight_layout()
        plt.savefig("rmse_table.png", dpi=300)
        plt.show()