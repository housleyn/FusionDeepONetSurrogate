import csv
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from model import FusionDeepONet
import matplotlib.pyplot as plt


class MethodsInference:
    """Utility mixin providing model loading and I/O helpers."""

    def _load_model(self, path: str) -> FusionDeepONet:
        model = FusionDeepONet(
            coord_dim=3,
            param_dim=2,
            hidden_size=32,
            num_hidden_layers=3,
            out_dim=5,
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _load_stats(self, npz_path: str) -> dict:
        data = np.load(npz_path)
        return {
            "coords_mean": torch.tensor(data["coords_mean"], dtype=torch.float32),
            "coords_std": torch.tensor(data["coords_std"], dtype=torch.float32),
            "radii_mean": torch.tensor(data["radii_mean"], dtype=torch.float32),
            "radii_std": torch.tensor(data["radii_std"], dtype=torch.float32),
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }

    def _denormalize(self, output: torch.Tensor) -> torch.Tensor:
        return output * self.stats["outputs_std"] + self.stats["outputs_mean"]

    def predict(self, coords_np: np.ndarray, params: Tuple[float, float]) -> np.ndarray:
        coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        params = torch.tensor([params], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model(coords, params)
        pred = self._denormalize(pred)
        return pred.squeeze(0).cpu().numpy()

    def load_csv_input(self, csv_path: str) -> Tuple[np.ndarray, float, float]:
        df = pd.read_csv(csv_path)
        coords_np = df[["X (m)", "Y (m)", "Z (m)"]].values
        if {"a", "b"}.issubset(df.columns):
            a = float(df["a"].iloc[0])
            b = float(df["b"].iloc[0])
            return coords_np, a, b
        raise ValueError("CSV must contain 'a' and 'b' columns for ellipse parameters")

    def save_to_csv(
        self,
        coords_np: np.ndarray,
        output_np: np.ndarray,
        params: Tuple[float, float],
        out_path: str = "predicted_output.csv",
    ) -> None:
        headers = [
            "Density (kg/m^3)",
            "Velocity[i] (m/s)",
            "Velocity[j] (m/s)",
            "Velocity[k] (m/s)",
            "Absolute Pressure (Pa)",
            "a",
            "b",
            "X (m)",
            "Y (m)",
            "Z (m)",
        ]
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)
            for i in range(coords_np.shape[0]):
                density = output_np[i, 0]
                vel = output_np[i, 1:4]
                pressure = output_np[i, 4]
                row = [density] + list(vel) + [pressure, params[0], params[1]] + list(coords_np[i])
                writer.writerow(row)

    def save_to_vtk(self, coords_np: np.ndarray, output_np: np.ndarray, out_path: str = "predicted_output.vtk") -> None:
        cloud = pv.PolyData(coords_np)
        cloud["velocity"] = output_np[:, 1:4]
        cloud["pressure"] = output_np[:, 4]
        cloud["density"] = output_np[:, 0]
        cloud.save(out_path)
        print(f"✅ VTK file saved to: {out_path}")

    @staticmethod
    def relative_L2_norm_error_per_field(csv_true: str, csv_pred: str, columns: Iterable[str]):
        df_true = pd.read_csv(csv_true)
        df_pred = pd.read_csv(csv_pred)
        assert np.allclose(
            df_true[["X (m)", "Y (m)", "Z (m)"]].values,
            df_pred[["X (m)", "Y (m)", "Z (m)"]].values,
        ), "Coordinate mismatch"
        data = []
        for col in columns:
            error = df_pred[col].values - df_true[col].values
            rel_rmse = np.sqrt(np.sum(error ** 2) / np.sum(df_true[col].values ** 2))
            data.append([col, f"{rel_rmse * 100:.2f}%"])
        return data

    @staticmethod
    def plot_table(data) -> None:
        fig, ax = plt.subplots()
        ax.axis("off")
        table = ax.table(cellText=data, colLabels=["Field", "% Error"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.tight_layout()
        plt.savefig("rmse_table.png", dpi=300)
        plt.close(fig)

