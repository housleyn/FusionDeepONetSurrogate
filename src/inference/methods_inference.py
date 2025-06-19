import torch
from ..model import FusionDeepONet
import csv
import numpy as np 
import pandas as pd
import os

class MethodsInference:
    def _load_model(self, path):
        model = FusionDeepONet(coord_dim=3, param_dim=2, hidden_size=32, num_hidden_layers=3, out_dim=5)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _load_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }

    def _denormalize(self, output):
        return output * self.stats["outputs_std"] + self.stats["outputs_mean"]

    def load_csv_input(self, csv_path):
        df = pd.read_csv(csv_path)
        coords_np = df[["X (m)", "Y (m)", "Z (m)"]].values
        params_np = df[self.param_columns].values
        return coords_np, params_np
    
    def predict(self, coords_np, params):
        coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        params = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(self.device)
               
        with torch.no_grad():
            pred = self.model(coords, params)
        pred = self._denormalize(pred)
        return pred.squeeze(0).cpu().numpy()
    
    def save_to_csv(self, coords_np, output_np, out_path=None):

        headers = [
            'Density (kg/m^3)', 'Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)',
            'Absolute Pressure (Pa)', "X (m)", "Y (m)", "Z (m)"
        ]

        if out_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "Outputs")
            out_path = os.path.join(outputs_dir, "predicted_output.csv")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, mode='w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)
            for i in range(coords_np.shape[0]):
                density = output_np[i, 0]
                vel = output_np[i, 1:4]
                pressure = output_np[i, 4]

                row = [density] + list(vel) + [pressure] + list(coords_np[i])
                writer.writerow(row)


